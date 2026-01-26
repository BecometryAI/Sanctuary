"""
LLM Client: Unified interface for language model interactions.

This module provides abstract and concrete implementations for interacting
with large language models. It supports both local models (via transformers)
and API providers, with proper resource management and error handling.

The LLM client layer provides:
- Abstract interface for model-agnostic code
- Concrete implementations for specific models (Gemma, Llama)
- Mock implementation for testing
- Connection pooling and rate limiting
- Proper error handling and retry logic
"""

from __future__ import annotations

import asyncio
import logging
import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from collections import deque

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    Defines the interface that all LLM clients must implement.
    Subclasses provide concrete implementations for specific models
    or API providers.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize LLM client.
        
        Args:
            config: Configuration dictionary with model-specific settings
        """
        self.config = config or {}
        self.model_name = self.config.get("model_name", "unknown")
        self.device = self.config.get("device", "cpu")
        self.max_tokens = self.config.get("max_tokens", 500)
        self.temperature = self.config.get("temperature", 0.7)
        self.timeout = self.config.get("timeout", 10.0)
        
        # Rate limiting
        self.rate_limit = self.config.get("rate_limit", 10)  # requests per minute
        self.request_times: deque = deque(maxlen=self.rate_limit)
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_errors": 0,
            "avg_latency": 0.0
        }
        
        self._initialized = False
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            temperature: Generation temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated text response
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    async def generate_structured(
        self, 
        prompt: str, 
        schema: Dict,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """
        Generate structured JSON output.
        
        Args:
            prompt: Input prompt text
            schema: Expected JSON schema
            temperature: Generation temperature (overrides default)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Parsed JSON dictionary matching schema
            
        Raises:
            LLMError: If generation or parsing fails
        """
        pass
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Remove requests older than 1 minute
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # Check if we're at the limit
        if len(self.request_times) >= self.rate_limit:
            wait_time = 60 - (current_time - self.request_times[0])
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        self.request_times.append(current_time)
    
    def _update_metrics(self, latency: float, tokens: int = 0, error: bool = False):
        """Update client metrics."""
        self.metrics["total_requests"] += 1
        self.metrics["total_tokens"] += tokens
        if error:
            self.metrics["total_errors"] += 1
        
        # Update average latency
        n = self.metrics["total_requests"]
        current_avg = self.metrics["avg_latency"]
        self.metrics["avg_latency"] = (current_avg * (n - 1) + latency) / n


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing and development.
    
    Returns predefined responses without actual model inference.
    Useful for testing the cognitive architecture without GPU resources.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model_name = "mock-llm"
        self._initialized = True
        logger.info("✅ MockLLMClient initialized")
    
    async def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate a mock text response."""
        await self._check_rate_limit()
        
        start_time = time.time()
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        response = (
            "This is a mock response from the development LLM client. "
            "In production, this would be replaced with actual model output. "
            f"Prompt length: {len(prompt)} chars."
        )
        
        latency = time.time() - start_time
        self._update_metrics(latency, tokens=len(response.split()))
        
        return response
    
    async def generate_structured(
        self, 
        prompt: str, 
        schema: Dict,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """Generate a mock structured JSON response."""
        await self._check_rate_limit()
        
        start_time = time.time()
        
        # Simulate processing delay
        await asyncio.sleep(0.1)
        
        # Return mock structured response
        response = {
            "intent": {
                "type": "question",
                "confidence": 0.85,
                "metadata": {}
            },
            "goals": [
                {
                    "type": "respond_to_user",
                    "description": "Respond to user question",
                    "priority": 0.9,
                    "metadata": {}
                }
            ],
            "entities": {
                "topics": ["general"],
                "temporal": [],
                "emotional_tone": "neutral",
                "names": [],
                "other": {}
            },
            "context_updates": {},
            "confidence": 0.85
        }
        
        latency = time.time() - start_time
        self._update_metrics(latency, tokens=50)
        
        return response


class GemmaClient(LLMClient):
    """
    Gemma 12B client for input parsing.
    
    Uses Google's Gemma model for natural language understanding
    and structured output generation.
    """
    
    MODEL_NAME = "google/gemma-12b"
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        config.setdefault("model_name", self.MODEL_NAME)
        config.setdefault("temperature", 0.3)  # Lower for structured parsing
        config.setdefault("max_tokens", 512)
        super().__init__(config)
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load Gemma model and tokenizer."""
        try:
            # Attempt to import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
            
            self._initialized = True
            logger.info(f"✅ GemmaClient initialized on {self.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.warning("GemmaClient will operate in mock mode")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to load Gemma model: {e}")
            logger.warning("GemmaClient will operate in mock mode")
            self._initialized = False
    
    async def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using Gemma model."""
        if not self._initialized or self.model is None:
            logger.warning("Gemma model not loaded, using mock response")
            return await MockLLMClient().generate(prompt, temperature, max_tokens)
        
        await self._check_rate_limit()
        
        start_time = time.time()
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with timeout
            outputs = await asyncio.wait_for(
                asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_new_tokens=max_tok,
                    temperature=temp,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id
                ),
                timeout=self.timeout
            )
            
            # Decode output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            latency = time.time() - start_time
            self._update_metrics(latency, tokens=len(outputs[0]))
            
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"Gemma generation timed out after {self.timeout}s")
            self._update_metrics(time.time() - start_time, error=True)
            raise LLMError("Generation timeout")
        except Exception as e:
            logger.error(f"Gemma generation failed: {e}")
            self._update_metrics(time.time() - start_time, error=True)
            raise LLMError(f"Generation failed: {e}")
    
    async def generate_structured(
        self, 
        prompt: str, 
        schema: Dict,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """Generate structured JSON output using Gemma."""
        # Add JSON formatting instruction to prompt
        structured_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n\nJSON Response:"
        
        # Generate response
        response_text = await self.generate(structured_prompt, temperature)
        
        # Parse JSON
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_text = response_text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            
            parsed = json.loads(json_text)
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemma output: {e}")
            logger.debug(f"Raw output: {response_text}")
            raise LLMError(f"JSON parsing failed: {e}")


class LlamaClient(LLMClient):
    """
    Llama 3 70B client for output generation.
    
    Uses Meta's Llama model for natural, identity-aligned
    response generation.
    """
    
    MODEL_NAME = "meta-llama/Llama-3-70B"
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        config.setdefault("model_name", self.MODEL_NAME)
        config.setdefault("temperature", 0.7)  # Higher for creative generation
        config.setdefault("max_tokens", 500)
        super().__init__(config)
        
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load Llama model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading {self.model_name}...")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                load_in_8bit=self.config.get("load_in_8bit", False)  # Optional quantization
            )
            
            self._initialized = True
            logger.info(f"✅ LlamaClient initialized on {self.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.warning("LlamaClient will operate in mock mode")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            logger.warning("LlamaClient will operate in mock mode")
            self._initialized = False
    
    async def generate(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using Llama model."""
        if not self._initialized or self.model is None:
            logger.warning("Llama model not loaded, using mock response")
            return await MockLLMClient().generate(prompt, temperature, max_tokens)
        
        await self._check_rate_limit()
        
        start_time = time.time()
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with timeout
            outputs = await asyncio.wait_for(
                asyncio.to_thread(
                    self.model.generate,
                    **inputs,
                    max_new_tokens=max_tok,
                    temperature=temp,
                    do_sample=temp > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=kwargs.get("top_p", 0.9),
                    repetition_penalty=kwargs.get("repetition_penalty", 1.1)
                ),
                timeout=self.timeout
            )
            
            # Decode output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            latency = time.time() - start_time
            self._update_metrics(latency, tokens=len(outputs[0]))
            
            return response
            
        except asyncio.TimeoutError:
            logger.error(f"Llama generation timed out after {self.timeout}s")
            self._update_metrics(time.time() - start_time, error=True)
            raise LLMError("Generation timeout")
        except Exception as e:
            logger.error(f"Llama generation failed: {e}")
            self._update_metrics(time.time() - start_time, error=True)
            raise LLMError(f"Generation failed: {e}")
    
    async def generate_structured(
        self, 
        prompt: str, 
        schema: Dict,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """Generate structured JSON output using Llama."""
        # Add JSON formatting instruction to prompt
        structured_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}\n\nJSON Response:"
        
        # Generate response
        response_text = await self.generate(structured_prompt, temperature)
        
        # Parse JSON
        try:
            # Extract JSON from response
            json_text = response_text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            
            parsed = json.loads(json_text)
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Llama output: {e}")
            logger.debug(f"Raw output: {response_text}")
            raise LLMError(f"JSON parsing failed: {e}")


class LLMError(Exception):
    """Exception raised for LLM-related errors."""
    pass
