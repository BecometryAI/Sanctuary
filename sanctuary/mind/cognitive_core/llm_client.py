"""
LLM Client: Unified interface for language model interactions.

This module provides abstract and concrete implementations for interacting
with large language models. It supports both local models (via transformers),
Ollama (local HTTP API), and API providers, with proper resource management
and error handling.

The LLM client layer provides:
- Abstract interface for model-agnostic code
- Concrete implementations for specific models (Gemma, Llama)
- Ollama integration for local model serving
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
from typing import AsyncIterator, Dict, Any, Optional, List
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
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream generated text token-by-token.

        Default implementation falls back to non-streaming generate().
        Subclasses may override for true token streaming.

        Args:
            prompt: Input prompt text
            temperature: Generation temperature (overrides default)
            max_tokens: Maximum tokens to generate (overrides default)
            **kwargs: Additional model-specific parameters

        Yields:
            Text chunks as they are generated
        """
        response = await self.generate(prompt, temperature, max_tokens, **kwargs)
        yield response

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
    
    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream mock response word-by-word."""
        await self._check_rate_limit()

        start_time = time.time()

        words = [
            "This", " is", " a", " mock", " streamed", " response",
            " from", " the", " development", " LLM", " client.",
        ]
        for word in words:
            await asyncio.sleep(0.02)
            yield word

        latency = time.time() - start_time
        self._update_metrics(latency, tokens=len(words))

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


class OllamaClient(LLMClient):
    """
    Ollama client for local model serving.

    Connects to a locally running Ollama instance via its HTTP API.
    Works with any model Ollama supports (llama3.2, gemma2, mistral, etc.)

    Requires:
        - Ollama installed and running (https://ollama.com)
        - A model pulled: e.g. `ollama pull gemma3:12b`

    Config example:
        {
            "backend": "ollama",
            "model_name": "gemma3:12b",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
            "max_tokens": 500,
        }
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        config.setdefault("model_name", "llama3.2")
        config.setdefault("temperature", 0.7)
        config.setdefault("max_tokens", 500)
        super().__init__(config)

        self.base_url = self.config.get("base_url", self.DEFAULT_BASE_URL).rstrip("/")
        self._initialized = False
        self._verify_connection()

    def _verify_connection(self):
        """Check that Ollama is running and the model is available."""
        import urllib.request
        import urllib.error

        try:
            # Check Ollama is running
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            available_models = [m.get("name", "") for m in data.get("models", [])]

            # Check if requested model is available (handle tag variants)
            model_found = False
            for m in available_models:
                if m == self.model_name or m.startswith(f"{self.model_name}:") or self.model_name.startswith(m.split(":")[0]):
                    model_found = True
                    break

            if model_found:
                logger.info(f"✅ OllamaClient connected: model={self.model_name}, url={self.base_url}")
                self._initialized = True
            else:
                logger.warning(
                    f"Model '{self.model_name}' not found in Ollama. "
                    f"Available: {available_models}. "
                    f"Pull it with: ollama pull {self.model_name}"
                )
                # Still mark as initialized — Ollama will pull on first use or give a clear error
                self._initialized = True

        except urllib.error.URLError as e:
            logger.error(
                f"Cannot connect to Ollama at {self.base_url}: {e}. "
                "Make sure Ollama is running (start it with: ollama serve)"
            )
            self._initialized = False
        except Exception as e:
            logger.error(f"Ollama connection check failed: {e}")
            self._initialized = False

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Generate text using Ollama's /api/generate endpoint."""
        if not self._initialized:
            logger.warning("Ollama not connected, using mock response")
            return await MockLLMClient().generate(prompt, temperature, max_tokens)

        await self._check_rate_limit()

        start_time = time.time()
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
            }
        }).encode("utf-8")

        try:
            response_text = await asyncio.wait_for(
                asyncio.to_thread(self._do_generate_request, payload),
                timeout=self.timeout
            )

            latency = time.time() - start_time
            self._update_metrics(latency, tokens=len(response_text.split()))
            logger.debug(f"Ollama generated {len(response_text)} chars in {latency:.2f}s")

            return response_text

        except asyncio.TimeoutError:
            logger.error(f"Ollama generation timed out after {self.timeout}s")
            self._update_metrics(time.time() - start_time, error=True)
            raise LLMError(f"Ollama generation timeout after {self.timeout}s")
        except LLMError:
            raise
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            self._update_metrics(time.time() - start_time, error=True)
            raise LLMError(f"Ollama generation failed: {e}")

    def _do_generate_request(self, payload: bytes) -> str:
        """Synchronous HTTP request to Ollama (run in thread)."""
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=int(self.timeout)) as resp:
                data = json.loads(resp.read().decode())
                return data.get("response", "")
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            raise LLMError(f"Ollama HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}: {e}")

    async def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream text using Ollama's native streaming endpoint."""
        if not self._initialized:
            logger.warning("Ollama not connected, using mock stream")
            async for chunk in MockLLMClient().generate_stream(prompt, temperature, max_tokens):
                yield chunk
            return

        await self._check_rate_limit()

        start_time = time.time()
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temp,
                "num_predict": max_tok,
            }
        }).encode("utf-8")

        try:
            token_count = 0
            for chunk in await asyncio.to_thread(self._do_stream_request, payload):
                token_count += 1
                yield chunk

            latency = time.time() - start_time
            self._update_metrics(latency, tokens=token_count)
            logger.debug(f"Ollama streamed {token_count} chunks in {latency:.2f}s")

        except Exception as e:
            logger.error(f"Ollama stream failed: {e}")
            self._update_metrics(time.time() - start_time, error=True)
            raise LLMError(f"Ollama stream failed: {e}")

    def _do_stream_request(self, payload: bytes) -> List[str]:
        """
        Synchronous streaming HTTP request to Ollama (run in thread).

        Returns list of text chunks (collected synchronously so the async
        generator in generate_stream can yield them without blocking).
        """
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        chunks = []
        try:
            with urllib.request.urlopen(req, timeout=int(self.timeout)) as resp:
                for line in resp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode())
                        chunk = data.get("response", "")
                        if chunk:
                            chunks.append(chunk)
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            raise LLMError(f"Ollama HTTP {e.code}: {body}")
        except urllib.error.URLError as e:
            raise LLMError(f"Cannot reach Ollama at {self.base_url}: {e}")

        return chunks

    async def generate_structured(
        self,
        prompt: str,
        schema: Dict,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """Generate structured JSON output using Ollama."""
        structured_prompt = (
            f"{prompt}\n\n"
            f"Respond with ONLY valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"JSON Response:"
        )

        response_text = await self.generate(
            structured_prompt,
            temperature=temperature or 0.3  # Lower temp for structured output
        )

        try:
            json_text = response_text.strip()
            # Strip markdown code fences if present
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            return json.loads(json_text)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Ollama output: {e}")
            logger.debug(f"Raw output: {response_text}")
            raise LLMError(f"JSON parsing failed: {e}")


class LLMError(Exception):
    """Exception raised for LLM-related errors."""
    pass
