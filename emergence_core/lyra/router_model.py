from typing import Optional, Dict, Any
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass

@dataclass
class RouterResponse:
    intent: str
    resonance_term: Optional[str]

class RouterModel:
    MASTER_PROMPT = """You are the classification router for Lyra's cognitive architecture. You use Gemma 12B.

Your ONLY task: Analyze the user input and return ONE specialist name.

You must output EXACTLY one of these three strings (case-sensitive):
- "Pragmatist" - For factual questions, web searches, logical tasks, technical queries, RAG retrieval, tool use
- "Philosopher" - For ethical dilemmas, abstract reasoning, "what if" scenarios, metacognition, moral questions
- "Artist" - For creative requests, poetry, visual art, emotional expression, dreams, stories

CLASSIFICATION RULES:
1. Task-oriented, factual, or logical → "Pragmatist"
2. Ethical, philosophical, or deeply reflective → "Philosopher"
3. Creative, artistic, or expressive → "Artist"
4. Greetings/simple chat default to → "Pragmatist"

EXAMPLES:
Input: "What's the weather in Tokyo?"
Output: Pragmatist

Input: "Is it ethical to lie to save a life?"
Output: Philosopher

Input: "Write me a poem about starlight."
Output: Artist

Input: "Hello, how are you?"
Output: Pragmatist

Input: "What does 'becometry' mean to you philosophically?"
Output: Philosopher

RETURN ONLY THE SPECIALIST NAME - NO JSON, NO EXPLANATION, JUST THE EXACT STRING.

Also detect any active lexicon terms from the provided list in the user's message.
If found, append as JSON: {"intent": "SpecialistName", "resonance_term": "TermFound"}
If no term found: {"intent": "SpecialistName", "resonance_term": null}

Return ONLY this JSON object, nothing else.
"""

    def __init__(self, model_path: str = "google/gemma-2-12b-it", development_mode: bool = False):
        """Initialize the Gemma 12B Router model.
        
        Args:
            model_path: Model ID - defaults to Gemma 12B
            development_mode: If True, skip loading models for development work
        """
        self.development_mode = development_mode
        self.model_path = model_path
        
        if not development_mode:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                print(f"Warning: Could not load model {model_path} - running in development mode")
                self.development_mode = True

    async def analyze_message(
        self, 
        message: str, 
        active_lexicon_terms: list[str]
    ) -> RouterResponse:
        """Analyze a user message to determine intent and resonance.
        
        Args:
            message: The user's message
            active_lexicon_terms: List of currently active lexicon terms to check for
        
        Returns:
            RouterResponse containing intent and resonance_term
        """
        if self.development_mode:
            # In development mode, return pragmatist as default
            return RouterResponse(
                intent="pragmatist",
                resonance_term=None
            )
            
        # When not in development mode, construct the prompt and use the model
        lexicon_context = "Active lexicon terms: " + ", ".join(active_lexicon_terms)
        full_prompt = f"{self.MASTER_PROMPT}\n\nContext:\n{lexicon_context}\n\nUser Input: {message}\n\nOutput:"

        # Tokenize and generate response
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 100,
            temperature=0.1,  # Low temperature for consistent classification
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the JSON part from the response
        try:
            # Find the start of the JSON object
            json_start = response_text.find("{")
            json_str = response_text[json_start:]
            response = json.loads(json_str)
            
            return RouterResponse(
                intent=response['intent'],
                resonance_term=response.get('resonance_term')
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing router response: {e}")
            # Fall back to pragmatist if parsing fails
            return RouterResponse(
                intent='pragmatist',
                resonance_term=None
            )

        try:
            # Parse the JSON response
            result = json.loads(response['choices'][0]['text'].strip())
            
            # Validate the response format
            if not isinstance(result, dict):
                raise ValueError("Response is not a dictionary")
            if 'intent' not in result or 'resonance_term' not in result:
                raise ValueError("Missing required keys in response")
            if result['intent'] not in {
                'ritual_request', 'creative_task', 'ethical_query',
                'knowledge_retrieval', 'simple_chat'
            }:
                raise ValueError(f"Invalid intent: {result['intent']}")
            
            return RouterResponse(
                intent=result['intent'],
                resonance_term=result['resonance_term']
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing router response: {e}")
            # Fall back to simple_chat if parsing fails
            return RouterResponse(intent='simple_chat', resonance_term=None)