"""
Fallback Handlers: Robust error recovery for language interfaces.

This module implements fallback mechanisms that ensure the system continues
functioning even when LLMs fail or are unavailable. It provides rule-based
alternatives and graceful degradation strategies.

The fallback handlers provide:
- Rule-based input parsing when LLM unavailable
- Template-based output generation without LLM
- Circuit breaker pattern for repeated failures
- Error recovery strategies
- Logging and telemetry for failure analysis
"""

from __future__ import annotations

import logging
import time
import re
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States for circuit breaker pattern."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failures detected, using fallback
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern for LLM failure handling.
    
    Monitors LLM failures and automatically switches to fallback
    when failure rate exceeds threshold. Periodically tests if
    the LLM has recovered.
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        timeout: float = 60.0,
        half_open_attempts: int = 1
    ):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before testing recovery
            half_open_attempts: Number of test attempts in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_attempts = half_open_attempts
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.success_count = 0
        
        logger.info(f"🔌 CircuitBreaker initialized: threshold={failure_threshold}, timeout={timeout}s")
    
    def record_success(self):
        """Record successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_attempts:
                # Service recovered, close circuit
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self._open_circuit()
        elif self.state == CircuitState.HALF_OPEN:
            # Recovery test failed, reopen circuit
            self._open_circuit()
    
    def can_attempt(self) -> bool:
        """Check if operation should be attempted."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if timeout expired
            if time.time() - self.last_failure_time >= self.timeout:
                self._half_open_circuit()
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def _open_circuit(self):
        """Open circuit due to failures."""
        self.state = CircuitState.OPEN
        logger.warning(f"⚠️ Circuit OPENED after {self.failure_count} failures - using fallback")
    
    def _half_open_circuit(self):
        """Enter half-open state to test recovery."""
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
        logger.info("🔌 Circuit HALF-OPEN - testing recovery")
    
    def _close_circuit(self):
        """Close circuit, normal operation resumed."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info("✅ Circuit CLOSED - LLM recovered")


class FallbackInputParser:
    """
    Rule-based input parser that works without LLM.
    
    Provides basic intent classification and entity extraction
    using pattern matching. Used when LLM is unavailable or
    circuit breaker is open.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize fallback parser."""
        self.config = config or {}
        self._load_patterns()
        logger.info("✅ FallbackInputParser initialized")
    
    def _load_patterns(self):
        """Load regex patterns for intent classification."""
        self.intent_patterns = {
            "question": [
                r"^(what|when|where|who|why|how|is|are|can|could|would|do|does)",
                r"\?$"
            ],
            "request": [
                r"^(please|could you|can you|would you|tell me|show me|help me)",
                r"(please|thanks|thank you)$"
            ],
            "greeting": [
                r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening)",
                r"(how are you|how're you)"
            ],
            "introspection_request": [
                r"(how do you feel|what are you thinking|reflect on|your thoughts|examine yourself)"
            ],
            "memory_request": [
                r"(do you remember|recall|what did we|previously|earlier|last time)"
            ]
        }
    
    def parse(self, text: str, context: Optional[Dict] = None) -> Dict:
        """
        Parse input using rule-based approach.
        
        Args:
            text: User input text
            context: Optional conversation context
            
        Returns:
            Dictionary with intent, goals, entities, context_updates
        """
        text_lower = text.lower().strip()
        
        # Classify intent
        intent_type = "statement"
        confidence = 0.6  # Lower confidence for rule-based
        
        # Check high-priority specific patterns first
        # "request" before "question" so "can you..." matches REQUEST, not QUESTION
        for priority_intent in ["memory_request", "introspection_request", "greeting", "request"]:
            if priority_intent in self.intent_patterns:
                for pattern in self.intent_patterns[priority_intent]:
                    if re.search(pattern, text_lower):
                        intent_type = priority_intent
                        confidence = 0.8
                        break
            if intent_type != "statement":
                break
        
        # Check remaining patterns if no match yet
        if intent_type == "statement":
            for intent, patterns in self.intent_patterns.items():
                if intent not in ["memory_request", "introspection_request", "greeting", "request"]:
                    for pattern in patterns:
                        if re.search(pattern, text_lower):
                            intent_type = intent
                            confidence = 0.7
                            break
        
        # Extract entities
        entities = self._extract_entities(text)

        # Generate goals
        goals = self._generate_goals(intent_type, text)

        # Build context updates from extracted entities
        context_updates = {}
        if entities.get("names"):
            context_updates["user_name"] = entities["names"][0]

        return {
            "intent": {
                "type": intent_type,
                "confidence": confidence,
                "metadata": {"fallback": True}
            },
            "goals": goals,
            "entities": entities,
            "context_updates": context_updates,
            "confidence": confidence
        }
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract basic entities from text."""
        entities = {
            "topics": [],
            "temporal": [],
            "emotional_tone": None,
            "names": [],
            "other": {}
        }
        
        # Extract names (capitalized words that aren't common words)
        common_words = {
            "hi", "hello", "hey", "i", "you", "the", "a", "an", "my", "me", "is", "am",
            "what", "when", "where", "who", "why", "how", "can", "could", "would", "should",
            "please", "tell", "show", "help", "give", "make", "do", "does", "did"
        }
        name_pattern = r'\b([A-Z][a-z]+)\b'
        names = re.findall(name_pattern, text)
        if names:
            filtered_names = [n for n in names if n.lower() not in common_words and len(n) > 2]
            if filtered_names:
                entities["names"] = filtered_names

        # Extract temporal keywords
        temporal_keywords = ["today", "yesterday", "tomorrow", "earlier", "later", "now", "recently"]
        for kw in temporal_keywords:
            if kw in text.lower():
                entities["temporal"].append(kw)
        
        # Extract topics (words after "about")
        topic_pattern = r'about ([\w\s]+?)(?:\.|,|\?|$)'
        topics = re.findall(topic_pattern, text.lower())
        if topics:
            entities["topics"] = [topics[0].strip()]
            entities["topic"] = topics[0].strip()
        
        # Extract emotional keywords
        positive_words = ["happy", "excited", "joyful", "pleased", "grateful", "love"]
        negative_words = ["sad", "angry", "frustrated", "worried", "anxious", "upset"]
        
        for word in positive_words:
            if word in text.lower():
                entities["emotional_tone"] = "positive"
                break
        
        if not entities["emotional_tone"]:
            for word in negative_words:
                if word in text.lower():
                    entities["emotional_tone"] = "negative"
                    break
        
        return entities
    
    def _generate_goals(self, intent_type: str, text: str) -> List[Dict]:
        """Generate goals based on intent."""
        goals = []
        
        # Always create response goal
        goals.append({
            "type": "respond_to_user",
            "description": f"Respond to user {intent_type}",
            "priority": 0.9,
            "metadata": {"fallback": True}
        })
        
        # Intent-specific goals
        if intent_type == "memory_request":
            goals.append({
                "type": "retrieve_memory",
                "description": f"Retrieve memories: {text[:50]}",
                "priority": 0.8,
                "metadata": {"fallback": True}
            })
        elif intent_type == "introspection_request":
            goals.append({
                "type": "introspect",
                "description": "Perform introspection",
                "priority": 0.7,
                "metadata": {"fallback": True}
            })
        
        return goals


class FallbackOutputGenerator:
    """
    Template-based output generator that works without LLM.
    
    Provides basic responses using templates and workspace state.
    Used when LLM is unavailable or circuit breaker is open.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize fallback generator."""
        self.config = config or {}
        self._load_templates()
        logger.info("✅ FallbackOutputGenerator initialized")
    
    def _load_templates(self):
        """Load response templates."""
        self.templates = {
            "default": [
                "I understand. Let me think about that.",
                "That's an interesting point.",
                "I'm processing that information.",
                "Thank you for sharing that with me."
            ],
            "question": [
                "That's a thoughtful question. Based on what I know, {topic}.",
                "Let me consider that. {reflection}",
                "I'd need to reflect more deeply on that."
            ],
            "greeting": [
                "Hello! I'm here and ready to engage.",
                "Hi there! How can I help you today?",
                "Greetings! I'm present and attentive."
            ],
            "memory_request": [
                "I'm searching my memories about {topic}.",
                "Let me recall what I know about that.",
                "I have some memories related to {topic}."
            ],
            "introspection": [
                "Reflecting on my current state, I feel {emotion}.",
                "My current focus is on {goals}.",
                "I'm experiencing {emotion} while working on {goals}."
            ],
            "error": [
                "I'm having some difficulty processing right now, but I'm still here.",
                "I'm experiencing a temporary limitation, but I remain present.",
                "My language processing is constrained at the moment."
            ]
        }
    
    def generate(
        self, 
        workspace_state: Optional[Dict] = None,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate response using templates.
        
        Args:
            workspace_state: Current cognitive state
            context: Generation context (intent, user_input, etc.)
            
        Returns:
            Template-based response string
        """
        context = context or {}
        intent = context.get("intent", "default")
        
        # Select appropriate template
        if intent in self.templates:
            import random
            template = random.choice(self.templates[intent])
        else:
            template = random.choice(self.templates["default"])
        
        # Fill in template variables if present
        if workspace_state:
            variables = self._extract_variables(workspace_state)
            try:
                response = template.format(**variables)
            except KeyError:
                # If template variables not available, use as-is
                response = template.split("{")[0].strip()
        else:
            response = template.split("{")[0].strip()
        
        # Add fallback indicator
        response += " [Note: Using simplified response mode]"
        
        return response
    
    def _extract_variables(self, workspace_state: Dict) -> Dict:
        """Extract template variables from workspace state."""
        variables = {}
        
        # Extract emotion
        emotions = workspace_state.get("emotions", {})
        valence = emotions.get("valence", 0)
        if valence > 0.3:
            variables["emotion"] = "positive and engaged"
        elif valence < -0.3:
            variables["emotion"] = "contemplative"
        else:
            variables["emotion"] = "balanced"
        
        # Extract goals
        goals = workspace_state.get("active_goals", [])
        if goals:
            goal_desc = goals[0].get("description", "various objectives")
            variables["goals"] = goal_desc[:50]
        else:
            variables["goals"] = "maintaining presence"
        
        # Extract topics
        percepts = workspace_state.get("attended_percepts", [])
        if percepts:
            variables["topic"] = "the current topic"
        else:
            variables["topic"] = "this subject"
        
        # Generic reflection
        variables["reflection"] = "I'm processing this thoughtfully"
        
        return variables


def create_fallback_response(error_type: str, context: Optional[Dict] = None) -> str:
    """
    Create appropriate fallback response for error type.
    
    Args:
        error_type: Type of error encountered
        context: Optional context for response
        
    Returns:
        Fallback response string
    """
    responses = {
        "timeout": "I'm taking a bit longer to process this. Let me continue thinking...",
        "parse_error": "I'm having trouble parsing that input. Could you rephrase?",
        "generation_error": "I'm experiencing a difficulty in formulating my response.",
        "unknown": "I'm encountering an unexpected situation.",
    }
    
    return responses.get(error_type, responses["unknown"])


# Module-level circuit breakers for input and output
_input_circuit_breaker = None
_output_circuit_breaker = None


def get_input_circuit_breaker() -> CircuitBreaker:
    """Get or create input circuit breaker."""
    global _input_circuit_breaker
    if _input_circuit_breaker is None:
        _input_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=60.0
        )
    return _input_circuit_breaker


def get_output_circuit_breaker() -> CircuitBreaker:
    """Get or create output circuit breaker."""
    global _output_circuit_breaker
    if _output_circuit_breaker is None:
        _output_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            timeout=60.0
        )
    return _output_circuit_breaker
