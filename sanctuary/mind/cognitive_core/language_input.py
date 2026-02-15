"""
Language Input Parser: Natural language to cognitive structures.

This module implements the LanguageInputParser class, which converts natural language
user input into structured Goals and Percepts that the cognitive core can process.
This is how Sanctuary "hears" and understands language.

The language input parser is responsible for:
- Converting natural language into structured cognitive formats
- Intent classification (question, request, statement, etc.)
- Goal generation from user intents
- Entity extraction (names, topics, temporal references, emotions)
- Context tracking across conversation turns
- Integration with perception subsystem for text encoding
- LLM-powered parsing with fallback to rule-based approach
"""

from __future__ import annotations

import asyncio
import logging
import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .workspace import Goal, GoalType, Percept
from .llm_client import LLMClient, LLMError
from .fallback_handlers import FallbackInputParser, get_input_circuit_breaker
from .structured_formats import (
    LLMInputParseRequest,
    LLMInputParseResponse,
    parse_response_to_goals,
    ConversationContext
)

logger = logging.getLogger(__name__)


class IntentType(str, Enum):
    """
    Types of user intent detected in natural language input.
    
    QUESTION: Asking for information or explanation
    REQUEST: Requesting action or assistance
    STATEMENT: Making a declaration or assertion
    GREETING: Social interaction (hello, goodbye)
    INTROSPECTION_REQUEST: Asking about system's internal state or feelings
    MEMORY_REQUEST: Asking about past events or memories
    UNKNOWN: Unable to classify intent
    """
    QUESTION = "question"
    REQUEST = "request"
    STATEMENT = "statement"
    GREETING = "greeting"
    INTROSPECTION_REQUEST = "introspection_request"
    MEMORY_REQUEST = "memory_request"
    UNKNOWN = "unknown"


@dataclass
class Intent:
    """
    Represents classified user intent.
    
    Attributes:
        type: The type of intent detected
        confidence: Confidence score (0.0-1.0)
        metadata: Additional intent-specific information
    """
    type: IntentType
    confidence: float
    metadata: Dict[str, Any]


@dataclass
class ParseResult:
    """
    Complete result of parsing natural language input.
    
    Contains all structured components extracted from the input text:
    goals to pursue, percept for workspace, detected intent, entities,
    and updated conversation context.
    
    Attributes:
        goals: List of Goal objects generated from the input
        percept: Percept object for workspace processing
        intent: Classified intent with confidence
        entities: Extracted entities (names, topics, etc.)
        context: Updated conversation context
    """
    goals: List[Goal]
    percept: Percept
    intent: Intent
    entities: Dict[str, Any]
    context: Dict[str, Any]


class LanguageInputParser:
    """
    Converts natural language input into cognitive structures.
    
    The LanguageInputParser serves as the language input boundary for the cognitive
    architecture. It converts user text into structured Goals and Percepts that
    the cognitive core can process. Can use either LLM-powered parsing (primary)
    or rule-based pattern matching (fallback).
    
    Key Responsibilities:
    - Classify user intent (question, request, statement, etc.)
    - Extract entities (names, topics, temporal references, emotions)
    - Generate appropriate goals based on intent type
    - Create percepts using the perception subsystem
    - Track conversation context across turns
    - Maintain dialogue state for contextual understanding
    - Integrate with LLM for intelligent parsing
    - Provide fallback mechanism when LLM unavailable
    
    Integration Points:
    - PerceptionSubsystem: Uses perception for text encoding into embeddings
    - GlobalWorkspace: Generated goals are added to workspace
    - CognitiveCore: Parsed percepts enter the cognitive loop
    - LLMClient: Uses LLM for natural language understanding
    - FallbackInputParser: Uses rule-based parsing when LLM fails
    
    Design Philosophy:
    This is a PERIPHERAL component that converts language into non-linguistic
    cognitive structures. The actual cognitive processing operates on the Goals
    and Percepts produced by this parser, not on raw text.
    
    Attributes:
        perception: Reference to PerceptionSubsystem for text encoding
        llm_client: Optional LLM client for intelligent parsing
        fallback_parser: Rule-based parser for when LLM unavailable
        circuit_breaker: Monitors LLM failures and switches to fallback
        config: Configuration dictionary
        conversation_context: Dialogue state tracking
        use_llm: Whether to attempt LLM parsing
        enable_cache: Whether to cache common patterns
        parse_cache: Cache of recent parse results
    """
    
    def __init__(
        self, 
        perception_subsystem, 
        llm_client: Optional[LLMClient] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the language input parser.
        
        Args:
            perception_subsystem: PerceptionSubsystem instance for text encoding
            llm_client: Optional LLM client for intelligent parsing
            config: Optional configuration dictionary with keys:
                - use_fallback_on_error: Use fallback when LLM fails (default: True)
                - max_retries: Maximum retry attempts for LLM (default: 2)
                - timeout: Parse timeout in seconds (default: 5.0)
                - enable_cache: Cache common patterns (default: True)
        """
        self.perception = perception_subsystem
        self.llm_client = llm_client
        self.config = config or {}
        
        # Initialize fallback parser
        self.fallback_parser = FallbackInputParser(config)
        
        # Get circuit breaker
        self.circuit_breaker = get_input_circuit_breaker()
        
        # Configuration
        self.use_fallback_on_error = self.config.get("use_fallback_on_error", True)
        self.max_retries = self.config.get("max_retries", 2)
        self.timeout = self.config.get("timeout", 5.0)
        self.enable_cache = self.config.get("enable_cache", True)
        
        # Parse cache for common patterns
        self.parse_cache: Dict[str, Dict] = {}
        self.cache_size_limit = 100
        
        # Context tracking across conversation turns
        self.conversation_context = {
            "turn_count": 0,
            "recent_topics": [],
            "user_name": None
        }
        
        # Load intent classification patterns (for fallback)
        self._load_intent_patterns()
        
        use_llm = "with LLM" if llm_client else "fallback-only"
        logger.info(f"✅ LanguageInputParser initialized ({use_llm})")
    
    def _load_intent_patterns(self):
        """Define regular expression patterns for intent classification."""
        self.intent_patterns = {
            IntentType.QUESTION: [
                r"^(what|when|where|who|why|how|is|are|can|could|would|do|does)",
                r"\?$"
            ],
            IntentType.REQUEST: [
                r"^(please|could you|can you|would you|tell me|show me|help me)",
                r"(please|thanks|thank you)$"
            ],
            IntentType.GREETING: [
                r"^(hi|hello|hey|greetings|good morning|good afternoon)",
                r"(how are you|how're you)"
            ],
            IntentType.INTROSPECTION_REQUEST: [
                r"(how do you feel|what are you thinking|reflect on|your thoughts)",
                r"(examine yourself|introspect|self-assess)"
            ],
            IntentType.MEMORY_REQUEST: [
                r"(do you remember|recall|what did we|previously|earlier|last time)"
            ]
        }
    
    async def parse(self, text: str, context: Optional[Dict] = None) -> ParseResult:
        """
        Parse natural language input into cognitive structures.
        
        Main entry point for language input parsing. Converts raw text into
        structured Goals, Percepts, Intent, and entities that can be processed
        by the cognitive core.
        
        Attempts LLM-powered parsing first (if available), with automatic
        fallback to rule-based parsing on failure or when circuit breaker is open.
        
        Args:
            text: User input text to parse
            context: Optional additional context to merge
            
        Returns:
            ParseResult containing goals, percept, intent, entities, and updated context
        """
        # Update conversation context
        if context:
            self.conversation_context.update(context)
        self.conversation_context["turn_count"] += 1
        
        # Check cache first
        if self.enable_cache and text in self.parse_cache:
            logger.debug(f"🔍 Using cached parse result for: {text[:50]}")
            cached_result = self.parse_cache[text]
            return await self._build_parse_result(text, cached_result)
        
        # Try LLM parsing if available and circuit breaker allows
        if self.llm_client and self.circuit_breaker.can_attempt():
            try:
                parse_data = await self._parse_with_llm(text)
                self.circuit_breaker.record_success()
                
                # Cache successful parse
                if self.enable_cache:
                    self._add_to_cache(text, parse_data)
                
                return await self._build_parse_result(text, parse_data)
                
            except Exception as e:
                logger.warning(f"LLM parsing failed: {e}")
                self.circuit_breaker.record_failure()
                
                if not self.use_fallback_on_error:
                    raise
        
        # Fallback to rule-based parsing
        logger.debug("Using fallback rule-based parsing")
        parse_data = self.fallback_parser.parse(text, self.conversation_context)
        
        return await self._build_parse_result(text, parse_data)
    
    async def _parse_with_llm(self, text: str) -> Dict:
        """
        Parse input using LLM with structured output.
        
        Args:
            text: User input text
            
        Returns:
            Dictionary with parsed components
            
        Raises:
            LLMError: If parsing fails after retries
        """
        # Build conversation context for LLM
        conv_context = ConversationContext(
            turn_count=self.conversation_context["turn_count"],
            recent_topics=self.conversation_context.get("recent_topics", []),
            user_name=self.conversation_context.get("user_name")
        )
        
        # Create structured request
        request = LLMInputParseRequest(
            user_text=text,
            conversation_context=conv_context
        )
        
        # Build prompt for LLM
        prompt = self._build_llm_prompt(request)
        
        # Define expected schema
        schema = {
            "intent": {"type": "str", "confidence": "float"},
            "goals": "list[dict]",
            "entities": "dict",
            "context_updates": "dict",
            "confidence": "float"
        }
        
        # Retry logic with exponential backoff
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Call LLM with timeout
                result = await asyncio.wait_for(
                    self.llm_client.generate_structured(prompt, schema),
                    timeout=self.timeout
                )
                
                # Validate and return
                return self._validate_llm_response(result)
                
            except asyncio.TimeoutError:
                last_error = "Timeout"
                logger.warning(f"LLM parse timeout (attempt {attempt + 1}/{self.max_retries + 1})")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except LLMError as e:
                last_error = str(e)
                logger.warning(f"LLM parse error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
        
        # All retries exhausted
        raise LLMError(f"LLM parsing failed after {self.max_retries + 1} attempts: {last_error}")
    
    def _build_llm_prompt(self, request: LLMInputParseRequest) -> str:
        """
        Build prompt for LLM input parsing.
        
        Args:
            request: Structured parse request
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are a natural language understanding system that parses user input into structured cognitive components.

USER INPUT: {request.user_text}

CONVERSATION CONTEXT:
- Turn: {request.conversation_context.turn_count if request.conversation_context else 0}
- Recent topics: {', '.join(request.conversation_context.recent_topics if request.conversation_context else [])}

Parse this input and extract:
1. Intent type (question, request, statement, greeting, introspection_request, memory_request)
2. Goals to pursue (respond_to_user, retrieve_memory, introspect, etc.)
3. Entities (topics, temporal references, emotional tone, names)
4. Any context updates

Respond with JSON in this format:
{{
    "intent": {{
        "type": "question|request|statement|greeting|introspection_request|memory_request",
        "confidence": 0.0-1.0,
        "metadata": {{}}
    }},
    "goals": [
        {{
            "type": "respond_to_user|retrieve_memory|introspect|explore|maintain|other",
            "description": "Brief goal description",
            "priority": 0.0-1.0,
            "metadata": {{}}
        }}
    ],
    "entities": {{
        "topics": ["topic1", "topic2"],
        "temporal": ["time references"],
        "emotional_tone": "positive|negative|neutral",
        "names": ["extracted names"],
        "other": {{}}
    }},
    "context_updates": {{}},
    "confidence": 0.0-1.0
}}

JSON Response:"""
        
        return prompt
    
    def _validate_llm_response(self, response: Dict) -> Dict:
        """
        Validate and sanitize LLM response.
        
        Args:
            response: Raw LLM response dictionary
            
        Returns:
            Validated response
            
        Raises:
            LLMError: If response invalid
        """
        # Check required fields
        required_fields = ["intent", "goals", "entities"]
        for field in required_fields:
            if field not in response:
                raise LLMError(f"Missing required field: {field}")
        
        # Validate intent
        if "type" not in response["intent"]:
            raise LLMError("Intent missing 'type' field")
        
        # Ensure at least one goal
        if not response.get("goals"):
            response["goals"] = [{
                "type": "respond_to_user",
                "description": "Respond to user",
                "priority": 0.9,
                "metadata": {}
            }]
        
        # Add defaults
        response.setdefault("context_updates", {})
        response.setdefault("confidence", 0.9)
        
        return response
    
    def _add_to_cache(self, text: str, parse_data: Dict):
        """Add parse result to cache with size limit."""
        if len(self.parse_cache) >= self.cache_size_limit:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self.parse_cache))
            del self.parse_cache[first_key]
        
        self.parse_cache[text] = parse_data
    
    async def _build_parse_result(self, text: str, parse_data: Dict) -> ParseResult:
        """
        Build ParseResult from parse data.
        
        Args:
            text: Original input text
            parse_data: Parsed components dictionary
            
        Returns:
            Complete ParseResult object
        """
        # Extract intent
        intent_data = parse_data["intent"]
        intent = Intent(
            type=IntentType(intent_data["type"]),
            confidence=intent_data.get("confidence", 0.8),
            metadata=intent_data.get("metadata", {})
        )
        
        # Extract entities
        entities = parse_data.get("entities", {})
        
        # Generate goals
        goals = self._parse_goals(parse_data.get("goals", []), text, intent)
        
        # Create percept for workspace
        percept = await self._create_percept(text, intent, entities)
        
        # Update topic tracking
        if entities.get("topics"):
            for topic in entities["topics"]:
                if topic not in self.conversation_context["recent_topics"]:
                    self.conversation_context["recent_topics"].append(topic)
            # Keep only last 5 topics
            self.conversation_context["recent_topics"] = \
                self.conversation_context["recent_topics"][-5:]
        
        # Apply context updates
        context_updates = parse_data.get("context_updates", {})
        self.conversation_context.update(context_updates)
        
        result = ParseResult(
            goals=goals,
            percept=percept,
            intent=intent,
            entities=entities,
            context=self.conversation_context.copy()
        )
        
        logger.info(f"📝 Parsed input: intent={intent.type}, "
                   f"goals={len(goals)}, entities={list(entities.keys())}")
        
        return result
    
    def _parse_goals(self, goals_data: List[Dict], text: str, intent: Intent) -> List[Goal]:
        """
        Convert goals data to Goal objects.
        
        Args:
            goals_data: List of goal dictionaries
            text: Original input text
            intent: Classified intent
            
        Returns:
            List of Goal objects
        """
        goals = []
        
        goal_type_mapping = {
            "respond_to_user": GoalType.RESPOND_TO_USER,
            "retrieve_memory": GoalType.RETRIEVE_MEMORY,
            "introspect": GoalType.INTROSPECT,
            "learn": GoalType.LEARN,
            "create": GoalType.CREATE,
            "other": GoalType.RESPOND_TO_USER  # Map other to respond_to_user
        }
        
        for goal_data in goals_data:
            goal_type_str = goal_data.get("type", "other")
            goal_type = goal_type_mapping.get(goal_type_str, GoalType.RESPOND_TO_USER)
            
            goals.append(Goal(
                type=goal_type,
                description=goal_data.get("description", "Process user input"),
                priority=goal_data.get("priority", 0.8),
                progress=0.0,
                metadata=goal_data.get("metadata", {})
            ))
        
        # Ensure at least one goal exists
        if not goals:
            goals.append(Goal(
                type=GoalType.RESPOND_TO_USER,
                description=f"Respond to user {intent.type}",
                priority=0.9,
                progress=0.0,
                metadata={"intent": intent.type, "user_input": text[:100]}
            ))
        
        return goals
    
    def _classify_intent(self, text: str) -> Intent:
        """
        Classify user intent using pattern matching.
        
        Uses regular expression patterns to identify the primary intent
        of the user input. Scores each intent type and returns the highest
        scoring one, defaulting to STATEMENT if no patterns match.
        
        More specific patterns (memory_request, introspection_request) are
        checked with higher priority to avoid conflicts with generic patterns.
        
        Args:
            text: Input text to classify
            
        Returns:
            Intent object with type, confidence, and metadata
        """
        text_lower = text.lower().strip()
        
        # Check high-priority specific patterns first
        # These are more specific and should override generic patterns
        # REQUEST before QUESTION so "can you..." matches REQUEST, not QUESTION
        high_priority_intents = [
            IntentType.MEMORY_REQUEST,
            IntentType.INTROSPECTION_REQUEST,
            IntentType.GREETING,
            IntentType.REQUEST,
        ]
        
        for intent_type in high_priority_intents:
            if intent_type in self.intent_patterns:
                for pattern in self.intent_patterns[intent_type]:
                    if re.search(pattern, text_lower):
                        return Intent(
                            type=intent_type,
                            confidence=0.9,  # High confidence for specific matches
                            metadata={}
                        )
        
        # Now check remaining patterns
        intent_scores = {intent_type: 0.0 for intent_type in IntentType}
        
        for intent_type, patterns in self.intent_patterns.items():
            if intent_type not in high_priority_intents:  # Skip already checked
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        intent_scores[intent_type] += 0.5
        
        # Get the highest scoring intent
        top_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        if top_intent[1] > 0:
            return Intent(
                type=top_intent[0],
                confidence=min(top_intent[1], 1.0),
                metadata={}
            )
        
        # Default to statement if no patterns match
        return Intent(
            type=IntentType.STATEMENT,
            confidence=0.5,
            metadata={}
        )
    
    def _generate_goals(self, text: str, intent: Intent, entities: Dict) -> List[Goal]:
        """
        Convert intent to appropriate goals.
        
        Different intent types generate different goals. For example:
        - MEMORY_REQUEST → RETRIEVE_MEMORY goal
        - INTROSPECTION_REQUEST → INTROSPECT goal
        - QUESTION (with memory keywords) → RETRIEVE_MEMORY goal
        - All intents → RESPOND_TO_USER goal
        
        Args:
            text: Original input text
            intent: Classified intent
            entities: Extracted entities
            
        Returns:
            List of Goal objects to add to workspace
        """
        goals = []
        
        # Always create response goal
        goals.append(Goal(
            type=GoalType.RESPOND_TO_USER,
            description=f"Respond to user {intent.type}",
            priority=0.9,
            progress=0.0,
            metadata={
                "intent": intent.type,
                "user_input": text[:100]  # Truncate for metadata
            }
        ))
        
        # Intent-specific goals
        if intent.type == IntentType.MEMORY_REQUEST:
            goals.append(Goal(
                type=GoalType.RETRIEVE_MEMORY,
                description=f"Retrieve memories about: {text[:50]}",
                priority=0.8,
                progress=0.0,
                metadata={"query": text}
            ))
        
        elif intent.type == IntentType.INTROSPECTION_REQUEST:
            goals.append(Goal(
                type=GoalType.INTROSPECT,
                description="Perform introspection as requested",
                priority=0.7,
                progress=0.0,
                metadata={"trigger": "user_request"}
            ))
        
        elif intent.type == IntentType.QUESTION:
            # Questions may need memory retrieval if they reference the past
            if any(kw in text.lower() for kw in ["remember", "recall", "earlier", "before"]):
                goals.append(Goal(
                    type=GoalType.RETRIEVE_MEMORY,
                    description="Search memory for answer",
                    priority=0.6,
                    progress=0.0,
                    metadata={"query": text}
                ))
        
        return goals
    
    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text using simple pattern matching.
        
        Extracts:
        - Names: Capitalized words (potential proper nouns)
        - Topics: Nouns following "about"
        - Temporal: Time references (today, yesterday, etc.)
        - Emotions: Emotional keywords and their valence
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        # Extract names (capitalized words)
        # Filter out common non-name words
        common_words = {
            "hi", "hello", "hey", "i", "you", "the", "a", "an", "my", "me", "is", "am",
            "what", "when", "where", "who", "why", "how", "can", "could", "would", "should",
            "please", "tell", "show", "help", "give", "make", "do", "does", "did"
        }
        name_pattern = r'\b([A-Z][a-z]+)\b'
        names = re.findall(name_pattern, text)
        if names:
            # Filter out common greeting/question words and short words
            filtered_names = [n for n in names if n.lower() not in common_words and len(n) > 2]
            if filtered_names:
                entities["names"] = filtered_names
                # Update context with user name if not already set
                if self.conversation_context["user_name"] is None and filtered_names:
                    self.conversation_context["user_name"] = filtered_names[0]
        
        # Extract topics (nouns after "about")
        topic_pattern = r'about ([\w\s]+?)(?:\.|,|\?|$)'
        topics = re.findall(topic_pattern, text.lower())
        if topics:
            entities["topic"] = topics[0].strip()
        
        # Extract temporal references
        temporal_keywords = ["today", "yesterday", "tomorrow", "earlier", "later", "now"]
        for kw in temporal_keywords:
            if kw in text.lower():
                entities["temporal"] = kw
                break
        
        # Extract emotional keywords
        emotion_keywords = {
            "positive": ["happy", "excited", "joyful", "pleased", "grateful"],
            "negative": ["sad", "angry", "frustrated", "worried", "anxious"],
        }
        
        for valence, keywords in emotion_keywords.items():
            for kw in keywords:
                if kw in text.lower():
                    entities["user_emotion"] = {"valence": valence, "keyword": kw}
                    break
        
        return entities
    
    async def _create_percept(self, text: str, intent: Intent, entities: Dict) -> Percept:
        """
        Create percept from parsed input using perception subsystem.
        
        Uses the perception subsystem to encode the text into an embedding,
        then enhances the percept with parsing metadata (intent, entities, etc.)
        and adjusts complexity based on intent type.
        
        Args:
            text: Original input text
            intent: Classified intent
            entities: Extracted entities
            
        Returns:
            Percept object ready for workspace processing
        """
        # Use perception subsystem to encode text
        percept = await self.perception.encode(text, modality="text")
        
        # Enhance with parsing metadata
        percept.metadata.update({
            "intent": intent.type,
            "intent_confidence": intent.confidence,
            "entities": entities,
            "turn_count": self.conversation_context["turn_count"]
        })
        
        # Adjust complexity based on intent type
        if intent.type == IntentType.QUESTION:
            percept.complexity += 10  # Questions require more processing
        elif intent.type == IntentType.INTROSPECTION_REQUEST:
            percept.complexity += 15  # Introspection is more complex
        
        return percept
