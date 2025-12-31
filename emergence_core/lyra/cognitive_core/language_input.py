"""
Language Input Parser: Natural language to cognitive structures.

This module implements the LanguageInputParser class, which converts natural language
user input into structured Goals and Percepts that the cognitive core can process.
This is how Lyra "hears" and understands language.

The language input parser is responsible for:
- Converting natural language into structured cognitive formats
- Intent classification (question, request, statement, etc.)
- Goal generation from user intents
- Entity extraction (names, topics, temporal references, emotions)
- Context tracking across conversation turns
- Integration with perception subsystem for text encoding
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from .workspace import Goal, GoalType, Percept

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
    the cognitive core can process. This is a rule-based parser (v1) that uses
    pattern matching for intent classification and simple extraction for entities.
    
    Key Responsibilities:
    - Classify user intent (question, request, statement, etc.)
    - Extract entities (names, topics, temporal references, emotions)
    - Generate appropriate goals based on intent type
    - Create percepts using the perception subsystem
    - Track conversation context across turns
    - Maintain dialogue state for contextual understanding
    
    Integration Points:
    - PerceptionSubsystem: Uses perception for text encoding into embeddings
    - GlobalWorkspace: Generated goals are added to workspace
    - CognitiveCore: Parsed percepts enter the cognitive loop
    
    Design Philosophy:
    This is a PERIPHERAL component that converts language into non-linguistic
    cognitive structures. The actual cognitive processing operates on the Goals
    and Percepts produced by this parser, not on raw text.
    
    Attributes:
        perception: Reference to PerceptionSubsystem for text encoding
        config: Configuration dictionary
        conversation_context: Dialogue state tracking
        intent_patterns: Regular expression patterns for intent classification
    """
    
    def __init__(self, perception_subsystem, config: Optional[Dict] = None):
        """
        Initialize the language input parser.
        
        Args:
            perception_subsystem: PerceptionSubsystem instance for text encoding
            config: Optional configuration dictionary
        """
        self.perception = perception_subsystem
        self.config = config or {}
        
        # Context tracking across conversation turns
        self.conversation_context = {
            "turn_count": 0,
            "recent_topics": [],
            "user_name": None
        }
        
        # Load intent classification patterns
        self._load_intent_patterns()
        
        logger.info("✅ LanguageInputParser initialized")
    
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
        
        # Classify intent
        intent = self._classify_intent(text)
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Generate goals based on intent
        goals = self._generate_goals(text, intent, entities)
        
        # Create percept for workspace
        percept = await self._create_percept(text, intent, entities)
        
        # Update topic tracking
        if entities.get("topic"):
            self.conversation_context["recent_topics"].append(entities["topic"])
            # Keep only last 5 topics
            self.conversation_context["recent_topics"] = \
                self.conversation_context["recent_topics"][-5:]
        
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
        high_priority_intents = [
            IntentType.MEMORY_REQUEST,
            IntentType.INTROSPECTION_REQUEST,
            IntentType.GREETING,
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
