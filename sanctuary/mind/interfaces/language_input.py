"""
Language Input Parser: Peripheral text input processing.

This module implements the LanguageInputParser class, which converts user text
into structured internal representations (goals, percepts, facts). It uses LLMs
for parsing but outputs structured data, not text. This is a peripheral interface,
not part of the core cognitive substrate.

The language input parser is responsible for:
- Converting natural language into structured internal formats
- Extracting goals, intentions, and requests from user input
- Identifying factual claims and assertions
- Detecting emotional tone and pragmatic intent
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class InputIntent(Enum):
    """
    Types of communicative intent in user input.

    QUERY: Requesting information or knowledge
    COMMAND: Requesting action or behavior
    STATEMENT: Asserting facts or beliefs
    SOCIAL: Social interaction (greetings, emotions)
    META: Discussion about the system itself
    """
    QUERY = "query"
    COMMAND = "command"
    STATEMENT = "statement"
    SOCIAL = "social"
    META = "meta"


@dataclass
class ParsedInput:
    """
    Structured representation of parsed natural language input.

    The parsed input breaks down natural language into components that can
    be directly used by the cognitive core: goals to pursue, percepts to
    attend to, facts to store, and emotional context to incorporate.

    Attributes:
        raw_text: Original user input
        intent: Primary communicative intent
        goals: Extracted goals or requests
        facts: Factual claims or assertions
        emotional_tone: Detected emotional valence (-1.0 to +1.0)
        entities: Extracted named entities or key concepts
        urgency: Time-sensitivity of the input (0.0-1.0)
        metadata: Additional parsing metadata
    """
    raw_text: str
    intent: InputIntent
    goals: List[str]
    facts: List[str]
    emotional_tone: float
    entities: List[str]
    urgency: float
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class LanguageInputParser:
    """
    Converts user text into structured internal representations.

    The LanguageInputParser serves as the input boundary between natural language
    and the internal cognitive architecture. It uses LLMs for parsing and semantic
    understanding, but its output is structured data (goals, percepts, facts), not
    generated text. This keeps language processing at the periphery while the core
    operates on non-linguistic representations.

    Key Responsibilities:
    - Parse natural language input into structured components
    - Extract goals, intentions, and action requests
    - Identify factual statements and beliefs
    - Detect emotional tone and pragmatic context
    - Handle multi-turn dialogue context
    - Resolve ambiguities using conversational context

    Integration Points:
    - PerceptionSubsystem: Parsed input becomes text percepts (via embedding)
    - GlobalWorkspace: Extracted goals enter as conscious intentions
    - ActionSubsystem: Commands trigger action consideration
    - AffectSubsystem: Emotional tone influences affective state
    - CognitiveCore: Parsed input feeds into the cognitive loop

    Parsing Process:
    1. Text → Semantic Analysis (using LLM)
       - Identify intent (query, command, statement, etc.)
       - Extract entities and key concepts
       - Detect emotional tone and urgency

    2. Structured Extraction
       - Goals: "Please help me understand X" → goal: "explain X"
       - Facts: "Paris is the capital of France" → fact: "capital(Paris, France)"
       - Commands: "Search for recent papers" → action: "search(papers, recent)"

    3. Context Integration
       - Use dialogue history for pronoun resolution
       - Maintain conversation state across turns
       - Detect topic shifts and continuations

    4. Output Structured Representation
       - ParsedInput object with all extracted components
       - Ready for consumption by cognitive subsystems

    Design Philosophy:
    This is explicitly a PERIPHERAL component. The LLM here is used as a tool for
    parsing language, not as the cognitive substrate. The actual "mind" operates
    on the structured representations (goals, percepts, embeddings) produced by
    this parser, not on the raw text or LLM-generated outputs.

    Attributes:
        llm_model: Language model used for parsing
        context_window: Recent dialogue history for context
        entity_extractor: Tool for named entity recognition
    """

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Meta-Llama-3-8B",
        context_window_size: int = 10,
        max_input_length: int = 2048,
    ) -> None:
        """
        Initialize the language input parser.

        Args:
            llm_model_name: Name/path of LLM to use for parsing
            context_window_size: Number of previous turns to maintain in context
            max_input_length: Maximum length of input to process (characters)
        """
        pass
