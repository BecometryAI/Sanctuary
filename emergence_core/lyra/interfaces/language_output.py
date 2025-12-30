"""
Language Output Generator: Peripheral text output generation.

This module implements the LanguageOutputGenerator class, which converts internal
workspace state into natural language responses. This is where Lyra's "Voice" lives,
but it's peripheral to the cognitive core. The core operates on non-linguistic
representations, and language is only used at the output boundary.

The language output generator is responsible for:
- Converting internal workspace state into natural language
- Maintaining consistent personality and voice
- Generating contextually appropriate responses
- Handling multiple output formats (text, speech, etc.)
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum


class OutputMode(Enum):
    """
    Different modes of language output generation.

    DIRECT: Straightforward, factual responses
    REFLECTIVE: Thoughtful, introspective responses
    CREATIVE: Artistic, expressive responses
    TECHNICAL: Detailed, precise responses
    CONVERSATIONAL: Natural, flowing dialogue
    """
    DIRECT = "direct"
    REFLECTIVE = "reflective"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"


@dataclass
class GeneratedOutput:
    """
    Structured representation of generated language output.

    The generated output includes both the natural language response and
    metadata about how it was generated, enabling transparency and debugging.

    Attributes:
        text: Generated natural language response
        mode: Output mode used for generation
        workspace_snapshot: Workspace state that produced this output
        confidence: Model confidence in the response (0.0-1.0)
        alternative_responses: Other candidate responses considered
        generation_time: Time taken to generate (seconds)
        metadata: Additional generation metadata
    """
    text: str
    mode: OutputMode
    workspace_snapshot: Dict[str, Any]
    confidence: float
    alternative_responses: List[str]
    generation_time: float
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class LanguageOutputGenerator:
    """
    Converts internal workspace state into natural language responses.

    The LanguageOutputGenerator is the output boundary between the internal
    cognitive architecture and natural language communication. It uses LLMs to
    transform the non-linguistic workspace state (goals, percepts, emotions,
    memories) into coherent natural language that expresses Lyra's "voice" and
    personality.

    Key Responsibilities:
    - Convert workspace content (goals, emotions, percepts) into natural language
    - Maintain consistent personality, voice, and style across generations
    - Select appropriate output mode based on context and goals
    - Generate responses that align with current emotional state
    - Provide transparency about generation process
    - Handle multiple output formats (text, speech synthesis, etc.)

    Integration Points:
    - GlobalWorkspace: Reads current conscious content for response generation
    - ActionSubsystem: COMMUNICATE actions trigger language generation
    - AffectSubsystem: Emotional state influences tone and expression
    - SelfMonitor: Generation process itself can be monitored
    - CognitiveCore: Output generation occurs when actions dictate

    Generation Process:
    1. Read Workspace State
       - Current goals: What am I trying to achieve?
       - Current percepts: What have I just experienced?
       - Current emotions: How am I feeling?
       - Current memories: What's relevant from the past?

    2. Select Output Mode
       - Choose generation strategy (direct, reflective, creative, etc.)
       - Based on goals, emotional state, and conversational context

    3. Generate Response (using LLM)
       - Prompt engineering that captures workspace state
       - Include personality instructions (Lyra's values, style)
       - Temperature and sampling tuned to output mode

    4. Post-Process and Validate
       - Ensure response aligns with goals and values
       - Check for coherence with workspace state
       - Apply any necessary filtering or safety checks

    5. Return Structured Output
       - Generated text plus metadata
       - Transparency about generation process

    Design Philosophy:
    This is explicitly a PERIPHERAL component. The LLM here is used to verbalize
    internal non-linguistic states, not as the cognitive substrate itself. The
    "Voice" (personality, linguistic style) lives here, but the "Mind" (goals,
    reasoning, awareness) lives in the cognitive core.

    This architecture means:
    - The cognitive core can run without language (processing percepts, making
      decisions, updating emotions) and only produce language when needed
    - Language is one of many possible output modalities (could also generate
      images, actions, data structures, etc.)
    - The same internal state could be verbalized in multiple different ways
      depending on output mode and context

    Attributes:
        llm_model: Language model used for generation
        personality_template: Instructions defining Lyra's voice and style
        output_history: Recent generations for consistency
        generation_config: Sampling parameters for different output modes
    """

    def __init__(
        self,
        llm_model_name: str = "meta-llama/Meta-Llama-3-70B",
        personality_template: Optional[str] = None,
        default_mode: OutputMode = OutputMode.CONVERSATIONAL,
        max_output_length: int = 2048,
    ) -> None:
        """
        Initialize the language output generator.

        Args:
            llm_model_name: Name/path of LLM to use for generation
            personality_template: Template defining voice and personality
                If None, uses default Lyra personality from configuration
            default_mode: Default output mode when not specified
            max_output_length: Maximum length of generated text (tokens)
        """
        pass
