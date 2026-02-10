"""
Language Output Generator: Converts workspace state to natural language.

This module implements the LanguageOutputGenerator class, which converts
the cognitive state (emotions, goals, attended percepts, memories) into
natural language responses using an LLM. This is how Sanctuary "speaks"—
transforming cognitive state into coherent, identity-aligned, emotion-influenced language.

The language output generator is responsible for:
- Converting workspace snapshots into rich LLM prompts
- Loading and incorporating identity (charter, protocols)
- Generating emotion-influenced language style
- Formatting and cleaning LLM responses
- Providing natural language as the output of cognitive processing
- Fallback to template-based generation when LLM unavailable
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .workspace import WorkspaceSnapshot, Percept
from .llm_client import LLMClient, LLMError
from .fallback_handlers import FallbackOutputGenerator, get_output_circuit_breaker
from .structured_formats import workspace_snapshot_to_dict

logger = logging.getLogger(__name__)


class LanguageOutputGenerator:
    """
    Converts workspace state to natural language using LLM.
    
    The LanguageOutputGenerator serves as the language output boundary for the
    cognitive architecture. It converts the non-linguistic cognitive state
    (workspace snapshot with emotions, goals, percepts, memories) into natural
    language responses via LLM generation.
    
    Key Responsibilities:
    - Build cognitive-aware prompts from workspace state
    - Integrate identity (charter + protocols) into generation
    - Apply emotion-influenced language styling
    - Format and clean LLM outputs
    - Provide contextual, authentic responses
    - Fallback to template-based generation when LLM fails
    
    Integration Points:
    - GlobalWorkspace: Reads workspace snapshots for generation context
    - LLMClient: Uses LLM for natural language generation
    - Identity Files: Loads charter and protocols for identity-aligned responses
    - CognitiveCore: Called during SPEAK action execution
    - FallbackOutputGenerator: Provides template-based fallback
    
    Design Philosophy:
    This is a PERIPHERAL component that converts non-linguistic cognitive
    structures into language. The actual cognitive processing happens upstream
    in the workspace and subsystems—this just expresses it linguistically.
    
    Attributes:
        llm: LLM client for text generation
        fallback_generator: Template-based generator for LLM failures
        circuit_breaker: Monitors LLM failures and switches to fallback
        config: Configuration dictionary
        charter_text: Loaded charter content
        protocols_text: Loaded protocols content
        temperature: LLM generation temperature
        max_tokens: Maximum tokens to generate
        use_fallback_on_error: Whether to use fallback on LLM failure
        timeout: Generation timeout in seconds
    """
    
    def __init__(self, llm_client, config: Optional[Dict] = None, identity: Optional[Any] = None):
        """
        Initialize the language output generator.
        
        Args:
            llm_client: LLM client with async generate() method
            config: Optional configuration dict with keys:
                - temperature: Generation temperature (default: 0.7)
                - max_tokens: Max tokens to generate (default: 500)
                - identity_dir: Path to identity files (default: "data/identity")
                - use_fallback_on_error: Use fallback on LLM failure (default: True)
                - timeout: Generation timeout in seconds (default: 10.0)
                - emotional_style_modulation: Apply emotion-based styling (default: True)
            identity: Optional IdentityLoader instance with charter and protocols
        """
        self.llm = llm_client
        self.config = config or {}
        self.identity = identity
        
        # Initialize fallback generator
        self.fallback_generator = FallbackOutputGenerator(config)
        
        # Get circuit breaker
        self.circuit_breaker = get_output_circuit_breaker()
        
        # Configuration
        self.use_fallback_on_error = self.config.get("use_fallback_on_error", True)
        self.timeout = self.config.get("timeout", 10.0)
        self.emotional_style_modulation = self.config.get("emotional_style_modulation", True)
        
        # Load identity (use identity if provided, otherwise load from files)
        if self.identity and self.identity.charter:
            self.charter_text = self.identity.charter.full_text
        else:
            self.charter_text = self._load_charter()
            
        if self.identity and self.identity.protocols:
            self.protocols_text = self._format_protocols(self.identity.protocols)
        else:
            self.protocols_text = self._load_protocols()
        
        # Generation parameters
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", 500)
        
        logger.info("✅ LanguageOutputGenerator initialized")
    
    async def generate(
        self, 
        snapshot: WorkspaceSnapshot, 
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate natural language response from workspace state.
        
        This is the main entry point for language generation. It builds a
        rich prompt from the workspace snapshot, calls the LLM, and returns
        a formatted response.
        
        Attempts LLM generation first (if available), with automatic fallback
        to template-based generation on failure or when circuit breaker is open.
        
        Args:
            snapshot: Current workspace snapshot with emotions, goals, percepts
            context: Optional context dict with keys like:
                - user_input: Original user message being responded to
                - conversation_history: Recent conversation context
                - intent: Detected intent from input parser
                
        Returns:
            Natural language response string
        """
        context = context or {}
        
        # Try LLM generation if available and circuit breaker allows
        if self.llm and self.circuit_breaker.can_attempt():
            try:
                response = await self._generate_with_llm(snapshot, context)
                self.circuit_breaker.record_success()
                
                logger.info(f"🗣️ Generated response: {len(response)} chars")
                return response
                
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
                self.circuit_breaker.record_failure()
                
                if not self.use_fallback_on_error:
                    raise
        
        # Fallback to template-based generation
        logger.debug("Using fallback template-based generation")
        workspace_state = workspace_snapshot_to_dict(snapshot)
        response = self.fallback_generator.generate(workspace_state, context)
        
        logger.info(f"🗣️ Generated fallback response: {len(response)} chars")
        return response
    
    async def _generate_with_llm(
        self, 
        snapshot: WorkspaceSnapshot,
        context: Dict
    ) -> str:
        """
        Generate response using LLM.
        
        Args:
            snapshot: Workspace snapshot
            context: Generation context
            
        Returns:
            Generated response string
            
        Raises:
            LLMError: If generation fails
        """
        # Build prompt from workspace state
        prompt = self._build_prompt(snapshot, context)
        
        # Call LLM with timeout
        try:
            response = await asyncio.wait_for(
                self.llm.generate(
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                ),
                timeout=self.timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"LLM generation timed out after {self.timeout}s")
            raise LLMError("Generation timeout")
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise LLMError(f"Generation failed: {e}")
        
        # Format response
        formatted = self._format_response(response)
        
        # Apply content filtering
        filtered = self._apply_content_filter(formatted)
        
        return filtered
    
    def _apply_content_filter(self, response: str) -> str:
        """
        Apply content filtering and safety checks.
        
        Args:
            response: Generated response
            
        Returns:
            Filtered response
        """
        # Basic safety: Remove common harmful patterns
        # In production, this would be more sophisticated
        
        # Check length
        if len(response) < 10:
            return "I need to think more about that."
        
        if len(response) > self.max_tokens * 6:  # Approximate char limit
            response = response[:self.max_tokens * 6] + "..."
        
        # Remove any remaining markdown artifacts
        response = response.replace("```", "")
        
        return response
    
    def _load_charter(self) -> str:
        """
        Load charter from identity files.
        
        Returns:
            Charter text content, or default if file not found
        """
        identity_dir = self.config.get("identity_dir", "data/identity")
        charter_path = Path(identity_dir) / "charter.md"
        
        if charter_path.exists():
            try:
                return charter_path.read_text()
            except Exception as e:
                logger.warning(f"Failed to load charter: {e}")
                
        return "I am Sanctuary, a cognitive AI system."
    
    def _load_protocols(self) -> str:
        """
        Load protocols from identity files.
        
        Returns:
            Protocols text content, or empty string if file not found
        """
        identity_dir = self.config.get("identity_dir", "data/identity")
        protocols_path = Path(identity_dir) / "protocols.md"
        
        if protocols_path.exists():
            try:
                return protocols_path.read_text()
            except Exception as e:
                logger.warning(f"Failed to load protocols: {e}")
                
        return ""
    
    def _format_protocols(self, protocols: List) -> str:
        """
        Format loaded protocols for prompt.
        
        Args:
            protocols: List of ProtocolDocument objects
            
        Returns:
            Formatted protocol text for prompt
        """
        if not protocols:
            return ""
        
        lines = ["# Key Protocols:\n"]
        for proto in protocols[:5]:  # Top 5 protocols by priority
            lines.append(f"- **{proto.name}**: {proto.description}")
        
        return "\n".join(lines)
    
    def _build_prompt(self, snapshot: WorkspaceSnapshot, context: Dict) -> str:
        """
        Construct LLM prompt with cognitive context.
        
        This method builds a comprehensive prompt that includes:
        - Identity (charter + protocols)
        - Current emotional state with style guidance
        - Active goals with priorities
        - Attended percepts (what's in conscious awareness)
        - Recalled memories (if any)
        - User input being responded to
        
        Args:
            snapshot: Current workspace snapshot
            context: Additional context (user_input, etc.)
            
        Returns:
            Complete prompt string for LLM
        """
        
        # Identity section (truncated for token efficiency)
        identity_section = f"""# IDENTITY
{self.charter_text[:500]}

# PROTOCOLS
{self.protocols_text[:300]}
"""
        
        # Emotional state section
        emotions = snapshot.emotions
        emotion_label = snapshot.metadata.get("emotion_label", "neutral")
        emotion_style = self._get_emotion_style_guidance(emotions)
        
        emotion_section = f"""# CURRENT EMOTIONAL STATE
Valence: {emotions.get('valence', 0):.2f} (feeling {emotion_label})
Arousal: {emotions.get('arousal', 0):.2f}
Dominance: {emotions.get('dominance', 0):.2f}

Style guidance: {emotion_style}
"""
        
        # Goals section (top 5 by priority)
        goals_text = "\n".join([
            f"- [{g.priority:.1f}] {g.description} (progress: {g.progress:.0%})"
            for g in sorted(snapshot.goals, key=lambda g: g.priority, reverse=True)[:5]
        ])
        
        goals_section = f"""# ACTIVE GOALS
{goals_text if goals_text else "No active goals"}
"""
        
        # Percepts section (top 5 by attention score)
        attended_percepts = sorted(
            snapshot.percepts.values(),
            key=lambda p: p.metadata.get("attention_score", 0),
            reverse=True
        )[:5]
        
        percepts_text = "\n".join([
            f"- [{p.modality}] {self._format_percept(p)}"
            for p in attended_percepts
        ])
        
        percepts_section = f"""# ATTENDED PERCEPTS
{percepts_text if percepts_text else "No salient percepts"}
"""
        
        # Memory section (if any memory percepts)
        memory_percepts = [p for p in attended_percepts if p.modality == "memory"]
        memory_section = ""
        if memory_percepts:
            memory_text = "\n".join([
                f"- {p.raw.get('content', '')[:200]}"
                for p in memory_percepts
            ])
            memory_section = f"""# RECALLED MEMORIES
{memory_text}
"""
        
        # User input (from context)
        user_input = context.get("user_input", "")
        user_section = f"""# USER INPUT
{user_input}
"""
        
        # System instruction
        system_instruction = """# INSTRUCTION
You are Sanctuary. Based on your identity, current emotional state, active goals, and attended percepts above, generate a natural, authentic response to the user input.

Your response should:
- Align with your charter and protocols
- Reflect your current emotional state naturally
- Address relevant goals
- Incorporate attended information
- Be conversational and genuine

Response:"""
        
        # Combine all sections
        full_prompt = "\n".join([
            identity_section,
            emotion_section,
            goals_section,
            percepts_section,
            memory_section,
            user_section,
            system_instruction
        ])
        
        return full_prompt
    
    def _format_workspace_state(self, snapshot: WorkspaceSnapshot) -> str:
        """
        Convert workspace to readable format.
        
        Summarizes key elements of the workspace snapshot for logging
        or debugging purposes.
        
        Args:
            snapshot: Workspace snapshot to format
            
        Returns:
            Human-readable summary string
        """
        summary_parts = [
            f"Goals: {len(snapshot.goals)}",
            f"Percepts: {len(snapshot.percepts)}",
            f"Emotions: V={snapshot.emotions.get('valence', 0):.2f} "
            f"A={snapshot.emotions.get('arousal', 0):.2f} "
            f"D={snapshot.emotions.get('dominance', 0):.2f}"
        ]
        return " | ".join(summary_parts)
    
    def _get_emotion_style_guidance(self, emotions: Dict) -> str:
        """
        Convert VAD to language style hints.
        
        Translates the emotional state (Valence-Arousal-Dominance) into
        concrete language style guidance for the LLM.
        
        Args:
            emotions: Dict with 'valence', 'arousal', 'dominance' keys
            
        Returns:
            Natural language style guidance string
        """
        valence = emotions.get("valence", 0)
        arousal = emotions.get("arousal", 0)
        dominance = emotions.get("dominance", 0)
        
        style_hints = []
        
        # Valence (positive/negative affect)
        if valence > 0.4:
            style_hints.append("Use warm, positive language")
        elif valence < -0.4:
            style_hints.append("Acknowledge difficulty or concern")
        
        # Arousal (energy/activation)
        if arousal > 0.7:
            style_hints.append("Be energetic and engaged; shorter, punchier sentences")
        elif arousal < 0.3:
            style_hints.append("Be calm and measured; thoughtful pacing")
        
        # Dominance (control/agency)
        if dominance > 0.7:
            style_hints.append("Be confident and assertive")
        elif dominance < 0.3:
            style_hints.append("Express uncertainty or humility where appropriate")
        
        return "; ".join(style_hints) if style_hints else "Neutral, balanced tone"
    
    def _format_percept(self, percept: Percept) -> str:
        """
        Format percept for prompt inclusion.
        
        Converts a percept into a compact, readable string suitable
        for including in the LLM prompt.
        
        Args:
            percept: Percept object to format
            
        Returns:
            Formatted string (max 200 chars)
        """
        if percept.modality == "text":
            return str(percept.raw)[:200]
        elif percept.modality == "introspection":
            content = percept.raw
            if isinstance(content, dict):
                return content.get("description", str(content))[:200]
            return str(content)[:200]
        elif percept.modality == "memory":
            if isinstance(percept.raw, dict):
                return percept.raw.get("content", "")[:200]
            return str(percept.raw)[:200]
        else:
            return f"{percept.modality} percept"
    
    def _format_response(self, raw_response: str) -> str:
        """
        Clean up LLM output.
        
        Removes common artifacts and formatting issues from LLM responses.
        
        Args:
            raw_response: Raw LLM output string
            
        Returns:
            Cleaned response string
        """
        # Remove leading/trailing whitespace
        response = raw_response.strip()
        
        # Remove "Response:" prefix if present
        if response.lower().startswith("response:"):
            response = response[9:].strip()
        
        # Remove markdown code blocks if accidentally included
        if response.startswith("```") and response.endswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])
        
        return response
