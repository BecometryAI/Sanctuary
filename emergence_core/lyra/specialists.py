"""
Specialist models for the adaptive router.

This module defines the specialist system: Philosopher, Pragmatist, Artist, and Voice.
Each specialist is a specialized model that processes specific types of queries.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Constants
MAX_IMAGE_PIXELS = 89478485  # PIL default safe limit


@dataclass
class SpecialistOutput:
    """
    Output from a specialist model.
    
    Attributes:
        content: The main response content
        confidence: Confidence score (0.0-1.0)
        metadata: Additional metadata about the processing
        specialist_type: Which specialist generated this
    """
    content: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    specialist_type: str = "unknown"


class BaseSpecialist:
    """Base class for all specialists."""
    
    def __init__(self, model_path: str, base_dir: Path, development_mode: bool = False):
        """
        Initialize specialist.
        
        Args:
            model_path: Path to model files
            base_dir: Base directory for data
            development_mode: If True, skip model loading for testing
        """
        self.model_path = model_path
        self.base_dir = Path(base_dir)
        self.development_mode = development_mode
        self.model = None
        
        if not development_mode:
            self._load_model()
    
    def _load_model(self):
        """Load the model. Override in subclasses."""
        pass
    
    async def process(self, message: str, context: Dict[str, Any]) -> SpecialistOutput:
        """
        Process a message with this specialist.
        
        Args:
            message: Input message
            context: Additional context
            
        Returns:
            SpecialistOutput with response
        """
        raise NotImplementedError("Subclasses must implement process()")


class PhilosopherSpecialist(BaseSpecialist):
    """
    Philosopher specialist for ethical reflection and meta-cognition.
    Uses Jamba 52B model.
    """
    
    def __init__(self, base_dir: Path, development_mode: bool = False):
        model_path = "ai21labs/Jamba-v0.1"
        super().__init__(model_path, base_dir, development_mode)
        self.specialist_type = "philosopher"
    
    async def process(self, message: str, context: Dict[str, Any]) -> SpecialistOutput:
        """Process with philosophical reasoning."""
        if self.development_mode:
            return SpecialistOutput(
                content="[MOCK] Philosophical reflection on: " + message[:50],
                confidence=0.9,
                metadata={"role": "philosopher", "mode": "mock"},
                specialist_type=self.specialist_type
            )
        
        # Real implementation would call Jamba model here
        # For now, return a placeholder
        return SpecialistOutput(
            content="Philosophical response placeholder",
            confidence=0.8,
            metadata={"role": "philosopher"},
            specialist_type=self.specialist_type
        )


class PragmatistSpecialist(BaseSpecialist):
    """
    Pragmatist specialist for tool use and practical reasoning.
    Uses Llama-3.3-Nemotron-49B model.
    """
    
    def __init__(self, base_dir: Path, development_mode: bool = False):
        model_path = "nvidia/Llama-3.3-Nemotron-70B-Instruct"
        super().__init__(model_path, base_dir, development_mode)
        self.specialist_type = "pragmatist"
    
    async def process(self, message: str, context: Dict[str, Any]) -> SpecialistOutput:
        """Process with practical reasoning and tool use."""
        if self.development_mode:
            return SpecialistOutput(
                content="[MOCK] Pragmatic response to: " + message[:50],
                confidence=0.85,
                metadata={"role": "pragmatist", "mode": "mock"},
                specialist_type=self.specialist_type
            )
        
        # Real implementation would call Nemotron model here
        return SpecialistOutput(
            content="Pragmatic response placeholder",
            confidence=0.85,
            metadata={"role": "pragmatist"},
            specialist_type=self.specialist_type
        )


class ArtistSpecialist(BaseSpecialist):
    """
    Artist specialist for creative and visual generation.
    Uses Flux.1-schnell model for image generation.
    """
    
    def __init__(self, base_dir: Path, development_mode: bool = False):
        model_path = "black-forest-labs/FLUX.1-schnell"
        super().__init__(model_path, base_dir, development_mode)
        self.specialist_type = "artist"
    
    async def process(self, message: str, context: Dict[str, Any]) -> SpecialistOutput:
        """Process with creative generation."""
        if self.development_mode:
            return SpecialistOutput(
                content="[MOCK] Creative response to: " + message[:50],
                confidence=0.9,
                metadata={"role": "artist", "mode": "mock"},
                specialist_type=self.specialist_type
            )
        
        # Real implementation would call Flux model here
        return SpecialistOutput(
            content="Creative response placeholder",
            confidence=0.9,
            metadata={"role": "artist"},
            specialist_type=self.specialist_type
        )


class VoiceSpecialist(BaseSpecialist):
    """
    Voice specialist for final synthesis and personality.
    Uses Llama 3 70B model with persistent self-model.
    """
    
    def __init__(self, base_dir: Path, development_mode: bool = False):
        model_path = "meta-llama/Meta-Llama-3-70B-Instruct"
        super().__init__(model_path, base_dir, development_mode)
        self.specialist_type = "voice"
        
        # Load persistent self-model
        self.self_model_path = self.base_dir / "lyra" / "persistent_self_model.txt"
        self.self_model = self._load_self_model()
    
    def _load_self_model(self) -> str:
        """Load persistent self-model from file."""
        try:
            if self.self_model_path.exists():
                return self.self_model_path.read_text()
        except Exception as e:
            logger.warning(f"Failed to load self-model: {e}")
        return ""
    
    async def process(
        self,
        message: str,
        specialist_outputs: Dict[str, str],
        context: Dict[str, Any]
    ) -> SpecialistOutput:
        """
        Synthesize final response with personality.
        
        Args:
            message: Original user message
            specialist_outputs: Outputs from other specialists
            context: Additional context
            
        Returns:
            Final synthesized response
        """
        if self.development_mode:
            return SpecialistOutput(
                content="[MOCK] Voice synthesis of: " + message[:50],
                confidence=0.95,
                metadata={"role": "voice", "mode": "mock"},
                specialist_type=self.specialist_type
            )
        
        # Real implementation would synthesize with Llama 70B here
        return SpecialistOutput(
            content="Voice response placeholder",
            confidence=0.95,
            metadata={"role": "voice"},
            specialist_type=self.specialist_type
        )


class PerceptionSpecialist(BaseSpecialist):
    """
    Perception specialist for image understanding.
    Uses LLaVA model for vision tasks.
    """
    
    def __init__(self, model_path: str, base_dir: Path, development_mode: bool = False):
        super().__init__(model_path, base_dir, development_mode)
        self.specialist_type = "perception"
    
    async def process(
        self,
        image: Any = None,
        prompt: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> SpecialistOutput:
        """
        Process image with perception model.
        
        Args:
            image: PIL Image or None
            prompt: Description prompt
            context: Additional context
            
        Returns:
            SpecialistOutput with description
        """
        # Input validation
        if image is None:
            return SpecialistOutput(
                content="Error: No image provided (None)",
                confidence=0.0,
                metadata={"role": "perception", "error": "no_image"},
                specialist_type=self.specialist_type
            )
        
        # Check image size
        if hasattr(image, 'size'):
            width, height = image.size
            if width <= 0 or height <= 0:
                return SpecialistOutput(
                    content="Error: Invalid image dimensions",
                    confidence=0.0,
                    metadata={"role": "perception", "validation_failed": True},
                    specialist_type=self.specialist_type
                )
            
            if width * height > MAX_IMAGE_PIXELS:
                return SpecialistOutput(
                    content=f"Error: Image too large ({width}x{height} pixels)",
                    confidence=0.0,
                    metadata={"role": "perception", "validation_failed": True},
                    specialist_type=self.specialist_type
                )
        
        if self.development_mode:
            return SpecialistOutput(
                content=f"[MOCK] Perception of image: {prompt[:30]}",
                confidence=0.85,
                metadata={"role": "perception", "mode": "mock", "image_size": getattr(image, 'size', 'unknown')},
                specialist_type=self.specialist_type
            )
        
        # Real implementation would call LLaVA model here
        return SpecialistOutput(
            content="Perception response placeholder",
            confidence=0.85,
            metadata={"role": "perception", "image_size": getattr(image, 'size', 'unknown')},
            specialist_type=self.specialist_type
        )


class SpecialistFactory:
    """Factory for creating specialist instances."""
    
    def __init__(self, development_mode: bool = False):
        """
        Initialize factory.
        
        Args:
            development_mode: If True, specialists will use mock implementations
        """
        self.development_mode = development_mode
    
    def create_specialist(
        self,
        specialist_type: str,
        base_dir: Path,
    ) -> BaseSpecialist:
        """
        Create a specialist instance.
        
        Args:
            specialist_type: Type of specialist ('philosopher', 'pragmatist', 'artist', 'voice')
            base_dir: Base directory for data
            
        Returns:
            Specialist instance
            
        Raises:
            ValueError: If specialist_type is unknown
        """
        specialists = {
            'philosopher': PhilosopherSpecialist,
            'pragmatist': PragmatistSpecialist,
            'artist': ArtistSpecialist,
            'voice': VoiceSpecialist,
        }
        
        if specialist_type not in specialists:
            raise ValueError(f"Unknown specialist type: {specialist_type}")
        
        specialist_class = specialists[specialist_type]
        return specialist_class(base_dir, development_mode=self.development_mode)
