"""
Perception Subsystem: Multimodal input processing.

This module implements the PerceptionSubsystem class, which converts raw multimodal
inputs (text, images, audio) into internal vector representations (embeddings).
It uses encoding models, not generative LLMs, to create a common representational
space for diverse input modalities.

The perception subsystem is responsible for:
- Converting raw sensory data into internal representations
- Maintaining consistent embedding spaces across modalities
- Providing pre-processed inputs to the attention system
- Detecting and handling perceptual anomalies
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from numpy.typing import NDArray


class ModalityType(Enum):
    """
    Types of perceptual input modalities.

    TEXT: Natural language text input
    IMAGE: Visual input (photos, diagrams, etc.)
    AUDIO: Auditory input (speech, sounds, music)
    PROPRIOCEPTIVE: Internal state signals (not external sensory)
    """
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    PROPRIOCEPTIVE = "proprioceptive"


@dataclass
class Percept:
    """
    Represents a single perceptual input after encoding.

    A percept is the internal representation of external sensory input.
    It includes both the vector embedding and metadata about the source,
    modality, and processing timestamp.

    Attributes:
        embedding: Vector representation of the input
        modality: Type of sensory input (text, image, audio, etc.)
        raw_content: Optional reference to original input
        timestamp: When the percept was created
        confidence: Model confidence in the encoding (0.0-1.0)
        metadata: Additional contextual information
    """
    embedding: NDArray[np.float32]
    modality: ModalityType
    raw_content: Optional[Any] = None
    timestamp: Optional[float] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class PerceptionSubsystem:
    """
    Converts raw multimodal inputs into internal vector representations.

    The PerceptionSubsystem is the boundary between the external world and the
    internal cognitive architecture. It uses specialized encoding models (not
    generative LLMs) to transform raw sensory inputs into a common vector space
    that can be processed by the attention and workspace systems.

    Key Responsibilities:
    - Encode text inputs using language embedding models
    - Encode visual inputs using vision transformers or CLIP-style models
    - Encode audio inputs using audio embedding models
    - Maintain consistent embedding spaces across modalities
    - Detect and flag perceptual anomalies or low-confidence encodings
    - Buffer recent percepts for temporal context

    Integration Points:
    - AttentionController: Provides candidate percepts for attention scoring
    - GlobalWorkspace: Selected percepts enter conscious awareness
    - CognitiveCore: Receives percepts on each cognitive cycle
    - LanguageInputParser: Uses perception for text encoding (at periphery)

    Design Philosophy:
    This subsystem explicitly uses ENCODING models, not generative LLMs. The goal
    is to create vector representations that capture semantic content without
    generating new text or imposing linguistic structure on non-linguistic inputs.

    For text: sentence-transformers or similar embedding models
    For images: CLIP image encoder, vision transformers, or similar
    For audio: wav2vec, audio spectrogram transformers, or similar

    The perception subsystem does NOT:
    - Generate text responses (that's LanguageOutputGenerator)
    - Make decisions about attention (that's AttentionController)
    - Store long-term memories (that's handled by external memory systems)

    Attributes:
        text_encoder: Model for encoding text inputs
        image_encoder: Model for encoding image inputs
        audio_encoder: Model for encoding audio inputs
        embedding_dim: Dimensionality of the common embedding space
        percept_buffer: Recent percepts maintained for temporal context
    """

    def __init__(
        self,
        text_encoder_name: str = "sentence-transformers/all-mpnet-base-v2",
        image_encoder_name: Optional[str] = None,
        audio_encoder_name: Optional[str] = None,
        embedding_dim: int = 768,
        buffer_size: int = 100,
    ) -> None:
        """
        Initialize the perception subsystem.

        Args:
            text_encoder_name: Name/path of text embedding model to use
            image_encoder_name: Name/path of image encoding model (optional)
            audio_encoder_name: Name/path of audio encoding model (optional)
            embedding_dim: Target dimensionality for all embeddings
            buffer_size: Maximum number of recent percepts to maintain

        Note: If modality encoders are None, that modality will not be supported.
        All encoders should project to the same embedding_dim for compatibility.
        """
        pass
