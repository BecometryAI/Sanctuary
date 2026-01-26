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

import logging
import time
import hashlib
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray

# Configure logging
logger = logging.getLogger(__name__)


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
        config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the perception subsystem.

        Args:
            config: Optional configuration dict with keys:
                - text_model: str = "all-MiniLM-L6-v2" (384-dim, 23MB)
                - cache_size: int = 1000
                - enable_image: bool = False
                - enable_audio: bool = False
                - device: str = "cpu" or "cuda"
        """
        self.config = config or {}
        
        # Load embedding model
        model_name = self.config.get("text_model", "all-MiniLM-L6-v2")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.text_encoder = SentenceTransformer(model_name)
            self.embedding_dim = self.text_encoder.get_sentence_embedding_dimension()
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load text encoder '{model_name}': {e}")
            raise
        
        # Cache for embeddings (OrderedDict for LRU)
        self.embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self.cache_size = self.config.get("cache_size", 1000)
        
        # Stats tracking
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_encodings": 0,
            "encoding_times": [],
        }
        
        # Optional image encoder
        self.image_encoder = None
        self.image_processor = None
        if self.config.get("enable_image", False):
            self._load_image_encoder()
        
        logger.info(f"✅ PerceptionSubsystem initialized with {model_name} "
                   f"(dim={self.embedding_dim})")
    
    def _load_image_encoder(self) -> bool:
        """Load CLIP for image encoding."""
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            logger.info("✅ CLIP image encoder loaded")
            return True
        except ImportError:
            logger.warning("CLIP not available (transformers/torch not installed)")
            return False
        except Exception as e:
            logger.warning(f"Failed to load CLIP: {e}")
            return False
    
    async def encode(self, raw_input: Any, modality: str) -> 'Percept':
        """
        Encode raw input into Percept with embedding.
        
        Main entry point for encoding inputs. Routes to appropriate
        modality handler and returns a Percept object.
        
        Args:
            raw_input: Raw data to encode (str, image, audio, dict)
            modality: Type of input ("text", "image", "audio", "introspection")
            
        Returns:
            Percept object with embedding and metadata
        """
        from .workspace import Percept as WorkspacePercept
        
        try:
            if modality == "text":
                embedding = self._encode_text(str(raw_input))
            elif modality == "image":
                embedding = self._encode_image(raw_input)
            elif modality == "audio":
                embedding = self._encode_audio(raw_input)
            elif modality == "introspection":
                # Introspective percepts are already structured
                if isinstance(raw_input, dict):
                    text = str(raw_input.get("description", ""))
                else:
                    text = str(raw_input)
                embedding = self._encode_text(text)
            else:
                raise ValueError(f"Unknown modality: {modality}")
            
            complexity = self._compute_complexity(raw_input, modality)
            
            percept = WorkspacePercept(
                modality=modality,
                raw=raw_input,
                embedding=embedding,
                complexity=complexity,
                timestamp=datetime.now(),
                metadata={"encoding_model": "sentence-transformers"}
            )
            
            self.stats["total_encodings"] += 1
            return percept
            
        except Exception as e:
            logger.error(f"Error encoding {modality} input: {e}", exc_info=True)
            # Return dummy percept on error
            return WorkspacePercept(
                modality=modality,
                raw=raw_input,
                embedding=[0.0] * self.embedding_dim,
                complexity=1,
                metadata={"error": str(e)}
            )
    
    def _encode_text(self, text: str) -> List[float]:
        """
        Encode text to embedding vector.
        
        Uses cache to avoid redundant encodings. Cache uses LRU eviction.
        
        Args:
            text: Text string to encode
            
        Returns:
            Normalized embedding vector (list of floats)
        """
        # Generate cache key
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache
        if cache_key in self.embedding_cache:
            self.stats["cache_hits"] += 1
            # Move to end (most recently used)
            self.embedding_cache.move_to_end(cache_key)
            return self.embedding_cache[cache_key]
        
        # Compute embedding
        self.stats["cache_misses"] += 1
        start_time = time.time()
        
        embedding = self.text_encoder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
        
        encoding_time = time.time() - start_time
        self.stats["encoding_times"].append(encoding_time)
        
        # Cache result (with LRU eviction)
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest entry (first item)
            self.embedding_cache.popitem(last=False)
        
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    def _encode_image(self, image: Any) -> List[float]:
        """
        Encode image to embedding using CLIP.
        
        Args:
            image: PIL Image, numpy array, or file path
            
        Returns:
            Normalized embedding vector
        """
        if self.image_encoder is None:
            logger.warning("Image encoding requested but CLIP not loaded")
            return [0.0] * self.embedding_dim
        
        try:
            from PIL import Image
            
            # Handle different image input types
            if isinstance(image, str):
                # File path
                image = Image.open(image)
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Encode with CLIP
            inputs = self.image_processor(images=image, return_tensors="pt")
            outputs = self.image_encoder.get_image_features(**inputs)
            
            # Normalize and convert to list
            embedding = outputs.detach().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return [0.0] * self.embedding_dim
    
    def _encode_audio(self, audio: Any) -> List[float]:
        """
        Encode audio by transcribing then encoding text.
        
        Args:
            audio: Audio data (future: integrate with Whisper)
            
        Returns:
            Placeholder embedding vector
        """
        # TODO: Integrate with existing Whisper implementation
        # For now, return placeholder
        logger.warning("Audio encoding not yet implemented")
        return [0.0] * self.embedding_dim
    
    def _compute_complexity(self, raw_input: Any, modality: str) -> int:
        """
        Estimate attention cost for processing this input.
        
        Complexity determines how much attention budget is consumed.
        
        Args:
            raw_input: The raw input data
            modality: Type of input
            
        Returns:
            Complexity score (1-100)
        """
        if modality == "text":
            text_length = len(str(raw_input))
            # 1 unit per ~20 characters, min 5, max 50
            return min(max(text_length // 20, 5), 50)
        
        elif modality == "image":
            # Images are expensive
            return 30
        
        elif modality == "audio":
            # Estimate based on duration (if available)
            if isinstance(raw_input, dict):
                duration = raw_input.get("duration_seconds", 5)
            else:
                duration = 5
            return min(int(duration * 5), 80)
        
        elif modality == "introspection":
            # Introspection is cognitively expensive
            return 20
        
        else:
            return 10  # Default
    
    def clear_cache(self) -> None:
        """Clear embedding cache. Useful for memory management."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics about encoding performance.
        
        Returns:
            Dict with cache hit rate, total encodings, and timing info
        """
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (
            self.stats["cache_hits"] / total_requests 
            if total_requests > 0 else 0.0
        )
        
        avg_encoding_time = (
            sum(self.stats["encoding_times"]) / len(self.stats["encoding_times"])
            if self.stats["encoding_times"] else 0.0
        )
        
        return {
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "total_encodings": self.stats["total_encodings"],
            "average_encoding_time_ms": avg_encoding_time * 1000,
            "cache_size": len(self.embedding_cache),
            "embedding_dim": self.embedding_dim,
        }
    
    async def process(self, raw_input: Any) -> Any:
        """
        Legacy compatibility method.
        
        Converts raw input into a percept with embedding.
        For new code, use encode() instead.
        
        Args:
            raw_input: Raw input data to process
            
        Returns:
            Percept with embedding
        """
        return await self.encode(raw_input, "text")
