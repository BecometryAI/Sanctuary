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
    SENSOR: Physical sensor data (temperature, motion, touch, etc.)
    PROPRIOCEPTIVE: Internal state signals (not external sensory)
    """
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    SENSOR = "sensor"
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
    Perception subsystem that converts raw multimodal inputs into internal vector representations.

    Uses encoding models (not generative LLMs) to transform sensory inputs
    into a common vector space for attention and workspace systems.

    Supported modalities: text, image, audio, sensor, introspection.
    """

    # Cached projection matrices (class-level for efficiency)
    _audio_projection: Optional[np.ndarray] = None
    _sensor_projection_cache: Dict[int, np.ndarray] = {}
    _mel_filterbank_cache: Dict[tuple, np.ndarray] = {}

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
            elif modality == "sensor":
                embedding = self._encode_sensor(raw_input)
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
        
        raw_embedding = self.text_encoder.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        # SentenceTransformer.encode() may return 2D array for single string;
        # squeeze to 1D before converting to list of floats
        if hasattr(raw_embedding, 'ndim') and raw_embedding.ndim == 2:
            raw_embedding = raw_embedding[0]
        embedding = raw_embedding.tolist()
        
        encoding_time = time.time() - start_time
        # Keep only last 100 encoding times to prevent memory leak
        if len(self.stats["encoding_times"]) >= 100:
            self.stats["encoding_times"].pop(0)
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

        Supports multiple input formats from the device abstraction layer:
        - PIL Image objects
        - Numpy arrays (BGR from OpenCV or RGB)
        - File paths (strings)
        - Dict with 'data' key containing numpy array (from DeviceDataPacket)

        Args:
            image: PIL Image, numpy array, file path, or dict with image data

        Returns:
            Normalized embedding vector
        """
        if self.image_encoder is None:
            logger.warning("Image encoding requested but CLIP not loaded")
            return [0.0] * self.embedding_dim

        try:
            from PIL import Image as PILImage

            # Handle dict format from DeviceDataPacket
            if isinstance(image, dict):
                if "data" in image:
                    image = image["data"]
                elif "raw_data" in image:
                    image = image["raw_data"]

            # Handle different image input types
            if isinstance(image, str):
                # File path
                image = PILImage.open(image)
            elif isinstance(image, np.ndarray):
                # Numpy array - could be BGR (from OpenCV) or RGB
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Check if it looks like BGR (OpenCV format) by checking metadata
                    # For now, assume BGR from camera devices and convert to RGB
                    try:
                        import cv2
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    except ImportError:
                        # If cv2 not available, assume RGB
                        pass
                image = PILImage.fromarray(image)
            elif isinstance(image, bytes):
                # Raw bytes - try to decode as image
                import io
                image = PILImage.open(io.BytesIO(image))

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
        Encode audio to embedding.

        Supports multiple input formats from the device abstraction layer:
        - Numpy arrays (float32 samples from sounddevice)
        - Raw bytes (int16 PCM audio)
        - Dict with audio data and metadata (from DeviceDataPacket)
        - Transcribed text (falls back to text encoding)

        For now, audio is encoded by computing spectral features and mapping
        to the text embedding space. Future versions will integrate with
        dedicated audio encoders (wav2vec, Whisper embeddings, etc.).

        Args:
            audio: Audio data in various formats

        Returns:
            Embedding vector
        """
        try:
            # Extract audio data from dict format (DeviceDataPacket)
            sample_rate = 16000  # Default
            if isinstance(audio, dict):
                sample_rate = audio.get("sample_rate", 16000)
                if "data" in audio:
                    audio = audio["data"]
                elif "raw_data" in audio:
                    audio = audio["raw_data"]
                elif "transcription" in audio:
                    # If already transcribed, encode the text
                    return self._encode_text(audio["transcription"])

            # Convert bytes to numpy array
            if isinstance(audio, bytes):
                # Assume int16 PCM
                audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

            # If string (transcription), encode as text
            if isinstance(audio, str):
                return self._encode_text(audio)

            # If numpy array, compute basic audio features
            if isinstance(audio, np.ndarray):
                # Flatten if needed
                if audio.ndim > 1:
                    audio = audio.flatten()

                # Compute simple spectral features
                audio_features = self._compute_audio_features(audio, sample_rate)

                # Map audio features to embedding space
                # For now, create a pseudo-embedding based on audio statistics
                # This preserves some audio characteristics while fitting the embedding dim
                embedding = self._audio_features_to_embedding(audio_features)
                return embedding

            # Fallback: return zero embedding
            logger.warning(f"Unsupported audio format: {type(audio)}")
            return [0.0] * self.embedding_dim

        except Exception as e:
            logger.error(f"Error encoding audio: {e}")
            return [0.0] * self.embedding_dim

    def _compute_audio_features(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Compute audio features for embedding: time-domain, spectral, MFCCs, pitch.
        """
        features = {}

        # Input validation
        if sample_rate <= 0:
            logger.warning(f"Invalid sample_rate: {sample_rate}")
            return {"empty": True}

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if len(audio) == 0:
            return {"empty": True}

        # Handle NaN/Inf values
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)

        # ========== Time-domain features ==========
        features["mean"] = float(np.mean(audio))
        features["std"] = float(np.std(audio))
        features["max_amplitude"] = float(np.max(np.abs(audio)))
        features["rms"] = float(np.sqrt(np.mean(audio ** 2)))

        # Zero crossing rate (voice activity indicator)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / 2
        features["zcr"] = float(zero_crossings / len(audio)) if len(audio) > 1 else 0.0

        # Duration
        features["duration"] = float(len(audio) / sample_rate)

        # ========== Frame-based analysis ==========
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        n_fft = 2 ** int(np.ceil(np.log2(frame_length)))  # Next power of 2

        if len(audio) < frame_length:
            # Too short for spectral analysis - use basic features only
            features["short_audio"] = True
            return features

        # Compute STFT
        num_frames = (len(audio) - frame_length) // hop_length + 1

        # Pre-compute windowed frames
        window = np.hanning(frame_length)
        frames = []
        for i in range(num_frames):
            start = i * hop_length
            end = start + frame_length
            frame = audio[start:end] * window
            frames.append(frame)

        frames = np.array(frames)

        # Compute magnitude spectra
        spectra = np.abs(np.fft.rfft(frames, n=n_fft, axis=1))
        power_spectra = spectra ** 2

        # Frequency bins
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        # ========== Spectral features (vectorized) ==========
        spectra_sum = np.sum(spectra, axis=1, keepdims=True)
        spectra_sum = np.maximum(spectra_sum, 1e-10)  # Avoid division by zero

        # Spectral centroid (brightness) - vectorized
        centroid_per_frame = np.sum(freqs * spectra, axis=1) / spectra_sum.squeeze()
        features["spectral_centroid_mean"] = float(np.mean(centroid_per_frame))
        features["spectral_centroid_std"] = float(np.std(centroid_per_frame))

        # Spectral bandwidth - vectorized
        freq_diff_sq = (freqs - centroid_per_frame[:, np.newaxis]) ** 2
        bandwidth_per_frame = np.sqrt(np.sum(freq_diff_sq * spectra, axis=1) / spectra_sum.squeeze())
        features["spectral_bandwidth_mean"] = float(np.mean(bandwidth_per_frame))
        features["spectral_bandwidth_std"] = float(np.std(bandwidth_per_frame))

        # Spectral rolloff (85% energy threshold)
        cumsum = np.cumsum(spectra, axis=1)
        thresholds = 0.85 * cumsum[:, -1:]
        rolloff_indices = np.argmax(cumsum >= thresholds, axis=1)
        rolloff_per_frame = freqs[np.minimum(rolloff_indices, len(freqs) - 1)]
        features["spectral_rolloff_mean"] = float(np.mean(rolloff_per_frame))

        # Spectral flux
        features["spectral_flux"] = float(np.mean(np.diff(spectra, axis=0) ** 2)) if len(spectra) > 1 else 0.0

        # Spectral flatness (vectorized)
        power_clean = power_spectra + 1e-10
        geo_mean = np.exp(np.mean(np.log(power_clean), axis=1))
        arith_mean = np.mean(power_clean, axis=1)
        flatness_per_frame = geo_mean / np.maximum(arith_mean, 1e-10)
        features["spectral_flatness_mean"] = float(np.mean(flatness_per_frame))

        # ========== MFCCs (speech/audio fingerprint) ==========
        mfccs = self._compute_mfccs(power_spectra, sample_rate, n_fft, n_mfcc=13)
        for i, mfcc_mean in enumerate(np.mean(mfccs, axis=0)):
            features[f"mfcc_{i}"] = float(mfcc_mean)

        # MFCC deltas (velocity)
        if mfccs.shape[0] > 2:
            mfcc_delta = np.diff(mfccs, axis=0)
            features["mfcc_delta_mean"] = float(np.mean(np.abs(mfcc_delta)))

        # ========== Frame energy statistics ==========
        frame_energies = np.sum(frames ** 2, axis=1)
        features["energy_mean"] = float(np.mean(frame_energies))
        features["energy_std"] = float(np.std(frame_energies))
        features["energy_max"] = float(np.max(frame_energies))

        # Energy ratio (speech activity)
        if features["energy_max"] > 0:
            features["energy_ratio"] = float(features["energy_mean"] / features["energy_max"])
        else:
            features["energy_ratio"] = 0.0

        # ========== Pitch estimation (for speech) ==========
        pitch = self._estimate_pitch(audio, sample_rate)
        if pitch is not None:
            features["pitch_hz"] = float(pitch)
            features["has_pitch"] = True
        else:
            features["has_pitch"] = False

        return features

    def _compute_mfccs(
        self,
        power_spectra: np.ndarray,
        sample_rate: int,
        n_fft: int,
        n_mfcc: int = 13,
        n_mels: int = 40
    ) -> np.ndarray:
        """
        Compute Mel-Frequency Cepstral Coefficients.

        MFCCs are the standard feature for speech recognition and audio
        classification. They capture the envelope of the short-term power
        spectrum on a mel scale (perceptually-weighted frequency).

        Args:
            power_spectra: Power spectra array (n_frames, n_freq_bins)
            sample_rate: Sample rate in Hz
            n_fft: FFT size used
            n_mfcc: Number of MFCCs to return
            n_mels: Number of mel filter banks

        Returns:
            MFCCs array (n_frames, n_mfcc)
        """
        # Create mel filterbank
        mel_filters = self._create_mel_filterbank(sample_rate, n_fft, n_mels)

        # Apply mel filterbank
        mel_spectra = np.dot(power_spectra, mel_filters.T)

        # Log compression (with floor to avoid log(0))
        log_mel_spectra = np.log(mel_spectra + 1e-10)

        # DCT (Type-II) to get MFCCs
        # Using manual DCT implementation for simplicity
        n_frames = log_mel_spectra.shape[0]
        mfccs = np.zeros((n_frames, n_mfcc))

        for i in range(n_mfcc):
            # DCT basis function
            basis = np.cos(np.pi * i * (np.arange(n_mels) + 0.5) / n_mels)
            mfccs[:, i] = np.dot(log_mel_spectra, basis)

        # Normalize
        mfccs[:, 0] *= np.sqrt(1.0 / n_mels)
        mfccs[:, 1:] *= np.sqrt(2.0 / n_mels)

        return mfccs

    def _create_mel_filterbank(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int = 40,
        fmin: float = 0.0,
        fmax: Optional[float] = None
    ) -> np.ndarray:
        """Create a mel filterbank matrix (cached for efficiency)."""
        if fmax is None:
            fmax = sample_rate / 2.0

        # Check cache
        cache_key = (sample_rate, n_fft, n_mels, fmin, fmax)
        if cache_key in PerceptionSubsystem._mel_filterbank_cache:
            return PerceptionSubsystem._mel_filterbank_cache[cache_key]

        # Mel scale conversion
        hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
        mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        n_freq = n_fft // 2 + 1
        fft_freqs = np.linspace(0, sample_rate / 2.0, n_freq)

        # Vectorized filterbank creation
        filterbank = np.zeros((n_mels, n_freq))
        for i in range(n_mels):
            left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            left_slope = (fft_freqs >= left) & (fft_freqs <= center)
            right_slope = (fft_freqs > center) & (fft_freqs <= right)
            filterbank[i, left_slope] = (fft_freqs[left_slope] - left) / (center - left + 1e-10)
            filterbank[i, right_slope] = (right - fft_freqs[right_slope]) / (right - center + 1e-10)

        # Cache and return
        PerceptionSubsystem._mel_filterbank_cache[cache_key] = filterbank
        return filterbank

    def _estimate_pitch(
        self,
        audio: np.ndarray,
        sample_rate: int,
        fmin: float = 50.0,
        fmax: float = 500.0
    ) -> Optional[float]:
        """
        Estimate fundamental frequency (pitch) using autocorrelation.

        Args:
            audio: Audio samples
            sample_rate: Sample rate in Hz
            fmin: Minimum pitch frequency (Hz)
            fmax: Maximum pitch frequency (Hz)

        Returns:
            Estimated pitch in Hz, or None if no clear pitch detected
        """
        # Lag range for pitch detection
        lag_min = int(sample_rate / fmax)
        lag_max = int(sample_rate / fmin)

        if len(audio) < lag_max * 2:
            return None

        # Compute autocorrelation for lag range
        # Use center portion of audio for stability
        center = len(audio) // 2
        segment = audio[center - lag_max:center + lag_max]

        if len(segment) < lag_max * 2:
            return None

        # Normalized autocorrelation
        autocorr = np.correlate(segment, segment, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Take positive lags only

        # Normalize
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]

        # Find peak in valid lag range
        valid_autocorr = autocorr[lag_min:lag_max]

        if len(valid_autocorr) == 0:
            return None

        peak_idx = np.argmax(valid_autocorr)
        peak_value = valid_autocorr[peak_idx]

        # Require minimum correlation for voiced detection
        if peak_value < 0.3:
            return None

        # Convert lag to frequency
        best_lag = lag_min + peak_idx
        pitch = sample_rate / best_lag

        return pitch

    def _audio_features_to_embedding(self, features: Dict[str, Any]) -> List[float]:
        """Map audio features to embedding space with semantic alignment."""
        if features.get("empty", False):
            return [0.0] * self.embedding_dim

        # Build feature vector (25 features total)
        all_features = [
            # Time-domain (4)
            min(features.get("rms", 0.0) * 10, 1.0),
            min(features.get("zcr", 0.0) * 50, 1.0),
            min(features.get("energy_ratio", 0.0), 1.0),
            min(features.get("duration", 0.0) / 10.0, 1.0),
            # Spectral (6)
            min(features.get("spectral_centroid_mean", 0.0) / 8000.0, 1.0),
            min(features.get("spectral_centroid_std", 0.0) / 2000.0, 1.0),
            min(features.get("spectral_bandwidth_mean", 0.0) / 4000.0, 1.0),
            min(features.get("spectral_rolloff_mean", 0.0) / 10000.0, 1.0),
            min(features.get("spectral_flux", 0.0) * 100, 1.0),
            features.get("spectral_flatness_mean", 0.0),
            # MFCCs (13)
            *[np.tanh(features.get(f"mfcc_{i}", 0.0) / 50.0) for i in range(13)],
            # Pitch (2)
            1.0 if features.get("has_pitch", False) else 0.0,
            min(features.get("pitch_hz", 0.0) / 500.0, 1.0) if features.get("has_pitch", False) else 0.0,
        ]
        feature_vector = np.array(all_features, dtype=np.float32)

        # Get or create cached projection matrix
        n_features = len(feature_vector)
        if PerceptionSubsystem._audio_projection is None or \
           PerceptionSubsystem._audio_projection.shape != (self.embedding_dim, n_features):
            rng = np.random.RandomState(42)
            PerceptionSubsystem._audio_projection = rng.randn(self.embedding_dim, n_features).astype(np.float32)
            PerceptionSubsystem._audio_projection /= np.sqrt(n_features)

        projected = np.dot(PerceptionSubsystem._audio_projection, feature_vector)

        # ========== Add semantic component ==========

        # Create a text description for semantic alignment
        # This helps the audio embedding relate to text descriptions
        descriptors = []

        # Volume descriptor
        rms = features.get("rms", 0.0)
        if rms > 0.3:
            descriptors.append("loud")
        elif rms < 0.05:
            descriptors.append("quiet")

        # Speech/tonal detection
        if features.get("has_pitch", False):
            pitch = features.get("pitch_hz", 0.0)
            if pitch > 200:
                descriptors.append("high-pitched voice")
            elif pitch < 120:
                descriptors.append("low-pitched voice")
            else:
                descriptors.append("speech")

        # Tonal vs noisy
        flatness = features.get("spectral_flatness_mean", 0.5)
        if flatness > 0.5:
            descriptors.append("noise")
        elif flatness < 0.1:
            descriptors.append("tonal")

        # Duration context
        duration = features.get("duration", 0.0)
        if duration > 5.0:
            descriptors.append("long audio")
        elif duration < 0.5:
            descriptors.append("short sound")

        # Get semantic embedding from description
        if descriptors:
            description = f"Audio: {', '.join(descriptors)}"
            semantic_embedding = np.array(self._encode_text(description), dtype=np.float32)

            # Blend projected features with semantic embedding
            # Weight: 70% acoustic features, 30% semantic
            alpha = 0.7
            blended = alpha * projected + (1 - alpha) * semantic_embedding
        else:
            blended = projected

        # ========== Final normalization ==========
        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm

        return blended.tolist()

    def _encode_sensor(self, sensor_data: Any) -> List[float]:
        """
        Encode sensor data to embedding with type-specific features.

        Supports sensor readings from the device abstraction layer:
        - Dict with sensor_type, value, unit (from SensorReading)
        - Raw numeric values
        - List of readings (time series)
        - Multi-axis sensors (accelerometer, gyroscope, magnetometer)

        Uses a hybrid encoding approach:
        1. Numerical features: Captures precise sensor values and derivatives
        2. Semantic features: Text embedding for categorical/interpretive meaning
        3. Blending: Combines both for robust representation

        Args:
            sensor_data: Sensor reading in various formats

        Returns:
            Embedding vector
        """
        try:
            # ========== Parse sensor data ==========
            sensor_type = "unknown"
            value = None
            unit = ""
            confidence = 1.0
            metadata = {}

            if isinstance(sensor_data, dict):
                sensor_type = sensor_data.get("sensor_type", "unknown")
                value = sensor_data.get("value", sensor_data.get("data", 0))
                unit = sensor_data.get("unit", "")
                confidence = sensor_data.get("confidence", 1.0)
                metadata = sensor_data.get("metadata", {})
            elif isinstance(sensor_data, (int, float)):
                value = sensor_data
            elif isinstance(sensor_data, list):
                value = sensor_data
            elif isinstance(sensor_data, np.ndarray):
                value = sensor_data.tolist() if sensor_data.size > 1 else float(sensor_data)

            # ========== Extract numerical features based on sensor type ==========
            numerical_features = self._extract_sensor_features(sensor_type, value, unit, metadata)

            # ========== Create semantic description ==========
            semantic_desc = self._create_sensor_description(sensor_type, value, unit, confidence)

            # ========== Hybrid encoding ==========
            return self._encode_sensor_hybrid(numerical_features, semantic_desc, sensor_type)

        except Exception as e:
            logger.error(f"Error encoding sensor data: {e}")
            return [0.0] * self.embedding_dim

    def _extract_sensor_features(
        self,
        sensor_type: str,
        value: Any,
        unit: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Extract numerical features based on sensor type.

        Different sensor types have different value ranges and semantics.
        This normalizes values to a common scale for embedding.

        Args:
            sensor_type: Type of sensor (TEMPERATURE, ACCELEROMETER, etc.)
            value: Sensor reading value
            unit: Unit of measurement
            metadata: Additional sensor metadata

        Returns:
            Dict of normalized numerical features
        """
        features = {}
        sensor_type_upper = sensor_type.upper() if isinstance(sensor_type, str) else "UNKNOWN"

        # ========== Environmental sensors ==========
        if sensor_type_upper == "TEMPERATURE":
            # Normalize to typical indoor range (celsius assumed)
            if isinstance(value, (int, float)):
                # Map typical range (-20 to 50°C) to 0-1
                features["temp_normalized"] = (float(value) + 20) / 70.0
                features["temp_cold"] = 1.0 if value < 10 else 0.0
                features["temp_hot"] = 1.0 if value > 30 else 0.0
                features["temp_comfortable"] = 1.0 if 18 <= value <= 26 else 0.0

        elif sensor_type_upper == "HUMIDITY":
            if isinstance(value, (int, float)):
                features["humidity_normalized"] = float(value) / 100.0
                features["humidity_dry"] = 1.0 if value < 30 else 0.0
                features["humidity_humid"] = 1.0 if value > 70 else 0.0

        elif sensor_type_upper == "PRESSURE":
            if isinstance(value, (int, float)):
                # Atmospheric pressure (hPa): typical range 950-1050
                features["pressure_normalized"] = (float(value) - 950) / 100.0
                features["pressure_low"] = 1.0 if value < 1000 else 0.0
                features["pressure_high"] = 1.0 if value > 1020 else 0.0

        elif sensor_type_upper == "LIGHT":
            if isinstance(value, (int, float)):
                # Log scale for light (lux can vary from 0 to 100000+)
                features["light_log"] = np.log1p(float(value)) / 12.0  # log(100000) ≈ 11.5
                features["light_dark"] = 1.0 if value < 50 else 0.0
                features["light_bright"] = 1.0 if value > 1000 else 0.0

        elif sensor_type_upper == "SOUND_LEVEL":
            if isinstance(value, (int, float)):
                # dB scale: typical range 0-120
                features["sound_normalized"] = float(value) / 120.0
                features["sound_quiet"] = 1.0 if value < 40 else 0.0
                features["sound_loud"] = 1.0 if value > 80 else 0.0

        # ========== Motion sensors (multi-axis) ==========
        elif sensor_type_upper in ("ACCELEROMETER", "GYROSCOPE", "MAGNETOMETER"):
            if isinstance(value, dict):
                # Multi-axis: {x, y, z}
                x = float(value.get("x", value.get("X", 0)))
                y = float(value.get("y", value.get("Y", 0)))
                z = float(value.get("z", value.get("Z", 0)))

                # Magnitude
                magnitude = np.sqrt(x**2 + y**2 + z**2)

                if sensor_type_upper == "ACCELEROMETER":
                    # Normalize assuming ±16g range, typical gravity ~9.8
                    features["accel_x"] = x / 16.0
                    features["accel_y"] = y / 16.0
                    features["accel_z"] = z / 16.0
                    features["accel_magnitude"] = magnitude / 16.0
                    features["accel_stationary"] = 1.0 if 9.0 < magnitude < 10.5 else 0.0
                    features["accel_moving"] = 1.0 if magnitude > 11.0 or magnitude < 9.0 else 0.0

                elif sensor_type_upper == "GYROSCOPE":
                    # Normalize assuming ±2000 dps range
                    features["gyro_x"] = x / 2000.0
                    features["gyro_y"] = y / 2000.0
                    features["gyro_z"] = z / 2000.0
                    features["gyro_magnitude"] = magnitude / 2000.0
                    features["gyro_rotating"] = 1.0 if magnitude > 50 else 0.0

                elif sensor_type_upper == "MAGNETOMETER":
                    # Normalize assuming ±1000 µT range
                    features["mag_x"] = x / 1000.0
                    features["mag_y"] = y / 1000.0
                    features["mag_z"] = z / 1000.0
                    features["mag_magnitude"] = magnitude / 1000.0

            elif isinstance(value, (list, tuple)) and len(value) >= 3:
                x, y, z = float(value[0]), float(value[1]), float(value[2])
                magnitude = np.sqrt(x**2 + y**2 + z**2)
                features["axis_x"] = x / 100.0  # Generic normalization
                features["axis_y"] = y / 100.0
                features["axis_z"] = z / 100.0
                features["axis_magnitude"] = magnitude / 100.0

        elif sensor_type_upper == "DISTANCE":
            if isinstance(value, (int, float)):
                # Distance sensors (cm typically): normalize to 0-500cm
                features["distance_normalized"] = min(float(value) / 500.0, 1.0)
                features["distance_close"] = 1.0 if value < 30 else 0.0
                features["distance_far"] = 1.0 if value > 200 else 0.0

        elif sensor_type_upper == "TOUCH":
            if isinstance(value, (int, float)):
                features["touch_active"] = 1.0 if value > 0 else 0.0
                features["touch_value"] = min(float(value) / 1024.0, 1.0)  # Typical 10-bit

        elif sensor_type_upper == "FORCE":
            if isinstance(value, (int, float)):
                # Force sensors: normalize to typical 0-100N range
                features["force_normalized"] = min(float(value) / 100.0, 1.0)
                features["force_light"] = 1.0 if value < 10 else 0.0
                features["force_heavy"] = 1.0 if value > 50 else 0.0

        # ========== Generic fallback ==========
        if not features:
            if isinstance(value, (int, float)):
                # Generic single value normalization
                features["value_raw"] = float(value)
                features["value_normalized"] = np.tanh(float(value) / 100.0)
            elif isinstance(value, (list, tuple)):
                # Time series: compute statistics
                arr = np.array(value, dtype=np.float32)
                features["series_mean"] = float(np.mean(arr))
                features["series_std"] = float(np.std(arr))
                features["series_min"] = float(np.min(arr))
                features["series_max"] = float(np.max(arr))
                features["series_trend"] = float(arr[-1] - arr[0]) if len(arr) > 1 else 0.0

        return features

    def _create_sensor_description(
        self,
        sensor_type: str,
        value: Any,
        unit: str,
        confidence: float
    ) -> str:
        """
        Create a human-readable description of the sensor reading.

        This description is used for semantic embedding alignment.

        Args:
            sensor_type: Type of sensor
            value: Sensor reading
            unit: Unit of measurement
            confidence: Reading confidence (0-1)

        Returns:
            Text description of the sensor state
        """
        sensor_type_upper = sensor_type.upper() if isinstance(sensor_type, str) else "UNKNOWN"
        descriptions = []

        # Add sensor type
        type_names = {
            "TEMPERATURE": "temperature",
            "HUMIDITY": "humidity level",
            "PRESSURE": "atmospheric pressure",
            "LIGHT": "light intensity",
            "SOUND_LEVEL": "sound level",
            "ACCELEROMETER": "acceleration",
            "GYROSCOPE": "rotation rate",
            "MAGNETOMETER": "magnetic field",
            "DISTANCE": "distance measurement",
            "TOUCH": "touch sensor",
            "FORCE": "force measurement",
        }
        type_name = type_names.get(sensor_type_upper, sensor_type.lower())

        # Format value
        if isinstance(value, dict):
            value_str = ", ".join(f"{k}={v:.2f}" for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            if len(value) <= 5:
                value_str = ", ".join(f"{v:.2f}" for v in value)
            else:
                value_str = f"series of {len(value)} readings"
        elif isinstance(value, (int, float)):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)

        # Build description
        descriptions.append(f"{type_name}: {value_str}")
        if unit:
            descriptions[-1] += f" {unit}"

        # Add qualitative descriptors
        if sensor_type_upper == "TEMPERATURE" and isinstance(value, (int, float)):
            if value < 0:
                descriptions.append("freezing cold")
            elif value < 15:
                descriptions.append("cold")
            elif value < 22:
                descriptions.append("cool")
            elif value < 28:
                descriptions.append("warm")
            else:
                descriptions.append("hot")

        elif sensor_type_upper == "HUMIDITY" and isinstance(value, (int, float)):
            if value < 30:
                descriptions.append("dry air")
            elif value > 70:
                descriptions.append("humid air")

        elif sensor_type_upper == "LIGHT" and isinstance(value, (int, float)):
            if value < 10:
                descriptions.append("darkness")
            elif value < 100:
                descriptions.append("dim light")
            elif value < 1000:
                descriptions.append("indoor lighting")
            else:
                descriptions.append("bright light")

        elif sensor_type_upper == "ACCELEROMETER" and isinstance(value, dict):
            mag = np.sqrt(sum(v**2 for v in value.values()))
            if 9.0 < mag < 10.5:
                descriptions.append("stationary")
            else:
                descriptions.append("motion detected")

        # Add confidence qualifier if low
        if confidence < 0.7:
            descriptions.append("uncertain reading")

        return " | ".join(descriptions)

    def _encode_sensor_hybrid(
        self,
        numerical_features: Dict[str, float],
        semantic_desc: str,
        sensor_type: str
    ) -> List[float]:
        """Blend numerical projection with semantic embedding based on sensor type."""
        feature_values = list(numerical_features.values())

        if feature_values:
            feature_vector = np.array(feature_values, dtype=np.float32)
            n_features = len(feature_vector)

            # Get or create cached projection matrix for this feature count
            if n_features not in PerceptionSubsystem._sensor_projection_cache:
                rng = np.random.RandomState(43)
                proj = rng.randn(self.embedding_dim, n_features).astype(np.float32)
                PerceptionSubsystem._sensor_projection_cache[n_features] = proj / np.sqrt(n_features)

            numerical_embedding = np.dot(
                PerceptionSubsystem._sensor_projection_cache[n_features],
                feature_vector
            )
        else:
            numerical_embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        # ========== Semantic embedding ==========
        semantic_embedding = np.array(self._encode_text(semantic_desc), dtype=np.float32)

        # ========== Determine blend ratio based on sensor type ==========
        sensor_type_upper = sensor_type.upper() if isinstance(sensor_type, str) else "UNKNOWN"

        # High-precision sensors get more numerical weight
        high_precision_sensors = {"ACCELEROMETER", "GYROSCOPE", "MAGNETOMETER", "FORCE"}
        # Qualitative sensors get more semantic weight
        qualitative_sensors = {"TEMPERATURE", "HUMIDITY", "LIGHT", "TOUCH"}

        if sensor_type_upper in high_precision_sensors:
            alpha = 0.7  # 70% numerical
        elif sensor_type_upper in qualitative_sensors:
            alpha = 0.4  # 40% numerical
        else:
            alpha = 0.5  # Balanced

        # ========== Blend and normalize ==========
        blended = alpha * numerical_embedding + (1 - alpha) * semantic_embedding

        norm = np.linalg.norm(blended)
        if norm > 0:
            blended = blended / norm

        return blended.tolist()
    
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
            # 1 unit per ~10 characters, min 1, max 50
            return min(max(text_length // 10, 1), 50)
        
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

        elif modality == "sensor":
            # Sensor readings are lightweight
            return 5

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
