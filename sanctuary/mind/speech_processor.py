"""
Speech-to-Text processing module for Sanctuary's auditory perception
Uses Whisper model with enhanced streaming capabilities
"""
import logging
import numpy as np
import torch
from typing import Optional, Generator, AsyncGenerator
import asyncio
from pathlib import Path
from transformers import (
    WhisperProcessor as HFWhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline
)
from .voice_analyzer import EmotionAnalyzer

logger = logging.getLogger(__name__)

class WhisperProcessor:
    def __init__(self):
        """Initialize Whisper model for Sanctuary's hearing"""
        # Create pipeline directly
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=self.device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Keep processor and model references for additional processing
        self.processor = self.asr_pipeline.feature_extractor
        self.model = self.asr_pipeline.model
        
        # Initialize emotion analyzer
        self.emotion_analyzer = EmotionAnalyzer()
        
        # Audio processing parameters
        self.sample_rate = 16000
        self.chunk_duration = 30  # seconds
        self.min_speech_probability = 0.5
        
        # Emotional context tracking
        self.voice_context = {
            "speaker_tone": None,
            "emotional_markers": [],
            "confidence": 0.0
        }
    
    async def process_audio_stream(self, 
                                 audio_generator: AsyncGenerator[np.ndarray, None],
                                 language: str = "en") -> AsyncGenerator[str, None]:
        """
        Process incoming audio stream with emotional context awareness
        
        Args:
            audio_generator: Generator yielding audio chunks
            language: Expected language code
            
        Yields:
            Transcribed text with high confidence
        """
        current_buffer = np.array([])
        
        async for chunk in audio_generator:
            # Append to buffer
            current_buffer = np.concatenate([current_buffer, chunk])
            
            # Process when we have enough audio
            if len(current_buffer) >= self.sample_rate * self.chunk_duration:
                # Transcribe with emotional context
                result = await self._transcribe_with_context(current_buffer, language)
                
                if result and result["confidence"] > self.min_speech_probability:
                    # Update emotional context
                    self._update_voice_context(result)
                    
                    # Yield transcribed text
                    yield result["text"]
                
                # Reset buffer with small overlap
                overlap = int(0.5 * self.sample_rate)  # 0.5 second overlap
                current_buffer = current_buffer[-overlap:] if len(current_buffer) > overlap else np.array([])
    
    async def _transcribe_with_context(self, 
                                     audio_data: np.ndarray, 
                                     language: str) -> Optional[dict]:
        """
        Transcribe audio while maintaining emotional context
        """
        try:
            transcription = await asyncio.to_thread(
                self.asr_pipeline,
                audio_data,
                generate_kwargs={"language": language, "task": "transcribe"}
            )
            
            result = {
                "text": transcription["text"],
                "confidence": 0.95,  # Placeholder until we implement confidence scoring
                "emotional_context": {
                    "tone": self._detect_tone(audio_data),
                    "confidence": 0.95,
                    "speaker_consistency": self._check_speaker_consistency(audio_data)
                }
            }
            
            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
    
    def _detect_tone(self, audio_data: np.ndarray) -> str:
        """
        Detect emotional tone in speech
        """
        # This would integrate with more sophisticated tone analysis
        # For now, return a neutral tone
        return "neutral"
    
    def _check_speaker_consistency(self, audio_data: np.ndarray) -> float:
        """
        Check if the current speaker is consistent
        """
        # This would integrate with speaker diarization
        # For now, return high consistency
        return 0.95
    
    def _update_voice_context(self, result: dict) -> None:
        """
        Update ongoing voice context tracking
        """
        self.voice_context["confidence"] = result["confidence"]
        self.voice_context["emotional_markers"].append(result["emotional_context"]["tone"])
        # Keep only recent context
        if len(self.voice_context["emotional_markers"]) > 10:
            self.voice_context["emotional_markers"] = self.voice_context["emotional_markers"][-10:]