"""
Voice processing module for Sanctuary's auditory perception and expression
Uses transformers for TTS and STT for compatibility with Python 3.13
"""
import asyncio
import io
import logging
import torch
from pathlib import Path
from typing import Optional, AsyncGenerator, Union
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import soundfile as sf
import numpy as np
from scipy import signal
from .voice_customizer import VoiceCustomizer, VoiceProfile

# Constants
WHISPER_MODEL = "openai/whisper-small"  # Small is fastest, still good quality
TTS_MODEL = "microsoft/speecht5_tts"     # High quality text-to-speech
EMOTION_MODEL = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"  # Emotion detection
DISCORD_SAMPLE_RATE = 48000  # Discord voice sample rate
DISCORD_FRAME_SIZE = 3840   # Size of each Discord audio frame (20ms at 48kHz)
DISCORD_CHANNELS = 2        # Discord uses stereo audio
DISCORD_CHUNK_SIZE = 4000   # Number of audio frames to process at once

# Emotion categories and their interpretations
EMOTIONS = {
    "anger": {"valence": -0.8, "arousal": 0.8, "dominance": 0.7},
    "disgust": {"valence": -0.6, "arousal": 0.2, "dominance": 0.5},
    "fear": {"valence": -0.7, "arousal": 0.7, "dominance": -0.7},
    "happiness": {"valence": 0.8, "arousal": 0.5, "dominance": 0.4},
    "sadness": {"valence": -0.7, "arousal": -0.3, "dominance": -0.5},
    "surprise": {"valence": 0.4, "arousal": 0.8, "dominance": -0.2},
    "neutral": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
}

# Map model labels to EMOTIONS keys
EMOTION_LABEL_MAP = {
    "happy": "happiness",
    "sad": "sadness",
    "angry": "anger",
    "fearful": "fear",
    "disgusted": "disgust",
    "surprised": "surprise",
    "neutral": "neutral",
    # Also include direct mappings
    "happiness": "happiness",
    "sadness": "sadness", 
    "anger": "anger",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise"
}

logger = logging.getLogger(__name__)

class VoiceProcessor:
    def __init__(self, voice_path: Optional[str] = None):
        """
        Initialize voice processing with modern transformers
        
        Args:
            voice_path: Path to custom voice file
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize speech recognition
        self.stt_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.stt_processor = AutoProcessor.from_pretrained(WHISPER_MODEL)
        
        # Initialize TTS (Text to Speech)
        self.tts = pipeline(
            "text-to-speech",
            TTS_MODEL,
            device=self.device,
        )
        
        # Initialize emotion recognition
        self.emotion_classifier = pipeline(
            task="audio-classification",
            model=EMOTION_MODEL,
            device=self.device,
            framework="pt"
        )
        
        # Initialize voice customization
        self.voice_customizer = VoiceCustomizer()
        self.current_voice = None
        
        # Load voice if provided
        if voice_path:
            self.load_voice(voice_path)
        
        # Volume control (0.0 to 1.0)
        self.volume = 1.0
        
        # Emotional context tracking
        self.emotional_context = {
            "current_emotion": "neutral",
            "emotion_history": [],
            "valence": 0.0,  # Positive/negative
            "arousal": 0.0,  # Energy level
            "dominance": 0.0  # Control/influence
        }
        
        self.voice_path = voice_path
        logger.info("Voice processor initialized")
        
    def set_volume(self, volume: float):
        """
        Set output volume for speech generation
        
        Args:
            volume: Volume level (0.0 to 1.0, where 1.0 is 100%)
        """
        self.volume = max(0.0, min(1.0, volume))
        logger.info(f"Volume set to {self.volume * 100:.0f}%")
    
    def get_volume(self) -> float:
        """Get current volume level (0.0 to 1.0)"""
        return self.volume
    
    async def transcribe_audio(self, audio_path: Union[str, Path]) -> dict:
        """
        Transcribe audio file to text using Whisper and detect emotion
        
        Args:
            audio_path: Path to audio file
        Returns:
            Dict containing transcribed text and emotional analysis
        """
        # Load and preprocess audio
        audio_input, sr = sf.read(audio_path)
        if len(audio_input.shape) > 1:
            audio_input = audio_input.mean(axis=1)  # Convert stereo to mono
            
        # Process with model
        inputs = self.stt_processor(
            audio_input,
            sampling_rate=sr,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.stt_model.generate(
                **inputs,
                max_length=448,
                return_timestamps=False
            )
            
        # Decode text
        transcription = self.stt_processor.batch_decode(
            outputs, 
            skip_special_tokens=True
        )[0].strip()
        
        # Detect emotion
        emotion_data = self.detect_emotion(audio_input, sr)
        
        return {
            "text": transcription,
            "emotion": emotion_data["emotion"],
            "confidence": emotion_data["confidence"],
            "emotional_context": emotion_data["metrics"]
        }
        
    def generate_speech(
        self, 
        text: str, 
        output_path: Union[str, Path],
        emotion: Optional[str] = None
    ) -> None:
        """
        Generate speech from text using Microsoft's SpeechT5 model
        
        Args:
            text: Text to convert to speech
            output_path: Where to save the audio file
            emotion: Optional emotion to apply to speech
        """
        try:
            # Get appropriate voice embeddings
            speaker_embeddings = None
            if self.current_voice:
                if emotion and emotion in self.current_voice.emotional_style:
                    speaker_embeddings = self.current_voice.emotional_style[emotion]
                else:
                    speaker_embeddings = self.current_voice.embeddings
                    
                # Apply voice characteristics
                if speaker_embeddings is not None:
                    speaker_embeddings = speaker_embeddings * torch.tensor([
                        self.current_voice.characteristics["pitch"],
                        1.0,  # Keep time-dimension unchanged
                        self.current_voice.characteristics["energy"]
                    ])
            
            # Generate speech
            speech = self.tts(
                text,
                forward_params={"speaker_embeddings": speaker_embeddings} if speaker_embeddings is not None else None
            )
            
            # Adjust speed if needed
            audio = speech["audio"]
            if self.current_voice and self.current_voice.characteristics["speed"] != 1.0:
                audio = signal.resample(
                    audio,
                    int(len(audio) / self.current_voice.characteristics["speed"])
                )
            
            # Apply volume control
            audio = audio * self.volume
            
            # Save audio file
            sf.write(output_path, audio, speech["sampling_rate"])
            
        except Exception as e:
            logger.error(f"Error generating speech: {e}")
            # For testing, generate a simple sine wave
            duration = 2  # seconds
            t = np.linspace(0, duration, int(16000 * duration))
            audio = np.sin(2 * np.pi * 440 * t) * self.volume  # Apply volume
            sf.write(output_path, audio, 16000)

    async def process_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[dict, None]:
        """
        Process live audio stream and yield transcribed text with emotional analysis
        
        Args:
            audio_stream: Generator yielding raw audio bytes from Discord
        Yields:
            Dict containing transcribed text and emotional analysis
        """
        buffer = bytearray()
        async for chunk in audio_stream:
            # Add new audio chunk
            buffer.extend(chunk)
            
            # Process when we have enough data
            if len(buffer) >= DISCORD_FRAME_SIZE * 10:  # Reduced chunk size for testing
                # Convert raw bytes to numpy array
                audio_data = np.frombuffer(buffer, dtype=np.int16)
                
                # Convert from stereo to mono by averaging channels
                if DISCORD_CHANNELS > 1:
                    audio_data = audio_data.reshape(-1, DISCORD_CHANNELS).mean(axis=1)
                
                # Normalize to float32 in [-1, 1] range
                audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Resample to 16kHz for Whisper
                resampled = signal.resample_poly(audio_data, 16000, DISCORD_SAMPLE_RATE)
                
                # Process with Whisper
                inputs = self.stt_processor(
                    resampled,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.stt_model.generate(
                        **inputs,
                        max_length=448,
                        return_timestamps=False
                    )
                    
                text = self.stt_processor.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )[0].strip()
                
                if text:
                    # Detect emotion in the audio segment
                    emotion_data = self.detect_emotion(resampled)
                    
                    yield {
                        "text": text,
                        "emotion": emotion_data["emotion"],
                        "confidence": emotion_data["confidence"],
                        "emotional_context": emotion_data["metrics"]
                    }
                    
                # Keep a small overlap for word boundaries (500ms)
                overlap = int(0.5 * DISCORD_SAMPLE_RATE * 2)  # 0.5s * sample_rate * bytes_per_sample
                buffer = buffer[-overlap:] if len(buffer) > overlap else buffer

    def detect_emotion(self, audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Detect emotion in audio using wav2vec model
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sampling rate of audio
            
        Returns:
            Dict containing emotion classification and emotional metrics
        """
        try:
            # Ensure audio data is normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32) / np.max(np.abs(audio_data))
                
            # Get emotion classification
            result = self.emotion_classifier({"sampling_rate": sample_rate, "raw": audio_data})
            emotion = max(result, key=lambda x: x["score"])
            detected_emotion = emotion["label"].lower()  # Normalize emotion label
            confidence = float(emotion["score"])
            
            # Map model label to EMOTIONS dictionary key
            mapped_emotion = EMOTION_LABEL_MAP.get(detected_emotion, "neutral")
            
            # Update emotional context
            self.emotional_context["current_emotion"] = mapped_emotion
            self.emotional_context["emotion_history"].append(mapped_emotion)
            if len(self.emotional_context["emotion_history"]) > 10:
                self.emotional_context["emotion_history"].pop(0)
                
            # Update emotional metrics
            metrics = EMOTIONS[mapped_emotion]
            alpha = confidence  # Use confidence as weight for update
            self.emotional_context["valence"] = (1 - alpha) * self.emotional_context["valence"] + alpha * metrics["valence"]
            self.emotional_context["arousal"] = (1 - alpha) * self.emotional_context["arousal"] + alpha * metrics["arousal"]
            self.emotional_context["dominance"] = (1 - alpha) * self.emotional_context["dominance"] + alpha * metrics["dominance"]
            
            return {
                "emotion": mapped_emotion,
                "confidence": confidence,
                "metrics": self.emotional_context
            }
            
        except Exception as e:
            logger.error(f"Error detecting emotion: {e}")
            return {
                "emotion": "neutral",
                "confidence": 0.0,
                "metrics": self.emotional_context
            }
            
    def load_voice(self, voice_path: str, speaker_name: str = "sanctuary") -> None:
        """
        Load or create Sanctuary's voice profile
        
        Args:
            voice_path: Path to voice sample audio file
            speaker_name: Name for the voice profile
        """
        try:
            # First try to load existing profile
            profile = self.voice_customizer.load_profile(speaker_name.lower())
            
            if profile is None:
                # Create new profile from voice sample
                profile = self.voice_customizer.extract_speaker_embeddings(
                    voice_path,
                    speaker_name
                )
                
            # Set as current voice
            self.current_voice = profile
            logger.info(f"Loaded voice profile for {speaker_name}")
            
        except Exception as e:
            logger.error(f"Failed to load voice: {e}")
            raise