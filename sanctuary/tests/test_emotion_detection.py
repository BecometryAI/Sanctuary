"""
Test the emotion detection functionality
"""
import os
import pytest

if os.environ.get("CI"):
    pytest.skip("Requires ML models — skipping in CI", allow_module_level=True)
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf

from mind.voice_processor import VoiceProcessor, EMOTIONS

@pytest.fixture
def voice_processor():
    return VoiceProcessor()

@pytest.fixture
def emotional_audio_samples():
    """Generate test audio samples with different emotional characteristics"""
    sr = 16000
    duration = 2  # seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    samples = {
        # Happy: Higher frequency modulation
        "happiness": 0.5 * np.sin(2 * np.pi * 440 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t)),
        
        # Sad: Lower frequency, amplitude modulation
        "sadness": 0.3 * np.sin(2 * np.pi * 220 * t) * (1 - 0.1 * np.sin(2 * np.pi * 2 * t)),
        
        # Anger: Higher amplitude, noise
        "anger": 0.8 * np.sin(2 * np.pi * 400 * t) + 0.2 * np.random.randn(len(t)),
        
        # Neutral: Clean sine wave
        "neutral": 0.5 * np.sin(2 * np.pi * 330 * t)
    }
    
    return samples, sr

def test_emotion_detection(voice_processor, emotional_audio_samples):
    """Test emotion detection on different audio samples"""
    samples, sr = emotional_audio_samples
    
    for emotion, audio in samples.items():
        result = voice_processor.detect_emotion(audio, sr)
        
        assert isinstance(result, dict)
        assert "emotion" in result
        assert "confidence" in result
        assert "metrics" in result
        assert result["confidence"] > 0
        
        # Verify emotional metrics
        metrics = result["metrics"]
        assert -1 <= metrics["valence"] <= 1
        assert -1 <= metrics["arousal"] <= 1
        assert -1 <= metrics["dominance"] <= 1

@pytest.mark.asyncio
async def test_emotional_transcription(voice_processor, emotional_audio_samples):
    """Test transcription with emotion detection"""
    samples, sr = emotional_audio_samples
    temp_path = Path(tempfile.gettempdir()) / "test_emotional_audio.wav"
    
    try:
        # Test with happy audio
        sf.write(temp_path, samples["happiness"], sr)
        result = await voice_processor.transcribe_audio(temp_path)
        
        assert isinstance(result, dict)
        assert "text" in result
        assert "emotion" in result
        assert "confidence" in result
        assert "emotional_context" in result
    finally:
        if temp_path.exists():
            temp_path.unlink()

@pytest.mark.asyncio
async def test_emotional_context_tracking(voice_processor, emotional_audio_samples):
    """Test emotional context tracking over time"""
    samples, sr = emotional_audio_samples
    
    # Process multiple emotions in sequence
    emotions = ["happiness", "sadness", "anger", "neutral"]
    for emotion in emotions:
        result = voice_processor.detect_emotion(samples[emotion], sr)
        assert len(voice_processor.emotional_context["emotion_history"]) > 0
        
    # Check that history is maintained
    assert len(voice_processor.emotional_context["emotion_history"]) <= 10  # Max history length
    
    # Verify emotional metrics are within bounds
    assert -1 <= voice_processor.emotional_context["valence"] <= 1
    assert -1 <= voice_processor.emotional_context["arousal"] <= 1
    assert -1 <= voice_processor.emotional_context["dominance"] <= 1