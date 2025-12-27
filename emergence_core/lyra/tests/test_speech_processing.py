"""
Test suite for speech processing and voice analysis components
"""
import pytest
import numpy as np
import asyncio
from typing import Generator, AsyncGenerator
from ..speech_processor import WhisperProcessor
from ..voice_analyzer import EmotionAnalyzer

@pytest.fixture
def whisper_processor():
    return WhisperProcessor()

@pytest.fixture
def emotion_analyzer():
    return EmotionAnalyzer()

@pytest.fixture
def sample_audio():
    # Generate 1 second of sample audio data at 16kHz
    duration = 1  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # Hz
    return np.sin(2 * np.pi * frequency * t)

def test_emotion_analyzer_initialization(emotion_analyzer):
    """Test emotion analyzer initialization and default states"""
    assert emotion_analyzer.emotion_states is not None
    assert "neutral" in emotion_analyzer.emotion_states
    assert emotion_analyzer.current_context["primary_emotion"] == "neutral"
    assert emotion_analyzer.current_context["confidence"] == 0.0

def test_emotion_analysis(emotion_analyzer, sample_audio):
    """Test basic emotion analysis functionality"""
    result = emotion_analyzer.analyze_segment(sample_audio)
    assert "primary_emotion" in result
    assert "confidence" in result
    assert "secondary_emotions" in result
    assert isinstance(result["confidence"], float)
    assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_whisper_processor_initialization(whisper_processor):
    """Test WhisperProcessor initialization"""
    assert whisper_processor.sample_rate == 16000
    assert whisper_processor.chunk_duration == 30
    assert whisper_processor.min_speech_probability == 0.5
    assert whisper_processor.voice_context is not None

@pytest.mark.asyncio
async def test_audio_stream_processing(whisper_processor, sample_audio):
    """Test audio stream processing with sample data"""
    async def mock_audio_generator() -> AsyncGenerator[np.ndarray, None]:
        # Generate 2 chunks of audio
        for _ in range(2):
            await asyncio.sleep(0)  # Simulate async generation
            yield sample_audio

    async for text in whisper_processor.process_audio_stream(mock_audio_generator()):
        assert isinstance(text, str)
        # Even if no speech is detected, the context should be updated
        assert whisper_processor.voice_context is not None
        assert isinstance(whisper_processor.voice_context["confidence"], float)

@pytest.mark.asyncio
async def test_transcription_with_context(whisper_processor, sample_audio):
    """Test transcription with emotional context"""
    result = await whisper_processor._transcribe_with_context(sample_audio, "en")
    assert result is not None
    assert "emotional_context" in result
    assert "tone" in result["emotional_context"]
    assert "confidence" in result["emotional_context"]
    assert "speaker_consistency" in result["emotional_context"]

def test_voice_context_updates(whisper_processor):
    """Test voice context tracking updates"""
    mock_result = {
        "confidence": 0.95,
        "emotional_context": {
            "tone": "happy"
        }
    }
    whisper_processor._update_voice_context(mock_result)
    assert whisper_processor.voice_context["confidence"] == 0.95
    assert "happy" in whisper_processor.voice_context["emotional_markers"]

@pytest.mark.asyncio
async def test_integration(whisper_processor, emotion_analyzer, sample_audio):
    """Test integration between WhisperProcessor and EmotionAnalyzer"""
    # Test that emotion analyzer results are properly integrated
    emotion_result = emotion_analyzer.analyze_segment(sample_audio)
    assert emotion_result["primary_emotion"] in whisper_processor.emotion_analyzer.emotion_states
    
    # Test that context is maintained across processing
    result = await whisper_processor._transcribe_with_context(sample_audio, "en")
    assert result is not None
    assert result["emotional_context"]["confidence"] > 0