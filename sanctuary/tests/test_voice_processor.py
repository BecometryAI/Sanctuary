"""
Test the voice processing functionality
"""
import os
import pytest

if os.environ.get("CI"):
    pytest.skip("Requires ML models — skipping in CI", allow_module_level=True)

import asyncio
import numpy as np
from pathlib import Path
import pytest
import soundfile as sf
import tempfile

from mind.voice_processor import VoiceProcessor, DISCORD_SAMPLE_RATE, DISCORD_CHANNELS

@pytest.fixture
def voice_processor():
    return VoiceProcessor()

@pytest.fixture
def sample_audio():
    """Generate a test sine wave audio signal"""
    duration = 2  # seconds
    freq = 440  # A4 note
    t = np.linspace(0, duration, int(DISCORD_SAMPLE_RATE * duration))
    audio = np.sin(2 * np.pi * freq * t)
    
    # Convert to stereo if needed
    if DISCORD_CHANNELS > 1:
        audio = np.tile(audio.reshape(-1, 1), (1, DISCORD_CHANNELS))
        
    # Convert to 16-bit PCM
    audio = (audio * 32767).astype(np.int16)
    return audio.tobytes()

@pytest.fixture
def audio_stream(sample_audio):
    """Simulate a Discord audio stream"""
    async def stream_generator():
        chunk_size = 3840  # Discord frame size
        for i in range(0, len(sample_audio), chunk_size):
            yield sample_audio[i:i + chunk_size]
    return stream_generator()

@pytest.mark.asyncio
async def test_transcribe_audio_file(voice_processor):
    """Test transcribing an audio file"""
    # Create temp audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Generate sine wave
        duration = 2
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Save to file
        sf.write(f.name, audio, sr)
        f.flush()
        
        # Transcribe
        result = await voice_processor.transcribe_audio(f.name)
        assert isinstance(result, dict)
        assert "text" in result
        assert "emotion" in result
        assert "confidence" in result
        assert "emotional_context" in result

def test_generate_speech(voice_processor):
    """Test speech generation"""
    test_text = "Hello, this is a test."
    temp_path = Path(tempfile.gettempdir()) / "test_speech.wav"
    try:
        # Generate speech
        voice_processor.generate_speech(test_text, temp_path)
        
        # Verify file was created and contains audio
        assert temp_path.exists()
        audio, sr = sf.read(temp_path)
        assert len(audio) > 0
        assert sr > 0
    finally:
        # Clean up
        if temp_path.exists():
            temp_path.unlink()

@pytest.mark.asyncio
async def test_process_stream(voice_processor, audio_stream):
    """Test processing a live audio stream"""
    results = []
    async for result in voice_processor.process_stream(audio_stream):
        results.append(result)
        
    # We should get at least one transcription
    assert len(results) > 0
    assert all(isinstance(r, dict) for r in results)
    assert all("text" in r and "emotion" in r for r in results)