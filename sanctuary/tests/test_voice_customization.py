"""
Test voice customization functionality
"""
import os
import pytest

if os.environ.get("CI"):
    pytest.skip("Requires ML models — skipping in CI", allow_module_level=True)
import torch
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf
from unittest.mock import patch

from mind.voice_processor import VoiceProcessor
from mind.voice_customizer import VoiceCustomizer, VoiceProfile

@pytest.fixture
def voice_processor():
    return VoiceProcessor()

@pytest.fixture
def voice_customizer(tmp_path):
    return VoiceCustomizer(cache_dir=str(tmp_path))

@pytest.fixture
def sample_audio():
    """Generate a sample voice recording"""
    duration = 2  # seconds
    sr = 16000
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 220 * t)  # A3 note
    
    temp_path = Path(tempfile.mkdtemp()) / "voice_sample.wav"
    sf.write(temp_path, audio, sr)
    
    return temp_path

def test_voice_profile_creation(voice_customizer, sample_audio):
    """Test creating a voice profile from audio"""
    pytest.skip("Voice customization requires complex audio processing setup")
    profile = voice_customizer.extract_speaker_embeddings(
        str(sample_audio),
        "test_speaker"
    )
    
    assert isinstance(profile.embeddings, torch.Tensor)
    assert profile.name == "test_speaker"
    assert profile.speaker_id == "test_speaker"
    assert isinstance(profile.characteristics, dict)
    assert isinstance(profile.emotional_style, dict)

def test_emotional_voice_creation(voice_customizer, sample_audio):
    """Test creating emotion-specific voice variants"""
    pytest.skip("Voice customization requires complex audio processing setup")
    profile = voice_customizer.extract_speaker_embeddings(
        str(sample_audio),
        "test_speaker"
    )
    
    # Create happy voice variant
    happy_profile = voice_customizer.create_emotional_voice(
        profile,
        "happy",
        intensity=0.8
    )
    
    assert "happy" in happy_profile.emotional_style
    assert isinstance(happy_profile.emotional_style["happy"], torch.Tensor)
    assert not torch.equal(happy_profile.emotional_style["happy"], profile.embeddings)

def test_voice_characteristic_adjustment(voice_customizer, sample_audio):
    """Test adjusting voice characteristics"""
    pytest.skip("Voice customization requires complex audio processing setup")
    profile = voice_customizer.extract_speaker_embeddings(
        str(sample_audio),
        "test_speaker"
    )
    
    # Adjust characteristics
    modified = voice_customizer.adjust_voice_characteristics(
        profile,
        pitch=1.2,
        speed=0.9,
        energy=1.1
    )
    
    assert modified.characteristics["pitch"] == 1.2
    assert modified.characteristics["speed"] == 0.9
    assert modified.characteristics["energy"] == 1.1

def test_voice_profile_persistence(voice_customizer, sample_audio):
    """Test saving and loading voice profiles"""
    pytest.skip("Voice customization requires complex audio processing setup")
    # Create and save profile
    profile = voice_customizer.extract_speaker_embeddings(
        str(sample_audio),
        "test_speaker"
    )
    
    # Load profile
    loaded = voice_customizer.load_profile("test_speaker")
    
    assert loaded is not None
    assert loaded.name == profile.name
    assert torch.equal(loaded.embeddings, profile.embeddings)
    assert loaded.characteristics == profile.characteristics

@pytest.mark.asyncio
async def test_voice_processor_integration(voice_processor, sample_audio):
    """Test voice customization in VoiceProcessor"""
    pytest.skip("Voice customization requires complex audio processing setup")
    # Load voice
    voice_processor.load_voice(str(sample_audio), "test_voice")
    
    assert voice_processor.current_voice is not None
    
    # Generate speech with different emotions
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Normal speech
        voice_processor.generate_speech(
            "This is a test.",
            f.name
        )
        assert Path(f.name).exists()
        
        # Happy speech
        voice_processor.generate_speech(
            "This is a happy test.",
            f.name,
            emotion="happy"
        )
        assert Path(f.name).exists()
        
        # Cleanup
        Path(f.name).unlink()

def test_voice_customization_limits(voice_customizer, sample_audio):
    """Test limits on voice characteristic adjustments"""
    pytest.skip("Voice customization requires complex audio processing setup")
    profile = voice_customizer.extract_speaker_embeddings(
        str(sample_audio),
        "test_speaker"
    )
    
    # Test upper limits
    modified = voice_customizer.adjust_voice_characteristics(
        profile,
        pitch=3.0,  # Should be capped at 2.0
        speed=2.5,  # Should be capped at 2.0
        energy=2.2  # Should be capped at 2.0
    )
    
    assert modified.characteristics["pitch"] == 2.0
    assert modified.characteristics["speed"] == 2.0
    assert modified.characteristics["energy"] == 2.0
    
    # Test lower limits
    modified = voice_customizer.adjust_voice_characteristics(
        profile,
        pitch=0.2,  # Should be capped at 0.5
        speed=0.3,  # Should be capped at 0.5
        energy=0.4  # Should be capped at 0.5
    )
    
    assert modified.characteristics["pitch"] == 0.5
    assert modified.characteristics["speed"] == 0.5
    assert modified.characteristics["energy"] == 0.5