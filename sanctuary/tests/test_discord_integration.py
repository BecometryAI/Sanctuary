"""
Integration tests for Discord client with voice processing
"""
import os
import pytest

if os.environ.get("CI"):
    pytest.skip("Requires ML models — skipping in CI", allow_module_level=True)
import asyncio
import numpy as np
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

from discord.voice_client import VoiceClient
from discord import VoiceState, Member, Guild
from mind.voice_processor import VoiceProcessor
from mind.discord_client import SanctuaryClient

class MockVoiceClient:
    def __init__(self):
        self.audio_stream = None
        self.recording = False
        
    def is_connected(self):
        return True
        
    async def disconnect(self):
        self.recording = False
        
    def start_recording(self):
        self.recording = True
        
    def stop_recording(self):
        self.recording = False
        
    async def create_stream(self):
        """Simulate Discord audio stream"""
        sr = 48000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(sr * duration))
        
        # Generate different emotional audio patterns
        streams = {
            "happy": 0.5 * np.sin(2 * np.pi * 440 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t)),
            "sad": 0.3 * np.sin(2 * np.pi * 220 * t) * (1 - 0.1 * np.sin(2 * np.pi * 2 * t)),
            "angry": 0.8 * np.sin(2 * np.pi * 400 * t) + 0.2 * np.random.randn(len(t))
        }
        
        for emotion, audio in streams.items():
            # Convert to bytes
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            # Yield in chunks
            chunk_size = 3840  # Discord frame size
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i + chunk_size]
                await asyncio.sleep(0.02)  # Simulate real-time streaming

@pytest.fixture
def mock_discord():
    guild = MagicMock(spec=Guild)
    guild.id = 123456789
    
    member = MagicMock(spec=Member)
    member.guild = guild
    member.id = 987654321
    member.name = "TestUser"
    
    voice_state = MagicMock(spec=VoiceState)
    voice_state.channel = MagicMock()
    voice_state.channel.guild = guild
    
    member.voice = voice_state
    
    return {
        "guild": guild,
        "member": member,
        "voice_state": voice_state
    }

@pytest.fixture
def sanctuary_client():
    client = SanctuaryClient()
    client.voice_processor = VoiceProcessor()
    return client

@pytest.mark.asyncio
async def test_voice_emotion_integration(sanctuary_client, mock_discord):
    """Test voice processing with emotion detection in Discord context"""
    member = mock_discord["member"]
    voice_client = MockVoiceClient()
    
    # Setup voice client
    sanctuary_client.voice_clients = [voice_client]
    
    # Start voice processing
    audio_stream = voice_client.create_stream()
    results = []
    
    async for result in sanctuary_client.voice_processor.process_stream(audio_stream):
        results.append(result)
        if len(results) >= 3:  # Get 3 emotional segments
            break
            
    # Verify emotional analysis
    assert len(results) > 0
    for result in results:
        assert "text" in result
        assert "emotion" in result
        assert "emotional_context" in result
        assert result["confidence"] > 0
        
        # Check emotional metrics
        context = result["emotional_context"]
        assert -1 <= context["valence"] <= 1
        assert -1 <= context["arousal"] <= 1
        assert -1 <= context["dominance"] <= 1

@pytest.mark.asyncio
async def test_voice_state_tracking(sanctuary_client, mock_discord):
    """Test voice state changes with emotional context"""
    member = mock_discord["member"]
    voice_client = MockVoiceClient()
    
    # Simulate voice state update
    await sanctuary_client.on_voice_state_update(
        member,
        member.voice,
        member.voice
    )
    
    # Start recording
    voice_client.start_recording()
    audio_stream = voice_client.create_stream()
    
    # Process some audio
    results = []
    async for result in sanctuary_client.voice_processor.process_stream(audio_stream):
        results.append(result)
        if len(results) >= 2:
            break
    
    # Check emotional state tracking
    assert len(sanctuary_client.voice_processor.emotional_context["emotion_history"]) > 0
    
    # Stop recording
    voice_client.stop_recording()
    
@pytest.mark.asyncio
async def test_emotional_response_generation(sanctuary_client, mock_discord):
    """Test generating responses with emotional awareness"""
    # Process some emotional audio first
    voice_client = MockVoiceClient()
    audio_stream = voice_client.create_stream()
    
    # Get emotional context
    async for result in sanctuary_client.voice_processor.process_stream(audio_stream):
        emotion = result["emotion"]
        context = result["emotional_context"]
        
        # Generate response with matching emotion
        response_text = "This is a test response."
        with open("test_response.wav", "wb") as f:
            sanctuary_client.voice_processor.generate_speech(
                response_text,
                "test_response.wav"
            )
            
        # Verify response was generated
        assert Path("test_response.wav").exists()
        
        # Clean up
        Path("test_response.wav").unlink()
        break  # One response is enough