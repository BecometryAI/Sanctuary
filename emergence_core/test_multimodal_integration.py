"""
Integration Tests for Multimodal Capabilities

Tests volume control, audio transcription, and image analysis.
"""

import asyncio
import numpy as np
from pathlib import Path
from PIL import Image
import soundfile as sf
import tempfile
import sys

# Add emergence_core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lyra.voice_processor import VoiceProcessor


class TestVolumeControl:
    """Test volume control functionality"""
    
    def test_volume_initialization(self):
        """Test that volume initializes to 1.0"""
        processor = VoiceProcessor()
        assert processor.get_volume() == 1.0
        print("[OK] Volume initializes to 100%")
    
    def test_volume_set_get(self):
        """Test setting and getting volume"""
        processor = VoiceProcessor()
        
        processor.set_volume(0.5)
        assert processor.get_volume() == 0.5
        
        processor.set_volume(0.75)
        assert processor.get_volume() == 0.75
        
        print("[OK] Volume set/get working (tested 50%, 75%)")
    
    def test_volume_clamping(self):
        """Test volume clamping"""
        processor = VoiceProcessor()
        
        # Test upper bound
        processor.set_volume(1.5)
        assert processor.get_volume() == 1.0
        
        # Test lower bound
        processor.set_volume(-0.5)
        assert processor.get_volume() == 0.0
        
        print("[OK] Volume clamping works (1.5->1.0, -0.5->0.0)")
    
    def test_volume_in_generation(self):
        """Test that volume affects generated audio"""
        processor = VoiceProcessor()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate at 100% volume
            processor.set_volume(1.0)
            path1 = Path(tmpdir) / "full.wav"
            processor.generate_speech("Test", path1)
            
            # Generate at 50% volume
            processor.set_volume(0.5)
            path2 = Path(tmpdir) / "half.wav"
            processor.generate_speech("Test", path2)
            
            # Load and compare
            audio1, sr1 = sf.read(path1)
            audio2, sr2 = sf.read(path2)
            
            max1 = np.max(np.abs(audio1))
            max2 = np.max(np.abs(audio2))
            
            ratio = max2 / max1
            assert 0.4 <= ratio <= 0.6, f"Expected ~0.5, got {ratio:.2f}"
            
            print(f"[OK] Volume affects amplitude (ratio: {ratio:.2f})")


class TestAudioTranscription:
    """Test audio file transcription"""
    
    def create_test_audio(self, duration=1.0):
        """Create a test audio file"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = Path(f.name)
        
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        sf.write(path, audio, sample_rate)
        
        return path
    
    def test_transcription_structure(self):
        """Test transcription returns proper structure"""
        processor = VoiceProcessor()
        audio_path = self.create_test_audio()
        
        try:
            result = asyncio.run(processor.transcribe_audio(audio_path))
            
            assert "text" in result
            assert "emotion" in result
            assert "confidence" in result
            assert "emotional_context" in result
            
            print(f"[OK] Transcription structure valid")
            print(f"  Keys: {list(result.keys())}")
            
        finally:
            if audio_path.exists():
                audio_path.unlink()
    
    def test_emotion_detection(self):
        """Test emotion detection"""
        processor = VoiceProcessor()
        audio_path = self.create_test_audio()
        
        try:
            result = asyncio.run(processor.transcribe_audio(audio_path))
            
            valid_emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
            assert result["emotion"] in valid_emotions
            assert 0.0 <= result["confidence"] <= 1.0
            
            print(f"[OK] Emotion detection working")
            print(f"  Emotion: {result['emotion']}, Confidence: {result['confidence']:.2f}")
            
        finally:
            if audio_path.exists():
                audio_path.unlink()


class TestComponentAvailability:
    """Test that all required components are available"""
    
    def test_voice_processor_available(self):
        """Test VoiceProcessor can be imported and initialized"""
        processor = VoiceProcessor()
        assert processor is not None
        assert hasattr(processor, 'set_volume')
        assert hasattr(processor, 'get_volume')
        assert hasattr(processor, 'generate_speech')
        assert hasattr(processor, 'transcribe_audio')
        print("[OK] VoiceProcessor available with all methods")
    
    def test_discord_bot_files_exist(self):
        """Test Discord bot files were created"""
        # Files are in emergence_core directory
        base = Path(__file__).parent
        
        bot_script = base / "run_discord_bot.py"
        assert bot_script.exists(), f"run_discord_bot.py not found at {bot_script}"
        
        env_example = base / ".env.example"
        assert env_example.exists(), f".env.example not found at {env_example}"
        
        print("[OK] Discord bot files exist")
        print(f"  - {bot_script.name}")
        print(f"  - {env_example.name}")
    
    def test_env_example_content(self):
        """Test .env.example has required variables"""
        env_example = Path(__file__).parent / ".env.example"
        content = env_example.read_text()
        
        required = [
            "DISCORD_BOT_TOKEN",
            "DISCORD_GUILD_ID",
            "DISCORD_VOICE_CHANNEL_ID",
            "AUTO_JOIN_VOICE"
        ]
        
        for var in required:
            assert var in content, f"{var} not in .env.example"
        
        print(f"[OK] .env.example contains {len(required)} required variables")
    
    def test_bot_script_structure(self):
        """Test bot script has required components"""
        bot_script = Path(__file__).parent / "run_discord_bot.py"
        content = bot_script.read_text()
        
        required = [
            "LyraBot",
            "join_voice",
            "leave_voice",
            "set_volume",
            "process_message",
            "speak_in_voice"
        ]
        
        for component in required:
            assert component in content, f"{component} not in bot script"
        
        print(f"[OK] Bot script has {len(required)} required components")


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("MULTIMODAL INTEGRATION TESTS")
    print("=" * 70)
    print()
    
    # Volume control tests
    print("Testing Volume Control...")
    volume_tests = TestVolumeControl()
    volume_tests.test_volume_initialization()
    volume_tests.test_volume_set_get()
    volume_tests.test_volume_clamping()
    volume_tests.test_volume_in_generation()
    print()
    
    # Audio transcription tests
    print("Testing Audio Transcription...")
    audio_tests = TestAudioTranscription()
    audio_tests.test_transcription_structure()
    audio_tests.test_emotion_detection()
    print()
    
    # Component availability tests
    print("Testing Component Availability...")
    component_tests = TestComponentAvailability()
    component_tests.test_voice_processor_available()
    component_tests.test_discord_bot_files_exist()
    component_tests.test_env_example_content()
    component_tests.test_bot_script_structure()
    print()
    
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("1. Copy .env.example to .env and add your Discord bot token")
    print("2. Run: python run_discord_bot.py")
    print("3. Test volume control with /volume command")
    print("4. Upload images and audio files in Discord")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
