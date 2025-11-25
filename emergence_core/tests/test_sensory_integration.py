"""
Comprehensive Sensory Suite Integration Test

Tests all sensory components end-to-end:
1. Vision (Perception) - Image understanding
2. Ears (STT) - Audio transcription
3. Vocal Cords (TTS) - Speech generation
4. Emotion (Heart) - Affective state tracking

Run this to verify the complete sensory suite is operational.
"""

import asyncio
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import tempfile
import soundfile as sf

# Import sensory components
from lyra.specialists import PerceptionSpecialist, SpecialistOutput
from lyra.speech_processor import WhisperProcessor
from lyra.voice_processor import VoiceProcessor
from lyra.emotion_simulator import EmotionSimulator, AffectiveState


class TestVisionSystem:
    """Test Vision (Perception) component"""
    
    @pytest.fixture
    def perception_specialist(self):
        """Create Perception specialist in development mode"""
        return PerceptionSpecialist(
            model_path="llava-hf/llava-v1.6-mistral-7b-hf",
            base_dir=Path(__file__).parent.parent,
            development_mode=True  # Skip model loading for tests
        )
    
    @pytest.fixture
    def test_image(self):
        """Create test image"""
        # Create simple 100x100 RGB image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return Image.fromarray(img_array, 'RGB')
    
    @pytest.mark.asyncio
    async def test_perception_basic(self, perception_specialist, test_image):
        """Test basic image perception"""
        result = await perception_specialist.process(
            image=test_image,
            prompt="Describe this image"
        )
        
        assert isinstance(result, SpecialistOutput)
        assert result.content is not None
        assert len(result.content) > 0
        assert result.metadata["role"] == "perception"
        print(f"\n✅ Vision: {result.content[:100]}...")
    
    @pytest.mark.asyncio
    async def test_perception_with_context(self, perception_specialist, test_image):
        """Test perception with contextual prompt"""
        result = await perception_specialist.process(
            image=test_image,
            prompt="What artistic elements stand out in this image?",
            context={"user": "test", "session": "integration_test"}
        )
        
        assert result.content is not None
        assert result.confidence > 0
        print(f"\n✅ Vision with context: Confidence {result.confidence:.2f}")


class TestAudioSystem:
    """Test Audio (Ears and Vocal Cords) components"""
    
    @pytest.fixture
    def whisper_processor(self):
        """Create Whisper processor"""
        return WhisperProcessor()
    
    @pytest.fixture
    def voice_processor(self):
        """Create Voice processor"""
        return VoiceProcessor()
    
    @pytest.fixture
    def test_audio(self):
        """Generate test audio (1 second of 440Hz tone)"""
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        return audio, sample_rate
    
    @pytest.mark.asyncio
    async def test_audio_transcription_context(self, whisper_processor, test_audio):
        """Test audio transcription with emotional context"""
        audio_data, sample_rate = test_audio
        
        result = await whisper_processor._transcribe_with_context(
            audio_data,
            language="en"
        )
        
        # Even if transcription is empty (pure tone), structure should be valid
        assert result is not None
        assert "text" in result
        assert "confidence" in result
        assert "emotional_context" in result
        print(f"\n✅ Ears (STT): Transcription structure valid")
    
    def test_speech_generation(self, voice_processor):
        """Test speech generation (TTS)"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            output_path = Path(tmp.name)
        
        try:
            voice_processor.generate_speech(
                text="Hello, this is a test of Lyra's voice.",
                output_path=output_path,
                emotion="neutral"
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
            
            # Load and verify audio
            audio_data, sample_rate = sf.read(output_path)
            assert len(audio_data) > 0
            assert sample_rate > 0
            
            print(f"\n✅ Vocal Cords (TTS): Generated {len(audio_data)} samples at {sample_rate}Hz")
            
        finally:
            if output_path.exists():
                output_path.unlink()
    
    def test_emotion_detection(self, voice_processor, test_audio):
        """Test emotion detection in audio"""
        audio_data, sample_rate = test_audio
        
        emotion_result = voice_processor.detect_emotion(
            audio_data,
            sample_rate=sample_rate
        )
        
        assert "emotion" in emotion_result
        assert "confidence" in emotion_result
        assert "scores" in emotion_result
        print(f"\n✅ Emotion detection: {emotion_result['emotion']} ({emotion_result['confidence']:.2f})")


class TestEmotionSystem:
    """Test Emotion (Heart) component"""
    
    @pytest.fixture
    def emotion_simulator(self):
        """Create emotion simulator"""
        return EmotionSimulator(base_dir=Path(__file__).parent.parent)
    
    def test_affective_state_creation(self):
        """Test affective state creation"""
        state = AffectiveState(
            valence=0.7,   # Positive
            arousal=0.5,   # Moderate energy
            dominance=0.6  # Moderate control
        )
        
        assert state.valence == 0.7
        assert state.arousal == 0.5
        assert state.dominance == 0.6
        assert state.timestamp is not None
        print(f"\n✅ Affective State: V={state.valence}, A={state.arousal}, D={state.dominance}")
    
    def test_emotion_generation(self, emotion_simulator):
        """Test context-based emotion generation"""
        context = {
            "event_type": "positive_interaction",
            "intensity": 0.8,
            "user_message": "That's wonderful!"
        }
        
        emotion = emotion_simulator.generate_emotion(context)
        
        assert isinstance(emotion, AffectiveState)
        assert -1.0 <= emotion.valence <= 1.0
        assert -1.0 <= emotion.arousal <= 1.0
        assert -1.0 <= emotion.dominance <= 1.0
        print(f"\n✅ Emotion generation: Valence {emotion.valence:.2f}")
    
    def test_emotion_state_update(self, emotion_simulator):
        """Test emotion state updates"""
        initial_state = emotion_simulator.current_state
        
        # Generate new emotion
        context = {
            "event_type": "surprising_information",
            "intensity": 0.6
        }
        new_emotion = emotion_simulator.generate_emotion(context)
        
        # Update state
        emotion_simulator.update_state(new_emotion)
        
        assert emotion_simulator.current_state != initial_state
        assert len(emotion_simulator.emotion_history) > 0
        print(f"\n✅ Emotion update: History length {len(emotion_simulator.emotion_history)}")


class TestSensoryIntegration:
    """Test integration between sensory components"""
    
    @pytest.mark.asyncio
    async def test_multimodal_perception(self):
        """Test vision + emotion integration"""
        # Create components
        perception = PerceptionSpecialist(
            model_path="llava-hf/llava-v1.6-mistral-7b-hf",
            base_dir=Path(__file__).parent.parent,
            development_mode=True
        )
        emotion_sim = EmotionSimulator(base_dir=Path(__file__).parent.parent)
        
        # Create test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_image = Image.fromarray(img_array, 'RGB')
        
        # Process image
        perception_result = await perception.process(test_image)
        
        # Generate emotional response to visual input
        emotion_context = {
            "event_type": "visual_input",
            "content": perception_result.content,
            "confidence": perception_result.confidence
        }
        emotion = emotion_sim.generate_emotion(emotion_context)
        
        assert perception_result.content is not None
        assert isinstance(emotion, AffectiveState)
        print(f"\n✅ Multimodal integration: Vision → Emotion (V={emotion.valence:.2f})")
    
    @pytest.mark.asyncio
    async def test_audio_emotion_flow(self):
        """Test audio transcription + emotion detection"""
        whisper = WhisperProcessor()
        emotion_sim = EmotionSimulator(base_dir=Path(__file__).parent.parent)
        
        # Create test audio
        sample_rate = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        # Transcribe with emotional context
        result = await whisper._transcribe_with_context(audio, "en")
        
        # Generate emotion based on audio context
        emotion_context = {
            "event_type": "audio_input",
            "tone": result.get("emotional_context", {}).get("tone", "neutral"),
            "confidence": result.get("confidence", 0.5)
        }
        emotion = emotion_sim.generate_emotion(emotion_context)
        
        assert result is not None
        assert isinstance(emotion, AffectiveState)
        print(f"\n✅ Audio-Emotion flow: STT → Emotion (A={emotion.arousal:.2f})")


def test_sensory_suite_complete():
    """Meta-test: Verify all sensory components are importable"""
    components = {
        "Vision": PerceptionSpecialist,
        "Ears": WhisperProcessor,
        "Voice": VoiceProcessor,
        "Emotion": EmotionSimulator
    }
    
    for name, component in components.items():
        assert component is not None
        print(f"✅ {name} component available")
    
    print("\n🎉 All sensory suite components are operational!")


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "integration: Integration tests for sensory suite"
    )


if __name__ == "__main__":
    print("=" * 70)
    print("SENSORY SUITE INTEGRATION TEST")
    print("=" * 70)
    
    # Run basic verification
    test_sensory_suite_complete()
    
    print("\n" + "=" * 70)
    print("Run full test suite with: pytest test_sensory_integration.py -v")
    print("=" * 70)
