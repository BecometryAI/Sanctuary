"""
Quick Sensory Suite Validation

Verifies that all sensory components can be imported and initialized
without loading heavy ML models. Tests the architecture, not the models.
"""

import asyncio
import sys
from pathlib import Path

# Add emergence_core to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_vision_import():
    """Test Vision (Perception) component imports"""
    try:
        from lyra.specialists import PerceptionSpecialist, SpecialistOutput
        print("[OK] Vision: PerceptionSpecialist imports successfully")
        return True
    except Exception as e:
        print(f"[FAILED] Vision import error: {e}")
        return False

def test_ears_import():
    """Test Ears (STT) component imports"""
    try:
        from lyra.speech_processor import WhisperProcessor
        print("[OK] Ears: WhisperProcessor imports successfully")
        return True
    except Exception as e:
        print(f"[FAILED] Ears import error: {e}")
        return False

def test_voice_import():
    """Test Voice (TTS) component imports"""
    try:
        from lyra.voice_processor import VoiceProcessor
        print("[OK] Voice: VoiceProcessor imports successfully")
        return True
    except Exception as e:
        print(f"[FAILED] Voice import error: {e}")
        return False

def test_emotion_import():
    """Test Emotion component imports"""
    try:
        from lyra.emotion_simulator import EmotionSimulator, AffectiveState
        print("[OK] Emotion: EmotionSimulator imports successfully")
        return True
    except Exception as e:
        print(f"[FAILED] Emotion import error: {e}")
        return False

def test_asr_gateway_import():
    """Test ASR Gateway components import"""
    try:
        from lyra.asr_server import ASRServer
        print("[OK] ASR Server: Imports successfully")
        
        from lyra.mic_client import MicrophoneClient
        print("[OK] Mic Client: Imports successfully")
        
        from lyra.audio_gateway import AudioGateway
        print("[OK] Audio Gateway: Imports successfully")
        
        return True
    except Exception as e:
        print(f"[FAILED] ASR Gateway import error: {e}")
        return False

def test_affective_state_creation():
    """Test basic AffectiveState creation"""
    try:
        from lyra.emotion_simulator import AffectiveState
        
        state = AffectiveState(
            valence=0.7,
            arousal=0.5,
            dominance=0.6
        )
        
        assert state.valence == 0.7
        assert state.arousal == 0.5
        assert state.dominance == 0.6
        assert state.timestamp is not None
        
        print(f"[OK] AffectiveState: V={state.valence}, A={state.arousal}, D={state.dominance}")
        return True
        
    except Exception as e:
        print(f"[FAILED] AffectiveState creation error: {e}")
        return False

def test_whisper_initialization():
    """Test WhisperProcessor can be initialized"""
    try:
        from lyra.speech_processor import WhisperProcessor
        
        processor = WhisperProcessor()
        assert processor.asr_pipeline is not None
        assert processor.device is not None
        assert processor.sample_rate == 16000
        
        print(f"[OK] WhisperProcessor: Initialized (device={processor.device})")
        return True
        
    except Exception as e:
        print(f"[FAILED] WhisperProcessor initialization: {e}")
        return False

async def test_asr_server_structure():
    """Test ASRServer structure without starting server"""
    try:
        from lyra.asr_server import ASRServer
        
        server = ASRServer(host="localhost", port=8765)
        assert server.host == "localhost"
        assert server.port == 8765
        assert server.whisper is not None
        assert server.language == "en"
        assert server.sample_rate == 16000
        
        print(f"[OK] ASRServer: Structure valid (ws://{server.host}:{server.port}, {server.sample_rate}Hz)")
        return True
        
    except Exception as e:
        print(f"[FAILED] ASRServer structure test: {e}")
        return False

def test_mic_client_structure():
    """Test MicrophoneClient structure"""
    try:
        from lyra.mic_client import MicrophoneClient
        
        client = MicrophoneClient(
            server_url="ws://localhost:8765",
            sample_rate=16000
        )
        
        assert client.server_url == "ws://localhost:8765"
        assert client.sample_rate == 16000
        assert client.channels == 1
        
        print(f"[OK] MicrophoneClient: Structure valid (sr={client.sample_rate}Hz)")
        return True
        
    except Exception as e:
        print(f"[FAILED] MicrophoneClient structure test: {e}")
        return False

def test_audio_gateway_structure():
    """Test AudioGateway structure"""
    try:
        from lyra.audio_gateway import AudioGateway
        
        gateway = AudioGateway(
            host="localhost",
            port=8765,
            auto_start_server=False  # Don't actually start server
        )
        
        assert gateway.host == "localhost"
        assert gateway.port == 8765
        assert gateway.sample_rate == 16000
        assert gateway.language == "en"
        
        print(f"[OK] AudioGateway: Structure valid (ws://{gateway.host}:{gateway.port})")
        return True
        
    except Exception as e:
        print(f"[FAILED] AudioGateway structure test: {e}")
        return False


def main():
    """Run all validation tests"""
    print("=" * 70)
    print("SENSORY SUITE QUICK VALIDATION")
    print("=" * 70)
    print()
    
    tests = [
        ("Vision (Perception)", test_vision_import),
        ("Ears (STT)", test_ears_import),
        ("Voice (TTS)", test_voice_import),
        ("Emotion (Heart)", test_emotion_import),
        ("ASR Gateway", test_asr_gateway_import),
        ("AffectiveState Creation", test_affective_state_creation),
        ("WhisperProcessor Init", test_whisper_initialization),
        ("ASRServer Structure", lambda: asyncio.run(test_asr_server_structure())),
        ("MicClient Structure", test_mic_client_structure),
        ("AudioGateway Structure", test_audio_gateway_structure),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nTesting {name}...")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[FAILED] {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    print("=" * 70)
    
    if passed == total:
        print("\n[SUCCESS] All sensory suite components operational!")
        return 0
    else:
        print(f"\n[WARNING] {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
