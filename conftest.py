"""Root conftest.py — prevents pytest from collecting legacy and hardware-dep tests."""

collect_ignore_glob = [
    # Legacy tests with dead langchain_classic imports
    "sanctuary/tests/test_consciousness_executive_integration.py",
    "sanctuary/tests/test_consciousness_self_awareness_integration.py",
    "sanctuary/tests/test_self_awareness_integration.py",
    "sanctuary/tests/mind/test_memory.py",
    "sanctuary/mind/test_memory.py",
    "sanctuary/mind/tests/test_memory_integration.py",
    # Hardware-dependent tests (soundfile, discord, torch)
    "sanctuary/tests/test_voice_processor.py",
    "sanctuary/tests/test_voice_customization.py",
    "sanctuary/tests/test_gpu_monitor.py",
    "sanctuary/tests/test_discord_integration.py",
    "sanctuary/tests/test_emotion_detection.py",
    # Legacy tests with bare 'mind' imports (need PYTHONPATH=sanctuary)
    "sanctuary/tests/mind/test_metacognition.py",
    "sanctuary/tests/mind/test_well_being.py",
]
