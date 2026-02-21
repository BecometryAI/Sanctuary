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
    # Broken imports — nonexistent modules or bare module names
    "sanctuary/tests/test_friction_based_memory.py",  # mind.economy does not exist
    "sanctuary/tests/test_emotional_modulation_standalone.py",  # bare 'import emotional_modulation'
    "sanctuary/tests/test_emotional_modulation_integration.py",  # bare 'import emotional_modulation'
    "sanctuary/tests/test_refactoring_backward_compatibility.py",  # emergence_core.sanctuary does not exist
    # External ML dependency (sklearn) — not installed in CI
    "sanctuary/tests/test_competitive_logic.py",
]
