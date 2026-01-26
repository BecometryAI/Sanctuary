"""
Test backward compatibility of the refactored CognitiveCore.

Verifies that all existing usage patterns still work after refactoring
the monolithic core.py into focused modules.
"""

import sys
import os

# Ensure we can import from the project
sys.path.insert(0, os.path.abspath('.'))


def test_import_cognitive_core():
    """Test that CognitiveCore can be imported from the expected location."""
    try:
        from emergence_core.lyra.cognitive_core import CognitiveCore
        print("✅ CognitiveCore import successful")
        return True
    except Exception as e:
        print(f"❌ CognitiveCore import failed: {e}")
        return False


def test_cognitive_core_attributes():
    """Test that CognitiveCore has all expected public attributes."""
    try:
        from emergence_core.lyra.cognitive_core import CognitiveCore
        
        # Test key public methods exist
        expected_methods = [
            'start', 'stop', 'inject_input', 'query_state', 
            'get_metrics', 'get_performance_breakdown',
            'process_language_input', 'chat', 'get_response',
            'save_state', 'restore_state', 
            'enable_auto_checkpoint', 'disable_auto_checkpoint'
        ]
        
        # Test key properties exist
        expected_properties = [
            'workspace', 'running', 'attention', 'perception', 
            'action', 'affect', 'meta_cognition', 'memory',
            'autonomous', 'temporal_awareness', 'identity',
            'language_input', 'language_output', 'checkpoint_manager'
        ]
        
        missing_methods = []
        missing_properties = []
        
        for method in expected_methods:
            if not hasattr(CognitiveCore, method):
                missing_methods.append(method)
        
        for prop in expected_properties:
            if not hasattr(CognitiveCore, prop):
                missing_properties.append(prop)
        
        if missing_methods:
            print(f"❌ Missing methods: {missing_methods}")
            return False
        
        if missing_properties:
            print(f"❌ Missing properties: {missing_properties}")
            return False
        
        print(f"✅ All {len(expected_methods)} methods present")
        print(f"✅ All {len(expected_properties)} properties present")
        return True
        
    except Exception as e:
        print(f"❌ Attribute test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cognitive_core_instantiation():
    """Test that CognitiveCore can be instantiated."""
    try:
        from emergence_core.lyra.cognitive_core import CognitiveCore
        
        # Create instance with default config
        core = CognitiveCore()
        
        # Verify basic attributes
        assert core.config is not None, "Config not initialized"
        assert core.state is not None, "State manager not initialized"
        assert core.subsystems is not None, "Subsystems not initialized"
        assert core.timing is not None, "Timing manager not initialized"
        assert core.lifecycle is not None, "Lifecycle manager not initialized"
        assert core.cycle_executor is not None, "Cycle executor not initialized"
        assert core.loop is not None, "Cognitive loop not initialized"
        
        # Verify subsystem access
        assert core.workspace is not None, "Workspace not accessible"
        assert core.attention is not None, "Attention subsystem not accessible"
        assert core.perception is not None, "Perception subsystem not accessible"
        assert core.action is not None, "Action subsystem not accessible"
        
        print("✅ CognitiveCore instantiation successful")
        print(f"✅ Config: cycle_rate={core.config['cycle_rate_hz']}Hz")
        print(f"✅ All subsystems initialized")
        return True
        
    except Exception as e:
        print(f"❌ Instantiation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_structure():
    """Test that the core module structure is correct."""
    try:
        # Test individual module imports
        from emergence_core.lyra.cognitive_core.core import CognitiveCore
        from emergence_core.lyra.cognitive_core.core.subsystem_coordinator import SubsystemCoordinator
        from emergence_core.lyra.cognitive_core.core.state_manager import StateManager
        from emergence_core.lyra.cognitive_core.core.timing import TimingManager
        from emergence_core.lyra.cognitive_core.core.lifecycle import LifecycleManager
        from emergence_core.lyra.cognitive_core.core.cycle_executor import CycleExecutor
        from emergence_core.lyra.cognitive_core.core.cognitive_loop import CognitiveLoop
        from emergence_core.lyra.cognitive_core.core.action_executor import ActionExecutor
        
        print("✅ All core submodules importable")
        print("✅ Module structure is correct")
        return True
        
    except Exception as e:
        print(f"❌ Module structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all backward compatibility tests."""
    print("=" * 60)
    print("Testing Backward Compatibility of Refactored CognitiveCore")
    print("=" * 60)
    print()
    
    tests = [
        ("Import CognitiveCore", test_import_cognitive_core),
        ("Check attributes", test_cognitive_core_attributes),
        ("Instantiate CognitiveCore", test_cognitive_core_instantiation),
        ("Module structure", test_module_structure),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")
        result = test_func()
        results.append((name, result))
        print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All backward compatibility tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
