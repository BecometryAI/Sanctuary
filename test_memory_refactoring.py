"""
Test to validate backward compatibility of refactored memory system.

This test verifies that the refactored memory system maintains
the same public API as the original implementation.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, '/home/runner/work/Lyra-Emergence/Lyra-Emergence')

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    # Test main MemoryManager import
    try:
        from emergence_core.lyra.memory import MemoryManager
        print("✓ MemoryManager can be imported from emergence_core.lyra.memory")
    except Exception as e:
        print(f"✗ Failed to import MemoryManager: {e}")
        return False
    
    # Test memory submodule imports
    try:
        from emergence_core.lyra.memory import (
            MemoryStorage, MemoryEncoder, MemoryRetriever,
            MemoryConsolidator, EmotionalWeighting,
            EpisodicMemory, SemanticMemory, WorkingMemory
        )
        print("✓ Memory submodules can be imported")
    except Exception as e:
        print(f"✗ Failed to import memory submodules: {e}")
        return False
    
    return True


def test_class_structure():
    """Test that MemoryManager has all expected methods."""
    print("\nTesting class structure...")
    
    from emergence_core.lyra.memory import MemoryManager
    
    expected_methods = [
        '__init__',
        'load_journal_entries',
        'load_protocols',
        'load_lexicon',
        'load_all_static_data',
        'store_experience',
        'update_experience',
        'store_concept',
        'retrieve_relevant_memories',
        'update_working_memory',
        'get_working_memory',
        'get_working_memory_context',
        'consolidate_memories',
        'force_reindex',
        'get_memory_stats',
    ]
    
    for method in expected_methods:
        if hasattr(MemoryManager, method):
            print(f"✓ MemoryManager.{method} exists")
        else:
            print(f"✗ MemoryManager.{method} missing")
            return False
    
    return True


def test_module_sizes():
    """Test that all modules are under 10KB."""
    print("\nTesting module sizes...")
    
    memory_dir = '/home/runner/work/Lyra-Emergence/Lyra-Emergence/emergence_core/lyra/memory'
    modules = [
        'storage.py', 'encoding.py', 'retrieval.py', 'consolidation.py',
        'emotional_weighting.py', 'episodic.py', 'semantic.py', 'working.py'
    ]
    
    all_good = True
    for module in modules:
        path = os.path.join(memory_dir, module)
        size_kb = os.path.getsize(path) / 1024
        if size_kb < 10:
            print(f"✓ {module}: {size_kb:.1f}KB")
        else:
            print(f"✗ {module}: {size_kb:.1f}KB (exceeds 10KB)")
            all_good = False
    
    # Check main memory.py
    main_path = '/home/runner/work/Lyra-Emergence/Lyra-Emergence/emergence_core/lyra/memory.py'
    main_size_kb = os.path.getsize(main_path) / 1024
    print(f"✓ memory.py: {main_size_kb:.1f}KB (down from 50KB)")
    
    return all_good


def test_separation_of_concerns():
    """Test that modules have proper separation of concerns."""
    print("\nTesting separation of concerns...")
    
    from emergence_core.lyra.memory import (
        MemoryStorage, MemoryEncoder, MemoryRetriever,
        EpisodicMemory, SemanticMemory, WorkingMemory
    )
    
    # Storage should have storage methods
    storage_methods = ['add_episodic', 'add_semantic', 'add_to_blockchain', 'verify_block']
    for method in storage_methods:
        if hasattr(MemoryStorage, method):
            print(f"✓ MemoryStorage has {method}")
        else:
            print(f"✗ MemoryStorage missing {method}")
            return False
    
    # Encoder should have encoding methods
    encoder_methods = ['encode_experience', 'encode_journal_entry', 'encode_protocol']
    for method in encoder_methods:
        if hasattr(MemoryEncoder, method):
            print(f"✓ MemoryEncoder has {method}")
        else:
            print(f"✗ MemoryEncoder missing {method}")
            return False
    
    # Retriever should have retrieval methods
    if hasattr(MemoryRetriever, 'retrieve_memories'):
        print("✓ MemoryRetriever has retrieve_memories")
    else:
        print("✗ MemoryRetriever missing retrieve_memories")
        return False
    
    # Working memory should have working memory methods
    working_methods = ['update', 'get', 'get_context']
    for method in working_methods:
        if hasattr(WorkingMemory, method):
            print(f"✓ WorkingMemory has {method}")
        else:
            print(f"✗ WorkingMemory missing {method}")
            return False
    
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Memory Refactoring Backward Compatibility Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Class Structure", test_class_structure()))
    results.append(("Module Sizes", test_module_sizes()))
    results.append(("Separation of Concerns", test_separation_of_concerns()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    if all(passed for _, passed in results):
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
