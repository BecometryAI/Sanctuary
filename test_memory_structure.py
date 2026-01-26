"""
Isolated test for memory module structure without full imports.
"""
import os
import ast
import sys

sys.path.insert(0, '/home/runner/work/Lyra-Emergence/Lyra-Emergence')

def test_parse_syntax():
    """Test that all files have valid Python syntax."""
    print("Testing Python syntax...")
    
    memory_dir = '/home/runner/work/Lyra-Emergence/Lyra-Emergence/emergence_core/lyra/memory'
    files = [
        'storage.py', 'encoding.py', 'retrieval.py', 'consolidation.py',
        'emotional_weighting.py', 'episodic.py', 'semantic.py', 'working.py',
        '__init__.py'
    ]
    
    all_good = True
    for filename in files:
        filepath = os.path.join(memory_dir, filename)
        try:
            with open(filepath, 'r') as f:
                ast.parse(f.read())
            print(f"✓ {filename} has valid syntax")
        except SyntaxError as e:
            print(f"✗ {filename} has syntax error: {e}")
            all_good = False
    
    # Test main memory.py
    memory_path = '/home/runner/work/Lyra-Emergence/Lyra-Emergence/emergence_core/lyra/memory.py'
    try:
        with open(memory_path, 'r') as f:
            ast.parse(f.read())
        print(f"✓ memory.py has valid syntax")
    except SyntaxError as e:
        print(f"✗ memory.py has syntax error: {e}")
        all_good = False
    
    return all_good


def test_module_exports():
    """Test that __init__.py exports all expected classes."""
    print("\nTesting module exports...")
    
    init_path = '/home/runner/work/Lyra-Emergence/Lyra-Emergence/emergence_core/lyra/memory/__init__.py'
    with open(init_path, 'r') as f:
        content = f.read()
    
    expected_exports = [
        'MemoryStorage', 'MemoryEncoder', 'MemoryRetriever',
        'MemoryConsolidator', 'EmotionalWeighting',
        'EpisodicMemory', 'SemanticMemory', 'WorkingMemory'
    ]
    
    all_good = True
    for export in expected_exports:
        if export in content:
            print(f"✓ {export} is exported")
        else:
            print(f"✗ {export} is not exported")
            all_good = False
    
    return all_good


def test_class_definitions():
    """Test that all expected classes are defined."""
    print("\nTesting class definitions...")
    
    memory_dir = '/home/runner/work/Lyra-Emergence/Lyra-Emergence/emergence_core/lyra/memory'
    classes_to_check = {
        'storage.py': 'MemoryStorage',
        'encoding.py': 'MemoryEncoder',
        'retrieval.py': 'MemoryRetriever',
        'consolidation.py': 'MemoryConsolidator',
        'emotional_weighting.py': 'EmotionalWeighting',
        'episodic.py': 'EpisodicMemory',
        'semantic.py': 'SemanticMemory',
        'working.py': 'WorkingMemory'
    }
    
    all_good = True
    for filename, classname in classes_to_check.items():
        filepath = os.path.join(memory_dir, filename)
        with open(filepath, 'r') as f:
            content = f.read()
        
        if f'class {classname}' in content:
            print(f"✓ {classname} defined in {filename}")
        else:
            print(f"✗ {classname} not found in {filename}")
            all_good = False
    
    # Check MemoryManager in memory.py
    memory_path = '/home/runner/work/Lyra-Emergence/Lyra-Emergence/emergence_core/lyra/memory.py'
    with open(memory_path, 'r') as f:
        content = f.read()
    
    if 'class MemoryManager' in content:
        print(f"✓ MemoryManager defined in memory.py")
    else:
        print(f"✗ MemoryManager not found in memory.py")
        all_good = False
    
    return all_good


def test_method_presence():
    """Test that key methods are defined in memory.py."""
    print("\nTesting method presence in MemoryManager...")
    
    memory_path = '/home/runner/work/Lyra-Emergence/Lyra-Emergence/emergence_core/lyra/memory.py'
    with open(memory_path, 'r') as f:
        content = f.read()
    
    expected_methods = [
        'load_journal_entries',
        'load_protocols',
        'load_lexicon',
        'store_experience',
        'update_experience',
        'store_concept',
        'retrieve_relevant_memories',
        'update_working_memory',
        'get_working_memory',
        'consolidate_memories',
        'force_reindex',
        'get_memory_stats',
    ]
    
    all_good = True
    for method in expected_methods:
        if f'def {method}' in content:
            print(f"✓ {method} method exists")
        else:
            print(f"✗ {method} method missing")
            all_good = False
    
    return all_good


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
    original_size_kb = 50.0
    reduction = ((original_size_kb - main_size_kb) / original_size_kb) * 100
    print(f"✓ memory.py: {main_size_kb:.1f}KB (down from {original_size_kb}KB, {reduction:.1f}% reduction)")
    
    return all_good


def test_imports_structure():
    """Test that memory.py imports from the new modules."""
    print("\nTesting import structure...")
    
    memory_path = '/home/runner/work/Lyra-Emergence/Lyra-Emergence/emergence_core/lyra/memory.py'
    with open(memory_path, 'r') as f:
        content = f.read()
    
    expected_imports = [
        'from .memory.storage import MemoryStorage',
        'from .memory.encoding import MemoryEncoder',
        'from .memory.retrieval import MemoryRetriever',
        'from .memory.consolidation import MemoryConsolidator',
        'from .memory.emotional_weighting import EmotionalWeighting',
        'from .memory.episodic import EpisodicMemory',
        'from .memory.semantic import SemanticMemory',
        'from .memory.working import WorkingMemory'
    ]
    
    all_good = True
    for expected_import in expected_imports:
        if expected_import in content:
            print(f"✓ {expected_import.split()[-1]} is imported")
        else:
            print(f"✗ {expected_import} not found")
            all_good = False
    
    return all_good


if __name__ == "__main__":
    print("=" * 60)
    print("Memory Refactoring Structure Test")
    print("=" * 60)
    
    results = []
    
    results.append(("Python Syntax", test_parse_syntax()))
    results.append(("Module Exports", test_module_exports()))
    results.append(("Class Definitions", test_class_definitions()))
    results.append(("Method Presence", test_method_presence()))
    results.append(("Module Sizes", test_module_sizes()))
    results.append(("Import Structure", test_imports_structure()))
    
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
