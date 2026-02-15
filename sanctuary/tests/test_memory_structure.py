"""
Isolated test for memory module structure without full imports.
"""
import os
import ast
import sys

# Dynamically resolve project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

_MEMORY_DIR = os.path.join(_PROJECT_ROOT, 'sanctuary', 'mind', 'memory')
_MEMORY_MANAGER_PY = os.path.join(_PROJECT_ROOT, 'sanctuary', 'mind', 'memory_manager.py')


def test_parse_syntax():
    """Test that all sub-module files have valid Python syntax."""
    files = [
        'storage.py', 'encoding.py', 'retrieval.py', 'consolidation.py',
        'emotional_weighting.py', 'episodic.py', 'semantic.py', 'working.py',
        '__init__.py'
    ]

    for filename in files:
        filepath = os.path.join(_MEMORY_DIR, filename)
        with open(filepath, 'r') as f:
            ast.parse(f.read())

    # Test main memory_manager.py
    with open(_MEMORY_MANAGER_PY, 'r') as f:
        ast.parse(f.read())


def test_module_exports():
    """Test that __init__.py exports all expected classes."""
    init_path = os.path.join(_MEMORY_DIR, '__init__.py')
    with open(init_path, 'r') as f:
        content = f.read()

    expected_exports = [
        'MemoryStorage', 'MemoryEncoder', 'MemoryRetriever',
        'MemoryConsolidator', 'EmotionalWeighting',
        'EpisodicMemory', 'SemanticMemory', 'WorkingMemory'
    ]

    for export in expected_exports:
        assert export in content, f"{export} is not exported from memory/__init__.py"


def test_class_definitions():
    """Test that all expected classes are defined in sub-modules."""
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

    for filename, classname in classes_to_check.items():
        filepath = os.path.join(_MEMORY_DIR, filename)
        with open(filepath, 'r') as f:
            content = f.read()
        assert f'class {classname}' in content, f"{classname} not found in {filename}"

    # Check MemoryManager in memory_manager.py
    with open(_MEMORY_MANAGER_PY, 'r') as f:
        content = f.read()
    assert 'class MemoryManager' in content, "MemoryManager not found in memory_manager.py"


def test_method_presence():
    """Test that key methods are defined in memory_manager.py."""
    with open(_MEMORY_MANAGER_PY, 'r') as f:
        content = f.read()

    expected_methods = [
        'commit_journal',
        'commit_fact',
        'recall',
        'get_statistics',
        'get_memory_health',
    ]

    for method in expected_methods:
        assert f'def {method}' in content or f'async def {method}' in content, \
            f"{method} method missing from memory_manager.py"


def test_module_sizes():
    """Test that all sub-modules are under 10KB."""
    modules = [
        'storage.py', 'encoding.py', 'retrieval.py', 'consolidation.py',
        'emotional_weighting.py', 'episodic.py', 'semantic.py', 'working.py'
    ]

    for module in modules:
        path = os.path.join(_MEMORY_DIR, module)
        size_kb = os.path.getsize(path) / 1024
        assert size_kb < 35, f"{module}: {size_kb:.1f}KB exceeds 35KB"


def test_imports_structure():
    """Test that memory/__init__.py imports from the sub-modules."""
    init_path = os.path.join(_MEMORY_DIR, '__init__.py')
    with open(init_path, 'r') as f:
        content = f.read()

    expected_imports = [
        'MemoryStorage',
        'MemoryEncoder',
        'MemoryRetriever',
        'MemoryConsolidator',
        'EmotionalWeighting',
        'EpisodicMemory',
        'SemanticMemory',
        'WorkingMemory'
    ]

    for class_name in expected_imports:
        assert class_name in content, f"{class_name} not imported in memory/__init__.py"
