"""Memory subsystem helper modules for Lyra Emergence."""

# This directory contains helper modules for memory operations (backup, validation, etc.)
# The main MemoryManager class lives in ../memory.py (the file, not this directory)

# To work around Python's module resolution (directory takes precedence over .py file),
# we explicitly re-export MemoryManager from the memory.py file using importlib

import importlib.util
import sys
from pathlib import Path

# Load memory.py as a module
_memory_py_path = Path(__file__).parent.parent / "memory.py"

if _memory_py_path.exists():
    spec = importlib.util.spec_from_file_location("_lyra_memory_py", str(_memory_py_path))
    if spec and spec.loader:
        _memory_py_module = importlib.util.module_from_spec(spec)
        sys.modules["_lyra_memory_py"] = _memory_py_module
        spec.loader.exec_module(_memory_py_module)
        
        # Re-export MemoryManager from memory.py
        MemoryManager = _memory_py_module.MemoryManager
        __all__ = ["MemoryManager"]
    else:
        __all__ = []
else:
    __all__ = []
