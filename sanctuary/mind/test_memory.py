"""
Test for Sanctuary Persistent Memory System
"""
import shutil
import os
import pytest

if os.environ.get("CI"):
    pytest.skip("Requires chromadb with models — skipping in CI", allow_module_level=True)

# Import from memory_legacy.py (renamed from memory.py)
from .memory_legacy import MemoryManager
from chromadb.config import Settings

def test_memory():
    # Clean up the entire test_memories directory before running test
    test_memories_path = "test_memories"
    if os.path.exists(test_memories_path):
        shutil.rmtree(test_memories_path)
    shared_settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )
    print(f"[TEST] shared_settings id: {id(shared_settings)}, contents: {shared_settings}")
    mem = MemoryManager(persistence_dir="test_memories", chain_dir="test_chain", chroma_settings=shared_settings)
    experience = {"event": "Test event", "details": "Testing episodic memory."}
    mem.store_experience(experience)
    mem.update_working_memory("test_key", "test_value")
    assert mem.get_working_memory("test_key") == "test_value"
    mem.consolidate_memories()
    print("Persistent memory test passed.")

if __name__ == "__main__":
    test_memory()
