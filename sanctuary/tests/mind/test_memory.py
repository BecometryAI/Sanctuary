import shutil
import os
import pytest
"""
Test for Sanctuary Persistent Memory System
"""

# Import from memory_legacy.py (renamed from memory.py)
from mind.memory_legacy import MemoryManager
from chromadb.config import Settings
from unittest.mock import Mock

@pytest.mark.integration
def test_memory():
    # Use a unique temporary directory for each test run
    import tempfile
    test_memories_path = tempfile.mkdtemp(prefix="test_memories_")
    shared_settings = Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )
    print(f"[TEST] shared_settings id: {id(shared_settings)}, contents: {shared_settings}")
    mem = MemoryManager(persistence_dir=test_memories_path, chain_dir="test_chain", chroma_settings=shared_settings)
    
    # Mock the vector_db.index() to skip LangChain/ChromaDB embedding issues
    mem.vector_db.index = Mock()
    
    experience = {"event": "Test event", "details": "Testing episodic memory."}
    mem.store_experience(experience)
    mem.update_working_memory("test_key", "test_value")
    assert mem.get_working_memory("test_key") == "test_value"
    mem.consolidate_memories()
    print("Persistent memory test passed.")

if __name__ == "__main__":
    test_memory()
