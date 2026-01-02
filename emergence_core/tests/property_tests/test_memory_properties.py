"""
Property-based tests for Memory system invariants.

Tests cover:
- Memory content preservation
- Significance bounds
- Timestamp ordering
- Tag management
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from datetime import datetime, timedelta

from lyra.cognitive_core.workspace import Memory
from .strategies import (
    memories,
    memory_lists,
    make_unique_by_id,
)


@pytest.mark.property
class TestMemoryProperties:
    """Property-based tests for Memory system invariants."""
    
    @given(memories())
    @settings(max_examples=100)
    def test_memory_significance_bounded(self, memory):
        """Property: Memory significance is always in [0, 1]."""
        assert 0.0 <= memory.significance <= 1.0
    
    @given(memories())
    @settings(max_examples=100)
    def test_memory_has_valid_timestamp(self, memory):
        """Property: Memory timestamp is valid and is a datetime object."""
        assert isinstance(memory.timestamp, datetime)
        # Timestamp should be within reasonable bounds (2024-2025)
        assert datetime(2024, 1, 1) <= memory.timestamp <= datetime(2026, 1, 1)
    
    @given(memories())
    @settings(max_examples=100)
    def test_memory_content_preserved(self, memory):
        """Property: Memory content is preserved through serialization."""
        # Serialize to dict
        data = memory.model_dump()
        
        # Deserialize back
        restored = Memory(**data)
        
        # Content should be identical
        assert restored.content == memory.content
        assert restored.id == memory.id
        assert restored.significance == memory.significance
    
    @given(memory_lists)
    @settings(max_examples=50)
    def test_memory_list_uniqueness(self, memories_list):
        """Property: Making memories unique by ID reduces to unique set."""
        unique_memories = make_unique_by_id(memories_list)
        
        # All IDs should be unique
        ids = [m.id for m in unique_memories]
        assert len(ids) == len(set(ids))
        
        # Count should be <= original count
        assert len(unique_memories) <= len(memories_list)
    
    @given(memory_lists)
    @settings(max_examples=50)
    def test_memory_timestamp_ordering(self, memories_list):
        """Property: Memories can be sorted by timestamp."""
        assume(len(memories_list) >= 2)
        
        # Sort by timestamp
        sorted_memories = sorted(memories_list, key=lambda m: m.timestamp)
        
        # Verify ordering
        for i in range(len(sorted_memories) - 1):
            assert sorted_memories[i].timestamp <= sorted_memories[i + 1].timestamp
    
    @given(memories(), st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=10))
    @settings(max_examples=50)
    def test_memory_tags_management(self, memory, new_tags):
        """Property: Memory tags can be updated and preserved."""
        original_tags = memory.tags.copy()
        
        # Add new tags
        combined_tags = list(set(original_tags + new_tags))
        memory.tags = combined_tags
        
        # All original tags should still be present
        for tag in original_tags:
            assert tag in memory.tags
    
    @given(memories())
    @settings(max_examples=100)
    def test_memory_id_immutable(self, memory):
        """Property: Memory ID should remain constant."""
        original_id = memory.id
        
        # Serialize and deserialize
        data = memory.model_dump()
        restored = Memory(**data)
        
        # ID should be unchanged
        assert restored.id == original_id
    
    @given(memory_lists)
    @settings(max_examples=50)
    def test_memory_filtering_by_significance(self, memories_list):
        """Property: Filtering by significance threshold works correctly."""
        assume(len(memories_list) > 0)
        
        threshold = 0.5  # Use fixed threshold for simplicity
        
        # Filter memories above threshold
        significant_memories = [m for m in memories_list if m.significance >= threshold]
        
        # All filtered memories should meet threshold
        for memory in significant_memories:
            assert memory.significance >= threshold
        
        # No filtered memory should be below threshold
        filtered_ids = {m.id for m in significant_memories}
        for memory in memories_list:
            if memory.id in filtered_ids:
                assert memory.significance >= threshold
    
    @given(memories(), st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=100)
    def test_memory_significance_update(self, memory, new_significance):
        """Property: Memory significance can be updated within bounds."""
        # Update significance
        memory.significance = new_significance
        
        # Should remain within bounds
        assert 0.0 <= memory.significance <= 1.0
    
    @given(memory_lists)
    @settings(max_examples=50)
    def test_memory_collection_operations(self, memories_list):
        """Property: Memory collections support standard operations."""
        assume(len(memories_list) > 0)
        
        # Test count
        count = len(memories_list)
        assert count >= 1
        
        # Test iteration
        iterated_count = 0
        for memory in memories_list:
            assert isinstance(memory, Memory)
            iterated_count += 1
        assert iterated_count == count
        
        # Test indexing (if list)
        first_memory = memories_list[0]
        assert isinstance(first_memory, Memory)
    
    @given(memories())
    @settings(max_examples=100)
    def test_memory_metadata_preservation(self, memory):
        """Property: Memory metadata is preserved through operations."""
        original_metadata = memory.metadata.copy()
        
        # Serialize
        data = memory.model_dump()
        
        # Deserialize
        restored = Memory(**data)
        
        # Metadata should be preserved
        assert restored.metadata == original_metadata
    
    @given(memories())
    @settings(max_examples=100)
    def test_memory_embedding_optional(self, memory):
        """Property: Memory embedding is optional and can be None or list."""
        # Embedding should be None or a list of floats
        assert memory.embedding is None or isinstance(memory.embedding, list)
        
        if memory.embedding is not None:
            # Should be a list of floats
            assert all(isinstance(x, (int, float)) for x in memory.embedding)
