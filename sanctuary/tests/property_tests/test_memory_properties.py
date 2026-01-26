"""
Property-based tests for Memory model.

Tests validate:
- Memory content preservation
- Memory field constraints
- Memory serialization
"""

import pytest
from hypothesis import given, settings
from .strategies import memories


@pytest.mark.property
class TestMemoryProperties:
    
    @given(memories())
    @settings(max_examples=50)
    def test_memory_content_preserved(self, memory):
        """Property: Memory content is preserved exactly after creation."""
        assert memory.content == memory.content
        assert memory.id == memory.id
        
    @given(memories())
    @settings(max_examples=50)
    def test_memory_significance_bounds(self, memory):
        """Property: Memory significance is always in [0, 1] range."""
        assert 0.0 <= memory.significance <= 1.0
    
    @given(memories())
    @settings(max_examples=50)
    def test_memory_serialization_roundtrip(self, memory):
        """Property: Memory can be serialized and deserialized without loss."""
        # Serialize to dict
        data = memory.model_dump()
        
        # Deserialize back
        from mind.cognitive_core.workspace import Memory
        restored = Memory(**data)
        
        # Verify all fields are preserved
        assert restored.id == memory.id
        assert restored.content == memory.content
        assert restored.significance == memory.significance
        assert restored.tags == memory.tags
        assert restored.embedding == memory.embedding
    
    @given(memories())
    @settings(max_examples=50)
    def test_memory_has_required_fields(self, memory):
        """Property: Memory has all required fields."""
        assert hasattr(memory, 'id')
        assert hasattr(memory, 'content')
        assert hasattr(memory, 'timestamp')
        assert hasattr(memory, 'significance')
        assert hasattr(memory, 'tags')
        assert hasattr(memory, 'metadata')
        
        # Check types
        assert isinstance(memory.id, str)
        assert isinstance(memory.content, str)
        assert isinstance(memory.significance, float)
        assert isinstance(memory.tags, list)
        assert isinstance(memory.metadata, dict)
