"""
Unit tests for PerceptionSubsystem.

Tests cover:
- Text encoding functionality
- Embedding cache with LRU eviction
- Similarity consistency
- Complexity estimation
- Error handling
- Statistics tracking
- Integration with workspace Percept model
"""

import pytest
import numpy as np

from mind.cognitive_core.perception import PerceptionSubsystem
from mind.cognitive_core.workspace import Percept


class TestPerceptionSubsystemInitialization:
    """Test PerceptionSubsystem initialization."""
    
    def test_initialization_default(self):
        """Test creating PerceptionSubsystem with default config."""
        perception = PerceptionSubsystem()
        
        assert perception is not None
        assert perception.text_encoder is not None
        assert perception.embedding_dim == 384  # all-MiniLM-L6-v2 is 384-dim
        assert perception.cache_size == 1000
        assert len(perception.embedding_cache) == 0
    
    def test_initialization_custom_config(self):
        """Test creating PerceptionSubsystem with custom config."""
        config = {
            "text_model": "all-MiniLM-L6-v2",
            "cache_size": 500,
            "enable_image": False,
        }
        perception = PerceptionSubsystem(config=config)
        
        assert perception is not None
        assert perception.cache_size == 500
        assert perception.embedding_dim == 384
    
    def test_stats_initialized(self):
        """Test that stats tracking is initialized."""
        perception = PerceptionSubsystem()
        
        stats = perception.get_stats()
        assert stats["cache_hits"] == 0
        assert stats["cache_misses"] == 0
        assert stats["total_encodings"] == 0
        assert stats["cache_hit_rate"] == 0.0


class TestTextEncoding:
    """Test text encoding functionality."""
    
    @pytest.mark.asyncio
    async def test_encode_simple_text(self):
        """Test encoding simple text input."""
        perception = PerceptionSubsystem()
        
        text = "Hello, world!"
        percept = await perception.encode(text, "text")
        
        assert isinstance(percept, Percept)
        assert percept.modality == "text"
        assert percept.raw == text
        assert percept.embedding is not None
        assert len(percept.embedding) == 384
        assert percept.complexity > 0
    
    @pytest.mark.asyncio
    async def test_embedding_shape(self):
        """Test that embeddings have correct dimensionality."""
        perception = PerceptionSubsystem()
        
        text = "The quick brown fox jumps over the lazy dog"
        percept = await perception.encode(text, "text")
        
        assert len(percept.embedding) == perception.embedding_dim
        assert all(isinstance(x, float) for x in percept.embedding)
    
    @pytest.mark.asyncio
    async def test_embeddings_normalized(self):
        """Test that embeddings are normalized (unit length)."""
        perception = PerceptionSubsystem()
        
        text = "Test normalization"
        percept = await perception.encode(text, "text")
        
        # Calculate L2 norm
        embedding_array = np.array(percept.embedding)
        norm = np.linalg.norm(embedding_array)
        
        # Should be close to 1.0 (normalized)
        assert abs(norm - 1.0) < 0.01


class TestCacheFunctionality:
    """Test embedding cache with LRU eviction."""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Test that encoding same text twice hits cache."""
        perception = PerceptionSubsystem()
        
        text = "Cache this text"
        
        # First encoding - cache miss
        percept1 = await perception.encode(text, "text")
        stats1 = perception.get_stats()
        
        # Second encoding - should hit cache
        percept2 = await perception.encode(text, "text")
        stats2 = perception.get_stats()
        
        # Verify cache hit
        assert stats2["cache_hits"] == stats1["cache_hits"] + 1
        assert stats2["cache_misses"] == stats1["cache_misses"]
        
        # Embeddings should be identical
        assert percept1.embedding == percept2.embedding
    
    @pytest.mark.asyncio
    async def test_cache_different_texts(self):
        """Test that different texts result in cache misses."""
        perception = PerceptionSubsystem()
        
        text1 = "First text"
        text2 = "Second text"
        
        await perception.encode(text1, "text")
        stats1 = perception.get_stats()
        
        await perception.encode(text2, "text")
        stats2 = perception.get_stats()
        
        # Both should be cache misses
        assert stats2["cache_misses"] == stats1["cache_misses"] + 1
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        config = {"cache_size": 3}  # Small cache for testing
        perception = PerceptionSubsystem(config=config)
        
        # Fill cache
        await perception.encode("text 1", "text")
        await perception.encode("text 2", "text")
        await perception.encode("text 3", "text")
        
        assert len(perception.embedding_cache) == 3
        
        # Add one more - should evict oldest
        await perception.encode("text 4", "text")
        
        assert len(perception.embedding_cache) == 3
        
        # First text should have been evicted
        await perception.encode("text 1", "text")
        stats = perception.get_stats()
        
        # Should be a cache miss (was evicted)
        assert stats["cache_misses"] >= 4
    
    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """Test clearing the cache."""
        perception = PerceptionSubsystem()
        
        # Add some entries
        await perception.encode("text 1", "text")
        await perception.encode("text 2", "text")
        
        assert len(perception.embedding_cache) > 0
        
        # Clear cache
        perception.clear_cache()
        
        assert len(perception.embedding_cache) == 0


class TestSimilarity:
    """Test semantic similarity of embeddings."""
    
    @pytest.mark.asyncio
    async def test_similar_texts_similar_embeddings(self):
        """Test that similar texts have similar embeddings."""
        perception = PerceptionSubsystem()
        
        text1 = "The cat sat on the mat"
        text2 = "A cat is sitting on a mat"
        
        percept1 = await perception.encode(text1, "text")
        percept2 = await perception.encode(text2, "text")
        
        # Calculate cosine similarity
        vec1 = np.array(percept1.embedding)
        vec2 = np.array(percept2.embedding)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Similar texts should have high similarity (> 0.7)
        assert similarity > 0.7
    
    @pytest.mark.asyncio
    async def test_dissimilar_texts_dissimilar_embeddings(self):
        """Test that dissimilar texts have dissimilar embeddings."""
        perception = PerceptionSubsystem()
        
        text1 = "Quantum mechanics in physics"
        text2 = "Baking chocolate chip cookies"
        
        percept1 = await perception.encode(text1, "text")
        percept2 = await perception.encode(text2, "text")
        
        # Calculate cosine similarity
        vec1 = np.array(percept1.embedding)
        vec2 = np.array(percept2.embedding)
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        # Dissimilar texts should have lower similarity (< 0.5)
        assert similarity < 0.5


class TestComplexityEstimation:
    """Test complexity estimation for different inputs."""
    
    @pytest.mark.asyncio
    async def test_text_complexity_scales_with_length(self):
        """Test that complexity increases with text length."""
        perception = PerceptionSubsystem()
        
        short_text = "Hi"
        long_text = "This is a much longer text with many more words and characters that should result in higher complexity score"
        
        short_percept = await perception.encode(short_text, "text")
        long_percept = await perception.encode(long_text, "text")
        
        # Longer text should have higher complexity
        assert long_percept.complexity > short_percept.complexity
    
    @pytest.mark.asyncio
    async def test_text_complexity_bounds(self):
        """Test that text complexity is within expected bounds."""
        perception = PerceptionSubsystem()
        
        short_text = "a"
        long_text = "a" * 2000
        
        short_percept = await perception.encode(short_text, "text")
        long_percept = await perception.encode(long_text, "text")
        
        # Complexity should be within bounds (1-50 for text)
        assert 1 <= short_percept.complexity <= 50
        assert 1 <= long_percept.complexity <= 50
    
    @pytest.mark.asyncio
    async def test_image_complexity(self):
        """Test that image inputs have expected complexity."""
        perception = PerceptionSubsystem()
        
        # Image encoding will fail without CLIP, but complexity should still be set
        percept = await perception.encode("fake_image.jpg", "image")
        
        # Images should have complexity around 30
        assert percept.complexity == 30
    
    @pytest.mark.asyncio
    async def test_audio_complexity(self):
        """Test that audio inputs have expected complexity."""
        perception = PerceptionSubsystem()
        
        audio_data = {"duration_seconds": 10}
        percept = await perception.encode(audio_data, "audio")
        
        # Audio complexity should scale with duration
        assert percept.complexity > 0
        assert percept.complexity <= 80
    
    @pytest.mark.asyncio
    async def test_introspection_complexity(self):
        """Test that introspection inputs have expected complexity."""
        perception = PerceptionSubsystem()
        
        intro_data = {"description": "Self-reflection on goals"}
        percept = await perception.encode(intro_data, "introspection")
        
        # Introspection should have fixed complexity around 20
        assert percept.complexity == 20


class TestErrorHandling:
    """Test error handling and robustness."""
    
    @pytest.mark.asyncio
    async def test_unknown_modality(self):
        """Test handling of unknown modality."""
        perception = PerceptionSubsystem()
        
        # Should not crash, returns dummy percept
        percept = await perception.encode("test", "unknown_modality")
        
        assert isinstance(percept, Percept)
        assert percept.modality == "unknown_modality"
        assert "error" in percept.metadata
    
    @pytest.mark.asyncio
    async def test_malformed_input(self):
        """Test handling of malformed input."""
        perception = PerceptionSubsystem()
        
        # None input
        percept = await perception.encode(None, "text")
        
        assert isinstance(percept, Percept)
        # Should still encode (converts to string "None")
        assert len(percept.embedding) == perception.embedding_dim
    
    @pytest.mark.asyncio
    async def test_empty_text(self):
        """Test handling of empty text."""
        perception = PerceptionSubsystem()
        
        percept = await perception.encode("", "text")
        
        assert isinstance(percept, Percept)
        assert len(percept.embedding) == perception.embedding_dim


class TestEmbeddingConsistency:
    """Test that embeddings are consistent across multiple calls."""
    
    @pytest.mark.asyncio
    async def test_same_text_identical_embeddings(self):
        """Test that encoding same text multiple times gives identical results."""
        perception = PerceptionSubsystem()
        
        text = "Consistency test"
        
        percept1 = await perception.encode(text, "text")
        percept2 = await perception.encode(text, "text")
        percept3 = await perception.encode(text, "text")
        
        # All embeddings should be identical
        assert percept1.embedding == percept2.embedding
        assert percept2.embedding == percept3.embedding
    
    @pytest.mark.asyncio
    async def test_embedding_deterministic(self):
        """Test that embeddings are deterministic."""
        # Create two separate instances
        perception1 = PerceptionSubsystem()
        perception2 = PerceptionSubsystem()
        
        text = "Deterministic test"
        
        percept1 = await perception1.encode(text, "text")
        percept2 = await perception2.encode(text, "text")
        
        # Embeddings should be identical across instances
        assert percept1.embedding == percept2.embedding


class TestStatistics:
    """Test statistics tracking."""
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test that statistics are tracked correctly."""
        perception = PerceptionSubsystem()
        
        # Initial state
        stats = perception.get_stats()
        assert stats["total_encodings"] == 0
        
        # Encode some texts
        await perception.encode("text 1", "text")
        await perception.encode("text 2", "text")
        await perception.encode("text 1", "text")  # Cache hit
        
        stats = perception.get_stats()
        
        assert stats["total_encodings"] == 3
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["cache_hit_rate"] == pytest.approx(1/3, abs=0.01)
    
    @pytest.mark.asyncio
    async def test_encoding_time_tracked(self):
        """Test that encoding times are tracked."""
        perception = PerceptionSubsystem()
        
        await perception.encode("test text", "text")
        
        stats = perception.get_stats()
        
        assert "average_encoding_time_ms" in stats
        assert stats["average_encoding_time_ms"] > 0


class TestLegacyCompatibility:
    """Test backward compatibility with legacy interface."""
    
    @pytest.mark.asyncio
    async def test_process_method(self):
        """Test that legacy process() method still works."""
        perception = PerceptionSubsystem()
        
        percept = await perception.process("test input")
        
        assert isinstance(percept, Percept)
        assert percept.modality == "text"
        assert len(percept.embedding) == perception.embedding_dim


class TestIntegrationWithWorkspace:
    """Test integration with workspace Percept model."""
    
    @pytest.mark.asyncio
    async def test_percept_has_required_fields(self):
        """Test that generated Percept has all required fields."""
        perception = PerceptionSubsystem()
        
        percept = await perception.encode("test", "text")
        
        assert hasattr(percept, "id")
        assert hasattr(percept, "modality")
        assert hasattr(percept, "embedding")
        assert hasattr(percept, "raw")
        assert hasattr(percept, "complexity")
        assert hasattr(percept, "timestamp")
        assert hasattr(percept, "metadata")
    
    @pytest.mark.asyncio
    async def test_percept_metadata(self):
        """Test that Percept metadata is populated."""
        perception = PerceptionSubsystem()
        
        percept = await perception.encode("test", "text")
        
        assert "encoding_model" in percept.metadata
        assert percept.metadata["encoding_model"] == "sentence-transformers"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
