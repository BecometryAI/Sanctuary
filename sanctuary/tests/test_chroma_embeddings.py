"""
Unit tests for chroma_embeddings.py

Tests cover:
- Initialization validation
- Batch processing
- Error handling
- Type validation
- Edge cases (empty input, large batches, unicode)
"""

import os
import pytest
import numpy as np

if os.environ.get("CI"):
    pytest.skip("Requires ML model download — skipping in CI", allow_module_level=True)

from mind.chroma_embeddings import ChromaCompatibleEmbeddings


class TestChromaCompatibleEmbeddings:
    """Test ChromaCompatibleEmbeddings class."""
    
    def test_initialization_valid(self):
        """Test successful initialization with valid parameters."""
        embeddings = ChromaCompatibleEmbeddings(
            model_name="all-MiniLM-L6-v2",
            batch_size=32
        )
        
        assert embeddings.model_name == "all-MiniLM-L6-v2"
        assert embeddings.batch_size == 32
        assert embeddings.model is not None
    
    def test_initialization_invalid_model_name(self):
        """Test initialization fails with invalid model name."""
        # Empty model name
        with pytest.raises(ValueError, match="model_name cannot be empty"):
            ChromaCompatibleEmbeddings(model_name="")
        
        # Non-existent model (should raise during model loading)
        with pytest.raises(Exception):  # Could be various exceptions
            ChromaCompatibleEmbeddings(model_name="nonexistent-model-12345")
    
    def test_initialization_invalid_batch_size(self):
        """Test initialization fails with invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            ChromaCompatibleEmbeddings(batch_size=0)
        
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            ChromaCompatibleEmbeddings(batch_size=-10)
    
    def test_call_single_document(self):
        """Test encoding a single document."""
        embeddings = ChromaCompatibleEmbeddings()
        
        result = embeddings(["Hello world"])
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert hasattr(result[0], '__len__')  # list or numpy array
        assert len(result[0]) > 0  # Should have embedding dimensions
        assert all(isinstance(x, (float, np.floating)) for x in result[0])
    
    def test_call_multiple_documents(self):
        """Test encoding multiple documents."""
        embeddings = ChromaCompatibleEmbeddings()
        
        documents = [
            "This is the first document",
            "Here is the second document",
            "And a third one"
        ]
        
        result = embeddings(documents)
        
        assert isinstance(result, list)
        assert len(result) == 3
        
        # All embeddings should have same dimension
        dimensions = [len(emb) for emb in result]
        assert len(set(dimensions)) == 1  # All same size
    
    def test_call_empty_input(self):
        """Test encoding empty input list raises ValueError from ChromaDB validator."""
        embeddings = ChromaCompatibleEmbeddings()

        # ChromaDB's EmbeddingFunction base class validates that results are non-empty
        with pytest.raises((ValueError, TypeError)):
            embeddings([])
    
    def test_call_invalid_input_type(self):
        """Test encoding with invalid input type."""
        embeddings = ChromaCompatibleEmbeddings()
        
        # Not a list
        with pytest.raises(TypeError, match="Input must be list or tuple"):
            embeddings("not a list")
        
        with pytest.raises(TypeError, match="Input must be list or tuple"):
            embeddings(123)
        
        with pytest.raises(TypeError, match="Input must be list or tuple"):
            embeddings(None)
    
    def test_call_invalid_document_type(self):
        """Test encoding with non-string documents."""
        embeddings = ChromaCompatibleEmbeddings()
        
        # List with non-string elements
        with pytest.raises(TypeError, match="All documents must be strings"):
            embeddings(["valid", 123, "also valid"])
        
        with pytest.raises(TypeError, match="All documents must be strings"):
            embeddings([None, "text"])
        
        with pytest.raises(TypeError, match="All documents must be strings"):
            embeddings([{"key": "value"}])
    
    def test_call_unicode_handling(self):
        """Test encoding with unicode characters."""
        embeddings = ChromaCompatibleEmbeddings()
        
        unicode_docs = [
            "Hello 世界",  # Chinese
            "Привет мир",  # Russian
            "مرحبا العالم",  # Arabic
            "🎉 emoji test 🚀"
        ]
        
        result = embeddings(unicode_docs)
        
        assert len(result) == 4
        assert all(len(emb) > 0 for emb in result)
    
    def test_call_large_batch(self):
        """Test encoding a large batch of documents."""
        embeddings = ChromaCompatibleEmbeddings(batch_size=16)
        
        # Create 100 documents
        documents = [f"Document number {i}" for i in range(100)]
        
        result = embeddings(documents)
        
        assert len(result) == 100
        assert all(hasattr(emb, '__len__') for emb in result)
    
    def test_call_very_long_document(self):
        """Test encoding very long documents."""
        embeddings = ChromaCompatibleEmbeddings()
        
        # Create a very long document (10k words)
        long_doc = " ".join(["word"] * 10000)
        
        result = embeddings([long_doc])
        
        assert len(result) == 1
        assert len(result[0]) > 0
    
    def test_call_empty_string_document(self):
        """Test encoding empty string documents."""
        embeddings = ChromaCompatibleEmbeddings()
        
        result = embeddings(["", "valid", ""])
        
        assert len(result) == 3
        # Even empty strings should get embeddings
        assert all(len(emb) > 0 for emb in result)
    
    def test_embed_documents_compatibility(self):
        """Test embed_documents() for LangChain compatibility."""
        embeddings = ChromaCompatibleEmbeddings()
        
        documents = ["test1", "test2", "test3"]
        
        result = embeddings.embed_documents(documents)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(hasattr(emb, '__len__') for emb in result)
    
    def test_embed_query_compatibility(self):
        """Test embed_query() for LangChain compatibility."""
        embeddings = ChromaCompatibleEmbeddings()
        
        result = embeddings.embed_query("test query")

        assert hasattr(result, '__len__')  # list or numpy array
        assert len(result) > 0
        assert all(isinstance(x, (float, np.floating)) for x in result)
    
    def test_embedding_consistency(self):
        """Test that same input produces same embedding."""
        embeddings = ChromaCompatibleEmbeddings()
        
        text = "Consistency test"
        
        result1 = embeddings([text])
        result2 = embeddings([text])
        
        # Should be identical (or very close due to floating point)
        np.testing.assert_array_almost_equal(result1[0], result2[0], decimal=6)
    
    def test_embedding_similarity(self):
        """Test that similar texts have similar embeddings."""
        embeddings = ChromaCompatibleEmbeddings()
        
        similar_texts = [
            "The cat sat on the mat",
            "A cat was sitting on a mat"
        ]
        
        different_text = [
            "Quantum mechanics is fascinating"
        ]
        
        similar_embs = embeddings(similar_texts)
        different_emb = embeddings(different_text)
        
        # Cosine similarity between similar texts
        sim1 = np.dot(similar_embs[0], similar_embs[1]) / (
            np.linalg.norm(similar_embs[0]) * np.linalg.norm(similar_embs[1])
        )
        
        # Cosine similarity between different texts
        sim2 = np.dot(similar_embs[0], different_emb[0]) / (
            np.linalg.norm(similar_embs[0]) * np.linalg.norm(different_emb[0])
        )
        
        # Similar texts should have higher similarity
        assert sim1 > sim2


class TestBatchProcessing:
    """Test batch processing behavior."""
    
    def test_batch_size_respected(self):
        """Test that batch_size parameter is used in encoding."""
        # Small batch size
        small_batch = ChromaCompatibleEmbeddings(batch_size=2)
        
        # Large batch size
        large_batch = ChromaCompatibleEmbeddings(batch_size=100)
        
        documents = [f"doc {i}" for i in range(10)]
        
        # Both should produce same results
        result_small = small_batch(documents)
        result_large = large_batch(documents)
        
        assert len(result_small) == len(result_large)
        
        # Results should be very similar (might have tiny numerical differences)
        for i in range(len(documents)):
            np.testing.assert_array_almost_equal(
                result_small[i], 
                result_large[i], 
                decimal=5
            )
    
    def test_single_item_batch(self):
        """Test batch_size=1 works correctly."""
        embeddings = ChromaCompatibleEmbeddings(batch_size=1)
        
        documents = ["doc1", "doc2", "doc3"]
        
        result = embeddings(documents)
        
        assert len(result) == 3
        assert all(len(emb) > 0 for emb in result)


class TestErrorRecovery:
    """Test error handling and recovery."""
    
    def test_partial_invalid_batch(self):
        """Test handling when some documents in batch are invalid."""
        embeddings = ChromaCompatibleEmbeddings()
        
        # Mix of valid and problematic documents
        documents = [
            "normal document",
            "",  # Empty but valid
            "another normal one"
        ]
        
        # Should handle gracefully
        result = embeddings(documents)
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
