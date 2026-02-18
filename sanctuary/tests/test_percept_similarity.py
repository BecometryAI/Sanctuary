"""
Tests for percept similarity detection.

Validates that the PerceptSimilarityDetector correctly identifies and
filters near-duplicate percepts using embedding cosine similarity.
"""

import pytest
import numpy as np

from mind.cognitive_core.percept_similarity import PerceptSimilarityDetector
from mind.cognitive_core.workspace import Percept


def _make_percept(embedding, modality="text", raw="test"):
    """Helper to create a Percept with a specific embedding."""
    return Percept(
        modality=modality,
        raw=raw,
        embedding=embedding,
        complexity=5,
    )


class TestPerceptSimilarityDetectorInitialization:
    """Test PerceptSimilarityDetector initialization."""

    def test_default_config(self):
        detector = PerceptSimilarityDetector()
        assert detector.similarity_threshold == 0.92
        assert detector.cross_modal_threshold == 0.95
        assert detector.max_history == 50

    def test_custom_config(self):
        detector = PerceptSimilarityDetector({
            "similarity_threshold": 0.85,
            "cross_modal_threshold": 0.90,
            "max_history": 100,
        })
        assert detector.similarity_threshold == 0.85
        assert detector.cross_modal_threshold == 0.90
        assert detector.max_history == 100

    def test_initial_stats(self):
        detector = PerceptSimilarityDetector()
        stats = detector.get_stats()
        assert stats["total_checked"] == 0
        assert stats["duplicates_detected"] == 0
        assert stats["duplicate_rate"] == 0.0


class TestFilterDuplicates:
    """Test duplicate filtering logic."""

    def test_empty_input_returns_empty(self):
        detector = PerceptSimilarityDetector()
        result = detector.filter_duplicates([])
        assert result == []

    def test_single_percept_passes_through(self):
        detector = PerceptSimilarityDetector()
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        percept = _make_percept(emb.tolist())
        result = detector.filter_duplicates([percept])
        assert len(result) == 1
        assert result[0] is percept

    def test_identical_embeddings_are_deduplicated(self):
        detector = PerceptSimilarityDetector({"similarity_threshold": 0.90})
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        emb_list = emb.tolist()

        p1 = _make_percept(emb_list, raw="first")
        p2 = _make_percept(emb_list, raw="duplicate")

        result = detector.filter_duplicates([p1, p2])
        assert len(result) == 1
        assert result[0] is p1

    def test_different_embeddings_both_kept(self):
        detector = PerceptSimilarityDetector()
        rng = np.random.RandomState(42)

        emb1 = rng.randn(384).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 = rng.randn(384).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)

        p1 = _make_percept(emb1.tolist(), raw="first")
        p2 = _make_percept(emb2.tolist(), raw="second")

        result = detector.filter_duplicates([p1, p2])
        assert len(result) == 2

    def test_temporal_dedup_across_calls(self):
        """Percepts similar to those from previous cycles are filtered."""
        detector = PerceptSimilarityDetector({"similarity_threshold": 0.90})
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        emb_list = emb.tolist()

        p1 = _make_percept(emb_list, raw="cycle 1")
        result1 = detector.filter_duplicates([p1])
        assert len(result1) == 1

        p2 = _make_percept(emb_list, raw="cycle 2")
        result2 = detector.filter_duplicates([p2])
        assert len(result2) == 0  # Filtered as duplicate of p1

    def test_workspace_dedup(self):
        """Percepts similar to those already in workspace are filtered."""
        detector = PerceptSimilarityDetector({"similarity_threshold": 0.90})
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        emb_list = emb.tolist()

        workspace_percept = _make_percept(emb_list, raw="in workspace")
        workspace_percepts = {"wp1": workspace_percept}

        new_percept = _make_percept(emb_list, raw="new input")
        result = detector.filter_duplicates([new_percept], workspace_percepts)
        assert len(result) == 0

    def test_cross_modal_uses_higher_threshold(self):
        """Cross-modal comparisons use the stricter threshold."""
        rng = np.random.RandomState(123)
        emb = rng.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)

        # Add perturbation so cosine sim is between 0.90 and 0.98
        noise = rng.randn(384).astype(np.float32) * 0.015
        emb2 = emb + noise
        emb2 /= np.linalg.norm(emb2)

        actual_sim = float(np.dot(emb, emb2))
        assert actual_sim > 0.90, f"Test setup: similarity {actual_sim:.3f} too low"
        assert actual_sim < 0.99, f"Test setup: similarity {actual_sim:.3f} too high"

        # Same modality with threshold 0.80: should be filtered
        detector1 = PerceptSimilarityDetector({
            "similarity_threshold": 0.80,
            "cross_modal_threshold": 0.99,
        })
        p1 = _make_percept(emb.tolist(), modality="text", raw="a")
        p2 = _make_percept(emb2.tolist(), modality="text", raw="b")
        result = detector1.filter_duplicates([p1, p2])
        assert len(result) == 1

        # Different modality with cross_modal_threshold=0.99: stricter, both kept
        detector2 = PerceptSimilarityDetector({
            "similarity_threshold": 0.80,
            "cross_modal_threshold": 0.99,
        })
        p3 = _make_percept(emb.tolist(), modality="text", raw="a")
        p4 = _make_percept(emb2.tolist(), modality="audio", raw="b")
        result2 = detector2.filter_duplicates([p3, p4])
        assert len(result2) == 2  # Different modality, cross_modal not met

    def test_zero_embedding_is_not_compared(self):
        """Percepts with zero embeddings (errors) should pass through."""
        detector = PerceptSimilarityDetector()
        zero_emb = [0.0] * 384
        p1 = _make_percept(zero_emb, raw="error percept 1")
        p2 = _make_percept(zero_emb, raw="error percept 2")
        result = detector.filter_duplicates([p1, p2])
        # Both should pass through since zero embeddings are skipped
        assert len(result) == 2

    def test_no_embedding_percept_passes_through(self):
        """Percepts without an embedding attribute should pass through."""
        detector = PerceptSimilarityDetector()
        p = Percept(modality="text", raw="hello", embedding=None, complexity=5)
        result = detector.filter_duplicates([p])
        assert len(result) == 1


class TestPerceptSimilarityStats:
    """Test statistics tracking."""

    def test_stats_track_filtering(self):
        detector = PerceptSimilarityDetector({"similarity_threshold": 0.90})
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        emb_list = emb.tolist()

        p1 = _make_percept(emb_list, raw="first")
        p2 = _make_percept(emb_list, raw="duplicate")

        detector.filter_duplicates([p1, p2])
        stats = detector.get_stats()

        assert stats["total_checked"] == 2
        assert stats["duplicates_detected"] == 1
        assert stats["duplicate_rate"] == 0.5
        assert stats["duplicates_by_modality"]["text"] == 1

    def test_clear_history(self):
        detector = PerceptSimilarityDetector()
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb)
        p = _make_percept(emb.tolist())
        detector.filter_duplicates([p])

        assert detector.get_stats()["history_size"] == 1
        detector.clear_history()
        assert detector.get_stats()["history_size"] == 0

    def test_history_eviction_at_max(self):
        detector = PerceptSimilarityDetector({"max_history": 3})
        rng = np.random.RandomState(42)

        for i in range(5):
            emb = rng.randn(384).astype(np.float32)
            emb /= np.linalg.norm(emb)
            p = _make_percept(emb.tolist(), raw=f"percept_{i}")
            detector.filter_duplicates([p])

        assert detector.get_stats()["history_size"] == 3


class TestCosineSimilarity:
    """Test the static cosine similarity helper."""

    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = PerceptSimilarityDetector._cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = PerceptSimilarityDetector._cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        sim = PerceptSimilarityDetector._cosine_similarity(a, b)
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector_returns_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        sim = PerceptSimilarityDetector._cosine_similarity(a, b)
        assert sim == 0.0

    def test_mismatched_shapes_return_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sim = PerceptSimilarityDetector._cosine_similarity(a, b)
        assert sim == 0.0
