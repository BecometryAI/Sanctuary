"""
Percept Similarity Detection: Deduplicate near-identical percepts.

Uses cosine similarity on percept embeddings to detect and filter out
duplicate or near-duplicate percepts before they consume attention budget.

This prevents the workspace from being flooded with redundant information
when, for example, a sensor sends repeated readings or the same text
input is submitted multiple times.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PerceptSimilarityDetector:
    """
    Detects and filters near-duplicate percepts using embedding similarity.

    Given a list of new percepts, compares each against recent percepts
    already in the workspace and against other new percepts. Percepts
    whose cosine similarity exceeds the threshold are marked as duplicates
    and can be filtered out before attention processing.

    Attributes:
        similarity_threshold: Cosine similarity above which two percepts
            are considered duplicates (default: 0.92)
        cross_modal_threshold: Threshold for cross-modal comparison
            (default: 0.95, higher because cross-modal similarity is noisier)
        max_history: Number of recent percept embeddings to retain for
            comparison (default: 50)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the similarity detector.

        Args:
            config: Optional configuration dict with keys:
                - similarity_threshold: float (default: 0.92)
                - cross_modal_threshold: float (default: 0.95)
                - max_history: int (default: 50)
        """
        config = config or {}
        self.similarity_threshold = config.get("similarity_threshold", 0.92)
        self.cross_modal_threshold = config.get("cross_modal_threshold", 0.95)
        self.max_history = config.get("max_history", 50)

        # Ring buffer of recent percept embeddings for comparison
        self._recent_embeddings: List[Dict[str, Any]] = []

        # Statistics
        self._stats = {
            "total_checked": 0,
            "duplicates_detected": 0,
            "duplicates_by_modality": {},
        }

        logger.info(
            f"PerceptSimilarityDetector initialized "
            f"(threshold={self.similarity_threshold}, "
            f"cross_modal={self.cross_modal_threshold}, "
            f"history={self.max_history})"
        )

    def filter_duplicates(
        self,
        new_percepts: list,
        workspace_percepts: Optional[dict] = None,
    ) -> list:
        """
        Filter out near-duplicate percepts.

        Compares each new percept against:
        1. Other new percepts in the same batch (intra-batch dedup)
        2. Recent percept history (temporal dedup)
        3. Current workspace percepts (workspace dedup)

        Args:
            new_percepts: List of Percept objects from the current cycle
            workspace_percepts: Optional dict of {id: Percept} currently in workspace

        Returns:
            Filtered list with near-duplicates removed
        """
        if not new_percepts:
            return []

        kept = []
        kept_entries = []  # list of (embedding, modality) for intra-batch comparison

        for percept in new_percepts:
            self._stats["total_checked"] += 1

            embedding = self._get_embedding(percept)
            if embedding is None:
                # No embedding available — keep the percept
                kept.append(percept)
                continue

            modality = getattr(percept, "modality", "unknown")
            is_duplicate = False

            # 1. Check against other percepts in this batch
            for prev_emb, prev_mod in kept_entries:
                threshold = (
                    self.similarity_threshold
                    if prev_mod == modality
                    else self.cross_modal_threshold
                )
                sim = self._cosine_similarity(embedding, prev_emb)
                if sim >= threshold:
                    is_duplicate = True
                    break

            # 2. Check against recent history
            if not is_duplicate:
                for entry in self._recent_embeddings:
                    threshold = (
                        self.similarity_threshold
                        if entry["modality"] == modality
                        else self.cross_modal_threshold
                    )
                    sim = self._cosine_similarity(embedding, entry["embedding"])
                    if sim >= threshold:
                        is_duplicate = True
                        break

            # 3. Check against workspace percepts
            if not is_duplicate and workspace_percepts:
                for wp in workspace_percepts.values():
                    wp_emb = self._get_embedding(wp)
                    if wp_emb is not None:
                        threshold = (
                            self.similarity_threshold
                            if getattr(wp, "modality", "") == modality
                            else self.cross_modal_threshold
                        )
                        sim = self._cosine_similarity(embedding, wp_emb)
                        if sim >= threshold:
                            is_duplicate = True
                            break

            if is_duplicate:
                self._stats["duplicates_detected"] += 1
                mod_key = str(modality)
                self._stats["duplicates_by_modality"][mod_key] = (
                    self._stats["duplicates_by_modality"].get(mod_key, 0) + 1
                )
                logger.debug(
                    f"Filtered duplicate {modality} percept "
                    f"(total filtered: {self._stats['duplicates_detected']})"
                )
            else:
                kept.append(percept)
                kept_entries.append((embedding, str(modality)))

                # Add to recent history
                self._recent_embeddings.append({
                    "embedding": embedding,
                    "modality": str(modality),
                })
                if len(self._recent_embeddings) > self.max_history:
                    self._recent_embeddings.pop(0)

        return kept

    @staticmethod
    def _get_embedding(percept) -> Optional[np.ndarray]:
        """Extract embedding from a percept, returning None if unavailable."""
        embedding = getattr(percept, "embedding", None)
        if embedding is None:
            return None

        if isinstance(embedding, np.ndarray):
            return embedding
        if isinstance(embedding, list):
            arr = np.array(embedding, dtype=np.float32)
            # Skip zero embeddings (error/placeholder)
            if np.allclose(arr, 0.0):
                return None
            return arr
        return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if a.shape != b.shape:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_stats(self) -> Dict[str, Any]:
        """Return similarity detection statistics."""
        total = self._stats["total_checked"]
        return {
            "total_checked": total,
            "duplicates_detected": self._stats["duplicates_detected"],
            "duplicate_rate": (
                self._stats["duplicates_detected"] / total if total > 0 else 0.0
            ),
            "duplicates_by_modality": dict(self._stats["duplicates_by_modality"]),
            "history_size": len(self._recent_embeddings),
            "similarity_threshold": self.similarity_threshold,
        }

    def clear_history(self) -> None:
        """Clear the recent embedding history."""
        self._recent_embeddings.clear()
        logger.debug("Percept similarity history cleared")
