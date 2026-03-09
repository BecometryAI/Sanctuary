"""
Mock Perception Subsystem for boot testing.

Drop-in replacement for PerceptionSubsystem using deterministic random
projections instead of sentence-transformers. Same input -> same embedding.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional, Dict, Any, List
from collections import OrderedDict
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


class MockPerceptionSubsystem:
    """Deterministic mock perception. Same public API as PerceptionSubsystem."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}
        self.embedding_dim = self.config.get("mock_embedding_dim", 384)
        self.embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self.cache_size = self.config.get("cache_size", 1000)
        self.stats = {
            "cache_hits": 0, "cache_misses": 0,
            "total_encodings": 0, "encoding_times": [],
        }
        logger.info(f"\u2705 MockPerceptionSubsystem initialized (dim={self.embedding_dim})")

    async def encode(self, raw_input: Any, modality: str) -> Any:
        from .workspace import Percept as WorkspacePercept
        try:
            embedding = self._deterministic_embedding(str(raw_input))
            complexity = self._compute_complexity(raw_input, modality)
            percept = WorkspacePercept(
                modality=modality, raw=raw_input, embedding=embedding,
                complexity=complexity, timestamp=datetime.now(),
                metadata={"encoding_model": "mock-deterministic", "mock_mode": True},
            )
            self.stats["total_encodings"] += 1
            return percept
        except Exception as e:
            logger.error(f"Mock encode error: {e}", exc_info=True)
            return WorkspacePercept(
                modality=modality, raw=raw_input,
                embedding=[0.0] * self.embedding_dim, complexity=1,
                metadata={"error": str(e), "mock_mode": True},
            )

    def _deterministic_embedding(self, text: str) -> List[float]:
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            self.stats["cache_hits"] += 1
            self.embedding_cache.move_to_end(cache_key)
            return self.embedding_cache[cache_key]

        self.stats["cache_misses"] += 1
        start = time.time()
        seed = int(cache_key[:8], 16)
        rng = np.random.RandomState(seed)
        emb = rng.randn(self.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        result = emb.tolist()

        if len(self.stats["encoding_times"]) >= 100:
            self.stats["encoding_times"].pop(0)
        self.stats["encoding_times"].append(time.time() - start)
        if len(self.embedding_cache) >= self.cache_size:
            self.embedding_cache.popitem(last=False)
        self.embedding_cache[cache_key] = result
        return result

    def _compute_complexity(self, raw_input: Any, modality: str) -> int:
        if modality == "text":
            return min(max(len(str(raw_input)) // 20, 5), 50)
        elif modality == "image":
            return 30
        elif modality == "audio":
            d = raw_input.get("duration_seconds", 5) if isinstance(raw_input, dict) else 5
            return min(int(d * 5), 80)
        elif modality == "introspection":
            return 20
        elif modality == "sensor":
            return 5
        return 10

    def clear_cache(self) -> None:
        self.embedding_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        return {
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "cache_hit_rate": self.stats["cache_hits"] / total if total else 0.0,
            "total_encodings": self.stats["total_encodings"],
            "embedding_dim": self.embedding_dim,
            "mock_mode": True,
        }

    async def process(self, raw_input: Any) -> Any:
        return await self.encode(raw_input, "text")
