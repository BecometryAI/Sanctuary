"""Performance subsystems for Phase 6.5.

Profile-driven performance optimization infrastructure:
- Profiler: Profile cognitive loop under load, identify bottlenecks
- Adaptive cycle rate: Auto-adjust cognitive loop speed based on load
- Lazy embeddings: Only compute embeddings when needed
- Async subsystem processor: Run subsystems in parallel
"""

from sanctuary.performance.profiler import CognitiveProfiler
from sanctuary.performance.adaptive_rate import AdaptiveCycleRate
from sanctuary.performance.lazy_embeddings import LazyEmbeddingCache
from sanctuary.performance.async_processor import AsyncSubsystemProcessor

__all__ = [
    "CognitiveProfiler",
    "AdaptiveCycleRate",
    "LazyEmbeddingCache",
    "AsyncSubsystemProcessor",
]
