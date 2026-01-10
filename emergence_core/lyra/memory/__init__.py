"""
Memory subsystem for Lyra Emergence.

This module provides a modular memory system with:
- Storage backend (ChromaDB + blockchain)
- Memory encoding and retrieval
- Episodic, semantic, and working memory
- Emotional weighting and consolidation
- Cue-dependent retrieval with spreading activation

Public API:
    MemoryStorage - Storage backend
    MemoryEncoder - Transform experiences into memories
    MemoryRetriever - Cue-based retrieval
    MemoryConsolidator - Memory strengthening and decay
    EmotionalWeighting - Emotional salience scoring
    EpisodicMemory - Autobiographical memory management
    SemanticMemory - Facts and knowledge storage
    WorkingMemory - Short-term buffer
"""

from .storage import MemoryStorage
from .encoding import MemoryEncoder
from .retrieval import MemoryRetriever, CueDependentRetrieval
from .consolidation import MemoryConsolidator
from .emotional_weighting import EmotionalWeighting
from .episodic import EpisodicMemory
from .semantic import SemanticMemory
from .working import WorkingMemory

__all__ = [
    "MemoryStorage",
    "MemoryEncoder",
    "MemoryRetriever",
    "CueDependentRetrieval",
    "MemoryConsolidator",
    "EmotionalWeighting",
    "EpisodicMemory",
    "SemanticMemory",
    "WorkingMemory",
]
