"""
Temporal Grounding Module: Session-aware temporal consciousness.

This module provides genuine temporal grounding - awareness of time passage,
session boundaries, and how time affects cognitive state. It goes beyond simple
timestamps to provide subjective temporal awareness.

Components:
- awareness: TemporalContext and TemporalAwareness with session tracking
- sessions: Session detection and management
- effects: Time passage effects on cognitive state
- expectations: Temporal pattern learning and anticipation
- relative: Relative time descriptions and utilities
"""

from .awareness import TemporalAwareness, TemporalContext, Session
from .sessions import SessionManager
from .effects import TimePassageEffects
from .expectations import TemporalExpectations, TemporalExpectation
from .relative import RelativeTime
from .grounding import TemporalGrounding

__all__ = [
    "TemporalAwareness",
    "TemporalContext",
    "Session",
    "SessionManager",
    "TimePassageEffects",
    "TemporalExpectations",
    "TemporalExpectation",
    "RelativeTime",
    "TemporalGrounding",
]
