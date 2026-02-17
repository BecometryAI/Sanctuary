"""
Identity module: Computed identity system.

This module provides functionality for computing identity from system state
rather than loading from configuration files. Identity emerges from memories,
behavioral patterns, goal structures, and emotional tendencies.
"""

from .computed import ComputedIdentity, Identity
from .continuity import IdentitySnapshot, IdentityContinuity, IdentityEvolutionEvent
from .manager import IdentityManager
from .behavior_logger import BehaviorLogger

__all__ = [
    'ComputedIdentity',
    'Identity',
    'IdentitySnapshot',
    'IdentityContinuity',
    'IdentityEvolutionEvent',
    'IdentityManager',
    'BehaviorLogger'
]
