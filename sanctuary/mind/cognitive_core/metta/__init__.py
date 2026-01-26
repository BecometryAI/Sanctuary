"""
MeTTa/Hyperon Integration (Pilot).

This module provides a bridge between Python cognitive core and MeTTa/Atomspace.
Currently a stub implementation with feature flag for gradual adoption.
"""

from .atomspace_bridge import AtomspaceBridge
from .metta_rules import COMMUNICATION_DECISION_RULES, PREDICTION_RULES

__all__ = [
    "AtomspaceBridge",
    "COMMUNICATION_DECISION_RULES",
    "PREDICTION_RULES",
]
