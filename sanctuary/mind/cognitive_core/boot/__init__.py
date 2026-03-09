"""
Boot module for Phase 1 integration testing.

Provides a minimal CognitiveCore that can instantiate and run
the cognitive loop without heavy ML dependencies.
"""

from .boot_core import BootCognitiveCore
from .boot_coordinator import BootCoordinator

__all__ = ["BootCognitiveCore", "BootCoordinator"]
