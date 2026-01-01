"""
Lyra Emergence Core - A system for developing emergent consciousness
"""

from .client import LyraAPI, Lyra
from .cognitive_core import (
    CognitiveCore,
    ConversationManager,
    ConversationTurn,
)

# Lazy imports to avoid loading heavy dependencies immediately
def __getattr__(name):
    """Lazy loading of heavy modules."""
    if name == "UnifiedCognitiveCore":
        from .unified_core import UnifiedCognitiveCore
        return UnifiedCognitiveCore
    elif name == "SharedMemoryBridge":
        from .unified_core import SharedMemoryBridge
        return SharedMemoryBridge
    elif name == "EmotionalStateBridge":
        from .unified_core import EmotionalStateBridge
        return EmotionalStateBridge
    elif name == "SpecialistFactory":
        from .specialists import SpecialistFactory
        return SpecialistFactory
    elif name == "SpecialistOutput":
        from .specialists import SpecialistOutput
        return SpecialistOutput
    elif name == "PhilosopherSpecialist":
        from .specialists import PhilosopherSpecialist
        return PhilosopherSpecialist
    elif name == "PragmatistSpecialist":
        from .specialists import PragmatistSpecialist
        return PragmatistSpecialist
    elif name == "ArtistSpecialist":
        from .specialists import ArtistSpecialist
        return ArtistSpecialist
    elif name == "VoiceSpecialist":
        from .specialists import VoiceSpecialist
        return VoiceSpecialist
    elif name == "PerceptionSpecialist":
        from .specialists import PerceptionSpecialist
        return PerceptionSpecialist
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__version__ = "0.1.0"

__all__ = [
    "LyraAPI",
    "Lyra",
    "CognitiveCore",
    "ConversationManager",
    "ConversationTurn",
]