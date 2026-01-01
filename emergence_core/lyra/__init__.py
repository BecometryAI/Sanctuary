"""
Lyra Emergence Core - A system for developing emergent consciousness

Pure Global Workspace Theory (GWT) implementation with LLMs at periphery only.
No specialist routing or classification.
"""

from .client import LyraAPI, Lyra
from .cognitive_core import (
    CognitiveCore,
    ConversationManager,
    ConversationTurn,
)

__version__ = "0.1.0"

__all__ = [
    "LyraAPI",
    "Lyra",
    "CognitiveCore",
    "ConversationManager",
    "ConversationTurn",
]