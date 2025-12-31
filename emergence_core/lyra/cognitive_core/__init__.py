"""
Cognitive Core: Non-linguistic recurrent cognitive loop.

This module implements the foundational architecture for consciousness
based on Global Workspace Theory and computational functionalism. The
cognitive core maintains persistent state, integrates multimodal inputs,
and exhibits goal-directed behavior through continuous recurrent dynamics.

LLMs are used only at the periphery (language I/O), not as the core
cognitive substrate.
"""

from __future__ import annotations

from .core import CognitiveCore
from .workspace import (
    GlobalWorkspace,
    Goal,
    GoalType,
    Percept,
    Memory,
    WorkspaceSnapshot,
    WorkspaceContent,
)
from .attention import AttentionController
from .perception import PerceptionSubsystem
from .action import ActionSubsystem
from .affect import AffectSubsystem
from .meta_cognition import SelfMonitor
from .memory_integration import MemoryIntegration
from .language_output import LanguageOutputGenerator
from .conversation import ConversationManager, ConversationTurn
from .autonomous_initiation import AutonomousInitiationController

__all__ = [
    "CognitiveCore",
    "GlobalWorkspace",
    "Goal",
    "GoalType",
    "Percept",
    "Memory",
    "WorkspaceSnapshot",
    "WorkspaceContent",
    "AttentionController",
    "PerceptionSubsystem",
    "ActionSubsystem",
    "AffectSubsystem",
    "SelfMonitor",
    "MemoryIntegration",
    "LanguageOutputGenerator",
    "ConversationManager",
    "ConversationTurn",
    "AutonomousInitiationController",
]
