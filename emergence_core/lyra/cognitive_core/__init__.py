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
from .action import ActionSubsystem, Action, ActionType
from .affect import AffectSubsystem
from .meta_cognition import SelfMonitor, IntrospectiveJournal
from .memory_integration import MemoryIntegration
from .language_input import LanguageInputParser, IntentType, Intent, ParseResult
from .language_output import LanguageOutputGenerator
from .llm_client import LLMClient, GemmaClient, LlamaClient, MockLLMClient, LLMError
from .structured_formats import (
    LLMInputParseRequest,
    LLMInputParseResponse,
    OutputGenerationRequest,
    OutputGenerationResponse,
    ConversationContext,
    EmotionalState,
    WorkspaceStateSnapshot
)
from .fallback_handlers import (
    FallbackInputParser,
    FallbackOutputGenerator,
    CircuitBreaker,
    CircuitState
)
from .conversation import ConversationManager, ConversationTurn
from .autonomous_initiation import AutonomousInitiationController
from .temporal_awareness import TemporalAwareness
from .autonomous_memory_review import AutonomousMemoryReview
from .existential_reflection import ExistentialReflection
from .interaction_patterns import InteractionPatternAnalysis
from .continuous_consciousness import ContinuousConsciousnessController
from .introspective_loop import IntrospectiveLoop, ActiveReflection, ReflectionTrigger
from .consciousness_tests import (
    ConsciousnessTest,
    TestResult,
    MirrorTest,
    UnexpectedSituationTest,
    SpontaneousReflectionTest,
    CounterfactualReasoningTest,
    MetaCognitiveAccuracyTest,
    ConsciousnessTestFramework,
    ConsciousnessReportGenerator
)

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
    "Action",
    "ActionType",
    "AffectSubsystem",
    "SelfMonitor",
    "IntrospectiveJournal",
    "MemoryIntegration",
    "LanguageInputParser",
    "IntentType",
    "Intent",
    "ParseResult",
    "LanguageOutputGenerator",
    "LLMClient",
    "GemmaClient",
    "LlamaClient",
    "MockLLMClient",
    "LLMError",
    "LLMInputParseRequest",
    "LLMInputParseResponse",
    "OutputGenerationRequest",
    "OutputGenerationResponse",
    "ConversationContext",
    "EmotionalState",
    "WorkspaceStateSnapshot",
    "FallbackInputParser",
    "FallbackOutputGenerator",
    "CircuitBreaker",
    "CircuitState",
    "ConversationManager",
    "ConversationTurn",
    "AutonomousInitiationController",
    "TemporalAwareness",
    "AutonomousMemoryReview",
    "ExistentialReflection",
    "InteractionPatternAnalysis",
    "ContinuousConsciousnessController",
    "IntrospectiveLoop",
    "ActiveReflection",
    "ReflectionTrigger",
    "ConsciousnessTest",
    "TestResult",
    "MirrorTest",
    "UnexpectedSituationTest",
    "SpontaneousReflectionTest",
    "CounterfactualReasoningTest",
    "MetaCognitiveAccuracyTest",
    "ConsciousnessTestFramework",
    "ConsciousnessReportGenerator",
]
