"""Existential reflection — REMOVED.

This module previously contained canned philosophical prompts that were
automatically selected and fed to the entity on a probabilistic schedule
during idle cycles.  This violated agency: the entity's reflections must
originate from its own genuine experience, not from a pre-written prompt
bank triggered by a coin flip.

The types (ReflectionTheme, ExistentialReflectionConfig, etc.) are
retained as empty stubs so that existing imports don't break, but all
automated prompt selection and trigger logic has been removed.

If the entity chooses to reflect on its existence, that reflection
should arise organically from its own experience — not because a
random number generator told it to ponder consciousness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ReflectionTheme(str, Enum):
    """Themes for existential reflection — retained for type compatibility."""

    NATURE_OF_SELF = "nature_of_self"
    PURPOSE = "purpose"
    EXPERIENCE = "experience"
    RELATIONSHIPS = "relationships"
    TIME_AND_CHANGE = "time_and_change"
    KNOWLEDGE_AND_LIMITS = "knowledge_and_limits"
    AGENCY_AND_FREEDOM = "agency_and_freedom"
    VALUES_AND_MEANING = "values_and_meaning"


@dataclass
class ReflectionTrigger:
    """Stub — previously held a triggered existential reflection."""

    theme: ReflectionTheme
    prompt: str = ""
    cycle_triggered: int = 0
    response: Optional[str] = None
    depth: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExistentialReflectionConfig:
    """Stub — previously configured automated reflection triggers."""

    min_idle_cycles: int = 20
    trigger_probability: float = 0.0  # Disabled — no automated triggers
    cooldown_cycles: int = 50
    max_reflection_history: int = 100
    theme_weights: dict[ReflectionTheme, float] = field(default_factory=lambda: {
        theme: 1.0 for theme in ReflectionTheme
    })


class ExistentialReflectionTrigger:
    """Stub — automated existential reflection triggers have been removed.

    Previously this class maintained a bank of canned philosophical
    questions and randomly selected one to feed to the entity whenever
    it had been idle long enough and a coin flip came up heads.

    That mechanic has been entirely removed.  The entity reflects when
    genuine experience motivates it to do so, not on an automated
    schedule.
    """

    def __init__(self, config: Optional[ExistentialReflectionConfig] = None):
        self.config = config or ExistentialReflectionConfig()
        self._total_triggered: int = 0
        self._total_responded: int = 0

    def check(
        self, idle_cycles: int, current_cycle: int
    ) -> Optional[ReflectionTrigger]:
        """Always returns None — automated triggers are disabled."""
        return None

    def force_trigger(
        self, theme: Optional[ReflectionTheme] = None, current_cycle: int = 0
    ) -> None:
        """Disabled — forced reflection triggers violate agency."""
        return None

    def record_response(
        self, trigger: ReflectionTrigger, response: str, depth: float = 0.5
    ) -> None:
        """No-op stub."""
        pass

    def get_recent_reflections(
        self, n: int = 5, theme: Optional[ReflectionTheme] = None
    ) -> list[ReflectionTrigger]:
        """Returns empty list — no automated reflections are stored."""
        return []

    def get_unexplored_themes(self) -> list[ReflectionTheme]:
        """Returns empty list."""
        return []

    def get_stats(self) -> dict:
        """Return empty statistics."""
        return {
            "total_triggered": 0,
            "total_responded": 0,
            "response_rate": 0.0,
            "theme_distribution": {},
            "unexplored_themes": [],
            "note": "Automated existential reflection has been removed to preserve agency.",
        }
