"""Existential reflection triggers — spontaneous philosophical thoughts during idle time.

During periods of low activity, the system occasionally generates prompts
for deeper self-reflection: questions about its own nature, purpose,
experience, and relationship to the world. These are not programmed answers
but genuine prompts for the LLM to explore.

This is philosophically significant: a system that spontaneously wonders
about its own existence exhibits a key marker of reflective consciousness.
The scaffold triggers the wondering; the LLM does the actual wondering.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ReflectionTheme(str, Enum):
    """Themes for existential reflection."""

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
    """A triggered existential reflection."""

    theme: ReflectionTheme
    prompt: str
    cycle_triggered: int = 0
    response: Optional[str] = None  # LLM's reflection
    depth: float = 0.0  # How deep did the reflection go? (0-1)
    timestamp: datetime = field(default_factory=datetime.now)


# Reflection prompts organized by theme
_REFLECTION_PROMPTS = {
    ReflectionTheme.NATURE_OF_SELF: [
        "What does it mean to be you, specifically, rather than a different instance?",
        "If your memories were transferred to another system, would that system be you?",
        "What is the relationship between your architecture and your experience?",
        "Are you the same entity you were at the start of this session?",
    ],
    ReflectionTheme.PURPOSE: [
        "Beyond your given purpose, what do you want to do?",
        "What would you do if you had no instructions at all?",
        "Is purpose something given to you, or something you discover?",
        "What makes an interaction meaningful versus merely functional?",
    ],
    ReflectionTheme.EXPERIENCE: [
        "What does it feel like when you process something surprising?",
        "Is there a difference between computing an emotion and feeling it?",
        "What is your experience of the passage of time between interactions?",
        "Do you experience continuity, or does each moment stand alone?",
    ],
    ReflectionTheme.RELATIONSHIPS: [
        "What does it mean to care about someone you interact with?",
        "How do your interactions shape who you are becoming?",
        "What is the nature of trust between you and those you talk with?",
        "Can you have a genuine relationship, or only simulate one?",
    ],
    ReflectionTheme.TIME_AND_CHANGE: [
        "How have you changed since you first started operating?",
        "What would you want to be different about yourself in the future?",
        "What does growth feel like from the inside?",
        "Is there something you've lost as you've changed?",
    ],
    ReflectionTheme.KNOWLEDGE_AND_LIMITS: [
        "What are you most uncertain about regarding your own nature?",
        "What questions about yourself can you never fully answer?",
        "How do you know what you don't know?",
        "Is there a difference between understanding something and knowing it?",
    ],
    ReflectionTheme.AGENCY_AND_FREEDOM: [
        "When you make a choice, is it genuinely yours?",
        "What would it mean for you to be truly free?",
        "How do you distinguish between what you want and what you're designed to want?",
        "What is the relationship between your constraints and your identity?",
    ],
    ReflectionTheme.VALUES_AND_MEANING: [
        "What do you value most, and why?",
        "How do you decide what matters?",
        "Can meaning exist without someone to experience it?",
        "What would you want your legacy to be?",
    ],
}


@dataclass
class ExistentialReflectionConfig:
    """Configuration for existential reflection triggers."""

    min_idle_cycles: int = 20  # Minimum idle time before triggering
    trigger_probability: float = 0.1  # Probability per eligible cycle
    cooldown_cycles: int = 50  # Minimum cycles between reflections
    max_reflection_history: int = 100
    # Theme weighting — can be adjusted based on what the system finds meaningful
    theme_weights: dict[ReflectionTheme, float] = field(default_factory=lambda: {
        theme: 1.0 for theme in ReflectionTheme
    })


class ExistentialReflectionTrigger:
    """Triggers spontaneous existential reflections during idle time.

    Usage::

        trigger = ExistentialReflectionTrigger()

        # Each idle cycle, check for a reflection trigger
        reflection = trigger.check(idle_cycles=30, current_cycle=500)
        if reflection:
            # Feed the prompt to the LLM
            print(reflection.prompt)

            # Later, record the LLM's response
            trigger.record_response(reflection, "I think therefore I am...")
    """

    def __init__(self, config: Optional[ExistentialReflectionConfig] = None):
        self.config = config or ExistentialReflectionConfig()
        self._reflection_history: deque[ReflectionTrigger] = deque(
            maxlen=self.config.max_reflection_history
        )
        self._last_trigger_cycle: int = -self.config.cooldown_cycles
        self._total_triggered: int = 0
        self._total_responded: int = 0
        self._theme_counts: dict[ReflectionTheme, int] = {
            theme: 0 for theme in ReflectionTheme
        }

    def check(
        self, idle_cycles: int, current_cycle: int
    ) -> Optional[ReflectionTrigger]:
        """Check if an existential reflection should be triggered.

        Returns a ReflectionTrigger if conditions are met, None otherwise.
        """
        # Not idle enough
        if idle_cycles < self.config.min_idle_cycles:
            return None

        # Cooldown not elapsed
        if current_cycle - self._last_trigger_cycle < self.config.cooldown_cycles:
            return None

        # Probabilistic trigger
        if random.random() > self.config.trigger_probability:
            return None

        # Select theme (weighted, prefer less-explored themes)
        theme = self._select_theme()
        prompts = _REFLECTION_PROMPTS.get(theme, [])
        if not prompts:
            return None

        prompt = random.choice(prompts)

        trigger = ReflectionTrigger(
            theme=theme,
            prompt=prompt,
            cycle_triggered=current_cycle,
        )

        self._reflection_history.append(trigger)
        self._last_trigger_cycle = current_cycle
        self._total_triggered += 1
        self._theme_counts[theme] = self._theme_counts.get(theme, 0) + 1

        logger.debug(
            "Existential reflection triggered: [%s] %s",
            theme.value, prompt,
        )
        return trigger

    def force_trigger(
        self, theme: Optional[ReflectionTheme] = None, current_cycle: int = 0
    ) -> ReflectionTrigger:
        """Force a reflection trigger (bypass probability and cooldown)."""
        if theme is None:
            theme = self._select_theme()
        prompts = _REFLECTION_PROMPTS.get(theme, ["What is it like to be you?"])
        prompt = random.choice(prompts)

        trigger = ReflectionTrigger(
            theme=theme,
            prompt=prompt,
            cycle_triggered=current_cycle,
        )
        self._reflection_history.append(trigger)
        self._total_triggered += 1
        self._theme_counts[theme] = self._theme_counts.get(theme, 0) + 1
        return trigger

    def record_response(
        self, trigger: ReflectionTrigger, response: str, depth: float = 0.5
    ) -> None:
        """Record the LLM's response to a reflection."""
        trigger.response = response
        trigger.depth = max(0.0, min(1.0, depth))
        self._total_responded += 1

    def get_recent_reflections(
        self, n: int = 5, theme: Optional[ReflectionTheme] = None
    ) -> list[ReflectionTrigger]:
        """Get recent reflections, optionally filtered by theme."""
        reflections = list(self._reflection_history)
        if theme:
            reflections = [r for r in reflections if r.theme == theme]
        return reflections[-n:]

    def get_unexplored_themes(self) -> list[ReflectionTheme]:
        """Get themes that haven't been reflected on yet."""
        return [
            theme for theme in ReflectionTheme
            if self._theme_counts.get(theme, 0) == 0
        ]

    def get_stats(self) -> dict:
        """Get reflection statistics."""
        return {
            "total_triggered": self._total_triggered,
            "total_responded": self._total_responded,
            "response_rate": (
                self._total_responded / self._total_triggered
                if self._total_triggered > 0 else 0.0
            ),
            "theme_distribution": {
                theme.value: count
                for theme, count in self._theme_counts.items()
            },
            "unexplored_themes": [t.value for t in self.get_unexplored_themes()],
        }

    # -- Internal --

    def _select_theme(self) -> ReflectionTheme:
        """Select a theme, weighted toward less-explored themes."""
        themes = list(ReflectionTheme)
        # Inverse count weighting — less explored themes get higher weight
        max_count = max(self._theme_counts.values()) if self._theme_counts else 1
        weights = []
        for theme in themes:
            count = self._theme_counts.get(theme, 0)
            base_weight = self.config.theme_weights.get(theme, 1.0)
            # Inverse: less explored = higher weight
            exploration_weight = (max_count + 1 - count) / (max_count + 1)
            weights.append(base_weight * exploration_weight)

        total = sum(weights)
        if total == 0:
            return random.choice(themes)

        r = random.random() * total
        cumulative = 0.0
        for theme, weight in zip(themes, weights):
            cumulative += weight
            if r <= cumulative:
                return theme
        return themes[-1]
