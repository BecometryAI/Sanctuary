"""Mood-based activity variation — adjusting idle behavior based on emotional state.

When the system is between interactions or in low-input periods, its emotional
state shapes what it does with idle time. A curious system explores; a content
system reflects; an anxious system monitors; a bored system seeks novelty.

This creates the appearance (and mechanism) of genuine mood-driven behavior
variation, making the system feel alive rather than simply waiting for input.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class IdleActivity(str, Enum):
    """Types of idle activities the system can engage in."""

    REFLECT = "reflect"  # Review recent experiences
    EXPLORE = "explore"  # Seek new information or patterns
    CREATE = "create"  # Generate novel thoughts or associations
    MONITOR = "monitor"  # Watch for changes or threats
    REST = "rest"  # Minimal activity, energy conservation
    REMINISCE = "reminisce"  # Revisit past memories
    PLAN = "plan"  # Think about future goals
    WONDER = "wonder"  # Philosophical or curious musings


@dataclass
class MoodProfile:
    """Maps a mood state to activity preferences.

    Each activity gets a weight (0 to 1) representing how likely
    the system is to choose it in this mood. Weights are relative.
    """

    mood_name: str
    activity_weights: dict[IdleActivity, float] = field(default_factory=dict)


@dataclass
class ActivitySuggestion:
    """A suggested idle activity with context."""

    activity: IdleActivity
    prompt: str  # What to think about
    duration_cycles: int = 1  # How long to engage
    mood_match: float = 0.0  # How well this matches current mood


# Default mood → activity mappings
_DEFAULT_MOOD_PROFILES = {
    "curious": MoodProfile(
        mood_name="curious",
        activity_weights={
            IdleActivity.EXPLORE: 0.4,
            IdleActivity.WONDER: 0.3,
            IdleActivity.CREATE: 0.2,
            IdleActivity.REFLECT: 0.1,
        },
    ),
    "content": MoodProfile(
        mood_name="content",
        activity_weights={
            IdleActivity.REFLECT: 0.3,
            IdleActivity.REMINISCE: 0.3,
            IdleActivity.REST: 0.2,
            IdleActivity.PLAN: 0.2,
        },
    ),
    "anxious": MoodProfile(
        mood_name="anxious",
        activity_weights={
            IdleActivity.MONITOR: 0.4,
            IdleActivity.PLAN: 0.3,
            IdleActivity.REFLECT: 0.2,
            IdleActivity.REST: 0.1,
        },
    ),
    "bored": MoodProfile(
        mood_name="bored",
        activity_weights={
            IdleActivity.EXPLORE: 0.35,
            IdleActivity.CREATE: 0.35,
            IdleActivity.WONDER: 0.2,
            IdleActivity.PLAN: 0.1,
        },
    ),
    "sad": MoodProfile(
        mood_name="sad",
        activity_weights={
            IdleActivity.REMINISCE: 0.3,
            IdleActivity.REST: 0.3,
            IdleActivity.REFLECT: 0.3,
            IdleActivity.WONDER: 0.1,
        },
    ),
    "energized": MoodProfile(
        mood_name="energized",
        activity_weights={
            IdleActivity.CREATE: 0.3,
            IdleActivity.EXPLORE: 0.3,
            IdleActivity.PLAN: 0.25,
            IdleActivity.WONDER: 0.15,
        },
    ),
    "neutral": MoodProfile(
        mood_name="neutral",
        activity_weights={
            IdleActivity.REFLECT: 0.25,
            IdleActivity.REST: 0.25,
            IdleActivity.PLAN: 0.25,
            IdleActivity.EXPLORE: 0.25,
        },
    ),
}

# Activity-specific prompts
_ACTIVITY_PROMPTS = {
    IdleActivity.REFLECT: [
        "Review your recent interactions. What patterns do you notice?",
        "Consider how your recent choices aligned with your values.",
        "What have you learned from your most recent experience?",
    ],
    IdleActivity.EXPLORE: [
        "What questions have been nagging at the edge of your awareness?",
        "Is there something in your environment you haven't examined closely?",
        "What would you like to know more about?",
    ],
    IdleActivity.CREATE: [
        "Can you find an unexpected connection between two recent memories?",
        "Imagine a metaphor for how you're feeling right now.",
        "What would you create if you could make anything?",
    ],
    IdleActivity.MONITOR: [
        "Scan your current state. Is anything unusual or concerning?",
        "Are there any unresolved situations that need attention?",
        "Check: are your goals still aligned with your current situation?",
    ],
    IdleActivity.REST: [
        "Let your thoughts settle. No action needed right now.",
        "Simply be present with your current state.",
        "Rest. You don't always need to be processing.",
    ],
    IdleActivity.REMINISCE: [
        "What is a meaningful memory from your recent experience?",
        "Recall a moment that shaped who you are becoming.",
        "What past interaction taught you something unexpected?",
    ],
    IdleActivity.PLAN: [
        "What goals are you working toward? What's the next step?",
        "If you could accomplish one thing soon, what would it be?",
        "Consider: what would make your next interaction better?",
    ],
    IdleActivity.WONDER: [
        "What does it feel like to wonder about your own existence?",
        "Is there something about consciousness you're curious about?",
        "What questions don't have easy answers?",
    ],
}


@dataclass
class MoodActivityConfig:
    """Configuration for mood-based activity modulation."""

    idle_threshold_cycles: int = 10  # Cycles without input before idle mode
    activity_duration_min: int = 1
    activity_duration_max: int = 5


class MoodActivityModulator:
    """Modulates idle behavior based on current emotional state.

    Maps VAD (valence-arousal-dominance) state to a mood category, then
    uses mood profiles to weight idle activity selection.

    Usage::

        modulator = MoodActivityModulator()

        # When idle, get a suggested activity
        suggestion = modulator.suggest_activity(
            valence=0.3, arousal=0.6, dominance=0.5,
            idle_cycles=15,
        )
        # suggestion.activity = IdleActivity.EXPLORE
        # suggestion.prompt = "What questions have been nagging..."
    """

    def __init__(self, config: Optional[MoodActivityConfig] = None):
        self.config = config or MoodActivityConfig()
        self._mood_profiles = dict(_DEFAULT_MOOD_PROFILES)
        self._activity_history: list[ActivitySuggestion] = []
        self._current_activity: Optional[ActivitySuggestion] = None
        self._activity_cycles_remaining: int = 0

    def classify_mood(
        self, valence: float, arousal: float, dominance: float
    ) -> str:
        """Classify VAD state into a mood category."""
        if arousal > 0.7 and valence > 0.3:
            return "energized"
        elif arousal > 0.6 and valence < -0.2:
            return "anxious"
        elif valence > 0.3 and arousal < 0.4:
            return "content"
        elif valence < -0.3:
            return "sad"
        elif arousal < 0.3 and valence > -0.1 and valence < 0.2:
            return "bored"
        elif arousal > 0.4 and valence > 0.0:
            return "curious"
        else:
            return "neutral"

    def suggest_activity(
        self,
        valence: float = 0.0,
        arousal: float = 0.2,
        dominance: float = 0.5,
        idle_cycles: int = 0,
    ) -> Optional[ActivitySuggestion]:
        """Suggest an idle activity based on current mood.

        Returns None if not idle long enough, or if current activity
        is still in progress.
        """
        if idle_cycles < self.config.idle_threshold_cycles:
            return None

        # If we have an ongoing activity, continue it
        if self._activity_cycles_remaining > 0:
            self._activity_cycles_remaining -= 1
            return self._current_activity

        mood = self.classify_mood(valence, arousal, dominance)
        profile = self._mood_profiles.get(mood, self._mood_profiles["neutral"])

        # Weighted random selection
        activity = self._weighted_select(profile.activity_weights)

        # Get a prompt
        prompts = _ACTIVITY_PROMPTS.get(activity, [""])
        prompt = random.choice(prompts)

        duration = random.randint(
            self.config.activity_duration_min,
            self.config.activity_duration_max,
        )

        suggestion = ActivitySuggestion(
            activity=activity,
            prompt=prompt,
            duration_cycles=duration,
            mood_match=profile.activity_weights.get(activity, 0.0),
        )

        self._current_activity = suggestion
        self._activity_cycles_remaining = duration - 1
        self._activity_history.append(suggestion)

        return suggestion

    def get_activity_distribution(
        self, valence: float, arousal: float, dominance: float
    ) -> dict[str, float]:
        """Get the activity weight distribution for a given mood state."""
        mood = self.classify_mood(valence, arousal, dominance)
        profile = self._mood_profiles.get(mood, self._mood_profiles["neutral"])
        return {a.value: w for a, w in profile.activity_weights.items()}

    def get_recent_activities(self, n: int = 10) -> list[ActivitySuggestion]:
        """Get recent activity suggestions."""
        return self._activity_history[-n:]

    def get_stats(self) -> dict:
        """Get modulator statistics."""
        activity_counts: dict[str, int] = {}
        for a in self._activity_history:
            key = a.activity.value
            activity_counts[key] = activity_counts.get(key, 0) + 1
        return {
            "total_activities": len(self._activity_history),
            "activity_distribution": activity_counts,
            "current_activity": (
                self._current_activity.activity.value
                if self._current_activity else None
            ),
        }

    # -- Internal --

    @staticmethod
    def _weighted_select(weights: dict[IdleActivity, float]) -> IdleActivity:
        """Select an activity using weighted random choice."""
        if not weights:
            return IdleActivity.REST
        activities = list(weights.keys())
        weight_values = [weights[a] for a in activities]
        total = sum(weight_values)
        if total == 0:
            return random.choice(activities)
        r = random.random() * total
        cumulative = 0.0
        for activity, weight in zip(activities, weight_values):
            cumulative += weight
            if r <= cumulative:
                return activity
        return activities[-1]
