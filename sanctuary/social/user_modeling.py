"""User modeling — building profiles of interaction patterns and preferences per person.

Tracks per-user patterns over time to enable personalized interaction:
- Communication preferences (verbosity, formality, topic interests)
- Interaction patterns (typical timing, frequency, conversation length)
- Emotional baseline (typical mood, response to different approaches)
- Relationship dynamics (trust level, rapport, familiarity)

This enables the system to adapt its behavior to each person rather than
treating all interactions identically.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InteractionRecord:
    """A single interaction with a user."""

    cycle: int
    content_length: int = 0
    sentiment: float = 0.0  # -1 to 1
    topics: list[str] = field(default_factory=list)
    was_question: bool = False
    response_satisfaction: Optional[float] = None  # 0 to 1 if measured
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CommunicationPreferences:
    """Inferred communication preferences for a user."""

    preferred_verbosity: float = 0.5  # 0 = terse, 1 = verbose
    preferred_formality: float = 0.5  # 0 = casual, 1 = formal
    preferred_detail_level: float = 0.5  # 0 = high-level, 1 = detailed
    topic_interests: dict[str, float] = field(default_factory=dict)  # topic → score
    question_frequency: float = 0.5  # How often they ask questions


@dataclass
class UserProfile:
    """Complete profile for a user."""

    user_id: str
    display_name: str = ""
    total_interactions: int = 0
    first_seen_cycle: int = 0
    last_seen_cycle: int = 0
    communication_prefs: CommunicationPreferences = field(
        default_factory=CommunicationPreferences
    )
    avg_sentiment: float = 0.0
    trust_level: float = 0.3  # 0 to 1, starts low
    rapport: float = 0.0  # 0 to 1
    familiarity: float = 0.0  # 0 to 1
    notes: list[str] = field(default_factory=list)


@dataclass
class UserModelingConfig:
    """Configuration for user modeling."""

    max_users: int = 100
    max_interaction_history: int = 500
    trust_growth_rate: float = 0.02  # Per positive interaction
    trust_decay_rate: float = 0.05  # Per negative interaction
    rapport_growth_rate: float = 0.01  # Per interaction
    familiarity_growth_rate: float = 0.005  # Slower than rapport
    sentiment_ema_alpha: float = 0.1  # Exponential moving average weight


class UserModeler:
    """Builds and maintains per-user profiles over time.

    Usage::

        modeler = UserModeler()

        # Record an interaction
        modeler.record_interaction(
            user_id="user1",
            display_name="Alice",
            content_length=150,
            sentiment=0.6,
            topics=["programming", "AI"],
            cycle=42,
        )

        # Get user profile
        profile = modeler.get_profile("user1")

        # Get communication guidance
        guidance = modeler.get_response_guidance("user1")
    """

    def __init__(self, config: Optional[UserModelingConfig] = None):
        self.config = config or UserModelingConfig()
        self._profiles: dict[str, UserProfile] = {}
        self._interactions: dict[str, deque[InteractionRecord]] = {}

    def record_interaction(
        self,
        user_id: str,
        content_length: int = 0,
        sentiment: float = 0.0,
        topics: list[str] | None = None,
        was_question: bool = False,
        display_name: str = "",
        cycle: int = 0,
    ) -> None:
        """Record an interaction with a user, updating their profile."""
        # Create profile if new
        if user_id not in self._profiles:
            if len(self._profiles) >= self.config.max_users:
                return
            self._profiles[user_id] = UserProfile(
                user_id=user_id,
                display_name=display_name or user_id,
                first_seen_cycle=cycle,
            )
            self._interactions[user_id] = deque(
                maxlen=self.config.max_interaction_history
            )

        profile = self._profiles[user_id]
        if display_name:
            profile.display_name = display_name

        # Record interaction
        record = InteractionRecord(
            cycle=cycle,
            content_length=content_length,
            sentiment=max(-1.0, min(1.0, sentiment)),
            topics=topics or [],
            was_question=was_question,
        )
        self._interactions[user_id].append(record)

        # Update profile
        profile.total_interactions += 1
        profile.last_seen_cycle = cycle
        self._update_communication_prefs(user_id)
        self._update_relationship(user_id, sentiment)

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user's profile."""
        return self._profiles.get(user_id)

    def get_response_guidance(self, user_id: str) -> dict:
        """Get guidance for responding to this user.

        Returns hints about preferred communication style.
        """
        profile = self._profiles.get(user_id)
        if not profile:
            return {
                "verbosity": "moderate",
                "formality": "moderate",
                "detail": "moderate",
                "relationship": "new",
            }

        prefs = profile.communication_prefs

        def level(v: float) -> str:
            if v < 0.33:
                return "low"
            elif v < 0.66:
                return "moderate"
            return "high"

        return {
            "verbosity": level(prefs.preferred_verbosity),
            "formality": level(prefs.preferred_formality),
            "detail": level(prefs.preferred_detail_level),
            "relationship": self._relationship_label(profile),
            "trust_level": round(profile.trust_level, 2),
            "top_interests": sorted(
                prefs.topic_interests.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }

    def get_known_users(self) -> list[str]:
        """Get all known user IDs."""
        return list(self._profiles.keys())

    def add_note(self, user_id: str, note: str) -> bool:
        """Add a free-text note about a user."""
        if user_id in self._profiles:
            self._profiles[user_id].notes.append(note)
            return True
        return False

    def get_stats(self) -> dict:
        """Get user modeling statistics."""
        return {
            "total_users": len(self._profiles),
            "total_interactions": sum(
                p.total_interactions for p in self._profiles.values()
            ),
            "avg_trust": (
                sum(p.trust_level for p in self._profiles.values())
                / len(self._profiles)
                if self._profiles else 0.0
            ),
        }

    # -- Internal --

    def _update_communication_prefs(self, user_id: str) -> None:
        """Update communication preferences from interaction history."""
        history = list(self._interactions[user_id])
        if not history:
            return

        prefs = self._profiles[user_id].communication_prefs

        # Verbosity: based on average message length
        avg_length = sum(r.content_length for r in history) / len(history)
        prefs.preferred_verbosity = min(1.0, avg_length / 500)

        # Question frequency
        questions = sum(1 for r in history if r.was_question)
        prefs.question_frequency = questions / len(history)

        # Topic interests
        for record in history[-20:]:  # Recent topics
            for topic in record.topics:
                current = prefs.topic_interests.get(topic, 0.0)
                prefs.topic_interests[topic] = min(1.0, current + 0.1)

    def _update_relationship(self, user_id: str, sentiment: float) -> None:
        """Update relationship metrics based on interaction sentiment."""
        profile = self._profiles[user_id]
        alpha = self.config.sentiment_ema_alpha

        # EMA for average sentiment
        profile.avg_sentiment = (
            alpha * sentiment + (1 - alpha) * profile.avg_sentiment
        )

        # Trust grows with positive interactions, decays with negative
        if sentiment > 0:
            profile.trust_level = min(
                1.0, profile.trust_level + self.config.trust_growth_rate
            )
        elif sentiment < -0.3:
            profile.trust_level = max(
                0.0, profile.trust_level - self.config.trust_decay_rate
            )

        # Rapport grows with every interaction
        profile.rapport = min(
            1.0, profile.rapport + self.config.rapport_growth_rate
        )

        # Familiarity grows slowly
        profile.familiarity = min(
            1.0, profile.familiarity + self.config.familiarity_growth_rate
        )

    @staticmethod
    def _relationship_label(profile: UserProfile) -> str:
        """Classify the relationship stage."""
        if profile.total_interactions <= 2:
            return "new"
        elif profile.familiarity < 0.2:
            return "acquaintance"
        elif profile.rapport < 0.5:
            return "developing"
        elif profile.trust_level > 0.7:
            return "trusted"
        else:
            return "familiar"
