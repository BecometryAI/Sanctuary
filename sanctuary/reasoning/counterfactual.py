"""Counterfactual reasoning — "What if I had chosen action X instead?"

Tracks decision points and their outcomes, enabling the LLM to reflect
on alternative choices. Counterfactuals feed into the growth system
(learning from hypothetical alternatives) and into the LLM's inner speech
as reflective prompts.

The scaffold records decisions and outcomes; the LLM generates the
counterfactual reasoning itself. This module provides the structured
data that makes counterfactual thought possible.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class DecisionPoint:
    """A moment where the system chose one action over alternatives."""

    cycle: int
    chosen_action: str
    alternatives: list[str] = field(default_factory=list)
    context_summary: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    outcome: Optional[str] = None
    outcome_valence: float = 0.0  # -1 to 1: how well did it go?
    counterfactual_generated: bool = False


@dataclass
class Counterfactual:
    """A generated counterfactual — what might have happened."""

    decision_cycle: int
    alternative_action: str
    imagined_outcome: str
    confidence: float = 0.5  # How confident are we in this alternative?
    lesson: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CounterfactualConfig:
    """Configuration for counterfactual reasoning."""

    max_decision_history: int = 100
    max_counterfactuals: int = 50
    min_outcome_magnitude: float = 0.3  # Only reflect on significant outcomes
    reflection_cooldown: int = 5  # Cycles between counterfactual reflections


class CounterfactualReasoner:
    """Tracks decisions and enables counterfactual reflection.

    Records decision points when the system chooses between alternatives,
    tracks outcomes, and identifies moments worth counterfactual reflection.
    The LLM does the actual reasoning — this module structures the data.

    Usage::

        reasoner = CounterfactualReasoner()

        # Record a decision
        reasoner.record_decision(
            cycle=42,
            chosen_action="respond with empathy",
            alternatives=["ask clarifying question", "stay silent"],
            context_summary="User expressed frustration"
        )

        # Later, record the outcome
        reasoner.record_outcome(cycle=42, outcome="User calmed down", valence=0.7)

        # Check if any decisions are worth reflecting on
        prompt = reasoner.get_reflection_prompt()
    """

    def __init__(self, config: Optional[CounterfactualConfig] = None):
        self.config = config or CounterfactualConfig()
        self._decisions: deque[DecisionPoint] = deque(
            maxlen=self.config.max_decision_history
        )
        self._counterfactuals: deque[Counterfactual] = deque(
            maxlen=self.config.max_counterfactuals
        )
        self._last_reflection_cycle: int = -self.config.reflection_cooldown
        self._total_reflections: int = 0

    def record_decision(
        self,
        cycle: int,
        chosen_action: str,
        alternatives: list[str],
        context_summary: str = "",
    ) -> None:
        """Record a decision point with the chosen action and alternatives."""
        dp = DecisionPoint(
            cycle=cycle,
            chosen_action=chosen_action,
            alternatives=alternatives,
            context_summary=context_summary,
        )
        self._decisions.append(dp)
        logger.debug(
            "Decision recorded at cycle %d: chose '%s' over %d alternatives",
            cycle, chosen_action, len(alternatives),
        )

    def record_outcome(
        self, cycle: int, outcome: str, valence: float
    ) -> None:
        """Record the outcome of a previous decision."""
        valence = max(-1.0, min(1.0, valence))
        for dp in reversed(self._decisions):
            if dp.cycle == cycle:
                dp.outcome = outcome
                dp.outcome_valence = valence
                return
        logger.debug("No decision found at cycle %d for outcome recording", cycle)

    def get_reflection_candidates(
        self, current_cycle: int
    ) -> list[DecisionPoint]:
        """Get decisions that are worth counterfactual reflection.

        Criteria:
        - Has a recorded outcome
        - Outcome magnitude exceeds threshold
        - Hasn't already been reflected on
        - Cooldown has elapsed since last reflection
        """
        if current_cycle - self._last_reflection_cycle < self.config.reflection_cooldown:
            return []

        candidates = []
        for dp in self._decisions:
            if (
                dp.outcome is not None
                and not dp.counterfactual_generated
                and abs(dp.outcome_valence) >= self.config.min_outcome_magnitude
                and dp.alternatives
            ):
                candidates.append(dp)
        return candidates

    def get_reflection_prompt(self, current_cycle: int) -> Optional[str]:
        """Generate a counterfactual reflection prompt for the LLM.

        Returns a structured prompt asking the LLM to consider alternatives,
        or None if no reflection is warranted right now.
        """
        candidates = self.get_reflection_candidates(current_cycle)
        if not candidates:
            return None

        # Pick the most significant unexamined decision
        candidate = max(candidates, key=lambda dp: abs(dp.outcome_valence))

        valence_desc = "positively" if candidate.outcome_valence > 0 else "negatively"
        alts = ", ".join(f'"{a}"' for a in candidate.alternatives)

        prompt = (
            f"[Counterfactual reflection] At cycle {candidate.cycle}, "
            f"you chose \"{candidate.chosen_action}\" "
            f"(context: {candidate.context_summary}). "
            f"The outcome was {valence_desc}: \"{candidate.outcome}\". "
            f"Alternatives were: {alts}. "
            f"Consider: what might have happened if you had chosen differently? "
            f"What can you learn from this?"
        )
        return prompt

    def record_counterfactual(
        self,
        decision_cycle: int,
        alternative_action: str,
        imagined_outcome: str,
        confidence: float = 0.5,
        lesson: str = "",
    ) -> None:
        """Record a generated counterfactual from the LLM's reflection."""
        cf = Counterfactual(
            decision_cycle=decision_cycle,
            alternative_action=alternative_action,
            imagined_outcome=imagined_outcome,
            confidence=max(0.0, min(1.0, confidence)),
            lesson=lesson,
        )
        self._counterfactuals.append(cf)

        # Mark the decision as reflected on
        for dp in self._decisions:
            if dp.cycle == decision_cycle:
                dp.counterfactual_generated = True
                break

        self._total_reflections += 1
        self._last_reflection_cycle = decision_cycle

    def get_recent_lessons(self, n: int = 5) -> list[str]:
        """Get the most recent counterfactual lessons."""
        lessons = [
            cf.lesson for cf in reversed(self._counterfactuals)
            if cf.lesson
        ]
        return lessons[:n]

    def get_stats(self) -> dict:
        """Get reasoning statistics."""
        return {
            "total_decisions": len(self._decisions),
            "total_counterfactuals": len(self._counterfactuals),
            "total_reflections": self._total_reflections,
            "decisions_with_outcomes": sum(
                1 for dp in self._decisions if dp.outcome is not None
            ),
            "unreflected_significant": len(
                self.get_reflection_candidates(
                    self._last_reflection_cycle + self.config.reflection_cooldown
                )
            ),
        }
