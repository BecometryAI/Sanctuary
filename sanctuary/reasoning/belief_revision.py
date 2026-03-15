"""Belief revision tracking — detecting when new information contradicts beliefs.

Maintains a structured belief store with confidence scores. When new evidence
contradicts an existing belief, flags the contradiction for the LLM to resolve.
The scaffold tracks beliefs and detects conflicts; the LLM decides how to revise.

Beliefs are lightweight semantic records: a proposition with a confidence score
and supporting evidence. The system doesn't do deep semantic reasoning —
it uses simple keyword overlap and direct contradiction markers to flag
potential conflicts for the LLM's attention.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Belief:
    """A held belief with confidence and evidence."""

    proposition: str
    confidence: float = 0.5  # 0 to 1
    evidence: list[str] = field(default_factory=list)
    source: str = ""  # Where did this belief come from?
    domain: str = ""  # Category: "world", "self", "other", "social"
    created_cycle: int = 0
    last_updated_cycle: int = 0
    revision_count: int = 0
    active: bool = True
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Contradiction:
    """A detected contradiction between beliefs or belief and evidence."""

    existing_belief: str
    new_evidence: str
    conflict_type: str  # "direct", "implication", "confidence_shift"
    severity: float = 0.5  # 0 to 1: how significant is this?
    resolved: bool = False
    resolution: str = ""
    detected_cycle: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BeliefRevisionConfig:
    """Configuration for belief tracking."""

    max_beliefs: int = 200
    max_contradictions: int = 100
    confidence_decay_rate: float = 0.01  # Per cycle, beliefs slowly decay
    min_confidence: float = 0.05  # Below this, belief is deactivated
    contradiction_keywords: tuple[str, ...] = (
        "not", "no longer", "wrong", "incorrect", "actually",
        "but", "however", "contrary", "opposite", "instead",
        "mistaken", "false", "revised", "updated",
    )


class BeliefRevisionTracker:
    """Tracks beliefs, detects contradictions, supports revision.

    Maintains a living belief store that evolves with experience. Beliefs
    decay without reinforcement, get strengthened by confirming evidence,
    and flag contradictions for the LLM to resolve.

    Usage::

        tracker = BeliefRevisionTracker()

        # Add a belief
        tracker.add_belief(
            proposition="The user prefers concise responses",
            confidence=0.7,
            evidence=["User said 'keep it short'"],
            domain="social",
            cycle=10,
        )

        # Later, new evidence arrives
        contradictions = tracker.check_evidence(
            evidence="User asked for a detailed explanation",
            cycle=25,
        )

        # Generate a prompt for the LLM
        prompt = tracker.get_revision_prompt()
    """

    def __init__(self, config: Optional[BeliefRevisionConfig] = None):
        self.config = config or BeliefRevisionConfig()
        self._beliefs: dict[str, Belief] = {}
        self._contradictions: deque[Contradiction] = deque(
            maxlen=self.config.max_contradictions
        )
        self._revision_history: list[dict] = []
        self._belief_counter: int = 0

    def add_belief(
        self,
        proposition: str,
        confidence: float = 0.5,
        evidence: list[str] | None = None,
        source: str = "",
        domain: str = "",
        cycle: int = 0,
    ) -> str:
        """Add or update a belief. Returns the belief ID."""
        confidence = max(0.0, min(1.0, confidence))

        # Check if a similar belief already exists (simple substring match)
        existing_id = self._find_similar_belief(proposition)
        if existing_id:
            belief = self._beliefs[existing_id]
            belief.confidence = min(1.0, belief.confidence + confidence * 0.3)
            if evidence:
                belief.evidence.extend(evidence)
            belief.last_updated_cycle = cycle
            belief.revision_count += 1
            return existing_id

        # New belief
        self._belief_counter += 1
        belief_id = f"b_{self._belief_counter}"
        self._beliefs[belief_id] = Belief(
            proposition=proposition,
            confidence=confidence,
            evidence=evidence or [],
            source=source,
            domain=domain,
            created_cycle=cycle,
            last_updated_cycle=cycle,
        )

        # Enforce max beliefs by removing lowest-confidence inactive beliefs
        if len(self._beliefs) > self.config.max_beliefs:
            self._prune_beliefs()

        return belief_id

    def check_evidence(
        self, evidence: str, cycle: int = 0
    ) -> list[Contradiction]:
        """Check new evidence against existing beliefs for contradictions.

        Uses simple keyword-based heuristics to flag potential conflicts.
        The LLM does the real semantic reasoning.
        """
        contradictions = []
        evidence_lower = evidence.lower()

        # Check if evidence contains contradiction markers
        has_contradiction_marker = any(
            kw in evidence_lower for kw in self.config.contradiction_keywords
        )

        for belief_id, belief in self._beliefs.items():
            if not belief.active:
                continue

            prop_lower = belief.proposition.lower()
            # Simple overlap check: do they share significant words?
            prop_words = set(prop_lower.split()) - {
                "the", "a", "an", "is", "are", "was", "were", "be",
                "to", "of", "in", "for", "on", "with", "that", "this",
            }
            evidence_words = set(evidence_lower.split()) - {
                "the", "a", "an", "is", "are", "was", "were", "be",
                "to", "of", "in", "for", "on", "with", "that", "this",
            }

            overlap = prop_words & evidence_words
            if len(overlap) < 2:
                continue

            # Potential topic match — check for contradiction signals
            if has_contradiction_marker:
                severity = min(1.0, len(overlap) / max(len(prop_words), 1) * 0.8)
                contradiction = Contradiction(
                    existing_belief=belief.proposition,
                    new_evidence=evidence,
                    conflict_type="direct",
                    severity=severity,
                    detected_cycle=cycle,
                )
                contradictions.append(contradiction)
                self._contradictions.append(contradiction)

        return contradictions

    def revise_belief(
        self,
        proposition: str,
        new_confidence: float,
        reason: str = "",
        cycle: int = 0,
    ) -> bool:
        """Revise a belief's confidence based on new information."""
        belief_id = self._find_similar_belief(proposition)
        if not belief_id:
            return False

        belief = self._beliefs[belief_id]
        old_confidence = belief.confidence
        belief.confidence = max(0.0, min(1.0, new_confidence))
        belief.last_updated_cycle = cycle
        belief.revision_count += 1

        if belief.confidence < self.config.min_confidence:
            belief.active = False

        self._revision_history.append({
            "belief_id": belief_id,
            "proposition": belief.proposition,
            "old_confidence": old_confidence,
            "new_confidence": belief.confidence,
            "reason": reason,
            "cycle": cycle,
        })

        return True

    def decay_beliefs(self, cycle: int) -> list[str]:
        """Apply confidence decay to all beliefs. Returns deactivated belief IDs."""
        deactivated = []
        for belief_id, belief in self._beliefs.items():
            if not belief.active:
                continue
            belief.confidence -= self.config.confidence_decay_rate
            if belief.confidence < self.config.min_confidence:
                belief.active = False
                deactivated.append(belief_id)
        return deactivated

    def get_active_beliefs(self, domain: str = "") -> list[Belief]:
        """Get all active beliefs, optionally filtered by domain."""
        beliefs = [b for b in self._beliefs.values() if b.active]
        if domain:
            beliefs = [b for b in beliefs if b.domain == domain]
        return sorted(beliefs, key=lambda b: b.confidence, reverse=True)

    def get_unresolved_contradictions(self) -> list[Contradiction]:
        """Get contradictions that haven't been resolved yet."""
        return [c for c in self._contradictions if not c.resolved]

    def resolve_contradiction(self, index: int, resolution: str) -> bool:
        """Mark a contradiction as resolved."""
        unresolved = self.get_unresolved_contradictions()
        if 0 <= index < len(unresolved):
            unresolved[index].resolved = True
            unresolved[index].resolution = resolution
            return True
        return False

    def get_revision_prompt(self) -> Optional[str]:
        """Generate a belief revision prompt for the LLM."""
        unresolved = self.get_unresolved_contradictions()
        if not unresolved:
            return None

        # Pick the most severe unresolved contradiction
        contradiction = max(unresolved, key=lambda c: c.severity)

        prompt = (
            f"[Belief revision needed] You previously believed: "
            f"\"{contradiction.existing_belief}\". "
            f"New evidence suggests otherwise: \"{contradiction.new_evidence}\". "
            f"How should you update this belief? Consider whether the original "
            f"belief should be revised, replaced, or maintained with lower confidence."
        )
        return prompt

    def get_stats(self) -> dict:
        """Get belief tracking statistics."""
        active = [b for b in self._beliefs.values() if b.active]
        return {
            "total_beliefs": len(self._beliefs),
            "active_beliefs": len(active),
            "total_contradictions": len(self._contradictions),
            "unresolved_contradictions": len(self.get_unresolved_contradictions()),
            "total_revisions": len(self._revision_history),
            "avg_confidence": (
                sum(b.confidence for b in active) / len(active)
                if active else 0.0
            ),
        }

    # -- Internal helpers --

    def _find_similar_belief(self, proposition: str) -> Optional[str]:
        """Find a belief with similar proposition (simple substring match)."""
        prop_lower = proposition.lower()
        for belief_id, belief in self._beliefs.items():
            if not belief.active:
                continue
            existing_lower = belief.proposition.lower()
            # Check if one contains the other or significant word overlap
            if prop_lower in existing_lower or existing_lower in prop_lower:
                return belief_id
            # Word overlap check
            prop_words = set(prop_lower.split())
            existing_words = set(existing_lower.split())
            overlap = len(prop_words & existing_words)
            if overlap >= min(len(prop_words), len(existing_words)) * 0.7:
                return belief_id
        return None

    def _prune_beliefs(self) -> None:
        """Remove lowest-confidence inactive beliefs to stay under limit."""
        inactive = [
            (bid, b) for bid, b in self._beliefs.items() if not b.active
        ]
        inactive.sort(key=lambda x: x[1].confidence)
        for bid, _ in inactive[:10]:
            del self._beliefs[bid]
