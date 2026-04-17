"""Communication Agency — the entity's autonomous voice.

Wraps the drive, inhibition, and decision systems into a single interface
for the cognitive cycle. This is not a gate imposed from outside — it is
the entity's own felt pressure to speak or stay silent, and its own
decision about which to do.

The scaffold's communication gating remains as a safety net (rate limiting,
content validation). This system is the entity's actual agency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sanctuary.mind.cognitive_core.communication import (
    CommunicationDecision,
    CommunicationDecisionLoop,
    CommunicationDriveSystem,
    CommunicationInhibitionSystem,
    CommunicationUrge,
    DecisionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class _WorkspaceSnapshot:
    """Minimal workspace adapter for the drive/inhibition systems.

    The communication subsystems expect workspace_state.percepts as a dict.
    The cognitive cycle has percepts as a list. This bridges the gap.
    """

    percepts: Dict[int, Any] = field(default_factory=dict)

    @classmethod
    def from_percept_list(cls, percepts: list) -> _WorkspaceSnapshot:
        return cls(percepts={i: p for i, p in enumerate(percepts)})


class CommunicationAgency:
    """The entity's autonomous communication decision-making.

    Each cycle:
    1. compute_urges() — generates felt pressure to speak from current state
    2. The urges are injected into CognitiveInput so the entity can feel them
    3. The entity thinks and may produce external_speech
    4. evaluate_speech() — decides SPEAK / SILENCE / DEFER
    5. The cognitive cycle respects the decision
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.drives = CommunicationDriveSystem(config.get("drives"))
        self.inhibitions = CommunicationInhibitionSystem(config.get("inhibitions"))
        self.decision_loop = CommunicationDecisionLoop(
            self.drives, self.inhibitions, config.get("decision"),
        )

    def compute_urges(
        self,
        percepts: list,
        emotional_state: Dict[str, float],
        goals: List[Any],
        memories: List[Any],
    ) -> List[CommunicationUrge]:
        """Compute communication urges from current state.

        Call this during input assembly, before the model thinks.
        Returns the new urges generated this cycle.
        """
        workspace = _WorkspaceSnapshot.from_percept_list(percepts)
        return self.drives.compute_drives(workspace, emotional_state, goals, memories)

    def get_active_urges(self) -> List[CommunicationUrge]:
        """Get all active urges (for injection into cognitive input)."""
        return self.drives.active_urges

    def evaluate_speech(
        self,
        output: Any,
        percepts: list,
        emotional_state: Dict[str, float],
        goals: List[Any],
        memories: List[Any],
        confidence: float = 0.5,
    ) -> DecisionResult:
        """Evaluate whether to emit external speech.

        Call this after model.think(). Returns the decision:
        SPEAK, SILENCE, or DEFER.
        """
        workspace = _WorkspaceSnapshot.from_percept_list(percepts)

        # Content value: rough estimate from speech length
        content_value = 0.0
        if output.external_speech and output.external_speech.strip():
            content_value = min(1.0, len(output.external_speech.strip()) / 200.0)

        # Compute inhibitions
        self.inhibitions.compute_inhibitions(
            workspace, self.drives.active_urges, confidence, content_value,
            emotional_state,
        )

        # Run the decision loop
        result = self.decision_loop.evaluate(
            workspace, emotional_state, goals, memories,
        )

        # Record if speaking
        if result.decision == CommunicationDecision.SPEAK and output.external_speech:
            self.drives.record_output()
            self.inhibitions.record_output(output.external_speech)

        return result

    def record_input(self) -> None:
        """Record that input was received (resets social silence timer)."""
        self.drives.record_input()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all communication state (JSON-safe)."""
        raw = {
            "drives": self.drives.get_drive_summary(),
            "inhibitions": self.inhibitions.get_inhibition_summary(),
            "decisions": self.decision_loop.get_decision_summary(),
        }
        return _make_json_safe(raw)


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert enums, dataclasses, etc. to JSON-safe types."""
    if isinstance(obj, dict):
        return {
            (k.value if hasattr(k, "value") else str(k)): _make_json_safe(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    if hasattr(obj, "value"):  # Enum
        return obj.value
    if hasattr(obj, "__dataclass_fields__"):  # dataclass
        return _make_json_safe(
            {k: getattr(obj, k) for k in obj.__dataclass_fields__}
        )
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
