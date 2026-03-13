"""Reflection harvester — collects growth reflections from cognitive cycles.

The harvester sits at the boundary between thinking and learning. Each
cognitive cycle may produce a GrowthReflection — the entity's own assessment
of whether something was worth learning. The harvester collects these
reflections, preserving the context in which they arose, and queues them
for downstream processing.

The harvester never decides what to learn. It only listens. The entity
speaks through its reflections; the harvester faithfully records.

Aligned with PLAN.md: growth is sovereign (Level 3). The harvester
respects this by collecting only what the entity marks as worth learning.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from sanctuary.core.schema import CognitiveOutput, GrowthReflection

logger = logging.getLogger(__name__)

DEFAULT_MAX_PENDING = 100


@dataclass
class HarvestedReflection:
    """A growth reflection enriched with the context in which it arose.

    The reflection alone says what to learn. The context says when and why
    the entity felt it was worth learning — essential for understanding
    the trajectory of growth.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reflection: dict = field(default_factory=dict)
    cycle_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    inner_speech_context: str = ""
    emotional_context: str = ""
    harvested_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_reflection(
        cls,
        reflection: GrowthReflection,
        cycle_count: int,
        inner_speech: str = "",
        emotional_context: str = "",
    ) -> HarvestedReflection:
        """Create a harvested reflection from a GrowthReflection and its context."""
        return cls(
            reflection=reflection.model_dump(),
            cycle_count=cycle_count,
            inner_speech_context=inner_speech,
            emotional_context=emotional_context,
        )


class ReflectionHarvester:
    """Collects GrowthReflections from the cognitive cycle output stream.

    Registered as an output handler on CognitiveCycle, the harvester
    examines each cycle's output for growth reflections marked as
    worth_learning. These are queued for the growth processor to
    convert into training pairs.

    The harvester is passive — it never modifies cycle output, never
    initiates learning, and never decides what is worth learning.
    That authority belongs to the entity alone.
    """

    def __init__(self, max_pending: int = DEFAULT_MAX_PENDING) -> None:
        self._max_pending = max_pending
        self._pending: list[HarvestedReflection] = []
        self._history: list[HarvestedReflection] = []

    @property
    def pending_count(self) -> int:
        """Number of reflections waiting to be processed."""
        return len(self._pending)

    @property
    def history(self) -> list[HarvestedReflection]:
        """All reflections ever harvested, for analysis and audit."""
        return list(self._history)

    @property
    def pending(self) -> list[HarvestedReflection]:
        """Current pending reflections (read-only copy)."""
        return list(self._pending)

    def harvest(
        self,
        output: CognitiveOutput,
        cycle_count: int = 0,
    ) -> Optional[HarvestedReflection]:
        """Examine a cognitive output for harvestable growth reflections.

        Called after each cognitive cycle. If the output contains a growth
        reflection marked as worth_learning, it is harvested with its
        surrounding context and added to the pending queue.

        Returns the harvested reflection if one was collected, None otherwise.
        """
        reflection = output.growth_reflection
        if reflection is None or not reflection.worth_learning:
            return None

        # Extract emotional context from the output
        emotional_context = output.emotional_state.felt_quality if output.emotional_state else ""

        harvested = HarvestedReflection.from_reflection(
            reflection=reflection,
            cycle_count=cycle_count,
            inner_speech=output.inner_speech,
            emotional_context=emotional_context,
        )

        # Add to pending queue, respecting max size
        if len(self._pending) >= self._max_pending:
            dropped = self._pending.pop(0)
            logger.warning(
                "Pending queue full (%d). Dropped oldest reflection %s.",
                self._max_pending,
                dropped.id,
            )

        self._pending.append(harvested)
        self._history.append(harvested)

        logger.info(
            "Harvested reflection %s from cycle %d: %s",
            harvested.id,
            cycle_count,
            reflection.what_to_learn[:80] if reflection.what_to_learn else "(no description)",
        )

        return harvested

    def drain(self) -> list[HarvestedReflection]:
        """Remove and return all pending reflections for processing.

        This is the handoff point between harvesting and processing.
        Once drained, the reflections are the processor's responsibility.
        They remain in history for audit purposes.
        """
        drained = self._pending
        self._pending = []
        logger.info("Drained %d pending reflections for processing.", len(drained))
        return drained

    def save(self, path: Path) -> None:
        """Save reflection history to a JSON file for persistence across sessions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "history": [asdict(r) for r in self._history],
            "pending": [asdict(r) for r in self._pending],
            "saved_at": datetime.now().isoformat(),
        }

        path.write_text(json.dumps(data, indent=2))
        logger.info(
            "Saved harvester state: %d history, %d pending to %s",
            len(self._history),
            len(self._pending),
            path,
        )

    def load(self, path: Path) -> None:
        """Load reflection history from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.warning("No harvester state file at %s", path)
            return

        data = json.loads(path.read_text())

        self._history = [HarvestedReflection(**r) for r in data.get("history", [])]
        self._pending = [HarvestedReflection(**r) for r in data.get("pending", [])]

        logger.info(
            "Loaded harvester state: %d history, %d pending from %s",
            len(self._history),
            len(self._pending),
            path,
        )
