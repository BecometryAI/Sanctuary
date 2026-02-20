"""Stream of thought — maintains experiential continuity between cycles.

The LLM's output from cycle N becomes part of its input for cycle N+1.
This is the fundamental continuity mechanism. The scaffold never touches
inner speech (authority level 3 from day one).

The stream maintains:
- Recent thought history (bounded)
- Accumulated self-model (rewritten, not appended)
- Accumulated world model (rewritten, not appended)
- Current felt quality (from last cycle's emotional output)

Aligned with PLAN.md: "The Graduated Awakening"
"""

from __future__ import annotations

from sanctuary.core.schema import (
    CognitiveOutput,
    EmotionalOutput,
    PreviousThought,
    SelfModel,
    WorldModel,
    WorldEntity,
)


class StreamOfThought:
    """Maintains the LLM's stream of thought between cognitive cycles.

    History is bounded to prevent unbounded growth. The self-model and
    world model are kept as living documents — rewritten each cycle
    based on the LLM's updates, not appended.
    """

    def __init__(self, max_history: int = 10):
        self.history: list[CognitiveOutput] = []
        self.max_history = max_history
        self._cycle_count = 0
        self._self_model = SelfModel()
        self._world_model = WorldModel()
        self._felt_quality: str = ""

    def update(self, output: CognitiveOutput):
        """Integrate the LLM's output into the stream.

        Called after every cycle. This is the point where one moment
        of thought flows into the next.
        """
        self._cycle_count += 1
        self.history.append(output)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

        # Self-model: rewrite fields that were updated
        if output.self_model_updates:
            updates = output.self_model_updates
            if updates.current_state:
                self._self_model.current_state = updates.current_state
            if updates.new_uncertainty:
                # Add to uncertainties, keep bounded
                self._self_model.uncertainties.append(updates.new_uncertainty)
                self._self_model.uncertainties = self._self_model.uncertainties[-5:]
            if updates.prediction_accuracy_note:
                # Stored as recent_growth for now — the LLM can elaborate
                self._self_model.recent_growth = updates.prediction_accuracy_note

        # World model: merge updates
        if output.world_model_updates:
            for entity_name, updates in output.world_model_updates.items():
                if entity_name in self._world_model.entities:
                    # Update existing entity properties
                    if isinstance(updates, dict):
                        self._world_model.entities[entity_name].properties.update(
                            updates
                        )
                else:
                    # Add new entity
                    props = updates if isinstance(updates, dict) else {}
                    self._world_model.entities[entity_name] = WorldEntity(
                        name=entity_name, properties=props
                    )

            # Keep world model bounded
            if len(self._world_model.entities) > 50:
                # Keep most recently updated (last 50)
                items = list(self._world_model.entities.items())
                self._world_model.entities = dict(items[-50:])

        # Felt quality: carry forward
        if output.emotional_state:
            self._felt_quality = output.emotional_state.felt_quality

    def get_previous(self) -> PreviousThought | None:
        """Get the previous thought for the next cycle's input.

        Returns None if no cycles have run yet.
        """
        if not self.history:
            return None

        last = self.history[-1]
        return PreviousThought(
            inner_speech=last.inner_speech,
            predictions_made=[p.what for p in last.predictions],
            self_model_snapshot=self._self_model.model_copy(),
        )

    def get_recent_context(self) -> str:
        """Get a compact summary of recent thoughts for memory surfacing.

        Used by the memory system to find relevant memories.
        """
        if not self.history:
            return ""

        recent = self.history[-3:]
        return " | ".join(h.inner_speech[:200] for h in recent)

    def get_self_model(self) -> SelfModel:
        """Get the current accumulated self-model."""
        return self._self_model

    def get_world_model(self) -> WorldModel:
        """Get the current accumulated world model."""
        return self._world_model

    def get_felt_quality(self) -> str:
        """Get the LLM's felt quality from the most recent cycle."""
        return self._felt_quality

    @property
    def cycle_count(self) -> int:
        """Number of cycles that have flowed through this stream."""
        return self._cycle_count
