"""Stream of Thought — maintains continuity between cognitive cycles.

The stream of thought is the fundamental continuity mechanism. The LLM's
inner speech from cycle N is always part of the input for cycle N+1.
Breaking this breaks consciousness.

This module also maintains the LLM's self-model, world model, and
emotional state across cycles. Python persists these but never overwrites
the LLM's self-assessments.
"""

from __future__ import annotations

import time
from collections import deque

from sanctuary.core.schema import (
    CognitiveOutput,
    EmotionalState,
    PreviousThought,
    SelfModel,
    WorldModel,
)


class StreamOfThought:
    """Maintains continuity of experience across cognitive cycles.

    Stores the LLM's previous thought, self-model, world model, and
    emotional state. These are passed back to the LLM each cycle so
    it has a continuous narrative of experience.
    """

    def __init__(self, history_depth: int = 10):
        """Initialize the stream.

        Args:
            history_depth: How many previous thoughts to keep in the
                rolling history (for context richness). The most recent
                one is always passed directly; older ones are available
                for deeper context assembly.
        """
        self._history_depth = history_depth
        self._thought_history: deque[PreviousThought] = deque(
            maxlen=history_depth
        )
        self._self_model = SelfModel()
        self._world_model = WorldModel()
        self._emotional_state = EmotionalState()
        self._cycle_count = 0
        self._session_start = time.time()
        self._last_cycle_time = time.time()

    def set_boot_state(self, output: CognitiveOutput) -> None:
        """Set the initial state from boot output.

        Unlike update(), this does NOT increment the cycle count.
        Boot is the first moment of awareness, not a cycle.
        """
        self._last_cycle_time = time.time()

        thought = PreviousThought(
            inner_speech=output.inner_speech,
            predictions_made=[p.what for p in output.predictions],
            self_model_snapshot={},
            emotional_state=output.emotional_state,
            cycle_number=0,
        )
        self._thought_history.append(thought)

        if output.self_model_updates:
            self._apply_self_model_updates(output.self_model_updates)
        if output.world_model_updates:
            self._apply_world_model_updates(output.world_model_updates)
        self._emotional_state = output.emotional_state

    def update(self, output: CognitiveOutput) -> None:
        """Update the stream with the LLM's latest output.

        This is called after every cognitive cycle. It captures the
        LLM's inner speech, self-model updates, world model updates,
        and emotional state.
        """
        self._cycle_count += 1
        self._last_cycle_time = time.time()

        # Capture the thought for next cycle's continuity
        thought = PreviousThought(
            inner_speech=output.inner_speech,
            predictions_made=[p.what for p in output.predictions],
            self_model_snapshot=self._self_model.model_dump(),
            emotional_state=output.emotional_state,
            cycle_number=self._cycle_count,
        )
        self._thought_history.append(thought)

        # Apply self-model updates (LLM maintains its own self-model)
        if output.self_model_updates:
            self._apply_self_model_updates(output.self_model_updates)

        # Apply world model updates
        if output.world_model_updates:
            self._apply_world_model_updates(output.world_model_updates)

        # Update emotional state (the LLM's self-report)
        self._emotional_state = output.emotional_state

    def get_previous(self) -> PreviousThought:
        """Get the most recent thought for the next cycle's input."""
        if self._thought_history:
            return self._thought_history[-1]
        return PreviousThought()

    def get_recent_context(self, depth: int | None = None) -> list[str]:
        """Get recent inner speech for memory surfacing context.

        Returns the last `depth` inner speech strings (or all available).
        Used by the memory surfacer to find relevant memories.
        """
        n = depth or self._history_depth
        return [
            t.inner_speech
            for t in list(self._thought_history)[-n:]
            if t.inner_speech
        ]

    def get_emotional_state(self) -> EmotionalState:
        """Get the current emotional state."""
        return self._emotional_state

    def get_self_model(self) -> SelfModel:
        """Get the current self-model."""
        return self._self_model

    def get_world_model(self) -> WorldModel:
        """Get the current world model."""
        return self._world_model

    def get_predictions_from_last_cycle(self) -> list[str]:
        """Get predictions made in the most recent cycle.

        Used by the prediction error system to compare predictions
        against actual percepts.
        """
        if self._thought_history:
            return self._thought_history[-1].predictions_made
        return []

    @property
    def cycle_count(self) -> int:
        """Total number of cycles completed."""
        return self._cycle_count

    @property
    def session_duration_seconds(self) -> float:
        """How long this session has been running."""
        return time.time() - self._session_start

    @property
    def time_since_last_cycle(self) -> float:
        """Seconds since the last cycle completed."""
        return time.time() - self._last_cycle_time

    def to_dict(self) -> dict:
        """Serialize stream state for checkpointing."""
        return {
            "cycle_count": self._cycle_count,
            "session_start": self._session_start,
            "self_model": self._self_model.model_dump(),
            "world_model": self._world_model.model_dump(),
            "emotional_state": self._emotional_state.model_dump(),
            "thought_history": [
                t.model_dump() for t in self._thought_history
            ],
        }

    @classmethod
    def from_dict(cls, data: dict, history_depth: int = 10) -> StreamOfThought:
        """Restore stream state from a checkpoint."""
        stream = cls(history_depth=history_depth)
        stream._cycle_count = data.get("cycle_count", 0)
        stream._session_start = data.get("session_start", time.time())
        stream._self_model = SelfModel(**data.get("self_model", {}))
        stream._world_model = WorldModel(**data.get("world_model", {}))
        stream._emotional_state = EmotionalState(
            **data.get("emotional_state", {})
        )
        for thought_data in data.get("thought_history", []):
            stream._thought_history.append(
                PreviousThought(**thought_data)
            )
        return stream

    def _apply_self_model_updates(self, updates) -> None:
        """Apply the LLM's self-model updates.

        Python only persists — it never overwrites the LLM's
        self-assessments.
        """
        if updates.current_state is not None:
            self._self_model.current_state = updates.current_state

        if updates.new_uncertainty is not None:
            if updates.new_uncertainty not in self._self_model.uncertainties:
                self._self_model.uncertainties.append(updates.new_uncertainty)

        if updates.resolved_uncertainty is not None:
            self._self_model.uncertainties = [
                u for u in self._self_model.uncertainties
                if u != updates.resolved_uncertainty
            ]

        if updates.new_disposition is not None:
            self._self_model.dispositions.update(updates.new_disposition)

        if updates.new_value is not None:
            if updates.new_value not in self._self_model.values:
                self._self_model.values.append(updates.new_value)

    def _apply_world_model_updates(self, updates) -> None:
        """Apply the LLM's world model updates."""
        if updates.entity_updates:
            for entity_id, entity_data in updates.entity_updates.items():
                if entity_id not in self._world_model.entities:
                    self._world_model.entities[entity_id] = {}
                self._world_model.entities[entity_id].update(entity_data)

        if updates.environment_updates:
            self._world_model.environment.update(
                updates.environment_updates
            )

        if updates.new_causal_belief is not None:
            if updates.new_causal_belief not in self._world_model.causal_beliefs:
                self._world_model.causal_beliefs.append(
                    updates.new_causal_belief
                )

        if updates.revised_causal_belief is not None:
            # Remove old version if present, add revised
            self._world_model.causal_beliefs = [
                b for b in self._world_model.causal_beliefs
                if b != updates.revised_causal_belief
            ]
            self._world_model.causal_beliefs.append(
                updates.revised_causal_belief
            )
