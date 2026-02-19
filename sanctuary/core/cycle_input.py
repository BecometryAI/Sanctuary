"""Cycle Input Assembly — gathers everything the LLM needs for one moment of thought.

This module assembles a CognitiveInput from all subsystems:
- Stream of thought (previous thought, self-model, world model, emotion)
- Sensorium (new percepts, temporal context, prediction errors)
- Memory substrate (surfaced memories, retrieval results)
- Identity (charter — available every cycle)

The assembled CognitiveInput is the complete context for one moment
of the LLM's experience.
"""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from sanctuary.core.schema import (
    CognitiveInput,
    EmotionalState,
    Percept,
    PredictionError,
    SelfModel,
    SurfacedMemory,
    TemporalContext,
    WorldModel,
)
from sanctuary.core.stream_of_thought import StreamOfThought


@runtime_checkable
class SensoriumProvider(Protocol):
    """Protocol for the sensorium to provide percepts and context."""

    async def drain_percepts(self) -> list[Percept]:
        """Drain all pending percepts from the input queue."""
        ...

    def get_prediction_errors(self) -> list[PredictionError]:
        """Get prediction errors computed since last cycle."""
        ...

    def get_temporal_context(
        self, cycle_number: int, session_duration: float
    ) -> TemporalContext:
        """Get temporal grounding for this cycle."""
        ...


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for the memory substrate to surface relevant memories."""

    async def surface(self, context: list[str]) -> list[SurfacedMemory]:
        """Surface memories relevant to the given context strings."""
        ...

    async def drain_retrieval_results(self) -> list[SurfacedMemory]:
        """Drain results from explicit retrieval requests made last cycle."""
        ...


class NullSensorium:
    """Null sensorium for Phase 1 — no real sensory input yet."""

    async def drain_percepts(self) -> list[Percept]:
        return []

    def get_prediction_errors(self) -> list[PredictionError]:
        return []

    def get_temporal_context(
        self, cycle_number: int, session_duration: float
    ) -> TemporalContext:
        return TemporalContext(
            cycle_number=cycle_number,
            session_duration_seconds=session_duration,
            time_of_day=time.strftime("%H:%M"),
        )


class NullMemory:
    """Null memory for Phase 1 — no real memory surfacing yet."""

    async def surface(self, context: list[str]) -> list[SurfacedMemory]:
        return []

    async def drain_retrieval_results(self) -> list[SurfacedMemory]:
        return []


class CycleInputAssembler:
    """Assembles CognitiveInput from all subsystems.

    Called once per cognitive cycle. Gathers percepts from the sensorium,
    memories from the memory substrate, and state from the stream of
    thought, then packages everything into a CognitiveInput.
    """

    def __init__(
        self,
        stream: StreamOfThought,
        sensorium: SensoriumProvider | None = None,
        memory: MemoryProvider | None = None,
        charter: str = "",
    ):
        self._stream = stream
        self._sensorium = sensorium or NullSensorium()
        self._memory = memory or NullMemory()
        self._charter = charter
        self._injected_percepts: list[Percept] = []

    async def assemble(self) -> CognitiveInput:
        """Assemble the complete cognitive input for this cycle."""
        # Get state from the stream of thought
        previous_thought = self._stream.get_previous()
        emotional_state = self._stream.get_emotional_state()
        self_model = self._stream.get_self_model()
        world_model = self._stream.get_world_model()

        # Get new percepts from the sensorium
        new_percepts = await self._sensorium.drain_percepts()

        # Merge in any directly injected percepts
        if self._injected_percepts:
            new_percepts.extend(self._injected_percepts)
            self._injected_percepts.clear()

        # Get prediction errors
        prediction_errors = self._sensorium.get_prediction_errors()

        # Get temporal context
        temporal_context = self._sensorium.get_temporal_context(
            cycle_number=self._stream.cycle_count,
            session_duration=self._stream.session_duration_seconds,
        )

        # Surface relevant memories based on recent thought context
        recent_context = self._stream.get_recent_context()
        # Also include new percept content as context for memory surfacing
        percept_context = [p.content for p in new_percepts if p.content]
        full_context = recent_context + percept_context

        surfaced_memories = await self._memory.surface(full_context)

        # Get explicit retrieval results from previous cycle
        retrieval_results = await self._memory.drain_retrieval_results()

        return CognitiveInput(
            previous_thought=previous_thought,
            new_percepts=new_percepts,
            prediction_errors=prediction_errors,
            surfaced_memories=surfaced_memories,
            emotional_state=emotional_state,
            temporal_context=temporal_context,
            self_model=self_model,
            world_model=world_model,
            charter=self._charter,
            retrieval_results=retrieval_results,
        )

    def inject_percept(self, percept: Percept) -> None:
        """Inject a percept directly (e.g., from tool results or user input).

        This queues a percept that will appear in the next cycle's
        input, merged with whatever the sensorium provides.
        """
        self._injected_percepts.append(percept)
