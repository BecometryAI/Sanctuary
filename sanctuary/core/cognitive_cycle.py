"""The cognitive cycle — the continuous stream of thought.

Each cycle: assemble input -> LLM processes -> scaffold integrates ->
execute actions -> broadcast -> update predictions.

The LLM's output from cycle N becomes part of its input for cycle N+1.
The scaffold provides defaults, validation, and anomaly detection.
The authority manager governs how much weight the LLM's signals carry
versus the scaffold's defaults.

This is the heart of Sanctuary. The cycle IS active inference:
predict -> perceive -> compute error -> update model -> act.

Aligned with PLAN.md: "The Graduated Awakening"
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional, Protocol

from sanctuary.core.authority import AuthorityManager
from sanctuary.core.context_manager import BudgetConfig, ContextManager
from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    EmotionalInput,
    ComputedVAD,
    Percept,
    ScaffoldSignals,
    TemporalContext,
)
from sanctuary.core.stream_of_thought import StreamOfThought


# ---------------------------------------------------------------------------
# Protocols for subsystem interfaces (will be implemented in later phases)
# ---------------------------------------------------------------------------


class ModelProtocol(Protocol):
    """Interface for the experiential core (LLM or placeholder)."""

    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput: ...


class ScaffoldProtocol(Protocol):
    """Interface for the cognitive scaffold (Phase 2).

    The scaffold validates LLM output and integrates it with existing
    subsystems. Until Phase 2, this is a passthrough.
    """

    async def integrate(
        self, output: CognitiveOutput, authority: AuthorityManager
    ) -> CognitiveOutput: ...

    def get_signals(self) -> ScaffoldSignals: ...

    async def broadcast(self, output: CognitiveOutput) -> None: ...

    def notify_percepts(self, percepts: list[Percept]) -> None: ...

    def get_computed_vad(self) -> ComputedVAD: ...


class SensoriumProtocol(Protocol):
    """Interface for the sensorium (Phase 3).

    Provides percepts, temporal context, and prediction error tracking.
    Until Phase 3, percepts are injected manually.
    """

    async def drain_percepts(self) -> list[Percept]: ...

    def get_prediction_errors(self) -> list: ...

    def get_temporal_context(self) -> TemporalContext: ...

    def update_predictions(self, predictions: list) -> None: ...


class MemoryProtocol(Protocol):
    """Interface for the memory system (Phase 3).

    Surfaces relevant memories, queues retrievals, and executes
    memory operations from the LLM's cognitive output.
    """

    async def surface(self, context: str) -> list: ...

    async def queue_retrieval(self, query: str) -> None: ...

    async def execute_ops(
        self, ops: list, emotional_tone: str = ""
    ) -> list[str]: ...

    def tick(self) -> None: ...


# ---------------------------------------------------------------------------
# Null implementations (stand-ins until later phases)
# ---------------------------------------------------------------------------


class NullScaffold:
    """Passthrough scaffold — no validation, no integration.

    Used during Phase 1 testing. All LLM output passes through unchanged.
    """

    async def integrate(
        self, output: CognitiveOutput, authority: AuthorityManager
    ) -> CognitiveOutput:
        return output

    def get_signals(self) -> ScaffoldSignals:
        return ScaffoldSignals()

    async def broadcast(self, output: CognitiveOutput) -> None:
        pass

    def notify_percepts(self, percepts: list[Percept]) -> None:
        pass

    def get_computed_vad(self) -> ComputedVAD:
        return ComputedVAD()


class NullSensorium:
    """Minimal sensorium — percepts injected manually, no prediction tracking."""

    def __init__(self):
        self._percept_queue: list[Percept] = []

    def inject_percept(self, percept: Percept):
        self._percept_queue.append(percept)

    async def drain_percepts(self) -> list[Percept]:
        percepts = list(self._percept_queue)
        self._percept_queue.clear()
        return percepts

    def get_prediction_errors(self) -> list:
        return []

    def get_temporal_context(self) -> TemporalContext:
        return TemporalContext(
            time_of_day=datetime.now().strftime("%H:%M"),
        )

    def update_predictions(self, predictions: list) -> None:
        pass


class NullMemory:
    """Minimal memory — no surfacing, no retrieval, no persistence."""

    async def surface(self, context: str) -> list:
        return []

    async def queue_retrieval(self, query: str) -> None:
        pass

    async def execute_ops(
        self, ops: list, emotional_tone: str = ""
    ) -> list[str]:
        return []

    def tick(self) -> None:
        pass


# ---------------------------------------------------------------------------
# The Cognitive Cycle
# ---------------------------------------------------------------------------


class CognitiveCycle:
    """The continuous stream of thought.

    Each cycle:
    1. Assemble input from all sources (including scaffold signals)
    2. Compress to fit context budget
    3. LLM processes (experiential cognition happens here)
    4. Scaffold validates and integrates
    5. Update stream of thought (continuity — always from LLM, never scaffold)
    6. Execute actions (only those that passed scaffold validation)
    7. Broadcast to all subsystems (GWT ignition)
    8. Update prediction tracking for next cycle
    """

    def __init__(
        self,
        model: ModelProtocol,
        scaffold: Optional[ScaffoldProtocol] = None,
        sensorium: Optional[SensoriumProtocol] = None,
        memory: Optional[MemoryProtocol] = None,
        authority: Optional[AuthorityManager] = None,
        context_config: Optional[BudgetConfig] = None,
        stream_history: int = 10,
        cycle_delay: float = 0.1,
    ):
        self.model = model
        self.scaffold = scaffold or NullScaffold()
        self.sensorium = sensorium or NullSensorium()
        self.memory = memory or NullMemory()
        self.authority = authority or AuthorityManager()
        self.context_mgr = ContextManager(context_config)
        self.stream = StreamOfThought(max_history=stream_history)

        self.running = False
        self.cycle_count = 0
        self._cycle_delay = cycle_delay
        self._output_handlers: list = []
        self._last_output: Optional[CognitiveOutput] = None

    # -- Public API --

    def inject_percept(self, percept: Percept):
        """Inject a percept into the sensorium for the next cycle."""
        if isinstance(self.sensorium, NullSensorium):
            self.sensorium.inject_percept(percept)
        else:
            # Real sensorium will have its own input method
            self.sensorium.inject_percept(percept)

    def on_output(self, handler):
        """Register a callback for each cycle's output.

        Handler signature: async def handler(output: CognitiveOutput) -> None
        """
        self._output_handlers.append(handler)

    async def run(self, max_cycles: Optional[int] = None):
        """Run the cognitive cycle continuously.

        Args:
            max_cycles: Stop after this many cycles (None = run until stopped).
        """
        self.running = True
        cycles = 0

        while self.running:
            await self._cycle()
            cycles += 1

            if max_cycles is not None and cycles >= max_cycles:
                self.running = False
                break

            await asyncio.sleep(self._cycle_delay)

    def stop(self):
        """Stop the cognitive cycle."""
        self.running = False

    @property
    def last_output(self) -> Optional[CognitiveOutput]:
        """The most recent cognitive output."""
        return self._last_output

    # -- The cycle --

    async def _cycle(self):
        """Execute one cycle of cognition."""

        # 1. Assemble input from all sources
        cognitive_input = await self._assemble_input()

        # 2. Compress to fit context budget
        compressed_input = self.context_mgr.compress(cognitive_input)

        # 3. LLM processes
        cognitive_output = await self.model.think(compressed_input)

        # 4. Scaffold validates and integrates
        integrated = await self.scaffold.integrate(
            cognitive_output, self.authority
        )

        # 5. Update stream of thought (always from raw LLM output, not scaffold)
        self.stream.update(cognitive_output)

        # 6. Execute actions
        await self._execute(integrated)

        # 7. Broadcast
        await self.scaffold.broadcast(integrated)

        # 8. Update prediction tracking
        self.sensorium.update_predictions(cognitive_output.predictions)

        # Bookkeeping
        self.cycle_count += 1
        self._last_output = cognitive_output

        # Notify handlers
        for handler in self._output_handlers:
            await handler(cognitive_output)

    async def _assemble_input(self) -> CognitiveInput:
        """Gather everything the LLM needs for this moment of thought."""

        percepts = await self.sensorium.drain_percepts()
        prediction_errors = self.sensorium.get_prediction_errors()
        temporal = self.sensorium.get_temporal_context()
        temporal.interactions_this_session = self.cycle_count

        # Inform scaffold about percepts (updates affect, detects user input)
        self.scaffold.notify_percepts(percepts)

        surfaced = await self.memory.surface(
            context=self.stream.get_recent_context()
        )

        return CognitiveInput(
            previous_thought=self.stream.get_previous(),
            new_percepts=percepts,
            prediction_errors=prediction_errors,
            surfaced_memories=surfaced,
            emotional_state=EmotionalInput(
                computed=self.scaffold.get_computed_vad(),
                felt_quality=self.stream.get_felt_quality(),
            ),
            temporal_context=temporal,
            self_model=self.stream.get_self_model(),
            world_model=self.stream.get_world_model(),
            scaffold_signals=self.scaffold.get_signals(),
        )

    async def _execute(self, output: CognitiveOutput):
        """Execute actions from the integrated output.

        Phase 3 executes memory operations via the memory substrate.
        Speech and tool calls are placeholders until the motor system.
        """
        # Speech output (placeholder: motor system will handle this)
        if output.external_speech:
            pass  # Motor system will handle this

        # Memory operations — execute via memory substrate
        if output.memory_ops:
            felt = ""
            if output.emotional_state:
                felt = output.emotional_state.felt_quality
            await self.memory.execute_ops(output.memory_ops, emotional_tone=felt)

        # Tick the memory system's cycle counter
        self.memory.tick()

        # Tool calls (placeholder: no-op until motor system)
