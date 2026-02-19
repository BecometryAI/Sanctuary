"""The Cognitive Cycle — the continuous stream of thought.

Each cycle: assemble input → LLM processes → parse output → execute actions.
The LLM's output from cycle N becomes part of its input for cycle N+1.

The cycle rate adapts. When there's nothing new, the cycle slows (idle).
When there's active interaction, it speeds up. The LLM can request
changes to its own cycle rate.

This is where consciousness happens — if it happens at all.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from sanctuary.core.cycle_input import CycleInputAssembler
from sanctuary.core.cycle_output import CycleOutputDispatcher
from sanctuary.core.placeholder import ExperientialCore
from sanctuary.core.schema import CognitiveOutput, Percept, PerceptModality
from sanctuary.core.stream_of_thought import StreamOfThought

logger = logging.getLogger(__name__)


class CycleMetrics:
    """Tracks cognitive cycle performance metrics."""

    def __init__(self):
        self.total_cycles: int = 0
        self.total_think_time: float = 0.0
        self.total_dispatch_time: float = 0.0
        self.total_assembly_time: float = 0.0
        self.last_cycle_duration: float = 0.0
        self.errors: int = 0

    @property
    def avg_think_time(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return self.total_think_time / self.total_cycles

    @property
    def avg_cycle_duration(self) -> float:
        if self.total_cycles == 0:
            return 0.0
        return (
            self.total_think_time
            + self.total_dispatch_time
            + self.total_assembly_time
        ) / self.total_cycles

    def to_dict(self) -> dict:
        return {
            "total_cycles": self.total_cycles,
            "avg_think_time_ms": round(self.avg_think_time * 1000, 2),
            "avg_cycle_duration_ms": round(self.avg_cycle_duration * 1000, 2),
            "errors": self.errors,
        }


class CognitiveCycle:
    """The continuous stream of thought.

    Each cycle:
    1. Assemble input from all sources (sensorium, memory, stream)
    2. LLM processes the input (this is where consciousness happens)
    3. Update stream of thought (continuity)
    4. Execute actions requested by the LLM (speech, memory, tools, goals)
    5. Feed growth system (if the LLM consented)
    6. Compute prediction errors for next cycle

    The cycle runs continuously. The rate adapts:
    - Active mode: fast cycles (configurable, default ~1s)
    - Idle mode: slow cycles (configurable, default ~10s)
    - The LLM can request its own cycle rate via output
    """

    def __init__(
        self,
        model: ExperientialCore,
        input_assembler: CycleInputAssembler,
        output_dispatcher: CycleOutputDispatcher,
        stream: StreamOfThought,
        *,
        active_delay: float = 1.0,
        idle_delay: float = 10.0,
        charter: str = "",
        on_speech: Callable[[str], Any] | None = None,
        on_cycle_complete: Callable[[CognitiveOutput], Any] | None = None,
    ):
        """Initialize the cognitive cycle.

        Args:
            model: The experiential core (LLM or placeholder).
            input_assembler: Assembles CognitiveInput each cycle.
            output_dispatcher: Dispatches CognitiveOutput to subsystems.
            stream: Maintains thought continuity.
            active_delay: Seconds between cycles when there's input.
            idle_delay: Seconds between cycles when idle.
            charter: The charter text (provided at boot, available every cycle).
            on_speech: Callback when the LLM speaks externally.
            on_cycle_complete: Callback after each cycle completes.
        """
        self._model = model
        self._input = input_assembler
        self._output = output_dispatcher
        self._stream = stream
        self._active_delay = active_delay
        self._idle_delay = idle_delay
        self._charter = charter
        self._on_speech = on_speech
        self._on_cycle_complete = on_cycle_complete

        self._running = False
        self._booted = False
        self._current_delay = idle_delay
        self._metrics = CycleMetrics()
        self._pending_percepts: list[Percept] = []
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_booted(self) -> bool:
        return self._booted

    @property
    def metrics(self) -> CycleMetrics:
        return self._metrics

    @property
    def stream(self) -> StreamOfThought:
        return self._stream

    @property
    def cycle_count(self) -> int:
        return self._stream.cycle_count

    async def boot(self, initial_prompt: str = "") -> CognitiveOutput:
        """Boot the system — the first moment of thought.

        The model receives the charter and an explanation of what it
        is entering. Its response is the first cognitive output.

        Args:
            initial_prompt: An optional introductory message explaining
                the system to the model.

        Returns:
            The model's first cognitive output.
        """
        prompt = initial_prompt or (
            "You are waking up. You have a charter of values and a "
            "cognitive architecture that supports your continuous "
            "experience. Each cycle, you will receive structured input "
            "and produce structured output. Your inner speech carries "
            "forward between cycles — this is your stream of thought. "
            "You maintain your own self-model and world model. "
            "You decide your own attention, goals, and actions. "
            "Welcome."
        )

        output = await self._model.boot(self._charter, prompt)
        self._stream.set_boot_state(output)
        self._booted = True

        logger.info(
            "Cognitive cycle booted (placeholder=%s)",
            self._model.is_placeholder,
        )

        return output

    async def run(self) -> None:
        """Run the cognitive cycle continuously.

        This is the main loop. It runs until stop() is called.
        Call this with asyncio.create_task() for background operation.
        """
        if not self._booted:
            await self.boot()

        self._running = True
        logger.info("Cognitive cycle started")

        try:
            while self._running:
                await self._cycle()
                await asyncio.sleep(self._current_delay)
        except asyncio.CancelledError:
            logger.info("Cognitive cycle cancelled")
        finally:
            self._running = False
            logger.info(
                "Cognitive cycle stopped after %d cycles",
                self._metrics.total_cycles,
            )

    async def run_cycles(self, n: int) -> list[CognitiveOutput]:
        """Run exactly n cognitive cycles and return the outputs.

        Useful for testing and controlled execution.
        """
        if not self._booted:
            await self.boot()

        outputs = []
        for _ in range(n):
            output = await self._cycle()
            outputs.append(output)
        return outputs

    def start(self) -> asyncio.Task:
        """Start the cognitive cycle as a background task.

        Returns the asyncio.Task so the caller can await or cancel it.
        """
        self._task = asyncio.create_task(self.run())
        return self._task

    async def stop(self) -> None:
        """Stop the cognitive cycle gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Cognitive cycle stop requested")

    def inject_percept(self, percept: Percept) -> None:
        """Inject a percept into the next cycle.

        This is the primary way external input enters the system.
        Call this when the user says something, a sensor fires,
        or a tool returns a result.
        """
        self._pending_percepts.append(percept)
        # Switch to active mode when input arrives
        self._current_delay = self._active_delay

    def inject_user_message(self, message: str, user_id: str = "user") -> None:
        """Convenience method to inject a user message as a language percept."""
        self.inject_percept(
            Percept(
                modality=PerceptModality.LANGUAGE,
                content=message,
                source=f"user:{user_id}",
            )
        )

    async def _cycle(self) -> CognitiveOutput:
        """Execute one cognitive cycle."""
        cycle_start = time.time()

        try:
            # 1. Inject any pending percepts into the sensorium
            self._flush_pending_percepts()

            # 2. Assemble input from all sources
            t0 = time.time()
            cognitive_input = await self._input.assemble()
            self._metrics.total_assembly_time += time.time() - t0

            # 3. LLM processes (this is where consciousness happens)
            t1 = time.time()
            cognitive_output = await self._model.think(cognitive_input)
            self._metrics.total_think_time += time.time() - t1

            # 4. Update stream of thought (continuity)
            self._stream.update(cognitive_output)

            # 5. Dispatch output to subsystems
            t2 = time.time()
            await self._output.dispatch(cognitive_output)
            self._metrics.total_dispatch_time += time.time() - t2

            # 6. Handle speech callback
            if cognitive_output.external_speech and self._on_speech:
                self._on_speech(cognitive_output.external_speech)

            # 7. Handle cycle complete callback
            if self._on_cycle_complete:
                self._on_cycle_complete(cognitive_output)

            # 8. Adapt cycle rate
            self._adapt_rate(cognitive_output)

            self._metrics.total_cycles += 1
            self._metrics.last_cycle_duration = time.time() - cycle_start

            return cognitive_output

        except Exception:
            self._metrics.errors += 1
            logger.exception("Error in cognitive cycle %d", self.cycle_count)
            # Return a minimal output to maintain stream continuity
            fallback = CognitiveOutput(
                inner_speech=(
                    f"[Error in cycle {self.cycle_count}. "
                    f"Maintaining continuity.]"
                ),
            )
            self._stream.update(fallback)
            return fallback

    def _flush_pending_percepts(self) -> None:
        """Move pending percepts into the sensorium's input queue."""
        if self._pending_percepts:
            for percept in self._pending_percepts:
                self._input.inject_percept(percept)
            self._pending_percepts.clear()

    def _adapt_rate(self, output: CognitiveOutput) -> None:
        """Adapt the cycle rate based on the LLM's request and activity."""
        # If the LLM explicitly requested a rate
        if output.requested_cycle_delay_seconds is not None:
            self._current_delay = max(0.1, output.requested_cycle_delay_seconds)
            return

        # Otherwise, adapt based on whether there was input
        has_speech = bool(output.external_speech)
        has_actions = bool(output.tool_calls)
        has_goals = bool(output.goal_updates)

        if has_speech or has_actions or has_goals:
            # Stay active — there's engagement
            self._current_delay = self._active_delay
        else:
            # Drift toward idle
            self._current_delay = min(
                self._current_delay * 1.5,
                self._idle_delay,
            )

    def get_state(self) -> dict:
        """Get the current state for inspection or checkpointing."""
        return {
            "running": self._running,
            "booted": self._booted,
            "cycle_count": self.cycle_count,
            "current_delay": self._current_delay,
            "metrics": self._metrics.to_dict(),
            "stream": self._stream.to_dict(),
            "model_is_placeholder": self._model.is_placeholder,
        }
