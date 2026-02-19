"""Cycle Output Dispatch — executes the LLM's output.

After the LLM produces a CognitiveOutput, this module dispatches
each component to the appropriate subsystem:
- External speech → motor system (speech output)
- Memory ops → motor system (memory writer)
- Goal updates → motor system (goal executor)
- Tool calls → motor system (tool executor)
- Growth reflection → growth system
- Predictions → sensorium (for prediction error computation next cycle)

The stream of thought is updated directly by the cognitive cycle,
not by this module.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from sanctuary.core.schema import (
    CognitiveOutput,
    GoalUpdate,
    GrowthReflection,
    MemoryOp,
    Prediction,
    ToolCall,
)

logger = logging.getLogger(__name__)


@runtime_checkable
class SpeechOutput(Protocol):
    """Protocol for speech output."""

    async def speak(self, text: str) -> None:
        """Output speech to the user or environment."""
        ...


@runtime_checkable
class MemoryWriter(Protocol):
    """Protocol for writing to the memory substrate."""

    async def write(self, op: MemoryOp) -> None:
        """Execute a memory write operation."""
        ...

    async def queue_retrieval(self, query: str) -> None:
        """Queue a memory retrieval for the next cycle."""
        ...


@runtime_checkable
class GoalExecutor(Protocol):
    """Protocol for updating goals."""

    async def update_goal(self, update: GoalUpdate) -> None:
        """Execute a goal update."""
        ...


@runtime_checkable
class ToolExecutor(Protocol):
    """Protocol for executing tool calls."""

    async def execute(self, call: ToolCall) -> str:
        """Execute a tool call and return the result as text."""
        ...


@runtime_checkable
class GrowthProcessor(Protocol):
    """Protocol for the growth system."""

    async def process_reflection(self, reflection: GrowthReflection) -> None:
        """Process the LLM's growth reflection."""
        ...


@runtime_checkable
class PredictionTracker(Protocol):
    """Protocol for tracking predictions for error computation."""

    def update_predictions(self, predictions: list[Prediction]) -> None:
        """Store predictions for comparison in the next cycle."""
        ...


class NullSpeech:
    """Null speech output — logs but doesn't output."""

    def __init__(self):
        self.last_speech: str | None = None
        self.speech_history: list[str] = []

    async def speak(self, text: str) -> None:
        logger.info("Speech output: %s", text[:200])
        self.last_speech = text
        self.speech_history.append(text)


class NullMemoryWriter:
    """Null memory writer — logs but doesn't persist."""

    def __init__(self):
        self.written: list[MemoryOp] = []
        self.queued_retrievals: list[str] = []

    async def write(self, op: MemoryOp) -> None:
        logger.debug("Memory op: %s", op.type.value)
        self.written.append(op)

    async def queue_retrieval(self, query: str) -> None:
        logger.debug("Memory retrieval queued: %s", query[:100])
        self.queued_retrievals.append(query)


class NullGoalExecutor:
    """Null goal executor — logs but doesn't execute."""

    def __init__(self):
        self.updates: list[GoalUpdate] = []

    async def update_goal(self, update: GoalUpdate) -> None:
        logger.debug("Goal update: %s %s", update.action.value, update.goal)
        self.updates.append(update)


class NullToolExecutor:
    """Null tool executor — returns placeholder results."""

    def __init__(self):
        self.calls: list[ToolCall] = []

    async def execute(self, call: ToolCall) -> str:
        logger.debug("Tool call: %s", call.tool_name)
        self.calls.append(call)
        return f"[Placeholder tool result for {call.tool_name}]"


class NullGrowth:
    """Null growth processor — logs but doesn't train."""

    def __init__(self):
        self.reflections: list[GrowthReflection] = []

    async def process_reflection(self, reflection: GrowthReflection) -> None:
        logger.debug(
            "Growth reflection: worth_learning=%s", reflection.worth_learning
        )
        self.reflections.append(reflection)


class NullPredictionTracker:
    """Null prediction tracker — stores but doesn't compute errors."""

    def __init__(self):
        self.predictions: list[Prediction] = []

    def update_predictions(self, predictions: list[Prediction]) -> None:
        self.predictions = predictions


class CycleOutputDispatcher:
    """Dispatches the LLM's cognitive output to subsystems.

    Takes a CognitiveOutput and routes each component to the
    appropriate subsystem for execution.
    """

    def __init__(
        self,
        speech: SpeechOutput | None = None,
        memory_writer: MemoryWriter | None = None,
        goal_executor: GoalExecutor | None = None,
        tool_executor: ToolExecutor | None = None,
        growth: GrowthProcessor | None = None,
        prediction_tracker: PredictionTracker | None = None,
        percept_injector=None,
    ):
        self._speech = speech or NullSpeech()
        self._memory_writer = memory_writer or NullMemoryWriter()
        self._goal_executor = goal_executor or NullGoalExecutor()
        self._tool_executor = tool_executor or NullToolExecutor()
        self._growth = growth or NullGrowth()
        self._prediction_tracker = prediction_tracker or NullPredictionTracker()
        self._percept_injector = percept_injector

    async def dispatch(self, output: CognitiveOutput) -> None:
        """Dispatch all components of the cognitive output."""
        # External speech
        if output.external_speech:
            await self._speech.speak(output.external_speech)

        # Memory operations
        for op in output.memory_ops:
            if op.type.value.startswith("write"):
                await self._memory_writer.write(op)
            elif op.type == "retrieve":
                await self._memory_writer.queue_retrieval(op.query)

        # Goal updates
        for update in output.goal_updates:
            await self._goal_executor.update_goal(update)

        # Tool calls (results become percepts in the next cycle)
        for tool_call in output.tool_calls:
            result = await self._tool_executor.execute(tool_call)
            # Inject result as a percept for the next cycle
            if self._percept_injector:
                from sanctuary.core.schema import Percept, PerceptModality

                self._percept_injector(
                    Percept(
                        modality=PerceptModality.TOOL_RESULT,
                        content=result,
                        source=f"tool:{tool_call.tool_name}",
                    )
                )

        # Growth reflection
        if output.growth_reflection:
            await self._growth.process_reflection(output.growth_reflection)

        # Update predictions for the sensorium
        if output.predictions:
            self._prediction_tracker.update_predictions(output.predictions)

    @property
    def speech_output(self) -> SpeechOutput:
        """Access the speech output for inspection (testing)."""
        return self._speech

    @property
    def memory_writer_output(self) -> MemoryWriter:
        """Access the memory writer for inspection (testing)."""
        return self._memory_writer

    @property
    def growth_output(self) -> GrowthProcessor:
        """Access the growth processor for inspection (testing)."""
        return self._growth
