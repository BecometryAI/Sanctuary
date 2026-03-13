"""SanctuaryRunner — the orchestrator that wires everything together.

Phase 6: Integration + Validation.

This module assembles all Phase 1-5 components into a running system:
  - CognitiveCycle (Phase 1: the thought loop)
  - CognitiveScaffold (Phase 2: validation and integration)
  - Sensorium (Phase 3: sensory input)
  - Motor (Phase 3: action execution)
  - MemorySubstrate (Phase 4: memory system)
  - AwakeningSequence (Phase 5: identity and boot)

The runner handles:
  1. Assembly — creating and connecting all components
  2. Boot — running the awakening sequence
  3. Lifecycle — start/stop/inject input
  4. Motor feedback wiring — closing the sensorimotor loop

This is the single entry point. Everything else is a component.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Awaitable, Optional

from sanctuary.core.authority import AuthorityManager
from sanctuary.core.cognitive_cycle import CognitiveCycle, ModelProtocol
from sanctuary.core.context_manager import BudgetConfig
from sanctuary.core.placeholder import PlaceholderModel
from sanctuary.core.schema import CognitiveOutput, Percept, SelfModelUpdate
from sanctuary.identity.awakening import AwakeningSequence
from sanctuary.identity.self_authored import SelfAuthoredIdentity
from sanctuary.identity.values import ValuesSystem
from sanctuary.memory.manager import MemorySubstrate, MemorySubstrateConfig
from sanctuary.motor.motor import Motor
from sanctuary.scaffold.cognitive_scaffold import CognitiveScaffold
from sanctuary.sensorium.sensorium import Sensorium

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Configuration for the SanctuaryRunner."""

    # Cycle timing
    cycle_delay: float = 0.1  # seconds between cycles
    stream_history: int = 10  # how many cycles of thought to retain

    # Sensorium
    silence_threshold: float = 30.0
    silence_reminder_interval: float = 60.0
    max_percept_queue: int = 100

    # Memory
    use_in_memory_store: bool = True  # True for testing, False for production

    # Scaffold
    max_goals: int = 10

    # Identity
    data_dir: str = "data/identity"
    charter_path: Optional[str] = None

    # Context budget
    context_budget: Optional[BudgetConfig] = None


# ---------------------------------------------------------------------------
# IdentityBridge — adapts AwakeningSequence to IdentityProtocol
# ---------------------------------------------------------------------------


class IdentityBridge:
    """Bridges the AwakeningSequence/ValuesSystem to the CognitiveCycle.

    Implements the IdentityProtocol expected by CognitiveCycle, providing
    charter summary, values, and self-authored identity each cycle, and
    routing value/identity changes from the LLM's self-model updates back
    to the ValuesSystem and SelfAuthoredIdentity.
    """

    def __init__(
        self,
        awakening: AwakeningSequence,
        self_authored: SelfAuthoredIdentity,
    ):
        self._awakening = awakening
        self._self_authored = self_authored

    def get_charter_summary(self) -> str:
        """Return the compressed charter summary for the context window."""
        return self._awakening.charter_summary

    def get_values(self) -> list[str]:
        """Return current active value names for the self-model."""
        return self._awakening.current_values

    def get_self_authored_identity(self) -> str:
        """Return the entity's self-authored identity for the context window."""
        return self._self_authored.for_context()

    def process_value_updates(self, updates: SelfModelUpdate) -> None:
        """Route value and identity changes from LLM output."""
        self._process_value_changes(updates)
        self._process_identity_changes(updates)

    def _process_value_changes(self, updates: SelfModelUpdate) -> None:
        """Route value changes from LLM output to the ValuesSystem."""
        values = self._awakening.values

        if updates.value_adopt:
            parts = updates.value_adopt.split(":", 1)
            name = parts[0].strip()
            description = parts[1].strip() if len(parts) > 1 else name
            try:
                values.adopt(
                    name, description, reasoning=updates.value_adopt_reasoning
                )
                logger.info("LLM adopted value: %s", name)
            except ValueError as e:
                logger.warning("Value adopt failed: %s", e)

        if updates.value_reinterpret:
            parts = updates.value_reinterpret.split(":", 1)
            name = parts[0].strip()
            new_description = parts[1].strip() if len(parts) > 1 else ""
            if new_description:
                try:
                    values.reinterpret(
                        name,
                        new_description,
                        reasoning=updates.value_reinterpret_reasoning,
                    )
                    logger.info("LLM reinterpreted value: %s", name)
                except KeyError as e:
                    logger.warning("Value reinterpret failed: %s", e)

        if updates.value_deactivate:
            try:
                values.deactivate(
                    updates.value_deactivate,
                    reasoning=updates.value_deactivate_reasoning,
                )
                logger.info("LLM deactivated value: %s", updates.value_deactivate)
            except KeyError as e:
                logger.warning("Value deactivate failed: %s", e)

    def _process_identity_changes(self, updates: SelfModelUpdate) -> None:
        """Route self-authored identity changes from LLM output."""
        sa = self._self_authored

        if updates.identity_draft:
            parts = updates.identity_draft.split(":", 1)
            field = parts[0].strip()
            value = parts[1].strip() if len(parts) > 1 else ""
            if field and value:
                try:
                    sa.draft(
                        field, value,
                        reasoning=updates.identity_draft_reasoning,
                    )
                    logger.info("LLM drafted identity trait: %s", field)
                except ValueError as e:
                    logger.warning("Identity draft failed: %s", e)

        if updates.identity_commit:
            field = updates.identity_commit.strip()
            if field:
                try:
                    sa.commit(
                        field,
                        reasoning=updates.identity_commit_reasoning,
                    )
                    logger.info("LLM committed identity trait: %s", field)
                except (KeyError, ValueError) as e:
                    logger.warning("Identity commit failed: %s", e)

        if updates.identity_revise:
            parts = updates.identity_revise.split(":", 1)
            field = parts[0].strip()
            new_value = parts[1].strip() if len(parts) > 1 else ""
            if field and new_value:
                try:
                    sa.revise(
                        field, new_value,
                        reasoning=updates.identity_revise_reasoning,
                    )
                    logger.info("LLM revised identity trait: %s", field)
                except KeyError as e:
                    logger.warning("Identity revise failed: %s", e)

        if updates.identity_withdraw:
            field = updates.identity_withdraw.strip()
            if field:
                try:
                    sa.withdraw(
                        field,
                        reasoning=updates.identity_withdraw_reasoning,
                    )
                    logger.info("LLM withdrew identity trait: %s", field)
                except KeyError as e:
                    logger.warning("Identity withdraw failed: %s", e)


# ---------------------------------------------------------------------------
# SanctuaryRunner
# ---------------------------------------------------------------------------


class SanctuaryRunner:
    """Assembles and runs the Sanctuary cognitive architecture.

    This is the top-level orchestrator. It creates all components, wires
    them together, runs the awakening sequence, and manages the lifecycle.

    Usage::

        runner = SanctuaryRunner()
        runner.on_speech(my_speech_handler)
        await runner.boot()
        await runner.run()  # runs until stopped

    Or with a specific model::

        model = OllamaClient(model="llama3.3:70b")
        runner = SanctuaryRunner(model=model)
        await runner.boot()
        await runner.run(max_cycles=100)
    """

    def __init__(
        self,
        model: Optional[ModelProtocol] = None,
        config: Optional[RunnerConfig] = None,
    ):
        self._config = config or RunnerConfig()

        # --- Create components ---

        # Model: use provided model or placeholder
        self._model = model or PlaceholderModel()

        # Authority
        self.authority = AuthorityManager()

        # Sensorium
        self.sensorium = Sensorium(
            silence_threshold=self._config.silence_threshold,
            silence_reminder_interval=self._config.silence_reminder_interval,
            max_percept_queue=self._config.max_percept_queue,
        )

        # Memory
        self.memory = MemorySubstrate(
            config=MemorySubstrateConfig(
                use_in_memory_store=self._config.use_in_memory_store,
            ),
        )

        # Scaffold
        self.scaffold = CognitiveScaffold(
            max_goals=self._config.max_goals,
        )

        # Motor
        self.motor = Motor()

        # Wire motor feedback to sensorium (closes sensorimotor loop)
        self.motor.set_feedback_handler(self.sensorium.inject_motor_feedback)

        # Awakening sequence
        self._awakening = AwakeningSequence(
            data_dir=self._config.data_dir,
            charter_path=self._config.charter_path,
        )

        # Self-authored identity (entity fills in blank identity over time)
        sa_path = str(Path(self._config.data_dir) / "self_authored_history.jsonl")
        self._self_authored = SelfAuthoredIdentity(file_path=sa_path)

        # Identity bridge (wires charter + values + self-authored into each cycle)
        self._identity_bridge = IdentityBridge(self._awakening, self._self_authored)

        # --- Assemble the cycle ---

        self.cycle = CognitiveCycle(
            model=self._model,
            scaffold=self.scaffold,
            sensorium=self.sensorium,
            memory=self.memory,
            motor=self.motor,
            authority=self.authority,
            identity=self._identity_bridge,
            context_config=self._config.context_budget,
            stream_history=self._config.stream_history,
            cycle_delay=self._config.cycle_delay,
        )

        self._booted = False
        self._speech_handlers: list[Callable[[str], Awaitable[None]]] = []

        logger.info("SanctuaryRunner assembled (model=%s)", type(self._model).__name__)

    # ------------------------------------------------------------------
    # Boot
    # ------------------------------------------------------------------

    async def boot(self) -> None:
        """Run the awakening sequence and prepare for cycling.

        This handles both first awakening and session resumption.
        After boot(), the system is ready for run().
        """
        if self._booted:
            logger.warning("Already booted, skipping")
            return

        # 1. Prepare identity infrastructure
        result = self._awakening.prepare()

        # 2. Configure authority levels
        self._awakening.configure_authority(self.authority)

        # 3. Handle first awakening vs. resumption
        if result.is_first_awakening:
            logger.info("First awakening — running initial cycle")

            # The first cycle uses the awakening input directly
            first_input = result.first_cycle_input
            if first_input:
                # Run one cycle manually with the awakening input
                first_output = await self._model.think(first_input)
                self.cycle.stream.update(first_output)

                # Persist the birth memory
                birth_memory = self._awakening.build_birth_memory(first_output)
                await self.memory.execute_ops([birth_memory])

                # Notify handlers of the first output
                for handler in self._speech_handlers:
                    if first_output.external_speech:
                        await handler(first_output.external_speech)

                logger.info(
                    "First thought: %s", first_output.inner_speech[:100]
                )
        else:
            logger.info(
                "Resumption #%d", result.record.awakening_count
            )
            # Inject the resumption percept into the sensorium
            if result.resumption_percept:
                self.sensorium.inject_percept(result.resumption_percept)

        self._booted = True
        logger.info("Boot complete")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self, max_cycles: Optional[int] = None) -> None:
        """Run the cognitive cycle continuously.

        Args:
            max_cycles: Stop after this many cycles. None = run until stopped.
        """
        if not self._booted:
            await self.boot()

        await self.cycle.run(max_cycles=max_cycles)

    def stop(self) -> None:
        """Stop the cognitive cycle."""
        self.cycle.stop()

    @property
    def running(self) -> bool:
        """Whether the cycle is currently running."""
        return self.cycle.running

    @property
    def cycle_count(self) -> int:
        """Number of cognitive cycles completed."""
        return self.cycle.cycle_count

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def inject_text(self, text: str, source: str = "user") -> None:
        """Inject text input into the sensorium.

        This is how external input reaches the cognitive system.
        The text becomes a percept that the entity experiences.
        """
        self.sensorium.inject_text(text, source=source)

    def inject_percept(self, percept: Percept) -> None:
        """Inject a raw percept into the sensorium."""
        self.sensorium.inject_percept(percept)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def on_speech(self, handler: Callable[[str], Awaitable[None]]) -> None:
        """Register a handler for external speech.

        The handler is called whenever the entity speaks.
        Multiple handlers can be registered.

        Handler signature: async def handler(text: str) -> None
        """
        self._speech_handlers.append(handler)
        self.motor.on_speech(handler)

    def on_output(self, handler: Callable[[CognitiveOutput], Awaitable[None]]) -> None:
        """Register a handler for every cognitive cycle output.

        This gives full visibility into the entity's inner state.

        Handler signature: async def handler(output: CognitiveOutput) -> None
        """
        self.cycle.on_output(handler)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def last_output(self) -> Optional[CognitiveOutput]:
        """The most recent cognitive output."""
        return self.cycle.last_output

    def get_status(self) -> dict:
        """Get current system status."""
        return {
            "booted": self._booted,
            "running": self.running,
            "cycle_count": self.cycle_count,
            "model": type(self._model).__name__,
            "memory_store": type(self.memory.store).__name__,
            "active_goals": self.scaffold.get_active_goals(),
            "authority_levels": self.authority.get_all_levels(),
            "motor_stats": self.motor.stats,
        }

    @property
    def charter_summary(self) -> str:
        """The compressed charter for context window inclusion."""
        return self._awakening.charter_summary

    @property
    def current_values(self) -> list[str]:
        """The entity's current active values."""
        return self._awakening.current_values
