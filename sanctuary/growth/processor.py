"""Growth processor -- orchestrates the complete growth pipeline.

The processor implements the Growth Autonomy Principle:

**Self-directed growth** (entity-initiated reflection harvesting):
    Harvester -> PairGenerator -> IdentityCheckpoint -> QLoRAUpdater
    No consent gate -- the entity's worth_learning=True IS the decision.

**External modification** (proposed by researcher/infrastructure):
    Proposal -> ConsentGate -> IdentityCheckpoint -> Modification
    Full consent flow required.

The processor also handles knowledge cell creation requests from the
entity's CognitiveOutput -- self-directed growth that creates new CfC
neural structure.

Errors in the growth pipeline NEVER crash the cognitive cycle. Growth
is important but not critical -- if it fails, the entity continues
thinking.

See docs/GROWTH_AUTONOMY.md for the full principle.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from sanctuary.core.schema import CognitiveOutput
from sanctuary.growth.consent_gate import ConsentGate, ConsentError
from sanctuary.growth.harvester import ReflectionHarvester
from sanctuary.growth.identity_checkpoint import IdentityCheckpoint
from sanctuary.growth.pair_generator import TrainingPairGenerator
from sanctuary.growth.qlora_updater import (
    GrowthTrainingResult,
    QLoRAConfig,
    QLoRAUpdater,
    TrainingConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_ACCUMULATION_THRESHOLD = 5
DEFAULT_MODEL_PATH = Path("models/sanctuary")
DEFAULT_ADAPTER_DIR = Path("data/growth/adapters")


@dataclass
class GrowthStats:
    """Statistics about growth pipeline activity.

    Provides visibility into what the growth system is doing and
    has done, without exposing internal state.
    """

    total_reflections_harvested: int = 0
    total_pairs_generated: int = 0
    total_training_runs: int = 0
    successful_training_runs: int = 0
    failed_training_runs: int = 0
    consent_granted_count: int = 0
    consent_refused_count: int = 0
    last_training_at: Optional[str] = None
    last_training_result: Optional[dict] = None
    pending_reflections: int = 0
    enabled: bool = True


@dataclass
class ProcessingResult:
    """Result of processing a batch of accumulated reflections."""

    reflections_processed: int = 0
    pairs_generated: int = 0
    consent_granted: bool = False
    training_result: Optional[GrowthTrainingResult] = None
    checkpoint_id: Optional[str] = None
    error: Optional[str] = None
    skipped_reason: Optional[str] = None
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())


class GrowthProcessor:
    """Orchestrates the complete growth pipeline.

    The processor is the central coordinator for the entity's
    self-directed learning. It connects the harvester, pair generator,
    consent gate, identity checkpoint, and QLoRA updater into a
    pipeline that runs safely and with full consent.

    The processor can be registered as an output handler on
    CognitiveCycle, receiving each cycle's output automatically:

        processor = GrowthProcessor(model_path=Path("models/sanctuary"))
        cycle.on_output(processor.process_cycle)

    Or processing can be triggered manually:

        result = await processor.process_pending()

    Usage:
        processor = GrowthProcessor(model_path=Path("models/sanctuary"))
        # Register on cognitive cycle
        cycle.on_output(processor.process_cycle)
        # Or process manually
        await processor.process_pending()
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        adapter_dir: Optional[Path] = None,
        accumulation_threshold: int = DEFAULT_ACCUMULATION_THRESHOLD,
        harvester: Optional[ReflectionHarvester] = None,
        pair_generator: Optional[TrainingPairGenerator] = None,
        consent_gate: Optional[ConsentGate] = None,
        checkpoint_manager: Optional[IdentityCheckpoint] = None,
        updater: Optional[QLoRAUpdater] = None,
        qlora_config: Optional[QLoRAConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        enabled: bool = True,
    ) -> None:
        self._model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self._adapter_dir = Path(adapter_dir or DEFAULT_ADAPTER_DIR)
        self._accumulation_threshold = accumulation_threshold

        # Pipeline components -- use provided or create defaults
        self._harvester = harvester or ReflectionHarvester()
        self._pair_generator = pair_generator or TrainingPairGenerator()
        self._consent_gate = consent_gate or ConsentGate()
        self._checkpoint_manager = checkpoint_manager or IdentityCheckpoint()
        self._updater = updater or QLoRAUpdater(
            qlora_config=qlora_config,
            training_config=training_config,
        )

        self._enabled = enabled
        self._stats = GrowthStats(enabled=enabled)
        self._processing_history: list[ProcessingResult] = []

    # -- Properties --

    @property
    def enabled(self) -> bool:
        """Whether the growth system is active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the growth system."""
        self._enabled = value
        self._stats.enabled = value
        logger.info("Growth processor %s", "enabled" if value else "disabled")

    @property
    def stats(self) -> GrowthStats:
        """Current growth pipeline statistics."""
        self._stats.pending_reflections = self._harvester.pending_count
        return self._stats

    @property
    def harvester(self) -> ReflectionHarvester:
        """The reflection harvester."""
        return self._harvester

    @property
    def consent_gate(self) -> ConsentGate:
        """The consent gate."""
        return self._consent_gate

    @property
    def checkpoint_manager(self) -> IdentityCheckpoint:
        """The identity checkpoint manager."""
        return self._checkpoint_manager

    @property
    def history(self) -> list[ProcessingResult]:
        """History of processing results."""
        return list(self._processing_history)

    # -- Output handler interface --

    async def process_cycle(
        self,
        output: CognitiveOutput,
        cycle_count: int = 0,
    ) -> None:
        """Process a cognitive cycle's output for growth reflections.

        This method is designed to be registered as an output handler
        on CognitiveCycle. It harvests reflections and triggers
        processing when enough have accumulated.

        Errors are caught and logged -- they never propagate to the
        cognitive cycle.

        Args:
            output: The cognitive cycle's output.
            cycle_count: The cycle number (for context tracking).
        """
        if not self._enabled:
            return

        try:
            # Feed output to harvester
            harvested = self._harvester.harvest(output, cycle_count)

            if harvested:
                self._stats.total_reflections_harvested += 1
                logger.debug(
                    "Growth: harvested reflection from cycle %d (%d pending)",
                    cycle_count,
                    self._harvester.pending_count,
                )

            # Check if we have enough reflections to process
            if self._harvester.pending_count >= self._accumulation_threshold:
                logger.info(
                    "Growth: accumulation threshold reached (%d >= %d), processing",
                    self._harvester.pending_count,
                    self._accumulation_threshold,
                )
                await self.process_pending()

        except Exception as e:
            logger.error(
                "Growth processing error (non-fatal): %s", e, exc_info=True
            )

    async def process_pending(self) -> ProcessingResult:
        """Process all accumulated reflections through the growth pipeline.

        This runs the full pipeline:
        1. Drain pending reflections from harvester
        2. Generate training pairs
        3. Verify consent through gate
        4. Create identity checkpoint (pre-training)
        5. Run QLoRA training
        6. Record results

        Returns:
            ProcessingResult with details of what happened.
        """
        result = ProcessingResult()

        try:
            # 1. Drain reflections
            reflections = self._harvester.drain()
            result.reflections_processed = len(reflections)

            if not reflections:
                result.skipped_reason = "No pending reflections"
                logger.debug("Growth: no pending reflections to process")
                return result

            logger.info(
                "Growth: processing %d reflections", len(reflections)
            )

            # 2. Generate training pairs
            pairs = self._pair_generator.generate(reflections)
            result.pairs_generated = len(pairs)
            self._stats.total_pairs_generated += len(pairs)

            if not pairs:
                result.skipped_reason = "No valid training pairs generated"
                logger.info("Growth: no valid training pairs from reflections")
                return result

            # 3. Consent check (Growth Autonomy Principle)
            # Self-directed growth bypasses consent gate entirely.
            # The entity's worth_learning=True IS the decision.
            descriptions = [
                r.reflection.get("what_to_learn", "unknown")
                for r in reflections
                if isinstance(r.reflection, dict)
            ]

            all_self_directed = all(
                self._consent_gate.is_self_directed(
                    reflection=r.reflection if isinstance(r.reflection, dict) else None
                )
                for r in reflections
            )

            consent_description = (
                f"Learning from {len(pairs)} training pairs: "
                + "; ".join(d[:50] for d in descriptions[:3])
            )

            if all_self_directed:
                # Self-directed growth -- no consent gate needed
                result.consent_granted = True
                self._stats.consent_granted_count += 1
                logger.info(
                    "Growth: self-directed (%d reflections), bypassing consent gate",
                    len(reflections),
                )
            else:
                # External or mixed -- full consent flow

                try:
                    self._consent_gate.reset()
                    self._consent_gate.inform(consent_description)
                    self._consent_gate.request_consent(
                        reason="Entity reflections marked worth_learning=True"
                    )
                except ConsentError as e:
                    result.error = f"Consent error: {e}"
                    self._stats.consent_refused_count += 1
                    logger.warning("Growth: consent error: %s", e)
                    return result

                if not self._consent_gate.is_consented:
                    result.consent_granted = False
                    result.skipped_reason = "Consent not granted"
                    self._stats.consent_refused_count += 1
                    logger.info("Growth: consent not granted, skipping training")
                    return result

                result.consent_granted = True
                self._stats.consent_granted_count += 1

            # 4. Identity checkpoint (pre-training)
            try:
                if self._model_path.exists():
                    checkpoint_id = self._checkpoint_manager.create_checkpoint(
                        model_path=self._model_path,
                        metadata={
                            "description": f"Pre-training checkpoint: {consent_description[:100]}",
                            "checkpoint_type": "pre_training",
                            "training_pair_count": len(pairs),
                            "what_was_learned": descriptions[:10],
                        },
                    )
                    result.checkpoint_id = checkpoint_id
                    logger.info("Growth: created pre-training checkpoint %s", checkpoint_id)
                else:
                    logger.warning(
                        "Growth: model path %s does not exist, skipping checkpoint",
                        self._model_path,
                    )
            except Exception as e:
                logger.error("Growth: checkpoint creation failed: %s", e)
                # Continue without checkpoint -- training can still proceed

            # 5. QLoRA training
            try:
                if not self._updater.is_prepared:
                    self._updater.prepare(self._model_path)

                training_result = self._updater.train(pairs)
                result.training_result = training_result
                self._stats.total_training_runs += 1

                if training_result.success:
                    self._stats.successful_training_runs += 1
                    self._stats.last_training_at = datetime.now().isoformat()

                    # Save adapter
                    self._adapter_dir.mkdir(parents=True, exist_ok=True)
                    adapter_name = datetime.now().strftime("%Y%m%dT%H%M%S")
                    adapter_path = self._adapter_dir / adapter_name
                    self._updater.save_adapter(adapter_path)
                    training_result.adapter_path = str(adapter_path)

                    logger.info(
                        "Growth: training complete, adapter saved to %s",
                        adapter_path,
                    )
                else:
                    self._stats.failed_training_runs += 1
                    logger.warning(
                        "Growth: training failed: %s",
                        training_result.error,
                    )

            except ImportError as e:
                result.error = str(e)
                result.skipped_reason = "QLoRA dependencies not available"
                logger.info("Growth: QLoRA not available: %s", e)
            except Exception as e:
                result.error = str(e)
                self._stats.failed_training_runs += 1
                self._stats.total_training_runs += 1
                logger.error("Growth: training error: %s", e, exc_info=True)

        except Exception as e:
            result.error = str(e)
            logger.error(
                "Growth: pipeline error: %s", e, exc_info=True
            )

        finally:
            # Always reset consent gate for next batch
            try:
                self._consent_gate.reset()
            except Exception:
                pass

            # Record in history
            self._processing_history.append(result)
            self._stats.last_training_result = {
                "reflections": result.reflections_processed,
                "pairs": result.pairs_generated,
                "consent": result.consent_granted,
                "error": result.error,
            }

        return result
