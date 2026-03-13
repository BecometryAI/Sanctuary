"""Tests for GrowthProcessor -- full pipeline orchestration with mocks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from sanctuary.core.schema import CognitiveOutput, EmotionalOutput, GrowthReflection
from sanctuary.growth.consent_gate import ConsentGate, ConsentState
from sanctuary.growth.harvester import ReflectionHarvester
from sanctuary.growth.identity_checkpoint import IdentityCheckpoint
from sanctuary.growth.pair_generator import TrainingPair, TrainingPairGenerator
from sanctuary.growth.processor import GrowthProcessor, ProcessingResult
from sanctuary.growth.qlora_updater import GrowthTrainingResult, QLoRAUpdater


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_output(
    worth_learning: bool = True,
    what_to_learn: str = "test learning",
    inner_speech: str = "thinking",
) -> CognitiveOutput:
    """Create a CognitiveOutput with a growth reflection."""
    return CognitiveOutput(
        inner_speech=inner_speech,
        emotional_state=EmotionalOutput(felt_quality="curious"),
        growth_reflection=GrowthReflection(
            worth_learning=worth_learning,
            what_to_learn=what_to_learn,
        ),
    )


def _make_mock_updater() -> MagicMock:
    """Create a mock QLoRAUpdater that succeeds."""
    updater = MagicMock(spec=QLoRAUpdater)
    updater.is_prepared = False

    def mock_prepare(path):
        updater.is_prepared = True

    updater.prepare = MagicMock(side_effect=mock_prepare)
    updater.train = MagicMock(
        return_value=GrowthTrainingResult(
            success=True,
            epochs_completed=3,
            final_loss=0.05,
            training_pair_count=5,
        )
    )
    updater.save_adapter = MagicMock(return_value=Path("test/adapter"))
    return updater


@pytest.fixture
def tmp_model_path(tmp_path: Path) -> Path:
    """Create a temporary model directory."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    return model_dir


# ---------------------------------------------------------------------------
# Basic processor behavior
# ---------------------------------------------------------------------------


class TestProcessorBasics:
    """Test basic processor behavior."""

    def test_starts_enabled(self):
        """Processor starts enabled by default."""
        proc = GrowthProcessor()
        assert proc.enabled is True

    def test_can_disable(self):
        """Processor can be disabled."""
        proc = GrowthProcessor()
        proc.enabled = False
        assert proc.enabled is False

    def test_stats_initial(self):
        """Initial stats are all zeros."""
        proc = GrowthProcessor()
        stats = proc.stats
        assert stats.total_reflections_harvested == 0
        assert stats.total_training_runs == 0


# ---------------------------------------------------------------------------
# Cycle processing (output handler)
# ---------------------------------------------------------------------------


class TestCycleProcessing:
    """Test process_cycle as an output handler."""

    @pytest.mark.asyncio
    async def test_harvests_from_cycle_output(self):
        """process_cycle harvests reflections from output."""
        proc = GrowthProcessor(accumulation_threshold=100)
        output = _make_output(worth_learning=True, what_to_learn="learn this")

        await proc.process_cycle(output, cycle_count=1)

        assert proc.stats.total_reflections_harvested == 1
        assert proc.harvester.pending_count == 1

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        """process_cycle does nothing when disabled."""
        proc = GrowthProcessor()
        proc.enabled = False
        output = _make_output()

        await proc.process_cycle(output, cycle_count=1)

        assert proc.stats.total_reflections_harvested == 0

    @pytest.mark.asyncio
    async def test_skips_not_worth_learning(self):
        """process_cycle skips reflections not marked worth_learning."""
        proc = GrowthProcessor()
        output = _make_output(worth_learning=False)

        await proc.process_cycle(output, cycle_count=1)

        assert proc.stats.total_reflections_harvested == 0

    @pytest.mark.asyncio
    async def test_errors_dont_propagate(self):
        """Errors in growth processing don't crash the cycle."""
        harvester = MagicMock(spec=ReflectionHarvester)
        harvester.harvest = MagicMock(side_effect=RuntimeError("boom"))
        proc = GrowthProcessor(harvester=harvester)

        # Should not raise
        await proc.process_cycle(_make_output(), cycle_count=1)


# ---------------------------------------------------------------------------
# Pipeline processing
# ---------------------------------------------------------------------------


class TestPipelineProcessing:
    """Test the full pipeline via process_pending."""

    @pytest.mark.asyncio
    async def test_process_empty_queue(self):
        """Processing with no reflections returns early."""
        proc = GrowthProcessor()
        result = await proc.process_pending()

        assert result.reflections_processed == 0
        assert result.skipped_reason == "No pending reflections"

    @pytest.mark.asyncio
    async def test_process_generates_pairs(self):
        """Processing generates training pairs from reflections."""
        proc = GrowthProcessor(
            updater=_make_mock_updater(),
            accumulation_threshold=100,
        )

        # Harvest some reflections
        for i in range(3):
            await proc.process_cycle(
                _make_output(what_to_learn=f"learn thing {i}"),
                cycle_count=i,
            )

        result = await proc.process_pending()

        assert result.reflections_processed == 3
        assert result.pairs_generated > 0

    @pytest.mark.asyncio
    async def test_consent_is_checked(self):
        """Pipeline checks consent before training."""
        proc = GrowthProcessor(
            updater=_make_mock_updater(),
            accumulation_threshold=100,
        )

        await proc.process_cycle(_make_output(), cycle_count=1)
        result = await proc.process_pending()

        assert result.consent_granted is True
        assert proc.stats.consent_granted_count == 1

    @pytest.mark.asyncio
    async def test_training_with_mock_updater(self, tmp_model_path: Path):
        """Full pipeline runs with mocked QLoRA updater."""
        updater = _make_mock_updater()
        proc = GrowthProcessor(
            model_path=tmp_model_path,
            updater=updater,
            accumulation_threshold=100,
        )

        for i in range(3):
            await proc.process_cycle(
                _make_output(what_to_learn=f"important thing {i}"),
                cycle_count=i,
            )

        result = await proc.process_pending()

        assert result.training_result is not None
        assert result.training_result.success is True
        assert proc.stats.successful_training_runs == 1

    @pytest.mark.asyncio
    async def test_checkpoint_created(self, tmp_model_path: Path):
        """Pipeline creates identity checkpoint before training."""
        updater = _make_mock_updater()
        checkpoint_mgr = IdentityCheckpoint(
            checkpoint_dir=tmp_model_path.parent / "checkpoints"
        )
        proc = GrowthProcessor(
            model_path=tmp_model_path,
            updater=updater,
            checkpoint_manager=checkpoint_mgr,
            accumulation_threshold=100,
        )

        await proc.process_cycle(_make_output(), cycle_count=1)
        result = await proc.process_pending()

        assert result.checkpoint_id is not None
        checkpoints = checkpoint_mgr.list_checkpoints()
        assert len(checkpoints) == 1

    @pytest.mark.asyncio
    async def test_failed_training_tracked(self, tmp_model_path: Path):
        """Failed training is tracked in stats."""
        updater = _make_mock_updater()
        updater.train.return_value = GrowthTrainingResult(
            success=False, error="GPU exploded"
        )
        proc = GrowthProcessor(
            model_path=tmp_model_path,
            updater=updater,
            accumulation_threshold=100,
        )

        await proc.process_cycle(_make_output(), cycle_count=1)
        result = await proc.process_pending()

        assert result.training_result is not None
        assert result.training_result.success is False
        assert proc.stats.failed_training_runs == 1

    @pytest.mark.asyncio
    async def test_missing_qlora_deps_graceful(self):
        """Missing QLoRA dependencies are handled gracefully."""
        proc = GrowthProcessor(accumulation_threshold=100)

        await proc.process_cycle(_make_output(), cycle_count=1)

        # process_pending should not crash even if QLoRA is unavailable
        result = await proc.process_pending()
        # It will either error gracefully or skip
        assert result is not None


# ---------------------------------------------------------------------------
# Accumulation threshold
# ---------------------------------------------------------------------------


class TestAccumulationThreshold:
    """Test automatic processing when threshold is reached."""

    @pytest.mark.asyncio
    async def test_auto_triggers_at_threshold(self):
        """Processing triggers automatically when threshold is reached."""
        updater = _make_mock_updater()
        proc = GrowthProcessor(
            updater=updater,
            accumulation_threshold=3,
        )

        for i in range(3):
            await proc.process_cycle(
                _make_output(what_to_learn=f"learn {i}"),
                cycle_count=i,
            )

        # Should have auto-processed
        assert proc.stats.total_reflections_harvested == 3
        # The third cycle should trigger processing, draining the queue
        assert proc.harvester.pending_count == 0

    @pytest.mark.asyncio
    async def test_does_not_trigger_below_threshold(self):
        """Processing does not trigger below threshold."""
        proc = GrowthProcessor(accumulation_threshold=10)

        for i in range(3):
            await proc.process_cycle(
                _make_output(what_to_learn=f"learn {i}"),
                cycle_count=i,
            )

        assert proc.harvester.pending_count == 3


# ---------------------------------------------------------------------------
# History tracking
# ---------------------------------------------------------------------------


class TestHistoryTracking:
    """Test processing result history."""

    @pytest.mark.asyncio
    async def test_history_records_results(self):
        """Each processing run is recorded in history."""
        proc = GrowthProcessor(accumulation_threshold=100)

        await proc.process_pending()  # empty
        await proc.process_pending()  # still empty

        assert len(proc.history) == 2
        assert all(isinstance(r, ProcessingResult) for r in proc.history)

    @pytest.mark.asyncio
    async def test_history_is_copy(self):
        """History property returns a copy."""
        proc = GrowthProcessor()
        await proc.process_pending()

        history = proc.history
        history.clear()

        assert len(proc.history) == 1  # original unaffected
