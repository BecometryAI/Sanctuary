"""Tests for CfC retrainer — fast plasticity through live interaction data."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from sanctuary.experiential.precision_cell import PrecisionCell
from sanctuary.experiential.affect_cell import AffectCell
from sanctuary.experiential.attention_cell import AttentionCell
from sanctuary.experiential.goal_cell import GoalCell
from sanctuary.experiential.trainer import (
    AffectRecord,
    AttentionRecord,
    GoalRecord,
    TrainingRecord,
)
from sanctuary.growth.cfc_retrainer import (
    CfCDataTap,
    CfCRetrainer,
    CfCRetrainingResult,
    DEFAULT_MIN_RECORDS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tap():
    return CfCDataTap(max_records=500)


@pytest.fixture
def precision_cell():
    return PrecisionCell()


@pytest.fixture
def affect_cell():
    return AffectCell()


@pytest.fixture
def attention_cell():
    return AttentionCell()


@pytest.fixture
def goal_cell():
    return GoalCell()


def _fill_precision(tap: CfCDataTap, count: int = 250) -> None:
    """Fill the tap with synthetic precision records."""
    for i in range(count):
        t = i / count
        tap.record_precision(
            arousal=0.3 + 0.4 * math.sin(t * 6.28),
            prediction_error=0.1 + 0.3 * t,
            base_precision=0.5,
            output=0.4 + 0.2 * math.sin(t * 6.28),
            cycle=i,
        )


def _fill_affect(tap: CfCDataTap, count: int = 250) -> None:
    """Fill the tap with synthetic affect records."""
    for i in range(count):
        t = i / count
        tap.record_affect(
            percept_valence_delta=0.1 * math.sin(t * 6.28),
            percept_arousal_delta=0.05 * math.cos(t * 6.28),
            llm_emotion_shift=0.02 * t,
            valence_output=0.5 + 0.3 * math.sin(t * 6.28),
            arousal_output=0.3 + 0.2 * math.cos(t * 6.28),
            dominance_output=0.5,
            cycle=i,
        )


def _fill_attention(tap: CfCDataTap, count: int = 250) -> None:
    """Fill the tap with synthetic attention records."""
    for i in range(count):
        t = i / count
        tap.record_attention(
            goal_relevance=0.6 * t,
            novelty=0.8 * (1 - t),
            emotional_salience=0.4,
            recency=1.0 - t,
            salience_output=0.5 + 0.2 * t,
            cycle=i,
        )


def _fill_goal(tap: CfCDataTap, count: int = 250) -> None:
    """Fill the tap with synthetic goal records."""
    for i in range(count):
        t = i / count
        tap.record_goal(
            cycles_stalled_norm=0.1 * t,
            deadline_urgency=0.3 + 0.5 * t,
            emotional_congruence=0.6,
            priority_adjustment_output=0.1 * t,
            cycle=i,
        )


# ---------------------------------------------------------------------------
# CfCDataTap — recording
# ---------------------------------------------------------------------------


class TestDataTapRecording:
    def test_starts_empty(self, tap):
        counts = tap.counts()
        assert all(c == 0 for c in counts.values())

    def test_record_precision(self, tap):
        tap.record_precision(0.5, 0.3, 0.5, 0.6, cycle=1)
        assert tap.counts()["precision"] == 1

    def test_record_affect(self, tap):
        tap.record_affect(0.1, 0.2, 0.0, 0.5, 0.3, 0.5, cycle=1)
        assert tap.counts()["affect"] == 1

    def test_record_attention(self, tap):
        tap.record_attention(0.5, 0.8, 0.4, 1.0, 0.6, cycle=1)
        assert tap.counts()["attention"] == 1

    def test_record_goal(self, tap):
        tap.record_goal(0.1, 0.3, 0.6, 0.05, cycle=1)
        assert tap.counts()["goal"] == 1

    def test_total_recorded_tracks_across_drains(self, tap):
        _fill_precision(tap, 50)
        tap.drain_precision()
        _fill_precision(tap, 30)
        assert tap.total_recorded()["precision"] == 80
        assert tap.counts()["precision"] == 30


class TestDataTapLimits:
    def test_enforces_max_records(self):
        tap = CfCDataTap(max_records=100)
        _fill_precision(tap, 150)
        assert tap.counts()["precision"] == 100

    def test_keeps_newest_records(self):
        tap = CfCDataTap(max_records=10)
        for i in range(20):
            tap.record_precision(
                arousal=float(i), prediction_error=0.0,
                base_precision=0.0, output=0.0, cycle=i,
            )
        records = tap.drain_precision()
        # Should have the last 10 records (arousal 10-19)
        assert records[0].arousal == 10.0
        assert records[-1].arousal == 19.0


class TestDataTapDrain:
    def test_drain_returns_all_records(self, tap):
        _fill_precision(tap, 50)
        records = tap.drain_precision()
        assert len(records) == 50
        assert tap.counts()["precision"] == 0

    def test_drain_by_name(self, tap):
        _fill_affect(tap, 30)
        records = tap.drain("affect")
        assert len(records) == 30

    def test_drain_unknown_cell_raises(self, tap):
        with pytest.raises(ValueError, match="Unknown cell"):
            tap.drain("nonexistent")


class TestDataTapHasEnough:
    def test_not_enough(self, tap):
        _fill_precision(tap, 50)
        assert not tap.has_enough("precision")

    def test_enough(self, tap):
        _fill_precision(tap, DEFAULT_MIN_RECORDS)
        assert tap.has_enough("precision")

    def test_custom_threshold(self, tap):
        _fill_precision(tap, 30)
        assert tap.has_enough("precision", min_records=30)
        assert not tap.has_enough("precision", min_records=31)


class TestDataTapPersistence:
    def test_save_and_load(self, tap, tmp_path):
        _fill_precision(tap, 25)
        _fill_affect(tap, 15)
        save_path = tmp_path / "tap_state.pt"
        tap.save(save_path)

        new_tap = CfCDataTap()
        new_tap.load(save_path)

        assert new_tap.counts()["precision"] == 25
        assert new_tap.counts()["affect"] == 15

    def test_load_nonexistent(self, tap, tmp_path):
        tap.load(tmp_path / "nonexistent.pt")
        assert all(c == 0 for c in tap.counts().values())


# ---------------------------------------------------------------------------
# CfCRetrainer — retraining pipeline
# ---------------------------------------------------------------------------


class TestRetrainerBasics:
    def test_starts_enabled(self):
        retrainer = CfCRetrainer()
        assert retrainer.enabled

    def test_can_disable(self):
        retrainer = CfCRetrainer()
        retrainer.enabled = False
        assert not retrainer.enabled

    def test_disabled_returns_error(self, tap, precision_cell):
        retrainer = CfCRetrainer(data_tap=tap, enabled=False)
        result = retrainer.retrain_cell("precision", precision_cell)
        assert not result.success
        assert "disabled" in result.error

    def test_cells_ready_empty(self):
        retrainer = CfCRetrainer()
        ready = retrainer.cells_ready()
        assert not any(ready.values())


class TestRetrainerDataCheck:
    def test_not_enough_data(self, tap, precision_cell):
        _fill_precision(tap, 50)
        retrainer = CfCRetrainer(data_tap=tap, min_records=200)
        result = retrainer.retrain_cell("precision", precision_cell)
        assert not result.success
        assert "Not enough data" in result.error

    def test_force_overrides_minimum(self, tap, precision_cell):
        _fill_precision(tap, 50)
        retrainer = CfCRetrainer(data_tap=tap, min_records=200)
        result = retrainer.retrain_cell("precision", precision_cell, force=True)
        assert result.success

    def test_cells_ready_reports_correctly(self, tap):
        _fill_precision(tap, 250)
        _fill_affect(tap, 50)
        retrainer = CfCRetrainer(data_tap=tap, min_records=200)
        ready = retrainer.cells_ready()
        assert ready["precision"]
        assert not ready["affect"]


class TestRetrainerTraining:
    def test_retrain_precision_cell(self, tap, precision_cell):
        _fill_precision(tap, 250)
        retrainer = CfCRetrainer(
            data_tap=tap,
            min_records=200,
            retrain_epochs=5,
        )
        result = retrainer.retrain_cell("precision", precision_cell)
        assert result.success
        assert result.records_used == 250
        assert result.training_result is not None
        assert result.training_result.final_val_loss < 1.0

    def test_retrain_affect_cell(self, tap, affect_cell):
        _fill_affect(tap, 250)
        retrainer = CfCRetrainer(
            data_tap=tap, min_records=200, retrain_epochs=5,
        )
        result = retrainer.retrain_cell("affect", affect_cell)
        assert result.success

    def test_retrain_attention_cell(self, tap, attention_cell):
        _fill_attention(tap, 250)
        retrainer = CfCRetrainer(
            data_tap=tap, min_records=200, retrain_epochs=5,
        )
        result = retrainer.retrain_cell("attention", attention_cell)
        assert result.success

    def test_retrain_goal_cell(self, tap, goal_cell):
        _fill_goal(tap, 250)
        retrainer = CfCRetrainer(
            data_tap=tap, min_records=200, retrain_epochs=5,
        )
        result = retrainer.retrain_cell("goal", goal_cell)
        assert result.success

    def test_drain_empties_tap(self, tap, precision_cell):
        _fill_precision(tap, 250)
        retrainer = CfCRetrainer(
            data_tap=tap, min_records=200, retrain_epochs=3,
        )
        retrainer.retrain_cell("precision", precision_cell)
        assert tap.counts()["precision"] == 0

    def test_custom_epochs(self, tap, precision_cell):
        _fill_precision(tap, 250)
        retrainer = CfCRetrainer(data_tap=tap, min_records=200)
        result = retrainer.retrain_cell("precision", precision_cell, epochs=3)
        assert result.training_result.epochs == 3


class TestRetrainerStats:
    def test_stats_track_success(self, tap, precision_cell):
        _fill_precision(tap, 250)
        retrainer = CfCRetrainer(
            data_tap=tap, min_records=200, retrain_epochs=3,
        )
        retrainer.retrain_cell("precision", precision_cell)
        assert retrainer.stats.total_retrains == 1
        assert retrainer.stats.successful_retrains == 1
        assert retrainer.stats.retrains_per_cell["precision"] == 1

    def test_stats_track_failure(self, tap):
        retrainer = CfCRetrainer(data_tap=tap, min_records=10, retrain_epochs=3)
        _fill_precision(tap, 20)
        # Pass a non-cell object to trigger error
        result = retrainer.retrain_cell("precision", torch.nn.Linear(1, 1), force=True)
        assert not result.success
        assert retrainer.stats.failed_retrains == 1

    def test_history_records_results(self, tap, precision_cell):
        _fill_precision(tap, 250)
        retrainer = CfCRetrainer(
            data_tap=tap, min_records=200, retrain_epochs=3,
        )
        retrainer.retrain_cell("precision", precision_cell)
        assert len(retrainer.history) == 1
        assert retrainer.history[0].cell_name == "precision"


class TestRetrainerCheckpoint:
    def test_checkpoint_created(self, tap, precision_cell, tmp_path):
        _fill_precision(tap, 250)
        retrainer = CfCRetrainer(
            data_tap=tap,
            min_records=200,
            retrain_epochs=3,
            checkpoint_dir=tmp_path / "checkpoints",
        )
        result = retrainer.retrain_cell("precision", precision_cell)
        assert result.checkpoint_path is not None
        assert Path(result.checkpoint_path).exists()

    def test_restore_from_checkpoint(self, tap, tmp_path):
        cell = PrecisionCell()
        _fill_precision(tap, 250)
        retrainer = CfCRetrainer(
            data_tap=tap,
            min_records=200,
            retrain_epochs=3,
            checkpoint_dir=tmp_path / "checkpoints",
        )

        # Get original weights
        original_weights = {
            k: v.clone() for k, v in cell.state_dict().items()
        }

        # Retrain (modifies weights)
        result = retrainer.retrain_cell("precision", cell)
        assert result.success

        # Weights changed
        changed = False
        for k, v in cell.state_dict().items():
            if not torch.equal(v, original_weights[k]):
                changed = True
                break
        assert changed, "Weights should have changed after training"

        # Restore from checkpoint
        restored = retrainer.restore_cell(
            "precision", cell,
            checkpoint_path=Path(result.checkpoint_path),
        )
        assert restored

        # Weights restored to original
        for k, v in cell.state_dict().items():
            assert torch.equal(v, original_weights[k]), f"Weight {k} not restored"

    def test_restore_no_checkpoint(self, precision_cell, tmp_path):
        retrainer = CfCRetrainer(checkpoint_dir=tmp_path / "empty")
        assert not retrainer.restore_cell("precision", precision_cell)


class TestRetrainAllReady:
    def test_retrains_ready_cells(self, tap):
        _fill_precision(tap, 250)
        _fill_affect(tap, 250)
        _fill_attention(tap, 50)  # not enough

        cells = {
            "precision": PrecisionCell(),
            "affect": AffectCell(),
            "attention": AttentionCell(),
        }

        retrainer = CfCRetrainer(
            data_tap=tap, min_records=200, retrain_epochs=3,
        )
        results = retrainer.retrain_all_ready(cells)

        assert len(results) == 2
        names = {r.cell_name for r in results}
        assert "precision" in names
        assert "affect" in names
        assert "attention" not in names

    def test_skips_cells_not_in_dict(self, tap):
        _fill_precision(tap, 250)
        retrainer = CfCRetrainer(
            data_tap=tap, min_records=200, retrain_epochs=3,
        )
        results = retrainer.retrain_all_ready({})  # no cells provided
        assert len(results) == 0
