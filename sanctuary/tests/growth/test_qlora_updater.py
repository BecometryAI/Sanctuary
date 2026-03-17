"""Tests for QLoRA updater — weight updates via QLoRA fine-tuning.

These tests validate the QLoRAUpdater without requiring peft/transformers
to be installed. The actual HuggingFace training path is tested through
the processor integration tests with mocked dependencies. Here we test
configuration, state management, and error handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sanctuary.growth.pair_generator import TrainingPair
from sanctuary.growth.qlora_updater import (
    GrowthTrainingResult,
    QLoRAConfig,
    QLoRAUpdater,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestQLoRAConfig:
    def test_default_rank(self):
        config = QLoRAConfig()
        assert config.rank == 8

    def test_default_alpha(self):
        config = QLoRAConfig()
        assert config.alpha == 16

    def test_default_target_modules(self):
        config = QLoRAConfig()
        assert config.target_modules == ["q_proj", "v_proj"]

    def test_default_dropout(self):
        config = QLoRAConfig()
        assert config.dropout == 0.05

    def test_custom_config(self):
        config = QLoRAConfig(rank=16, alpha=32, dropout=0.1)
        assert config.rank == 16
        assert config.alpha == 32
        assert config.dropout == 0.1


class TestTrainingConfig:
    def test_default_epochs(self):
        config = TrainingConfig()
        assert config.epochs == 3

    def test_default_lr(self):
        config = TrainingConfig()
        assert config.learning_rate == 2e-4

    def test_default_batch_size(self):
        config = TrainingConfig()
        assert config.batch_size == 1

    def test_custom_config(self):
        config = TrainingConfig(epochs=5, learning_rate=1e-3, batch_size=4)
        assert config.epochs == 5
        assert config.learning_rate == 1e-3
        assert config.batch_size == 4


# ---------------------------------------------------------------------------
# GrowthTrainingResult
# ---------------------------------------------------------------------------


class TestTrainingResult:
    def test_default_not_successful(self):
        result = GrowthTrainingResult()
        assert not result.success

    def test_records_pair_count(self):
        result = GrowthTrainingResult(training_pair_count=10)
        assert result.training_pair_count == 10

    def test_has_timestamps(self):
        result = GrowthTrainingResult()
        assert result.started_at is not None

    def test_successful_result(self):
        result = GrowthTrainingResult(
            success=True,
            epochs_completed=3,
            final_loss=0.05,
            training_pair_count=10,
            adapter_path="/some/path",
        )
        assert result.success
        assert result.final_loss == 0.05
        assert result.adapter_path == "/some/path"


# ---------------------------------------------------------------------------
# QLoRAUpdater — state management
# ---------------------------------------------------------------------------


class TestUpdaterState:
    def test_starts_unprepared(self):
        updater = QLoRAUpdater()
        assert not updater.is_prepared

    def test_configs_accessible(self):
        qlora_config = QLoRAConfig(rank=16)
        training_config = TrainingConfig(epochs=5)
        updater = QLoRAUpdater(
            qlora_config=qlora_config,
            training_config=training_config,
        )
        assert updater.qlora_config.rank == 16
        assert updater.training_config.epochs == 5

    def test_train_without_prepare_raises(self):
        updater = QLoRAUpdater()
        pairs = [
            TrainingPair(
                system_prompt="test",
                user_input="hello",
                assistant_response="world",
            )
        ]
        with pytest.raises(RuntimeError, match="not prepared"):
            updater.train(pairs)

    def test_train_empty_pairs(self):
        updater = QLoRAUpdater()
        # Bypass prepare check manually
        updater._prepared = True
        result = updater.train([])
        assert not result.success
        assert "No training pairs" in result.error

    def test_save_without_prepare_raises(self):
        updater = QLoRAUpdater()
        with pytest.raises(RuntimeError, match="not prepared"):
            updater.save_adapter(Path("/tmp/test"))

    def test_merge_without_prepare_raises(self):
        updater = QLoRAUpdater()
        with pytest.raises(RuntimeError, match="not prepared"):
            updater.merge_and_save(Path("/tmp/test"))

    def test_prepare_nonexistent_model_raises(self, tmp_path):
        updater = QLoRAUpdater()
        # This will raise ImportError if peft not installed,
        # or FileNotFoundError if it is
        with pytest.raises((ImportError, FileNotFoundError)):
            updater.prepare(tmp_path / "nonexistent_model")
