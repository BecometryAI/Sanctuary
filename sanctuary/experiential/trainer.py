"""CfC cell trainer — learns from scaffold heuristic data.

The scaffold runs first, logging input/output pairs. The trainer uses
those pairs to train CfC cells via supervised learning. Once a cell
approximates the heuristic, it can generalize beyond it.

Training workflow:
    1. Run scaffold for N cycles, collecting data via DataCollector
    2. Create training sequences from collected data
    3. Train CfC cell on sequences (MSE loss, Adam optimizer)
    4. Validate: CfC output ≈ scaffold output on held-out data
    5. Wire CfC cell into cognitive cycle (ExperientialManager)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from sanctuary.experiential.precision_cell import PrecisionCell

logger = logging.getLogger(__name__)

# Training defaults
DEFAULT_EPOCHS = 50
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 16
DEFAULT_SEQ_LEN = 10      # group data points into sequences of this length
DEFAULT_TRAIN_SPLIT = 0.8


@dataclass
class TrainingRecord:
    """A single input/output pair from the scaffold heuristic."""

    arousal: float
    prediction_error: float
    base_precision: float
    precision_output: float  # the scaffold's computed precision


@dataclass
class TrainingResult:
    """Result of a training run."""

    epochs: int
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    best_epoch: int
    num_train_samples: int
    num_val_samples: int


class DataCollector:
    """Collects training data from the scaffold's precision weighting system.

    Attach this to the existing PrecisionWeighting instance to passively
    log every computation as a training record.
    """

    def __init__(self):
        self._records: list[TrainingRecord] = []

    def record(
        self,
        arousal: float,
        prediction_error: float,
        base_precision: float,
        precision_output: float,
    ):
        """Log one scaffold precision computation."""
        self._records.append(
            TrainingRecord(
                arousal=arousal,
                prediction_error=prediction_error,
                base_precision=base_precision,
                precision_output=precision_output,
            )
        )

    @property
    def count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[TrainingRecord]:
        return list(self._records)

    def clear(self):
        self._records.clear()

    def save(self, path: Path):
        """Save collected records to disk."""
        data = [
            {
                "arousal": r.arousal,
                "prediction_error": r.prediction_error,
                "base_precision": r.base_precision,
                "precision_output": r.precision_output,
            }
            for r in self._records
        ]
        torch.save(data, path)
        logger.info("Saved %d training records to %s", len(data), path)

    def load(self, path: Path):
        """Load records from disk."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        self._records = [TrainingRecord(**d) for d in data]
        logger.info("Loaded %d training records from %s", len(self._records), path)


class CfCTrainer:
    """Trains CfC cells from scaffold data.

    Takes a DataCollector's records, creates sequential training data,
    and trains a PrecisionCell to approximate the scaffold's behavior.
    """

    def __init__(
        self,
        cell: PrecisionCell,
        learning_rate: float = DEFAULT_LR,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seq_len: int = DEFAULT_SEQ_LEN,
        train_split: float = DEFAULT_TRAIN_SPLIT,
    ):
        self.cell = cell
        self.lr = learning_rate
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.train_split = train_split

    def prepare_data(
        self, records: list[TrainingRecord]
    ) -> tuple[TensorDataset, TensorDataset]:
        """Convert records into sequential training/validation datasets.

        Groups consecutive records into sequences of length seq_len.
        This preserves temporal ordering, which is critical for CfC cells.
        """
        if len(records) < self.seq_len:
            raise ValueError(
                f"Need at least {self.seq_len} records, got {len(records)}"
            )

        inputs = []
        targets = []

        # Create overlapping sequences
        for i in range(len(records) - self.seq_len + 1):
            seq = records[i : i + self.seq_len]
            inp = [[r.arousal, r.prediction_error, r.base_precision] for r in seq]
            tgt = [[r.precision_output] for r in seq]
            inputs.append(inp)
            targets.append(tgt)

        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        targets_t = torch.tensor(targets, dtype=torch.float32)

        # Split train/val
        n = len(inputs_t)
        split_idx = int(n * self.train_split)

        train_ds = TensorDataset(inputs_t[:split_idx], targets_t[:split_idx])
        val_ds = TensorDataset(inputs_t[split_idx:], targets_t[split_idx:])

        return train_ds, val_ds

    def train(
        self,
        records: list[TrainingRecord],
        epochs: int = DEFAULT_EPOCHS,
    ) -> TrainingResult:
        """Train the precision cell on scaffold data.

        Returns a TrainingResult with loss metrics.
        """
        train_ds, val_ds = self.prepare_data(records)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False
        )

        optimizer = torch.optim.Adam(self.cell.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_epoch = 0
        final_train_loss = 0.0
        final_val_loss = 0.0

        self.cell.train()
        for epoch in range(epochs):
            # Training
            epoch_loss = 0.0
            n_batches = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                _, loss = self.cell.forward_training(inputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            final_train_loss = epoch_loss / max(n_batches, 1)

            # Validation
            self.cell.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    _, loss = self.cell.forward_training(inputs, targets)
                    val_loss += loss.item()
                    n_val += 1
            final_val_loss = val_loss / max(n_val, 1)
            self.cell.train()

            if final_val_loss < best_val_loss:
                best_val_loss = final_val_loss
                best_epoch = epoch

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d — train_loss=%.6f, val_loss=%.6f",
                    epoch + 1,
                    epochs,
                    final_train_loss,
                    final_val_loss,
                )

        self.cell.eval()

        result = TrainingResult(
            epochs=epochs,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            num_train_samples=len(train_ds),
            num_val_samples=len(val_ds),
        )

        logger.info(
            "Training complete: best_val_loss=%.6f at epoch %d",
            best_val_loss,
            best_epoch,
        )
        return result
