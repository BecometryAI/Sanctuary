#!/usr/bin/env python3
"""Train the CfC precision cell on collected scaffold data, then validate.

This script completes the scaffold-to-CfC handoff for precision weighting:
    1. Load training records collected by collect_training_data.py
    2. Train the CfC precision cell (supervised, MSE loss)
    3. Validate: CfC output ≈ scaffold output on held-out sequences
    4. Run novel scenarios to test generalization beyond the heuristic
    5. Save the trained cell for use in ExperientialManager

The cell is tiny (~1K parameters). Training takes seconds on CPU.

Usage:
    python scripts/train_precision_cell.py [--data data/training/precision_records.pt]
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanctuary.experiential.precision_cell import PrecisionCell
from sanctuary.experiential.trainer import CfCTrainer, DataCollector
from sanctuary.mind.cognitive_core.precision_weighting import PrecisionWeighting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def validate_against_scaffold(
    cell: PrecisionCell,
    n_tests: int = 100,
    seed: int = 99,
) -> dict:
    """Compare trained CfC cell outputs against scaffold heuristic.

    Runs both systems on identical inputs and measures agreement.
    The CfC cell should approximate the scaffold closely on familiar
    input ranges, while potentially showing richer temporal dynamics.

    Returns:
        Dict with max_error, mean_error, agreement_rate (within 0.1),
        and per-test details.
    """
    rng = random.Random(seed)
    pw = PrecisionWeighting()
    cell.reset_hidden()

    errors = []
    details = []

    for i in range(n_tests):
        arousal = rng.uniform(0.0, 0.8)
        pred_error = rng.uniform(0.0, 0.7)

        scaffold_out = pw.compute_precision(
            percept=f"validation_{i}",
            emotional_state={"arousal": arousal},
            prediction_error=pred_error,
        )
        cfc_out = cell.step(
            arousal=arousal,
            prediction_error=pred_error,
            base_precision=0.5,
        )

        err = abs(scaffold_out - cfc_out)
        errors.append(err)
        details.append({
            "arousal": arousal,
            "pred_error": pred_error,
            "scaffold": scaffold_out,
            "cfc": cfc_out,
            "error": err,
        })

    mean_error = sum(errors) / len(errors)
    max_error = max(errors)
    within_01 = sum(1 for e in errors if e < 0.1) / len(errors)
    within_005 = sum(1 for e in errors if e < 0.05) / len(errors)

    return {
        "mean_error": mean_error,
        "max_error": max_error,
        "agreement_01": within_01,   # % within 0.1 of scaffold
        "agreement_005": within_005, # % within 0.05 of scaffold
        "n_tests": n_tests,
        "details": details,
    }


def test_temporal_dynamics(cell: PrecisionCell) -> dict:
    """Verify the CfC cell captures temporal patterns the heuristic cannot.

    The scaffold is memoryless — same inputs always produce the same output.
    The CfC cell has hidden state, so its output should differ based on
    recent history. This is the "temporal thickness" that justifies the
    cell's existence.

    Test: Run two sequences with identical final inputs but different
    histories. If the CfC outputs differ, it has learned temporal context.
    """
    # Sequence A: calm history, then moderate input
    cell.reset_hidden()
    for _ in range(20):
        cell.step(arousal=0.1, prediction_error=0.05, base_precision=0.5)
    calm_then_moderate = cell.step(arousal=0.4, prediction_error=0.3, base_precision=0.5)

    # Sequence B: agitated history, then same moderate input
    cell.reset_hidden()
    for _ in range(20):
        cell.step(arousal=0.7, prediction_error=0.5, base_precision=0.5)
    agitated_then_moderate = cell.step(arousal=0.4, prediction_error=0.3, base_precision=0.5)

    temporal_difference = abs(calm_then_moderate - agitated_then_moderate)
    has_temporal_context = temporal_difference > 0.01

    # Sequence C: gradual arousal increase (should show momentum)
    cell.reset_hidden()
    rising = []
    for i in range(30):
        t = i / 30
        p = cell.step(arousal=0.1 + 0.5 * t, prediction_error=0.2, base_precision=0.5)
        rising.append(p)

    # Check for smooth trajectory (not just repeating the heuristic)
    diffs = [abs(rising[i+1] - rising[i]) for i in range(len(rising)-1)]
    smooth = max(diffs) < 0.3  # No wild jumps

    return {
        "calm_then_moderate": calm_then_moderate,
        "agitated_then_moderate": agitated_then_moderate,
        "temporal_difference": temporal_difference,
        "has_temporal_context": has_temporal_context,
        "rising_trajectory_smooth": smooth,
        "rising_range": (min(rising), max(rising)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and validate CfC precision cell",
    )
    parser.add_argument(
        "--data", type=str,
        default="data/training/precision_records.pt",
        help="Path to collected training records",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/training/precision_cell_trained.pt",
        help="Path to save trained cell",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs (default: 100)",
    )
    parser.add_argument(
        "--seq-len", type=int, default=10,
        help="Sequence length for training windows (default: 10)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Training batch size (default: 16)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 0.001)",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    output_path = Path(args.output)

    # --- Load data ---
    if not data_path.exists():
        print(f"No training data found at {data_path}")
        print("Run collect_training_data.py first.")
        sys.exit(1)

    collector = DataCollector()
    collector.load(data_path)
    print(f"Loaded {collector.count} training records from {data_path}")

    # --- Train ---
    print(f"\nTraining CfC precision cell...")
    print(f"  epochs={args.epochs}, seq_len={args.seq_len}, batch_size={args.batch_size}, lr={args.lr}")

    cell = PrecisionCell()
    trainer = CfCTrainer(
        cell,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )
    result = trainer.train(collector.records, epochs=args.epochs)

    print(f"\n--- Training Results ---")
    print(f"  Final train loss:  {result.final_train_loss:.6f}")
    print(f"  Final val loss:    {result.final_val_loss:.6f}")
    print(f"  Best val loss:     {result.best_val_loss:.6f} (epoch {result.best_epoch})")
    print(f"  Train samples:     {result.num_train_samples}")
    print(f"  Val samples:       {result.num_val_samples}")

    # --- Validate vs scaffold ---
    print(f"\n--- Scaffold Validation ---")
    val = validate_against_scaffold(cell, n_tests=200)
    print(f"  Mean error:        {val['mean_error']:.4f}")
    print(f"  Max error:         {val['max_error']:.4f}")
    print(f"  Within 0.10:       {val['agreement_01']:.1%}")
    print(f"  Within 0.05:       {val['agreement_005']:.1%}")

    passed = val["mean_error"] < 0.15 and val["agreement_01"] > 0.5
    print(f"  Validation:        {'PASSED' if passed else 'NEEDS MORE TRAINING'}")

    # --- Temporal dynamics ---
    print(f"\n--- Temporal Dynamics ---")
    temporal = test_temporal_dynamics(cell)
    print(f"  Calm→moderate:     {temporal['calm_then_moderate']:.4f}")
    print(f"  Agitated→moderate: {temporal['agitated_then_moderate']:.4f}")
    print(f"  Temporal diff:     {temporal['temporal_difference']:.4f}")
    print(f"  Has temporal ctx:  {temporal['has_temporal_context']}")
    print(f"  Rising smooth:     {temporal['rising_trajectory_smooth']}")
    print(f"  Rising range:      [{temporal['rising_range'][0]:.4f}, {temporal['rising_range'][1]:.4f}]")

    # --- Save ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cell.save(output_path)
    print(f"\nTrained cell saved to {output_path}")

    # --- Overall verdict ---
    print(f"\n{'='*50}")
    if passed and temporal["has_temporal_context"]:
        print("READY: Cell approximates scaffold AND shows temporal dynamics.")
        print("It can be promoted from SCAFFOLD_ONLY to LLM_ADVISES.")
    elif passed:
        print("PARTIAL: Cell approximates scaffold but shows limited temporal dynamics.")
        print("Consider more training data with varied temporal patterns.")
    else:
        print("NOT READY: Cell does not yet approximate scaffold well enough.")
        print("Try: more epochs, more training data, or adjusted learning rate.")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
