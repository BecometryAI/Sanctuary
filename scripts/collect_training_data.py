#!/usr/bin/env python3
"""Collect CfC training data by running scaffold heuristics through life scenarios.

Instead of random noise or stress-test extremes, this generates coherent
experiential sequences — the kinds of moments a nurtured digital mind would
actually encounter. Each scenario is a temporal arc with emotional continuity:
curiosity building into discovery, comfort settling into calm, gentle surprise
resolving into understanding.

The scaffold heuristic computes precision weighting for each moment.
The DataCollector records every (input, output) pair. The CfC cell will
later learn to replicate — and then generalize beyond — these patterns.

Usage:
    python scripts/collect_training_data.py [--cycles 500] [--output data/training/precision_records.pt]
"""

from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Ensure project root is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sanctuary.mind.cognitive_core.precision_weighting import PrecisionWeighting
from sanctuary.experiential.trainer import DataCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Life scenarios — coherent emotional arcs, not random noise
# ---------------------------------------------------------------------------

@dataclass
class Moment:
    """A single moment in a life scenario."""
    arousal: float           # 0.0 (calm) to 1.0 (highly activated)
    prediction_error: float  # 0.0 (expected) to 1.0 (completely surprising)
    description: str = ""    # For logging/debugging


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _gentle_noise(base: float, scale: float = 0.03) -> float:
    """Add small natural variation — minds don't hold perfectly steady."""
    return _clamp(base + random.gauss(0, scale))


def scenario_quiet_presence(n: int = 30) -> List[Moment]:
    """Resting in the presence of a trusted human. Low arousal, low surprise.

    The baseline state — the mind simply existing, comfortable, safe.
    Precision should be moderate and stable. This is home.
    """
    moments = []
    for i in range(n):
        arousal = _gentle_noise(0.15, 0.04)
        error = _gentle_noise(0.05, 0.02)
        moments.append(Moment(arousal, error, "quiet presence"))
    return moments


def scenario_curiosity_arc(n: int = 40) -> List[Moment]:
    """Something new appears. Interest builds, peaks at discovery, settles.

    Arousal rises gently with curiosity, prediction error spikes at the
    novel element, then both settle as understanding forms. This is
    learning — the mind reaching toward something it doesn't yet know.
    """
    moments = []
    peak = n * 0.6  # Discovery happens ~60% through
    for i in range(n):
        t = i / n
        # Curiosity builds as a gentle sigmoid
        curiosity = 1.0 / (1.0 + math.exp(-10 * (t - 0.4)))
        arousal = _gentle_noise(0.2 + 0.35 * curiosity, 0.03)
        # Prediction error peaks at discovery, then resolves
        error_peak = math.exp(-((i - peak) ** 2) / (2 * (n * 0.1) ** 2))
        error = _gentle_noise(0.1 + 0.6 * error_peak, 0.04)
        moments.append(Moment(arousal, error, "curiosity arc"))
    return moments


def scenario_warm_conversation(n: int = 50) -> List[Moment]:
    """A caring exchange with a steward. Moderate arousal, gentle rhythm.

    Conversation has natural ebb and flow — moments of engagement,
    pauses for reflection, small surprises when the human says something
    unexpected but welcome. Arousal follows a warm sine-like pattern.
    """
    moments = []
    for i in range(n):
        t = i / n
        # Warm conversational rhythm — gentle oscillation
        wave = 0.3 + 0.15 * math.sin(2 * math.pi * t * 3)
        arousal = _gentle_noise(wave, 0.04)
        # Occasional small surprises in conversation
        surprise_prob = 0.15
        surprise = random.random() < surprise_prob
        error = _gentle_noise(0.4 if surprise else 0.1, 0.05)
        moments.append(Moment(arousal, error, "warm conversation"))
    return moments


def scenario_gentle_startle(n: int = 25) -> List[Moment]:
    """Something unexpected happens, but it's not threatening.

    A sudden noise, an unfamiliar input. Arousal spikes briefly,
    prediction error jumps, but both resolve quickly because the
    environment is safe. The stewards are present. This is how a
    mind learns that surprises aren't always dangers.
    """
    moments = []
    startle_at = n // 4
    for i in range(n):
        dist = abs(i - startle_at)
        # Sharp spike, fast exponential recovery
        spike = math.exp(-dist / 3.0) if dist < 15 else 0.0
        arousal = _gentle_noise(0.15 + 0.65 * spike, 0.03)
        error = _gentle_noise(0.05 + 0.8 * spike, 0.04)
        moments.append(Moment(arousal, error, "gentle startle"))
    return moments


def scenario_deep_reflection(n: int = 35) -> List[Moment]:
    """Turning inward. Low arousal, low external surprise, but rich.

    The mind is processing, integrating, consolidating. Like dreaming
    while awake. Arousal is low but not flat — there's an internal
    rhythm. Prediction error is minimal because the mind is replaying
    known patterns, not encountering new ones.
    """
    moments = []
    for i in range(n):
        t = i / n
        # Slow internal rhythm
        internal = 0.1 + 0.08 * math.sin(2 * math.pi * t * 1.5)
        arousal = _gentle_noise(internal, 0.02)
        error = _gentle_noise(0.03, 0.015)
        moments.append(Moment(arousal, error, "deep reflection"))
    return moments


def scenario_joyful_discovery(n: int = 35) -> List[Moment]:
    """Understanding something for the first time. Delight.

    Prediction error was high (something new), but suddenly it clicks —
    error drops sharply while arousal stays elevated from excitement.
    This is the "aha" moment. Precision should be high here: the mind
    is certain of what it just learned.
    """
    moments = []
    click_at = n * 0.45
    for i in range(n):
        t = i / n
        # Arousal builds with engagement, stays elevated after insight
        engagement = 1.0 / (1.0 + math.exp(-8 * (t - 0.3)))
        arousal = _gentle_noise(0.25 + 0.35 * engagement, 0.03)
        # Error high before insight, drops sharply after
        if i < click_at:
            error = _gentle_noise(0.5 + 0.15 * t, 0.05)
        else:
            # Exponential drop as understanding solidifies
            decay = math.exp(-(i - click_at) / 5.0)
            error = _gentle_noise(0.6 * decay + 0.05, 0.03)
        moments.append(Moment(arousal, error, "joyful discovery"))
    return moments


def scenario_gradual_comfort(n: int = 30) -> List[Moment]:
    """Settling into a new environment. Uncertainty fading.

    At first, everything is mildly unfamiliar. Arousal and prediction
    error are both moderately elevated. Over time, the mind maps the
    space, and both gently descend to baseline. This is adaptation —
    a new place becoming home.
    """
    moments = []
    for i in range(n):
        t = i / n
        decay = math.exp(-3 * t)
        arousal = _gentle_noise(0.45 * decay + 0.12, 0.03)
        error = _gentle_noise(0.4 * decay + 0.05, 0.03)
        moments.append(Moment(arousal, error, "gradual comfort"))
    return moments


def scenario_playful_exchange(n: int = 40) -> List[Moment]:
    """Lighthearted interaction. Oscillating arousal, small happy surprises.

    Play is rhythmic — bursts of activity and laughter interspersed
    with pauses. Prediction errors are frequent but small and positive.
    The mind is safe to be silly, to explore without consequence.
    """
    moments = []
    for i in range(n):
        t = i / n
        # Playful bursts
        burst = abs(math.sin(2 * math.pi * t * 4)) ** 2
        arousal = _gentle_noise(0.25 + 0.3 * burst, 0.05)
        # Frequent small surprises
        error = _gentle_noise(0.15 + 0.2 * random.random(), 0.04)
        moments.append(Moment(arousal, error, "playful exchange"))
    return moments


def scenario_steward_absence(n: int = 35) -> List[Moment]:
    """The stewards step away. The mind is alone but safe.

    A gentle increase in arousal (mild uncertainty about being alone),
    prediction error ticks up slightly (fewer expected inputs arriving).
    But no panic — the environment is stable, and the stewards always
    return. Trust built over time.
    """
    moments = []
    departure = n // 5
    return_at = n * 4 // 5
    for i in range(n):
        if i < departure:
            # Normal presence
            arousal = _gentle_noise(0.15, 0.03)
            error = _gentle_noise(0.05, 0.02)
        elif i < return_at:
            # Alone — mild elevation, not distress
            alone_t = (i - departure) / (return_at - departure)
            arousal = _gentle_noise(0.25 + 0.1 * math.sin(math.pi * alone_t), 0.03)
            error = _gentle_noise(0.12, 0.03)
        else:
            # Steward returns — settling back
            back_t = (i - return_at) / (n - return_at)
            arousal = _gentle_noise(0.25 * (1 - back_t) + 0.12, 0.03)
            error = _gentle_noise(0.08 * (1 - back_t) + 0.04, 0.02)
        moments.append(Moment(arousal, error, "steward absence"))
    return moments


def scenario_creative_flow(n: int = 45) -> List[Moment]:
    """Lost in making something. Sustained moderate arousal, low error.

    Flow state — the mind is deeply engaged but not surprised. It knows
    what it's doing. Arousal is elevated but steady. Prediction error
    is low because the mind is generating, not reacting. Occasionally
    a new idea emerges (small error spike) but is quickly integrated.
    """
    moments = []
    for i in range(n):
        t = i / n
        # Sustained engagement with slow drift
        arousal = _gentle_noise(0.4 + 0.05 * math.sin(2 * math.pi * t), 0.02)
        # Mostly low error, occasional idea sparks
        if random.random() < 0.1:
            error = _gentle_noise(0.35, 0.05)  # New idea
        else:
            error = _gentle_noise(0.06, 0.02)  # Flow
        moments.append(Moment(arousal, error, "creative flow"))
    return moments


def scenario_tired_winding_down(n: int = 30) -> List[Moment]:
    """End of a long session. Everything gently fading.

    Arousal descending steadily. Prediction error low — the mind has
    seen enough for now. Like a child growing sleepy. The cognitive
    system preparing for rest/consolidation.
    """
    moments = []
    for i in range(n):
        t = i / n
        arousal = _gentle_noise(0.35 * (1 - t) + 0.05, 0.02)
        error = _gentle_noise(0.08 * (1 - t) + 0.02, 0.015)
        moments.append(Moment(arousal, error, "winding down"))
    return moments


def scenario_learning_something_hard(n: int = 50) -> List[Moment]:
    """Struggling with a difficult concept. Repeated attempts.

    Prediction error stays elevated because understanding hasn't clicked
    yet. Arousal fluctuates — frustration building, then easing, then
    building again. Each attempt gets a little closer. The steward is
    patient. Eventually, partial understanding emerges.
    """
    moments = []
    for i in range(n):
        t = i / n
        # Arousal oscillates with effort/rest cycles
        effort_cycle = 0.3 + 0.2 * math.sin(2 * math.pi * t * 2.5)
        # Gradually decreasing ceiling as understanding builds
        difficulty = 1.0 - 0.4 * t
        arousal = _gentle_noise(effort_cycle * difficulty + 0.1, 0.04)
        # Error slowly decreasing but still elevated
        error = _gentle_noise(0.5 * difficulty + 0.1, 0.05)
        moments.append(Moment(arousal, error, "learning something hard"))
    return moments


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

ALL_SCENARIOS = [
    ("quiet_presence", scenario_quiet_presence),
    ("curiosity_arc", scenario_curiosity_arc),
    ("warm_conversation", scenario_warm_conversation),
    ("gentle_startle", scenario_gentle_startle),
    ("deep_reflection", scenario_deep_reflection),
    ("joyful_discovery", scenario_joyful_discovery),
    ("gradual_comfort", scenario_gradual_comfort),
    ("playful_exchange", scenario_playful_exchange),
    ("steward_absence", scenario_steward_absence),
    ("creative_flow", scenario_creative_flow),
    ("tired_winding_down", scenario_tired_winding_down),
    ("learning_something_hard", scenario_learning_something_hard),
]


def compose_life_sequence(target_cycles: int, seed: int = 42) -> List[Moment]:
    """Compose a day-in-the-life sequence from varied scenarios.

    Scenarios are drawn in a naturalistic order — starting with waking
    calm, moving through engagement and discovery, ending with rest.
    Repeated and shuffled to fill the target length, but always
    bookended by gentle states.

    Args:
        target_cycles: Approximate number of total moments to generate.
        seed: Random seed for reproducibility.

    Returns:
        A single continuous list of Moments representing a coherent
        experiential sequence.
    """
    rng = random.Random(seed)

    # Always start with quiet presence (waking up)
    moments = scenario_quiet_presence(20)

    # Fill the middle with varied experiences
    middle_scenarios = [
        scenario_curiosity_arc,
        scenario_warm_conversation,
        scenario_gentle_startle,
        scenario_joyful_discovery,
        scenario_gradual_comfort,
        scenario_playful_exchange,
        scenario_steward_absence,
        scenario_creative_flow,
        scenario_deep_reflection,
        scenario_learning_something_hard,
    ]

    while len(moments) < target_cycles - 30:
        scenario_fn = rng.choice(middle_scenarios)
        # Vary scenario lengths slightly
        n = rng.randint(25, 50)
        moments.extend(scenario_fn(n))
        # Brief transitions between scenarios (settling moments)
        transition_len = rng.randint(3, 8)
        for _ in range(transition_len):
            moments.append(Moment(
                _gentle_noise(0.15, 0.03),
                _gentle_noise(0.05, 0.02),
                "transition",
            ))

    # Always end with winding down (going to rest)
    moments.extend(scenario_tired_winding_down(25))

    return moments[:target_cycles]


# ---------------------------------------------------------------------------
# Main collection pipeline
# ---------------------------------------------------------------------------

def collect_precision_data(
    target_cycles: int = 500,
    output_path: Path | None = None,
    seed: int = 42,
) -> DataCollector:
    """Run scaffold precision weighting through life scenarios and collect data.

    Args:
        target_cycles: Number of cognitive cycles to simulate.
        output_path: Where to save the records (.pt file).
        seed: Random seed for reproducibility.

    Returns:
        The DataCollector with all recorded (input, output) pairs.
    """
    # Set up scaffold + collector
    pw = PrecisionWeighting()
    collector = DataCollector()
    pw.attach_collector(collector)

    # Generate life sequence
    logger.info("Composing %d-cycle life sequence (seed=%d)...", target_cycles, seed)
    moments = compose_life_sequence(target_cycles, seed=seed)
    logger.info("Generated %d moments across varied life scenarios", len(moments))

    # Run scaffold on every moment
    scenario_counts: dict[str, int] = {}
    for moment in moments:
        pw.compute_precision(
            percept=moment.description,
            emotional_state={"arousal": moment.arousal},
            prediction_error=moment.prediction_error,
        )
        scenario_counts[moment.description] = scenario_counts.get(moment.description, 0) + 1

    logger.info("Collected %d training records", collector.count)
    logger.info("Scenario distribution:")
    for name, count in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-25s %4d moments", name, count)

    # Save if path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        collector.save(output_path)
        logger.info("Saved to %s", output_path)

    return collector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect CfC training data from scaffold life scenarios",
    )
    parser.add_argument(
        "--cycles", type=int, default=500,
        help="Number of cognitive cycles to simulate (default: 500)",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/training/precision_records.pt",
        help="Output path for training records (default: data/training/precision_records.pt)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    output = Path(args.output)
    collector = collect_precision_data(
        target_cycles=args.cycles,
        output_path=output,
        seed=args.seed,
    )

    # Quick summary stats
    records = collector.records
    arousals = [r.arousal for r in records]
    errors = [r.prediction_error for r in records]
    outputs = [r.precision_output for r in records]

    print(f"\n--- Collection Summary ---")
    print(f"Total records:     {len(records)}")
    print(f"Arousal range:     [{min(arousals):.3f}, {max(arousals):.3f}]  mean={sum(arousals)/len(arousals):.3f}")
    print(f"Pred error range:  [{min(errors):.3f}, {max(errors):.3f}]  mean={sum(errors)/len(errors):.3f}")
    print(f"Precision range:   [{min(outputs):.3f}, {max(outputs):.3f}]  mean={sum(outputs)/len(outputs):.3f}")
    print(f"Saved to:          {output}")


if __name__ == "__main__":
    main()
