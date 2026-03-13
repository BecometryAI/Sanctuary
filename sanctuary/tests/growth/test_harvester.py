"""Tests for ReflectionHarvester -- reflection collection, queue limits, save/load."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanctuary.core.schema import CognitiveOutput, GrowthReflection, EmotionalOutput
from sanctuary.growth.harvester import HarvestedReflection, ReflectionHarvester


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_output(
    worth_learning: bool = True,
    what_to_learn: str = "Test learning",
    training_pair_suggestion: dict | None = None,
    inner_speech: str = "I was thinking about this",
    felt_quality: str = "curious",
) -> CognitiveOutput:
    """Create a CognitiveOutput with a growth reflection."""
    return CognitiveOutput(
        inner_speech=inner_speech,
        emotional_state=EmotionalOutput(felt_quality=felt_quality),
        growth_reflection=GrowthReflection(
            worth_learning=worth_learning,
            what_to_learn=what_to_learn,
            training_pair_suggestion=training_pair_suggestion,
        ),
    )


# ---------------------------------------------------------------------------
# Basic harvesting
# ---------------------------------------------------------------------------


class TestHarvestBasics:
    """Test basic reflection harvesting behavior."""

    def test_harvest_worth_learning(self):
        """Reflections marked worth_learning are harvested."""
        harvester = ReflectionHarvester()
        output = _make_output(worth_learning=True, what_to_learn="empathy")

        result = harvester.harvest(output, cycle_count=1)

        assert result is not None
        assert result.cycle_count == 1
        assert result.reflection["what_to_learn"] == "empathy"
        assert harvester.pending_count == 1

    def test_skip_not_worth_learning(self):
        """Reflections not marked worth_learning are skipped."""
        harvester = ReflectionHarvester()
        output = _make_output(worth_learning=False)

        result = harvester.harvest(output, cycle_count=1)

        assert result is None
        assert harvester.pending_count == 0

    def test_skip_no_reflection(self):
        """Outputs without growth_reflection are skipped."""
        harvester = ReflectionHarvester()
        output = CognitiveOutput(inner_speech="just thinking")

        result = harvester.harvest(output, cycle_count=1)

        assert result is None
        assert harvester.pending_count == 0

    def test_harvested_reflection_has_context(self):
        """Harvested reflections include cycle context."""
        harvester = ReflectionHarvester()
        output = _make_output(
            inner_speech="deep thought",
            felt_quality="wonder",
        )

        result = harvester.harvest(output, cycle_count=42)

        assert result is not None
        assert result.inner_speech_context == "deep thought"
        assert result.emotional_context == "wonder"
        assert result.cycle_count == 42

    def test_harvested_reflection_has_id(self):
        """Each harvested reflection gets a unique ID."""
        harvester = ReflectionHarvester()
        out1 = _make_output(what_to_learn="thing 1")
        out2 = _make_output(what_to_learn="thing 2")

        r1 = harvester.harvest(out1, cycle_count=1)
        r2 = harvester.harvest(out2, cycle_count=2)

        assert r1 is not None and r2 is not None
        assert r1.id != r2.id


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------


class TestQueueManagement:
    """Test pending queue behavior and limits."""

    def test_drain_returns_all_pending(self):
        """Drain returns all pending reflections and clears the queue."""
        harvester = ReflectionHarvester()
        for i in range(3):
            harvester.harvest(_make_output(what_to_learn=f"learn {i}"), cycle_count=i)

        drained = harvester.drain()

        assert len(drained) == 3
        assert harvester.pending_count == 0

    def test_drain_empty_queue(self):
        """Draining an empty queue returns an empty list."""
        harvester = ReflectionHarvester()
        drained = harvester.drain()
        assert drained == []

    def test_max_pending_drops_oldest(self):
        """When queue is full, oldest reflection is dropped."""
        harvester = ReflectionHarvester(max_pending=3)

        for i in range(5):
            harvester.harvest(_make_output(what_to_learn=f"learn {i}"), cycle_count=i)

        assert harvester.pending_count == 3
        pending = harvester.pending
        # The oldest two (0, 1) should have been dropped
        assert pending[0].reflection["what_to_learn"] == "learn 2"
        assert pending[2].reflection["what_to_learn"] == "learn 4"

    def test_history_keeps_all(self):
        """History keeps all reflections, even those dropped from pending."""
        harvester = ReflectionHarvester(max_pending=2)

        for i in range(5):
            harvester.harvest(_make_output(what_to_learn=f"learn {i}"), cycle_count=i)

        assert len(harvester.history) == 5
        assert harvester.pending_count == 2


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """Test persistence of harvester state."""

    def test_save_and_load(self, tmp_path: Path):
        """Harvester state survives save/load cycle."""
        harvester = ReflectionHarvester()
        harvester.harvest(_make_output(what_to_learn="persist this"), cycle_count=1)
        harvester.harvest(_make_output(what_to_learn="and this"), cycle_count=2)

        save_path = tmp_path / "harvester.json"
        harvester.save(save_path)

        # Load into fresh harvester
        loaded = ReflectionHarvester()
        loaded.load(save_path)

        assert len(loaded.history) == 2
        assert loaded.pending_count == 2
        assert loaded.history[0].reflection["what_to_learn"] == "persist this"

    def test_save_creates_directories(self, tmp_path: Path):
        """Save creates parent directories if needed."""
        harvester = ReflectionHarvester()
        save_path = tmp_path / "deep" / "nested" / "harvester.json"
        harvester.save(save_path)
        assert save_path.exists()

    def test_load_nonexistent_file(self, tmp_path: Path):
        """Loading from a nonexistent file is a no-op."""
        harvester = ReflectionHarvester()
        harvester.load(tmp_path / "nonexistent.json")
        assert harvester.pending_count == 0

    def test_save_format_is_json(self, tmp_path: Path):
        """Saved file is valid JSON."""
        harvester = ReflectionHarvester()
        harvester.harvest(_make_output(), cycle_count=1)

        save_path = tmp_path / "harvester.json"
        harvester.save(save_path)

        data = json.loads(save_path.read_text())
        assert "history" in data
        assert "pending" in data
        assert "saved_at" in data
