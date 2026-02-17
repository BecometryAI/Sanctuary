"""
Tests for identity evolution tracking and consistency checks.
"""

import pytest
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from mind.cognitive_core.identity.continuity import (
    IdentityContinuity,
    IdentitySnapshot,
    IdentityEvolutionEvent,
)


class _SimpleIdentity:
    """Simple identity stand-in that won't trip MagicMock auto-attrs."""
    def __init__(self, core_values, emotional_disposition, autobiographical_self,
                 behavioral_tendencies, source):
        self.core_values = core_values
        self.emotional_disposition = emotional_disposition
        self.autobiographical_self = autobiographical_self
        self.behavioral_tendencies = behavioral_tendencies
        self.source = source


def _make_identity(
    core_values=None,
    emotional_disposition=None,
    autobiographical_self=None,
    behavioral_tendencies=None,
    source="computed",
):
    return _SimpleIdentity(
        core_values=core_values or ["honesty", "curiosity"],
        emotional_disposition=emotional_disposition or {"valence": 0.1, "arousal": 0.3, "dominance": 0.6},
        autobiographical_self=autobiographical_self or [],
        behavioral_tendencies=behavioral_tendencies or {"tendency_speak": 0.4, "proactivity": 0.5},
        source=source,
    )


class TestIdentityEvolutionTracking:
    @pytest.fixture
    def tmp_dir(self, tmp_path):
        return str(tmp_path / "evolution")

    def test_snapshot_persisted_to_disk(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        identity = _make_identity()
        ic.take_snapshot(identity, trigger="test")

        snapshot_file = Path(tmp_dir) / "snapshots.jsonl"
        assert snapshot_file.exists()
        lines = snapshot_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert "honesty" in data["core_values"]

    def test_multiple_snapshots_persisted(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        for i in range(5):
            ic.take_snapshot(_make_identity(), trigger="periodic")

        snapshot_file = Path(tmp_dir) / "snapshots.jsonl"
        lines = snapshot_file.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_snapshots_loaded_on_init(self, tmp_dir):
        # First instance writes snapshots
        ic1 = IdentityContinuity(config={"persistence_dir": tmp_dir})
        ic1.take_snapshot(_make_identity(core_values=["honesty"]))
        ic1.take_snapshot(_make_identity(core_values=["honesty", "kindness"]))

        # Second instance loads them
        ic2 = IdentityContinuity(config={"persistence_dir": tmp_dir})
        assert len(ic2.snapshots) == 2
        assert "honesty" in ic2.snapshots[0].core_values

    def test_value_added_event(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        ic.take_snapshot(_make_identity(core_values=["honesty"]))
        ic.take_snapshot(_make_identity(core_values=["honesty", "compassion"]))

        added_events = [e for e in ic.evolution_events if e.event_type == "value_added"]
        assert len(added_events) == 1
        assert added_events[0].new_value == "compassion"

    def test_value_removed_event(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        ic.take_snapshot(_make_identity(core_values=["honesty", "curiosity"]))
        ic.take_snapshot(_make_identity(core_values=["honesty"]))

        removed_events = [e for e in ic.evolution_events if e.event_type == "value_removed"]
        assert len(removed_events) == 1
        assert removed_events[0].old_value == "curiosity"

    def test_disposition_shift_event(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        ic.take_snapshot(_make_identity(
            emotional_disposition={"valence": 0.1, "arousal": 0.2, "dominance": 0.5}
        ))
        ic.take_snapshot(_make_identity(
            emotional_disposition={"valence": 0.8, "arousal": 0.9, "dominance": 0.9}
        ))

        shift_events = [e for e in ic.evolution_events if e.event_type == "disposition_shift"]
        assert len(shift_events) == 1

    def test_tendency_change_event(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        ic.take_snapshot(_make_identity(
            behavioral_tendencies={"tendency_speak": 0.3, "proactivity": 0.5}
        ))
        ic.take_snapshot(_make_identity(
            behavioral_tendencies={"tendency_speak": 0.7, "proactivity": 0.5}
        ))

        tendency_events = [e for e in ic.evolution_events if e.event_type == "tendency_change"]
        assert len(tendency_events) == 1
        assert "tendency_speak" in tendency_events[0].description

    def test_no_event_for_minor_change(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        ic.take_snapshot(_make_identity(
            emotional_disposition={"valence": 0.1, "arousal": 0.3, "dominance": 0.6}
        ))
        ic.take_snapshot(_make_identity(
            emotional_disposition={"valence": 0.12, "arousal": 0.31, "dominance": 0.6}
        ))

        # Tiny change should not generate a disposition_shift event
        shift_events = [e for e in ic.evolution_events if e.event_type == "disposition_shift"]
        assert len(shift_events) == 0

    def test_events_persisted_to_disk(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        ic.take_snapshot(_make_identity(core_values=["a"]))
        ic.take_snapshot(_make_identity(core_values=["a", "b"]))

        events_file = Path(tmp_dir) / "events.jsonl"
        assert events_file.exists()
        lines = events_file.read_text().strip().split("\n")
        assert len(lines) >= 1

    def test_get_evolution_summary(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        ic.take_snapshot(_make_identity(core_values=["a"]))
        ic.take_snapshot(_make_identity(core_values=["a", "b"]))
        ic.take_snapshot(_make_identity(core_values=["b"]))

        summary = ic.get_evolution_summary()
        assert summary["total_events"] >= 2
        assert "value_added" in summary["event_counts"]

    def test_trigger_recorded_in_snapshot_metadata(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        ic.take_snapshot(_make_identity(), trigger="behavior_pattern")
        assert ic.snapshots[-1].metadata["trigger"] == "behavior_pattern"

    def test_continuity_score_still_works(self, tmp_dir):
        ic = IdentityContinuity(config={"persistence_dir": tmp_dir})
        ic.take_snapshot(_make_identity(core_values=["a", "b"]))
        ic.take_snapshot(_make_identity(core_values=["a", "b"]))
        score = ic.get_continuity_score()
        assert score == 1.0  # identical snapshots = perfect continuity


class TestIdentityConsistencyCheck:
    """Test the SelfMonitor.check_identity_consistency() method."""

    def test_no_identity_returns_none(self):
        from mind.cognitive_core.meta_cognition import SelfMonitor
        sm = SelfMonitor(identity=None, identity_manager=None)
        result = sm.check_identity_consistency()
        assert result is None

    def test_empty_identity_returns_none(self):
        from mind.cognitive_core.meta_cognition import SelfMonitor

        identity = MagicMock()
        identity.charter = MagicMock()
        identity.charter.core_values = ["Honesty"]
        identity.charter.behavioral_guidelines = []
        identity.protocols = []

        manager = MagicMock()
        empty_id = MagicMock()
        empty_id.source = "empty"
        manager.get_identity.return_value = empty_id

        sm = SelfMonitor(identity=identity, identity_manager=manager)
        result = sm.check_identity_consistency()
        assert result is None

    def test_divergence_detected(self):
        from mind.cognitive_core.meta_cognition import SelfMonitor

        identity = MagicMock()
        identity.charter = MagicMock()
        identity.charter.core_values = ["Honesty", "Helpfulness", "Harmlessness"]
        identity.charter.behavioral_guidelines = []
        identity.protocols = []

        computed = MagicMock()
        computed.source = "computed"
        computed.core_values = ["curiosity", "persistence"]  # Different from charter
        computed.behavioral_tendencies = {}

        manager = MagicMock()
        manager.get_identity.return_value = computed
        manager.get_identity_drift.return_value = {"has_drift": False}

        sm = SelfMonitor(identity=identity, identity_manager=manager)
        result = sm.check_identity_consistency()
        assert result is not None
        assert "divergence" in result.raw.lower()

    def test_no_divergence_when_aligned(self):
        from mind.cognitive_core.meta_cognition import SelfMonitor

        identity = MagicMock()
        identity.charter = MagicMock()
        identity.charter.core_values = ["honesty"]
        identity.charter.behavioral_guidelines = []
        identity.protocols = []

        computed = MagicMock()
        computed.source = "computed"
        computed.core_values = ["honesty"]
        computed.behavioral_tendencies = {}

        manager = MagicMock()
        manager.get_identity.return_value = computed
        manager.get_identity_drift.return_value = {"has_drift": False}

        sm = SelfMonitor(identity=identity, identity_manager=manager)
        result = sm.check_identity_consistency()
        assert result is None

    def test_drift_generates_percept(self):
        from mind.cognitive_core.meta_cognition import SelfMonitor

        identity = MagicMock()
        identity.charter = MagicMock()
        identity.charter.core_values = []
        identity.charter.behavioral_guidelines = []
        identity.protocols = []

        computed = MagicMock()
        computed.source = "computed"
        computed.core_values = []
        computed.behavioral_tendencies = {}

        manager = MagicMock()
        manager.get_identity.return_value = computed
        manager.get_identity_drift.return_value = {
            "has_drift": True,
            "disposition_change": 0.6,
        }

        sm = SelfMonitor(identity=identity, identity_manager=manager)
        result = sm.check_identity_consistency()
        assert result is not None
        assert "drift" in result.raw.lower()
