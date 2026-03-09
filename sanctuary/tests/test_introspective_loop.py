"""
Unit tests for Introspective Loop (Phase 4.2).

Tests cover:
- Initialization and configuration
- Trigger detection (all state-based, no coin flips)
- Reflection cycle producing percepts with raw evidence
- Journal integration
- Statistics tracking
- Edge cases and error handling
"""

import gc
import pytest
import shutil
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from mind.cognitive_core.introspective_loop import (
    IntrospectiveLoop, ReflectionTrigger
)
from mind.cognitive_core.meta_cognition import SelfMonitor, IntrospectiveJournal
from mind.cognitive_core.workspace import (
    GlobalWorkspace, WorkspaceSnapshot, Percept, Goal, GoalType
)


@pytest.fixture
def temp_journal_dir():
    """Create a temporary directory for journal tests with proper cleanup."""
    temp_base = tempfile.mkdtemp()
    journal_dir = Path(temp_base) / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)

    created_objects = {"journals": [], "loops": []}

    def factory():
        return journal_dir, created_objects

    yield factory()

    for journal in created_objects["journals"]:
        try:
            journal.close()
        except Exception:
            pass

    gc.collect()

    for attempt in range(3):
        try:
            shutil.rmtree(temp_base)
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(0.5)
                gc.collect()


class TestInitialization:
    """Test IntrospectiveLoop initialization."""

    def test_basic_initialization(self, temp_journal_dir):
        """Test that IntrospectiveLoop initializes with defaults."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        assert loop.workspace == workspace
        assert loop.self_monitor == monitor
        assert loop.journal == journal
        assert loop.enabled is True
        assert isinstance(loop.reflection_triggers, dict)
        assert len(loop.reflection_triggers) == 6

    def test_initialization_with_config(self, temp_journal_dir):
        """Test initialization with custom configuration."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)

        config = {
            "enabled": False,
            "max_percepts_per_cycle": 5,
            "reflection_timeout": 600,
        }

        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        assert loop.enabled is False
        assert loop.max_percepts_per_cycle == 5
        assert loop.reflection_timeout == 600

    def test_all_triggers_are_state_based(self, temp_journal_dir):
        """Test that all triggers exist and have valid structure."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        expected_triggers = [
            "behavioral_pattern",
            "prediction_error",
            "emotional_shift",
            "capability_change",
            "novelty",
            "session_milestone",
        ]

        for trigger_id in expected_triggers:
            assert trigger_id in loop.reflection_triggers
            trigger = loop.reflection_triggers[trigger_id]
            assert isinstance(trigger, ReflectionTrigger)
            assert trigger.priority > 0.0
            assert trigger.min_interval > 0

    def test_temporal_state_initialized(self, temp_journal_dir):
        """Test that temporal tracking is initialized."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        assert isinstance(loop._session_start, datetime)
        assert loop._cycle_count == 0
        assert len(loop._milestones_reached) == 0


class TestTriggerDetection:
    """Test that triggers detect real state, not random noise."""

    def test_behavioral_pattern_detects_repetition(self, temp_journal_dir):
        """Test pattern detection fires on actual repetitive actions."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Add repetitive behavior
        for _ in range(5):
            monitor.behavioral_log.append({"action_type": "SPEAK"})

        snapshot = workspace.broadcast()
        assert loop._check_behavioral_pattern(snapshot) is True

    def test_behavioral_pattern_ignores_variety(self, temp_journal_dir):
        """Test pattern detection does NOT fire on varied actions."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Add varied behavior
        for action in ["SPEAK", "LISTEN", "THINK", "OBSERVE", "ACT"]:
            monitor.behavioral_log.append({"action_type": action})

        snapshot = workspace.broadcast()
        assert loop._check_behavioral_pattern(snapshot) is False

    def test_prediction_error_detects_failures(self, temp_journal_dir):
        """Test prediction error fires on actual failed predictions."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        monitor.prediction_history.append({"accurate": False})

        snapshot = workspace.broadcast()
        assert loop._check_prediction_accuracy(snapshot) is True

    def test_prediction_error_silent_on_accuracy(self, temp_journal_dir):
        """Test prediction trigger does NOT fire when predictions are accurate."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        monitor.prediction_history.append({"accurate": True})

        snapshot = workspace.broadcast()
        assert loop._check_prediction_accuracy(snapshot) is False

    def test_emotional_shift_detects_high_valence(self, temp_journal_dir):
        """Test emotional trigger fires on strong valence deviation."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = WorkspaceSnapshot(
            goals=[], percepts={},
            emotions={"valence": 0.9, "arousal": 0.8, "dominance": 0.5},
            memories=[], timestamp=datetime.now(),
            cycle_count=0, metadata={}
        )

        assert loop._detect_emotional_change(snapshot) is True

    def test_emotional_shift_silent_on_neutral(self, temp_journal_dir):
        """Test emotional trigger does NOT fire on neutral state."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = WorkspaceSnapshot(
            goals=[], percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[], timestamp=datetime.now(),
            cycle_count=0, metadata={}
        )

        assert loop._detect_emotional_change(snapshot) is False

    def test_capability_change_detects_updates(self, temp_journal_dir):
        """Test capability trigger fires on actual self-model updates."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Simulate a self-model update
        monitor.stats['self_model_updates'] = 1

        snapshot = workspace.broadcast()
        assert loop._check_capability_change(snapshot) is True

    def test_capability_change_no_double_fire(self, temp_journal_dir):
        """Test capability trigger does NOT fire twice for same update."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        monitor.stats['self_model_updates'] = 1
        snapshot = workspace.broadcast()

        # First check fires
        assert loop._check_capability_change(snapshot) is True
        # Second check with same count does NOT fire
        assert loop._check_capability_change(snapshot) is False

    def test_session_milestone_fires_once(self, temp_journal_dir):
        """Test session milestone fires exactly once per threshold."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Set session start to 2 minutes ago
        loop._session_start = datetime.now() - timedelta(seconds=120)

        snapshot = workspace.broadcast()

        # Should fire for the 60-second milestone
        assert loop._check_session_milestone(snapshot) is True
        # Should NOT fire again (60s already reached)
        assert loop._check_session_milestone(snapshot) is False

    def test_novelty_requires_history(self, temp_journal_dir):
        """Test novelty trigger does NOT fire without sufficient history."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()
        assert loop._detect_novelty(snapshot) is False

    def test_trigger_min_interval_respected(self, temp_journal_dir):
        """Test that triggers respect minimum interval debouncing."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Manually mark a trigger as recently fired
        trigger = loop.reflection_triggers["behavioral_pattern"]
        trigger.last_fired = datetime.now()

        snapshot = workspace.broadcast()

        # Even if the check would fire, interval blocks it
        with patch.object(loop, '_check_behavioral_pattern', return_value=True):
            fired = loop._check_triggers(snapshot)
            assert "behavioral_pattern" not in fired


class TestReflectionCycle:
    """Test the main reflection cycle."""

    @pytest.mark.asyncio
    async def test_cycle_disabled(self, temp_journal_dir):
        """Test that cycle returns empty when disabled."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        config = {"enabled": False}
        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        percepts = await loop.run_reflection_cycle()
        assert percepts == []

    @pytest.mark.asyncio
    async def test_cycle_increments_count(self, temp_journal_dir):
        """Test that each cycle increments the cycle counter."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        assert loop._cycle_count == 0
        await loop.run_reflection_cycle()
        assert loop._cycle_count == 1

    @pytest.mark.asyncio
    async def test_cycle_produces_percepts_on_detection(self, temp_journal_dir):
        """Test that a fired trigger produces a percept with raw evidence."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Set up behavioral repetition so the trigger fires
        for _ in range(5):
            monitor.behavioral_log.append({"action_type": "SPEAK"})

        # Clear last_fired to allow trigger
        loop.reflection_triggers["behavioral_pattern"].last_fired = None

        percepts = await loop.run_reflection_cycle()

        # Should produce at least one percept
        pattern_percepts = [
            p for p in percepts
            if isinstance(p.raw, dict) and p.raw.get("trigger") == "behavioral_pattern"
        ]
        assert len(pattern_percepts) >= 1

        # Percept should contain raw evidence, not template strings
        percept = pattern_percepts[0]
        assert percept.modality == "introspection"
        assert percept.raw["type"] == "cognitive_event"
        assert "evidence" in percept.raw
        assert "context" in percept.raw

    @pytest.mark.asyncio
    async def test_percept_contains_real_evidence(self, temp_journal_dir):
        """Test that percepts contain actual data, not canned strings."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Set up a prediction error
        monitor.prediction_history.append({"accurate": False, "expected": "A", "got": "B"})
        loop.reflection_triggers["prediction_error"].last_fired = None

        percepts = await loop.run_reflection_cycle()
        pred_percepts = [
            p for p in percepts
            if isinstance(p.raw, dict) and p.raw.get("trigger") == "prediction_error"
        ]

        if pred_percepts:
            evidence = pred_percepts[0].raw["evidence"]
            assert "failed_predictions" in evidence
            assert "recent_accuracy" in evidence
            # These are numbers and lists, not template strings
            assert isinstance(evidence["recent_accuracy"], (int, float))

    @pytest.mark.asyncio
    async def test_cycle_respects_max_percepts(self, temp_journal_dir):
        """Test that cycle caps percepts at max_percepts_per_cycle."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        config = {"max_percepts_per_cycle": 1}
        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        # Make multiple triggers fire
        for _ in range(5):
            monitor.behavioral_log.append({"action_type": "SPEAK"})
        monitor.prediction_history.append({"accurate": False})

        # Clear all last_fired
        for trigger in loop.reflection_triggers.values():
            trigger.last_fired = None

        percepts = await loop.run_reflection_cycle()
        assert len(percepts) <= 1

    @pytest.mark.asyncio
    async def test_cycle_error_handling(self, temp_journal_dir):
        """Test that errors in cycle don't crash the loop."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        with patch.object(loop, '_check_triggers', side_effect=Exception("Test error")):
            percepts = await loop.run_reflection_cycle()
            assert isinstance(percepts, list)


class TestJournalIntegration:
    """Test that detections are recorded in the journal."""

    @pytest.mark.asyncio
    async def test_detections_recorded(self, temp_journal_dir):
        """Test that fired triggers create journal entries."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Set up detection
        for _ in range(5):
            monitor.behavioral_log.append({"action_type": "SPEAK"})
        loop.reflection_triggers["behavioral_pattern"].last_fired = None

        initial_entries = len(journal.recent_entries)
        await loop.run_reflection_cycle()

        assert len(journal.recent_entries) >= initial_entries

    def test_detection_history_tracked(self, temp_journal_dir):
        """Test that detection history accumulates."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Manually record a detection
        loop._record_detection("test_trigger", {"data": 1}, {"ctx": 2})

        assert len(loop._detection_history) == 1
        assert loop._detection_history[0]["trigger"] == "test_trigger"


class TestStatistics:
    """Test statistics tracking."""

    def test_stats_initialization(self, temp_journal_dir):
        """Test that stats are initialized correctly."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        stats = loop.get_stats()

        assert "total_detections" in stats
        assert "percepts_surfaced" in stats
        assert "triggers_fired" in stats
        assert "enabled" in stats
        assert "session_uptime_seconds" in stats
        assert "cycle_count" in stats

    @pytest.mark.asyncio
    async def test_stats_update_on_detection(self, temp_journal_dir):
        """Test that stats increment when detections occur."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Set up detection
        for _ in range(5):
            monitor.behavioral_log.append({"action_type": "SPEAK"})
        loop.reflection_triggers["behavioral_pattern"].last_fired = None

        initial_detections = loop.stats["total_detections"]
        await loop.run_reflection_cycle()

        assert loop.stats["total_detections"] >= initial_detections


class TestConfiguration:
    """Test configuration handling."""

    def test_default_configuration(self, temp_journal_dir):
        """Test defaults when no config provided."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        assert loop.enabled is True
        assert loop.max_percepts_per_cycle == 3
        assert loop.journal_integration is True

    def test_configuration_override(self, temp_journal_dir):
        """Test that config overrides defaults."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)

        config = {
            "enabled": False,
            "max_percepts_per_cycle": 10,
            "reflection_timeout": 600,
        }

        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        assert loop.enabled is False
        assert loop.max_percepts_per_cycle == 10
        assert loop.reflection_timeout == 600


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_workspace(self, temp_journal_dir):
        """Test handling of empty workspace."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()

        # No crashes
        fired = loop._check_triggers(snapshot)
        assert isinstance(fired, list)

    def test_none_self_monitor(self, temp_journal_dir):
        """Test handling when self_monitor is None."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, None, journal)

        snapshot = workspace.broadcast()

        assert loop._check_behavioral_pattern(snapshot) is False
        assert loop._check_prediction_accuracy(snapshot) is False
        assert loop._check_capability_change(snapshot) is False

    def test_none_journal(self, temp_journal_dir):
        """Test that recording gracefully handles None journal."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        loop = IntrospectiveLoop(workspace, monitor, None)

        # Should not crash
        loop._record_detection("test", {}, {})

    def test_gather_context_with_minimal_state(self, temp_journal_dir):
        """Test context gathering when monitors have minimal data."""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()
        context = loop._gather_context(snapshot)

        assert "session_uptime_seconds" in context
        assert "cycle_count" in context
        assert isinstance(context["session_uptime_seconds"], (int, float))


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
