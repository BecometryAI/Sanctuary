"""
Tests for confidence-based action modulation in the action subsystem.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from mind.cognitive_core.action import ActionSubsystem, ActionType, Action
from mind.cognitive_core.workspace import WorkspaceSnapshot, Goal, GoalType


def _make_snapshot(goals=None, emotions=None, metadata=None, percepts=None):
    return WorkspaceSnapshot(
        goals=goals or [],
        percepts=percepts or {},
        emotions=emotions or {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
        memories=[],
        metadata=metadata or {},
        timestamp=datetime.now(),
        cycle_count=0,
    )


class TestConfidenceModulation:
    @pytest.fixture
    def action_sub(self):
        return ActionSubsystem(config={})

    def test_low_confidence_boosts_introspect(self, action_sub):
        """Low IWMT confidence should boost INTROSPECT action score."""
        snapshot = _make_snapshot(metadata={"iwmt_confidence": 0.2})
        introspect_action = Action(type=ActionType.INTROSPECT, priority=0.5)
        score = action_sub._score_action(introspect_action, snapshot)

        # Compare with no confidence
        snapshot_no_conf = _make_snapshot(metadata={})
        score_no_conf = action_sub._score_action(introspect_action, snapshot_no_conf)

        assert score > score_no_conf

    def test_low_confidence_penalizes_speak(self, action_sub):
        """Low IWMT confidence should penalize SPEAK action score."""
        snapshot = _make_snapshot(metadata={"iwmt_confidence": 0.1})
        speak_action = Action(type=ActionType.SPEAK, priority=0.7)
        score = action_sub._score_action(speak_action, snapshot)

        snapshot_high_conf = _make_snapshot(metadata={"iwmt_confidence": 0.8})
        score_high = action_sub._score_action(speak_action, snapshot_high_conf)

        assert score < score_high

    def test_high_confidence_no_penalty(self, action_sub):
        """High confidence (>0.5) should not apply modulation."""
        snapshot = _make_snapshot(metadata={"iwmt_confidence": 0.8})
        speak_action = Action(type=ActionType.SPEAK, priority=0.7)
        score_with = action_sub._score_action(speak_action, snapshot)

        snapshot_none = _make_snapshot(metadata={})
        score_without = action_sub._score_action(speak_action, snapshot_none)

        # Should be equal — no modulation above 0.5
        assert abs(score_with - score_without) < 0.01

    def test_very_low_confidence_generates_introspect_candidate(self, action_sub):
        """Very low confidence (<0.3) should add an INTROSPECT candidate."""
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="test")
        snapshot = _make_snapshot(
            goals=[goal],
            metadata={"iwmt_confidence": 0.15}
        )
        candidates = action_sub._generate_candidates(snapshot)

        introspect_candidates = [
            c for c in candidates
            if c.type == ActionType.INTROSPECT
            and c.metadata.get("trigger") == "low_confidence"
        ]
        assert len(introspect_candidates) == 1

    def test_normal_confidence_no_extra_introspect(self, action_sub):
        """Normal confidence should not generate extra INTROSPECT candidate."""
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="test")
        snapshot = _make_snapshot(
            goals=[goal],
            metadata={"iwmt_confidence": 0.6}
        )
        candidates = action_sub._generate_candidates(snapshot)

        low_conf_introspect = [
            c for c in candidates
            if c.type == ActionType.INTROSPECT
            and c.metadata.get("trigger") == "low_confidence"
        ]
        assert len(low_conf_introspect) == 0

    def test_confidence_modulation_wait_action(self, action_sub):
        """Low confidence should also boost WAIT action."""
        snapshot = _make_snapshot(metadata={"iwmt_confidence": 0.2})
        wait_action = Action(type=ActionType.WAIT, priority=0.3)
        score_low = action_sub._score_action(wait_action, snapshot)

        snapshot_none = _make_snapshot(metadata={})
        score_none = action_sub._score_action(wait_action, snapshot_none)

        assert score_low > score_none

    def test_confidence_modulation_clamped(self, action_sub):
        """Scores should always be clamped to [0.0, 1.0]."""
        snapshot = _make_snapshot(metadata={"iwmt_confidence": 0.0})
        action = Action(type=ActionType.INTROSPECT, priority=0.95)
        score = action_sub._score_action(action, snapshot)
        assert 0.0 <= score <= 1.0


class TestWorkspaceMetadata:
    def test_metadata_in_broadcast(self):
        from mind.cognitive_core.workspace import GlobalWorkspace
        ws = GlobalWorkspace()
        ws.metadata["iwmt_confidence"] = 0.75
        snap = ws.broadcast()
        assert snap.metadata["iwmt_confidence"] == 0.75

    def test_metadata_cleared(self):
        from mind.cognitive_core.workspace import GlobalWorkspace
        ws = GlobalWorkspace()
        ws.metadata["test"] = True
        ws.clear()
        assert ws.metadata == {}
