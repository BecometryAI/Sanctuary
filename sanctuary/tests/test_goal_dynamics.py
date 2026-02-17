"""
Tests for GoalDynamics: dynamic priority adjustment, deadline urgency,
emotional congruence, and staleness/frustration boost.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from mind.cognitive_core.goals.dynamics import (
    GoalDynamics,
    GoalAdjustment,
    GoalDynamicsState,
    STALL_CYCLE_THRESHOLD,
)


def _make_goal(
    goal_id="g1",
    goal_type="respond_to_user",
    priority=0.5,
    progress=0.0,
    deadline=None,
    metadata=None,
    created_at=None,
):
    g = MagicMock()
    g.id = goal_id
    g.type = MagicMock()
    g.type.value = goal_type
    g.priority = priority
    g.progress = progress
    g.deadline = deadline
    g.metadata = metadata or {}
    g.created_at = created_at or datetime.now()
    return g


class TestGoalDynamics:
    def test_init_defaults(self):
        gd = GoalDynamics()
        assert gd.stall_threshold == STALL_CYCLE_THRESHOLD
        assert gd._state is not None

    def test_no_adjustment_for_fresh_goal(self):
        gd = GoalDynamics()
        goal = _make_goal()
        adjustments = gd.adjust_priorities([goal], cycle_count=1)
        assert adjustments == []

    def test_stall_boost_after_threshold(self):
        gd = GoalDynamics()
        goal = _make_goal(priority=0.5, progress=0.0)

        # Simulate cycles without progress
        for cycle in range(STALL_CYCLE_THRESHOLD + 10):
            adjustments = gd.adjust_priorities([goal], cycle_count=cycle)

        # Should now have a stall boost
        assert len(adjustments) > 0
        adj = adjustments[0]
        assert adj.new_priority > adj.old_priority
        assert "stalled" in adj.reason

    def test_no_stall_boost_when_progressing(self):
        gd = GoalDynamics()
        goal = _make_goal(priority=0.5, progress=0.0)

        for cycle in range(STALL_CYCLE_THRESHOLD + 5):
            # Make progress each cycle
            goal.progress = cycle * 0.02
            adjustments = gd.adjust_priorities([goal], cycle_count=cycle)

        # Should not have stall-related adjustments
        stall_adjs = [a for a in adjustments if "stalled" in a.reason]
        assert len(stall_adjs) == 0

    def test_deadline_boost_overdue(self):
        gd = GoalDynamics()
        # Overdue deadline
        goal = _make_goal(priority=0.5, deadline=datetime.now() - timedelta(hours=1))
        adjustments = gd.adjust_priorities([goal], cycle_count=100)
        assert len(adjustments) > 0
        adj = adjustments[0]
        assert adj.new_priority > adj.old_priority
        assert "deadline" in adj.reason

    def test_deadline_boost_imminent(self):
        gd = GoalDynamics()
        # Deadline in 30 minutes
        goal = _make_goal(priority=0.5, deadline=datetime.now() + timedelta(minutes=30))
        adjustments = gd.adjust_priorities([goal], cycle_count=100)
        assert len(adjustments) > 0
        adj = adjustments[0]
        assert adj.new_priority > adj.old_priority

    def test_no_deadline_boost_far_future(self):
        gd = GoalDynamics()
        # Deadline in 2 weeks
        goal = _make_goal(priority=0.5, deadline=datetime.now() + timedelta(weeks=2))
        adjustments = gd.adjust_priorities([goal], cycle_count=100)
        # Should not have significant deadline boost
        deadline_adjs = [a for a in adjustments if "deadline" in a.reason]
        assert len(deadline_adjs) == 0

    def test_emotion_congruence_high_arousal(self):
        gd = GoalDynamics()
        goal = _make_goal(goal_type="respond_to_user", priority=0.5)
        emotional_state = {"arousal": 0.8, "valence": 0.0, "label": "anticipation"}
        adjustments = gd.adjust_priorities([goal], cycle_count=100, emotional_state=emotional_state)
        assert len(adjustments) > 0
        assert adjustments[0].new_priority > adjustments[0].old_priority

    def test_emotion_congruence_negative_valence_introspect(self):
        gd = GoalDynamics()
        goal = _make_goal(goal_type="introspect", priority=0.5)
        emotional_state = {"arousal": 0.3, "valence": -0.6, "label": "sadness"}
        adjustments = gd.adjust_priorities([goal], cycle_count=100, emotional_state=emotional_state)
        assert len(adjustments) > 0
        assert adjustments[0].new_priority > adjustments[0].old_priority

    def test_progress_decay(self):
        gd = GoalDynamics()
        goal = _make_goal(priority=0.8, progress=0.7)
        # First cycle to establish tracking
        gd.adjust_priorities([goal], cycle_count=0)
        # Make more progress
        goal.progress = 0.72
        adjustments = gd.adjust_priorities([goal], cycle_count=1)
        # Should reduce priority slightly
        if adjustments:
            assert adjustments[0].new_priority < adjustments[0].old_priority

    def test_cleanup_stale_goals(self):
        gd = GoalDynamics()
        goal1 = _make_goal(goal_id="g1")
        goal2 = _make_goal(goal_id="g2")
        gd.adjust_priorities([goal1, goal2], cycle_count=0)
        assert "g1" in gd._state.first_seen_cycle
        assert "g2" in gd._state.first_seen_cycle

        # Remove goal2
        gd.adjust_priorities([goal1], cycle_count=1)
        assert "g1" in gd._state.first_seen_cycle
        assert "g2" not in gd._state.first_seen_cycle

    def test_adjustment_capped(self):
        gd = GoalDynamics(config={"adjustment_cap": 0.10})
        goal = _make_goal(
            priority=0.5,
            deadline=datetime.now() - timedelta(hours=10),
        )
        # Force many stall cycles + overdue deadline
        for c in range(STALL_CYCLE_THRESHOLD + 50):
            adjustments = gd.adjust_priorities([goal], cycle_count=c)
        assert len(adjustments) > 0
        adj = adjustments[0]
        assert abs(adj.adjustment) <= 0.10 + 0.001  # slight float tolerance

    def test_multiple_goals(self):
        gd = GoalDynamics()
        goals = [
            _make_goal(goal_id=f"g{i}", priority=0.5)
            for i in range(5)
        ]
        adjustments = gd.adjust_priorities(goals, cycle_count=0)
        # First cycle — no adjustments expected
        assert adjustments == []
