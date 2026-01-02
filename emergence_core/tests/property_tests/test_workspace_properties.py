"""
Property-based tests for GlobalWorkspace invariants.

Tests cover:
- Workspace snapshot immutability
- Percept addition consistency
- Goal management consistency
- Emotional state bounds
- Serialization/deserialization
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pydantic import ValidationError

from lyra.cognitive_core.workspace import (
    GlobalWorkspace,
    Goal,
    GoalType,
    WorkspaceSnapshot,
)
from .strategies import (
    percepts,
    goals,
    emotional_states,
    percept_lists,
    goal_lists,
    make_unique_by_id,
)


@pytest.mark.property
class TestWorkspaceProperties:
    """Property-based tests for GlobalWorkspace invariants."""
    
    @given(percept_lists, goal_lists, emotional_states())
    @settings(max_examples=100, deadline=None)
    def test_workspace_snapshot_immutability(self, percepts_list, goals_list, emotions):
        """Property: Workspace snapshots are immutable."""
        workspace = GlobalWorkspace()
        
        # Add percepts
        for percept in percepts_list:
            workspace.active_percepts[percept.id] = percept
        
        # Add goals
        for goal in goals_list:
            workspace.add_goal(goal)
        
        # Update emotions
        workspace.emotional_state["valence"] = emotions.valence
        workspace.emotional_state["arousal"] = emotions.arousal
        workspace.emotional_state["dominance"] = emotions.dominance
        
        # Get snapshot
        snapshot = workspace.broadcast()
        
        # Verify snapshot is frozen (attempting modification raises error)
        with pytest.raises((ValidationError, AttributeError, TypeError)):
            snapshot.cycle_count = 999
        
        # Verify snapshot unchanged after workspace modification
        original_goal_count = len(snapshot.goals)
        new_goal = Goal(type=GoalType.LEARN, description="New goal after snapshot")
        workspace.add_goal(new_goal)
        assert len(snapshot.goals) == original_goal_count
    
    @given(percepts())
    @settings(max_examples=100)
    def test_percept_addition_increases_count(self, percept):
        """Property: Adding a percept increases count by 1."""
        workspace = GlobalWorkspace()
        initial_count = len(workspace.active_percepts)
        
        workspace.active_percepts[percept.id] = percept
        
        assert len(workspace.active_percepts) == initial_count + 1
        assert percept.id in workspace.active_percepts
    
    @given(percept_lists)
    @settings(max_examples=100)
    def test_percept_count_equals_additions(self, percepts_list):
        """Property: Final percept count equals number of unique additions."""
        workspace = GlobalWorkspace()
        
        # Ensure unique IDs
        unique_percepts = make_unique_by_id(percepts_list)
        
        for percept in unique_percepts:
            workspace.active_percepts[percept.id] = percept
        
        assert len(workspace.active_percepts) == len(unique_percepts)
    
    @given(emotional_states())
    @settings(max_examples=100)
    def test_emotional_state_bounded(self, emotions):
        """Property: Emotional state VAD values always remain in [-1, 1]."""
        workspace = GlobalWorkspace()
        workspace.emotional_state["valence"] = emotions.valence
        workspace.emotional_state["arousal"] = emotions.arousal
        workspace.emotional_state["dominance"] = emotions.dominance
        
        snapshot = workspace.broadcast()
        assert -1.0 <= snapshot.emotions["valence"] <= 1.0
        assert -1.0 <= snapshot.emotions["arousal"] <= 1.0
        assert -1.0 <= snapshot.emotions["dominance"] <= 1.0
    
    @given(goal_lists)
    @settings(max_examples=100)
    def test_goal_priority_ordering(self, goals_list):
        """Property: Goals are maintained in priority order (highest first)."""
        assume(len(goals_list) > 0)
        
        workspace = GlobalWorkspace()
        unique_goals = make_unique_by_id(goals_list)
        
        for goal in unique_goals:
            workspace.add_goal(goal)
        
        # Verify goals are sorted by priority (highest first)
        priorities = [g.priority for g in workspace.current_goals]
        assert priorities == sorted(priorities, reverse=True)
    
    @given(goals())
    @settings(max_examples=100)
    def test_goal_duplicate_prevention(self, goal):
        """Property: Adding the same goal twice results in only one instance."""
        workspace = GlobalWorkspace()
        
        workspace.add_goal(goal)
        initial_count = len(workspace.current_goals)
        
        workspace.add_goal(goal)  # Try to add same goal again
        
        assert len(workspace.current_goals) == initial_count
    
    @given(st.lists(percepts(), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_update_increments_cycle_count(self, percepts_list):
        """Property: Each update() call increments cycle_count by 1."""
        workspace = GlobalWorkspace()
        initial_cycle = workspace.cycle_count
        
        outputs = [{"type": "percept", "data": p} for p in percepts_list[:3]]
        workspace.update(outputs)
        
        assert workspace.cycle_count == initial_cycle + 1
        
        workspace.update([])
        assert workspace.cycle_count == initial_cycle + 2
    
    @given(goal_lists, percept_lists, emotional_states())
    @settings(max_examples=50, deadline=None)
    def test_serialization_roundtrip(self, goals_list, percepts_list, emotions):
        """Property: Workspace can be serialized and deserialized without data loss."""
        workspace1 = GlobalWorkspace(capacity=10)
        
        # Add state
        unique_goals = make_unique_by_id(goals_list)
        for goal in unique_goals[:5]:  # Limit to 5 goals
            workspace1.add_goal(goal)
        
        unique_percepts = make_unique_by_id(percepts_list)
        for percept in unique_percepts[:5]:  # Limit to 5 percepts
            workspace1.active_percepts[percept.id] = percept
        
        workspace1.emotional_state["valence"] = emotions.valence
        workspace1.emotional_state["arousal"] = emotions.arousal
        workspace1.cycle_count = 10
        
        # Serialize and deserialize
        data = workspace1.to_dict()
        workspace2 = GlobalWorkspace.from_dict(data)
        
        # Verify equivalence
        assert workspace2.capacity == workspace1.capacity
        assert len(workspace2.current_goals) == len(workspace1.current_goals)
        assert len(workspace2.active_percepts) == len(workspace1.active_percepts)
        assert workspace2.emotional_state["valence"] == workspace1.emotional_state["valence"]
        assert workspace2.emotional_state["arousal"] == workspace1.emotional_state["arousal"]
        assert workspace2.cycle_count == workspace1.cycle_count
    
    @given(percepts(), goals())
    @settings(max_examples=100)
    def test_clear_resets_state(self, percept, goal):
        """Property: clear() resets workspace to initial state."""
        workspace = GlobalWorkspace()
        
        # Add state
        workspace.active_percepts[percept.id] = percept
        workspace.add_goal(goal)
        workspace.emotional_state["valence"] = 0.8
        workspace.cycle_count = 5
        
        # Clear
        workspace.clear()
        
        # Verify reset to initial state
        assert len(workspace.current_goals) == 0
        assert len(workspace.active_percepts) == 0
        assert len(workspace.attended_memories) == 0
        assert workspace.emotional_state["valence"] == 0.0
        assert workspace.emotional_state["arousal"] == 0.0
        assert workspace.emotional_state["dominance"] == 0.0
        assert workspace.cycle_count == 0
    
    @given(percept_lists, goal_lists)
    @settings(max_examples=50)
    def test_broadcast_reflects_current_state(self, percepts_list, goals_list):
        """Property: broadcast() returns a snapshot reflecting current state."""
        workspace = GlobalWorkspace()
        
        # Add state
        unique_percepts = make_unique_by_id(percepts_list)
        for percept in unique_percepts[:5]:
            workspace.active_percepts[percept.id] = percept
        
        unique_goals = make_unique_by_id(goals_list)
        for goal in unique_goals[:5]:
            workspace.add_goal(goal)
        
        # Get snapshot
        snapshot = workspace.broadcast()
        
        # Verify snapshot reflects current state
        assert len(snapshot.percepts) == len(workspace.active_percepts)
        assert len(snapshot.goals) == len(workspace.current_goals)
        assert isinstance(snapshot, WorkspaceSnapshot)
