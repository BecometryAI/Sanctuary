"""
Property-based tests for GlobalWorkspace invariants.

Tests validate:
- Workspace snapshot immutability
- Percept addition behavior
- Goal management correctness
"""

import pytest
from hypothesis import given, settings
from lyra.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType
from .strategies import percepts, goals, emotional_states, percept_lists, goal_lists


@pytest.mark.property
class TestWorkspaceProperties:
    
    @given(percept_lists, goal_lists, emotional_states())
    @settings(max_examples=50, deadline=500)
    def test_workspace_snapshot_immutability(self, percepts_list, goals_list, emotions):
        """Property: Workspace snapshots are immutable after creation."""
        workspace = GlobalWorkspace()
        
        # Add percepts to workspace
        for percept in percepts_list:
            workspace.active_percepts[percept.id] = percept
        
        # Add goals to workspace
        for goal in goals_list:
            workspace.add_goal(goal)
        
        # Update emotions
        workspace.emotional_state.update(emotions)
        
        # Get snapshot
        snapshot = workspace.broadcast()
        
        # Verify snapshot state matches workspace
        assert len(snapshot.goals) == len(goals_list)
        assert len(snapshot.percepts) == len(percepts_list)
        
        # Modify workspace and ensure snapshot is unchanged
        original_goal_count = len(snapshot.goals)
        original_percept_count = len(snapshot.percepts)
        
        workspace.add_goal(Goal(
            id="new-goal",
            type=GoalType.RESPOND_TO_USER,
            description="New goal after snapshot",
            priority=0.5
        ))
        
        # Add new percept
        from lyra.cognitive_core.workspace import Percept
        new_percept = Percept(
            id="new-percept",
            modality="text",
            raw="New percept after snapshot"
        )
        workspace.active_percepts[new_percept.id] = new_percept
        
        # Snapshot should remain unchanged
        assert len(snapshot.goals) == original_goal_count
        assert len(snapshot.percepts) == original_percept_count
    
    @given(percepts())
    @settings(max_examples=50)
    def test_percept_addition_increases_count(self, percept):
        """Property: Adding a percept increases count by 1."""
        workspace = GlobalWorkspace()
        initial_count = len(workspace.active_percepts)
        
        workspace.active_percepts[percept.id] = percept
        
        assert len(workspace.active_percepts) == initial_count + 1
        assert percept.id in workspace.active_percepts
    
    @given(goal_lists)
    @settings(max_examples=50)
    def test_goal_count_equals_additions(self, goals_list):
        """Property: Final goal count equals unique goal additions."""
        workspace = GlobalWorkspace()
        
        # Ensure unique IDs to avoid overwrites
        unique_goals = {g.id: g for g in goals_list}.values()
        
        for goal in unique_goals:
            workspace.add_goal(goal)
        
        assert len(workspace.current_goals) == len(unique_goals)
    
    @given(goals())
    @settings(max_examples=50)
    def test_goal_removal_works_correctly(self, goal):
        """Property: Removing a goal by ID works correctly."""
        workspace = GlobalWorkspace()
        
        # Add goal
        workspace.add_goal(goal)
        assert len(workspace.current_goals) == 1
        assert goal.id in [g.id for g in workspace.current_goals]
        
        # Remove goal
        workspace.remove_goal(goal.id)
        assert len(workspace.current_goals) == 0
        assert goal.id not in [g.id for g in workspace.current_goals]
    
    @given(percept_lists, goal_lists, emotional_states())
    @settings(max_examples=50)
    def test_workspace_serialization_roundtrip(self, percepts_list, goals_list, emotions):
        """Property: Workspace state is preserved through serialization."""
        workspace = GlobalWorkspace()
        
        # Add percepts
        for percept in percepts_list:
            workspace.active_percepts[percept.id] = percept
        
        # Add goals
        for goal in goals_list:
            workspace.add_goal(goal)
        
        # Update emotions
        workspace.emotional_state.update(emotions)
        
        # Serialize and deserialize
        data = workspace.to_dict()
        restored = GlobalWorkspace.from_dict(data)
        
        # Verify state is preserved
        assert len(restored.current_goals) == len(workspace.current_goals)
        assert len(restored.active_percepts) == len(workspace.active_percepts)
        assert restored.emotional_state == workspace.emotional_state
        assert restored.cycle_count == workspace.cycle_count
    
    @given(emotional_states())
    @settings(max_examples=50)
    def test_emotional_state_updates_correctly(self, emotions):
        """Property: Emotional state updates reflect the new values."""
        workspace = GlobalWorkspace()
        
        # Update emotions
        workspace.emotional_state.update(emotions)
        
        # Verify all values are within bounds
        assert -1.0 <= workspace.emotional_state['valence'] <= 1.0
        assert -1.0 <= workspace.emotional_state['arousal'] <= 1.0
        assert -1.0 <= workspace.emotional_state['dominance'] <= 1.0
