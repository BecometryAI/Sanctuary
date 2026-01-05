"""
Integration tests for GlobalWorkspace broadcasting and state management.

Tests that the workspace correctly broadcasts immutable snapshots
and integrates updates from subsystems.
"""
import pytest
from datetime import datetime

from lyra.cognitive_core.workspace import (
    GlobalWorkspace, 
    Goal, 
    GoalType, 
    Percept,
    Memory,
    WorkspaceSnapshot
)


@pytest.mark.integration
class TestWorkspaceBroadcasting:
    """Test workspace broadcast mechanism."""
    
    def test_broadcast_returns_immutable_snapshot(self):
        """Test that broadcast() returns an immutable WorkspaceSnapshot."""
        workspace = GlobalWorkspace()
        
        # Add some state
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Test goal",
            priority=0.8
        )
        workspace.add_goal(goal)
        
        # Get snapshot
        snapshot = workspace.broadcast()
        
        # Verify it's a WorkspaceSnapshot (Pydantic frozen model)
        assert isinstance(snapshot, WorkspaceSnapshot)
        
        # Verify immutability - should raise ValidationError if we try to modify attributes
        with pytest.raises(Exception):  # Pydantic raises ValidationError for frozen models
            snapshot.goals = []
    
    def test_broadcast_contains_all_state(self):
        """Test that broadcast includes goals, percepts, emotions, memories."""
        workspace = GlobalWorkspace()
        
        # Add goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Test goal",
            priority=0.8
        )
        workspace.add_goal(goal)
        
        # Add percept
        percept = Percept(
            modality="text",
            raw="Hello world",
            complexity=5
        )
        workspace.active_percepts[percept.id] = percept
        
        # Update emotion
        workspace.emotional_state["valence"] = 0.5
        
        # Add memory
        memory = Memory(
            id="mem-1",
            content="Test memory",
            timestamp=datetime.now(),
            significance=0.7
        )
        workspace.attended_memories.append(memory)
        
        # Broadcast
        snapshot = workspace.broadcast()
        
        # Verify all state present
        assert len(snapshot.goals) == 1
        assert snapshot.goals[0].description == "Test goal"
        
        assert len(snapshot.percepts) == 1
        assert percept.id in snapshot.percepts
        
        assert snapshot.emotions["valence"] == 0.5
        
        assert len(snapshot.memories) == 1
        assert snapshot.memories[0]["content"] == "Test memory"
    
    def test_snapshot_isolation(self):
        """Test that modifying workspace doesn't affect existing snapshots."""
        workspace = GlobalWorkspace()
        
        # Create initial state
        goal1 = Goal(type=GoalType.RESPOND_TO_USER, description="Goal 1", priority=0.5)
        workspace.add_goal(goal1)
        
        # Take snapshot
        snapshot1 = workspace.broadcast()
        
        # Modify workspace
        goal2 = Goal(type=GoalType.INTROSPECT, description="Goal 2", priority=0.8)
        workspace.add_goal(goal2)
        
        # Take second snapshot
        snapshot2 = workspace.broadcast()
        
        # Verify snapshots are isolated
        assert len(snapshot1.goals) == 1
        assert snapshot1.goals[0].description == "Goal 1"
        
        assert len(snapshot2.goals) == 2
        assert snapshot2.goals[0].description == "Goal 2"  # Sorted by priority


@pytest.mark.integration
class TestWorkspaceUpdate:
    """Test workspace update mechanism."""
    
    def test_update_handles_goal_addition(self):
        """Test that update() correctly adds goals."""
        workspace = GlobalWorkspace()
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Test", priority=0.5)
        
        updates = [
            {'type': 'goal', 'data': goal}
        ]
        
        workspace.update(updates)
        
        assert len(workspace.current_goals) == 1
        assert workspace.current_goals[0].description == "Test"
    
    def test_update_handles_percept_addition(self):
        """Test that update() correctly adds percepts."""
        workspace = GlobalWorkspace()
        
        percept = Percept(modality="text", raw="Hello", complexity=3)
        
        updates = [
            {'type': 'percept', 'data': percept}
        ]
        
        workspace.update(updates)
        
        assert len(workspace.active_percepts) == 1
        assert percept.id in workspace.active_percepts
    
    def test_update_handles_emotion_update(self):
        """Test that update() correctly updates emotions."""
        workspace = GlobalWorkspace()
        
        emotion_update = {'valence': 0.7, 'arousal': 0.3}
        
        updates = [
            {'type': 'emotion', 'data': emotion_update}
        ]
        
        workspace.update(updates)
        
        assert workspace.emotional_state['valence'] == 0.7
        assert workspace.emotional_state['arousal'] == 0.3
    
    def test_update_increments_cycle_count(self):
        """Test that update() increments the cycle counter."""
        workspace = GlobalWorkspace()
        
        initial_cycle = workspace.cycle_count
        
        workspace.update([])
        
        assert workspace.cycle_count == initial_cycle + 1
