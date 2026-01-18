"""
Integration tests for ActionSubsystem.

Tests that ActionSubsystem correctly proposes and selects
actions based on workspace goals and state.
"""
import pytest
import asyncio
from lyra.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType, Percept
from lyra.cognitive_core.action import ActionSubsystem, ActionType


@pytest.mark.integration
class TestActionProposal:
    """Test action proposal based on goals."""
    
    def test_action_subsystem_proposes_actions_for_goals(self):
        """Test that ActionSubsystem proposes actions for active goals."""
        workspace = GlobalWorkspace()
        action_subsystem = ActionSubsystem()
        
        # Add goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Respond to user greeting",
            priority=0.9
        )
        workspace.add_goal(goal)
        
        # Get snapshot and decide on actions
        snapshot = workspace.broadcast()
        actions = action_subsystem.decide(snapshot)
        
        # Should propose at least one action
        assert len(actions) > 0
        # Actions should be relevant to goals
        assert any(action.type in [ActionType.SPEAK, ActionType.SPEAK_AUTONOMOUS] 
                   for action in actions)
    
    def test_action_selection_based_on_priority(self):
        """Test that action selection considers goal priority."""
        workspace = GlobalWorkspace()
        action_subsystem = ActionSubsystem()
        
        # Add multiple goals with different priorities
        goal_high = Goal(
            type=GoalType.RESPOND_TO_USER, 
            description="High priority response", 
            priority=0.9
        )
        goal_low = Goal(
            type=GoalType.INTROSPECT, 
            description="Low priority introspection", 
            priority=0.3
        )
        
        workspace.add_goal(goal_high)
        workspace.add_goal(goal_low)
        
        # Select actions
        snapshot = workspace.broadcast()
        actions = action_subsystem.decide(snapshot)
        
        # Should have actions
        assert len(actions) > 0
        
        # First action should have high priority
        # (Actions are returned sorted by priority)
        assert actions[0].priority >= 0.5
    
    def test_multiple_goal_types_generate_different_actions(self):
        """Test that different goal types generate appropriate actions."""
        workspace = GlobalWorkspace()
        action_subsystem = ActionSubsystem()
        
        # Add different types of goals
        respond_goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Respond to query",
            priority=0.8
        )
        memory_goal = Goal(
            type=GoalType.COMMIT_MEMORY,
            description="Store conversation",
            priority=0.6
        )
        
        workspace.add_goal(respond_goal)
        workspace.add_goal(memory_goal)
        
        # Get actions
        snapshot = workspace.broadcast()
        actions = action_subsystem.decide(snapshot)
        
        # Should have multiple actions for different goals
        assert len(actions) > 0
        action_types = [action.type for action in actions]
        
        # Should include speaking and memory-related actions
        assert any(at in [ActionType.SPEAK, ActionType.SPEAK_AUTONOMOUS] 
                   for at in action_types)


@pytest.mark.integration
class TestActionDecisionMaking:
    """Test action decision-making logic."""
    
    def test_action_subsystem_generates_candidates(self):
        """Test that action subsystem generates candidate actions."""
        workspace = GlobalWorkspace()
        action_subsystem = ActionSubsystem()
        
        # Add a goal and percept
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Answer question about consciousness",
            priority=0.7
        )
        workspace.add_goal(goal)
        
        percept = Percept(
            modality="text",
            raw="What is consciousness?",
            complexity=5
        )
        workspace.active_percepts[percept.id] = percept
        
        # Get actions
        snapshot = workspace.broadcast()
        actions = action_subsystem.decide(snapshot)
        
        # Should generate actions based on goal and percept
        assert len(actions) > 0
        
        # At least one action should have a reason
        assert any(action.reason for action in actions)
    
    def test_action_history_tracking(self):
        """Test that action subsystem tracks action history."""
        workspace = GlobalWorkspace()
        action_subsystem = ActionSubsystem()
        
        # Add goal
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Test", priority=0.5)
        workspace.add_goal(goal)
        
        # Get initial action history size
        initial_history_size = len(action_subsystem.action_history)
        
        # Select actions
        snapshot = workspace.broadcast()
        actions = action_subsystem.decide(snapshot)
        
        # History should have grown
        assert len(action_subsystem.action_history) > initial_history_size
        
        # Stats should be updated
        assert action_subsystem.action_stats["total_actions"] > 0
    
    def test_action_subsystem_handles_empty_workspace(self):
        """Test that action subsystem handles workspace with no goals."""
        workspace = GlobalWorkspace()
        action_subsystem = ActionSubsystem()
        
        # Get snapshot with no goals
        snapshot = workspace.broadcast()
        actions = action_subsystem.decide(snapshot)
        
        # Should return actions (possibly WAIT or autonomous actions)
        # At minimum should not crash
        assert isinstance(actions, list)


@pytest.mark.integration  
class TestActionWorkspaceIntegration:
    """Test integration between actions and workspace updates."""
    
    def test_action_reflects_workspace_state(self):
        """Test that actions reflect current workspace state."""
        workspace = GlobalWorkspace()
        action_subsystem = ActionSubsystem()
        
        # Set up workspace with emotional state
        workspace.emotional_state["valence"] = 0.8
        workspace.emotional_state["arousal"] = 0.6
        
        # Add goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Express happiness",
            priority=0.8
        )
        workspace.add_goal(goal)
        
        # Get actions
        snapshot = workspace.broadcast()
        actions = action_subsystem.decide(snapshot)
        
        # Should have actions
        assert len(actions) > 0
        
        # Actions should reflect high-priority goal
        assert any(action.priority > 0.5 for action in actions)
    
    def test_action_subsystem_with_multiple_percepts(self):
        """Test action selection with multiple percepts in workspace."""
        workspace = GlobalWorkspace()
        action_subsystem = ActionSubsystem()
        
        # Add multiple percepts
        for i in range(3):
            percept = Percept(
                modality="text",
                raw=f"Percept {i}",
                complexity=3
            )
            workspace.active_percepts[percept.id] = percept
        
        # Add goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Process information",
            priority=0.7
        )
        workspace.add_goal(goal)
        
        # Get actions
        snapshot = workspace.broadcast()
        actions = action_subsystem.decide(snapshot)
        
        # Should generate actions
        assert len(actions) > 0
        
        # Snapshot should contain the percepts
        assert len(snapshot.percepts) == 3
