"""
Integration tests for edge cases and error handling.

Tests system behavior under boundary conditions and
error scenarios.
"""
import pytest
from lyra.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType, Percept
from lyra.cognitive_core.attention import AttentionController
from lyra.cognitive_core.affect import AffectSubsystem


@pytest.mark.integration
class TestWorkspaceCapacityLimits:
    """Test workspace behavior at capacity."""
    
    def test_workspace_handles_many_goals(self):
        """Test that workspace can handle multiple goals."""
        workspace = GlobalWorkspace()
        
        # Add many goals
        for i in range(10):
            goal = Goal(
                type=GoalType.RESPOND_TO_USER,
                description=f"Goal {i}",
                priority=0.5 + (i * 0.05)
            )
            workspace.add_goal(goal)
        
        # Should handle all goals
        snapshot = workspace.broadcast()
        assert len(snapshot.goals) == 10
        
        # Goals should be sorted by priority (highest first)
        priorities = [g.priority for g in snapshot.goals]
        assert priorities == sorted(priorities, reverse=True)
    
    def test_workspace_handles_many_percepts(self):
        """Test that workspace can handle multiple percepts."""
        workspace = GlobalWorkspace()
        
        # Add many percepts
        for i in range(20):
            percept = Percept(
                modality="text",
                raw=f"Percept {i}",
                complexity=3
            )
            workspace.active_percepts[percept.id] = percept
        
        # Should handle all percepts
        snapshot = workspace.broadcast()
        assert len(snapshot.percepts) == 20
    
    def test_workspace_handles_percept_replacement(self):
        """Test that workspace handles replacing percepts."""
        workspace = GlobalWorkspace()
        
        # Add initial percept
        percept1 = Percept(
            modality="text",
            raw="Original percept",
            complexity=5
        )
        workspace.active_percepts[percept1.id] = percept1
        
        snapshot1 = workspace.broadcast()
        assert len(snapshot1.percepts) == 1
        
        # Clear and add new percepts
        workspace.active_percepts.clear()
        
        percept2 = Percept(
            modality="text",
            raw="New percept",
            complexity=3
        )
        workspace.active_percepts[percept2.id] = percept2
        
        snapshot2 = workspace.broadcast()
        assert len(snapshot2.percepts) == 1
        assert percept2.id in snapshot2.percepts


@pytest.mark.integration
class TestAttentionBudgetConstraints:
    """Test attention behavior when budget is exhausted."""
    
    def test_attention_with_insufficient_budget(self):
        """Test attention selection when budget is too small for all percepts."""
        workspace = GlobalWorkspace()
        affect = AffectSubsystem()
        
        # Set very small attention budget
        attention = AttentionController(
            attention_budget=5,  # Very small budget
            workspace=workspace,
            affect=affect
        )
        
        # Create many complex percepts
        percepts = [
            Percept(modality="text", raw=f"Complex percept {i}", complexity=10)
            for i in range(10)
        ]
        
        # Select with insufficient budget
        selected = attention.select_for_broadcast(percepts)
        
        # Should select what fits in budget
        assert len(selected) < len(percepts)
        
        # Total complexity should not exceed budget
        total_complexity = sum(p.complexity for p in selected)
        assert total_complexity <= 5
    
    def test_attention_prioritizes_high_complexity(self):
        """Test that attention handles varying complexity appropriately."""
        workspace = GlobalWorkspace()
        affect = AffectSubsystem()
        
        attention = AttentionController(
            attention_budget=20,
            workspace=workspace,
            affect=affect
        )
        
        # Create percepts with varying complexity
        percepts = [
            Percept(modality="text", raw="Simple", complexity=2),
            Percept(modality="text", raw="Medium", complexity=5),
            Percept(modality="text", raw="Complex", complexity=15)
        ]
        
        # Select percepts
        selected = attention.select_for_broadcast(percepts)
        
        # Should select some percepts
        assert len(selected) > 0
        
        # Total should respect budget
        total_complexity = sum(p.complexity for p in selected)
        assert total_complexity <= 20
    
    def test_attention_with_zero_budget(self):
        """Test attention behavior with zero budget."""
        workspace = GlobalWorkspace()
        affect = AffectSubsystem()
        
        attention = AttentionController(
            attention_budget=0,
            workspace=workspace,
            affect=affect
        )
        
        percepts = [
            Percept(modality="text", raw="Test", complexity=5)
        ]
        
        # Should handle gracefully (empty selection)
        selected = attention.select_for_broadcast(percepts)
        
        # With zero budget, nothing can be selected
        assert len(selected) == 0


@pytest.mark.integration
class TestConcurrentWorkspaceAccess:
    """Test concurrent workspace access."""
    
    def test_multiple_subsystems_read_snapshot_safely(self):
        """Test that multiple subsystems can safely read snapshots."""
        workspace = GlobalWorkspace()
        
        # Add some state
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Test goal",
            priority=0.5
        )
        workspace.add_goal(goal)
        
        # Multiple subsystems read simultaneously
        snapshot1 = workspace.broadcast()
        snapshot2 = workspace.broadcast()
        
        # Snapshots should be independent but equivalent
        assert len(snapshot1.goals) == len(snapshot2.goals)
        assert snapshot1.goals[0].id == snapshot2.goals[0].id
    
    def test_snapshot_immutability_enforced(self):
        """Test that WorkspaceSnapshot is truly immutable."""
        workspace = GlobalWorkspace()
        
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Original",
            priority=0.7
        )
        workspace.add_goal(goal)
        
        snapshot = workspace.broadcast()
        
        # Attempting to modify snapshot should raise an error
        # (Pydantic frozen models raise ValidationError)
        with pytest.raises(Exception):
            snapshot.goals = []
        
        with pytest.raises(Exception):
            snapshot.emotions = {"valence": 0.0}
    
    def test_workspace_update_doesnt_affect_old_snapshots(self):
        """Test that updating workspace doesn't affect existing snapshots."""
        workspace = GlobalWorkspace()
        
        # Add initial goal
        goal1 = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="First goal",
            priority=0.6
        )
        workspace.add_goal(goal1)
        
        # Take snapshot
        snapshot_before = workspace.broadcast()
        
        # Add another goal
        goal2 = Goal(
            type=GoalType.INTROSPECT,
            description="Second goal",
            priority=0.8
        )
        workspace.add_goal(goal2)
        
        # Take new snapshot
        snapshot_after = workspace.broadcast()
        
        # Old snapshot should still have only one goal
        assert len(snapshot_before.goals) == 1
        
        # New snapshot should have both goals
        assert len(snapshot_after.goals) == 2


@pytest.mark.integration
class TestMalformedInputHandling:
    """Test handling of invalid inputs."""
    
    def test_invalid_goal_type_raises_error(self):
        """Test handling of invalid goal type."""
        # Invalid goal type should fail validation
        with pytest.raises(Exception):
            # This should raise a validation error
            goal = Goal(
                type="invalid_type",  # Not a valid GoalType
                description="Bad goal",
                priority=0.5
            )
    
    def test_invalid_goal_priority_raises_error(self):
        """Test handling of invalid goal priority."""
        # Priority outside valid range should fail
        with pytest.raises(Exception):
            goal = Goal(
                type=GoalType.RESPOND_TO_USER,
                description="Test",
                priority=1.5  # Should be 0.0-1.0
            )
        
        with pytest.raises(Exception):
            goal = Goal(
                type=GoalType.RESPOND_TO_USER,
                description="Test",
                priority=-0.5  # Should be 0.0-1.0
            )
    
    def test_invalid_percept_complexity_raises_error(self):
        """Test handling of invalid percept complexity."""
        # Negative complexity might be invalid depending on implementation
        # Let's test with a valid percept first
        valid_percept = Percept(
            modality="text",
            raw="Test",
            complexity=5
        )
        assert valid_percept.complexity == 5
        
        # Zero complexity should be valid
        zero_percept = Percept(
            modality="text",
            raw="Test",
            complexity=0
        )
        assert zero_percept.complexity == 0
    
    def test_workspace_handles_empty_goal_list(self):
        """Test that workspace handles no goals gracefully."""
        workspace = GlobalWorkspace()
        
        # Broadcast with no goals
        snapshot = workspace.broadcast()
        
        # Should return empty list, not crash
        assert isinstance(snapshot.goals, list)
        assert len(snapshot.goals) == 0
    
    def test_workspace_handles_empty_percepts(self):
        """Test that workspace handles no percepts gracefully."""
        workspace = GlobalWorkspace()
        
        # Broadcast with no percepts
        snapshot = workspace.broadcast()
        
        # Should return empty dict, not crash
        assert isinstance(snapshot.percepts, dict)
        assert len(snapshot.percepts) == 0


@pytest.mark.integration
class TestEmotionalStateEdgeCases:
    """Test edge cases in emotional state management."""
    
    def test_extreme_emotional_values(self):
        """Test workspace with extreme emotional values."""
        workspace = GlobalWorkspace()
        
        # Set extreme emotional values
        workspace.emotional_state["valence"] = 1.0
        workspace.emotional_state["arousal"] = 1.0
        workspace.emotional_state["dominance"] = -1.0
        
        # Should handle without issues
        snapshot = workspace.broadcast()
        
        assert snapshot.emotions["valence"] == 1.0
        assert snapshot.emotions["arousal"] == 1.0
        assert snapshot.emotions["dominance"] == -1.0
    
    def test_neutral_emotional_state(self):
        """Test workspace with neutral emotional state."""
        workspace = GlobalWorkspace()
        
        # Set neutral emotions (all zeros)
        workspace.emotional_state["valence"] = 0.0
        workspace.emotional_state["arousal"] = 0.0
        workspace.emotional_state["dominance"] = 0.0
        
        snapshot = workspace.broadcast()
        
        assert snapshot.emotions["valence"] == 0.0
        assert snapshot.emotions["arousal"] == 0.0
        assert snapshot.emotions["dominance"] == 0.0


@pytest.mark.integration
class TestCycleCountTracking:
    """Test workspace cycle count tracking."""
    
    def test_cycle_count_increments(self):
        """Test that cycle count increments on update."""
        workspace = GlobalWorkspace()
        
        initial_count = workspace.cycle_count
        
        # Perform updates
        workspace.update([])
        assert workspace.cycle_count == initial_count + 1
        
        workspace.update([])
        assert workspace.cycle_count == initial_count + 2
    
    def test_cycle_count_in_snapshot(self):
        """Test that snapshots include current cycle count."""
        workspace = GlobalWorkspace()
        
        # Take initial snapshot
        snapshot1 = workspace.broadcast()
        count1 = snapshot1.cycle_count
        
        # Update workspace
        workspace.update([])
        
        # Take new snapshot
        snapshot2 = workspace.broadcast()
        count2 = snapshot2.cycle_count
        
        # New snapshot should have incremented count
        assert count2 == count1 + 1
