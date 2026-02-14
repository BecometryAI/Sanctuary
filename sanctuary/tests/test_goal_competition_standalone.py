"""
Standalone tests for Goal Competition System

These tests can run without the full Sanctuary system dependencies.
"""

import sys
import os
import importlib.util

# Direct import without going through package __init__
base_path = os.path.join(os.path.dirname(__file__), 'emergence_core', 'sanctuary', 'cognitive_core', 'goals')

def import_module_from_path(module_name, file_path):
    """Import a module directly from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import modules directly
resources = import_module_from_path('resources', os.path.join(base_path, 'resources.py'))
competition = import_module_from_path('competition', os.path.join(base_path, 'competition.py'))
interactions = import_module_from_path('interactions', os.path.join(base_path, 'interactions.py'))
metrics = import_module_from_path('metrics', os.path.join(base_path, 'metrics.py'))

CognitiveResources = resources.CognitiveResources
ResourcePool = resources.ResourcePool
GoalCompetition = competition.GoalCompetition
ActiveGoal = competition.ActiveGoal
GoalInteraction = interactions.GoalInteraction
GoalCompetitionMetrics = metrics.GoalCompetitionMetrics
MetricsTracker = metrics.MetricsTracker

import pytest
from datetime import datetime, timedelta


# Mock Goal class for testing
class MockGoal:
    """Mock goal for testing."""
    def __init__(
        self,
        goal_id: str,
        importance: float = 0.5,
        progress: float = 0.0,
        resource_needs: CognitiveResources = None,
        deadline: datetime = None,
        subgoal_ids: list = None,
        metadata: dict = None
    ):
        self.id = goal_id
        self.importance = importance
        self.progress = progress
        self.resource_needs = resource_needs or CognitiveResources(
            attention_budget=0.2,
            processing_budget=0.2,
            action_budget=0.2,
            time_budget=0.2
        )
        self.deadline = deadline
        self.subgoal_ids = subgoal_ids or []
        self.metadata = metadata or {}


def test_resource_allocation():
    """Test basic resource allocation."""
    pool = ResourcePool()
    request = CognitiveResources(0.3, 0.3, 0.3, 0.3)
    
    granted = pool.allocate("goal1", request)
    
    assert granted.attention_budget == 0.3
    assert pool.resources.attention_budget == 0.7
    print("✓ Resource allocation works")


def test_resource_release():
    """Test releasing resources."""
    pool = ResourcePool()
    pool.allocate("goal1", CognitiveResources(0.5, 0.5, 0.5, 0.5))
    
    assert pool.resources.attention_budget == 0.5
    
    pool.release("goal1")
    
    assert pool.resources.attention_budget == 1.0
    print("✓ Resource release works")


def test_goal_competition():
    """Test goal competition dynamics."""
    comp = GoalCompetition()
    high_priority = MockGoal("high", importance=0.9)
    low_priority = MockGoal("low", importance=0.2)
    
    activations = comp.compete([high_priority, low_priority])
    
    assert activations["high"] > activations["low"]
    print("✓ Goal competition works - high priority wins")


def test_resource_constraints():
    """Test that resource constraints limit concurrent goals."""
    comp = GoalCompetition()
    pool = ResourcePool()
    
    # Create goals with high resource needs
    goals = [
        MockGoal(
            f"g{i}",
            importance=0.8,
            resource_needs=CognitiveResources(0.5, 0.5, 0.5, 0.5)
        )
        for i in range(5)
    ]
    
    active = comp.select_active_goals(goals, pool)
    
    # Should only activate 2 goals due to resource limits
    assert len(active) <= 2
    print(f"✓ Resource constraints work - only {len(active)} of 5 goals activated")


def test_goal_interference():
    """Test goal interference detection."""
    tracker = GoalInteraction()
    
    goal1 = MockGoal(
        "g1",
        resource_needs=CognitiveResources(0.8, 0.8, 0.8, 0.8)
    )
    goal2 = MockGoal(
        "g2",
        resource_needs=CognitiveResources(0.8, 0.8, 0.8, 0.8)
    )
    
    interactions = tracker.compute_interactions([goal1, goal2])
    
    # Should have negative interaction (interference)
    interaction = interactions[("g1", "g2")]
    assert interaction < 0
    print("✓ Goal interference detection works")


def test_goal_facilitation():
    """Test goal facilitation detection."""
    tracker = GoalInteraction()
    
    goal1 = MockGoal("g1", subgoal_ids=["s1", "s2", "s3"])
    goal2 = MockGoal("g2", subgoal_ids=["s2", "s3", "s4"])
    
    interactions = tracker.compute_interactions([goal1, goal2])
    
    # Should have positive interaction (facilitation)
    interaction = interactions[("g1", "g2")]
    assert interaction > 0
    print("✓ Goal facilitation detection works")


def test_metrics_tracking():
    """Test competition metrics tracking."""
    tracker = MetricsTracker()
    
    metrics1 = GoalCompetitionMetrics(active_goals=2, waiting_goals=1)
    metrics2 = GoalCompetitionMetrics(active_goals=3, waiting_goals=0)
    
    tracker.record(metrics1)
    tracker.record(metrics2)
    
    assert len(tracker.history) == 2
    latest = tracker.get_latest()
    assert latest.active_goals == 3
    print("✓ Metrics tracking works")


def test_goal_switches():
    """Test tracking goal switches."""
    tracker = MetricsTracker()
    
    tracker.track_goal_switch("g1")
    tracker.track_goal_switch("g1")
    assert tracker.get_goal_switches() == 0  # Same goal
    
    tracker.track_goal_switch("g2")
    assert tracker.get_goal_switches() == 1  # Switch detected
    print("✓ Goal switch tracking works")


def test_integration():
    """Test full integration: competition -> selection -> allocation."""
    comp = GoalCompetition(inhibition_strength=0.3)
    pool = ResourcePool()
    
    goals = [
        MockGoal("urgent", importance=0.9, deadline=datetime.now() + timedelta(hours=1)),
        MockGoal("important", importance=0.8),
        MockGoal("low_priority", importance=0.3),
    ]
    
    active = comp.select_active_goals(goals, pool)
    
    # Highest priority should be first
    assert len(active) > 0
    assert active[0].goal.id == "urgent"
    
    # Resources should be allocated
    for active_goal in active:
        assert active_goal.resources.total() > 0
        alloc = pool.get_allocation(active_goal.goal.id)
        assert alloc is not None
    
    print(f"✓ Integration test passed - {len(active)} goals activated")


if __name__ == "__main__":
    print("\nRunning Goal Competition System Tests\n" + "=" * 50)
    
    test_resource_allocation()
    test_resource_release()
    test_goal_competition()
    test_resource_constraints()
    test_goal_interference()
    test_goal_facilitation()
    test_metrics_tracking()
    test_goal_switches()
    test_integration()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓\n")
