"""
Tests for Goal Competition System

Tests cover:
- Resource allocation and release
- Goal competition with lateral inhibition
- Resource constraints limiting concurrent goals
- Goal interference and facilitation
- Metrics tracking
"""

import pytest
from datetime import datetime, timedelta

from mind.cognitive_core.goals.resources import CognitiveResources, ResourcePool
from mind.cognitive_core.goals.competition import GoalCompetition, ActiveGoal
from mind.cognitive_core.goals.interactions import GoalInteraction
from mind.cognitive_core.goals.metrics import GoalCompetitionMetrics, MetricsTracker


# Test fixtures

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


# ============================================================================
# Resource Pool Tests
# ============================================================================

class TestCognitiveResources:
    """Test CognitiveResources dataclass."""
    
    def test_creation_default(self):
        """Test creating resources with defaults."""
        resources = CognitiveResources()
        assert resources.attention_budget == 1.0
        assert resources.processing_budget == 1.0
        assert resources.action_budget == 1.0
        assert resources.time_budget == 1.0
        assert resources.total() == 4.0
    
    def test_creation_custom(self):
        """Test creating resources with custom values."""
        resources = CognitiveResources(
            attention_budget=0.5,
            processing_budget=0.3,
            action_budget=0.2,
            time_budget=0.1
        )
        assert resources.attention_budget == 0.5
        assert resources.total() == 1.1
    
    def test_negative_validation(self):
        """Test that negative values raise ValueError."""
        with pytest.raises(ValueError, match="attention_budget must be >= 0"):
            CognitiveResources(attention_budget=-0.1)
        
        with pytest.raises(ValueError, match="processing_budget must be >= 0"):
            CognitiveResources(processing_budget=-0.1)
    
    def test_is_empty(self):
        """Test empty resource detection."""
        empty = CognitiveResources(0, 0, 0, 0)
        assert empty.is_empty()
        
        not_empty = CognitiveResources(0.1, 0, 0, 0)
        assert not not_empty.is_empty()


class TestResourcePool:
    """Test ResourcePool allocation."""
    
    def test_initialization(self):
        """Test pool initialization."""
        pool = ResourcePool()
        assert pool.resources.total() == 4.0
        assert len(pool.allocations) == 0
    
    def test_allocate_full_request(self):
        """Test allocating when resources are available."""
        pool = ResourcePool()
        request = CognitiveResources(0.3, 0.3, 0.3, 0.3)
        
        granted = pool.allocate("goal1", request)
        
        assert granted.attention_budget == 0.3
        assert granted.total() == 1.2
        assert pool.resources.attention_budget == 0.7
        assert "goal1" in pool.allocations
    
    def test_allocate_partial_request(self):
        """Test allocating when only partial resources available."""
        pool = ResourcePool()
        # Exhaust most attention
        pool.allocate("goal1", CognitiveResources(0.9, 0, 0, 0))
        
        # Request more attention than available
        request = CognitiveResources(0.5, 0, 0, 0)
        granted = pool.allocate("goal2", request)
        
        # Should only get what's available (0.1)
        assert abs(granted.attention_budget - 0.1) < 1e-9
        assert abs(pool.resources.attention_budget) < 1e-9
    
    def test_allocate_duplicate_goal(self):
        """Test that allocating to same goal twice raises error."""
        pool = ResourcePool()
        pool.allocate("goal1", CognitiveResources(0.2, 0, 0, 0))
        
        with pytest.raises(ValueError, match="already has allocated resources"):
            pool.allocate("goal1", CognitiveResources(0.1, 0, 0, 0))
    
    def test_release(self):
        """Test releasing resources."""
        pool = ResourcePool()
        request = CognitiveResources(0.3, 0.3, 0.3, 0.3)
        pool.allocate("goal1", request)
        
        assert pool.resources.attention_budget == 0.7
        
        success = pool.release("goal1")
        
        assert success
        assert pool.resources.attention_budget == 1.0
        assert "goal1" not in pool.allocations
    
    def test_release_nonexistent(self):
        """Test releasing goal that has no allocation."""
        pool = ResourcePool()
        success = pool.release("nonexistent")
        
        assert not success
    
    def test_get_allocation(self):
        """Test retrieving allocation."""
        pool = ResourcePool()
        request = CognitiveResources(0.3, 0, 0, 0)
        pool.allocate("goal1", request)
        
        alloc = pool.get_allocation("goal1")
        assert alloc is not None
        assert alloc.attention_budget == 0.3
        
        alloc = pool.get_allocation("nonexistent")
        assert alloc is None
    
    def test_can_allocate(self):
        """Test checking if allocation is possible."""
        pool = ResourcePool()
        
        # Should be able to allocate small request
        small_request = CognitiveResources(0.2, 0.2, 0.2, 0.2)
        assert pool.can_allocate(small_request)
        
        # Exhaust resources
        pool.allocate("goal1", CognitiveResources(1.0, 1.0, 1.0, 1.0))
        
        # Should not be able to allocate any more
        assert not pool.can_allocate(CognitiveResources(0.1, 0, 0, 0))
    
    def test_utilization(self):
        """Test resource utilization calculation."""
        pool = ResourcePool()
        assert pool.utilization() == 0.0
        
        pool.allocate("goal1", CognitiveResources(0.5, 0.5, 0.5, 0.5))
        # 2.0 out of 4.0 = 0.5
        assert pool.utilization() == 0.5
        
        pool.allocate("goal2", CognitiveResources(0.5, 0.5, 0.5, 0.5))
        # 4.0 out of 4.0 = 1.0
        assert pool.utilization() == 1.0
    
    def test_reset(self):
        """Test resetting pool."""
        pool = ResourcePool()
        pool.allocate("goal1", CognitiveResources(0.5, 0.5, 0.5, 0.5))
        pool.allocate("goal2", CognitiveResources(0.3, 0.3, 0.3, 0.3))
        
        pool.reset()
        
        assert pool.resources.total() == 4.0
        assert len(pool.allocations) == 0


# ============================================================================
# Goal Competition Tests
# ============================================================================

class TestGoalCompetition:
    """Test goal competition dynamics."""
    
    def test_initialization(self):
        """Test competition system initialization."""
        comp = GoalCompetition(inhibition_strength=0.4)
        assert comp.inhibition_strength == 0.4
    
    def test_initialization_invalid_strength(self):
        """Test that invalid inhibition strength raises error."""
        with pytest.raises(ValueError, match="inhibition_strength must be in"):
            GoalCompetition(inhibition_strength=1.5)
    
    def test_compete_single_goal(self):
        """Test competition with single goal."""
        comp = GoalCompetition()
        goal = MockGoal("g1", importance=0.8)
        
        activations = comp.compete([goal])
        
        assert "g1" in activations
        assert 0 <= activations["g1"] <= 1
    
    def test_compete_high_priority_wins(self):
        """Test that high-priority goals get higher activation."""
        comp = GoalCompetition()
        high_priority = MockGoal("high", importance=0.9)
        low_priority = MockGoal("low", importance=0.2)
        
        activations = comp.compete([high_priority, low_priority])
        
        assert activations["high"] > activations["low"]
    
    def test_compete_with_inhibition(self):
        """Test that conflicting goals inhibit each other."""
        comp = GoalCompetition(inhibition_strength=0.5)
        
        # Create conflicting goals (same resource needs)
        goal1 = MockGoal(
            "g1",
            importance=0.7,
            resource_needs=CognitiveResources(0.8, 0.8, 0.8, 0.8)
        )
        goal2 = MockGoal(
            "g2",
            importance=0.7,
            resource_needs=CognitiveResources(0.8, 0.8, 0.8, 0.8)
        )
        
        activations = comp.compete([goal1, goal2], iterations=20)
        
        # Both should have reduced activation due to inhibition
        # At least one should be suppressed significantly
        assert min(activations.values()) < 0.5
    
    def test_compete_urgency_factor(self):
        """Test that urgent goals (near deadline) get higher activation."""
        comp = GoalCompetition()
        
        urgent = MockGoal(
            "urgent",
            importance=0.5,
            deadline=datetime.now() + timedelta(minutes=30)  # Very soon
        )
        not_urgent = MockGoal(
            "not_urgent",
            importance=0.5,
            deadline=datetime.now() + timedelta(days=7)  # Week away
        )
        
        activations = comp.compete([urgent, not_urgent])
        
        assert activations["urgent"] > activations["not_urgent"]
    
    def test_select_active_goals_with_resources(self):
        """Test selecting active goals based on resources."""
        comp = GoalCompetition()
        pool = ResourcePool()
        
        goal1 = MockGoal("g1", importance=0.9)
        goal2 = MockGoal("g2", importance=0.7)
        goal3 = MockGoal("g3", importance=0.5)
        
        active = comp.select_active_goals([goal1, goal2, goal3], pool)
        
        # Should select goals in priority order until resources exhausted
        assert len(active) > 0
        assert active[0].goal.id == "g1"  # Highest priority first
    
    def test_select_active_goals_resource_limit(self):
        """Test that resource constraints limit concurrent goals."""
        comp = GoalCompetition()
        pool = ResourcePool()
        
        # Create goals with high resource needs
        goals = [
            MockGoal(
                f"g{i}",
                importance=0.8 - i * 0.1,
                resource_needs=CognitiveResources(0.5, 0.5, 0.5, 0.5)
            )
            for i in range(5)
        ]
        
        active = comp.select_active_goals(goals, pool)
        
        # With 1.0 in each resource dimension and needs of 0.5 each,
        # should only be able to activate 2 goals
        assert len(active) <= 2
    
    def test_select_active_goals_max_active(self):
        """Test max_active parameter limits selection."""
        comp = GoalCompetition()
        pool = ResourcePool()
        
        goals = [MockGoal(f"g{i}", importance=0.5) for i in range(10)]
        
        active = comp.select_active_goals(goals, pool, max_active=3)
        
        assert len(active) <= 3


# ============================================================================
# Goal Interaction Tests
# ============================================================================

class TestGoalInteraction:
    """Test goal interference and facilitation."""
    
    def test_initialization(self):
        """Test interaction tracker initialization."""
        tracker = GoalInteraction()
        assert len(tracker._interaction_cache) == 0
    
    def test_compute_interactions_empty(self):
        """Test computing interactions with no goals."""
        tracker = GoalInteraction()
        interactions = tracker.compute_interactions([])
        assert len(interactions) == 0
    
    def test_compute_interactions_shared_subgoals(self):
        """Test that shared subgoals create facilitation."""
        tracker = GoalInteraction()
        
        goal1 = MockGoal("g1", subgoal_ids=["s1", "s2", "s3"])
        goal2 = MockGoal("g2", subgoal_ids=["s2", "s3", "s4"])
        
        interactions = tracker.compute_interactions([goal1, goal2])
        
        # Should have positive interaction (facilitation)
        interaction = interactions[("g1", "g2")]
        assert interaction > 0
    
    def test_compute_interactions_resource_conflict(self):
        """Test that resource overlap creates interference."""
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
    
    def test_compute_interactions_explicit_conflict(self):
        """Test explicit conflict in metadata."""
        tracker = GoalInteraction()
        
        goal1 = MockGoal("g1", metadata={"conflicts_with": ["g2"]})
        goal2 = MockGoal("g2")
        
        interactions = tracker.compute_interactions([goal1, goal2])
        
        # Should have strong negative interaction
        interaction = interactions[("g1", "g2")]
        assert interaction < -0.3
    
    def test_compute_interactions_caching(self):
        """Test that interactions are cached."""
        tracker = GoalInteraction()
        
        goal1 = MockGoal("g1")
        goal2 = MockGoal("g2")
        
        # First computation
        tracker.compute_interactions([goal1, goal2])
        assert len(tracker._interaction_cache) == 1
        
        # Second computation should use cache
        tracker.compute_interactions([goal1, goal2])
        assert len(tracker._interaction_cache) == 1
    
    def test_get_facilitating_goals(self):
        """Test finding facilitating goals."""
        tracker = GoalInteraction()
        
        target = MockGoal("target", subgoal_ids=["s1", "s2"])
        facilitator = MockGoal("helper", subgoal_ids=["s1", "s2", "s3"])
        unrelated = MockGoal("unrelated", subgoal_ids=["s4"])
        
        all_goals = [target, facilitator, unrelated]
        facilitating = tracker.get_facilitating_goals(target, all_goals, threshold=0.1)
        
        assert len(facilitating) >= 1
        assert facilitating[0][0].id == "helper"
    
    def test_get_interfering_goals(self):
        """Test finding interfering goals."""
        tracker = GoalInteraction()
        
        target = MockGoal(
            "target",
            resource_needs=CognitiveResources(0.8, 0.8, 0.8, 0.8)
        )
        interferer = MockGoal(
            "interferer",
            resource_needs=CognitiveResources(0.8, 0.8, 0.8, 0.8)
        )
        compatible = MockGoal(
            "compatible",
            resource_needs=CognitiveResources(0.1, 0.1, 0.1, 0.1)
        )
        
        all_goals = [target, interferer, compatible]
        interfering = tracker.get_interfering_goals(target, all_goals, threshold=-0.1)
        
        assert len(interfering) >= 1
        assert interfering[0][0].id == "interferer"
    
    def test_clear_cache(self):
        """Test clearing interaction cache."""
        tracker = GoalInteraction()
        
        goal1 = MockGoal("g1")
        goal2 = MockGoal("g2")
        
        tracker.compute_interactions([goal1, goal2])
        assert len(tracker._interaction_cache) > 0
        
        tracker.clear_cache()
        assert len(tracker._interaction_cache) == 0


# ============================================================================
# Metrics Tests
# ============================================================================

class TestGoalCompetitionMetrics:
    """Test competition metrics."""
    
    def test_creation(self):
        """Test creating metrics."""
        metrics = GoalCompetitionMetrics(
            active_goals=3,
            waiting_goals=2,
            total_resource_utilization=0.75
        )
        
        assert metrics.active_goals == 3
        assert metrics.waiting_goals == 2
        assert metrics.total_resource_utilization == 0.75
    
    def test_validation(self):
        """Test metrics validation."""
        with pytest.raises(ValueError, match="active_goals must be >= 0"):
            GoalCompetitionMetrics(active_goals=-1)
        
        with pytest.raises(ValueError, match="total_resource_utilization must be in"):
            GoalCompetitionMetrics(total_resource_utilization=1.5)
    
    def test_to_dict(self):
        """Test converting metrics to dict."""
        metrics = GoalCompetitionMetrics(
            active_goals=2,
            resource_conflicts=[("g1", "g2", 0.8)]
        )
        
        data = metrics.to_dict()
        
        assert data["active_goals"] == 2
        assert len(data["resource_conflicts"]) == 1
        assert data["resource_conflicts"][0]["goal1"] == "g1"
    
    def test_from_dict(self):
        """Test creating metrics from dict."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "active_goals": 3,
            "waiting_goals": 1,
            "total_resource_utilization": 0.6,
            "resource_conflicts": [
                {"goal1": "g1", "goal2": "g2", "level": 0.7}
            ]
        }
        
        metrics = GoalCompetitionMetrics.from_dict(data)
        
        assert metrics.active_goals == 3
        assert metrics.waiting_goals == 1
        assert len(metrics.resource_conflicts) == 1


class TestMetricsTracker:
    """Test metrics tracking over time."""
    
    def test_initialization(self):
        """Test tracker initialization."""
        tracker = MetricsTracker(max_history=50)
        assert tracker.max_history == 50
        assert len(tracker.history) == 0
    
    def test_record(self):
        """Test recording metrics."""
        tracker = MetricsTracker()
        metrics = GoalCompetitionMetrics(active_goals=2)
        
        tracker.record(metrics)
        
        assert len(tracker.history) == 1
        assert tracker.history[0].active_goals == 2
    
    def test_record_max_history(self):
        """Test that history is trimmed to max_history."""
        tracker = MetricsTracker(max_history=3)
        
        for i in range(5):
            tracker.record(GoalCompetitionMetrics(active_goals=i))
        
        assert len(tracker.history) == 3
        # Should keep most recent 3
        assert tracker.history[-1].active_goals == 4
    
    def test_track_goal_switch(self):
        """Test tracking goal switches."""
        tracker = MetricsTracker()
        
        tracker.track_goal_switch("g1")
        assert tracker.get_goal_switches() == 0  # First goal, no switch
        
        tracker.track_goal_switch("g1")
        assert tracker.get_goal_switches() == 0  # Same goal, no switch
        
        tracker.track_goal_switch("g2")
        assert tracker.get_goal_switches() == 1  # Different goal, switch
        
        tracker.track_goal_switch("g1")
        assert tracker.get_goal_switches() == 2  # Another switch
    
    def test_get_average_utilization(self):
        """Test calculating average utilization."""
        tracker = MetricsTracker()
        
        tracker.record(GoalCompetitionMetrics(total_resource_utilization=0.5))
        tracker.record(GoalCompetitionMetrics(total_resource_utilization=0.7))
        tracker.record(GoalCompetitionMetrics(total_resource_utilization=0.9))
        
        avg = tracker.get_average_utilization()
        assert abs(avg - 0.7) < 0.01
    
    def test_get_latest(self):
        """Test getting latest metrics."""
        tracker = MetricsTracker()
        
        with pytest.raises(IndexError):
            tracker.get_latest()
        
        tracker.record(GoalCompetitionMetrics(active_goals=5))
        latest = tracker.get_latest()
        
        assert latest.active_goals == 5
    
    def test_clear(self):
        """Test clearing tracker."""
        tracker = MetricsTracker()
        tracker.record(GoalCompetitionMetrics(active_goals=2))
        tracker.track_goal_switch("g1")
        tracker.track_goal_switch("g2")
        
        tracker.clear()
        
        assert len(tracker.history) == 0
        assert tracker.get_goal_switches() == 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestGoalCompetitionIntegration:
    """Integration tests for the complete system."""
    
    def test_full_competition_cycle(self):
        """Test complete cycle: competition -> selection -> allocation."""
        comp = GoalCompetition(inhibition_strength=0.3)
        pool = ResourcePool()
        
        # Create diverse goals
        goals = [
            MockGoal("urgent", importance=0.9, deadline=datetime.now() + timedelta(hours=1)),
            MockGoal("important", importance=0.8),
            MockGoal("low_priority", importance=0.3),
        ]
        
        # Select active goals
        active = comp.select_active_goals(goals, pool)
        
        # Verify highest priority goals are selected
        assert len(active) > 0
        assert active[0].goal.id == "urgent"
        
        # Verify resources are allocated
        for active_goal in active:
            assert active_goal.resources.total() > 0
            alloc = pool.get_allocation(active_goal.goal.id)
            assert alloc is not None
    
    def test_resource_constraint_enforcement(self):
        """Test that resource constraints truly limit concurrent goals."""
        comp = GoalCompetition()
        pool = ResourcePool()
        
        # Create many goals with significant resource needs
        goals = [
            MockGoal(
                f"goal{i}",
                importance=0.5,
                resource_needs=CognitiveResources(0.4, 0.4, 0.4, 0.4)
            )
            for i in range(10)
        ]
        
        active = comp.select_active_goals(goals, pool)
        
        # With resources of 1.0 each and needs of 0.4 each,
        # should only activate 2 goals (0.8 out of 1.0)
        assert len(active) <= 2
        
        # Verify total allocated doesn't exceed available
        total_allocated = pool.total_allocated()
        assert total_allocated <= 4.0  # Initial total
    
    def test_competition_with_interactions(self):
        """Test competition considering goal interactions."""
        comp = GoalCompetition(inhibition_strength=0.5)
        interaction = GoalInteraction()
        pool = ResourcePool()
        
        # Create goals with shared subgoals (facilitation)
        goal1 = MockGoal("g1", importance=0.7, subgoal_ids=["s1", "s2"])
        goal2 = MockGoal("g2", importance=0.7, subgoal_ids=["s1", "s2"])
        
        # Create conflicting goal (interference)
        goal3 = MockGoal(
            "g3",
            importance=0.7,
            resource_needs=CognitiveResources(0.9, 0.9, 0.9, 0.9)
        )
        
        all_goals = [goal1, goal2, goal3]
        
        # Compute interactions
        interactions = interaction.compute_interactions(all_goals)
        
        # g1 and g2 should facilitate each other
        assert interactions[("g1", "g2")] > 0
        
        # Run competition
        activations = comp.compete(all_goals)
        
        # All should have some activation
        assert all(act > 0 for act in activations.values())
    
    def test_metrics_tracking_during_competition(self):
        """Test tracking metrics during goal competition."""
        comp = GoalCompetition()
        pool = ResourcePool()
        tracker = MetricsTracker()
        
        goals = [
            MockGoal("g1", importance=0.9),
            MockGoal("g2", importance=0.7),
            MockGoal("g3", importance=0.5),
        ]
        
        # First cycle
        active1 = comp.select_active_goals(goals, pool)
        metrics1 = GoalCompetitionMetrics(
            active_goals=len(active1),
            waiting_goals=len(goals) - len(active1),
            total_resource_utilization=pool.utilization()
        )
        tracker.record(metrics1)
        tracker.track_goal_switch(active1[0].goal.id if active1 else "")
        
        # Release resources
        for ag in active1:
            pool.release(ag.goal.id)
        
        # Change priorities
        goals[2].importance = 0.95  # Low priority becomes highest
        
        # Second cycle
        active2 = comp.select_active_goals(goals, pool)
        metrics2 = GoalCompetitionMetrics(
            active_goals=len(active2),
            waiting_goals=len(goals) - len(active2),
            total_resource_utilization=pool.utilization()
        )
        tracker.record(metrics2)
        tracker.track_goal_switch(active2[0].goal.id if active2 else "")
        
        # Verify tracking
        assert len(tracker.history) == 2
        assert tracker.get_goal_switches() >= 1  # Priority changed


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_empty_resource_allocation(self):
        """Test allocating zero resources."""
        pool = ResourcePool()
        empty = CognitiveResources(0, 0, 0, 0)
        granted = pool.allocate("g1", empty)
        
        assert granted.total() == 0.0
        assert pool.utilization() == 0.0
    
    def test_over_allocation_request(self):
        """Test requesting more resources than available."""
        pool = ResourcePool()
        huge_request = CognitiveResources(10.0, 10.0, 10.0, 10.0)
        granted = pool.allocate("g1", huge_request)
        
        # Should only grant what's available
        assert granted.attention_budget == 1.0
        assert granted.processing_budget == 1.0
        assert pool.utilization() == 1.0
    
    def test_competition_with_zero_iterations(self):
        """Test competition with invalid iteration count."""
        comp = GoalCompetition()
        goals = [MockGoal("g1")]
        
        with pytest.raises(ValueError, match="iterations must be >= 1"):
            comp.compete(goals, iterations=0)
    
    def test_competition_with_identical_goals(self):
        """Test competition when all goals have same priority."""
        comp = GoalCompetition()
        goals = [MockGoal(f"g{i}", importance=0.5) for i in range(5)]
        
        activations = comp.compete(goals)
        
        # All should have similar activation (no clear winner)
        values = list(activations.values())
        assert max(values) - min(values) < 0.3  # Small variance
    
    def test_resource_pool_with_zero_initial(self):
        """Test pool initialized with zero resources."""
        empty_resources = CognitiveResources(0, 0, 0, 0)
        pool = ResourcePool(empty_resources)
        
        assert pool.utilization() == 0.0
        assert not pool.can_allocate(CognitiveResources(0.1, 0, 0, 0))
    
    def test_negative_resource_validation(self):
        """Test that negative resources are rejected."""
        with pytest.raises(ValueError, match="must be >= 0"):
            CognitiveResources(-0.1, 0, 0, 0)
    
    def test_interaction_with_single_goal(self):
        """Test interaction computation with only one goal."""
        tracker = GoalInteraction()
        goal = MockGoal("g1")
        
        interactions = tracker.compute_interactions([goal])
        
        # Single goal has no interactions
        assert len(interactions) == 0
    
    def test_metrics_with_invalid_utilization(self):
        """Test metrics validation with out-of-range utilization."""
        with pytest.raises(ValueError, match="total_resource_utilization must be in"):
            GoalCompetitionMetrics(total_resource_utilization=1.5)
        
        with pytest.raises(ValueError, match="total_resource_utilization must be in"):
            GoalCompetitionMetrics(total_resource_utilization=-0.1)
    
    def test_release_nonexistent_goal(self):
        """Test releasing resources for goal that was never allocated."""
        pool = ResourcePool()
        result = pool.release("nonexistent_goal")
        
        assert result is False
        assert pool.utilization() == 0.0
    
    def test_double_allocation_prevention(self):
        """Test that allocating to same goal twice raises error."""
        pool = ResourcePool()
        pool.allocate("g1", CognitiveResources(0.2, 0.2, 0.2, 0.2))
        
        with pytest.raises(ValueError, match="already has allocated resources"):
            pool.allocate("g1", CognitiveResources(0.1, 0.1, 0.1, 0.1))
    
    def test_goal_without_id_attribute(self):
        """Test handling goals without proper ID attribute."""
        comp = GoalCompetition()
        bad_goal = object()  # No 'id' attribute
        
        with pytest.raises(ValueError, match="Cannot extract ID"):
            comp._get_goal_id(bad_goal)
    
    def test_inhibition_strength_validation(self):
        """Test inhibition strength must be in valid range."""
        with pytest.raises(ValueError, match="inhibition_strength must be in"):
            GoalCompetition(inhibition_strength=1.5)
        
        with pytest.raises(ValueError, match="inhibition_strength must be in"):
            GoalCompetition(inhibition_strength=-0.1)
    
    def test_activation_clamping(self):
        """Test that activation values are properly clamped to [0,1]."""
        comp = GoalCompetition(inhibition_strength=0.0)  # No inhibition
        
        # Create goal with extreme importance
        goal = MockGoal("extreme", importance=1.0)
        
        activations = comp.compete([goal])
        
        # Should be clamped to <= 1.0
        assert 0.0 <= activations["extreme"] <= 1.0
    
    def test_pool_reset_with_allocations(self):
        """Test resetting pool releases all allocations."""
        pool = ResourcePool()
        pool.allocate("g1", CognitiveResources(0.3, 0.3, 0.3, 0.3))
        pool.allocate("g2", CognitiveResources(0.2, 0.2, 0.2, 0.2))
        
        assert pool.utilization() > 0
        
        pool.reset()
        
        assert pool.utilization() == 0.0
        assert len(pool.allocations) == 0
    
    def test_metrics_tracker_history_limit(self):
        """Test that metrics history respects max_history limit."""
        tracker = MetricsTracker(max_history=3)
        
        for i in range(10):
            tracker.record(GoalCompetitionMetrics(active_goals=i))
        
        # Should only keep last 3
        assert len(tracker.history) == 3
        assert tracker.history[-1].active_goals == 9
    
    def test_empty_goals_list(self):
        """Test handling empty goals list."""
        comp = GoalCompetition()
        pool = ResourcePool()
        
        activations = comp.compete([])
        assert activations == {}
        
        active = comp.select_active_goals([], pool)
        assert active == []
