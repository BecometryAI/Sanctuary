"""
Unit tests for executive_function.py (Element 4)

Tests cover:
- Goal creation and management
- Priority handling and ranking
- Action dependencies and sequencing
- Decision tree evaluation
- Edge cases and error handling
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mind.executive_function import (
    ExecutiveFunction,
    Goal,
    Action,
    DecisionNode,
    GoalStatus,
    ActionStatus,
    DecisionType
)


class TestGoal:
    """Test Goal dataclass validation and methods"""
    
    def test_goal_creation_valid(self):
        """Test creating a valid goal"""
        goal = Goal(
            id="test_goal",
            description="Test goal description",
            priority=0.8
        )
        
        assert goal.id == "test_goal"
        assert goal.description == "Test goal description"
        assert goal.priority == 0.8
        assert goal.status == GoalStatus.PENDING
        assert goal.progress == 0.0
    
    def test_goal_invalid_priority(self):
        """Test goal creation fails with invalid priority"""
        with pytest.raises(ValueError, match="Priority must be in"):
            Goal(id="test", description="Test", priority=1.5)
        
        with pytest.raises(ValueError, match="Priority must be in"):
            Goal(id="test", description="Test", priority=-0.1)
    
    def test_goal_invalid_progress(self):
        """Test goal creation fails with invalid progress"""
        with pytest.raises(ValueError, match="Progress must be in"):
            Goal(id="test", description="Test", priority=0.5, progress=1.2)
    
    def test_goal_empty_description(self):
        """Test goal creation fails with empty description"""
        with pytest.raises(ValueError, match="description cannot be empty"):
            Goal(id="test", description="", priority=0.5)
        
        with pytest.raises(ValueError, match="description cannot be empty"):
            Goal(id="test", description="   ", priority=0.5)
    
    def test_goal_empty_id(self):
        """Test goal creation fails with empty ID"""
        with pytest.raises(ValueError, match="ID cannot be empty"):
            Goal(id="", description="Test", priority=0.5)
    
    def test_goal_serialization(self):
        """Test goal to_dict and from_dict"""
        original = Goal(
            id="goal1",
            description="Test goal",
            priority=0.7,
            deadline=datetime(2025, 12, 31)
        )
        
        # Convert to dict
        data = original.to_dict()
        
        # Recreate from dict
        restored = Goal.from_dict(data)
        
        assert restored.id == original.id
        assert restored.description == original.description
        assert restored.priority == original.priority
        assert restored.deadline == original.deadline


class TestAction:
    """Test Action dataclass validation and methods"""
    
    def test_action_creation_valid(self):
        """Test creating a valid action"""
        action = Action(
            id="action1",
            description="Do something",
            goal_id="goal1"
        )
        
        assert action.id == "action1"
        assert action.description == "Do something"
        assert action.goal_id == "goal1"
        assert action.status == ActionStatus.PENDING
        assert action.dependencies == []
    
    def test_action_empty_description(self):
        """Test action creation fails with empty description"""
        with pytest.raises(ValueError, match="description cannot be empty"):
            Action(id="a1", description="", goal_id="g1")
    
    def test_action_empty_goal_id(self):
        """Test action creation fails without goal"""
        with pytest.raises(ValueError, match="must be associated with a goal"):
            Action(id="a1", description="Test", goal_id="")
    
    def test_action_with_dependencies(self):
        """Test action with dependencies"""
        action = Action(
            id="a1",
            description="Test",
            goal_id="g1",
            dependencies=["a0"]
        )
        
        assert action.dependencies == ["a0"]
    
    def test_action_serialization(self):
        """Test action to_dict and from_dict"""
        original = Action(
            id="a1",
            description="Test action",
            goal_id="g1",
            dependencies=["a0"],
            estimated_duration=timedelta(minutes=30)
        )
        
        data = original.to_dict()
        restored = Action.from_dict(data)
        
        assert restored.id == original.id
        assert restored.goal_id == original.goal_id
        assert restored.dependencies == original.dependencies
        assert restored.estimated_duration == original.estimated_duration


class TestDecisionNode:
    """Test DecisionNode validation"""
    
    def test_decision_creation_valid(self):
        """Test creating a valid decision"""
        decision = DecisionNode(
            id="d1",
            question="Should we proceed?",
            decision_type=DecisionType.BINARY,
            options=["yes", "no"]
        )
        
        assert decision.question == "Should we proceed?"
        assert len(decision.options) == 2
    
    def test_decision_empty_question(self):
        """Test decision fails with empty question"""
        with pytest.raises(ValueError, match="question cannot be empty"):
            DecisionNode(
                id="d1",
                question="",
                decision_type=DecisionType.BINARY,
                options=["yes", "no"]
            )
    
    def test_decision_insufficient_options(self):
        """Test decision fails with too few options"""
        with pytest.raises(ValueError, match="at least 2 options"):
            DecisionNode(
                id="d1",
                question="Test?",
                decision_type=DecisionType.BINARY,
                options=["only_one"]
            )
    
    def test_decision_invalid_confidence(self):
        """Test decision fails with invalid confidence"""
        with pytest.raises(ValueError, match="Confidence must be in"):
            DecisionNode(
                id="d1",
                question="Test?",
                decision_type=DecisionType.BINARY,
                options=["yes", "no"],
                confidence=1.5
            )


class TestExecutiveFunction:
    """Test ExecutiveFunction core functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)
    
    @pytest.fixture
    def exec_func(self, temp_dir):
        """Create ExecutiveFunction instance"""
        return ExecutiveFunction(persistence_dir=temp_dir)
    
    # ========================================================================
    # Goal Management Tests
    # ========================================================================
    
    def test_create_goal(self, exec_func):
        """Test goal creation"""
        goal = exec_func.create_goal(
            description="Test goal",
            priority=0.8
        )
        
        assert goal.id in exec_func.goals
        assert goal.description == "Test goal"
        assert goal.priority == 0.8
    
    def test_create_goal_with_custom_id(self, exec_func):
        """Test goal creation with custom ID"""
        goal = exec_func.create_goal(
            description="Test",
            priority=0.5,
            goal_id="custom_id"
        )
        
        assert goal.id == "custom_id"
    
    def test_create_hierarchical_goal(self, exec_func):
        """Test creating parent-child goals"""
        parent = exec_func.create_goal(
            description="Parent goal",
            priority=0.9
        )
        
        child = exec_func.create_goal(
            description="Child goal",
            priority=0.7,
            parent_goal_id=parent.id
        )
        
        assert child.parent_goal_id == parent.id
        assert child.id in parent.subgoal_ids
    
    def test_create_goal_invalid_parent(self, exec_func):
        """Test creating goal with non-existent parent fails"""
        with pytest.raises(ValueError, match="Parent goal .* does not exist"):
            exec_func.create_goal(
                description="Child",
                priority=0.5,
                parent_goal_id="nonexistent"
            )
    
    def test_update_goal_priority(self, exec_func):
        """Test updating goal priority"""
        goal = exec_func.create_goal("Test", priority=0.5)
        
        updated = exec_func.update_goal_priority(goal.id, 0.9)
        
        assert updated.priority == 0.9
        assert exec_func.goals[goal.id].priority == 0.9
    
    def test_update_goal_priority_invalid(self, exec_func):
        """Test updating priority to invalid value fails"""
        goal = exec_func.create_goal("Test", priority=0.5)
        
        with pytest.raises(ValueError, match="Priority must be in"):
            exec_func.update_goal_priority(goal.id, 1.5)
    
    def test_update_goal_status(self, exec_func):
        """Test updating goal status"""
        goal = exec_func.create_goal("Test", priority=0.5)
        
        updated = exec_func.update_goal_status(goal.id, GoalStatus.ACTIVE)
        
        assert updated.status == GoalStatus.ACTIVE
        assert goal.id in exec_func._active_goals
    
    def test_get_top_priority_goals(self, exec_func):
        """Test retrieving top priority goals"""
        # Create goals with different priorities
        g1 = exec_func.create_goal("Low priority", priority=0.3)
        g2 = exec_func.create_goal("High priority", priority=0.9)
        g3 = exec_func.create_goal("Medium priority", priority=0.6)
        
        # Make them all active
        for gid in [g1.id, g2.id, g3.id]:
            exec_func.update_goal_status(gid, GoalStatus.ACTIVE)
        
        # Get top 2
        top_goals = exec_func.get_top_priority_goals(n=2, active_only=True)
        
        assert len(top_goals) == 2
        assert top_goals[0].id == g2.id  # Highest priority
        assert top_goals[1].id == g3.id  # Second highest
    
    def test_get_top_priority_goals_inactive_filter(self, exec_func):
        """Test active_only filter works"""
        g1 = exec_func.create_goal("Active", priority=0.9)
        g2 = exec_func.create_goal("Inactive", priority=0.8)
        
        exec_func.update_goal_status(g1.id, GoalStatus.ACTIVE)
        # g2 stays pending
        
        active_goals = exec_func.get_top_priority_goals(n=5, active_only=True)
        all_goals = exec_func.get_top_priority_goals(n=5, active_only=False)
        
        assert len(active_goals) == 1
        assert len(all_goals) == 2
    
    # ========================================================================
    # Action Management Tests
    # ========================================================================
    
    def test_create_action(self, exec_func):
        """Test action creation"""
        goal = exec_func.create_goal("Test goal", priority=0.5)
        
        action = exec_func.create_action(
            description="Do something",
            goal_id=goal.id
        )
        
        assert action.id in exec_func.actions
        assert action.goal_id == goal.id
    
    def test_create_action_invalid_goal(self, exec_func):
        """Test action creation fails with non-existent goal"""
        with pytest.raises(ValueError, match="Goal .* does not exist"):
            exec_func.create_action(
                description="Test",
                goal_id="nonexistent"
            )
    
    def test_create_action_with_dependencies(self, exec_func):
        """Test creating actions with dependencies"""
        goal = exec_func.create_goal("Test", priority=0.5)
        
        a1 = exec_func.create_action("Step 1", goal_id=goal.id)
        a2 = exec_func.create_action(
            "Step 2",
            goal_id=goal.id,
            dependencies=[a1.id]
        )
        
        assert a1.id in a2.dependencies
    
    def test_create_action_invalid_dependency(self, exec_func):
        """Test action creation fails with non-existent dependency"""
        goal = exec_func.create_goal("Test", priority=0.5)
        
        with pytest.raises(ValueError, match="Dependency action .* does not exist"):
            exec_func.create_action(
                "Test",
                goal_id=goal.id,
                dependencies=["nonexistent"]
            )
    
    def test_create_action_circular_dependency(self, exec_func):
        """Test action cannot depend on itself"""
        goal = exec_func.create_goal("Test", priority=0.5)
        
        # Create first action
        a1 = exec_func.create_action(
            "Step 1",
            goal_id=goal.id,
            action_id="a1"
        )
        
        # Try to create action that depends on itself (using existing action as ID)
        # This should be caught by self-dependency validation
        with pytest.raises(ValueError, match="cannot depend on itself"):
            exec_func.create_action(
                "Self-dependent",
                goal_id=goal.id,
                action_id="a1",  # Reusing existing ID
                dependencies=["a1"]  # Self-dependency
            )
    
    def test_get_ready_actions(self, exec_func):
        """Test getting ready actions"""
        goal = exec_func.create_goal("Test", priority=0.5)
        
        a1 = exec_func.create_action("Step 1", goal_id=goal.id)
        a2 = exec_func.create_action(
            "Step 2",
            goal_id=goal.id,
            dependencies=[a1.id]
        )
        
        # Initially, only a1 is ready (no dependencies)
        ready = exec_func.get_ready_actions()
        assert len(ready) == 1
        assert ready[0].id == a1.id
        
        # Complete a1
        exec_func.actions[a1.id].status = ActionStatus.COMPLETED
        
        # Now a2 should be ready
        ready = exec_func.get_ready_actions()
        assert len(ready) == 1
        assert ready[0].id == a2.id
    
    def test_get_action_sequence(self, exec_func):
        """Test topological sort for action ordering"""
        goal = exec_func.create_goal("Test", priority=0.5)
        
        # Create action DAG:
        # a1 -> a2 -> a4
        # a1 -> a3 -> a4
        a1 = exec_func.create_action("Step 1", goal_id=goal.id, action_id="a1")
        a2 = exec_func.create_action("Step 2", goal_id=goal.id, action_id="a2", dependencies=["a1"])
        a3 = exec_func.create_action("Step 3", goal_id=goal.id, action_id="a3", dependencies=["a1"])
        a4 = exec_func.create_action("Step 4", goal_id=goal.id, action_id="a4", dependencies=["a2", "a3"])
        
        sequence = exec_func.get_action_sequence(goal.id)
        
        # Verify sequence structure
        assert len(sequence) == 3
        assert sequence[0] == ["a1"]  # First batch: a1
        assert set(sequence[1]) == {"a2", "a3"}  # Second batch: a2 and a3 (parallel)
        assert sequence[2] == ["a4"]  # Third batch: a4
    
    def test_get_action_sequence_circular_dependency(self, exec_func):
        """Test circular dependency detection in sequence"""
        goal = exec_func.create_goal("Test", priority=0.5)
        
        # Create actions
        a1 = exec_func.create_action("Step 1", goal_id=goal.id, action_id="a1")
        a2 = exec_func.create_action("Step 2", goal_id=goal.id, action_id="a2", dependencies=["a1"])
        
        # Manually create circular dependency (a1 depends on a2)
        exec_func.actions["a1"].dependencies = ["a2"]
        
        # Should detect cycle
        with pytest.raises(ValueError, match="Circular dependencies detected"):
            exec_func.get_action_sequence(goal.id)
    
    # ========================================================================
    # Decision Making Tests
    # ========================================================================
    
    def test_create_decision(self, exec_func):
        """Test decision creation"""
        decision = exec_func.create_decision(
            question="Should we proceed?",
            options=["yes", "no"]
        )
        
        assert decision.id in [d.id for d in exec_func.decisions]
        assert decision.question == "Should we proceed?"
    
    def test_create_binary_decision_validation(self, exec_func):
        """Test binary decision must have exactly 2 options"""
        with pytest.raises(ValueError, match="exactly 2 options"):
            exec_func.create_decision(
                question="Test?",
                options=["a", "b", "c"],
                decision_type=DecisionType.BINARY
            )
    
    def test_evaluate_decision_default(self, exec_func):
        """Test decision evaluation with default scoring"""
        decision = exec_func.create_decision(
            question="Choose one",
            options=["A", "B", "C"]
        )
        
        selected, confidence, rationale = exec_func.evaluate_decision(decision.id)
        
        assert selected in ["A", "B", "C"]
        assert 0 <= confidence <= 1
        assert isinstance(rationale, str)
    
    def test_evaluate_decision_custom_scoring(self, exec_func):
        """Test decision evaluation with custom scoring function"""
        decision = exec_func.create_decision(
            question="Pick best option",
            options=["low", "medium", "high"]
        )
        
        # Custom scoring: prefer "high"
        def score_func(opt):
            scores = {"low": 0.2, "medium": 0.5, "high": 1.0}
            return scores.get(opt, 0.0)
        
        selected, confidence, _ = exec_func.evaluate_decision(
            decision.id,
            scoring_function=score_func
        )
        
        assert selected == "high"
        assert confidence > 0.5
    
    # ========================================================================
    # Persistence Tests
    # ========================================================================
    
    def test_save_and_load_state(self, temp_dir):
        """Test state persistence"""
        # Create instance and add data
        exec1 = ExecutiveFunction(persistence_dir=temp_dir)
        
        goal = exec1.create_goal("Test goal", priority=0.8)
        action = exec1.create_action("Test action", goal_id=goal.id)
        decision = exec1.create_decision("Test?", options=["yes", "no"])
        
        # Save state
        exec1.save_state()
        
        # Create new instance (should load state)
        exec2 = ExecutiveFunction(persistence_dir=temp_dir)
        
        # Verify data loaded
        assert goal.id in exec2.goals
        assert action.id in exec2.actions
        assert len(exec2.decisions) == 1
        
        # Verify goal data
        loaded_goal = exec2.goals[goal.id]
        assert loaded_goal.description == "Test goal"
        assert loaded_goal.priority == 0.8
    
    def test_save_state_no_persistence_dir(self):
        """Test save_state with no persistence directory"""
        exec_func = ExecutiveFunction(persistence_dir=None)
        
        # Should log warning but not crash
        exec_func.save_state()
    
    # ========================================================================
    # Statistics Tests
    # ========================================================================
    
    def test_get_statistics(self, exec_func):
        """Test statistics generation"""
        # Create some data
        g1 = exec_func.create_goal("Goal 1", priority=0.9)
        g2 = exec_func.create_goal("Goal 2", priority=0.7)
        exec_func.update_goal_status(g1.id, GoalStatus.ACTIVE)
        
        a1 = exec_func.create_action("Action 1", goal_id=g1.id)
        a2 = exec_func.create_action("Action 2", goal_id=g2.id)
        
        d1 = exec_func.create_decision("Decision 1", options=["a", "b"])
        
        stats = exec_func.get_statistics()
        
        assert stats["total_goals"] == 2
        assert stats["active_goals"] == 1
        assert stats["total_actions"] == 2
        assert stats["total_decisions"] == 1


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_goal_list(self):
        """Test behavior with no goals"""
        exec_func = ExecutiveFunction()
        
        top_goals = exec_func.get_top_priority_goals(n=5)
        assert top_goals == []
    
    def test_invalid_n_parameter(self):
        """Test invalid n parameter"""
        exec_func = ExecutiveFunction()
        
        with pytest.raises(ValueError):
            exec_func.get_top_priority_goals(n=0)
        
        with pytest.raises(ValueError):
            exec_func.get_top_priority_goals(n=-1)
    
    def test_persistence_dir_is_file(self):
        """Test initialization fails if persistence_dir is a file"""
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(ValueError, match="must be a directory"):
                ExecutiveFunction(persistence_dir=Path(f.name))
    
    def test_goal_sequence_empty(self):
        """Test action sequence with no actions"""
        exec_func = ExecutiveFunction()
        goal = exec_func.create_goal("Test", priority=0.5)
        
        sequence = exec_func.get_action_sequence(goal.id)
        assert sequence == []


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
