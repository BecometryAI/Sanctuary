"""
Integration tests for ConsciousnessCore + ExecutiveFunction

Tests verify that:
- ConsciousnessCore correctly initializes ExecutiveFunction
- Helper methods properly integrate conversation context
- Goal/action/decision tracking works end-to-end
"""

import pytest
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mind.consciousness import ConsciousnessCore
from mind.executive_function import GoalStatus, ActionStatus


class TestConsciousnessExecutiveIntegration:
    """Test integration between ConsciousnessCore and ExecutiveFunction"""
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing"""
        memory_dir = tempfile.mkdtemp()
        executive_dir = tempfile.mkdtemp()
        yield Path(memory_dir), Path(executive_dir)
        # Cleanup handled by consciousness fixture

    @pytest.fixture
    def consciousness(self, temp_dirs):
        """Create ConsciousnessCore instance"""
        memory_dir, executive_dir = temp_dirs
        core = ConsciousnessCore(
            memory_persistence_dir=str(memory_dir),
            executive_persistence_dir=str(executive_dir)
        )
        yield core
        # Close to release ChromaDB file locks before cleanup
        core.close()
        # Now cleanup directories
        shutil.rmtree(memory_dir, ignore_errors=True)
        shutil.rmtree(executive_dir, ignore_errors=True)
    
    # ========================================================================
    # Initialization Tests
    # ========================================================================
    
    def test_executive_function_initialization(self, consciousness):
        """Test that ExecutiveFunction is properly initialized"""
        assert hasattr(consciousness, 'executive')
        assert consciousness.executive is not None
    
    # ========================================================================
    # Goal Management Integration Tests
    # ========================================================================
    
    def test_set_goal_basic(self, consciousness):
        """Test setting a goal through consciousness"""
        goal = consciousness.set_goal(
            description="Learn about user preferences",
            priority=0.8
        )
        
        assert goal.description == "Learn about user preferences"
        assert goal.priority == 0.8
        assert goal.id in consciousness.executive.goals
    
    def test_set_goal_with_context_enrichment(self, consciousness):
        """Test that goals are enriched with conversation context"""
        # Simulate conversation state
        consciousness.context.current_topic = "machine learning"
        consciousness.context.interaction_count = 5
        
        goal = consciousness.set_goal(
            description="Understand ML concepts",
            priority=0.9
        )
        
        # Goal should have context in metadata
        assert "context" in goal.context
        assert goal.context["context"]["topic"] == "machine learning"
        assert goal.context["context"]["interaction_count"] == 5
    
    def test_get_active_goals(self, consciousness):
        """Test retrieving active goals"""
        # Create goals
        g1 = consciousness.set_goal("Goal 1", priority=0.9)
        g2 = consciousness.set_goal("Goal 2", priority=0.7)
        
        # Activate them
        consciousness.executive.update_goal_status(g1.id, GoalStatus.ACTIVE)
        consciousness.executive.update_goal_status(g2.id, GoalStatus.ACTIVE)
        
        # Get active goals
        active = consciousness.get_active_goals(n=5)
        
        assert len(active) == 2
        assert all(isinstance(g, dict) for g in active)
        assert active[0]["priority"] >= active[1]["priority"]  # Sorted
    
    # ========================================================================
    # Action Planning Integration Tests
    # ========================================================================
    
    def test_plan_actions_for_goal(self, consciousness):
        """Test planning actions for a goal"""
        goal = consciousness.set_goal("Complete project", priority=0.8)
        
        # Plan sequential actions
        actions = consciousness.plan_actions_for_goal(
            goal_id=goal.id,
            action_descriptions=[
                "Gather requirements",
                "Design architecture",
                "Implement features"
            ],
            sequential=True
        )
        
        assert len(actions) == 3
        
        # Verify dependencies (sequential)
        assert actions[0].dependencies == []
        assert actions[1].dependencies == [actions[0].id]
        assert actions[2].dependencies == [actions[1].id]
    
    def test_plan_actions_parallel(self, consciousness):
        """Test planning parallel actions"""
        goal = consciousness.set_goal("Research topics", priority=0.7)
        
        actions = consciousness.plan_actions_for_goal(
            goal_id=goal.id,
            action_descriptions=[
                "Read paper A",
                "Read paper B",
                "Read paper C"
            ],
            sequential=False
        )
        
        # All actions should have no dependencies
        assert all(len(a.dependencies) == 0 for a in actions)
    
    def test_get_next_actions(self, consciousness):
        """Test getting next ready actions"""
        goal = consciousness.set_goal("Task sequence", priority=0.8)
        
        # Create sequential actions
        actions = consciousness.plan_actions_for_goal(
            goal_id=goal.id,
            action_descriptions=["Step 1", "Step 2", "Step 3"],
            sequential=True
        )
        
        # Initially, only first action is ready
        next_actions = consciousness.get_next_actions(goal_id=goal.id)
        assert len(next_actions) == 1
        assert next_actions[0]["id"] == actions[0].id
        
        # Complete first action
        consciousness.executive.actions[actions[0].id].status = ActionStatus.COMPLETED
        
        # Now second action should be ready
        next_actions = consciousness.get_next_actions(goal_id=goal.id)
        assert len(next_actions) == 1
        assert next_actions[0]["id"] == actions[1].id
    
    # ========================================================================
    # Decision Making Integration Tests
    # ========================================================================
    
    def test_make_decision_basic(self, consciousness):
        """Test making a decision through consciousness"""
        question = "Should we proceed with implementation?"
        options = ["yes", "no", "wait for more info"]
        
        decision, selected, confidence, rationale = consciousness.make_decision(
            question=question,
            options=options
        )
        
        assert decision.question == question
        assert selected in options
        assert 0 <= confidence <= 1
        assert isinstance(rationale, str)
    
    def test_make_decision_with_context(self, consciousness):
        """Test decision includes conversation context"""
        # Set conversation context
        consciousness.context.current_topic = "API design"
        consciousness.context.interaction_count = 10
        
        decision, _, _, _ = consciousness.make_decision(
            question="Choose API style",
            options=["REST", "GraphQL", "gRPC"]
        )
        
        # Decision should have context
        assert "context" in decision.context
        assert decision.context["context"]["topic"] == "API design"
    
    # ========================================================================
    # Executive Summary Tests
    # ========================================================================
    
    def test_get_executive_summary(self, consciousness):
        """Test getting comprehensive executive summary"""
        # Create goals and actions
        g1 = consciousness.set_goal("Goal 1", priority=0.9)
        consciousness.executive.update_goal_status(g1.id, GoalStatus.ACTIVE)
        
        consciousness.plan_actions_for_goal(
            goal_id=g1.id,
            action_descriptions=["Action 1", "Action 2"],
            sequential=True
        )
        
        # Get summary
        summary = consciousness.get_executive_summary(top_n_goals=5)
        
        # Verify summary structure
        assert "statistics" in summary
        assert "top_active_goals" in summary
        assert "next_actions" in summary
        
        # Verify statistics
        assert summary["statistics"]["total_goals"] >= 1
        assert summary["statistics"]["total_actions"] >= 2
        
        # Verify active goals
        assert len(summary["top_active_goals"]) >= 1
        
        # Verify next actions
        assert len(summary["next_actions"]) >= 1
    
    # ========================================================================
    # End-to-End Workflow Tests
    # ========================================================================
    
    def test_complete_goal_workflow(self, consciousness):
        """Test complete workflow: goal -> actions -> decisions -> completion"""
        # 1. Set a goal
        goal = consciousness.set_goal(
            description="Implement new feature",
            priority=0.85
        )
        consciousness.executive.update_goal_status(goal.id, GoalStatus.ACTIVE)
        
        # 2. Plan actions
        actions = consciousness.plan_actions_for_goal(
            goal_id=goal.id,
            action_descriptions=[
                "Design feature",
                "Implement code",
                "Write tests",
                "Deploy"
            ],
            sequential=True
        )
        
        # 3. Make decision during planning
        decision, choice, _, _ = consciousness.make_decision(
            question="Which architecture pattern?",
            options=["MVC", "MVVM", "Clean Architecture"]
        )
        
        # 4. Simulate executing actions
        for i, action in enumerate(actions):
            # Get next ready actions
            next_actions = consciousness.get_next_actions(goal_id=goal.id)
            assert len(next_actions) >= 1
            
            # Complete current action
            consciousness.executive.actions[action.id].status = ActionStatus.COMPLETED
        
        # 5. Complete goal
        consciousness.executive.update_goal_status(goal.id, GoalStatus.COMPLETED)
        consciousness.executive.goals[goal.id].progress = 1.0
        
        # 6. Verify final state
        summary = consciousness.get_executive_summary()
        assert summary["statistics"]["total_goals"] >= 1
        assert summary["statistics"]["total_actions"] == 4
        assert summary["statistics"]["total_decisions"] >= 1
    
    def test_hierarchical_goal_planning(self, consciousness):
        """Test planning with parent-child goals"""
        # Create parent goal
        parent = consciousness.set_goal(
            description="Master Python",
            priority=0.9
        )
        
        # Create child goals
        child1 = consciousness.executive.create_goal(
            description="Learn basics",
            priority=0.8,
            parent_goal_id=parent.id
        )
        
        child2 = consciousness.executive.create_goal(
            description="Learn advanced topics",
            priority=0.7,
            parent_goal_id=parent.id
        )
        
        # Verify hierarchy
        assert child1.parent_goal_id == parent.id
        assert child2.parent_goal_id == parent.id
        assert child1.id in parent.subgoal_ids
        assert child2.id in parent.subgoal_ids
        
        # Plan actions for child goals
        actions1 = consciousness.plan_actions_for_goal(
            goal_id=child1.id,
            action_descriptions=["Learn syntax", "Practice basics"],
            sequential=True
        )
        
        actions2 = consciousness.plan_actions_for_goal(
            goal_id=child2.id,
            action_descriptions=["Study decorators", "Master metaclasses"],
            sequential=True
        )
        
        # Verify actions are associated correctly
        assert all(a.goal_id == child1.id for a in actions1)
        assert all(a.goal_id == child2.id for a in actions2)


class TestPersistence:
    """Test persistence of executive state through consciousness"""

    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories"""
        memory_dir = tempfile.mkdtemp()
        executive_dir = tempfile.mkdtemp()
        yield Path(memory_dir), Path(executive_dir)
        # Use ignore_errors=True for Windows where ChromaDB may hold locks
        shutil.rmtree(memory_dir, ignore_errors=True)
        shutil.rmtree(executive_dir, ignore_errors=True)
    
    def test_executive_state_persistence(self, temp_dirs):
        """Test executive state persists across consciousness instances"""
        memory_dir, executive_dir = temp_dirs

        # Create first instance and add data
        consciousness1 = ConsciousnessCore(
            memory_persistence_dir=str(memory_dir),
            executive_persistence_dir=str(executive_dir)
        )

        goal = consciousness1.set_goal("Persistent goal", priority=0.8)
        consciousness1.plan_actions_for_goal(
            goal_id=goal.id,
            action_descriptions=["Action 1", "Action 2"]
        )
        consciousness1.executive.save_state()

        # Close first instance to release ChromaDB locks before creating second
        goal_id = goal.id  # Save ID before closing
        consciousness1.close()

        # Create second instance (should load state)
        consciousness2 = ConsciousnessCore(
            memory_persistence_dir=str(memory_dir),
            executive_persistence_dir=str(executive_dir)
        )

        try:
            # Verify data loaded
            assert goal_id in consciousness2.executive.goals
            loaded_goal = consciousness2.executive.goals[goal_id]
            assert loaded_goal.description == "Persistent goal"
            assert loaded_goal.priority == 0.8
        finally:
            consciousness2.close()
            shutil.rmtree(memory_dir, ignore_errors=True)
            shutil.rmtree(executive_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
