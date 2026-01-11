"""
Tests for unified MetaCognitiveSystem.

Tests cover:
- System initialization
- Integration of monitoring, action learning, and attention history
- Self-assessment generation
- Introspection queries
"""

import pytest
from datetime import datetime

from lyra.cognitive_core.meta_cognition import (
    MetaCognitiveSystem,
    SelfAssessment
)
from lyra.cognitive_core.goals.resources import CognitiveResources


class TestMetaCognitiveSystem:
    """Test unified MetaCognitiveSystem."""
    
    def test_system_initialization(self):
        """Test system initialization."""
        system = MetaCognitiveSystem()
        
        assert system.monitor is not None
        assert system.action_learner is not None
        assert system.attention_history is not None
    
    def test_system_with_custom_params(self):
        """Test initialization with custom parameters."""
        system = MetaCognitiveSystem(
            min_observations_for_patterns=5,
            min_outcomes_for_model=10
        )
        
        assert system.monitor.pattern_detector.min_observations == 5
        assert system.action_learner.min_outcomes == 10
    
    def test_monitor_integration(self):
        """Test monitoring through unified system."""
        system = MetaCognitiveSystem()
        
        with system.monitor.observe("test_process") as ctx:
            ctx.input_complexity = 0.7
            ctx.output_quality = 0.9
            ctx.resources = CognitiveResources(attention_budget=0.5)
        
        assert len(system.monitor.observations) == 1
    
    def test_action_outcome_recording(self):
        """Test recording action outcomes."""
        system = MetaCognitiveSystem()
        
        system.record_action_outcome(
            action_id="act_1",
            action_type="speak",
            intended="provide helpful response",
            actual="provided helpful response",
            context={"ready": True}
        )
        
        assert len(system.action_learner.outcomes) == 1
    
    def test_attention_recording(self):
        """Test recording attention allocation."""
        system = MetaCognitiveSystem()
        
        alloc_id = system.record_attention(
            allocation={"goal_1": 0.6, "goal_2": 0.4},
            trigger="goal_priority",
            workspace_state="test_state"
        )
        
        assert alloc_id is not None
        assert len(system.attention_history.allocations) == 1
    
    def test_attention_outcome_recording(self):
        """Test recording attention outcomes."""
        system = MetaCognitiveSystem()
        
        alloc_id = system.record_attention(
            allocation={"goal_1": 0.6, "goal_2": 0.4},
            trigger="test",
            workspace_state="test"
        )
        
        system.record_attention_outcome(
            allocation_id=alloc_id,
            goal_progress={"goal_1": 0.3, "goal_2": 0.1},
            discoveries=["insight"],
            missed=[]
        )
        
        assert alloc_id in system.attention_history.outcomes
    
    def test_get_action_reliability(self):
        """Test getting action reliability."""
        system = MetaCognitiveSystem()
        
        # Record some outcomes
        for i in range(5):
            system.record_action_outcome(
                action_id=f"act_{i}",
                action_type="test_action",
                intended="succeed",
                actual="succeed" if i < 4 else "fail",
                context={}
            )
        
        reliability = system.get_action_reliability("test_action")
        
        assert reliability.action_type == "test_action"
        assert not reliability.unknown
        assert 0 <= reliability.success_rate <= 1
    
    def test_predict_action_outcome(self):
        """Test predicting action outcomes."""
        system = MetaCognitiveSystem(min_outcomes_for_model=3)
        
        # Build a model
        for i in range(10):
            success = i % 2 == 0
            system.record_action_outcome(
                action_id=f"act_{i}",
                action_type="test_action",
                intended="succeed",
                actual="succeed" if success else "fail",
                context={"favorable": success}
            )
        
        # Make prediction
        prediction = system.predict_action_outcome(
            action_type="test_action",
            context={"favorable": True}
        )
        
        assert 0 <= prediction.probability_success <= 1
    
    def test_get_recommended_attention(self):
        """Test getting recommended attention allocation."""
        system = MetaCognitiveSystem()
        
        # Mock goals
        class MockGoal:
            def __init__(self, id, priority):
                self.id = id
                self.priority = priority
        
        goals = [
            MockGoal("goal_1", 0.7),
            MockGoal("goal_2", 0.3)
        ]
        
        recommendation = system.get_recommended_attention(
            context="test",
            goals=goals
        )
        
        assert isinstance(recommendation, dict)


class TestSelfAssessment:
    """Test self-assessment generation."""
    
    def test_self_assessment_empty(self):
        """Test self-assessment with no data."""
        system = MetaCognitiveSystem()
        
        assessment = system.get_self_assessment()
        
        assert isinstance(assessment, SelfAssessment)
        assert isinstance(assessment.processing_patterns, list)
        assert isinstance(assessment.action_reliability, dict)
        assert isinstance(assessment.identified_strengths, list)
        assert isinstance(assessment.identified_weaknesses, list)
        assert isinstance(assessment.suggested_adaptations, list)
    
    def test_self_assessment_with_data(self):
        """Test self-assessment with recorded data."""
        system = MetaCognitiveSystem(
            min_observations_for_patterns=3,
            min_outcomes_for_model=3
        )
        
        # Add processing observations
        for i in range(10):
            with system.monitor.observe("reasoning") as ctx:
                ctx.input_complexity = 0.5
                ctx.output_quality = 0.8
        
        # Add action outcomes
        for i in range(10):
            system.record_action_outcome(
                action_id=f"act_{i}",
                action_type="speak",
                intended="succeed",
                actual="succeed" if i < 8 else "fail",
                context={}
            )
        
        # Add attention allocations
        for i in range(10):
            alloc_id = system.record_attention(
                allocation={"goal_1": 0.7, "goal_2": 0.3},
                trigger="test",
                workspace_state=f"state_{i}"
            )
            system.record_attention_outcome(
                allocation_id=alloc_id,
                goal_progress={"goal_1": 0.4},
                discoveries=[],
                missed=[],
                efficiency=0.8
            )
        
        assessment = system.get_self_assessment()
        
        # Should have some content
        assert len(assessment.action_reliability) > 0
    
    def test_identify_strengths(self):
        """Test identification of strengths."""
        system = MetaCognitiveSystem(min_outcomes_for_model=3)
        
        # Create strong performance pattern
        for i in range(10):
            system.record_action_outcome(
                action_id=f"act_{i}",
                action_type="reliable_action",
                intended="succeed",
                actual="succeed",
                context={}
            )
        
        assessment = system.get_self_assessment()
        
        # Should identify strength
        assert len(assessment.identified_strengths) > 0
    
    def test_identify_weaknesses(self):
        """Test identification of weaknesses."""
        system = MetaCognitiveSystem(
            min_observations_for_patterns=3,
            min_outcomes_for_model=3
        )
        
        # Create failure pattern in processing
        for i in range(10):
            try:
                with system.monitor.observe("difficult_process") as ctx:
                    ctx.input_complexity = 0.9
                    ctx.output_quality = 0.2
                    if i % 2 == 0:
                        raise ValueError("Test failure")
            except ValueError:
                pass  # Expected for test
        
        # Create unreliable action pattern
        for i in range(10):
            system.record_action_outcome(
                action_id=f"act_{i}",
                action_type="unreliable_action",
                intended="succeed",
                actual="fail",
                context={}
            )
        
        assessment = system.get_self_assessment()
        
        # Should identify weaknesses
        assert len(assessment.identified_weaknesses) > 0
    
    def test_suggest_adaptations(self):
        """Test adaptation suggestions."""
        system = MetaCognitiveSystem(min_observations_for_patterns=3)
        
        # Create pattern that suggests adaptation
        for i in range(10):
            try:
                with system.monitor.observe("test_process") as ctx:
                    ctx.input_complexity = 0.9 if i < 5 else 0.3
                    ctx.output_quality = 0.2 if i < 5 else 0.9
                    if i < 5:
                        raise ValueError("Failed on complexity")
            except ValueError:
                pass  # Expected for test
        
        assessment = system.get_self_assessment()
        
        # Should suggest adaptations
        assert len(assessment.suggested_adaptations) > 0


class TestIntrospection:
    """Test introspection query handling."""
    
    def test_introspect_failures(self):
        """Test introspection about failures."""
        system = MetaCognitiveSystem(min_observations_for_patterns=3)
        
        # Create failure patterns
        for i in range(10):
            try:
                with system.monitor.observe("reasoning") as ctx:
                    ctx.input_complexity = 0.9
                    ctx.output_quality = 0.1
                    if i % 2 == 0:
                        raise ValueError("Test failure")
            except ValueError:
                pass  # Expected for test
        
        response = system.introspect("What do I tend to fail at?")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_introspect_successes(self):
        """Test introspection about successes."""
        system = MetaCognitiveSystem(min_observations_for_patterns=3)
        
        # Create success patterns
        for i in range(10):
            with system.monitor.observe("reasoning") as ctx:
                ctx.input_complexity = 0.5
                ctx.output_quality = 0.9
        
        response = system.introspect("What am I good at?")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_introspect_attention(self):
        """Test introspection about attention."""
        system = MetaCognitiveSystem()
        
        # Record some attention allocations
        for i in range(5):
            alloc_id = system.record_attention(
                allocation={"goal_1": 0.7, "goal_2": 0.3},
                trigger="test",
                workspace_state=f"state_{i}"
            )
            system.record_attention_outcome(
                allocation_id=alloc_id,
                goal_progress={"goal_1": 0.3},
                discoveries=[],
                missed=[],
                efficiency=0.7
            )
        
        response = system.introspect("How effective is my attention?")
        
        assert isinstance(response, str)
        assert "attention" in response.lower() or "efficiency" in response.lower()
    
    def test_introspect_actions(self):
        """Test introspection about actions."""
        system = MetaCognitiveSystem()
        
        # Record action outcomes
        for i in range(5):
            system.record_action_outcome(
                action_id=f"act_{i}",
                action_type="test_action",
                intended="succeed",
                actual="succeed" if i < 4 else "fail",
                context={}
            )
        
        response = system.introspect("How reliable are my actions?")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_introspect_general(self):
        """Test general introspection."""
        system = MetaCognitiveSystem()
        
        response = system.introspect("Tell me about myself")
        
        assert isinstance(response, str)
        assert len(response) > 0


class TestSystemSummary:
    """Test system summary generation."""
    
    def test_get_summary(self):
        """Test getting comprehensive summary."""
        system = MetaCognitiveSystem()
        
        # Add some data to each subsystem
        with system.monitor.observe("test") as ctx:
            ctx.input_complexity = 0.5
            ctx.output_quality = 0.8
        
        system.record_action_outcome(
            action_id="act_1",
            action_type="test",
            intended="succeed",
            actual="succeed",
            context={}
        )
        
        alloc_id = system.record_attention(
            allocation={"goal_1": 1.0},
            trigger="test",
            workspace_state="test"
        )
        system.record_attention_outcome(
            allocation_id=alloc_id,
            goal_progress={"goal_1": 0.5},
            discoveries=[],
            missed=[],
            efficiency=0.7
        )
        
        summary = system.get_summary()
        
        assert "monitoring" in summary
        assert "action_learning" in summary
        assert "attention_history" in summary
        
        # Each subsystem should have its own summary
        assert isinstance(summary["monitoring"], dict)
        assert isinstance(summary["action_learning"], dict)
        assert isinstance(summary["attention_history"], dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
