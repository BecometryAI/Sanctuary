"""
Tests for meta-cognitive monitoring, action learning, and attention history.

Tests verify:
- Processing observation and pattern detection
- Action outcome tracking and reliability
- Attention allocation history and learning
- Unified meta-cognitive system integration
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from lyra.cognitive_core.meta_cognition import (
    MetaCognitiveMonitor,
    ProcessingObservation,
    ProcessingContext,
    CognitivePattern,
    ProcessStats,
    CognitiveResources,
    ActionOutcomeLearner,
    ActionOutcome,
    ActionReliability,
    AttentionHistory,
    AttentionAllocation,
    AttentionOutcome,
    MetaCognitiveSystem,
    SelfAssessment,
)


class TestMetaCognitiveMonitor:
    """Test MetaCognitiveMonitor functionality."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = MetaCognitiveMonitor()
        assert monitor is not None
        assert len(monitor.observations) == 0
        assert monitor.pattern_detector is not None
    
    def test_observe_context_manager(self):
        """Test ProcessingContext as context manager."""
        monitor = MetaCognitiveMonitor()
        
        with monitor.observe("reasoning") as ctx:
            ctx.set_complexity(0.7)
            ctx.set_quality(0.8)
            # Simulate some work
            pass
        
        # Check observation was recorded
        assert len(monitor.observations) == 1
        obs = monitor.observations[0]
        assert obs.process_type == "reasoning"
        assert obs.success is True
        assert obs.input_complexity == 0.7
        assert obs.output_quality == 0.8
    
    def test_observe_with_error(self):
        """Test observation recording on error."""
        monitor = MetaCognitiveMonitor()
        
        try:
            with monitor.observe("memory_retrieval") as ctx:
                ctx.set_complexity(0.5)
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Check error was recorded
        assert len(monitor.observations) == 1
        obs = monitor.observations[0]
        assert obs.success is False
        assert obs.error == "Test error"
    
    def test_process_statistics(self):
        """Test statistics computation."""
        monitor = MetaCognitiveMonitor()
        
        # Record several observations
        for i in range(5):
            with monitor.observe("goal_selection") as ctx:
                ctx.set_complexity(0.5)
                ctx.set_quality(0.7)
        
        stats = monitor.get_process_statistics("goal_selection")
        assert stats.total_executions == 5
        assert stats.success_rate == 1.0
        assert stats.avg_quality == 0.7
    
    def test_pattern_detection_failure_mode(self):
        """Test failure mode pattern detection."""
        monitor = MetaCognitiveMonitor()
        
        # Record successes with low complexity
        for i in range(5):
            with monitor.observe("reasoning") as ctx:
                ctx.set_complexity(0.3)
                ctx.set_quality(0.8)
        
        # Record failures with high complexity
        for i in range(5):
            try:
                with monitor.observe("reasoning") as ctx:
                    ctx.set_complexity(0.9)
                    raise Exception("Complex input failed")
            except Exception:
                pass
        
        patterns = monitor.get_identified_patterns()
        failure_patterns = [p for p in patterns if p.pattern_type == 'failure_mode']
        
        assert len(failure_patterns) > 0
        assert any("high-complexity" in p.description for p in failure_patterns)
    
    def test_pattern_detection_success_condition(self):
        """Test success condition pattern detection."""
        monitor = MetaCognitiveMonitor()
        
        # Record many successes with simple inputs
        for i in range(10):
            with monitor.observe("memory_retrieval") as ctx:
                ctx.set_complexity(0.2)
                ctx.set_quality(0.9)
        
        patterns = monitor.get_identified_patterns()
        success_patterns = [p for p in patterns if p.pattern_type == 'success_condition']
        
        assert len(success_patterns) > 0
    
    def test_summary_generation(self):
        """Test summary generation."""
        monitor = MetaCognitiveMonitor()
        
        # Add some observations
        for i in range(3):
            with monitor.observe("introspection") as ctx:
                ctx.set_complexity(0.5)
                ctx.set_quality(0.6)
        
        summary = monitor.get_summary()
        assert summary["total_observations"] == 3
        assert "introspection" in summary["process_types"]


class TestActionOutcomeLearner:
    """Test ActionOutcomeLearner functionality."""
    
    def test_initialization(self):
        """Test learner initialization."""
        learner = ActionOutcomeLearner()
        assert learner is not None
        assert len(learner.outcomes) == 0
    
    def test_record_outcome_success(self):
        """Test recording successful outcome."""
        learner = ActionOutcomeLearner()
        
        learner.record_outcome(
            action_id="action1",
            action_type="speak",
            intended="say hello",
            actual="said hello",
            context={}
        )
        
        assert len(learner.outcomes) == 1
        outcome = learner.outcomes[0]
        assert outcome.action_type == "speak"
        assert outcome.success is True
    
    def test_record_outcome_failure(self):
        """Test recording failed outcome."""
        learner = ActionOutcomeLearner()
        
        learner.record_outcome(
            action_id="action2",
            action_type="retrieve_memory",
            intended="find specific memory",
            actual="found nothing",
            context={}
        )
        
        assert len(learner.outcomes) == 1
        outcome = learner.outcomes[0]
        assert outcome.success is False
    
    def test_action_reliability(self):
        """Test action reliability computation."""
        learner = ActionOutcomeLearner()
        
        # Record multiple outcomes
        for i in range(10):
            success = i < 8  # 80% success rate
            learner.record_outcome(
                action_id=f"action{i}",
                action_type="commit_memory",
                intended="store memory",
                actual="memory stored" if success else "storage failed",
                context={}
            )
        
        reliability = learner.get_action_reliability("commit_memory")
        assert reliability.success_rate == 0.8
        assert reliability.total_executions == 10
        assert reliability.unknown is False
    
    def test_action_reliability_unknown(self):
        """Test reliability for unknown action."""
        learner = ActionOutcomeLearner()
        
        reliability = learner.get_action_reliability("unknown_action")
        assert reliability.unknown is True
        assert reliability.success_rate == 0.0
    
    def test_outcome_prediction(self):
        """Test outcome prediction."""
        learner = ActionOutcomeLearner()
        
        # Record enough outcomes to build a model
        for i in range(10):
            learner.record_outcome(
                action_id=f"action{i}",
                action_type="test_action",
                intended="test",
                actual="test completed",
                context={"feature1": True}
            )
        
        # Predict with similar context
        prediction = learner.predict_outcome("test_action", {"feature1": True})
        assert prediction.confidence > 0
    
    def test_side_effect_tracking(self):
        """Test side effect identification."""
        learner = ActionOutcomeLearner()
        
        learner.record_outcome(
            action_id="action1",
            action_type="speak",
            intended="say hello",
            actual="said hello with enthusiasm",
            context={"emotional_change": True}
        )
        
        outcome = learner.outcomes[0]
        assert len(outcome.side_effects) > 0
    
    def test_summary_generation(self):
        """Test summary generation."""
        learner = ActionOutcomeLearner()
        
        # Add some outcomes
        for i in range(5):
            learner.record_outcome(
                action_id=f"action{i}",
                action_type="test",
                intended="test",
                actual="test done",
                context={}
            )
        
        summary = learner.get_summary()
        assert summary["total_outcomes"] == 5
        assert summary["action_types"] >= 1


class TestAttentionHistory:
    """Test AttentionHistory functionality."""
    
    def test_initialization(self):
        """Test history initialization."""
        history = AttentionHistory()
        assert history is not None
        assert len(history.allocations) == 0
    
    def test_record_allocation(self):
        """Test allocation recording."""
        history = AttentionHistory()
        
        allocation = {"goal1": 0.6, "goal2": 0.4}
        workspace_state = Mock()
        
        allocation_id = history.record_allocation(
            allocation=allocation,
            trigger="new_percept",
            workspace_state=workspace_state
        )
        
        assert len(history.allocations) == 1
        assert history.allocations[0].id == allocation_id
        assert history.allocations[0].trigger == "new_percept"
    
    def test_record_outcome(self):
        """Test outcome recording."""
        history = AttentionHistory()
        
        # Record allocation
        allocation_id = history.record_allocation(
            allocation={"goal1": 1.0},
            trigger="test",
            workspace_state=Mock()
        )
        
        # Record outcome
        history.record_outcome(
            allocation_id=allocation_id,
            goal_progress={"goal1": 0.5},
            discoveries=["insight1"],
            missed=[]
        )
        
        assert allocation_id in history.outcomes
        outcome = history.outcomes[allocation_id]
        assert outcome.efficiency > 0
    
    def test_efficiency_computation(self):
        """Test efficiency calculation."""
        history = AttentionHistory()
        
        # High progress, discoveries, no misses = high efficiency
        efficiency = history._compute_efficiency(
            goal_progress={"goal1": 0.9},
            discoveries=["insight1", "insight2"],
            missed=[]
        )
        assert efficiency > 0.8
        
        # Low progress, misses = low efficiency
        efficiency = history._compute_efficiency(
            goal_progress={"goal1": 0.1},
            discoveries=[],
            missed=["missed1", "missed2"]
        )
        assert efficiency < 0.3
    
    def test_pattern_learning(self):
        """Test attention pattern learning."""
        history = AttentionHistory()
        
        # Record several allocations with outcomes
        for i in range(10):
            allocation_id = history.record_allocation(
                allocation={"goal1": 1.0},  # Focused allocation
                trigger="focused_test",
                workspace_state=Mock()
            )
            
            history.record_outcome(
                allocation_id=allocation_id,
                goal_progress={"goal1": 0.8},
                discoveries=["insight"],
                missed=[]
            )
        
        patterns = history.get_attention_patterns()
        assert len(patterns) > 0
        
        # Check that focused pattern was learned
        assert any(p.pattern == "focused_single" for p in patterns)
    
    def test_recommended_allocation(self):
        """Test allocation recommendation."""
        history = AttentionHistory()
        
        # Train with successful focused allocations
        for i in range(10):
            allocation_id = history.record_allocation(
                allocation={"goal1": 1.0},
                trigger="test",
                workspace_state=Mock()
            )
            history.record_outcome(
                allocation_id=allocation_id,
                goal_progress={"goal1": 0.9},
                discoveries=[],
                missed=[]
            )
        
        # Get recommendation
        mock_goal = Mock()
        mock_goal.id = "goal1"
        mock_goal.priority = 0.8
        
        recommended = history.get_recommended_allocation(
            context=Mock(),
            goals=[mock_goal]
        )
        
        assert len(recommended) > 0
    
    def test_summary_generation(self):
        """Test summary generation."""
        history = AttentionHistory()
        
        # Add some data
        allocation_id = history.record_allocation(
            allocation={"goal1": 1.0},
            trigger="test",
            workspace_state=Mock()
        )
        history.record_outcome(
            allocation_id=allocation_id,
            goal_progress={"goal1": 0.7},
            discoveries=[],
            missed=[]
        )
        
        summary = history.get_summary()
        assert summary["total_allocations"] == 1
        assert summary["total_outcomes"] == 1


class TestMetaCognitiveSystem:
    """Test unified MetaCognitiveSystem."""
    
    def test_initialization(self):
        """Test system initialization."""
        system = MetaCognitiveSystem()
        assert system is not None
        assert system.monitor is not None
        assert system.action_learner is not None
        assert system.attention_history is not None
    
    def test_self_assessment(self):
        """Test self-assessment generation."""
        system = MetaCognitiveSystem()
        
        # Add some data to each subsystem
        with system.monitor.observe("test_process") as ctx:
            ctx.set_complexity(0.5)
            ctx.set_quality(0.7)
        
        system.action_learner.record_outcome(
            action_id="action1",
            action_type="test",
            intended="test",
            actual="test done",
            context={}
        )
        
        allocation_id = system.attention_history.record_allocation(
            allocation={"goal1": 1.0},
            trigger="test",
            workspace_state=Mock()
        )
        system.attention_history.record_outcome(
            allocation_id=allocation_id,
            goal_progress={"goal1": 0.8},
            discoveries=[],
            missed=[]
        )
        
        # Generate assessment
        assessment = system.get_self_assessment()
        assert isinstance(assessment, SelfAssessment)
        assert isinstance(assessment.processing_patterns, list)
        assert isinstance(assessment.action_reliability, dict)
        assert isinstance(assessment.identified_strengths, list)
        assert isinstance(assessment.identified_weaknesses, list)
    
    def test_introspection_failures(self):
        """Test introspection about failures."""
        system = MetaCognitiveSystem()
        
        # Add failure data
        for i in range(5):
            try:
                with system.monitor.observe("test") as ctx:
                    ctx.set_complexity(0.9)
                    raise Exception("Test failure")
            except Exception:
                pass
        
        response = system.introspect("What do I fail at?")
        assert "failure" in response.lower() or "fail" in response.lower()
    
    def test_introspection_attention(self):
        """Test introspection about attention."""
        system = MetaCognitiveSystem()
        
        # Add attention data
        for i in range(10):
            allocation_id = system.attention_history.record_allocation(
                allocation={"goal1": 1.0},
                trigger="test",
                workspace_state=Mock()
            )
            system.attention_history.record_outcome(
                allocation_id=allocation_id,
                goal_progress={"goal1": 0.7},
                discoveries=[],
                missed=[]
            )
        
        response = system.introspect("How effective is my attention?")
        assert "attention" in response.lower()
    
    def test_introspection_actions(self):
        """Test introspection about actions."""
        system = MetaCognitiveSystem()
        
        # Add action data
        for i in range(10):
            system.action_learner.record_outcome(
                action_id=f"action{i}",
                action_type="test_action",
                intended="test",
                actual="test done" if i < 8 else "failed",
                context={}
            )
        
        response = system.introspect("How reliable are my actions?")
        assert "action" in response.lower() or "reliable" in response.lower()
    
    def test_introspection_general(self):
        """Test general introspection."""
        system = MetaCognitiveSystem()
        
        response = system.introspect("Tell me about myself")
        assert len(response) > 0
        assert "meta-cognitive" in response.lower() or "cognitive" in response.lower()
    
    def test_monitoring_summary(self):
        """Test comprehensive monitoring summary."""
        system = MetaCognitiveSystem()
        
        # Add data to all subsystems
        with system.monitor.observe("test") as ctx:
            ctx.set_complexity(0.5)
            ctx.set_quality(0.7)
        
        system.action_learner.record_outcome(
            action_id="action1",
            action_type="test",
            intended="test",
            actual="test done",
            context={}
        )
        
        allocation_id = system.attention_history.record_allocation(
            allocation={"goal1": 1.0},
            trigger="test",
            workspace_state=Mock()
        )
        
        summary = system.get_monitoring_summary()
        assert "processing_monitor" in summary
        assert "action_learner" in summary
        assert "attention_history" in summary


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_processing_context_invalid_complexity(self):
        """Test invalid complexity values."""
        monitor = MetaCognitiveMonitor()
        
        with pytest.raises(TypeError):
            with monitor.observe("test") as ctx:
                ctx.set_complexity("invalid")
        
        # Test boundary clamping
        with monitor.observe("test") as ctx:
            ctx.set_complexity(-1.0)
            assert ctx.input_complexity == 0.0
            
        with monitor.observe("test") as ctx:
            ctx.set_complexity(2.0)
            assert ctx.input_complexity == 1.0
    
    def test_processing_context_invalid_quality(self):
        """Test invalid quality values."""
        monitor = MetaCognitiveMonitor()
        
        with pytest.raises(TypeError):
            with monitor.observe("test") as ctx:
                ctx.set_quality("invalid")
    
    def test_action_learner_invalid_inputs(self):
        """Test action learner with invalid inputs."""
        learner = ActionOutcomeLearner()
        
        # Empty action_id
        with pytest.raises(ValueError):
            learner.record_outcome("", "type", "intended", "actual", {})
        
        # Non-string action_type
        with pytest.raises(ValueError):
            learner.record_outcome("id", "", "intended", "actual", {})
        
        # Non-string intended
        with pytest.raises(TypeError):
            learner.record_outcome("id", "type", 123, "actual", {})
        
        # Non-dict context
        with pytest.raises(TypeError):
            learner.record_outcome("id", "type", "intended", "actual", "not_dict")
    
    def test_attention_history_invalid_inputs(self):
        """Test attention history with invalid inputs."""
        history = AttentionHistory()
        
        # Non-dict allocation
        with pytest.raises(TypeError):
            history.record_allocation("not_dict", "trigger", Mock())
        
        # Empty trigger
        with pytest.raises(ValueError):
            history.record_allocation({}, "", Mock())
    
    def test_empty_observations(self):
        """Test behavior with no observations."""
        monitor = MetaCognitiveMonitor()
        
        patterns = monitor.get_identified_patterns()
        assert patterns == []
        
        stats = monitor.get_process_statistics("nonexistent")
        assert stats.total_executions == 0
    
    def test_empty_action_outcomes(self):
        """Test behavior with no action outcomes."""
        learner = ActionOutcomeLearner()
        
        reliability = learner.get_action_reliability("nonexistent")
        assert reliability.unknown is True
        
        prediction = learner.predict_outcome("nonexistent", {})
        assert prediction.confidence == 0.0
    
    def test_memory_limits(self):
        """Test that memory limits are respected."""
        config = {"max_observations": 10}
        monitor = MetaCognitiveMonitor(config=config)
        
        # Add more than max
        for i in range(20):
            with monitor.observe("test") as ctx:
                ctx.set_complexity(0.5)
        
        assert len(monitor.observations) == 10
    
    def test_division_by_zero_protection(self):
        """Test protection against division by zero."""
        learner = ActionOutcomeLearner()
        
        # Record with empty strings
        learner.record_outcome("id", "type", "", "", {})
        
        reliability = learner.get_action_reliability("type")
        assert reliability.success_rate == 1.0  # Empty matches empty
    
    def test_concurrent_operations(self):
        """Test thread-safety considerations."""
        system = MetaCognitiveSystem()
        
        # Multiple rapid operations
        for i in range(10):
            with system.monitor.observe("test") as ctx:
                ctx.set_complexity(0.5)
            
            system.action_learner.record_outcome(
                f"action_{i}", "test", "intended", "actual", {}
            )
        
        # Should not crash
        assessment = system.get_self_assessment()
        assert assessment is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
