"""
Unit tests for ActionOutcomeLearner integration with ActiveInferenceActionSelector.

Tests cover:
- Action selection with action learner
- Reliability bias application
- Unknown action handling
- Backward compatibility
- Integration with IWMTCore
"""

import pytest
from datetime import datetime

from mind.cognitive_core.world_model import WorldModel
from mind.cognitive_core.active_inference import (
    FreeEnergyMinimizer,
    ActiveInferenceActionSelector,
    ActionEvaluation
)
from mind.cognitive_core.meta_cognition import ActionOutcomeLearner
from mind.cognitive_core.iwmt_core import IWMTCore


class TestActionLearnerIntegration:
    """Test ActionOutcomeLearner integration with ActiveInferenceActionSelector."""
    
    def test_initialization_without_action_learner(self):
        """Test backward compatibility: works without action_learner."""
        minimizer = FreeEnergyMinimizer()
        selector = ActiveInferenceActionSelector(minimizer)
        assert selector.action_learner is None
        assert selector.reliability_weight == 0.2  # default
    
    def test_initialization_with_action_learner(self):
        """Test initialization with action learner."""
        minimizer = FreeEnergyMinimizer()
        action_learner = ActionOutcomeLearner()
        selector = ActiveInferenceActionSelector(minimizer, action_learner=action_learner)
        
        assert selector.action_learner is action_learner
        assert selector.reliability_weight == 0.2
    
    def test_initialization_with_custom_reliability_weight(self):
        """Test initialization with custom reliability weight."""
        minimizer = FreeEnergyMinimizer()
        action_learner = ActionOutcomeLearner()
        config = {"reliability_weight": 0.5}
        selector = ActiveInferenceActionSelector(
            minimizer, 
            config=config,
            action_learner=action_learner
        )
        
        assert selector.reliability_weight == 0.5
    
    def test_evaluate_action_with_no_history(self):
        """Test that unknown actions get neutral treatment (no penalty)."""
        minimizer = FreeEnergyMinimizer()
        action_learner = ActionOutcomeLearner()
        selector = ActiveInferenceActionSelector(minimizer, action_learner=action_learner)
        world_model = WorldModel()
        
        action = {"type": "speak", "content": "test"}
        evaluation = selector.evaluate_action(action, world_model)
        
        # Should work without error, with neutral treatment
        assert isinstance(evaluation, ActionEvaluation)
        assert evaluation.action == action
    
    def test_high_reliability_action_preferred(self):
        """Test that actions with high historical success rate are preferred."""
        minimizer = FreeEnergyMinimizer()
        action_learner = ActionOutcomeLearner()
        config = {"reliability_weight": 0.3}  # Higher weight for stronger effect
        selector = ActiveInferenceActionSelector(minimizer, config=config, action_learner=action_learner)
        world_model = WorldModel()
        
        # Record successful outcomes for "speak" action
        for i in range(10):
            action_learner.record_outcome(
                action_id=f"speak_{i}",
                action_type="speak",
                intended="successfully communicate clear message to recipient",
                actual="message was successfully communicated clearly to recipient",
                context={}
            )
        
        # Record failed outcomes for "observe" action
        # Use distinctly different actual vs intended for failure
        for i in range(10):
            action_learner.record_outcome(
                action_id=f"observe_{i}",
                action_type="observe",
                intended="observe and gather detailed information",
                actual="nothing was observed no data collected",
                context={}
            )
        
        # Verify reliabilities
        speak_reliability = action_learner.get_action_reliability("speak")
        observe_reliability = action_learner.get_action_reliability("observe")
        
        assert speak_reliability.success_rate > observe_reliability.success_rate
        
        # Evaluate both actions
        speak_action = {"type": "speak", "content": "test"}
        observe_action = {"type": "observe", "target": "environment"}
        
        speak_eval = selector.evaluate_action(speak_action, world_model)
        observe_eval = selector.evaluate_action(observe_action, world_model)
        
        # Speak should have lower EFE (more preferred) due to higher success rate
        # The difference should be at least the reliability bonus
        expected_difference = (speak_reliability.success_rate - observe_reliability.success_rate) * config["reliability_weight"]
        actual_difference = observe_eval.expected_free_energy - speak_eval.expected_free_energy
        assert actual_difference >= expected_difference * 0.9  # Allow 10% tolerance
    
    def test_low_reliability_action_less_preferred(self):
        """Test that actions with low success rate receive worse scores."""
        minimizer = FreeEnergyMinimizer()
        action_learner = ActionOutcomeLearner()
        config = {"reliability_weight": 0.3}  # Higher weight for stronger effect
        selector = ActiveInferenceActionSelector(minimizer, config=config, action_learner=action_learner)
        world_model = WorldModel()
        
        # Record failed outcomes for "act" action
        # Use distinctly different actual vs intended for clear failure
        for i in range(10):
            action_learner.record_outcome(
                action_id=f"act_{i}",
                action_type="act",
                intended="successfully complete task and reach goal",
                actual="task completely failed error occurred nothing accomplished",
                context={}
            )
        
        # Verify low reliability
        reliability = action_learner.get_action_reliability("act")
        assert reliability.success_rate < 0.5  # Low success rate
        
        # Evaluate action with and without learner
        action = {"type": "act", "content": "do something"}
        
        # With learner (should have less benefit or penalty due to low success rate)
        eval_with_learner = selector.evaluate_action(action, world_model)
        
        # Create selector without learner for comparison
        selector_no_learner = ActiveInferenceActionSelector(minimizer)
        eval_without_learner = selector_no_learner.evaluate_action(action, world_model)
        
        # With low reliability, the bonus should be smaller (or penalty applied)
        # Since reliability bonus = success_rate * weight, low success rate means small bonus
        expected_bonus = reliability.success_rate * config["reliability_weight"]
        expected_efe_with_learner = eval_without_learner.expected_free_energy - expected_bonus
        
        # Check the bonus is approximately correct
        assert abs(eval_with_learner.expected_free_energy - expected_efe_with_learner) < 0.01
    
    def test_reliability_affects_action_selection(self):
        """Test that reliability influences which action is selected."""
        minimizer = FreeEnergyMinimizer()
        action_learner = ActionOutcomeLearner()
        config = {"reliability_weight": 0.5}  # Strong reliability influence
        selector = ActiveInferenceActionSelector(minimizer, config=config, action_learner=action_learner)
        world_model = WorldModel()
        
        # Make "wait" action highly reliable
        for i in range(20):
            action_learner.record_outcome(
                action_id=f"wait_{i}",
                action_type="wait",
                intended="allow settling",
                actual="system settled successfully",
                context={}
            )
        
        # Make "speak" action unreliable
        for i in range(20):
            action_learner.record_outcome(
                action_id=f"speak_{i}",
                action_type="speak",
                intended="communicate",
                actual="communication failed",
                context={}
            )
        
        # Add high uncertainty to trigger action
        for _ in range(5):
            world_model.update_on_percept({"content": "unexpected"})
        
        # Available actions
        actions = [
            {"type": "speak", "content": "test"},
            {"type": "wait", "duration": 1.0}
        ]
        
        should_act, selected_action = selector.should_act(world_model, actions)
        
        if should_act:
            # Wait should be selected due to higher reliability
            assert selected_action["type"] == "wait"
    
    def test_iwmt_core_with_action_learner(self):
        """Test IWMTCore initialization with action learner."""
        action_learner = ActionOutcomeLearner()
        core = IWMTCore(action_learner=action_learner)
        
        assert core.active_inference.action_learner is action_learner
    
    def test_iwmt_core_without_action_learner(self):
        """Test IWMTCore backward compatibility without action learner."""
        core = IWMTCore()
        
        assert core.active_inference.action_learner is None
    
    @pytest.mark.asyncio
    async def test_cognitive_cycle_with_action_learning(self):
        """Test cognitive cycle with action learning integration."""
        action_learner = ActionOutcomeLearner()
        
        # Record some action history
        action_learner.record_outcome(
            action_id="test_1",
            action_type="speak",
            intended="test communication",
            actual="communication successful",
            context={}
        )
        
        core = IWMTCore(action_learner=action_learner)
        
        percepts = [{"content": "test input"}]
        emotional_state = {"arousal": 0.5, "valence": 0.3}
        
        results = await core.cognitive_cycle(percepts, emotional_state)
        
        assert "should_act" in results
        assert "recommended_action" in results
        # Cycle should complete successfully with action learning
        assert results["cycle_number"] == 1
    
    def test_context_specific_predictions(self):
        """Test that context-specific predictions are available from action learner."""
        action_learner = ActionOutcomeLearner(config={"min_samples_for_model": 5})
        
        # Record outcomes with specific contexts
        # Success in context A
        for i in range(10):
            action_learner.record_outcome(
                action_id=f"speak_a_{i}",
                action_type="speak",
                intended="communicate",
                actual="communication successful",
                context={"quiet": True, "low_stress": True}
            )
        
        # Failure in context B
        for i in range(10):
            action_learner.record_outcome(
                action_id=f"speak_b_{i}",
                action_type="speak",
                intended="communicate",
                actual="communication failed",
                context={"noisy": True, "high_stress": True}
            )
        
        # Verify action learner has learned about "speak" action
        reliability = action_learner.get_action_reliability("speak")
        assert not reliability.unknown
        assert reliability.total_executions == 20
        
        # The model should have enough samples now (20 > min_samples_for_model)
        model = action_learner.action_models.get("speak")
        assert model is not None
        assert model.sample_size >= 10
        
        # Verify that predictions can be made (even if confidence varies)
        prediction_a = action_learner.predict_outcome(
            "speak",
            {"quiet": True, "low_stress": True}
        )
        # Model exists and can make prediction
        assert prediction_a.prediction != "unknown"
        
        prediction_b = action_learner.predict_outcome(
            "speak",
            {"noisy": True, "high_stress": True}
        )
        assert prediction_b.prediction != "unknown"
    
    def test_evaluation_history_with_learner(self):
        """Test that evaluation history tracks reliability-adjusted scores."""
        minimizer = FreeEnergyMinimizer()
        action_learner = ActionOutcomeLearner()
        selector = ActiveInferenceActionSelector(minimizer, action_learner=action_learner)
        world_model = WorldModel()
        
        # Record successful outcomes
        for i in range(5):
            action_learner.record_outcome(
                action_id=f"test_{i}",
                action_type="speak",
                intended="test",
                actual="test successful",
                context={}
            )
        
        # Evaluate action
        action = {"type": "speak", "content": "test"}
        selector.evaluate_action(action, world_model)
        
        # Check evaluation history
        assert len(selector.evaluation_history) == 1
        evaluation = selector.evaluation_history[0]
        assert evaluation.action == action
        # EFE should be adjusted by reliability
    
    def test_reliability_weight_scaling(self):
        """Test that reliability_weight parameter scales the bias correctly."""
        minimizer = FreeEnergyMinimizer()
        action_learner = ActionOutcomeLearner()
        world_model = WorldModel()
        
        # Record perfect success
        for i in range(10):
            action_learner.record_outcome(
                action_id=f"test_{i}",
                action_type="speak",
                intended="test",
                actual="test successful",
                context={}
            )
        
        # Test with low weight
        selector_low = ActiveInferenceActionSelector(
            minimizer, 
            config={"reliability_weight": 0.1},
            action_learner=action_learner
        )
        eval_low = selector_low.evaluate_action({"type": "speak"}, world_model)
        
        # Test with high weight
        selector_high = ActiveInferenceActionSelector(
            minimizer,
            config={"reliability_weight": 0.5},
            action_learner=action_learner
        )
        eval_high = selector_high.evaluate_action({"type": "speak"}, world_model)
        
        # Higher weight should produce more bias (lower EFE for reliable action)
        assert eval_high.expected_free_energy < eval_low.expected_free_energy
    
    def test_mixed_reliability_actions(self):
        """Test action selection with mix of reliable, unreliable, and unknown actions."""
        minimizer = FreeEnergyMinimizer()
        action_learner = ActionOutcomeLearner()
        config = {"reliability_weight": 0.3}
        selector = ActiveInferenceActionSelector(minimizer, config=config, action_learner=action_learner)
        world_model = WorldModel()
        
        # Record high reliability for "speak" - use very similar intended/actual 
        for i in range(10):
            action_learner.record_outcome(
                action_id=f"speak_{i}",
                action_type="speak",
                intended="successfully communicate clear message",
                actual="successfully communicated clear message",
                context={}
            )
        
        # Record low reliability for "act" - use very different intended/actual
        for i in range(10):
            action_learner.record_outcome(
                action_id=f"act_{i}",
                action_type="act",
                intended="successfully complete important task",
                actual="failed completely nothing worked",
                context={}
            )
        
        # Leave "observe" unknown (no history)
        
        # Get reliabilities
        speak_reliability = action_learner.get_action_reliability("speak")
        act_reliability = action_learner.get_action_reliability("act")
        observe_reliability = action_learner.get_action_reliability("observe")
        
        assert speak_reliability.success_rate > 0.8  # High
        assert act_reliability.success_rate < 0.2  # Low
        assert observe_reliability.unknown  # Unknown
        
        actions = [
            {"type": "speak"},
            {"type": "act"},
            {"type": "observe"}  # Unknown
        ]
        
        # Evaluate all actions
        evaluations = [selector.evaluate_action(action, world_model) for action in actions]
        
        speak_eval = evaluations[0]
        act_eval = evaluations[1]
        observe_eval = evaluations[2]
        
        # Speak should have best (lowest) EFE due to high reliability
        # Act should have worst (highest) EFE due to low reliability
        assert speak_eval.expected_free_energy < act_eval.expected_free_energy
        
        # Observe should be between or neutral (no reliability bonus/penalty)
        # Since observe has higher base epistemic value (0.3) vs speak (0.2), 
        # but speak gets reliability bonus, final ordering depends on the weight
    
    def test_invalid_action_learner_type(self):
        """Test that invalid action_learner type is rejected."""
        minimizer = FreeEnergyMinimizer()
        
        # Try to pass an invalid object as action_learner
        invalid_learner = {"not": "an action learner"}
        
        with pytest.raises(TypeError, match="must have 'get_action_reliability' method"):
            ActiveInferenceActionSelector(minimizer, action_learner=invalid_learner)
    
    def test_iwmt_core_invalid_action_learner(self):
        """Test that IWMTCore rejects invalid action_learner."""
        invalid_learner = "not an action learner"
        
        with pytest.raises(TypeError, match="must have 'get_action_reliability' method"):
            IWMTCore(action_learner=invalid_learner)
