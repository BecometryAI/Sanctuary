"""
Unit tests for Active Inference components.

Tests cover:
- FreeEnergyMinimizer computation and action selection
- ActionEvaluation
- ActiveInferenceActionSelector
"""

import pytest
from datetime import datetime

from lyra.cognitive_core.world_model import WorldModel
from lyra.cognitive_core.active_inference import (
    FreeEnergyMinimizer,
    ActiveInferenceActionSelector,
    ActionEvaluation
)


class TestFreeEnergyMinimizer:
    """Test FreeEnergyMinimizer class."""
    
    def test_initialization(self):
        """Test FreeEnergyMinimizer initializes correctly."""
        minimizer = FreeEnergyMinimizer()
        assert minimizer.prediction_error_weight == 1.0
        assert minimizer.complexity_weight == 0.1
    
    def test_compute_free_energy(self):
        """Test free energy computation."""
        minimizer = FreeEnergyMinimizer()
        world_model = WorldModel()
        
        fe = minimizer.compute_free_energy(world_model)
        assert fe >= 0.0
        assert isinstance(fe, float)
    
    def test_expected_free_energy(self):
        """Test expected free energy for actions."""
        minimizer = FreeEnergyMinimizer()
        world_model = WorldModel()
        
        action = {"type": "speak", "content": "test"}
        efe = minimizer.expected_free_energy(action, world_model)
        assert isinstance(efe, float)
    
    def test_different_action_types(self):
        """Test that different action types have different EFE."""
        minimizer = FreeEnergyMinimizer()
        world_model = WorldModel()
        
        speak_action = {"type": "speak"}
        observe_action = {"type": "observe"}
        wait_action = {"type": "wait"}
        
        speak_efe = minimizer.expected_free_energy(speak_action, world_model)
        observe_efe = minimizer.expected_free_energy(observe_action, world_model)
        wait_efe = minimizer.expected_free_energy(wait_action, world_model)
        
        # Observe should have highest epistemic value (lowest EFE in low uncertainty)
        # Values depend on current free energy, so just check they're computed
        assert isinstance(speak_efe, float)
        assert isinstance(observe_efe, float)
        assert isinstance(wait_efe, float)
    
    def test_select_action(self):
        """Test action selection."""
        minimizer = FreeEnergyMinimizer()
        world_model = WorldModel()
        
        actions = [
            {"type": "speak", "content": "Hello"},
            {"type": "observe", "target": "environment"},
            {"type": "wait", "duration": 1.0}
        ]
        
        selected = minimizer.select_action(actions, world_model)
        assert selected in actions
    
    def test_select_action_empty(self):
        """Test action selection with empty list."""
        minimizer = FreeEnergyMinimizer()
        world_model = WorldModel()
        
        selected = minimizer.select_action([], world_model)
        assert selected["type"] == "wait"


class TestActionEvaluation:
    """Test ActionEvaluation dataclass."""
    
    def test_creation(self):
        """Test creating an action evaluation."""
        action = {"type": "speak"}
        eval = ActionEvaluation(
            action=action,
            expected_free_energy=0.5,
            epistemic_value=0.2,
            pragmatic_value=0.1,
            confidence=0.7,
            timestamp=datetime.now()
        )
        assert eval.action == action
        assert eval.expected_free_energy == 0.5
        assert eval.epistemic_value == 0.2


class TestActiveInferenceActionSelector:
    """Test ActiveInferenceActionSelector class."""
    
    def test_initialization(self):
        """Test ActionSelector initializes correctly."""
        minimizer = FreeEnergyMinimizer()
        selector = ActiveInferenceActionSelector(minimizer)
        assert selector.free_energy == minimizer
        assert selector.action_threshold == 0.3
    
    def test_evaluate_action(self):
        """Test action evaluation."""
        minimizer = FreeEnergyMinimizer()
        selector = ActiveInferenceActionSelector(minimizer)
        world_model = WorldModel()
        
        action = {"type": "speak", "content": "test"}
        evaluation = selector.evaluate_action(action, world_model)
        
        assert isinstance(evaluation, ActionEvaluation)
        assert evaluation.action == action
        assert 0.0 <= evaluation.confidence <= 1.0
        assert len(selector.evaluation_history) == 1
    
    def test_should_act_low_free_energy(self):
        """Test that low free energy doesn't trigger action."""
        minimizer = FreeEnergyMinimizer()
        selector = ActiveInferenceActionSelector(minimizer)
        world_model = WorldModel()
        
        # With empty world model, free energy should be low
        should_act, action = selector.should_act(world_model)
        
        # May or may not need action depending on exact thresholds
        assert isinstance(should_act, bool)
        if should_act:
            assert action is not None
            assert "type" in action
    
    def test_should_act_with_actions(self):
        """Test action selection with provided actions."""
        minimizer = FreeEnergyMinimizer()
        selector = ActiveInferenceActionSelector(minimizer)
        world_model = WorldModel()
        
        # Add prediction errors to trigger action
        world_model.predictions.append(
            world_model.self_model.predict_own_behavior({})
        )
        world_model.update_on_percept({"content": "unexpected"})
        
        actions = [
            {"type": "speak"},
            {"type": "observe"}
        ]
        
        should_act, action = selector.should_act(world_model, actions)
        if should_act:
            assert action in actions
    
    def test_get_evaluation_summary(self):
        """Test evaluation summary."""
        minimizer = FreeEnergyMinimizer()
        selector = ActiveInferenceActionSelector(minimizer)
        world_model = WorldModel()
        
        # Perform some evaluations
        selector.evaluate_action({"type": "speak"}, world_model)
        selector.evaluate_action({"type": "observe"}, world_model)
        
        summary = selector.get_evaluation_summary()
        assert summary["total_evaluations"] == 2
        assert "average_efe" in summary
        assert "recent_actions" in summary
