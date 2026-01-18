"""
Active Inference Action Selection.

This module implements action selection based on active inference principles,
where actions are chosen to minimize expected free energy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime

if TYPE_CHECKING:
    from ..world_model import WorldModel

logger = logging.getLogger(__name__)


@dataclass
class ActionEvaluation:
    """
    Evaluation of a proposed action.
    
    Attributes:
        action: The action being evaluated
        expected_free_energy: Expected free energy after action
        epistemic_value: Information gain (uncertainty reduction)
        pragmatic_value: Goal achievement value
        confidence: Confidence in this evaluation
        timestamp: When evaluation was performed
    """
    action: Dict[str, Any]
    expected_free_energy: float
    epistemic_value: float
    pragmatic_value: float
    confidence: float
    timestamp: datetime


class ActiveInferenceActionSelector:
    """
    Select actions to confirm predictions or reduce uncertainty.
    
    Actions are chosen based on:
    1. Epistemic value: Do they reduce uncertainty?
    2. Pragmatic value: Do they achieve goals?
    """
    
    def __init__(self, free_energy_minimizer, config: Optional[Dict[str, Any]] = None):
        """
        Initialize action selector.
        
        Args:
            free_energy_minimizer: FreeEnergyMinimizer instance
            config: Optional configuration
        """
        self.free_energy = free_energy_minimizer
        config = config or {}
        
        # Thresholds
        self.action_threshold = config.get("action_threshold", 0.3)
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.5)
        
        # History
        self.evaluation_history: List[ActionEvaluation] = []
        
        logger.info("ActiveInferenceActionSelector initialized")
    
    def evaluate_action(
        self,
        action: Dict[str, Any],
        world_model: WorldModel
    ) -> ActionEvaluation:
        """
        Evaluate action by expected prediction error reduction.
        
        Args:
            action: Action to evaluate
            world_model: Current world model
            
        Returns:
            ActionEvaluation with detailed assessment
        """
        # Compute expected free energy
        efe = self.free_energy.expected_free_energy(action, world_model)
        
        # Decompose into epistemic and pragmatic values
        action_type = action.get("type", "unknown")
        
        # Heuristic value estimates
        epistemic_value = 0.0
        pragmatic_value = 0.0
        
        if action_type == "speak":
            epistemic_value = 0.2  # Speaking tests predictions
            pragmatic_value = 0.1
        elif action_type == "observe":
            epistemic_value = 0.3  # Observing reduces uncertainty
            pragmatic_value = 0.0
        elif action_type == "act":
            epistemic_value = 0.1
            pragmatic_value = 0.3  # Acting achieves goals
        elif action_type == "wait":
            epistemic_value = 0.05
            pragmatic_value = 0.0
        
        # Confidence based on world model state
        error_summary = world_model.get_prediction_error_summary()
        uncertainty = error_summary.get("average_surprise", 0.5)
        confidence = 1.0 - min(uncertainty, 1.0)
        
        evaluation = ActionEvaluation(
            action=action,
            expected_free_energy=efe,
            epistemic_value=epistemic_value,
            pragmatic_value=pragmatic_value,
            confidence=confidence,
            timestamp=datetime.now()
        )
        
        self.evaluation_history.append(evaluation)
        
        # Keep history bounded
        if len(self.evaluation_history) > 100:
            self.evaluation_history = self.evaluation_history[-100:]
        
        return evaluation
    
    def should_act(
        self,
        world_model: WorldModel,
        available_actions: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Determine if action is needed to reduce free energy.
        
        Args:
            world_model: Current world model
            available_actions: Optional list of available actions
            
        Returns:
            Tuple of (should_act, recommended_action)
        """
        # Compute current free energy
        current_fe = self.free_energy.compute_free_energy(world_model)
        
        # Get error summary for uncertainty assessment
        error_summary = world_model.get_prediction_error_summary()
        avg_surprise = error_summary.get("average_surprise", 0.0)
        
        # Decide if action is needed
        needs_action = (
            current_fe > self.action_threshold or
            avg_surprise > self.uncertainty_threshold
        )
        
        if not needs_action:
            logger.debug(f"No action needed (FE={current_fe:.3f}, surprise={avg_surprise:.3f})")
            return False, None
        
        # If action is needed, select best action
        if available_actions is None:
            # Default actions
            available_actions = [
                {"type": "speak", "reason": "reduce_uncertainty"},
                {"type": "observe", "reason": "gather_information"},
                {"type": "wait", "reason": "allow_settling"},
            ]
        
        # Evaluate all actions
        evaluations = [
            self.evaluate_action(action, world_model)
            for action in available_actions
        ]
        
        # Select best action (lowest expected free energy)
        best_eval = min(evaluations, key=lambda e: e.expected_free_energy)
        
        logger.info(f"Action recommended: {best_eval.action.get('type')} (EFE={best_eval.expected_free_energy:.3f})")
        return True, best_eval.action
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent action evaluations.
        
        Returns:
            Dictionary with evaluation statistics
        """
        if not self.evaluation_history:
            return {
                "total_evaluations": 0,
                "average_efe": 0.0,
                "average_epistemic": 0.0,
                "average_pragmatic": 0.0,
            }
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "average_efe": sum(e.expected_free_energy for e in self.evaluation_history) / len(self.evaluation_history),
            "average_epistemic": sum(e.epistemic_value for e in self.evaluation_history) / len(self.evaluation_history),
            "average_pragmatic": sum(e.pragmatic_value for e in self.evaluation_history) / len(self.evaluation_history),
            "recent_actions": [
                {
                    "type": e.action.get("type"),
                    "efe": e.expected_free_energy,
                    "epistemic": e.epistemic_value,
                    "pragmatic": e.pragmatic_value,
                }
                for e in self.evaluation_history[-5:]
            ]
        }
