"""
Free Energy Minimization for Active Inference.

This module implements free energy computation and minimization,
which is central to active inference and IWMT.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..world_model import WorldModel

logger = logging.getLogger(__name__)


class FreeEnergyMinimizer:
    """
    Compute and minimize free energy (prediction error).
    
    In active inference, free energy is a bound on surprise.
    Minimizing free energy means:
    1. Improving model predictions (perceptual inference)
    2. Acting to confirm predictions (active inference)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize free energy minimizer.
        
        Args:
            config: Optional configuration parameters
        """
        config = config or {}
        
        # Weighting factors
        self.prediction_error_weight = config.get("prediction_error_weight", 1.0)
        self.complexity_weight = config.get("complexity_weight", 0.1)
        
        logger.info("FreeEnergyMinimizer initialized")
    
    def compute_free_energy(self, world_model: WorldModel) -> float:
        """
        Current free energy (sum of prediction errors).
        
        Free energy = Prediction Error + Model Complexity
        
        Args:
            world_model: Current world model
            
        Returns:
            Free energy value (lower is better)
        """
        # Get prediction error summary
        error_summary = world_model.get_prediction_error_summary()
        
        # Prediction error term
        avg_magnitude = error_summary.get("average_magnitude", 0.0)
        avg_surprise = error_summary.get("average_surprise", 0.0)
        
        prediction_error = (avg_magnitude + avg_surprise) / 2.0
        
        # Model complexity term (based on number of entities/predictions)
        num_entities = world_model.environment_model.to_dict()["num_entities"]
        num_predictions = len(world_model.predictions)
        
        complexity = (num_entities + num_predictions) / 100.0  # Normalize
        
        # Total free energy
        free_energy = (
            self.prediction_error_weight * prediction_error +
            self.complexity_weight * complexity
        )
        
        logger.debug(f"Free energy: {free_energy:.3f} (error={prediction_error:.3f}, complexity={complexity:.3f})")
        return free_energy
    
    def expected_free_energy(
        self,
        action: Dict[str, Any],
        world_model: WorldModel
    ) -> float:
        """
        Expected free energy if action is taken.
        
        Expected free energy combines:
        1. Epistemic value: Does this reduce uncertainty?
        2. Pragmatic value: Does this achieve goals?
        
        Args:
            action: Proposed action
            world_model: Current world model
            
        Returns:
            Expected free energy after taking action (lower is better)
        """
        # Current free energy as baseline
        current_fe = self.compute_free_energy(world_model)
        
        # Estimate action effects
        action_type = action.get("type", "unknown")
        
        # Heuristic estimates based on action type
        epistemic_gain = 0.0  # Information gain
        pragmatic_gain = 0.0  # Goal achievement
        
        if action_type == "speak":
            # Speaking tests predictions and provides feedback
            epistemic_gain = 0.2
            pragmatic_gain = 0.1
        elif action_type == "observe":
            # Observing gathers information
            epistemic_gain = 0.3
            pragmatic_gain = 0.0
        elif action_type == "act":
            # Acting achieves goals
            epistemic_gain = 0.1
            pragmatic_gain = 0.3
        elif action_type == "wait":
            # Waiting provides minimal gain
            epistemic_gain = 0.05
            pragmatic_gain = 0.0
        
        # Expected free energy = current - expected reduction
        expected_fe = current_fe - (epistemic_gain + pragmatic_gain)
        
        logger.debug(f"Expected FE for {action_type}: {expected_fe:.3f} (epistemic={epistemic_gain:.2f}, pragmatic={pragmatic_gain:.2f})")
        return expected_fe
    
    def select_action(
        self,
        available_actions: List[Dict[str, Any]],
        world_model: WorldModel
    ) -> Dict[str, Any]:
        """
        Select action that minimizes expected free energy.
        
        Args:
            available_actions: List of possible actions
            world_model: Current world model
            
        Returns:
            Selected action (the one with lowest expected free energy)
        """
        if not available_actions:
            logger.warning("No available actions")
            return {"type": "wait", "reason": "no_actions_available"}
        
        # Evaluate each action
        best_action = None
        best_efe = float('inf')
        
        for action in available_actions:
            efe = self.expected_free_energy(action, world_model)
            
            if efe < best_efe:
                best_efe = efe
                best_action = action
        
        logger.info(f"Selected action: {best_action.get('type', 'unknown')} (EFE={best_efe:.3f})")
        return best_action or available_actions[0]
