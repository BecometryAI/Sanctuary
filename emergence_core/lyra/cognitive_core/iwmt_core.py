"""
IWMT Core: Central coordinator for IWMT-based cognition.

This module integrates all IWMT components:
- WorldModel (predictive processing)
- FreeEnergyMinimizer (active inference)
- PrecisionWeighting (precision-weighted attention)
- ActiveInferenceActionSelector (action selection)
- AtomspaceBridge (MeTTa integration - optional)
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .world_model import WorldModel
from .active_inference import FreeEnergyMinimizer, ActiveInferenceActionSelector
from .precision_weighting import PrecisionWeighting
from .metta import AtomspaceBridge

logger = logging.getLogger(__name__)


class IWMTCore:
    """
    Central coordinator for IWMT-based cognition.
    
    Implements the IWMT cognitive cycle:
    1. Update world model with new percepts
    2. Compute prediction errors
    3. Apply precision-weighted attention
    4. Select actions via active inference
    5. Update self-model from outcomes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IWMT core with all components.
        
        Args:
            config: Optional configuration for all subsystems
        """
        config = config or {}
        
        # Core IWMT components
        self.world_model = WorldModel()
        self.free_energy = FreeEnergyMinimizer(config.get("free_energy", {}))
        self.precision = PrecisionWeighting(config.get("precision", {}))
        self.active_inference = ActiveInferenceActionSelector(
            self.free_energy,
            config.get("action_selection", {})
        )
        
        # Optional MeTTa integration
        metta_config = config.get("metta", {"use_metta": False})
        self.metta_bridge = AtomspaceBridge(metta_config)
        
        # Cycle state
        self.cycle_count = 0
        self.last_cycle_time: Optional[datetime] = None
        
        logger.info("IWMTCore initialized")
    
    async def cognitive_cycle(
        self,
        percepts: List[Any],
        emotional_state: Dict[str, float],
        goals: Optional[List[Any]] = None,
        available_actions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute one IWMT cognitive cycle.
        
        Steps:
        1. Update world model with new percepts
        2. Compute prediction errors (surprises)
        3. Compute precision weights for attention
        4. Evaluate need for action
        5. If action needed, select via active inference
        6. Return cycle results
        
        Args:
            percepts: New perceptual inputs
            emotional_state: Current emotional state (arousal, valence)
            goals: Current goals (optional)
            available_actions: Available actions (optional)
            
        Returns:
            Dictionary with cycle results including:
            - prediction_errors: List of errors detected
            - free_energy: Current free energy level
            - action_recommended: Recommended action (if any)
            - precision_summary: Attention precision statistics
        """
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        logger.debug(f"Starting IWMT cycle {self.cycle_count}")
        
        # Step 1: Generate predictions before processing percepts
        context = {
            "goals": goals or [],
            "emotional_state": emotional_state,
            "percepts": percepts
        }
        predictions = self.world_model.predict(time_horizon=1.0, context=context)
        
        # Step 2: Process percepts and compute prediction errors
        prediction_errors = []
        for percept in percepts:
            error = self.world_model.update_on_percept(percept)
            if error:
                prediction_errors.append(error)
        
        # Step 3: Compute precision weights for attention
        precision_weights = {}
        for percept in percepts:
            percept_id = str(id(percept))
            
            # Get prediction error for this percept (if any)
            error_magnitude = None
            for error in prediction_errors:
                if str(id(error.actual)) == percept_id:
                    error_magnitude = error.magnitude
                    break
            
            # Compute precision
            precision = self.precision.compute_precision(
                percept,
                emotional_state,
                error_magnitude
            )
            precision_weights[percept_id] = precision
        
        # Step 4: Compute free energy
        current_fe = self.free_energy.compute_free_energy(self.world_model)
        
        # Step 5: Determine if action is needed via active inference
        should_act, recommended_action = self.active_inference.should_act(
            self.world_model,
            available_actions
        )
        
        # Prepare cycle results
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        self.last_cycle_time = cycle_start
        
        results = {
            "cycle_number": self.cycle_count,
            "cycle_time_seconds": cycle_time,
            "timestamp": cycle_start.isoformat(),
            
            # Predictions
            "num_predictions": len(predictions),
            "predictions": [
                {
                    "content": p.content,
                    "confidence": p.confidence,
                    "source": p.source
                }
                for p in predictions[:5]  # First 5 predictions
            ],
            
            # Prediction errors
            "num_prediction_errors": len(prediction_errors),
            "prediction_errors": [
                {
                    "predicted": e.prediction.content,
                    "actual": str(e.actual)[:100],
                    "magnitude": e.magnitude,
                    "surprise": e.surprise
                }
                for e in prediction_errors[:5]  # First 5 errors
            ],
            
            # Free energy and uncertainty
            "free_energy": current_fe,
            "prediction_error_summary": self.world_model.get_prediction_error_summary(),
            
            # Attention precision
            "precision_summary": self.precision.get_precision_summary(),
            "num_precision_weights": len(precision_weights),
            
            # Action recommendation
            "should_act": should_act,
            "recommended_action": recommended_action,
            
            # World model state
            "world_model_state": self.world_model.to_dict(),
        }
        
        logger.info(f"Completed IWMT cycle {self.cycle_count}: FE={current_fe:.3f}, "
                   f"errors={len(prediction_errors)}, should_act={should_act}")
        
        return results
    
    def update_from_action_outcome(
        self,
        action: Dict[str, Any],
        outcome: Dict[str, Any]
    ):
        """
        Update IWMT core based on action outcome.
        
        This updates the world model's self-model and environment model
        based on what happened when an action was taken.
        
        Args:
            action: The action that was taken
            outcome: The result of that action
        """
        self.world_model.update_from_action_outcome(action, outcome)
        logger.debug(f"Updated world model from action: {action.get('type', 'unknown')}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of IWMT core.
        
        Returns:
            Dictionary with status information
        """
        return {
            "cycle_count": self.cycle_count,
            "last_cycle": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            "world_model": self.world_model.to_dict(),
            "free_energy": self.free_energy.compute_free_energy(self.world_model),
            "precision_summary": self.precision.get_precision_summary(),
            "action_evaluation_summary": self.active_inference.get_evaluation_summary(),
            "metta_available": self.metta_bridge.is_available(),
        }
