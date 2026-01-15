"""
IWMT Core: Central coordinator for IWMT-based cognition.

Integrates all IWMT components in a unified cognitive cycle.
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

# Configuration constants
DEFAULT_TIME_HORIZON = 1.0
MAX_RESULT_PREDICTIONS = 5
MAX_RESULT_ERRORS = 5


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
        """Initialize IWMT core with all components."""
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
        self.metta_bridge = AtomspaceBridge(config.get("metta", {"use_metta": False}))
        
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
        
        Cycle steps:
        1. Generate predictions based on context
        2. Process percepts and compute prediction errors
        3. Compute precision weights for attention
        4. Evaluate action need via free energy
        5. Select action if needed
        """
        self.cycle_count += 1
        cycle_start = datetime.now()
        
        logger.debug(f"Starting IWMT cycle {self.cycle_count}")
        
        # Step 1: Generate predictions
        context = {
            "goals": goals or [],
            "emotional_state": emotional_state,
            "percepts": percepts
        }
        predictions = self.world_model.predict(DEFAULT_TIME_HORIZON, context)
        
        # Step 2: Process percepts and compute errors
        prediction_errors = [
            error for error in 
            (self.world_model.update_on_percept(p) for p in percepts)
            if error is not None
        ]
        
        # Step 3: Compute precision weights
        precision_weights = self._compute_precision_weights(
            percepts, prediction_errors, emotional_state
        )
        
        # Step 4: Compute free energy
        current_fe = self.free_energy.compute_free_energy(self.world_model)
        
        # Step 5: Determine action
        should_act, recommended_action = self.active_inference.should_act(
            self.world_model,
            available_actions
        )
        
        # Build and return results
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        self.last_cycle_time = cycle_start
        
        return self._build_cycle_results(
            cycle_start, cycle_time, predictions, prediction_errors,
            precision_weights, current_fe, should_act, recommended_action
        )
    
    def _compute_precision_weights(
        self,
        percepts: List[Any],
        prediction_errors: List[Any],
        emotional_state: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute precision weights for all percepts."""
        precision_weights = {}
        
        for percept in percepts:
            percept_id = str(id(percept))
            
            # Find matching prediction error
            error_magnitude = next(
                (e.magnitude for e in prediction_errors if str(id(e.actual)) == percept_id),
                None
            )
            
            # Compute precision
            precision = self.precision.compute_precision(
                percept, emotional_state, error_magnitude
            )
            precision_weights[percept_id] = precision
        
        return precision_weights
    
    def _build_cycle_results(
        self,
        cycle_start: datetime,
        cycle_time: float,
        predictions: List[Any],
        prediction_errors: List[Any],
        precision_weights: Dict[str, float],
        current_fe: float,
        should_act: bool,
        recommended_action: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build cycle results dictionary."""
        return {
            "cycle_number": self.cycle_count,
            "cycle_time_seconds": cycle_time,
            "timestamp": cycle_start.isoformat(),
            "num_predictions": len(predictions),
            "predictions": [
                {"content": p.content, "confidence": p.confidence, "source": p.source}
                for p in predictions[:MAX_RESULT_PREDICTIONS]
            ],
            "num_prediction_errors": len(prediction_errors),
            "prediction_errors": [
                {
                    "predicted": e.prediction.content,
                    "actual": str(e.actual)[:100],
                    "magnitude": e.magnitude,
                    "surprise": e.surprise,
                }
                for e in prediction_errors[:MAX_RESULT_ERRORS]
            ],
            "free_energy": current_fe,
            "prediction_error_summary": self.world_model.get_prediction_error_summary(),
            "precision_summary": self.precision.get_precision_summary(),
            "num_precision_weights": len(precision_weights),
            "should_act": should_act,
            "recommended_action": recommended_action,
            "world_model_state": self.world_model.to_dict(),
        }
    
    def update_from_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any]):
        """Update IWMT core based on action outcome."""
        self.world_model.update_from_action_outcome(action, outcome)
        logger.debug(f"Updated world model: {action.get('type', 'unknown')}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of IWMT core."""
        return {
            "cycle_count": self.cycle_count,
            "last_cycle": self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            "world_model": self.world_model.to_dict(),
            "free_energy": self.free_energy.compute_free_energy(self.world_model),
            "precision_summary": self.precision.get_precision_summary(),
            "action_evaluation_summary": self.active_inference.get_evaluation_summary(),
            "metta_available": self.metta_bridge.is_available(),
        }
