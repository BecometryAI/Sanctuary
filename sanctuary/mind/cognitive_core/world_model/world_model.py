"""
WorldModel: Hierarchical predictive world model (IWMT core).

Integrates self-model, environment model, and prediction tracking.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .prediction import Prediction, PredictionError
from .self_model import SelfModel
from .environment_model import EnvironmentModel

logger = logging.getLogger(__name__)

# Configuration constants
MAX_PREDICTIONS = 200
MAX_ERRORS = 100


class WorldModel:
    """
    Hierarchical predictive world model (IWMT core).
    
    Integrates:
    - Self-model: Embodied representation of the agent
    - Environment model: External world representation
    - Predictions: Active predictions about future states
    - Prediction errors: Mismatches between predictions and observations
    """
    
    def __init__(self):
        """Initialize world model with self and environment models."""
        self.self_model = SelfModel()
        self.environment_model = EnvironmentModel()
        
        # Active predictions (combined from both models)
        self.predictions: List[Prediction] = []
        
        # Prediction errors (surprises/mismatches)
        self.prediction_errors: List[PredictionError] = []
        
        logger.info("WorldModel initialized")
    
    def predict(self, time_horizon: float, context: Optional[Dict[str, Any]] = None) -> List[Prediction]:
        """
        Generate predictions about future states.
        
        Combines predictions from both self-model and environment model.
        
        Args:
            time_horizon: How far into the future to predict (seconds)
            context: Current context for making predictions
            
        Returns:
            List of predictions
        """
        context = context or {}
        predictions = []
        
        # Get self-predictions
        self_prediction = self.self_model.predict_own_behavior(context)
        predictions.append(self_prediction)
        
        # Get environment predictions
        env_predictions = self.environment_model.predict_environment(time_horizon, context)
        predictions.extend(env_predictions)
        
        # Store predictions and keep bounded
        self.predictions.extend(predictions)
        if len(self.predictions) > MAX_PREDICTIONS:
            self.predictions = self.predictions[-MAX_PREDICTIONS:]
        
        logger.debug(f"Generated {len(predictions)} predictions for horizon {time_horizon}s")
        return predictions
    
    def update_on_percept(self, percept: Any) -> Optional[PredictionError]:
        """
        Compare percept to predictions, compute prediction error.
        
        Args:
            percept: New perceptual input
            
        Returns:
            PredictionError if mismatch detected, None otherwise
        """
        if not self.predictions:
            return None
        
        # Use most recent prediction
        relevant_prediction = self.predictions[-1]
        
        # Extract percept content
        percept_content = str(percept) if not isinstance(percept, dict) else percept.get("content", str(percept))
        
        # Compare prediction to percept (simple word overlap heuristic)
        prediction_words = set(relevant_prediction.content.lower().split())
        percept_words = set(percept_content.lower().split())
        
        overlap = len(prediction_words & percept_words)
        total_words = len(prediction_words | percept_words)
        
        if total_words == 0:
            match_score = 0.0
        else:
            match_score = overlap / total_words
        
        # If match is low, we have a prediction error
        if match_score < 0.3:
            magnitude = 1.0 - match_score
            surprise = PredictionError.compute_surprise(relevant_prediction.confidence)
            
            error = PredictionError(
                prediction=relevant_prediction,
                actual=percept_content,
                magnitude=magnitude,
                surprise=surprise,
                timestamp=datetime.now()
            )
            
            self.prediction_errors.append(error)
            if len(self.prediction_errors) > MAX_ERRORS:
                self.prediction_errors = self.prediction_errors[-MAX_ERRORS:]
            
            logger.debug(f"Prediction error: magnitude={magnitude:.2f}, surprise={surprise:.2f}")
            return error
        
        return None
    
    def get_prediction_error_summary(self) -> Dict[str, Any]:
        """Summary of current prediction errors."""
        if not self.prediction_errors:
            return {
                "total_errors": 0,
                "average_magnitude": 0.0,
                "average_surprise": 0.0,
                "max_surprise": 0.0,
            }
        
        magnitudes = [e.magnitude for e in self.prediction_errors]
        surprises = [e.surprise for e in self.prediction_errors]
        num_errors = len(self.prediction_errors)
        
        return {
            "total_errors": num_errors,
            "average_magnitude": sum(magnitudes) / num_errors,
            "average_surprise": sum(surprises) / num_errors,
            "max_surprise": max(surprises),
            "recent_errors": [
                {
                    "prediction": e.prediction.content,
                    "actual": str(e.actual)[:100],
                    "magnitude": e.magnitude,
                    "surprise": e.surprise,
                }
                for e in self.prediction_errors[-5:]
            ]
        }
    
    def update_from_action_outcome(self, action: Dict[str, Any], outcome: Dict[str, Any]):
        """
        Update world model based on action outcome.
        
        Args:
            action: The action that was taken
            outcome: The result of that action
        """
        # Update self-model
        self.self_model.update_from_action(action, outcome)
        
        # Update environment model if outcome contains observations
        if "observation" in outcome:
            self.environment_model.update_from_observation(outcome["observation"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Export world model state as dictionary."""
        return {
            "self_model": self.self_model.to_dict(),
            "environment_model": self.environment_model.to_dict(),
            "num_predictions": len(self.predictions),
            "num_prediction_errors": len(self.prediction_errors),
            "prediction_error_summary": self.get_prediction_error_summary(),
        }
