"""
WorldModel: Hierarchical predictive world model (IWMT core).

This module implements the core WorldModel class that integrates:
- SelfModel: Representation of the agent itself
- EnvironmentModel: Representation of the external world
- Prediction tracking and prediction error computation
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .prediction import Prediction, PredictionError
from .self_model import SelfModel
from .environment_model import EnvironmentModel

logger = logging.getLogger(__name__)


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
        
        # Store predictions
        self.predictions.extend(predictions)
        
        # Keep predictions list bounded
        if len(self.predictions) > 200:
            self.predictions = self.predictions[-200:]
        
        logger.debug(f"Generated {len(predictions)} predictions for horizon {time_horizon}s")
        return predictions
    
    def update_on_percept(self, percept: Any) -> Optional[PredictionError]:
        """
        Compare percept to predictions, compute prediction error.
        
        Args:
            percept: New perceptual input
            
        Returns:
            PredictionError if there's a mismatch, None otherwise
        """
        # Simple heuristic: check if percept contradicts any active predictions
        # In a full implementation, this would use semantic comparison
        
        if not self.predictions:
            # No predictions to compare against
            return None
        
        # Find most relevant prediction (for now, use most recent)
        relevant_prediction = self.predictions[-1] if self.predictions else None
        
        if relevant_prediction is None:
            return None
        
        # Compute prediction error
        # For this implementation, we'll use a simple heuristic
        # In practice, this would involve semantic comparison
        
        # Extract percept content
        percept_content = str(percept) if not isinstance(percept, dict) else percept.get("content", str(percept))
        
        # Compare prediction to percept (simple string-based heuristic)
        prediction_text = relevant_prediction.content.lower()
        percept_text = percept_content.lower()
        
        # Simple overlap check
        overlap = len(set(prediction_text.split()) & set(percept_text.split()))
        total_words = len(set(prediction_text.split()) | set(percept_text.split()))
        
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
            
            # Keep error list bounded
            if len(self.prediction_errors) > 100:
                self.prediction_errors = self.prediction_errors[-100:]
            
            logger.debug(f"Prediction error detected: magnitude={magnitude:.2f}, surprise={surprise:.2f}")
            return error
        
        return None
    
    def get_prediction_error_summary(self) -> Dict[str, Any]:
        """
        Summary of current prediction errors.
        
        Returns:
            Dictionary with error statistics
        """
        if not self.prediction_errors:
            return {
                "total_errors": 0,
                "average_magnitude": 0.0,
                "average_surprise": 0.0,
                "max_surprise": 0.0,
            }
        
        magnitudes = [e.magnitude for e in self.prediction_errors]
        surprises = [e.surprise for e in self.prediction_errors]
        
        return {
            "total_errors": len(self.prediction_errors),
            "average_magnitude": sum(magnitudes) / len(magnitudes),
            "average_surprise": sum(surprises) / len(surprises),
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
