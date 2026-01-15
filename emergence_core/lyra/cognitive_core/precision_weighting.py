"""
Precision-weighted attention for IWMT.

Computes precision (inverse uncertainty) for attention modulation.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_AROUSAL_DAMPENING = 0.5
DEFAULT_ERROR_BOOST = 0.3
DEFAULT_VALENCE_BIAS = 0.2
DEFAULT_BASE_PRECISION = 0.5
MAX_HISTORY = 100


@dataclass
class PrecisionWeights:
    """
    Precision weights for different percepts/stimuli.
    
    Attributes:
        percept_id: Identifier for the percept
        precision: Precision value (0.0 to 1.0, higher = more certain)
        base_precision: Base precision without emotional modulation
        emotional_modulation: Adjustment due to emotional state
        prediction_error_boost: Boost due to prediction error
    """
    percept_id: str
    precision: float
    base_precision: float
    emotional_modulation: float
    prediction_error_boost: float


class PrecisionWeighting:
    """
    Compute precision (inverse uncertainty) for attention.
    
    Precision determines how much to weight different sources of information.
    High precision = high certainty = strong attention
    Low precision = high uncertainty = weak attention
    
    Factors affecting precision:
    - Prediction errors: Higher error -> higher precision (attend to surprises)
    - Emotional arousal: Higher arousal -> lower precision (more uncertain)
    - Emotional valence: Negative valence -> bias toward threat-related
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize precision weighting system."""
        config = config or {}
        self.arousal_dampening = config.get("arousal_dampening", DEFAULT_AROUSAL_DAMPENING)
        self.prediction_error_boost = config.get("prediction_error_boost", DEFAULT_ERROR_BOOST)
        self.valence_bias = config.get("valence_bias", DEFAULT_VALENCE_BIAS)
        self.base_precision = config.get("base_precision", DEFAULT_BASE_PRECISION)
        self.precision_history: List[PrecisionWeights] = []
        logger.info("PrecisionWeighting initialized")
    
    def compute_precision(
        self,
        percept: Any,
        emotional_state: Dict[str, float],
        prediction_error: Optional[float] = None
    ) -> float:
        """
        Compute precision (inverse uncertainty) for a percept.
        
        High arousal → lower precision
        High prediction error → higher precision (attend to surprises)
        """
        # Start with base and apply modulations
        arousal = emotional_state.get("arousal", 0.0)
        arousal_effect = -self.arousal_dampening * arousal
        
        error_boost = 0.0
        if prediction_error is not None:
            error_boost = self.prediction_error_boost * prediction_error
        
        precision = self.base_precision + arousal_effect + error_boost
        
        # Clamp to valid range and record
        precision = max(0.0, min(1.0, precision))
        
        weights = PrecisionWeights(
            percept_id=str(id(percept)),
            precision=precision,
            base_precision=self.base_precision,
            emotional_modulation=arousal_effect,
            prediction_error_boost=error_boost
        )
        self.precision_history.append(weights)
        
        if len(self.precision_history) > MAX_HISTORY:
            self.precision_history = self.precision_history[-MAX_HISTORY:]
        
        logger.debug(f"Precision: {precision:.3f} (base={self.base_precision:.2f}, arousal={arousal_effect:.2f}, error={error_boost:.2f})")
        return precision
    
    def apply_precision_weighting(
        self,
        salience_scores: Dict[str, float],
        precisions: Dict[str, float]
    ) -> Dict[str, float]:
        """Weight salience by precision: attention = salience × precision."""
        return {
            percept_id: salience * precisions.get(percept_id, self.base_precision)
            for percept_id, salience in salience_scores.items()
        }
    
    def get_precision_summary(self) -> Dict[str, Any]:
        """Get summary of recent precision computations."""
        if not self.precision_history:
            return {
                "total_computations": 0,
                "average_precision": self.base_precision,
                "average_emotional_modulation": 0.0,
                "average_error_boost": 0.0,
            }
        
        num_computations = len(self.precision_history)
        return {
            "total_computations": num_computations,
            "average_precision": sum(p.precision for p in self.precision_history) / num_computations,
            "average_emotional_modulation": sum(p.emotional_modulation for p in self.precision_history) / num_computations,
            "average_error_boost": sum(p.prediction_error_boost for p in self.precision_history) / num_computations,
            "recent_precisions": [
                {
                    "precision": p.precision,
                    "base": p.base_precision,
                    "emotional": p.emotional_modulation,
                    "error_boost": p.prediction_error_boost,
                }
                for p in self.precision_history[-5:]
            ]
        }
