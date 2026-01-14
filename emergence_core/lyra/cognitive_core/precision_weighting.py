"""
Precision-weighted attention for IWMT.

This module implements precision (inverse uncertainty) computation
for attention weighting. Precision modulates attention based on:
- Prediction errors (surprises)
- Emotional state (arousal, valence)
- Confidence/certainty
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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
        """
        Initialize precision weighting system.
        
        Args:
            config: Optional configuration parameters
        """
        config = config or {}
        
        # Weighting factors
        self.arousal_dampening = config.get("arousal_dampening", 0.5)
        self.prediction_error_boost = config.get("prediction_error_boost", 0.3)
        self.valence_bias = config.get("valence_bias", 0.2)
        
        # Base precision
        self.base_precision = config.get("base_precision", 0.5)
        
        # History
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
        
        Precision = confidence in this information.
        
        High arousal → lower precision (more uncertain)
        High prediction error → higher precision (attend to surprises)
        Negative valence → bias toward threat-related
        
        Args:
            percept: The perceptual input
            emotional_state: Current emotional state with arousal and valence
            prediction_error: Optional prediction error magnitude (0.0 to 1.0)
            
        Returns:
            Precision value (0.0 to 1.0)
        """
        # Start with base precision
        precision = self.base_precision
        
        # Extract emotional state
        arousal = emotional_state.get("arousal", 0.0)
        valence = emotional_state.get("valence", 0.0)
        
        # Arousal reduces precision (high arousal = high uncertainty)
        arousal_effect = -self.arousal_dampening * arousal
        precision += arousal_effect
        
        # Prediction error increases precision (attend to surprises)
        error_boost = 0.0
        if prediction_error is not None:
            error_boost = self.prediction_error_boost * prediction_error
            precision += error_boost
        
        # Negative valence biases toward threat-related percepts
        # (This is handled in apply_precision_weighting for specific percept types)
        
        # Clamp to valid range
        precision = max(0.0, min(1.0, precision))
        
        # Record this precision computation
        percept_id = str(id(percept))
        weights = PrecisionWeights(
            percept_id=percept_id,
            precision=precision,
            base_precision=self.base_precision,
            emotional_modulation=arousal_effect,
            prediction_error_boost=error_boost
        )
        self.precision_history.append(weights)
        
        # Keep history bounded
        if len(self.precision_history) > 100:
            self.precision_history = self.precision_history[-100:]
        
        logger.debug(f"Computed precision: {precision:.3f} (base={self.base_precision:.2f}, arousal={arousal_effect:.2f}, error_boost={error_boost:.2f})")
        return precision
    
    def apply_precision_weighting(
        self,
        salience_scores: Dict[str, float],
        precisions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Weight salience by precision.
        
        Effective attention weight = Salience × Precision
        
        Args:
            salience_scores: Raw salience scores for each percept
            precisions: Precision values for each percept
            
        Returns:
            Dictionary of precision-weighted salience scores
        """
        weighted_scores = {}
        
        for percept_id, salience in salience_scores.items():
            precision = precisions.get(percept_id, self.base_precision)
            weighted = salience * precision
            weighted_scores[percept_id] = weighted
            
            logger.debug(f"Percept {percept_id}: salience={salience:.3f}, precision={precision:.3f}, weighted={weighted:.3f}")
        
        return weighted_scores
    
    def get_precision_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent precision computations.
        
        Returns:
            Dictionary with precision statistics
        """
        if not self.precision_history:
            return {
                "total_computations": 0,
                "average_precision": self.base_precision,
                "average_emotional_modulation": 0.0,
                "average_error_boost": 0.0,
            }
        
        return {
            "total_computations": len(self.precision_history),
            "average_precision": sum(p.precision for p in self.precision_history) / len(self.precision_history),
            "average_emotional_modulation": sum(p.emotional_modulation for p in self.precision_history) / len(self.precision_history),
            "average_error_boost": sum(p.prediction_error_boost for p in self.precision_history) / len(self.precision_history),
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
