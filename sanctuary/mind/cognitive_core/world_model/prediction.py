"""
Prediction and PredictionError classes for IWMT.

This module defines the basic structures for predictions and prediction errors
in the predictive processing framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
import math


@dataclass
class Prediction:
    """
    A prediction about future states.
    
    Attributes:
        content: What is being predicted (text description)
        confidence: Confidence in this prediction (0.0 to 1.0)
        time_horizon: How far into the future (in seconds)
        source: Which model/component generated this prediction
        created_at: When this prediction was made
    """
    content: str
    confidence: float
    time_horizon: float
    source: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate prediction parameters."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.time_horizon < 0:
            raise ValueError(f"Time horizon must be non-negative, got {self.time_horizon}")


@dataclass
class PredictionError:
    """
    Error/mismatch between prediction and actual observation.
    
    Attributes:
        prediction: The prediction that was made
        actual: What actually happened/was observed
        magnitude: Size of the error (0.0 to 1.0)
        surprise: Information-theoretic surprise (in nats or bits)
        timestamp: When this error was computed
    """
    prediction: Prediction
    actual: Any
    magnitude: float
    surprise: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate prediction error parameters."""
        if not 0.0 <= self.magnitude <= 1.0:
            raise ValueError(f"Magnitude must be in [0, 1], got {self.magnitude}")
        if self.surprise < 0:
            raise ValueError(f"Surprise must be non-negative, got {self.surprise}")
    
    @staticmethod
    def compute_surprise(confidence: float) -> float:
        """
        Compute information-theoretic surprise from confidence.
        
        Surprise = -log(P(observation|model))
        Higher confidence in wrong prediction = higher surprise
        
        Args:
            confidence: Confidence in the (incorrect) prediction
            
        Returns:
            Surprise value in nats (natural logarithm)
        """
        # Clamp confidence away from 0 to avoid infinity
        confidence = max(0.001, min(0.999, confidence))
        # Surprise is negative log probability
        # If we were confident in the prediction, but it was wrong, surprise is high
        return -math.log(1.0 - confidence)
