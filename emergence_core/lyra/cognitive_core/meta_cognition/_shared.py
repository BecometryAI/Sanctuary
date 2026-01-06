"""
Shared data structures for meta-cognition subsystem.

This module contains dataclasses used across multiple meta-cognition components.
"""

from __future__ import annotations

from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class PredictionRecord:
    """
    Comprehensive record of a single prediction.
    
    Attributes:
        id: Unique prediction identifier
        timestamp: When prediction was made
        category: Type of prediction (action, emotion, capability, etc.)
        predicted_state: What was predicted
        predicted_confidence: Confidence in prediction (0.0-1.0)
        actual_state: What actually happened (filled after observation)
        correct: Whether prediction was correct (filled after validation)
        error_magnitude: Size of prediction error if continuous
        context: Contextual information at prediction time
        validated_at: When prediction was validated
        self_model_version: Self-model state at prediction time
    """
    id: str
    timestamp: datetime
    category: str
    predicted_state: Dict[str, Any]
    predicted_confidence: float
    actual_state: Optional[Dict[str, Any]] = None
    correct: Optional[bool] = None
    error_magnitude: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    validated_at: Optional[datetime] = None
    self_model_version: int = 0


@dataclass
class AccuracySnapshot:
    """
    Point-in-time accuracy snapshot.
    
    Attributes:
        timestamp: When snapshot was taken
        overall_accuracy: Overall accuracy at this time
        category_accuracies: Accuracies by category
        calibration_score: Calibration quality
        prediction_count: Number of predictions in window
        self_model_version: Self-model version at this time
    """
    timestamp: datetime
    overall_accuracy: float
    category_accuracies: Dict[str, float]
    calibration_score: float
    prediction_count: int
    self_model_version: int
