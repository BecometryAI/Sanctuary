"""
World Model: Hierarchical predictive world model (IWMT core).

This module implements the predictive processing layer for IWMT, including:
- WorldModel: Core hierarchical predictive model
- Prediction and PredictionError tracking
- SelfModel: Embodied self representation
- EnvironmentModel: External world representation
"""

from .world_model import WorldModel
from .prediction import Prediction, PredictionError
from .self_model import SelfModel
from .environment_model import EnvironmentModel, EntityModel, Relationship

__all__ = [
    "WorldModel",
    "Prediction",
    "PredictionError",
    "SelfModel",
    "EnvironmentModel",
    "EntityModel",
    "Relationship",
]
