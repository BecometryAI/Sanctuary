"""
Goal Competition System

This module implements resource-based goal competition for Sanctuary's cognitive system.
Goals compete for limited cognitive resources using activation-based dynamics with
lateral inhibition, allowing for realistic goal selection and prioritization.
"""

from .resources import CognitiveResources, ResourcePool
from .competition import GoalCompetition, ActiveGoal
from .interactions import GoalInteraction
from .metrics import GoalCompetitionMetrics
from .dynamics import GoalDynamics, GoalAdjustment

__all__ = [
    "CognitiveResources",
    "ResourcePool",
    "GoalCompetition",
    "ActiveGoal",
    "GoalInteraction",
    "GoalCompetitionMetrics",
    "GoalDynamics",
    "GoalAdjustment",
]
