"""
Active Inference: Free energy minimization and action selection.

This module implements the active inference framework for IWMT, including:
- FreeEnergyMinimizer: Computes and minimizes free energy
- ActiveInferenceActionSelector: Selects actions to reduce uncertainty
"""

from .free_energy import FreeEnergyMinimizer
from .action_selection import ActiveInferenceActionSelector, ActionEvaluation

__all__ = [
    "FreeEnergyMinimizer",
    "ActiveInferenceActionSelector",
    "ActionEvaluation",
]
