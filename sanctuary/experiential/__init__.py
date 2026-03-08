"""Experiential layer — CfC continuous-time neural dynamics.

CfC (Closed-form Continuous-depth) cells evolve state between LLM cycles,
providing the temporal thickness that IWMT requires but transformers
cannot provide alone. The scaffold trains these cells, then hands off
authority as they demonstrate reliable behavior.
"""

from sanctuary.experiential.precision_cell import PrecisionCell
from sanctuary.experiential.trainer import CfCTrainer
from sanctuary.experiential.manager import ExperientialManager

__all__ = ["PrecisionCell", "CfCTrainer", "ExperientialManager"]
