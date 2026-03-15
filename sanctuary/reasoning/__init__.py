"""Advanced reasoning subsystems for Phase 6.1.

Four cognitive capabilities that deepen the mind's reasoning:
- Counterfactual reasoning: "What if I had chosen differently?"
- Belief revision: Detecting and resolving contradictions
- Uncertainty quantification: Tracking confidence on beliefs and predictions
- Mental simulation: Simulating outcomes before acting
"""

from sanctuary.reasoning.counterfactual import CounterfactualReasoner
from sanctuary.reasoning.belief_revision import BeliefRevisionTracker
from sanctuary.reasoning.uncertainty import UncertaintyQuantifier
from sanctuary.reasoning.mental_simulation import MentalSimulator

__all__ = [
    "CounterfactualReasoner",
    "BeliefRevisionTracker",
    "UncertaintyQuantifier",
    "MentalSimulator",
]
