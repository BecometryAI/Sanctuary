"""
Action-Outcome Learning: Tracks what actions actually achieve.

This module implements learning from action outcomes - tracking what actions
actually accomplish versus what was intended, identifying reliable action types,
and predicting likely outcomes of future actions.
"""

from __future__ import annotations

import math
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ActionOutcome:
    """Records what an action actually achieved."""
    action_id: str
    action_type: str
    intended_outcome: str
    actual_outcome: str
    success: bool
    partial_success: float  # 0-1, how much of intended was achieved
    side_effects: List[str]  # Unintended consequences
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionReliability:
    """Reliability metrics for an action type."""
    action_type: str
    success_rate: float
    avg_partial_success: float
    common_side_effects: List[Tuple[str, float]]  # (effect, probability)
    best_contexts: List[Dict[str, Any]]
    worst_contexts: List[Dict[str, Any]]
    unknown: bool = False
    total_executions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomePrediction:
    """Prediction of action outcome."""
    confidence: float
    probability_success: float = 0.5
    prediction: str = "unknown"
    likely_side_effects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionModel:
    """Learned model of what an action does."""
    action_type: str
    success_predictors: Dict[str, float]  # Context features that predict success
    failure_predictors: Dict[str, float]  # Context features that predict failure
    typical_side_effects: List[Tuple[str, float]]  # Side effect and probability
    sample_size: int = 0
    
    def predict(self, context: Dict[str, Any]) -> OutcomePrediction:
        """Predict outcome in context."""
        success_score = sum(
            weight for feature, weight in self.success_predictors.items()
            if context.get(feature)
        )
        failure_score = sum(
            weight for feature, weight in self.failure_predictors.items()
            if context.get(feature)
        )
        
        net_score = success_score - failure_score
        
        # Sigmoid function for probability
        probability = 1 / (1 + math.exp(-net_score)) if net_score != 0 else 0.5
        
        # Confidence based on score magnitude and sample size
        confidence = min(1.0, (abs(net_score) / (abs(net_score) + 1)) * (self.sample_size / 10))
        
        # Filter likely side effects
        likely_side_effects = [
            effect for effect, prob in self.typical_side_effects 
            if prob > 0.3
        ]
        
        return OutcomePrediction(
            confidence=confidence,
            probability_success=probability,
            prediction="likely_success" if probability > 0.6 else "likely_failure" if probability < 0.4 else "uncertain",
            likely_side_effects=likely_side_effects
        )


class ActionOutcomeLearner:
    """Learns what actions actually achieve."""
    
    # Thresholds for outcome comparison
    SUCCESS_OVERLAP_THRESHOLD = 0.5  # At least 50% of intended terms should be in actual
    SIDE_EFFECT_FREQUENCY_THRESHOLD = 0.2  # 20% frequency to be considered "common"
    
    def __init__(self, config: Optional[Dict] = None):
        self.outcomes: List[ActionOutcome] = []
        self.action_models: Dict[str, ActionModel] = {}
        self.config = config or {}
        self.max_outcomes = self.config.get("max_outcomes", 1000)
        self.min_samples_for_model = self.config.get("min_samples_for_model", 5)
        
        logger.info("✅ ActionOutcomeLearner initialized")
    
    def record_outcome(self, action_id: str, action_type: str,
                      intended: str, actual: str, context: Dict[str, Any]):
        """Record the outcome of an action."""
        # Input validation
        if not action_id or not isinstance(action_id, str):
            raise ValueError("action_id must be a non-empty string")
        if not action_type or not isinstance(action_type, str):
            raise ValueError("action_type must be a non-empty string")
        if not isinstance(intended, str):
            raise TypeError("intended must be a string")
        if not isinstance(actual, str):
            raise TypeError("actual must be a string")
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
        
        success = self._compare_outcomes(intended, actual)
        partial = self._compute_partial_success(intended, actual)
        side_effects = self._identify_side_effects(intended, actual, context)
        
        outcome = ActionOutcome(
            action_id=action_id,
            action_type=action_type,
            intended_outcome=intended,
            actual_outcome=actual,
            success=success,
            partial_success=partial,
            side_effects=side_effects,
            timestamp=datetime.now(),
            context=context
        )
        
        self.outcomes.append(outcome)
        
        # Limit memory usage
        if len(self.outcomes) > self.max_outcomes:
            self.outcomes = self.outcomes[-self.max_outcomes:]
        
        self._update_action_model(action_type, outcome)
        
        logger.debug(f"📝 Recorded action outcome: {action_type} "
                    f"(success={success}, partial={partial:.2f})")
    
    def _compare_outcomes(self, intended: str, actual: str) -> bool:
        """Compare intended and actual outcomes."""
        # Simple string comparison - in practice, this would be more sophisticated
        intended_lower = intended.lower()
        actual_lower = actual.lower()
        
        # Check if key terms from intended are in actual
        intended_terms = set(intended_lower.split())
        actual_terms = set(actual_lower.split())
        
        # At least 50% of intended terms should be in actual for success
        if not intended_terms:
            return True
            
        overlap = len(intended_terms & actual_terms)
        return overlap >= len(intended_terms) * self.SUCCESS_OVERLAP_THRESHOLD
    
    def _compute_partial_success(self, intended: str, actual: str) -> float:
        """Compute partial success score (0.0-1.0)."""
        if not intended:
            return 1.0
            
        intended_lower = intended.lower()
        actual_lower = actual.lower()
        
        intended_terms = set(intended_lower.split())
        actual_terms = set(actual_lower.split())
        
        if not intended_terms:
            return 1.0
        
        overlap = len(intended_terms & actual_terms)
        return overlap / len(intended_terms)
    
    def _identify_side_effects(self, intended: str, actual: str, 
                               context: Dict[str, Any]) -> List[str]:
        """Identify unintended consequences."""
        side_effects = []
        
        intended_lower = intended.lower()
        actual_lower = actual.lower()
        
        intended_terms = set(intended_lower.split())
        actual_terms = set(actual_lower.split())
        
        # Terms in actual but not intended are potential side effects
        unexpected_terms = actual_terms - intended_terms
        
        # Filter out common terms
        common_terms = {'the', 'a', 'an', 'is', 'was', 'were', 'be', 'been', 
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would'}
        significant_terms = unexpected_terms - common_terms
        
        if significant_terms:
            side_effects.append(f"unexpected_elements: {', '.join(list(significant_terms)[:5])}")
        
        # Check context for unexpected changes
        if context.get("emotional_change"):
            side_effects.append("emotional_impact")
        
        if context.get("resource_usage_high"):
            side_effects.append("high_resource_usage")
        
        return side_effects
    
    def _update_action_model(self, action_type: str, outcome: ActionOutcome):
        """Update the learned model for an action type."""
        if action_type not in self.action_models:
            self.action_models[action_type] = ActionModel(
                action_type=action_type,
                success_predictors={},
                failure_predictors={},
                typical_side_effects=[],
                sample_size=0
            )
        
        model = self.action_models[action_type]
        model.sample_size += 1
        
        # Update predictors based on outcome
        context_features = self._extract_context_features(outcome.context)
        
        if outcome.success:
            for feature in context_features:
                model.success_predictors[feature] = \
                    model.success_predictors.get(feature, 0) + 0.1
        else:
            for feature in context_features:
                model.failure_predictors[feature] = \
                    model.failure_predictors.get(feature, 0) + 0.1
        
        # Update side effects
        for effect in outcome.side_effects:
            # Find existing or add new
            found = False
            for i, (existing_effect, prob) in enumerate(model.typical_side_effects):
                if existing_effect == effect:
                    # Update probability
                    new_prob = (prob * (model.sample_size - 1) + 1) / model.sample_size
                    model.typical_side_effects[i] = (effect, new_prob)
                    found = True
                    break
            
            if not found:
                model.typical_side_effects.append((effect, 1.0 / model.sample_size))
    
    def _extract_context_features(self, context: Dict[str, Any]) -> List[str]:
        """Extract relevant features from context."""
        features = []
        
        # Boolean features
        for key, value in context.items():
            if isinstance(value, bool) and value:
                features.append(key)
            elif isinstance(value, (int, float)) and value > 0:
                features.append(f"{key}_present")
        
        return features
    
    def get_action_reliability(self, action_type: str) -> ActionReliability:
        """How reliable is this action type?"""
        relevant = [o for o in self.outcomes if o.action_type == action_type]
        
        if not relevant:
            return ActionReliability(
                action_type=action_type,
                success_rate=0.0,
                avg_partial_success=0.0,
                common_side_effects=[],
                best_contexts=[],
                worst_contexts=[],
                unknown=True
            )
        
        success_rate = sum(1 for o in relevant if o.success) / len(relevant)
        avg_partial = sum(o.partial_success for o in relevant) / len(relevant)
        
        # Get common side effects
        side_effect_counts: Dict[str, int] = defaultdict(int)
        for outcome in relevant:
            for effect in outcome.side_effects:
                side_effect_counts[effect] += 1
        
        common_side_effects = [
            (effect, count / len(relevant))
            for effect, count in side_effect_counts.items()
            if count > len(relevant) * self.SIDE_EFFECT_FREQUENCY_THRESHOLD
        ]
        common_side_effects.sort(key=lambda x: -x[1])
        
        # Identify best and worst contexts
        successes = [o for o in relevant if o.success]
        failures = [o for o in relevant if not o.success]
        
        best_contexts = [o.context for o in successes[:3]]
        worst_contexts = [o.context for o in failures[:3]]
        
        return ActionReliability(
            action_type=action_type,
            success_rate=success_rate,
            avg_partial_success=avg_partial,
            common_side_effects=common_side_effects,
            best_contexts=best_contexts,
            worst_contexts=worst_contexts,
            total_executions=len(relevant)
        )
    
    def predict_outcome(self, action_type: str, context: Dict[str, Any]) -> OutcomePrediction:
        """Predict likely outcome of an action in context."""
        if action_type not in self.action_models:
            return OutcomePrediction(
                confidence=0.0,
                prediction="unknown",
                metadata={"reason": "no_model"}
            )
        
        model = self.action_models[action_type]
        
        if model.sample_size < self.min_samples_for_model:
            return OutcomePrediction(
                confidence=0.0,
                prediction="insufficient_data",
                metadata={"sample_size": model.sample_size}
            )
        
        return model.predict(context)
    
    def get_all_reliabilities(self) -> Dict[str, ActionReliability]:
        """Get reliability metrics for all action types."""
        action_types = set(o.action_type for o in self.outcomes)
        return {
            action_type: self.get_action_reliability(action_type)
            for action_type in action_types
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of action learning data."""
        action_types = set(o.action_type for o in self.outcomes)
        reliabilities = {
            at: self.get_action_reliability(at)
            for at in action_types
        }
        
        return {
            "total_outcomes": len(self.outcomes),
            "action_types": len(action_types),
            "action_models": len(self.action_models),
            "average_success_rate": sum(r.success_rate for r in reliabilities.values()) / len(reliabilities) if reliabilities else 0.0,
            "reliabilities": {
                at: {
                    "success_rate": r.success_rate,
                    "executions": r.total_executions,
                    "side_effects": len(r.common_side_effects)
                }
                for at, r in reliabilities.items()
            }
        }
