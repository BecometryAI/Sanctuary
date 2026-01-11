"""
Action-Outcome Learning: Track what actions actually achieve.

This module implements learning from action outcomes, tracking intended
vs actual results, building predictive models, and identifying reliable
action patterns.
"""

from __future__ import annotations

import math
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ActionOutcome:
    """
    Records what an action actually achieved.
    
    Compares intended outcomes with actual results to enable learning
    about action reliability and side effects.
    
    Attributes:
        action_id: Unique identifier for the action
        action_type: Category of action
        intended_outcome: What the action was meant to achieve
        actual_outcome: What actually happened
        success: Whether intended outcome was achieved
        partial_success: How much of intended was achieved (0.0-1.0)
        side_effects: Unintended consequences
        timestamp: When the outcome was observed
        context: Contextual information at action time
    """
    action_id: str
    action_type: str
    intended_outcome: str
    actual_outcome: str
    success: bool
    partial_success: float
    side_effects: List[str]
    timestamp: datetime
    context: Dict[str, Any]


@dataclass
class ActionReliability:
    """
    Summary of how reliable an action type is.
    
    Attributes:
        action_type: Type of action
        success_rate: Fraction of attempts that fully succeeded
        avg_partial_success: Average partial success across attempts
        common_side_effects: Frequently occurring side effects
        best_contexts: Contexts where action performs best
        worst_contexts: Contexts where action performs worst
        unknown: True if insufficient data for assessment
    """
    action_type: str
    success_rate: float = 0.0
    avg_partial_success: float = 0.0
    common_side_effects: List[Tuple[str, float]] = field(default_factory=list)
    best_contexts: List[Dict[str, Any]] = field(default_factory=list)
    worst_contexts: List[Dict[str, Any]] = field(default_factory=list)
    unknown: bool = False


@dataclass
class OutcomePrediction:
    """
    Predicted outcome of an action in context.
    
    Attributes:
        confidence: How confident the prediction is (0.0-1.0)
        probability_success: Probability of full success (0.0-1.0)
        prediction: Text description of likely outcome
        likely_side_effects: Side effects that may occur
    """
    confidence: float
    probability_success: float = 0.5
    prediction: str = "unknown"
    likely_side_effects: List[str] = field(default_factory=list)


@dataclass
class ActionModel:
    """
    Learned model of what an action does.
    
    Attributes:
        action_type: Type of action this model represents
        success_predictors: Context features that predict success (feature -> weight)
        failure_predictors: Context features that predict failure (feature -> weight)
        typical_side_effects: Common side effects (effect, probability)
    """
    action_type: str
    success_predictors: Dict[str, float] = field(default_factory=dict)
    failure_predictors: Dict[str, float] = field(default_factory=dict)
    typical_side_effects: List[Tuple[str, float]] = field(default_factory=list)
    
    def predict(self, context: Dict[str, Any]) -> OutcomePrediction:
        """
        Predict outcome in given context.
        
        Args:
            context: Contextual information
            
        Returns:
            OutcomePrediction with likelihood assessment
        """
        # Calculate success and failure scores based on context features
        success_score = sum(
            weight for feature, weight in self.success_predictors.items()
            if context.get(feature)
        )
        
        failure_score = sum(
            weight for feature, weight in self.failure_predictors.items()
            if context.get(feature)
        )
        
        # Net score (positive = likely success, negative = likely failure)
        net_score = success_score - failure_score
        
        # Convert to probability using sigmoid
        probability = 1 / (1 + math.exp(-net_score))
        
        # Confidence based on how extreme the net score is
        confidence = abs(net_score) / (abs(net_score) + 1)
        
        # Get likely side effects (those with >30% probability)
        likely_side_effects = [
            effect for effect, prob in self.typical_side_effects
            if prob > 0.3
        ]
        
        return OutcomePrediction(
            confidence=confidence,
            probability_success=probability,
            prediction=f"{'likely success' if probability > 0.5 else 'likely failure'} "
                      f"(p={probability:.2f})",
            likely_side_effects=likely_side_effects
        )


class ActionOutcomeLearner:
    """
    Learns what actions actually achieve.
    
    Tracks action outcomes, builds predictive models, and provides
    reliability assessments for different action types.
    """
    
    # Class constants
    MAX_OUTCOMES = 5000
    SIMILARITY_THRESHOLD = 0.6
    
    def __init__(self, min_outcomes_for_model: int = 5):
        """
        Initialize action-outcome learner.
        
        Args:
            min_outcomes_for_model: Min outcomes for model building
        """
        if min_outcomes_for_model < 3:
            raise ValueError("min_outcomes_for_model must be >= 3")
        
        self.outcomes: List[ActionOutcome] = []
        self.action_models: Dict[str, ActionModel] = {}
        self.min_outcomes = min_outcomes_for_model
        
        logger.info("✅ ActionOutcomeLearner initialized")
    
    def record_outcome(
        self,
        action_id: str,
        action_type: str,
        intended: str,
        actual: str,
        context: Dict[str, Any]
    ):
        """Record action outcome with validation."""
        if not action_id or not action_type:
            raise ValueError("action_id and action_type required")
        if not intended or not actual:
            raise ValueError("intended and actual outcomes required")
        
        success = self._compare_outcomes(intended, actual)
        partial = self._compute_partial_success(intended, actual)
        side_effects = self._identify_side_effects(intended, actual, context)
        
        outcome = ActionOutcome(
            action_id=action_id,
            action_type=action_type,
            intended_outcome=intended,
            actual_outcome=actual,
            success=success,
            partial_success=max(0.0, min(1.0, partial)),  # Clamp to [0,1]
            side_effects=side_effects,
            timestamp=datetime.now(),
            context=context or {}
        )
        
        self.outcomes.append(outcome)
        self._update_action_model(action_type, outcome)
        
        logger.debug(f"Recorded {action_type}: success={success}, partial={partial:.2f}")
        
        # Prune old outcomes
        if len(self.outcomes) > self.MAX_OUTCOMES:
            self.outcomes = self.outcomes[-self.MAX_OUTCOMES:]
    
    def _compare_outcomes(self, intended: str, actual: str) -> bool:
        """
        Compare intended and actual outcomes.
        
        Args:
            intended: Intended outcome
            actual: Actual outcome
            
        Returns:
            True if outcomes match (success), False otherwise
        """
        # Simple keyword-based comparison
        # In a real implementation, this would use semantic similarity
        intended_lower = intended.lower()
        actual_lower = actual.lower()
        
        # Check for key terms match
        intended_words = set(intended_lower.split())
        actual_words = set(actual_lower.split())
        
        # Success if significant overlap
        overlap = len(intended_words & actual_words)
        union = len(intended_words | actual_words)
        
        if union == 0:
            return False
        
        similarity = overlap / union
        return similarity > self.SIMILARITY_THRESHOLD
    
    def _compute_partial_success(self, intended: str, actual: str) -> float:
        """
        Compute how much of intended outcome was achieved.
        
        Args:
            intended: Intended outcome
            actual: Actual outcome
            
        Returns:
            Partial success score (0.0-1.0)
        """
        # Simple keyword overlap measure
        intended_words = set(intended.lower().split())
        actual_words = set(actual.lower().split())
        
        if not intended_words:
            return 0.5
        
        overlap = len(intended_words & actual_words)
        return min(1.0, overlap / len(intended_words))
    
    def _identify_side_effects(
        self,
        intended: str,
        actual: str,
        context: Dict[str, Any]
    ) -> List[str]:
        """
        Identify unintended consequences.
        
        Args:
            intended: Intended outcome
            actual: Actual outcome
            context: Contextual information
            
        Returns:
            List of identified side effects
        """
        side_effects = []
        
        # Look for unexpected keywords in actual outcome
        intended_words = set(intended.lower().split())
        actual_words = set(actual.lower().split())
        
        unexpected = actual_words - intended_words
        
        # Flag certain unexpected terms as side effects
        negative_terms = {'error', 'fail', 'wrong', 'unexpected', 'problem', 'issue'}
        if unexpected & negative_terms:
            side_effects.append("unexpected_error")
        
        # Check context for side effects
        if context.get("goal_interference"):
            side_effects.append("interfered_with_other_goals")
        
        if context.get("resource_exhaustion"):
            side_effects.append("exhausted_resources")
        
        return side_effects
    
    def _update_action_model(self, action_type: str, outcome: ActionOutcome):
        """
        Update the learned model for an action type.
        
        Args:
            action_type: Type of action
            outcome: New outcome to learn from
        """
        # Get or create model
        if action_type not in self.action_models:
            self.action_models[action_type] = ActionModel(action_type=action_type)
        
        model = self.action_models[action_type]
        
        # Get all outcomes for this action type
        action_outcomes = [o for o in self.outcomes if o.action_type == action_type]
        
        if len(action_outcomes) < self.min_outcomes:
            return  # Not enough data yet
        
        # Update success predictors
        successes = [o for o in action_outcomes if o.success]
        failures = [o for o in action_outcomes if not o.success]
        
        # Extract common context features
        if successes:
            success_features = self._extract_common_features(
                [s.context for s in successes]
            )
            model.success_predictors = success_features
        
        if failures:
            failure_features = self._extract_common_features(
                [f.context for f in failures]
            )
            model.failure_predictors = failure_features
        
        # Update side effect probabilities
        side_effect_counts: Dict[str, int] = defaultdict(int)
        for outcome in action_outcomes:
            for effect in outcome.side_effects:
                side_effect_counts[effect] += 1
        
        model.typical_side_effects = [
            (effect, count / len(action_outcomes))
            for effect, count in side_effect_counts.items()
        ]
    
    def _extract_common_features(
        self,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Extract features common across contexts.
        
        Args:
            contexts: List of context dictionaries
            
        Returns:
            Feature weights (feature -> weight)
        """
        if not contexts:
            return {}
        
        # Count feature occurrences
        feature_counts: Dict[str, int] = defaultdict(int)
        for context in contexts:
            for key, value in context.items():
                # Only consider boolean features or presence
                if isinstance(value, bool) and value:
                    feature_counts[key] += 1
                elif value and not isinstance(value, (dict, list)):
                    feature_counts[key] += 1
        
        # Convert to weights (proportion of contexts with feature)
        return {
            feature: count / len(contexts)
            for feature, count in feature_counts.items()
            if count / len(contexts) > 0.5  # At least 50% occurrence
        }
    
    def get_action_reliability(self, action_type: str) -> ActionReliability:
        """
        Get reliability assessment for an action type.
        
        Args:
            action_type: Type of action
            
        Returns:
            ActionReliability assessment
        """
        relevant = [o for o in self.outcomes if o.action_type == action_type]
        
        if not relevant:
            return ActionReliability(action_type=action_type, unknown=True)
        
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
            if count / len(relevant) > 0.2  # At least 20% occurrence
        ]
        common_side_effects.sort(key=lambda x: -x[1])
        
        # Identify best and worst contexts
        best_contexts = self._identify_best_contexts(relevant)
        worst_contexts = self._identify_worst_contexts(relevant)
        
        return ActionReliability(
            action_type=action_type,
            success_rate=success_rate,
            avg_partial_success=avg_partial,
            common_side_effects=common_side_effects,
            best_contexts=best_contexts,
            worst_contexts=worst_contexts
        )
    
    def _identify_best_contexts(
        self,
        outcomes: List[ActionOutcome]
    ) -> List[Dict[str, Any]]:
        """
        Identify contexts where action performs best.
        
        Args:
            outcomes: List of outcomes for an action type
            
        Returns:
            List of contexts associated with success
        """
        successes = [o for o in outcomes if o.success or o.partial_success > 0.8]
        
        if len(successes) < 3:
            return []
        
        # Return contexts of top successes (by partial success)
        successes.sort(key=lambda o: -o.partial_success)
        return [o.context for o in successes[:3]]
    
    def _identify_worst_contexts(
        self,
        outcomes: List[ActionOutcome]
    ) -> List[Dict[str, Any]]:
        """
        Identify contexts where action performs worst.
        
        Args:
            outcomes: List of outcomes for an action type
            
        Returns:
            List of contexts associated with failure
        """
        failures = [o for o in outcomes if not o.success and o.partial_success < 0.3]
        
        if len(failures) < 3:
            return []
        
        # Return contexts of worst failures (by partial success)
        failures.sort(key=lambda o: o.partial_success)
        return [o.context for o in failures[:3]]
    
    def predict_outcome(
        self,
        action_type: str,
        context: Dict[str, Any]
    ) -> OutcomePrediction:
        """
        Predict likely outcome of an action in context.
        
        Args:
            action_type: Type of action
            context: Contextual information
            
        Returns:
            OutcomePrediction with likelihood assessment
        """
        if action_type not in self.action_models:
            return OutcomePrediction(
                confidence=0.0,
                prediction="unknown - no prior data"
            )
        
        model = self.action_models[action_type]
        return model.predict(context)
    
    def get_all_action_types(self) -> List[str]:
        """
        Get list of all action types with recorded outcomes.
        
        Returns:
            List of action type names
        """
        return list(set(o.action_type for o in self.outcomes))
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of action learning data.
        
        Returns:
            Dictionary with overall statistics
        """
        action_types = self.get_all_action_types()
        
        return {
            "total_outcomes": len(self.outcomes),
            "action_types": len(action_types),
            "models_built": len(self.action_models),
            "overall_success_rate": (
                sum(1 for o in self.outcomes if o.success) / len(self.outcomes)
                if self.outcomes else 0.0
            ),
            "reliability_by_action": {
                action_type: self.get_action_reliability(action_type).success_rate
                for action_type in action_types
            }
        }
