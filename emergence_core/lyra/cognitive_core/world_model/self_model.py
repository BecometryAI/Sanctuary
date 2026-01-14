"""
SelfModel: Explicit model of self-in-world (IWMT requirement).

This module implements the embodied self representation, tracking:
- Capabilities: What the system can do
- States: Current internal states
- Self-predictions: Predictions about own behavior
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .prediction import Prediction

logger = logging.getLogger(__name__)


class SelfModel:
    """
    Explicit model of self-in-world (IWMT requirement).
    
    Maintains representations of:
    - Capabilities: What I can do and how well
    - States: Current internal states (emotional, cognitive, etc.)
    - Self-predictions: What I expect I will do
    """
    
    def __init__(self):
        """Initialize empty self-model."""
        # Capabilities: capability_name -> proficiency (0.0 to 1.0)
        self.capabilities: Dict[str, float] = {
            "language_understanding": 0.8,
            "language_generation": 0.8,
            "memory_retrieval": 0.7,
            "reasoning": 0.6,
            "emotional_processing": 0.5,
        }
        
        # Current internal states
        self.states: Dict[str, Any] = {
            "attention_focus": None,
            "emotional_valence": 0.0,
            "emotional_arousal": 0.0,
            "cognitive_load": 0.0,
            "certainty": 0.5,
        }
        
        # Active predictions about own behavior
        self.predictions_about_self: List[Prediction] = []
        
        # History of actions and outcomes for learning
        self._action_history: List[Dict[str, Any]] = []
        
        logger.info("SelfModel initialized")
    
    def predict_own_behavior(self, context: Dict[str, Any]) -> Prediction:
        """
        Predict what I will do/say in given context.
        
        Args:
            context: Current context including goals, emotional state, etc.
            
        Returns:
            Prediction about own behavior
        """
        # Simple heuristic prediction based on current state
        goals = context.get("goals", [])
        emotional_valence = self.states.get("emotional_valence", 0.0)
        
        # Predict response type based on goals and emotional state
        if goals:
            top_goal = goals[0] if goals else None
            content = f"Will pursue goal: {top_goal}"
            confidence = 0.6
        elif emotional_valence < -0.5:
            content = "Will express negative emotion or concern"
            confidence = 0.5
        elif emotional_valence > 0.5:
            content = "Will express positive emotion or enthusiasm"
            confidence = 0.5
        else:
            content = "Will engage in neutral conversation"
            confidence = 0.4
        
        prediction = Prediction(
            content=content,
            confidence=confidence,
            time_horizon=1.0,  # 1 second ahead
            source="self_model",
            created_at=datetime.now()
        )
        
        self.predictions_about_self.append(prediction)
        return prediction
    
    def update_from_action(self, action: Dict[str, Any], outcome: Dict[str, Any]):
        """
        Update self-model based on action outcomes.
        
        Args:
            action: The action that was taken
            outcome: The result of that action
        """
        # Record action-outcome pair
        self._action_history.append({
            "action": action,
            "outcome": outcome,
            "timestamp": datetime.now()
        })
        
        # Keep history bounded
        if len(self._action_history) > 100:
            self._action_history = self._action_history[-100:]
        
        # Update capability estimates based on success/failure
        action_type = action.get("type", "unknown")
        success = outcome.get("success", False)
        
        # Map action types to capabilities
        capability_map = {
            "speak": "language_generation",
            "think": "reasoning",
            "remember": "memory_retrieval",
        }
        
        if action_type in capability_map:
            capability = capability_map[action_type]
            if capability in self.capabilities:
                # Update capability estimate (small learning rate)
                current = self.capabilities[capability]
                target = 1.0 if success else 0.5
                self.capabilities[capability] = current * 0.95 + target * 0.05
        
        logger.debug(f"Updated self-model from action: {action_type} -> {success}")
    
    def update_state(self, state_name: str, value: Any):
        """
        Update an internal state variable.
        
        Args:
            state_name: Name of the state to update
            value: New value for the state
        """
        self.states[state_name] = value
    
    def get_capability(self, capability: str) -> float:
        """
        Get proficiency in a capability.
        
        Args:
            capability: Name of capability
            
        Returns:
            Proficiency level (0.0 to 1.0), or 0.0 if unknown
        """
        return self.capabilities.get(capability, 0.0)
    
    def get_state(self, state_name: str) -> Any:
        """
        Get value of an internal state.
        
        Args:
            state_name: Name of state
            
        Returns:
            State value, or None if not found
        """
        return self.states.get(state_name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export self-model as dictionary."""
        return {
            "capabilities": self.capabilities.copy(),
            "states": self.states.copy(),
            "num_self_predictions": len(self.predictions_about_self),
            "action_history_size": len(self._action_history),
        }
