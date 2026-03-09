"""
Time Passage Effects: How time affects cognitive state.

This module implements the effects of time passage on emotional state, working memory,
goal urgency, and other cognitive components.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..affect import EmotionalState
    from ..workspace import Goal

logger = logging.getLogger(__name__)


class TimePassageEffects:
    """
    Applies effects of time passage to cognitive state.
    
    Time affects:
    - Emotional decay toward baseline
    - Context fading in working memory
    - Goal urgency updates
    - Memory consolidation triggers
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize time passage effects system.
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        
        # Emotional decay parameters
        self.emotion_decay_rate = self.config.get("emotion_decay_rate", 0.9)  # per hour
        self.emotion_baseline = self.config.get("emotion_baseline", {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.5
        })
        
        # Context fading parameters
        self.context_fade_rate = self.config.get("context_fade_rate", 0.85)  # per hour
        
        # Consolidation threshold
        self.consolidation_threshold = self.config.get(
            "consolidation_threshold_hours", 1.0
        )
        
        logger.info("✅ TimePassageEffects initialized")
    
    def apply(self, elapsed: timedelta, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply effects of time passage to cognitive state.
        
        Args:
            elapsed: Time elapsed since last update
            state: Current cognitive state dict containing emotions, goals, etc.
            
        Returns:
            Updated cognitive state
        """
        # Apply emotional decay
        if "emotions" in state:
            state["emotions"] = self._decay_emotions(state["emotions"], elapsed)
        
        # Apply context fading
        if "working_memory" in state:
            state["working_memory"] = self._fade_context(
                state["working_memory"], elapsed
            )
        
        # Update goal urgencies
        if "goals" in state:
            state["goals"] = self._update_urgencies(state["goals"], elapsed)
        
        # Check if consolidation should be triggered
        hours = elapsed.total_seconds() / 3600
        if hours > self.consolidation_threshold:
            state["consolidation_needed"] = True
        
        return state
    
    def _decay_emotions(
        self, emotions: Dict[str, float], elapsed: timedelta
    ) -> Dict[str, float]:
        """Apply exponential decay to emotions toward baseline."""
        hours = elapsed.total_seconds() / 3600
        decay_factor = self.emotion_decay_rate ** hours
        
        decayed = {}
        for key in ["valence", "arousal", "dominance"]:
            if key in emotions:
                baseline = self.emotion_baseline.get(key, 0.5 if key == "dominance" else 0.0)
                decayed[key] = baseline + (emotions[key] - baseline) * decay_factor
        
        # Preserve other emotional attributes
        decayed.update({k: v for k, v in emotions.items() if k not in decayed})
        
        return decayed
    
    def _fade_context(
        self, working_memory: List[Any], elapsed: timedelta
    ) -> List[Any]:
        """Fade context in working memory based on time passage."""
        hours = elapsed.total_seconds() / 3600
        fade_factor = self.context_fade_rate ** hours
        
        faded_memory = []
        for item in working_memory:
            if isinstance(item, dict) and "salience" in item:
                faded_item = item.copy()
                faded_item["salience"] *= fade_factor
                if faded_item["salience"] > 0.1:  # Threshold
                    faded_memory.append(faded_item)
            else:
                faded_memory.append(item)
        
        return faded_memory
    
    def _update_urgencies(self, goals: List[Any], elapsed: timedelta) -> List[Any]:
        """Update goal urgency based on approaching deadlines. Modifies goals in-place."""
        now = datetime.now()
        
        for goal in goals:
            deadline = self._get_goal_deadline(goal)
            if not deadline:
                continue
            
            remaining = deadline - now
            
            # Check in order: expired first, then approaching
            if remaining < timedelta(0):
                # Past deadline - mark as expired
                self._set_goal_urgency(goal, 0.0)
                self._set_goal_status(goal, "expired")
            elif remaining < timedelta(hours=24):
                # Less than 24 hours - increase urgency
                self._set_goal_urgency(goal, min(1.0, self._get_goal_urgency(goal) + 0.2))
        
        return goals  # Return modified goals for clarity
    
    @staticmethod
    def _get_goal_deadline(goal: Any) -> Optional[datetime]:
        """Extract deadline from goal (dict or object)."""
        deadline = goal.get("deadline") if isinstance(goal, dict) else getattr(goal, "deadline", None)
        if isinstance(deadline, str):
            try:
                return datetime.fromisoformat(deadline)
            except (ValueError, AttributeError):
                return None
        return deadline
    
    @staticmethod
    def _get_goal_urgency(goal: Any) -> float:
        """Get urgency from goal."""
        return goal.get("urgency", 0.5) if isinstance(goal, dict) else getattr(goal, "urgency", 0.5)
    
    @staticmethod
    def _set_goal_urgency(goal: Any, urgency: float) -> None:
        """Set urgency on goal."""
        if isinstance(goal, dict):
            goal["urgency"] = urgency
        else:
            goal.urgency = urgency
    
    @staticmethod
    def _set_goal_status(goal: Any, status: str) -> None:
        """Set status on goal."""
        if isinstance(goal, dict):
            goal["status"] = status
        else:
            goal.status = status
    
    def trigger_consolidation(self, elapsed: timedelta) -> bool:
        """
        Check if memory consolidation should be triggered.
        
        Args:
            elapsed: Time elapsed since last consolidation
            
        Returns:
            True if consolidation should be triggered
        """
        hours = elapsed.total_seconds() / 3600
        should_consolidate = hours > self.consolidation_threshold
        
        if should_consolidate:
            logger.info(f"💭 Memory consolidation triggered after {hours:.1f} hours")
        
        return should_consolidate
