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
        """
        Emotions decay toward baseline over time.
        
        Uses exponential decay: value(t) = baseline + (current - baseline) * decay^t
        
        Args:
            emotions: Current emotional state (valence, arousal, dominance)
            elapsed: Time elapsed
            
        Returns:
            Updated emotional state
        """
        hours = elapsed.total_seconds() / 3600
        decay_factor = self.emotion_decay_rate ** hours
        
        # Get baseline values
        baseline_valence = self.emotion_baseline.get("valence", 0.0)
        baseline_arousal = self.emotion_baseline.get("arousal", 0.0)
        baseline_dominance = self.emotion_baseline.get("dominance", 0.5)
        
        # Apply decay toward baseline
        decayed = {}
        
        if "valence" in emotions:
            decayed["valence"] = baseline_valence + (
                emotions["valence"] - baseline_valence
            ) * decay_factor
        
        if "arousal" in emotions:
            decayed["arousal"] = baseline_arousal + (
                emotions["arousal"] - baseline_arousal
            ) * decay_factor
        
        if "dominance" in emotions:
            decayed["dominance"] = baseline_dominance + (
                emotions["dominance"] - baseline_dominance
            ) * decay_factor
        
        # Preserve other emotional attributes
        for key, value in emotions.items():
            if key not in decayed:
                decayed[key] = value
        
        logger.debug(f"😌 Emotions decayed toward baseline (factor: {decay_factor:.3f})")
        
        return decayed
    
    def _fade_context(
        self, working_memory: List[Any], elapsed: timedelta
    ) -> List[Any]:
        """
        Context in working memory fades over time.
        
        Items become less salient as time passes. Very old items may be removed.
        
        Args:
            working_memory: List of items in working memory
            elapsed: Time elapsed
            
        Returns:
            Updated working memory with faded context
        """
        hours = elapsed.total_seconds() / 3600
        fade_factor = self.context_fade_rate ** hours
        
        faded_memory = []
        
        for item in working_memory:
            # If item has salience/strength, reduce it
            if isinstance(item, dict):
                faded_item = item.copy()
                if "salience" in faded_item:
                    faded_item["salience"] *= fade_factor
                    # Only keep items above threshold
                    if faded_item["salience"] > 0.1:
                        faded_memory.append(faded_item)
                else:
                    faded_memory.append(faded_item)
            else:
                faded_memory.append(item)
        
        removed = len(working_memory) - len(faded_memory)
        if removed > 0:
            logger.debug(f"🌫️  Context faded: {removed} items removed from working memory")
        
        return faded_memory
    
    def _update_urgencies(self, goals: List[Any], elapsed: timedelta) -> List[Any]:
        """
        Goal urgency changes with time passage.
        
        Goals approaching deadlines become more urgent.
        Goals past deadlines are marked as expired.
        
        Args:
            goals: List of current goals
            elapsed: Time elapsed
            
        Returns:
            Updated goals with modified urgencies
        """
        updated_goals = []
        
        for goal in goals:
            # Handle both dict and object representations
            if isinstance(goal, dict):
                updated_goal = goal.copy()
                deadline = updated_goal.get("deadline")
                urgency = updated_goal.get("urgency", 0.5)
                status = updated_goal.get("status", "active")
            else:
                # Assume it's a Goal object
                updated_goal = goal
                deadline = getattr(goal, "deadline", None)
                urgency = getattr(goal, "urgency", 0.5)
                status = getattr(goal, "status", "active")
            
            if deadline:
                # Convert deadline to datetime if it's a string
                if isinstance(deadline, str):
                    try:
                        deadline = datetime.fromisoformat(deadline)
                    except (ValueError, AttributeError):
                        deadline = None
                
                if deadline:
                    now = datetime.now()
                    remaining = deadline - now
                    
                    # Update urgency based on time remaining
                    if remaining < timedelta(hours=24):
                        # Less than 24 hours - increase urgency
                        new_urgency = min(1.0, urgency + 0.2)
                        if isinstance(updated_goal, dict):
                            updated_goal["urgency"] = new_urgency
                        else:
                            updated_goal.urgency = new_urgency
                    elif remaining < timedelta(0):
                        # Past deadline - mark as expired
                        if isinstance(updated_goal, dict):
                            updated_goal["urgency"] = 0.0
                            updated_goal["status"] = "expired"
                        else:
                            updated_goal.urgency = 0.0
                            updated_goal.status = "expired"
                        
                        logger.debug(f"⏰ Goal expired: {updated_goal.get('description', 'unknown') if isinstance(updated_goal, dict) else getattr(updated_goal, 'description', 'unknown')}")
            
            updated_goals.append(updated_goal)
        
        return updated_goals
    
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
