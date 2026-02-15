"""
Behavior Logger: Track actions and behavioral patterns.

This module provides logging of actions and decisions to support
identity computation from behavioral patterns.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from collections import deque, defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


class BehaviorLogger:
    """
    Logs actions and behavioral patterns for identity computation.
    
    Tracks what actions the system takes, when, and why, enabling
    the identity system to infer values and tendencies from actual
    behavior rather than declared intentions.
    
    Attributes:
        action_history: Recent actions taken
        max_history: Maximum actions to keep in memory
        config: Configuration dictionary
    """
    
    def __init__(self, max_history: int = 500, config: Optional[Dict] = None):
        """
        Initialize behavior logger.
        
        Args:
            max_history: Maximum actions to keep in history
            config: Optional configuration dictionary
        
        Raises:
            ValueError: If max_history is less than 1
        """
        if max_history < 1:
            raise ValueError("max_history must be at least 1")
        
        self.action_history: deque = deque(maxlen=max_history)
        self.max_history = max_history
        self.config = config or {}
        
        logger.debug(f"BehaviorLogger initialized (max_history={max_history})")
    
    def log_action(self, action: Any) -> None:
        """
        Log an action that was taken.
        
        Args:
            action: Action object or dictionary
        """
        # Convert action to dict if needed
        try:
            if action is None:
                action_dict = {'action': 'None'}
            elif hasattr(action, 'model_dump'):
                action_dict = action.model_dump()
            elif hasattr(action, '__dict__'):
                action_dict = action.__dict__.copy()
            elif isinstance(action, dict):
                action_dict = dict(action)
            else:
                action_dict = {'action': str(action)}
        except Exception as e:
            logger.debug(f"Could not convert action to dict: {e}")
            action_dict = {'action': str(action)}
        
        # Add timestamp
        action_dict['logged_at'] = datetime.now().isoformat()
        
        # Add to history
        self.action_history.append(action_dict)
        
        logger.debug(f"Action logged: {action_dict.get('type', 'unknown')}")
    
    def get_action_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent action history.
        
        Args:
            limit: Maximum number of actions to return (None for all)
            
        Returns:
            List of action dictionaries
        """
        if limit is None:
            return list(self.action_history)
        return list(self.action_history)[-limit:]
    
    def analyze_tendencies(self) -> Dict[str, float]:
        """
        Analyze behavioral tendencies from action history.
        
        Returns:
            Dictionary mapping tendency types to scores
        """
        if not self.action_history:
            return {}
        
        tendencies = {}
        
        # Action type frequencies
        action_types = defaultdict(int)
        total_actions = len(self.action_history)
        
        for action in self.action_history:
            action_type = action.get('type', 'unknown')
            action_types[action_type] += 1
        
        # Convert to tendencies (normalized frequencies)
        for action_type, count in action_types.items():
            tendencies[f"tendency_{action_type}"] = count / total_actions
        
        # Analyze priority patterns
        priorities = [a.get('priority', 0.5) for a in self.action_history]
        if priorities:
            avg_priority = sum(priorities) / len(priorities)
            tendencies['average_urgency'] = avg_priority
        
        # Analyze reasoning patterns
        reasons = [a.get('reason', '') for a in self.action_history if a.get('reason')]
        if reasons:
            # Count reasoning patterns
            proactive_count = sum(1 for r in reasons if 'proactive' in r.lower() or 'initiate' in r.lower())
            reactive_count = sum(1 for r in reasons if 'respond' in r.lower() or 'react' in r.lower())
            
            if proactive_count + reactive_count > 0:
                tendencies['proactivity'] = proactive_count / (proactive_count + reactive_count)
        
        # Analyze action complexity
        complex_actions = ['tool_call', 'introspect', 'retrieve_memory']
        complex_count = sum(1 for a in self.action_history if a.get('type') in complex_actions)
        tendencies['complexity_preference'] = complex_count / total_actions if total_actions > 0 else 0.0
        
        logger.debug(f"Analyzed {len(tendencies)} behavioral tendencies")
        return tendencies
    
    def get_tradeoff_decisions(self, min_priority: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get actions that represented tradeoff decisions (high priority).
        
        Args:
            min_priority: Minimum priority to consider a tradeoff
            
        Returns:
            List of tradeoff decision dictionaries
        """
        tradeoffs = [
            action for action in self.action_history
            if action.get('priority', 0.0) >= min_priority
        ]
        
        logger.debug(f"Found {len(tradeoffs)} tradeoff decisions")
        return tradeoffs
    
    def clear_history(self) -> None:
        """Clear all action history."""
        self.action_history.clear()
        logger.info("Action history cleared")
