"""
Idle Detection Module

Detects when the cognitive system has low activity and can run
memory consolidation processes without interfering with active cognition.

Author: Sanctuary Team
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class IdleDetector:
    """
    Detects system idle periods for memory consolidation.
    
    Monitors cognitive activity and determines when the system
    is idle enough to perform background memory maintenance.
    
    Attributes:
        idle_threshold_seconds: Seconds of inactivity before considered idle
        activity_decay: Decay rate for activity level (0.0-1.0)
        last_activity: Timestamp of last recorded activity
        activity_level: Current activity level (0.0-1.0, 1.0 = active)
    """
    
    def __init__(
        self,
        idle_threshold_seconds: float = 30.0,
        activity_decay: float = 0.9
    ):
        """
        Initialize idle detector.
        
        Args:
            idle_threshold_seconds: Seconds of inactivity before idle (> 0)
            activity_decay: Activity decay rate per check (0.0-1.0)
        """
        if idle_threshold_seconds <= 0:
            raise ValueError(f"idle_threshold_seconds must be > 0, got {idle_threshold_seconds}")
        if not (0.0 <= activity_decay <= 1.0):
            raise ValueError(f"activity_decay must be in [0.0, 1.0], got {activity_decay}")
        
        self.idle_threshold = idle_threshold_seconds
        self.activity_decay = activity_decay
        self.last_activity = datetime.now()
        self.activity_level = 1.0
        
        logger.info(
            f"IdleDetector initialized (threshold: {idle_threshold_seconds}s, "
            f"decay: {activity_decay})"
        )
    
    def record_activity(self) -> None:
        """
        Record that cognitive activity has occurred.
        
        Called when:
        - Memory retrieval happens
        - New memories are stored
        - Cognitive processes execute
        - User interaction occurs
        """
        self.last_activity = datetime.now()
        self.activity_level = 1.0
        logger.debug("Activity recorded, system active")
    
    def is_idle(self) -> bool:
        """
        Check if system is idle enough for consolidation.
        
        Returns:
            True if system has been idle for threshold duration
        """
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        is_idle = elapsed > self.idle_threshold
        
        if is_idle and self.activity_level > 0.1:
            # Decay activity level over time
            self.activity_level *= self.activity_decay
        
        return is_idle
    
    def get_consolidation_budget(self) -> float:
        """
        Calculate how much consolidation work can be done.
        
        Returns a budget from 0.0 to 1.0 indicating how much
        consolidation processing is appropriate:
        - 0.0: System is active, no consolidation
        - 0.0-0.2: Just became idle, minimal consolidation
        - 0.2-0.5: Moderate idle time, standard consolidation
        - 0.5-1.0: Extended idle time, full consolidation
        
        Returns:
            Consolidation budget (0.0-1.0)
        """
        if not self.is_idle():
            return 0.0
        
        # Calculate how long we've been idle beyond threshold
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        idle_duration = elapsed - self.idle_threshold
        
        # Budget scales with idle time, capped at 1.0
        # Full budget after 60 seconds of idle time
        budget = min(1.0, idle_duration / 60.0)
        
        logger.debug(f"Consolidation budget: {budget:.2f} (idle: {idle_duration:.1f}s)")
        return budget
    
    def get_idle_duration(self) -> float:
        """
        Get duration of current idle period.
        
        Returns:
            Seconds since last activity, or 0.0 if not idle
        """
        if not self.is_idle():
            return 0.0
        
        elapsed = (datetime.now() - self.last_activity).total_seconds()
        return elapsed - self.idle_threshold
    
    def reset(self) -> None:
        """Reset idle detector to active state."""
        self.last_activity = datetime.now()
        self.activity_level = 1.0
        logger.debug("IdleDetector reset to active state")
