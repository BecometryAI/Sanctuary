"""
Deferred Communication Queue - Queue communications for better timing.

This module implements a queue for communications that should be deferred
rather than immediately spoken or silenced. It supports:
- Multiple deferral reasons (bad timing, waiting for response, topic change, etc.)
- Time-based and condition-based release
- Expiration of old items
- Priority ordering for ready items
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DeferralReason(Enum):
    """Reasons for deferring communication."""
    BAD_TIMING = "bad_timing"                    # Just spoke, need spacing
    WAIT_FOR_RESPONSE = "wait_for_response"      # Asked question, waiting for answer
    TOPIC_CHANGE = "topic_change"                # Save for when topic returns
    PROCESSING = "processing"                    # Still thinking, will share when ready
    COURTESY = "courtesy"                        # Let them finish their thought
    CUSTOM = "custom"                            # Custom reason specified in description


@dataclass
class DeferredCommunication:
    """
    Represents a communication deferred for later.
    
    Attributes:
        urge: The CommunicationUrge being deferred
        reason: Why this communication is being deferred
        deferred_at: When this was deferred
        release_condition: Human-readable description of when to release
        release_after: Time-based release condition (release after this datetime)
        priority: Priority for ordering when multiple items are ready (0.0 to 1.0)
        max_age_seconds: Maximum age before expiration (default: 300 = 5 minutes)
        attempts: Number of times this has been reconsidered
    """
    urge: Any  # CommunicationUrge instance
    reason: DeferralReason
    deferred_at: datetime = field(default_factory=datetime.now)
    release_condition: str = ""
    release_after: Optional[datetime] = None
    priority: float = 0.5
    max_age_seconds: float = 300.0
    attempts: int = 0
    
    def is_ready(self) -> bool:
        """
        Check if ready to be released.
        
        Currently only checks time-based conditions.
        Future: Could check topic detection, input received, etc.
        
        Returns:
            True if release conditions are met, False otherwise
        """
        if self.is_expired():
            return False
        
        # Time-based release
        if self.release_after is not None:
            return datetime.now() >= self.release_after
        
        # No explicit release condition means ready immediately
        # (should be checked each cycle)
        return True
    
    def is_expired(self) -> bool:
        """
        Check if too old and should be discarded.
        
        Returns:
            True if age exceeds max_age_seconds, False otherwise
        """
        age_seconds = (datetime.now() - self.deferred_at).total_seconds()
        return age_seconds > self.max_age_seconds
    
    def increment_attempts(self) -> None:
        """Increment the reconsideration attempt counter."""
        self.attempts += 1
    
    def get_age_seconds(self) -> float:
        """Get age of deferred item in seconds."""
        return (datetime.now() - self.deferred_at).total_seconds()


class DeferredQueue:
    """
    Queue for deferred communications.
    
    Manages communications that should be reconsidered later based on
    timing, context, or other conditions.
    
    Attributes:
        queue: Active deferred communications
        released_history: History of released items
        expired_history: History of expired items
        config: Configuration dictionary
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize deferred queue.
        
        Args:
            config: Optional configuration with keys:
                - max_queue_size: Maximum queue size (default: 20)
                - max_history_size: Maximum history size (default: 50)
                - default_defer_seconds: Default defer duration (default: 30)
                - max_defer_attempts: Max reconsideration attempts (default: 3)
        """
        self.queue: List[DeferredCommunication] = []
        self.released_history: List[DeferredCommunication] = []
        self.expired_history: List[DeferredCommunication] = []
        
        # Load configuration
        config = config or {}
        self.max_queue_size = max(1, config.get("max_queue_size", 20))
        self.max_history_size = max(1, config.get("max_history_size", 50))
        self.default_defer_seconds = max(1, config.get("default_defer_seconds", 30))
        self.max_defer_attempts = max(1, config.get("max_defer_attempts", 3))
        
        logger.debug(f"DeferredQueue initialized: max_queue={self.max_queue_size}, "
                    f"default_defer={self.default_defer_seconds}s")
    
    def defer(
        self,
        urge: Any,
        reason: DeferralReason,
        release_seconds: Optional[float] = None,
        condition: Optional[str] = None,
        priority: Optional[float] = None,
        max_age_seconds: Optional[float] = None
    ) -> DeferredCommunication:
        """
        Add communication to deferred queue.
        
        Args:
            urge: CommunicationUrge to defer
            reason: DeferralReason enum value
            release_seconds: Seconds to wait before release (None = use default)
            condition: Human-readable release condition description
            priority: Priority override (None = use urge priority)
            max_age_seconds: Expiration override (None = use default 300s)
            
        Returns:
            Created DeferredCommunication instance
        """
        # Calculate release time
        if release_seconds is None:
            release_seconds = self.default_defer_seconds
        
        release_after = datetime.now() + timedelta(seconds=release_seconds)
        
        # Determine priority (from urge if not specified)
        if priority is None:
            priority = getattr(urge, 'priority', 0.5)
        
        # Create deferred item
        deferred = DeferredCommunication(
            urge=urge,
            reason=reason,
            release_after=release_after,
            release_condition=condition or f"Wait {release_seconds:.0f} seconds",
            priority=priority,
            max_age_seconds=max_age_seconds or 300.0
        )
        
        # Add to queue
        self.queue.append(deferred)
        
        # Maintain size limit by removing lowest priority item if needed
        if len(self.queue) > self.max_queue_size:
            self._remove_lowest_priority()
        
        logger.debug(f"Deferred {reason.value}: {condition or 'default'} "
                    f"(queue size: {len(self.queue)})")
        
        return deferred
    
    def _remove_lowest_priority(self) -> None:
        """Remove lowest priority item from queue to maintain size limit."""
        if not self.queue:
            return
        
        # Find lowest priority item (considering both priority and age)
        lowest = min(self.queue, key=lambda d: d.priority * (1.0 - d.get_age_seconds() / 600.0))
        self.queue.remove(lowest)
        logger.debug(f"Removed lowest priority deferred: {lowest.reason.value}")
    
    def check_ready(self) -> Optional[DeferredCommunication]:
        """
        Get highest priority ready item, if any.
        
        Checks all items for readiness, returns the one with highest
        weighted priority (priority * urge intensity). Items that have
        exceeded max_defer_attempts are not returned.
        
        Returns:
            Highest priority ready DeferredCommunication, or None
        """
        # Filter to ready items that haven't exceeded attempts
        ready_items = [
            d for d in self.queue
            if d.is_ready() and d.attempts < self.max_defer_attempts
        ]
        
        if not ready_items:
            return None
        
        # Sort by weighted priority: priority * urge intensity
        def priority_score(d: DeferredCommunication) -> float:
            urge_intensity = getattr(d.urge, 'get_current_intensity', lambda: 0.5)()
            return d.priority * urge_intensity
        
        best = max(ready_items, key=priority_score)
        best.increment_attempts()
        
        # If this was the last attempt, move to released history
        if best.attempts >= self.max_defer_attempts:
            self.queue.remove(best)
            self._add_to_released_history(best)
            logger.debug(f"Released after {best.attempts} attempts: {best.reason.value}")
        
        return best
    
    def cleanup_expired(self) -> List[DeferredCommunication]:
        """
        Remove and return expired items.
        
        Expired items are moved to expired_history.
        
        Returns:
            List of expired DeferredCommunication instances
        """
        expired = [d for d in self.queue if d.is_expired()]
        
        if not expired:
            return []
        
        # Remove from queue and add to history
        for item in expired:
            self.queue.remove(item)
            self._add_to_expired_history(item)
        
        logger.debug(f"Cleaned up {len(expired)} expired deferrals")
        return expired
    
    def _add_to_released_history(self, deferred: DeferredCommunication) -> None:
        """Add to released history with size limit."""
        self.released_history.append(deferred)
        if len(self.released_history) > self.max_history_size:
            self.released_history = self.released_history[-self.max_history_size:]
    
    def _add_to_expired_history(self, deferred: DeferredCommunication) -> None:
        """Add to expired history with size limit."""
        self.expired_history.append(deferred)
        if len(self.expired_history) > self.max_history_size:
            self.expired_history = self.expired_history[-self.max_history_size:]
    
    def remove(self, deferred: DeferredCommunication) -> bool:
        """
        Remove specific item from queue.
        
        Args:
            deferred: DeferredCommunication to remove
            
        Returns:
            True if removed, False if not found
        """
        try:
            self.queue.remove(deferred)
            return True
        except ValueError:
            return False
    
    def clear(self) -> int:
        """
        Clear all items from queue.
        
        Returns:
            Number of items cleared
        """
        count = len(self.queue)
        self.queue.clear()
        return count
    
    def get_queue_summary(self) -> Dict[str, Any]:
        """
        Get summary of queue state.
        
        Returns:
            Dictionary with queue statistics and state
        """
        # Count by reason
        reason_counts = {reason: 0 for reason in DeferralReason}
        for item in self.queue:
            reason_counts[item.reason] += 1
        
        # Count ready items
        ready_count = sum(1 for d in self.queue if d.is_ready())
        
        # Find oldest and newest
        oldest_age = max((d.get_age_seconds() for d in self.queue), default=0.0)
        newest_age = min((d.get_age_seconds() for d in self.queue), default=0.0)
        
        return {
            "queue_size": len(self.queue),
            "ready_count": ready_count,
            "released_count": len(self.released_history),
            "expired_count": len(self.expired_history),
            "reasons": {reason.value: count for reason, count in reason_counts.items()},
            "oldest_age_seconds": oldest_age,
            "newest_age_seconds": newest_age,
            "average_priority": (
                sum(d.priority for d in self.queue) / len(self.queue)
                if self.queue else 0.0
            )
        }
