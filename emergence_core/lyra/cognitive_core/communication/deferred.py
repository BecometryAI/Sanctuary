"""
Deferred Communication Queue - Queue communications for better timing.

Manages communications that should be deferred rather than immediately 
spoken or silenced, with priority ordering and expiration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configuration defaults
DEFAULT_MAX_QUEUE_SIZE = 20
DEFAULT_MAX_HISTORY_SIZE = 50
DEFAULT_DEFER_SECONDS = 30
DEFAULT_MAX_AGE_SECONDS = 300.0
DEFAULT_MAX_ATTEMPTS = 3
MIN_CONFIG_VALUE = 1


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
    """Communication deferred for later reconsideration."""
    urge: Any  # CommunicationUrge instance
    reason: DeferralReason
    deferred_at: datetime = field(default_factory=datetime.now)
    release_condition: str = ""
    release_after: Optional[datetime] = None
    priority: float = 0.5
    max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS
    attempts: int = 0
    
    def is_ready(self) -> bool:
        """Check if ready to be released (not expired and time-based condition met)."""
        if self.is_expired():
            return False
        return self.release_after is None or datetime.now() >= self.release_after
    
    def is_expired(self) -> bool:
        """Check if too old and should be discarded."""
        return self.get_age_seconds() > self.max_age_seconds
    
    def increment_attempts(self) -> None:
        """Increment reconsideration attempt counter."""
        self.attempts += 1
    
    def get_age_seconds(self) -> float:
        """Get age of deferred item in seconds."""
        return (datetime.now() - self.deferred_at).total_seconds()


class DeferredQueue:
    """Queue for deferred communications with priority ordering and expiration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize deferred queue with optional configuration."""
        self.queue: List[DeferredCommunication] = []
        self.released_history: List[DeferredCommunication] = []
        self.expired_history: List[DeferredCommunication] = []
        
        # Load and validate configuration
        config = config or {}
        self.max_queue_size = max(MIN_CONFIG_VALUE, config.get("max_queue_size", DEFAULT_MAX_QUEUE_SIZE))
        self.max_history_size = max(MIN_CONFIG_VALUE, config.get("max_history_size", DEFAULT_MAX_HISTORY_SIZE))
        self.default_defer_seconds = max(MIN_CONFIG_VALUE, config.get("default_defer_seconds", DEFAULT_DEFER_SECONDS))
        self.max_defer_attempts = max(MIN_CONFIG_VALUE, config.get("max_defer_attempts", DEFAULT_MAX_ATTEMPTS))
        
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
        """Add communication to deferred queue with priority ordering."""
        # Use defaults if not specified
        if release_seconds is None:
            release_seconds = self.default_defer_seconds
        if priority is None:
            priority = getattr(urge, 'priority', 0.5)
        if max_age_seconds is None:
            max_age_seconds = DEFAULT_MAX_AGE_SECONDS
        
        # Create deferred item
        deferred = DeferredCommunication(
            urge=urge,
            reason=reason,
            release_after=datetime.now() + timedelta(seconds=release_seconds),
            release_condition=condition or f"Wait {release_seconds:.0f}s",
            priority=priority,
            max_age_seconds=max_age_seconds
        )
        
        self.queue.append(deferred)
        
        # Maintain size limit
        if len(self.queue) > self.max_queue_size:
            self._remove_lowest_priority()
        
        logger.debug(f"Deferred {reason.value} (queue size: {len(self.queue)})")
        return deferred
    
    def _remove_lowest_priority(self) -> None:
        """Remove lowest priority item to maintain queue size limit."""
        if self.queue:
            # Consider both priority and age (newer items preferred)
            lowest = min(self.queue, key=lambda d: d.priority * (1.0 - min(d.get_age_seconds() / 600.0, 1.0)))
            self.queue.remove(lowest)
            logger.debug(f"Removed lowest priority: {lowest.reason.value}")
    
    def check_ready(self) -> Optional[DeferredCommunication]:
        """Get highest priority ready item that hasn't exceeded max attempts."""
        # Filter ready items with attempts remaining
        ready_items = [d for d in self.queue if d.is_ready() and d.attempts < self.max_defer_attempts]
        
        if not ready_items:
            return None
        
        # Get item with highest weighted priority
        best = max(ready_items, key=lambda d: d.priority * getattr(d.urge, 'get_current_intensity', lambda: 0.5)())
        best.increment_attempts()
        
        # Remove and track if max attempts reached
        if best.attempts >= self.max_defer_attempts:
            self.queue.remove(best)
            self._add_to_history(self.released_history, best)
            logger.debug(f"Released after {best.attempts} attempts: {best.reason.value}")
        
        return best
    
    def cleanup_expired(self) -> List[DeferredCommunication]:
        """Remove and return expired items, moving them to history."""
        expired = [d for d in self.queue if d.is_expired()]
        
        if expired:
            # More efficient: rebuild queue without expired items
            self.queue = [d for d in self.queue if not d.is_expired()]
            for item in expired:
                self._add_to_history(self.expired_history, item)
            logger.debug(f"Cleaned up {len(expired)} expired deferrals")
        
        return expired
    
    def _add_to_history(self, history: List[DeferredCommunication], item: DeferredCommunication) -> None:
        """Add item to history with size limit enforcement."""
        history.append(item)
        if len(history) > self.max_history_size:
            del history[:-self.max_history_size]
    
    def remove(self, deferred: DeferredCommunication) -> bool:
        """Remove specific item from queue. Returns True if removed."""
        try:
            self.queue.remove(deferred)
            return True
        except ValueError:
            return False
    
    def clear(self) -> int:
        """Clear all items from queue. Returns number of items cleared."""
        count = len(self.queue)
        self.queue.clear()
        return count
    
    def get_queue_summary(self) -> Dict[str, Any]:
        """Get summary of queue state with statistics."""
        if not self.queue:
            return {
                "queue_size": 0,
                "ready_count": 0,
                "released_count": len(self.released_history),
                "expired_count": len(self.expired_history),
                "reasons": {r.value: 0 for r in DeferralReason},
                "oldest_age_seconds": 0.0,
                "newest_age_seconds": 0.0,
                "average_priority": 0.0
            }
        
        # Single pass for efficiency
        reason_counts = {r: 0 for r in DeferralReason}
        ready_count = 0
        total_priority = 0.0
        ages = []
        
        for item in self.queue:
            reason_counts[item.reason] += 1
            if item.is_ready():
                ready_count += 1
            total_priority += item.priority
            ages.append(item.get_age_seconds())
        
        return {
            "queue_size": len(self.queue),
            "ready_count": ready_count,
            "released_count": len(self.released_history),
            "expired_count": len(self.expired_history),
            "reasons": {r.value: count for r, count in reason_counts.items()},
            "oldest_age_seconds": max(ages),
            "newest_age_seconds": min(ages),
            "average_priority": total_priority / len(self.queue)
        }
