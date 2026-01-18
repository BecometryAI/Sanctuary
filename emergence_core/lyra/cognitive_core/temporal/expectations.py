"""
Temporal Expectations: Learning and anticipating when events should occur.

This module implements temporal pattern learning to develop expectations about
when certain events should happen based on past patterns.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class TemporalExpectation:
    """
    Represents an expectation about when an event should occur.
    
    Attributes:
        event_type: Type of event expected
        expected_time: When the event is expected to occur
        confidence: Confidence in the expectation (0.0-1.0)
        is_overdue: Whether the expected time has passed
        average_interval: Average time between occurrences
        pattern_strength: How consistent the pattern is
    """
    event_type: str
    expected_time: datetime
    confidence: float
    is_overdue: bool
    average_interval: Optional[float] = None
    pattern_strength: Optional[float] = None


class TemporalExpectations:
    """
    Learns temporal patterns and generates expectations.
    
    Tracks when events occur and learns patterns to anticipate future occurrences.
    """
    
    def __init__(self, min_observations: int = 3):
        """
        Initialize temporal expectations system.
        
        Args:
            min_observations: Minimum observations needed to form expectations
        """
        self.patterns: Dict[str, List[datetime]] = {}
        self.min_observations = min_observations
        
        logger.info(f"✅ TemporalExpectations initialized (min observations: {min_observations})")
    
    def record_event(self, event_type: str, time: Optional[datetime] = None) -> None:
        """
        Record when an event occurred to learn patterns.
        
        Args:
            event_type: Type of event that occurred
            time: When it occurred (default: now)
        """
        if time is None:
            time = datetime.now()
        
        if event_type not in self.patterns:
            self.patterns[event_type] = []
        
        self.patterns[event_type].append(time)
        
        # Keep only recent history (last 100 occurrences)
        if len(self.patterns[event_type]) > 100:
            self.patterns[event_type] = self.patterns[event_type][-100:]
        
        logger.debug(f"📊 Event recorded: {event_type} at {time.isoformat()}")
    
    def get_expectation(self, event_type: str) -> Optional[TemporalExpectation]:
        """
        Get expected timing for event type based on learned patterns.
        
        Args:
            event_type: Type of event to get expectation for
            
        Returns:
            TemporalExpectation or None if insufficient data
        """
        if event_type not in self.patterns:
            return None
        
        times = self.patterns[event_type]
        
        if len(times) < self.min_observations:
            logger.debug(f"⏳ Insufficient data for {event_type}: {len(times)}/{self.min_observations}")
            return None
        
        # Calculate intervals between occurrences
        intervals = [
            (times[i+1] - times[i]).total_seconds()
            for i in range(len(times) - 1)
        ]
        
        if not intervals:
            return None
        
        # Calculate statistics
        avg_interval = statistics.mean(intervals)
        
        # Calculate pattern strength (inverse of coefficient of variation)
        if len(intervals) > 1:
            std_dev = statistics.stdev(intervals)
            cv = std_dev / avg_interval if avg_interval > 0 else 1.0
            pattern_strength = 1.0 / (1.0 + cv)  # 0 to 1, higher is more consistent
        else:
            pattern_strength = 0.5
        
        # Predict next occurrence
        last_time = times[-1]
        expected_time = last_time + timedelta(seconds=avg_interval)
        
        # Calculate confidence based on pattern strength and recency
        now = datetime.now()
        time_since_last = (now - last_time).total_seconds()
        recency_factor = min(1.0, avg_interval / (time_since_last + 1))
        confidence = pattern_strength * recency_factor
        
        # Check if overdue
        is_overdue = now > expected_time
        
        return TemporalExpectation(
            event_type=event_type,
            expected_time=expected_time,
            confidence=confidence,
            is_overdue=is_overdue,
            average_interval=avg_interval,
            pattern_strength=pattern_strength
        )
    
    def get_active_expectations(self) -> List[TemporalExpectation]:
        """
        Get all active temporal expectations.
        
        Returns:
            List of expectations for all tracked event types
        """
        expectations = []
        
        for event_type in self.patterns.keys():
            expectation = self.get_expectation(event_type)
            if expectation:
                expectations.append(expectation)
        
        # Sort by confidence (highest first)
        expectations.sort(key=lambda e: e.confidence, reverse=True)
        
        return expectations
    
    def get_overdue_expectations(self) -> List[TemporalExpectation]:
        """
        Get expectations that are currently overdue.
        
        Returns:
            List of overdue expectations
        """
        all_expectations = self.get_active_expectations()
        overdue = [e for e in all_expectations if e.is_overdue]
        
        if overdue:
            logger.debug(f"⚠️  {len(overdue)} overdue expectations")
        
        return overdue
    
    def get_pattern_summary(self) -> Dict[str, Dict]:
        """
        Get summary of all learned temporal patterns.
        
        Returns:
            Dictionary mapping event types to their pattern summaries
        """
        summary = {}
        
        for event_type in self.patterns.keys():
            times = self.patterns[event_type]
            expectation = self.get_expectation(event_type)
            
            summary[event_type] = {
                "observation_count": len(times),
                "first_observed": times[0].isoformat() if times else None,
                "last_observed": times[-1].isoformat() if times else None,
                "has_expectation": expectation is not None
            }
            
            if expectation:
                summary[event_type].update({
                    "expected_time": expectation.expected_time.isoformat(),
                    "confidence": expectation.confidence,
                    "is_overdue": expectation.is_overdue,
                    "pattern_strength": expectation.pattern_strength
                })
        
        return summary
    
    def clear_pattern(self, event_type: str) -> None:
        """
        Clear learned pattern for an event type.
        
        Args:
            event_type: Event type to clear
        """
        if event_type in self.patterns:
            del self.patterns[event_type]
            logger.info(f"🗑️  Cleared pattern for: {event_type}")
    
    def clear_all_patterns(self) -> None:
        """Clear all learned patterns."""
        self.patterns.clear()
        logger.info("🗑️  All patterns cleared")
