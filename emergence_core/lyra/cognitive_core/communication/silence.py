"""
Silence-as-Action - Explicit choice not to respond.

This module treats silence as an explicit, logged action rather than just
absence of output. Silence decisions are categorized, tracked, and can
influence future communication decisions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Classification thresholds for silence type determination
_CLASSIFICATION_HIGH_INHIBITION = 0.7  # Threshold for choosing discretion
_CLASSIFICATION_LOW_DRIVE = 0.3  # Threshold for nothing to add

# Pressure calculation constants
_PRESSURE_MAX_DURATION_MULTIPLIER = 3.0  # Duration reaches max at 3x threshold
_PRESSURE_DURATION_WEIGHT = 0.6  # 60% weight on duration
_PRESSURE_STREAK_WEIGHT = 0.4  # 40% weight on streak


class SilenceType(Enum):
    """Types of silence decisions."""
    NOTHING_TO_ADD = "nothing_to_add"           # No valuable content
    RESPECTING_SPACE = "respecting_space"       # Giving human room
    STILL_THINKING = "still_thinking"           # Processing continues
    CHOOSING_DISCRETION = "choosing_discretion" # Deliberate restraint
    UNCERTAINTY = "uncertainty"                 # Too unsure to commit
    TIMING = "timing"                           # Waiting for better moment
    REDUNDANCY = "redundancy"                   # Already addressed


@dataclass
class SilenceAction:
    """
    Represents an explicit silence decision.
    
    Attributes:
        silence_type: The category of silence
        reason: Human-readable explanation for silence
        inhibitions: Inhibition factors that caused this silence
        suppressed_urges: Communication urges that were suppressed
        timestamp: When this silence decision was made
        duration: How long silence lasted (None until ended)
    """
    silence_type: SilenceType
    reason: str
    inhibitions: List[Any]  # InhibitionFactor instances
    suppressed_urges: List[Any]  # CommunicationUrge instances
    timestamp: datetime = field(default_factory=datetime.now)
    duration: Optional[float] = None  # Duration in seconds
    
    def end_silence(self) -> float:
        """
        Mark this silence period as ended.
        
        Returns:
            Duration in seconds
        """
        if self.duration is None:
            elapsed = (datetime.now() - self.timestamp).total_seconds()
            self.duration = elapsed
            return elapsed
        return self.duration


class SilenceTracker:
    """
    Tracks silence decisions and their effects on future communication.
    
    Maintains history of silence actions, tracks current silence periods,
    and computes pressure to break silence based on duration and frequency.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize silence tracker.
        
        Args:
            config: Configuration dict with optional keys:
                - max_silence_history: Max silence actions to track (default: 100)
                - silence_pressure_threshold: Silence duration before pressure builds (default: 60)
                - max_silence_streak: Max consecutive silences before pressure (default: 5)
        """
        self.silence_history: List[SilenceAction] = []
        self.current_silence: Optional[SilenceAction] = None
        self.silence_streak: int = 0  # Consecutive silent cycles
        
        # Load configuration
        config = config or {}
        self.max_history = max(1, config.get("max_silence_history", 100))
        self.pressure_threshold_seconds = max(10, config.get("silence_pressure_threshold", 60))
        self.max_silence_streak = max(1, config.get("max_silence_streak", 5))
        
        logger.debug(f"SilenceTracker initialized: max_history={self.max_history}, "
                    f"pressure_threshold={self.pressure_threshold_seconds}s, "
                    f"max_streak={self.max_silence_streak}")
    
    def record_silence(
        self,
        decision_result: Any,
        silence_type: Optional[SilenceType] = None
    ) -> SilenceAction:
        """
        Record an explicit silence decision.
        
        Args:
            decision_result: DecisionResult instance with inhibitions and urges
            silence_type: Optional override for silence type classification
            
        Returns:
            Created SilenceAction
        """
        # Infer silence type if not provided
        if silence_type is None:
            silence_type = self._classify_silence_type(decision_result)
        
        # Extract data from decision result with defaults
        inhibitions = getattr(decision_result, 'inhibitions', [])
        urges = getattr(decision_result, 'urges', [])
        reason = getattr(decision_result, 'reason', 'Silence decision')
        
        # Create silence action
        silence_action = SilenceAction(
            silence_type=silence_type,
            reason=reason,
            inhibitions=inhibitions,
            suppressed_urges=urges
        )
        
        # Update tracking state
        self.current_silence = silence_action
        self.silence_streak += 1
        self.silence_history.append(silence_action)
        
        # Maintain history size limit
        if len(self.silence_history) > self.max_history:
            self.silence_history = self.silence_history[-self.max_history:]
        
        logger.debug(f"Recorded silence: {silence_type.value} - {reason} "
                    f"(streak: {self.silence_streak})")
        
        return silence_action
    
    def _classify_silence_type(self, decision_result: Any) -> SilenceType:
        """
        Classify silence type based on decision result.
        
        Uses keyword matching on reason and fallback to drive/inhibition levels.
        """
        reason_lower = getattr(decision_result, 'reason', '').lower()
        
        # Keyword-based classification (most specific first)
        keyword_map = {
            ('uncertainty', 'confidence'): SilenceType.UNCERTAINTY,
            ('redundant', 'already'): SilenceType.REDUNDANCY,
            ('timing', 'spacing'): SilenceType.TIMING,
            ('processing', 'thinking'): SilenceType.STILL_THINKING,
            ('respect', 'silence appropriate'): SilenceType.RESPECTING_SPACE,
            ('discretion', 'restraint'): SilenceType.CHOOSING_DISCRETION,
        }
        
        for keywords, silence_type in keyword_map.items():
            if any(kw in reason_lower for kw in keywords):
                return silence_type
        
        # Fallback to drive/inhibition level classification
        inhibition_level = getattr(decision_result, 'inhibition_level', 0.0)
        drive_level = getattr(decision_result, 'drive_level', 0.0)
        
        if inhibition_level > _CLASSIFICATION_HIGH_INHIBITION:
            return SilenceType.CHOOSING_DISCRETION
        
        if drive_level < _CLASSIFICATION_LOW_DRIVE:
            return SilenceType.NOTHING_TO_ADD
        
        return SilenceType.NOTHING_TO_ADD
    
    def end_silence(self) -> Optional[SilenceAction]:
        """
        End current silence period when speaking.
        
        Returns:
            The ended SilenceAction with duration set, or None if no active silence
        """
        if self.current_silence is None:
            return None
        
        ended_silence = self.current_silence
        ended_silence.end_silence()
        
        # Reset tracking state
        self.current_silence = None
        self.silence_streak = 0
        
        logger.debug(f"Ended silence after {ended_silence.duration:.1f}s: "
                    f"{ended_silence.silence_type.value}")
        
        return ended_silence
    
    def get_silence_pressure(self) -> float:
        """
        Get pressure to break silence based on duration and streak.
        
        Pressure increases with:
        - Duration of current silence (60% weight)
        - Number of consecutive silence decisions (40% weight)
        
        Duration pressure reaches 1.0 at 3x the threshold.
        Streak pressure reaches 1.0 at max_silence_streak.
        
        Returns:
            Pressure value in range [0.0, 1.0]
        """
        if self.current_silence is None:
            return 0.0
        
        # Duration pressure: 0 at threshold, 1.0 at max_duration_multiplier * threshold
        elapsed = (datetime.now() - self.current_silence.timestamp).total_seconds()
        max_duration = self.pressure_threshold_seconds * _PRESSURE_MAX_DURATION_MULTIPLIER
        duration_pressure = min(1.0, elapsed / max_duration)
        
        # Streak pressure: 0 at 1, 1.0 at max_silence_streak
        streak_pressure = min(1.0, self.silence_streak / self.max_silence_streak)
        
        # Combined pressure using configured weights
        total_pressure = (duration_pressure * _PRESSURE_DURATION_WEIGHT) + \
                        (streak_pressure * _PRESSURE_STREAK_WEIGHT)
        
        return min(1.0, total_pressure)
    
    def get_recent_silences(self, minutes: int = 5) -> List[SilenceAction]:
        """
        Get recent silence actions for introspection.
        
        Args:
            minutes: How many minutes back to look (must be positive)
            
        Returns:
            List of SilenceAction instances from the specified time window
        """
        if not self.silence_history or minutes <= 0:
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            silence for silence in self.silence_history
            if silence.timestamp > cutoff_time
        ]
    
    def get_silence_by_type(self, silence_type: SilenceType) -> List[SilenceAction]:
        """
        Get all silence actions of a specific type.
        
        Args:
            silence_type: The silence type to filter by
            
        Returns:
            List of matching SilenceAction instances
        """
        return [
            silence for silence in self.silence_history
            if silence.silence_type == silence_type
        ]
    
    def get_silence_summary(self) -> Dict[str, Any]:
        """
        Get summary of silence tracker state.
        
        Returns:
            Dictionary with statistics about silence decisions
        """
        # Single pass through history for efficiency
        cutoff_time = datetime.now() - timedelta(minutes=5)
        recent_count = 0
        type_counts = {st: 0 for st in SilenceType}
        
        for silence in self.silence_history:
            if silence.timestamp > cutoff_time:
                recent_count += 1
            type_counts[silence.silence_type] += 1
        
        return {
            "total_silences": len(self.silence_history),
            "recent_silences": recent_count,
            "current_silence": self.current_silence is not None,
            "silence_streak": self.silence_streak,
            "silence_pressure": self.get_silence_pressure(),
            "silence_by_type": {st.value: count for st, count in type_counts.items()},
            "current_silence_duration": (
                (datetime.now() - self.current_silence.timestamp).total_seconds()
                if self.current_silence else None
            )
        }
