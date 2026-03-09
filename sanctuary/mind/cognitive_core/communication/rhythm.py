"""
Conversational Rhythm Model - Understanding conversation timing and flow.

This module tracks conversation dynamics to inform timing decisions:
turn-taking, pause detection, response latencies, and conversation tempo.
It provides timing appropriateness signals to the inhibition system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configuration constants
MIN_PAUSE_THRESHOLD = 0.5  # Minimum natural pause threshold in seconds
MIN_RAPID_THRESHOLD = 0.1  # Minimum rapid exchange threshold in seconds
MIN_TURN_HISTORY = 10      # Minimum turn history to maintain
MIN_RESPONSE_TIME = 0.5    # Minimum average response time in seconds
MIN_TURN_LENGTH = 10.0     # Minimum average turn length in characters
PAUSE_NORMALIZATION_FACTOR = 2.0  # Multiplier for pause appropriateness normalization


class ConversationPhase(Enum):
    """Current phase of conversation flow."""
    HUMAN_SPEAKING = "human_speaking"        # Human is actively speaking
    HUMAN_PAUSED = "human_paused"            # Human paused, may continue or be done
    SYSTEM_SPEAKING = "system_speaking"      # System is speaking
    MUTUAL_SILENCE = "mutual_silence"        # Neither party speaking
    RAPID_EXCHANGE = "rapid_exchange"        # Fast back-and-forth conversation


@dataclass
class ConversationTurn:
    """
    Represents a single conversational turn.
    
    Attributes:
        speaker: "human" or "system"
        started_at: When this turn began
        ended_at: When this turn ended (None if ongoing)
        content_length: Length of content in characters
    """
    speaker: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    content_length: int = 0
    
    @property
    def duration(self) -> float:
        """Get duration of this turn in seconds."""
        if self.ended_at is None:
            return (datetime.now() - self.started_at).total_seconds()
        return (self.ended_at - self.started_at).total_seconds()
    
    @property
    def is_complete(self) -> bool:
        """Check if this turn has ended."""
        return self.ended_at is not None


class ConversationalRhythmModel:
    """
    Tracks conversation rhythm and timing to inform communication decisions.
    
    Monitors turn-taking patterns, response latencies, pause durations,
    and conversation tempo to determine when it's appropriate to speak.
    
    Attributes:
        turns: History of conversation turns
        current_phase: Current conversation phase
        avg_response_time: Average time between turns (seconds)
        avg_turn_length: Average turn length (characters)
        config: Configuration parameters
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize conversational rhythm model.
        
        Args:
            config: Optional configuration dict with keys:
                - natural_pause_threshold: Seconds to consider natural pause (default: 2.0)
                - rapid_exchange_threshold: Seconds for rapid exchange (default: 1.0)
                - max_turn_history: Maximum turns to track (default: 50)
                - default_response_time: Initial avg response time (default: 2.0)
                - default_turn_length: Initial avg turn length chars (default: 50.0)
        """
        self.turns: List[ConversationTurn] = []
        self.current_phase: ConversationPhase = ConversationPhase.MUTUAL_SILENCE
        self.avg_response_time: float = 2.0
        self.avg_turn_length: float = 50.0
        self._last_phase_update: datetime = datetime.now()
        
        # Load configuration
        config = config or {}
        self.natural_pause_threshold = max(MIN_PAUSE_THRESHOLD, config.get("natural_pause_threshold", 2.0))
        self.rapid_exchange_threshold = max(MIN_RAPID_THRESHOLD, config.get("rapid_exchange_threshold", 1.0))
        self.max_turn_history = max(MIN_TURN_HISTORY, config.get("max_turn_history", 50))
        self.avg_response_time = max(MIN_RESPONSE_TIME, config.get("default_response_time", 2.0))
        self.avg_turn_length = max(MIN_TURN_LENGTH, config.get("default_turn_length", 50.0))
        
        logger.debug(
            f"ConversationalRhythmModel initialized: "
            f"pause_threshold={self.natural_pause_threshold:.1f}s, "
            f"rapid_threshold={self.rapid_exchange_threshold:.1f}s"
        )
    
    def record_human_input(self, content: str) -> None:
        """Record human started/continued speaking."""
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        
        now = datetime.now()
        content_len = len(content)
        
        if self.turns and self.turns[-1].speaker == "human" and not self.turns[-1].is_complete:
            self.turns[-1].content_length += content_len
            logger.debug(f"Continued human turn, new length: {self.turns[-1].content_length}")
        else:
            if self.turns and not self.turns[-1].is_complete:
                self.turns[-1].ended_at = now
            
            self.turns.append(ConversationTurn(
                speaker="human",
                started_at=now,
                content_length=content_len
            ))
            logger.debug(f"Started new human turn, length: {content_len}")
        
        self._limit_history()
        self.update_phase()
        self._update_averages()
    
    def record_system_output(self, content: str) -> None:
        """Record system spoke."""
        if not isinstance(content, str):
            raise TypeError(f"content must be str, got {type(content).__name__}")
        
        now = datetime.now()
        
        if self.turns and not self.turns[-1].is_complete:
            self.turns[-1].ended_at = now
        
        self.turns.append(ConversationTurn(
            speaker="system",
            started_at=now,
            ended_at=now,
            content_length=len(content)
        ))
        logger.debug(f"Recorded system turn, length: {len(content)}")
        
        self._limit_history()
        self.update_phase()
        self._update_averages()
    
    def _get_silence_duration(self) -> float:
        """Get silence duration since last completed turn."""
        if not self.turns or not self.turns[-1].is_complete:
            return 0.0
        return (datetime.now() - self.turns[-1].ended_at).total_seconds()
    
    def _get_time_since_last_event(self) -> float:
        """
        Get time since last turn event.
        
        For incomplete turns: time since turn started (turn duration)
        For complete turns: time since turn ended (silence duration)
        """
        if not self.turns:
            return 0.0
        return self._get_silence_duration() if self.turns[-1].is_complete else \
               (datetime.now() - self.turns[-1].started_at).total_seconds()
    
    def _limit_history(self) -> None:
        """Limit turn history to configured maximum."""
        if len(self.turns) > self.max_turn_history:
            self.turns = self.turns[-self.max_turn_history:]
    
    def update_phase(self) -> None:
        """Update current conversation phase based on timing and turn state."""
        now = datetime.now()
        
        if not self.turns:
            self.current_phase = ConversationPhase.MUTUAL_SILENCE
            self._last_phase_update = now
            return
        
        last_turn = self.turns[-1]
        
        # Check for rapid exchange
        if self._is_rapid_exchange():
            self.current_phase = ConversationPhase.RAPID_EXCHANGE
            self._last_phase_update = now
            return
        
        # Determine phase based on last turn
        if not last_turn.is_complete:
            self.current_phase = ConversationPhase.HUMAN_SPEAKING if last_turn.speaker == "human" \
                                 else ConversationPhase.SYSTEM_SPEAKING
        elif last_turn.speaker == "human":
            silence = self._get_silence_duration()
            self.current_phase = ConversationPhase.HUMAN_PAUSED if silence < self.natural_pause_threshold \
                                 else ConversationPhase.MUTUAL_SILENCE
        else:
            self.current_phase = ConversationPhase.MUTUAL_SILENCE
        
        self._last_phase_update = now
    
    def _is_rapid_exchange(self) -> bool:
        """Check if conversation is in rapid exchange mode."""
        if len(self.turns) < 3:
            return False
        
        recent_turns = self.turns[-3:]
        if not all(turn.is_complete for turn in recent_turns):
            return False
        
        gaps = [(recent_turns[i + 1].started_at - recent_turns[i].ended_at).total_seconds()
                for i in range(len(recent_turns) - 1)]
        return sum(gaps) / len(gaps) < self.rapid_exchange_threshold
    
    def get_timing_appropriateness(self) -> float:
        """
        Get how appropriate it is to speak now (0.0-1.0).
        
        Returns 1.0 for highly appropriate, 0.0 for inappropriate timing.
        """
        self.update_phase()
        
        if not self.turns:
            return 0.8
        
        time_since_last = self._get_time_since_last_event()
        
        # Return appropriateness based on phase
        phase_handlers = {
            ConversationPhase.HUMAN_SPEAKING: lambda: 0.0,
            ConversationPhase.HUMAN_PAUSED: lambda: self._calc_pause_appropriateness(time_since_last),
            ConversationPhase.SYSTEM_SPEAKING: lambda: 0.1,
            ConversationPhase.MUTUAL_SILENCE: lambda: self._calc_silence_appropriateness(time_since_last),
            ConversationPhase.RAPID_EXCHANGE: lambda: 0.9 if self.turns[-1].speaker == "human" else 0.3
        }
        
        return phase_handlers.get(self.current_phase, lambda: 0.5)()
    
    def _calc_pause_appropriateness(self, time_since_last: float) -> float:
        """Calculate appropriateness during human pause."""
        pause_threshold_full = self.natural_pause_threshold * PAUSE_NORMALIZATION_FACTOR
        normalized_pause = min(1.0, time_since_last / pause_threshold_full)
        return 0.3 + (0.7 * normalized_pause)
    
    def _calc_silence_appropriateness(self, time_since_last: float) -> float:
        """Calculate appropriateness during mutual silence."""
        if time_since_last < self.avg_response_time:
            return 0.5 + (0.3 * (time_since_last / self.avg_response_time))
        excess = time_since_last - self.avg_response_time
        return min(1.0, 0.8 + (0.2 * (excess / self.avg_response_time)))
    
    def get_suggested_wait_time(self) -> float:
        """Get suggested seconds to wait before speaking."""
        appropriateness = self.get_timing_appropriateness()
        
        if appropriateness >= 0.8 or not self.turns:
            return 0.0 if appropriateness >= 0.8 else 0.5
        
        time_since_last = self._get_time_since_last_event()
        
        # Calculate wait time based on phase
        wait_handlers = {
            ConversationPhase.HUMAN_SPEAKING: lambda: self.natural_pause_threshold,
            ConversationPhase.HUMAN_PAUSED: lambda: max(0.0, self.natural_pause_threshold - time_since_last),
            ConversationPhase.SYSTEM_SPEAKING: lambda: 0.5,
            ConversationPhase.MUTUAL_SILENCE: lambda: (self.avg_response_time - time_since_last) * 0.5 
                                                      if time_since_last < self.avg_response_time else 0.0,
            ConversationPhase.RAPID_EXCHANGE: lambda: 0.1 if self.turns[-1].speaker == "human" else 1.0
        }
        
        return wait_handlers.get(self.current_phase, lambda: 0.0)()
    
    def is_natural_pause(self) -> bool:
        """Detect if we're at a natural conversation pause."""
        if not self.turns:
            return True
        
        self.update_phase()
        
        if self.current_phase in [ConversationPhase.HUMAN_PAUSED, ConversationPhase.MUTUAL_SILENCE]:
            return self._get_silence_duration() >= self.natural_pause_threshold
        
        return self.current_phase == ConversationPhase.RAPID_EXCHANGE and \
               self.turns[-1].speaker == "human"
    
    def get_rhythm_summary(self) -> Dict[str, Any]:
        """Get summary of conversation rhythm metrics."""
        tempo = "normal"
        if self.avg_response_time < 1.5:
            tempo = "fast"
        elif self.avg_response_time > 4.0:
            tempo = "slow"
        
        return {
            "current_phase": self.current_phase.value,
            "total_turns": len(self.turns),
            "avg_response_time": round(self.avg_response_time, 2),
            "avg_turn_length": round(self.avg_turn_length, 1),
            "conversation_tempo": tempo,
            "last_speaker": self.turns[-1].speaker if self.turns else None,
            "silence_duration": round(self._get_silence_duration(), 2),
            "is_natural_pause": self.is_natural_pause(),
            "timing_appropriateness": round(self.get_timing_appropriateness(), 2),
            "suggested_wait_time": round(self.get_suggested_wait_time(), 2),
            "phase_updated_at": self._last_phase_update.isoformat()
        }
    
    def _update_averages(self) -> None:
        """Update running averages for response time and turn length."""
        # Calculate response times (gaps between turns) - needs at least 2 turns
        if len(self.turns) >= 2:
            response_times = []
            for i in range(len(self.turns) - 1):
                if self.turns[i].is_complete:
                    gap = (self.turns[i + 1].started_at - self.turns[i].ended_at).total_seconds()
                    if gap >= 0:  # Ignore negative gaps (shouldn't happen but be safe)
                        response_times.append(gap)

            if response_times:
                # Use exponential moving average for responsiveness
                recent_responses = response_times[-10:]  # Last 10 gaps
                new_avg = sum(recent_responses) / len(recent_responses)
                self.avg_response_time = (0.7 * self.avg_response_time + 0.3 * new_avg)

        # Calculate turn lengths - works with any number of turns
        turn_lengths = [turn.content_length for turn in self.turns if turn.content_length > 0]
        if turn_lengths:
            recent_lengths = turn_lengths[-10:]  # Last 10 turns
            new_avg = sum(recent_lengths) / len(recent_lengths)
            self.avg_turn_length = (0.7 * self.avg_turn_length + 0.3 * new_avg)
