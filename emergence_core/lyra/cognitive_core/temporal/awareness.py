"""
Temporal Context and Awareness: Enhanced temporal grounding with subjective time awareness.

This module implements temporal grounding that goes beyond simple timestamps to provide
subjective awareness of time passage, session boundaries, and how time affects cognitive state.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TemporalContext:
    """
    Current temporal awareness state with subjective time perception.
    
    Attributes:
        current_time: Current moment
        session_start: When the current session began
        last_interaction: When the last interaction occurred
        elapsed_since_last: Time elapsed since last interaction
        session_duration: How long the current session has been running
        is_new_session: Whether this is the start of a new session
        session_number: Sequential session counter
    """
    current_time: datetime
    session_start: datetime
    last_interaction: datetime
    elapsed_since_last: timedelta
    session_duration: timedelta
    is_new_session: bool
    session_number: int
    
    @property
    def time_description(self) -> str:
        """Human-readable time since last interaction."""
        return self._format_elapsed(self.elapsed_since_last)
    
    @staticmethod
    def _format_elapsed(elapsed: timedelta) -> str:
        """Format elapsed time into natural language."""
        seconds = elapsed.total_seconds()
        
        if seconds < 300:  # 5 minutes
            return "moments ago"
        elif seconds < 3600:  # 1 hour
            return f"{int(seconds / 60)} minutes ago"
        elif seconds < 86400:  # 1 day
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds < 604800:  # 7 days
            days = elapsed.days
            return f"{days} day{'s' if days != 1 else ''} ago"
        else:
            weeks = int(elapsed.days / 7)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
    
    @property
    def session_description(self) -> str:
        """Human-readable session duration."""
        seconds = self.session_duration.total_seconds()
        
        if seconds < 60:
            return "just started"
        elif seconds < 3600:  # 1 hour
            return f"{int(seconds / 60)} minute{'s' if int(seconds / 60) != 1 else ''}"
        else:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''}"


@dataclass
class Session:
    """Conversation session with temporal and contextual metadata."""
    id: str
    start_time: datetime
    last_interaction: datetime
    interaction_count: int
    emotional_arc: List[Any]
    topics: List[str]
    summary: Optional[str] = None


class TemporalAwareness:
    """
    Enhanced temporal awareness with session tracking and subjective time perception.
    
    This class provides genuine temporal grounding - awareness of time passage,
    session boundaries, and how time affects cognitive state.
    """
    
    def __init__(self, session_gap_threshold: Optional[timedelta] = None):
        """
        Initialize temporal awareness system.
        
        Args:
            session_gap_threshold: Time gap that indicates a new session (default: 1 hour)
        """
        self.session_gap_threshold = session_gap_threshold or timedelta(hours=1)
        self.current_session: Optional[Session] = None
        self.session_history: List[Session] = []
        self._session_counter = 0
        
        logger.info(f"✅ Enhanced TemporalAwareness initialized (gap threshold: {self.session_gap_threshold})")
    
    def update(self, interaction_time: Optional[datetime] = None) -> TemporalContext:
        """
        Update temporal context with new interaction.
        
        Args:
            interaction_time: Time of interaction (default: now)
            
        Returns:
            TemporalContext with current temporal state
        """
        interaction_time = interaction_time or datetime.now()
        
        # Determine session status
        is_new = self._should_start_new_session(interaction_time)
        
        if is_new:
            if self.current_session:
                self._end_session()
            self._start_new_session(interaction_time)
        
        # Calculate elapsed and update session
        elapsed = interaction_time - self.current_session.last_interaction
        self.current_session.last_interaction = interaction_time
        self.current_session.interaction_count += 1
        
        # Create and return context
        return TemporalContext(
            current_time=interaction_time,
            session_start=self.current_session.start_time,
            last_interaction=interaction_time,
            elapsed_since_last=elapsed,
            session_duration=interaction_time - self.current_session.start_time,
            is_new_session=is_new,
            session_number=self._session_counter
        )
    
    def _should_start_new_session(self, interaction_time: datetime) -> bool:
        """Check if a new session should be started."""
        if self.current_session is None:
            return True
        time_gap = interaction_time - self.current_session.last_interaction
        return time_gap > self.session_gap_threshold
    
    def _start_new_session(self, start_time: datetime) -> None:
        """
        Start a new session.
        
        Args:
            start_time: When the session begins
        """
        self._session_counter += 1
        self.current_session = Session(
            id=f"session_{self._session_counter}_{start_time.isoformat()}",
            start_time=start_time,
            last_interaction=start_time,
            interaction_count=0,
            emotional_arc=[],
            topics=[]
        )
        logger.info(f"🔔 New session started: #{self._session_counter}")
    
    def _end_session(self) -> None:
        """End the current session and archive it."""
        if self.current_session is not None:
            self.session_history.append(self.current_session)
            logger.info(f"📝 Session ended: #{self._session_counter}, "
                       f"{self.current_session.interaction_count} interactions")
            self.current_session = None
    
    def get_last_session(self) -> Optional[Session]:
        """
        Get the most recent completed session.
        
        Returns:
            Last session or None if no history
        """
        return self.session_history[-1] if self.session_history else None
    
    @property
    def session_count(self) -> int:
        """
        Get the total number of sessions.
        
        Returns:
            Total session count
        """
        return self._session_counter
    
    def get_context(self) -> dict:
        """
        Get complete temporal context as a dictionary.
        
        Returns:
            Dictionary with temporal state information
        """
        if self.current_session is None:
            return {"status": "no_active_session"}
        
        now = datetime.now()
        elapsed = now - self.current_session.last_interaction
        
        return {
            "session_id": self.current_session.id,
            "session_number": self._session_counter,
            "session_start": self.current_session.start_time.isoformat(),
            "last_interaction": self.current_session.last_interaction.isoformat(),
            "elapsed_seconds": elapsed.total_seconds(),
            "elapsed_description": self._format_duration(elapsed),
            "interaction_count": self.current_session.interaction_count,
            "total_sessions": len(self.session_history) + (1 if self.current_session else 0)
        }
    
    @staticmethod
    def _format_duration(duration: timedelta) -> str:
        """
        Format a timedelta into human-readable string.
        
        Args:
            duration: Time duration to format
            
        Returns:
            Human-readable duration string
        """
        seconds = duration.total_seconds()
        
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''}"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''}"
