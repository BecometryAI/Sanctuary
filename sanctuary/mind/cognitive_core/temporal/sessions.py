"""
Session Management: Detection and management of conversation sessions.

This module handles session boundaries, context retrieval, and greeting generation
based on temporal gaps between interactions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .awareness import TemporalAwareness, Session
    from ...memory.storage import MemorySystem

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages conversation sessions with awareness of temporal boundaries.
    
    Recognizes new conversations vs. continuations vs. resumptions, and provides
    appropriate contextual information for each situation.
    """
    
    def __init__(self, temporal: TemporalAwareness, memory: Optional[Any] = None):
        """
        Initialize session manager.
        
        Args:
            temporal: TemporalAwareness instance for time tracking
            memory: Optional memory system for session storage/retrieval
        """
        self.temporal = temporal
        self.memory = memory
        
        logger.info("✅ SessionManager initialized")
    
    def on_session_start(self, session: Session) -> None:
        """
        Handle new session beginning.
        
        Retrieves context from last session and applies time passage effects.
        
        Args:
            session: The newly started session
        """
        last_session = self._get_last_session()
        
        if last_session:
            gap = session.start_time - last_session.last_interaction
            logger.info(f"🔄 Resuming after {self._format_gap(gap)}")
            
            # Could trigger time passage effects here
            # Could prepare resumption context here
        else:
            logger.info("🆕 First session - no previous context")
    
    def on_session_end(self, session: Session) -> None:
        """
        Handle session ending.
        
        Generates session summary and stores in memory if available.
        
        Args:
            session: The session that ended
        """
        # Generate basic summary
        session.summary = self._summarize_session(session)
        
        # Store in memory if available
        if self.memory:
            try:
                self._store_session(session)
            except Exception as e:
                logger.warning(f"Failed to store session in memory: {e}")
        
        logger.info(f"📝 Session summary: {session.summary}")
    
    def get_session_greeting_context(self) -> Dict[str, Any]:
        """
        Get context for how to greet based on temporal situation.
        
        Returns:
            Dictionary with greeting context including type and details
        """
        last_session = self._get_last_session()
        
        if not last_session:
            return {
                "type": "first_meeting",
                "context": "This is our first interaction",
                "greeting_hint": "friendly introduction"
            }
        
        gap = datetime.now() - last_session.last_interaction
        
        if gap < timedelta(hours=1):
            return {
                "type": "continuation",
                "context": "we were just talking",
                "gap": gap,
                "greeting_hint": "seamless continuation"
            }
        elif gap < timedelta(days=1):
            return {
                "type": "same_day",
                "context": "earlier today",
                "gap": gap,
                "greeting_hint": "friendly re-engagement",
                "topics": last_session.topics[:3] if last_session.topics else []
            }
        elif gap < timedelta(weeks=1):
            days = gap.days
            return {
                "type": "recent",
                "context": f"{days} day{'s' if days != 1 else ''} ago",
                "gap": gap,
                "greeting_hint": "warm reconnection",
                "topics": last_session.topics[:3] if last_session.topics else [],
                "may_need_context": True
            }
        else:
            weeks = int(gap.days / 7)
            return {
                "type": "long_gap",
                "context": f"{weeks} week{'s' if weeks != 1 else ''}",
                "gap": gap,
                "greeting_hint": "careful reintroduction",
                "may_need_reintroduction": True,
                "topics": last_session.topics[:2] if last_session.topics else []
            }
    
    def get_current_session_info(self) -> Dict[str, Any]:
        """
        Get information about the current session.
        
        Returns:
            Dictionary with current session details
        """
        if not self.temporal.current_session:
            return {"status": "no_active_session"}
        
        session = self.temporal.current_session
        now = datetime.now()
        duration = now - session.start_time
        
        return {
            "session_id": session.id,
            "session_number": self.temporal._session_counter,
            "duration": duration.total_seconds(),
            "duration_formatted": self._format_gap(duration),
            "interaction_count": session.interaction_count,
            "topics": session.topics,
            "emotional_states": len(session.emotional_arc)
        }
    
    def record_topic(self, topic: str) -> None:
        """
        Record a topic discussed in the current session.
        
        Args:
            topic: Topic identifier or description
        """
        if self.temporal.current_session:
            if topic not in self.temporal.current_session.topics:
                self.temporal.current_session.topics.append(topic)
                logger.debug(f"📌 Topic recorded: {topic}")
    
    def record_emotional_state(self, emotional_state: Any) -> None:
        """
        Record an emotional state in the current session.
        
        Args:
            emotional_state: Emotional state object or dict
        """
        if self.temporal.current_session:
            self.temporal.current_session.emotional_arc.append(emotional_state)
    
    def _get_last_session(self) -> Optional[Session]:
        """Get the most recent completed session."""
        return self.temporal.get_last_session()
    
    def _summarize_session(self, session: Session) -> str:
        """
        Generate a summary of a session.
        
        Args:
            session: Session to summarize
            
        Returns:
            Summary string
        """
        duration = session.last_interaction - session.start_time
        
        summary_parts = [
            f"{session.interaction_count} interactions",
            f"duration: {self._format_gap(duration)}"
        ]
        
        if session.topics:
            summary_parts.append(f"topics: {', '.join(session.topics[:3])}")
        
        return "; ".join(summary_parts)
    
    def _store_session(self, session: Session) -> None:
        """
        Store session in memory system.
        
        Args:
            session: Session to store
        """
        # This would integrate with the actual memory system
        # For now, just log
        logger.debug(f"💾 Would store session: {session.id}")
    
    @staticmethod
    def _format_gap(gap: timedelta) -> str:
        """
        Format a time gap into human-readable string.
        
        Args:
            gap: Time gap to format
            
        Returns:
            Formatted string
        """
        seconds = gap.total_seconds()
        
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
