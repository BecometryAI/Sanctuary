"""
Temporal Grounding: Central temporal awareness integration.

This module provides the main TemporalGrounding class that integrates all
temporal components into a cohesive system.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from .awareness import TemporalAwareness, TemporalContext
from .sessions import SessionManager
from .effects import TimePassageEffects
from .expectations import TemporalExpectations
from .relative import RelativeTime

logger = logging.getLogger(__name__)


class TemporalGrounding:
    """
    Central temporal awareness for the cognitive system.
    
    Integrates all temporal components to provide:
    - Session-aware time tracking
    - Time passage effects on cognitive state
    - Temporal pattern learning and expectations
    - Relative time descriptions
    """
    
    def __init__(self, config: Optional[Dict] = None, memory: Optional[Any] = None):
        """
        Initialize temporal grounding system.
        
        Args:
            config: Optional configuration dict
            memory: Optional memory system for session storage
        """
        self.config = config or {}
        
        # Initialize components
        session_gap = timedelta(
            seconds=self.config.get("session_gap_threshold_seconds", 3600)
        )
        
        self.awareness = TemporalAwareness(session_gap_threshold=session_gap)
        self.sessions = SessionManager(self.awareness, memory)
        self.effects = TimePassageEffects(self.config.get("effects", {}))
        self.expectations = TemporalExpectations(
            min_observations=self.config.get("min_observations", 3)
        )
        # RelativeTime is a utility class with static methods only
        
        # Track last interaction time for effects
        self._last_effect_time: Optional[datetime] = None
        
        # Track last input/action/output times for cognitive cycle integration
        self._last_input_time: Optional[datetime] = None
        self._last_action_time: Optional[datetime] = None
        self._last_output_time: Optional[datetime] = None
        self._cycle_count: int = 0
        
        logger.info("✅ TemporalGrounding initialized")
    
    def on_interaction(self, time: Optional[datetime] = None) -> TemporalContext:
        """
        Update temporal state on interaction.
        
        Args:
            time: Interaction time (default: now)
            
        Returns:
            TemporalContext with current temporal state
        """
        time = time or datetime.now()
        
        # Update temporal awareness
        context = self.awareness.update(time)
        
        # Handle new session
        if context.is_new_session and self.awareness.current_session:
            self.sessions.on_session_start(self.awareness.current_session)
        
        # Record pattern
        self.expectations.record_event("user_interaction", time)
        
        # Initialize effect time
        if self._last_effect_time is None:
            self._last_effect_time = time
        
        return context
    
    def apply_time_passage_effects(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply time passage effects to cognitive state.
        
        Args:
            state: Current cognitive state
            
        Returns:
            Updated cognitive state
        """
        if self._last_effect_time is None:
            self._last_effect_time = datetime.now()
            return state
        
        now = datetime.now()
        elapsed = now - self._last_effect_time
        self._last_effect_time = now
        
        return self.effects.apply(elapsed, state)
    
    def get_temporal_state(self) -> Dict[str, Any]:
        """
        Get full temporal state for cognitive processing.
        
        Returns:
            Dictionary with complete temporal state
        """
        return {
            "context": self.awareness.get_context(),
            "session": self.sessions.get_current_session_info(),
            "greeting_context": self.sessions.get_session_greeting_context(),
            "expectations": [
                {
                    "event_type": exp.event_type,
                    "expected_time": exp.expected_time.isoformat(),
                    "confidence": exp.confidence,
                    "is_overdue": exp.is_overdue
                }
                for exp in self.expectations.get_active_expectations()
            ],
            "overdue_expectations": [
                exp.event_type for exp in self.expectations.get_overdue_expectations()
            ]
        }
    
    def record_topic(self, topic: str) -> None:
        """
        Record a topic in the current session.
        
        Args:
            topic: Topic identifier or description
        """
        self.sessions.record_topic(topic)
    
    def record_emotional_state(self, emotional_state: Any) -> None:
        """
        Record an emotional state in the current session.
        
        Args:
            emotional_state: Emotional state object or dict
        """
        self.sessions.record_emotional_state(emotional_state)
    
    def record_event(self, event_type: str, time: Optional[datetime] = None) -> None:
        """
        Record an event for pattern learning.
        
        Args:
            event_type: Type of event
            time: When it occurred (default: now)
        """
        self.expectations.record_event(event_type, time)
    
    def describe_time(self, timestamp: datetime) -> str:
        """
        Get human-readable description of a timestamp.
        
        Args:
            timestamp: Time to describe
            
        Returns:
            Relative time description
        """
        return RelativeTime.describe(timestamp)
    
    def end_session(self) -> None:
        """End the current session and archive it."""
        if self.awareness.current_session:
            self.sessions.on_session_end(self.awareness.current_session)
            self.awareness._end_session()
    
    def record_input(self, time: Optional[datetime] = None) -> None:
        """
        Record that an input was received.
        
        Args:
            time: Input time (default: now)
        """
        self._last_input_time = time or datetime.now()
        logger.debug(f"📥 Input recorded at {self._last_input_time}")
    
    def record_action(self, time: Optional[datetime] = None) -> None:
        """
        Record that an action was executed.
        
        Args:
            time: Action time (default: now)
        """
        self._last_action_time = time or datetime.now()
        logger.debug(f"⚡ Action recorded at {self._last_action_time}")
    
    def record_output(self, time: Optional[datetime] = None) -> None:
        """
        Record that an output was generated.
        
        Args:
            time: Output time (default: now)
        """
        self._last_output_time = time or datetime.now()
        logger.debug(f"📤 Output recorded at {self._last_output_time}")
    
    def get_temporal_context(self) -> Dict[str, Any]:
        """
        Get temporal context for the current cognitive cycle.
        
        This method returns temporal awareness information that should be
        passed to subsystems and included in workspace broadcasts.
        
        Returns:
            Dictionary with temporal context including:
            - cycle_timestamp: Current time
            - session_start: When current session began
            - session_duration_seconds: Duration of current session
            - time_since_last_input_seconds: Time since last input (or None)
            - time_since_last_action_seconds: Time since last action (or None)
            - time_since_last_output_seconds: Time since last output (or None)
            - cycles_this_session: Number of cycles in current session
            - session_id: Current session identifier
        """
        now = datetime.now()
        self._cycle_count += 1
        
        # Get session information
        session_start = None
        session_duration_seconds = None
        session_id = None
        
        if self.awareness.current_session:
            session_start = self.awareness.current_session.start_time
            session_duration_seconds = (now - session_start).total_seconds()
            session_id = self.awareness.current_session.id
        
        # Calculate time since events
        time_since_last_input = None
        if self._last_input_time:
            time_since_last_input = (now - self._last_input_time).total_seconds()
        
        time_since_last_action = None
        if self._last_action_time:
            time_since_last_action = (now - self._last_action_time).total_seconds()
        
        time_since_last_output = None
        if self._last_output_time:
            time_since_last_output = (now - self._last_output_time).total_seconds()
        
        return {
            "cycle_timestamp": now,
            "session_start": session_start,
            "session_duration_seconds": session_duration_seconds,
            "time_since_last_input_seconds": time_since_last_input,
            "time_since_last_action_seconds": time_since_last_action,
            "time_since_last_output_seconds": time_since_last_output,
            "cycles_this_session": self._cycle_count,
            "session_id": session_id,
        }
