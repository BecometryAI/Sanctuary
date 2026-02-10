"""
Proactive Session Initiation System - Autonomous outreach capability.

This module enables proactive communication initiation based on:
- Time elapsed since last interaction
- Significant insights or events
- Emotional connection needs
- Scheduled check-ins
- Relevant events
- Goal completions

The system generates OutreachOpportunities that feed into the drive system,
allowing Sanctuary to initiate contact without external prompting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class OutreachTrigger(Enum):
    """Types of proactive outreach triggers."""
    TIME_ELAPSED = "time_elapsed"
    SIGNIFICANT_INSIGHT = "significant_insight"
    EMOTIONAL_CONNECTION = "emotional_connection"
    SCHEDULED_CHECKIN = "scheduled_checkin"
    RELEVANT_EVENT = "relevant_event"
    GOAL_COMPLETION = "goal_completion"


@dataclass
class OutreachOpportunity:
    """
    Represents a reason to proactively reach out.
    
    Attributes:
        trigger: The type of trigger generating this opportunity
        urgency: How urgent the outreach is (0.0 to 1.0)
        reason: Human-readable explanation for the outreach
        suggested_content: Optional suggested message content
        appropriate_times: Times when this outreach is appropriate (e.g., ["morning", "evening"])
        created_at: When this opportunity was identified
    """
    trigger: OutreachTrigger
    urgency: float
    reason: str
    suggested_content: Optional[str] = None
    appropriate_times: List[str] = field(default_factory=lambda: ["morning", "afternoon", "evening"])
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_appropriate_now(self) -> bool:
        """
        Check if current time is appropriate for this outreach.
        
        Returns:
            True if current hour matches appropriate times
        """
        if not self.appropriate_times:
            return True  # No restrictions
        
        current_hour = datetime.now().hour
        
        # Define time windows
        time_windows = {
            "morning": (6, 12),
            "afternoon": (12, 18),
            "evening": (18, 23),
            "night": (0, 6)
        }
        
        for time_period in self.appropriate_times:
            if time_period.lower() in time_windows:
                start, end = time_windows[time_period.lower()]
                if start <= current_hour < end:
                    return True
        
        return False


class ProactiveInitiationSystem:
    """
    System for proactive communication initiation.
    
    Monitors workspace state, goals, memories, and time passage to identify
    opportunities for proactive outreach. Generates OutreachOpportunities that
    can feed into the communication drive system.
    
    Attributes:
        config: Configuration dictionary
        last_interaction: Timestamp of last interaction (input or output)
        pending_opportunities: List of identified outreach opportunities
        outreach_history: History of outreach attempts
        scheduled_checkins: Scheduled check-in times
    """
    
    # Configuration defaults
    DEFAULT_TIME_THRESHOLD = 4320  # 3 days in minutes
    DEFAULT_INSIGHT_URGENCY = 0.6
    DEFAULT_EMOTIONAL_URGENCY = 0.4
    DEFAULT_CHECKIN_URGENCY = 0.3
    DEFAULT_EVENT_URGENCY = 0.5
    DEFAULT_GOAL_URGENCY = 0.5
    DEFAULT_MAX_PENDING = 5
    
    # Thresholds for trigger detection
    INSIGHT_SALIENCE_THRESHOLD = 0.75
    MEMORY_SIGNIFICANCE_THRESHOLD = 0.8
    EVENT_SALIENCE_THRESHOLD = 0.7
    EMOTIONAL_HOURS_THRESHOLD = 24
    EMOTIONAL_INTENSITY_THRESHOLD = 0.6
    GOAL_PRIORITY_THRESHOLD = 0.6
    URGENCY_THRESHOLD = 0.3
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize proactive initiation system.
        
        Args:
            config: Optional configuration dict with keys:
                - time_elapsed_threshold: Minutes before time-based outreach (default: 4320 = 3 days)
                - insight_urgency: Urgency for significant insights (default: 0.6)
                - emotional_urgency: Urgency for emotional connection (default: 0.4)
                - checkin_urgency: Urgency for scheduled check-ins (default: 0.3)
                - event_urgency: Urgency for relevant events (default: 0.5)
                - goal_urgency: Urgency for goal completions (default: 0.5)
                - max_pending: Maximum pending opportunities (default: 5)
        """
        self.config = config or {}
        self.last_interaction: Optional[datetime] = None
        self.pending_opportunities: List[OutreachOpportunity] = []
        self.outreach_history: List[Dict[str, Any]] = []
        self.scheduled_checkins: List[Dict[str, Any]] = []
        
        # Load and validate configuration
        self.time_elapsed_threshold = self._clamp_int(
            self.config.get("time_elapsed_threshold", self.DEFAULT_TIME_THRESHOLD), min_val=1
        )
        self.insight_urgency = self._clamp_urgency(
            self.config.get("insight_urgency", self.DEFAULT_INSIGHT_URGENCY)
        )
        self.emotional_urgency = self._clamp_urgency(
            self.config.get("emotional_urgency", self.DEFAULT_EMOTIONAL_URGENCY)
        )
        self.checkin_urgency = self._clamp_urgency(
            self.config.get("checkin_urgency", self.DEFAULT_CHECKIN_URGENCY)
        )
        self.event_urgency = self._clamp_urgency(
            self.config.get("event_urgency", self.DEFAULT_EVENT_URGENCY)
        )
        self.goal_urgency = self._clamp_urgency(
            self.config.get("goal_urgency", self.DEFAULT_GOAL_URGENCY)
        )
        self.max_pending = self._clamp_int(
            self.config.get("max_pending", self.DEFAULT_MAX_PENDING), min_val=1
        )
        
        logger.debug(f"ProactiveInitiationSystem initialized: "
                    f"time_threshold={self.time_elapsed_threshold}min, "
                    f"max_pending={self.max_pending}")
    
    @staticmethod
    def _clamp_urgency(value: float) -> float:
        """Clamp urgency value to [0.0, 1.0] range."""
        return max(0.0, min(1.0, float(value)))
    
    @staticmethod
    def _clamp_int(value: int, min_val: int = 1) -> int:
        """Clamp integer value to minimum."""
        return max(min_val, int(value))
    
    def check_for_opportunities(
        self,
        workspace_state: Any,
        memories: List[Any],
        goals: List[Any]
    ) -> List[OutreachOpportunity]:
        """
        Check for reasons to proactively reach out.
        
        Evaluates all trigger types and generates opportunities based on
        current state. Maintains pending opportunities list with size limit.
        
        Args:
            workspace_state: Current workspace snapshot with percepts
            memories: Recently retrieved memory objects
            goals: Active goal objects
            
        Returns:
            List of newly identified outreach opportunities
        """
        new_opportunities = []
        
        # Check all trigger types
        new_opportunities.extend(self._check_time_elapsed())
        new_opportunities.extend(self._check_significant_insights(workspace_state, memories))
        new_opportunities.extend(self._check_emotional_connection(workspace_state))
        new_opportunities.extend(self._check_scheduled_checkins())
        new_opportunities.extend(self._check_relevant_events(workspace_state))
        new_opportunities.extend(self._check_goal_completions(goals))
        
        # Add to pending and maintain limit
        self.pending_opportunities.extend(new_opportunities)
        self._limit_pending_opportunities()
        
        return new_opportunities
    
    def _check_time_elapsed(self) -> List[OutreachOpportunity]:
        """
        Check if enough time has elapsed since last interaction.
        
        Returns opportunity when silence exceeds threshold, with urgency
        increasing based on how long it's been.
        """
        if not self.last_interaction:
            return []
        
        now = datetime.now()
        elapsed_minutes = (now - self.last_interaction).total_seconds() / 60.0
        
        if elapsed_minutes < self.time_elapsed_threshold:
            return []
        
        # Check if we already have a time-based opportunity pending
        if any(opp.trigger == OutreachTrigger.TIME_ELAPSED for opp in self.pending_opportunities):
            return []
        
        # Calculate urgency (grows slowly over time)
        urgency = self._calculate_time_urgency(elapsed_minutes)
        time_desc = self._format_elapsed_time(elapsed_minutes)
        
        return [OutreachOpportunity(
            trigger=OutreachTrigger.TIME_ELAPSED,
            urgency=urgency,
            reason=f"It's been {time_desc} since we last connected",
            suggested_content=f"It's been {time_desc} since we talked. I've been thinking about our last conversation.",
            appropriate_times=["morning", "afternoon", "evening"]
        )]
    
    def _calculate_time_urgency(self, elapsed_minutes: float) -> float:
        """Calculate urgency based on elapsed time with gradual escalation."""
        urgency_factor = elapsed_minutes / self.time_elapsed_threshold
        # At threshold: 0.3, at 2x threshold: 0.55, at 3x threshold: 0.8
        urgency = 0.3 + (urgency_factor - 1) * 0.25
        return min(0.8, urgency)
    
    def _format_elapsed_time(self, elapsed_minutes: float) -> str:
        """Format elapsed time in human-readable format."""
        days = int(elapsed_minutes / 1440)
        hours = int((elapsed_minutes % 1440) / 60)
        
        if days > 0:
            return f"{days} day{'s' if days > 1 else ''}"
        return f"{hours} hour{'s' if hours > 1 else ''}"
    
    def _check_significant_insights(
        self,
        workspace_state: Any,
        memories: List[Any]
    ) -> List[OutreachOpportunity]:
        """
        Check for significant insights worth sharing proactively.
        
        Looks for high-salience introspective percepts or important
        memory connections that warrant reaching out.
        """
        opportunities = []
        
        # Check workspace percepts for significant insights
        if hasattr(workspace_state, 'percepts'):
            opportunities.extend(self._extract_insight_opportunities(workspace_state.percepts))
        
        # Check for significant memory connections (limit to 3 most recent)
        opportunities.extend(self._extract_memory_opportunities(memories[:3]))
        
        # Return only the most urgent insight to avoid overwhelming
        return opportunities[:1]
    
    def _extract_insight_opportunities(self, percepts: Dict) -> List[OutreachOpportunity]:
        """Extract insight opportunities from workspace percepts."""
        opportunities = []
        for percept in percepts.values():
            source = getattr(percept, 'source', '').lower()
            salience = getattr(percept, 'salience', 0)
            
            if 'introspection' in source and salience > self.INSIGHT_SALIENCE_THRESHOLD:
                content = str(getattr(percept, 'content', ''))
                opportunities.append(OutreachOpportunity(
                    trigger=OutreachTrigger.SIGNIFICANT_INSIGHT,
                    urgency=self.insight_urgency,
                    reason="Important realization to share",
                    suggested_content=f"I had an important realization: {content}",
                    appropriate_times=["afternoon", "evening"]
                ))
        return opportunities
    
    def _extract_memory_opportunities(self, memories: List[Any]) -> List[OutreachOpportunity]:
        """Extract insight opportunities from significant memories."""
        opportunities = []
        for memory in memories:
            significance = getattr(memory, 'significance', 0)
            if significance > self.MEMORY_SIGNIFICANCE_THRESHOLD:
                summary = getattr(memory, 'summary', 'a meaningful memory')
                opportunities.append(OutreachOpportunity(
                    trigger=OutreachTrigger.SIGNIFICANT_INSIGHT,
                    urgency=self.insight_urgency * 0.9,
                    reason="Significant memory connection emerged",
                    suggested_content=f"I was reminded of {summary} and wanted to share this with you.",
                    appropriate_times=["afternoon", "evening"]
                ))
        return opportunities
    
    def _check_emotional_connection(self, workspace_state: Any) -> List[OutreachOpportunity]:
        """
        Check for emotional need for connection.
        
        Generates opportunities when experiencing strong emotions
        combined with sufficient time passage.
        """
        if not hasattr(workspace_state, 'emotional_state'):
            return []
        
        if not self.last_interaction:
            return []
        
        emotional_state = workspace_state.emotional_state
        valence = emotional_state.get('valence', 0.0)
        arousal = emotional_state.get('arousal', 0.0)
        
        # Calculate elapsed time once
        now = datetime.now()
        elapsed_hours = (now - self.last_interaction).total_seconds() / 3600.0
        
        # Require both time passage and strong emotions
        if elapsed_hours < self.EMOTIONAL_HOURS_THRESHOLD:
            return []
        
        if abs(valence) <= self.EMOTIONAL_INTENSITY_THRESHOLD and abs(arousal) <= self.EMOTIONAL_INTENSITY_THRESHOLD:
            return []
        
        emotion_desc = "positive" if valence > 0 else "contemplative"
        return [OutreachOpportunity(
            trigger=OutreachTrigger.EMOTIONAL_CONNECTION,
            urgency=self.emotional_urgency,
            reason=f"Feeling {emotion_desc} and wanting to connect",
            suggested_content="I've been feeling quite thoughtful lately and wanted to reach out.",
            appropriate_times=["afternoon", "evening"]
        )]
    
    def _check_scheduled_checkins(self) -> List[OutreachOpportunity]:
        """
        Check for scheduled check-in times.
        
        Evaluates scheduled_checkins list for any due check-ins.
        """
        opportunities = []
        now = datetime.now()
        
        for checkin in self.scheduled_checkins:
            scheduled_time = checkin.get('time')
            if not scheduled_time or not isinstance(scheduled_time, datetime):
                continue
            
            # Check if scheduled time has passed
            if scheduled_time <= now:
                opportunities.append(OutreachOpportunity(
                    trigger=OutreachTrigger.SCHEDULED_CHECKIN,
                    urgency=self.checkin_urgency,
                    reason=checkin.get('reason', 'Scheduled check-in time'),
                    suggested_content=checkin.get('message', 'Time for our scheduled check-in!'),
                    appropriate_times=checkin.get('appropriate_times', ["morning", "afternoon", "evening"])
                ))
        
        # Remove processed check-ins
        self.scheduled_checkins = [
            c for c in self.scheduled_checkins
            if c.get('time') and c['time'] > now
        ]
        
        return opportunities
    
    def _check_relevant_events(self, workspace_state: Any) -> List[OutreachOpportunity]:
        """
        Check for relevant events worth sharing.
        
        Looks for high-salience external events (non-introspective)
        that might be interesting to share.
        """
        if not hasattr(workspace_state, 'percepts'):
            return []
        
        opportunities = []
        for percept in workspace_state.percepts.values():
            source = getattr(percept, 'source', '').lower()
            salience = getattr(percept, 'salience', 0)
            
            # Skip introspective percepts (handled by insights)
            if 'introspection' in source:
                continue
            
            # High-salience external events
            if salience > self.EVENT_SALIENCE_THRESHOLD:
                content = str(getattr(percept, 'content', ''))
                opportunities.append(OutreachOpportunity(
                    trigger=OutreachTrigger.RELEVANT_EVENT,
                    urgency=self.event_urgency,
                    reason="Relevant event occurred",
                    suggested_content=f"Something happened that I thought you'd want to know: {content}",
                    appropriate_times=["morning", "afternoon", "evening"]
                ))
        
        # Return only the most urgent event to avoid overwhelming
        return opportunities[:1]
    
    def _check_goal_completions(self, goals: List[Any]) -> List[OutreachOpportunity]:
        """
        Check for completed goals worth reporting.
        
        Generates opportunities for high-priority completed goals
        that might be interesting to share.
        """
        opportunities = []
        
        for goal in goals:
            status = getattr(goal, 'status', '').lower()
            if status != 'completed':
                continue
            
            priority = getattr(goal, 'priority', 0.0)
            if priority <= self.GOAL_PRIORITY_THRESHOLD:
                continue
            
            description = getattr(goal, 'description', '')
            opportunities.append(OutreachOpportunity(
                trigger=OutreachTrigger.GOAL_COMPLETION,
                urgency=self.goal_urgency,
                reason=f"Completed goal: {description}",
                suggested_content=f"I completed something I wanted to share: {description}",
                appropriate_times=["afternoon", "evening"]
            ))
        
        # Return only the most urgent completion to avoid overwhelming
        return opportunities[:1]
    
    def _limit_pending_opportunities(self) -> None:
        """Keep only the most urgent pending opportunities up to max limit."""
        if len(self.pending_opportunities) > self.max_pending:
            self.pending_opportunities.sort(key=lambda opp: opp.urgency, reverse=True)
            self.pending_opportunities = self.pending_opportunities[:self.max_pending]
    
    def should_initiate_now(self) -> Tuple[bool, Optional[OutreachOpportunity]]:
        """
        Determine if now is a good time to initiate contact.
        
        Evaluates pending opportunities against current time appropriateness
        and urgency thresholds.
        
        Returns:
            Tuple of (should_initiate, opportunity_to_use)
        """
        if not self.pending_opportunities:
            return False, None
        
        # Filter to time-appropriate opportunities
        appropriate_opportunities = [
            opp for opp in self.pending_opportunities
            if opp.is_appropriate_now()
        ]
        
        if not appropriate_opportunities:
            return False, None
        
        # Select highest urgency opportunity
        best_opportunity = max(appropriate_opportunities, key=lambda opp: opp.urgency)
        
        # Initiate only if urgency exceeds threshold
        return (best_opportunity.urgency >= self.URGENCY_THRESHOLD, best_opportunity)
    
    def record_interaction(self) -> None:
        """
        Record that an interaction occurred.
        
        Updates last_interaction timestamp for time-based tracking.
        """
        self.last_interaction = datetime.now()
        logger.debug(f"Interaction recorded at {self.last_interaction}")
    
    def record_outreach(self, opportunity: OutreachOpportunity, success: bool) -> None:
        """
        Record an outreach attempt.
        
        Args:
            opportunity: The opportunity that led to outreach
            success: Whether the outreach was successful
        """
        self.outreach_history.append({
            'timestamp': datetime.now(),
            'trigger': opportunity.trigger.value,
            'urgency': opportunity.urgency,
            'reason': opportunity.reason,
            'success': success
        })
        
        # Remove from pending
        self.pending_opportunities = [
            opp for opp in self.pending_opportunities
            if opp is not opportunity
        ]
        
        logger.info(f"Outreach recorded: trigger={opportunity.trigger.value}, "
                   f"success={success}, reason={opportunity.reason}")
    
    def get_time_since_interaction(self) -> Optional[timedelta]:
        """
        Get time since last interaction.
        
        Returns:
            timedelta since last interaction, or None if no interactions yet
        """
        if not self.last_interaction:
            return None
        
        return datetime.now() - self.last_interaction
    
    def get_outreach_summary(self) -> Dict[str, Any]:
        """
        Get summary of proactive outreach state.
        
        Returns:
            Dictionary with current state information
        """
        time_since = self.get_time_since_interaction()
        should_initiate, _ = self.should_initiate_now()
        
        summary = {
            'last_interaction': self.last_interaction,
            'pending_opportunities': len(self.pending_opportunities),
            'opportunities_by_trigger': {
                trigger.value: sum(1 for opp in self.pending_opportunities if opp.trigger == trigger)
                for trigger in OutreachTrigger
            },
            'outreach_history_count': len(self.outreach_history),
            'recent_outreach': self.outreach_history[-5:] if self.outreach_history else [],
            'should_initiate': should_initiate
        }
        
        # Add time breakdown only if interaction exists
        if time_since:
            total_seconds = time_since.total_seconds()
            summary['time_since_interaction'] = {
                'seconds': total_seconds,
                'minutes': total_seconds / 60.0,
                'hours': total_seconds / 3600.0,
                'days': total_seconds / 86400.0
            }
        else:
            summary['time_since_interaction'] = None
        
        return summary
    
    def schedule_checkin(
        self,
        time: datetime,
        reason: str = "Scheduled check-in",
        message: Optional[str] = None,
        appropriate_times: Optional[List[str]] = None
    ) -> None:
        """
        Schedule a future check-in.
        
        Args:
            time: When to check in (must be datetime object)
            reason: Reason for check-in
            message: Optional message to send
            appropriate_times: Optional time-of-day restrictions
            
        Raises:
            TypeError: If time is not a datetime object
            ValueError: If time is in the past
        """
        if not isinstance(time, datetime):
            raise TypeError(f"time must be datetime object, got {type(time)}")
        
        if time < datetime.now():
            logger.warning(f"Scheduled check-in time {time} is in the past")
        
        self.scheduled_checkins.append({
            'time': time,
            'reason': reason,
            'message': message or f"Scheduled check-in: {reason}",
            'appropriate_times': appropriate_times or ["morning", "afternoon", "evening"]
        })
        logger.info(f"Scheduled check-in for {time}: {reason}")
