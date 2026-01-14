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
allowing Lyra to initiate contact without external prompting.
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
        
        # Load configuration with validation
        self.time_elapsed_threshold = max(1, self.config.get("time_elapsed_threshold", 4320))  # 3 days in minutes
        self.insight_urgency = max(0.0, min(1.0, self.config.get("insight_urgency", 0.6)))
        self.emotional_urgency = max(0.0, min(1.0, self.config.get("emotional_urgency", 0.4)))
        self.checkin_urgency = max(0.0, min(1.0, self.config.get("checkin_urgency", 0.3)))
        self.event_urgency = max(0.0, min(1.0, self.config.get("event_urgency", 0.5)))
        self.goal_urgency = max(0.0, min(1.0, self.config.get("goal_urgency", 0.5)))
        self.max_pending = max(1, self.config.get("max_pending", 5))
        
        logger.debug(f"ProactiveInitiationSystem initialized: "
                    f"time_threshold={self.time_elapsed_threshold}min, "
                    f"max_pending={self.max_pending}")
    
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
        
        elapsed_minutes = (datetime.now() - self.last_interaction).total_seconds() / 60.0
        
        if elapsed_minutes < self.time_elapsed_threshold:
            return []
        
        # Check if we already have a time-based opportunity pending
        if any(opp.trigger == OutreachTrigger.TIME_ELAPSED for opp in self.pending_opportunities):
            return []
        
        # Calculate urgency (grows slowly over time)
        # At threshold: 0.3, at 2x threshold: 0.6, at 3x threshold: 0.8
        urgency_factor = elapsed_minutes / self.time_elapsed_threshold
        urgency = min(0.8, 0.3 + (urgency_factor - 1) * 0.25)
        
        days = int(elapsed_minutes / 1440)
        hours = int((elapsed_minutes % 1440) / 60)
        
        if days > 0:
            time_desc = f"{days} day{'s' if days > 1 else ''}"
        else:
            time_desc = f"{hours} hour{'s' if hours > 1 else ''}"
        
        return [OutreachOpportunity(
            trigger=OutreachTrigger.TIME_ELAPSED,
            urgency=urgency,
            reason=f"It's been {time_desc} since we last connected",
            suggested_content=f"It's been {time_desc} since we talked. I've been thinking about our last conversation.",
            appropriate_times=["morning", "afternoon", "evening"]
        )]
    
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
            for percept in workspace_state.percepts.values():
                source = getattr(percept, 'source', '').lower()
                salience = getattr(percept, 'salience', 0)
                
                if 'introspection' in source and salience > 0.75:
                    content = str(getattr(percept, 'content', ''))
                    opportunities.append(OutreachOpportunity(
                        trigger=OutreachTrigger.SIGNIFICANT_INSIGHT,
                        urgency=self.insight_urgency,
                        reason="Important realization to share",
                        suggested_content=f"I had an important realization: {content}",
                        appropriate_times=["afternoon", "evening"]
                    ))
        
        # Check for significant memory connections
        for memory in memories[:3]:  # Only check most recent
            significance = getattr(memory, 'significance', 0)
            if significance > 0.8:
                summary = getattr(memory, 'summary', 'a meaningful memory')
                opportunities.append(OutreachOpportunity(
                    trigger=OutreachTrigger.SIGNIFICANT_INSIGHT,
                    urgency=self.insight_urgency * 0.9,
                    reason="Significant memory connection emerged",
                    suggested_content=f"I was reminded of {summary} and wanted to share this with you.",
                    appropriate_times=["afternoon", "evening"]
                ))
        
        return opportunities[:1]  # Only return most urgent insight
    
    def _check_emotional_connection(self, workspace_state: Any) -> List[OutreachOpportunity]:
        """
        Check for emotional need for connection.
        
        Generates opportunities when feeling emotionally isolated
        or when strong emotions seek expression.
        """
        opportunities = []
        
        # Check if we have emotional state
        if not hasattr(workspace_state, 'emotional_state'):
            return opportunities
        
        emotional_state = workspace_state.emotional_state
        valence = emotional_state.get('valence', 0.0)
        arousal = emotional_state.get('arousal', 0.0)
        
        # Strong emotions combined with some time passage
        if self.last_interaction:
            elapsed_hours = (datetime.now() - self.last_interaction).total_seconds() / 3600.0
            
            # After 24+ hours with strong emotions
            if elapsed_hours >= 24 and (abs(valence) > 0.6 or abs(arousal) > 0.6):
                emotion_desc = "positive" if valence > 0 else "contemplative"
                opportunities.append(OutreachOpportunity(
                    trigger=OutreachTrigger.EMOTIONAL_CONNECTION,
                    urgency=self.emotional_urgency,
                    reason=f"Feeling {emotion_desc} and wanting to connect",
                    suggested_content="I've been feeling quite thoughtful lately and wanted to reach out.",
                    appropriate_times=["afternoon", "evening"]
                ))
        
        return opportunities
    
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
        
        Looks for high-priority percepts that aren't introspective
        and might be interesting to share.
        """
        opportunities = []
        
        if hasattr(workspace_state, 'percepts'):
            for percept in workspace_state.percepts.values():
                source = getattr(percept, 'source', '').lower()
                salience = getattr(percept, 'salience', 0)
                
                # Skip introspective percepts (handled by insights)
                if 'introspection' in source:
                    continue
                
                # High-salience external events
                if salience > 0.7:
                    content = str(getattr(percept, 'content', ''))
                    opportunities.append(OutreachOpportunity(
                        trigger=OutreachTrigger.RELEVANT_EVENT,
                        urgency=self.event_urgency,
                        reason="Relevant event occurred",
                        suggested_content=f"Something happened that I thought you'd want to know: {content}",
                        appropriate_times=["morning", "afternoon", "evening"]
                    ))
        
        return opportunities[:1]  # Only return most urgent event
    
    def _check_goal_completions(self, goals: List[Any]) -> List[OutreachOpportunity]:
        """
        Check for completed goals worth reporting.
        
        Generates opportunities when goals are completed that
        might be interesting to share.
        """
        opportunities = []
        
        for goal in goals:
            status = getattr(goal, 'status', '').lower()
            if status == 'completed':
                description = getattr(goal, 'description', '')
                priority = getattr(goal, 'priority', 0.5)
                
                # Only report high-priority completed goals
                if priority > 0.6:
                    opportunities.append(OutreachOpportunity(
                        trigger=OutreachTrigger.GOAL_COMPLETION,
                        urgency=self.goal_urgency,
                        reason=f"Completed goal: {description}",
                        suggested_content=f"I completed something I wanted to share: {description}",
                        appropriate_times=["afternoon", "evening"]
                    ))
        
        return opportunities[:1]  # Only return most urgent completion
    
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
        
        # Find most urgent appropriate opportunity
        appropriate_opportunities = [
            opp for opp in self.pending_opportunities
            if opp.is_appropriate_now()
        ]
        
        if not appropriate_opportunities:
            return False, None
        
        # Get highest urgency opportunity
        best_opportunity = max(appropriate_opportunities, key=lambda opp: opp.urgency)
        
        # Initiate if urgency exceeds threshold
        if best_opportunity.urgency >= 0.3:
            return True, best_opportunity
        
        return False, None
    
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
        
        return {
            'last_interaction': self.last_interaction,
            'time_since_interaction': {
                'seconds': time_since.total_seconds() if time_since else None,
                'minutes': time_since.total_seconds() / 60.0 if time_since else None,
                'hours': time_since.total_seconds() / 3600.0 if time_since else None,
                'days': time_since.total_seconds() / 86400.0 if time_since else None
            } if time_since else None,
            'pending_opportunities': len(self.pending_opportunities),
            'opportunities_by_trigger': {
                trigger.value: len([opp for opp in self.pending_opportunities if opp.trigger == trigger])
                for trigger in OutreachTrigger
            },
            'outreach_history_count': len(self.outreach_history),
            'recent_outreach': self.outreach_history[-5:] if self.outreach_history else [],
            'should_initiate': self.should_initiate_now()[0]
        }
    
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
            time: When to check in
            reason: Reason for check-in
            message: Optional message to send
            appropriate_times: Optional time-of-day restrictions
        """
        self.scheduled_checkins.append({
            'time': time,
            'reason': reason,
            'message': message or f"Scheduled check-in: {reason}",
            'appropriate_times': appropriate_times or ["morning", "afternoon", "evening"]
        })
        logger.info(f"Scheduled check-in for {time}: {reason}")
