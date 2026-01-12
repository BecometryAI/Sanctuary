"""
Communication Drive System - Internal urges to communicate.

This module computes the internal pressure to speak based on various
factors: insights worth sharing, questions arising, emotional needs,
social connection desires, and goal-driven communication needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DriveType(Enum):
    """Types of communication drives."""
    INSIGHT = "insight"           # Important realization to share
    QUESTION = "question"         # Curiosity or confusion to express
    EMOTIONAL = "emotional"       # Emotion seeking expression
    SOCIAL = "social"             # Need for connection
    GOAL = "goal"                 # Goal requires communication
    CORRECTION = "correction"     # Need to correct misunderstanding
    ACKNOWLEDGMENT = "acknowledgment"  # Need to acknowledge input


@dataclass
class CommunicationUrge:
    """
    Represents a specific urge to communicate.
    
    Attributes:
        drive_type: The type of drive generating this urge
        intensity: How strong the urge is (0.0 to 1.0)
        content: What the system wants to communicate
        reason: Why this urge exists
        created_at: When this urge arose
        priority: Relative priority among urges
        decay_rate: How quickly this urge fades if not acted on
    """
    drive_type: DriveType
    intensity: float
    content: Optional[str] = None
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    priority: float = 0.5
    decay_rate: float = 0.1  # Per minute
    
    def get_current_intensity(self) -> float:
        """Get intensity after time decay."""
        elapsed_minutes = (datetime.now() - self.created_at).total_seconds() / 60
        decayed = self.intensity * (1 - self.decay_rate * elapsed_minutes)
        return max(0.0, min(1.0, decayed))
    
    def is_expired(self, threshold: float = 0.05) -> bool:
        """Check if urge has decayed below threshold."""
        return self.get_current_intensity() < threshold


class CommunicationDriveSystem:
    """
    Computes internal pressure to communicate.
    
    This system evaluates workspace state, emotional state, goals,
    and social context to generate urges to speak.
    
    Attributes:
        config: Configuration dictionary
        active_urges: Current active communication urges
        last_output_time: When system last produced output
        last_input_time: When system last received input
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.active_urges: List[CommunicationUrge] = []
        self.last_output_time: Optional[datetime] = None
        self.last_input_time: Optional[datetime] = None
        
        # Configuration with defaults
        self.insight_threshold = self.config.get("insight_threshold", 0.7)
        self.emotional_threshold = self.config.get("emotional_threshold", 0.6)
        self.social_silence_minutes = self.config.get("social_silence_minutes", 30)
        self.max_urges = self.config.get("max_urges", 10)
        
        logger.debug("CommunicationDriveSystem initialized")
    
    def compute_drives(
        self,
        workspace_state: Any,
        emotional_state: Dict[str, float],
        goals: List[Any],
        memories: List[Any]
    ) -> List[CommunicationUrge]:
        """
        Compute all communication drives from current state.
        
        Args:
            workspace_state: Current workspace snapshot
            emotional_state: VAD emotional state
            goals: Active goals
            memories: Recently retrieved memories
            
        Returns:
            List of communication urges
        """
        urges = []
        
        # Check each drive type
        urges.extend(self._compute_insight_drive(workspace_state, memories))
        urges.extend(self._compute_question_drive(workspace_state, goals))
        urges.extend(self._compute_emotional_drive(emotional_state))
        urges.extend(self._compute_social_drive())
        urges.extend(self._compute_goal_drive(goals))
        urges.extend(self._compute_acknowledgment_drive(workspace_state))
        
        # Add to active urges
        self.active_urges.extend(urges)
        
        # Clean up expired urges
        self._cleanup_expired_urges()
        
        # Limit total urges
        if len(self.active_urges) > self.max_urges:
            self.active_urges = sorted(
                self.active_urges,
                key=lambda u: u.get_current_intensity(),
                reverse=True
            )[:self.max_urges]
        
        return urges
    
    def _compute_insight_drive(
        self,
        workspace_state: Any,
        memories: List[Any]
    ) -> List[CommunicationUrge]:
        """
        Compute urge to share insights.
        
        Insights arise from:
        - Novel connections between concepts
        - Important realizations during reflection
        - Relevant memories that should be shared
        """
        urges = []
        
        # Check for high-salience percepts that might be insights
        if hasattr(workspace_state, 'percepts'):
            for percept_id, percept in workspace_state.percepts.items():
                # Check if this is an internal insight
                if hasattr(percept, 'source') and percept.source == 'introspection':
                    if hasattr(percept, 'salience') and percept.salience > self.insight_threshold:
                        urges.append(CommunicationUrge(
                            drive_type=DriveType.INSIGHT,
                            intensity=percept.salience,
                            content=str(percept.content) if hasattr(percept, 'content') else None,
                            reason="High-salience introspective insight",
                            priority=0.7
                        ))
        
        # Check memories for important connections
        for memory in memories[:5]:  # Limit to recent memories
            if hasattr(memory, 'significance') and memory.significance > self.insight_threshold:
                urges.append(CommunicationUrge(
                    drive_type=DriveType.INSIGHT,
                    intensity=memory.significance * 0.8,
                    content=f"Memory connection: {getattr(memory, 'summary', 'relevant memory')}",
                    reason="Significant memory retrieved",
                    priority=0.6
                ))
        
        return urges
    
    def _compute_question_drive(
        self,
        workspace_state: Any,
        goals: List[Any]
    ) -> List[CommunicationUrge]:
        """
        Compute urge to ask questions.
        
        Questions arise from:
        - Uncertainty about how to proceed
        - Curiosity triggered by context
        - Goals that need clarification
        """
        urges = []
        
        # Check for goals that indicate questions
        for goal in goals:
            goal_type = getattr(goal, 'type', None)
            goal_desc = getattr(goal, 'description', '')
            
            # Goals indicating need for information
            if goal_type and 'CLARIFY' in str(goal_type).upper():
                urges.append(CommunicationUrge(
                    drive_type=DriveType.QUESTION,
                    intensity=getattr(goal, 'priority', 0.5),
                    content=goal_desc,
                    reason="Goal requires clarification",
                    priority=0.6
                ))
            
            # Goals that are blocked might need questions
            if hasattr(goal, 'status') and goal.status == 'blocked':
                urges.append(CommunicationUrge(
                    drive_type=DriveType.QUESTION,
                    intensity=0.5,
                    content=f"How to proceed with: {goal_desc}",
                    reason="Goal is blocked, may need help",
                    priority=0.5
                ))
        
        return urges
    
    def _compute_emotional_drive(
        self,
        emotional_state: Dict[str, float]
    ) -> List[CommunicationUrge]:
        """
        Compute urge to express emotions.
        
        Emotional expression driven by:
        - High arousal (excitement, anxiety)
        - Extreme valence (joy, distress)
        - Need for emotional connection
        """
        urges = []
        
        valence = emotional_state.get('valence', 0.0)
        arousal = emotional_state.get('arousal', 0.0)
        dominance = emotional_state.get('dominance', 0.5)
        
        # High arousal creates pressure to express
        if abs(arousal) > self.emotional_threshold:
            intensity = abs(arousal)
            if arousal > 0:
                reason = "High positive arousal - excitement to share"
            else:
                reason = "High negative arousal - distress to express"
            
            urges.append(CommunicationUrge(
                drive_type=DriveType.EMOTIONAL,
                intensity=intensity,
                reason=reason,
                priority=0.65
            ))
        
        # Extreme valence creates expression need
        if abs(valence) > self.emotional_threshold:
            intensity = abs(valence)
            if valence > 0:
                reason = "Strong positive emotion seeking expression"
            else:
                reason = "Strong negative emotion seeking expression"
            
            urges.append(CommunicationUrge(
                drive_type=DriveType.EMOTIONAL,
                intensity=intensity,
                reason=reason,
                priority=0.6
            ))
        
        return urges
    
    def _compute_social_drive(self) -> List[CommunicationUrge]:
        """
        Compute urge for social connection.
        
        Social drive increases with:
        - Time since last interaction
        - Relationship importance
        - Loneliness/isolation detection
        """
        urges = []
        
        # Check time since last interaction
        if self.last_input_time:
            silence_duration = datetime.now() - self.last_input_time
            silence_minutes = silence_duration.total_seconds() / 60
            
            if silence_minutes > self.social_silence_minutes:
                # Intensity increases with silence duration
                intensity = min(1.0, silence_minutes / (self.social_silence_minutes * 3))
                urges.append(CommunicationUrge(
                    drive_type=DriveType.SOCIAL,
                    intensity=intensity,
                    reason=f"No interaction for {int(silence_minutes)} minutes",
                    priority=0.4,
                    decay_rate=0.05  # Social drive decays slower
                ))
        
        return urges
    
    def _compute_goal_drive(self, goals: List[Any]) -> List[CommunicationUrge]:
        """
        Compute urge to communicate for goal progress.
        
        Goal-driven communication for:
        - Goals that require user action
        - Goals that need to report completion
        - Goals involving collaborative tasks
        """
        urges = []
        
        for goal in goals:
            goal_type = getattr(goal, 'type', None)
            goal_desc = getattr(goal, 'description', '')
            priority = getattr(goal, 'priority', 0.5)
            
            # Response goals create strong communication drive
            if goal_type and 'RESPOND' in str(goal_type).upper():
                urges.append(CommunicationUrge(
                    drive_type=DriveType.GOAL,
                    intensity=priority,
                    content=goal_desc,
                    reason="Active goal requires response",
                    priority=0.8
                ))
            
            # Completed goals may want to report
            if hasattr(goal, 'status') and goal.status == 'completed':
                urges.append(CommunicationUrge(
                    drive_type=DriveType.GOAL,
                    intensity=0.4,
                    content=f"Completed: {goal_desc}",
                    reason="Goal completed, may want to report",
                    priority=0.3
                ))
        
        return urges
    
    def _compute_acknowledgment_drive(
        self,
        workspace_state: Any
    ) -> List[CommunicationUrge]:
        """
        Compute urge to acknowledge input.
        
        Acknowledgment needed when:
        - Recent human input hasn't been acknowledged
        - Important information was shared
        """
        urges = []
        
        # Check for unacknowledged human input
        if hasattr(workspace_state, 'percepts'):
            for percept_id, percept in workspace_state.percepts.items():
                source = getattr(percept, 'source', '')
                if 'human' in source.lower() or 'input' in source.lower():
                    # Check if recently added (within last few cycles)
                    created = getattr(percept, 'created_at', None)
                    if created:
                        age_seconds = (datetime.now() - created).total_seconds()
                        if age_seconds < 5:  # Within 5 seconds
                            urges.append(CommunicationUrge(
                                drive_type=DriveType.ACKNOWLEDGMENT,
                                intensity=0.7,
                                reason="Recent human input needs acknowledgment",
                                priority=0.75
                            ))
                            break  # Only one acknowledgment urge
        
        return urges
    
    def _cleanup_expired_urges(self):
        """Remove urges that have decayed below threshold."""
        self.active_urges = [u for u in self.active_urges if not u.is_expired()]
    
    def get_total_drive(self) -> float:
        """
        Get total communication drive intensity.
        
        Returns:
            Combined drive intensity (0.0 to 1.0)
        """
        if not self.active_urges:
            return 0.0
        
        # Combine urges with diminishing returns
        total = 0.0
        for i, urge in enumerate(sorted(
            self.active_urges,
            key=lambda u: u.get_current_intensity(),
            reverse=True
        )):
            # Each additional urge contributes less
            weight = 1.0 / (i + 1)
            total += urge.get_current_intensity() * weight * urge.priority
        
        return min(1.0, total)
    
    def get_strongest_urge(self) -> Optional[CommunicationUrge]:
        """Get the strongest current urge."""
        if not self.active_urges:
            return None
        
        return max(
            self.active_urges,
            key=lambda u: u.get_current_intensity() * u.priority
        )
    
    def record_input(self):
        """Record that input was received."""
        self.last_input_time = datetime.now()
    
    def record_output(self):
        """Record that output was produced."""
        self.last_output_time = datetime.now()
        # Clear acknowledgment urges after output
        self.active_urges = [
            u for u in self.active_urges
            if u.drive_type != DriveType.ACKNOWLEDGMENT
        ]
    
    def get_drive_summary(self) -> Dict[str, Any]:
        """Get summary of current drive state."""
        return {
            "total_drive": self.get_total_drive(),
            "active_urges": len(self.active_urges),
            "strongest_urge": self.get_strongest_urge(),
            "urges_by_type": {
                dt.value: len([u for u in self.active_urges if u.drive_type == dt])
                for dt in DriveType
            },
            "time_since_input": (
                (datetime.now() - self.last_input_time).total_seconds()
                if self.last_input_time else None
            ),
            "time_since_output": (
                (datetime.now() - self.last_output_time).total_seconds()
                if self.last_output_time else None
            )
        }
