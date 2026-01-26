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
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .proactive import ProactiveInitiationSystem

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
        """
        Get intensity after time decay.
        
        Uses linear decay: intensity * (1 - decay_rate * elapsed_minutes).
        Clamped to [0.0, 1.0] range.
        """
        elapsed_minutes = (datetime.now() - self.created_at).total_seconds() / 60.0
        decayed = self.intensity * (1.0 - self.decay_rate * elapsed_minutes)
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
        """
        Initialize communication drive system.
        
        Args:
            config: Optional configuration dict with keys:
                - insight_threshold: Min salience for insight drives (default: 0.7)
                - emotional_threshold: Min arousal/valence for emotional drives (default: 0.6)
                - social_silence_minutes: Silence duration before social drive (default: 30)
                - max_urges: Maximum active urges to track (default: 10)
                - enable_proactive: Enable proactive initiation system (default: True)
        """
        self.config = config or {}
        self.active_urges: List[CommunicationUrge] = []
        self.last_output_time: Optional[datetime] = None
        self.last_input_time: Optional[datetime] = None
        
        # Load and validate configuration
        self.insight_threshold = max(0.0, min(1.0, self.config.get("insight_threshold", 0.7)))
        self.emotional_threshold = max(0.0, min(1.0, self.config.get("emotional_threshold", 0.6)))
        self.social_silence_minutes = max(1, self.config.get("social_silence_minutes", 30))
        self.max_urges = max(1, self.config.get("max_urges", 10))
        
        # Initialize proactive system if enabled
        self.proactive_system: Optional[ProactiveInitiationSystem] = None
        if self.config.get("enable_proactive", True):
            try:
                from .proactive import ProactiveInitiationSystem
            except ImportError:
                from proactive import ProactiveInitiationSystem
            self.proactive_system = ProactiveInitiationSystem(self.config.get("proactive_config", {}))
        
        
        logger.debug(f"CommunicationDriveSystem initialized: "
                    f"insight_threshold={self.insight_threshold:.2f}, "
                    f"emotional_threshold={self.emotional_threshold:.2f}, "
                    f"social_silence_minutes={self.social_silence_minutes}, "
                    f"max_urges={self.max_urges}, "
                    f"proactive_enabled={self.proactive_system is not None}")
    
    def compute_drives(
        self,
        workspace_state: Any,
        emotional_state: Dict[str, float],
        goals: List[Any],
        memories: List[Any]
    ) -> List[CommunicationUrge]:
        """
        Compute all communication drives from current state.
        
        Evaluates 6 drive types, checks proactive opportunities, and manages 
        active urge list with decay cleanup.
        
        Args:
            workspace_state: Current workspace snapshot with percepts
            emotional_state: VAD emotional state dict (valence, arousal, dominance)
            goals: Active goal objects with type, description, priority attributes
            memories: Recently retrieved memory objects with significance attribute
            
        Returns:
            List of newly generated communication urges
        """
        # Compute all drive types in order of importance
        new_urges = []
        new_urges.extend(self._compute_goal_drive(goals))
        new_urges.extend(self._compute_acknowledgment_drive(workspace_state))
        new_urges.extend(self._compute_emotional_drive(emotional_state))
        new_urges.extend(self._compute_insight_drive(workspace_state, memories))
        new_urges.extend(self._compute_question_drive(workspace_state, goals))
        new_urges.extend(self._compute_social_drive())
        
        # Check proactive opportunities if system is enabled
        if self.proactive_system:
            new_urges.extend(self._compute_proactive_drive(workspace_state, memories, goals))
        
        # Add to active urges and maintain size limit
        self.active_urges.extend(new_urges)
        self._cleanup_expired_urges()
        self._limit_active_urges()
        
        return new_urges
    
    def _limit_active_urges(self) -> None:
        """Keep only the strongest urges up to max_urges limit."""
        if len(self.active_urges) > self.max_urges:
            self.active_urges.sort(
                key=lambda u: u.get_current_intensity() * u.priority,
                reverse=True
            )
            self.active_urges = self.active_urges[:self.max_urges]
    
    def _compute_insight_drive(
        self,
        workspace_state: Any,
        memories: List[Any]
    ) -> List[CommunicationUrge]:
        """
        Compute urge to share insights from introspection and significant memories.
        
        Returns urges when:
        - Introspective percepts exceed salience threshold
        - Retrieved memories exceed significance threshold
        """
        urges = []
        
        # Check workspace percepts for high-salience insights
        if hasattr(workspace_state, 'percepts'):
            urges.extend(self._check_introspective_percepts(workspace_state.percepts))
        
        # Check recently retrieved memories (limit to 5 most recent)
        urges.extend(self._check_significant_memories(memories[:5]))
        
        return urges
    
    def _check_introspective_percepts(self, percepts: Dict) -> List[CommunicationUrge]:
        """Extract insight urges from introspective percepts."""
        urges = []
        for percept in percepts.values():
            if (getattr(percept, 'source', None) == 'introspection' and
                getattr(percept, 'salience', 0) > self.insight_threshold):
                urges.append(CommunicationUrge(
                    drive_type=DriveType.INSIGHT,
                    intensity=percept.salience,
                    content=str(getattr(percept, 'content', None)),
                    reason="High-salience introspective insight",
                    priority=0.7
                ))
        return urges
    
    def _check_significant_memories(self, memories: List[Any]) -> List[CommunicationUrge]:
        """Extract insight urges from significant memories."""
        urges = []
        for memory in memories:
            significance = getattr(memory, 'significance', 0)
            if significance > self.insight_threshold:
                urges.append(CommunicationUrge(
                    drive_type=DriveType.INSIGHT,
                    intensity=significance * 0.8,
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
        Compute urge to ask questions from uncertain or blocked goals.
        
        Returns urges when:
        - Goals require clarification (type contains 'CLARIFY')
        - Goals are blocked and may need help
        """
        urges = []
        for goal in goals:
            goal_type = str(getattr(goal, 'type', '')).upper()
            
            # Check for clarification goals
            if 'CLARIFY' in goal_type:
                urges.append(CommunicationUrge(
                    drive_type=DriveType.QUESTION,
                    intensity=getattr(goal, 'priority', 0.5),
                    content=getattr(goal, 'description', ''),
                    reason="Goal requires clarification",
                    priority=0.6
                ))
            
            # Check for blocked goals
            elif getattr(goal, 'status', None) == 'blocked':
                urges.append(CommunicationUrge(
                    drive_type=DriveType.QUESTION,
                    intensity=0.5,
                    content=f"How to proceed with: {getattr(goal, 'description', '')}",
                    reason="Goal is blocked, may need help",
                    priority=0.5
                ))
        
        return urges
    
    def _compute_emotional_drive(
        self,
        emotional_state: Dict[str, float]
    ) -> List[CommunicationUrge]:
        """
        Compute urge to express emotions from arousal or valence extremes.
        
        Returns urges when:
        - Arousal magnitude exceeds threshold (excitement/anxiety)
        - Valence magnitude exceeds threshold (joy/distress)
        """
        urges = []
        valence = emotional_state.get('valence', 0.0)
        arousal = emotional_state.get('arousal', 0.0)
        
        # High arousal creates expression pressure
        if abs(arousal) > self.emotional_threshold:
            urges.append(self._create_emotional_urge(
                intensity=abs(arousal),
                reason=("High positive arousal - excitement to share" if arousal > 0
                        else "High negative arousal - distress to express"),
                priority=0.65
            ))
        
        # Extreme valence creates expression need
        if abs(valence) > self.emotional_threshold:
            urges.append(self._create_emotional_urge(
                intensity=abs(valence),
                reason=("Strong positive emotion seeking expression" if valence > 0
                        else "Strong negative emotion seeking expression"),
                priority=0.6
            ))
        
        return urges
    
    def _create_emotional_urge(self, intensity: float, reason: str, priority: float) -> CommunicationUrge:
        """Helper to create emotional urge with standard parameters."""
        return CommunicationUrge(
            drive_type=DriveType.EMOTIONAL,
            intensity=intensity,
            reason=reason,
            priority=priority
        )
    
    def _compute_social_drive(self) -> List[CommunicationUrge]:
        """
        Compute urge for social connection based on silence duration.
        
        Returns urge when silence exceeds threshold, with intensity
        increasing up to 3x the threshold (caps at 1.0).
        """
        if not self.last_input_time:
            return []
        
        silence_minutes = (datetime.now() - self.last_input_time).total_seconds() / 60.0
        
        if silence_minutes <= self.social_silence_minutes:
            return []
        
        # Intensity grows to 1.0 at 3x threshold
        intensity = min(1.0, silence_minutes / (self.social_silence_minutes * 3.0))
        
        return [CommunicationUrge(
            drive_type=DriveType.SOCIAL,
            intensity=intensity,
            reason=f"No interaction for {int(silence_minutes)} minutes",
            priority=0.4,
            decay_rate=0.05  # Social drive decays slower
        )]
    
    def _compute_goal_drive(self, goals: List[Any]) -> List[CommunicationUrge]:
        """
        Compute urge to communicate for goal-driven needs.
        
        Returns urges when:
        - Goals require user response (type contains 'RESPOND')
        - Goals are completed and may want acknowledgment
        """
        urges = []
        for goal in goals:
            goal_type = str(getattr(goal, 'type', '')).upper()
            description = getattr(goal, 'description', '')
            
            # Response goals create strong communication drive
            if 'RESPOND' in goal_type:
                urges.append(CommunicationUrge(
                    drive_type=DriveType.GOAL,
                    intensity=getattr(goal, 'priority', 0.5),
                    content=description,
                    reason="Active goal requires response",
                    priority=0.8
                ))
            
            # Completed goals may want to report
            elif getattr(goal, 'status', None) == 'completed':
                urges.append(CommunicationUrge(
                    drive_type=DriveType.GOAL,
                    intensity=0.4,
                    content=f"Completed: {description}",
                    reason="Goal completed, may want to report",
                    priority=0.3
                ))
        
        return urges
    
    def _compute_acknowledgment_drive(
        self,
        workspace_state: Any
    ) -> List[CommunicationUrge]:
        """
        Compute urge to acknowledge recent human input.
        
        Returns single urge if human input detected within 5 seconds.
        Only generates one acknowledgment urge at a time.
        """
        if not hasattr(workspace_state, 'percepts'):
            return []
        
        now = datetime.now()
        for percept in workspace_state.percepts.values():
            source = getattr(percept, 'source', '').lower()
            
            # Check if this is human input
            if 'human' not in source and 'input' not in source:
                continue
            
            # Check if recent (within 5 seconds)
            created = getattr(percept, 'created_at', None)
            if created and (now - created).total_seconds() < 5.0:
                return [CommunicationUrge(
                    drive_type=DriveType.ACKNOWLEDGMENT,
                    intensity=0.7,
                    reason="Recent human input needs acknowledgment",
                    priority=0.75
                )]
        
        return []
    
    def _cleanup_expired_urges(self) -> None:
        """Remove urges that have decayed below expiration threshold."""
        self.active_urges = [u for u in self.active_urges if not u.is_expired()]
    
    def get_total_drive(self) -> float:
        """
        Get combined communication drive intensity using diminishing returns.
        
        Urges are sorted by weighted intensity (intensity * priority),
        with each additional urge contributing less (1/n weight).
        Result clamped to [0.0, 1.0].
        
        Returns:
            Combined drive intensity
        """
        if not self.active_urges:
            return 0.0
        
        # Sort by weighted intensity for consistent ordering
        sorted_urges = sorted(
            self.active_urges,
            key=lambda u: u.get_current_intensity() * u.priority,
            reverse=True
        )
        
        # Diminishing returns: 1/1, 1/2, 1/3, ...
        total = sum(
            urge.get_current_intensity() * urge.priority / (i + 1)
            for i, urge in enumerate(sorted_urges)
        )
        
        return min(1.0, total)
    
    def get_strongest_urge(self) -> Optional[CommunicationUrge]:
        """Get the urge with highest weighted intensity (intensity * priority)."""
        if not self.active_urges:
            return None
        
        return max(
            self.active_urges,
            key=lambda u: u.get_current_intensity() * u.priority
        )
    
    def record_input(self) -> None:
        """Record timestamp of received input for social drive and proactive tracking."""
        self.last_input_time = datetime.now()
        if self.proactive_system:
            self.proactive_system.record_interaction()
    
    def record_output(self) -> None:
        """
        Record timestamp of produced output and clear acknowledgment urges.
        
        Acknowledgment urges are cleared since output implies acknowledgment.
        Also updates proactive system interaction tracking.
        """
        self.last_output_time = datetime.now()
        self.active_urges = [
            u for u in self.active_urges
            if u.drive_type != DriveType.ACKNOWLEDGMENT
        ]
        if self.proactive_system:
            self.proactive_system.record_interaction()
    
    def _compute_proactive_drive(
        self,
        workspace_state: Any,
        memories: List[Any],
        goals: List[Any]
    ) -> List[CommunicationUrge]:
        """
        Compute urges from proactive outreach opportunities.
        
        Checks for proactive opportunities (time-based, event-based, etc.)
        and generates urges for high-urgency opportunities.
        
        Args:
            workspace_state: Current workspace snapshot
            memories: Recently retrieved memories
            goals: Active goals
            
        Returns:
            List of communication urges from proactive opportunities
        """
        # Check for new opportunities
        opportunities = self.proactive_system.check_for_opportunities(
            workspace_state, memories, goals
        )
        
        # Generate urges for high-urgency opportunities (threshold: 0.3)
        urges = []
        for opp in opportunities:
            if opp.urgency > 0.3:
                drive_type = self._map_trigger_to_drive(opp.trigger)
                
                # Priority scales with urgency: base 0.5 + (0.0 to 0.3)
                priority = 0.5 + (opp.urgency * 0.3)
                
                urges.append(CommunicationUrge(
                    drive_type=drive_type,
                    intensity=opp.urgency,
                    content=opp.suggested_content,
                    reason=f"Proactive: {opp.reason}",
                    priority=priority,
                    decay_rate=0.05  # Slower decay for proactive urges
                ))
        
        return urges
    
    def _map_trigger_to_drive(self, trigger) -> DriveType:
        """Map OutreachTrigger to DriveType."""
        try:
            from .proactive import OutreachTrigger
        except ImportError:
            from proactive import OutreachTrigger
        
        mapping = {
            OutreachTrigger.TIME_ELAPSED: DriveType.SOCIAL,
            OutreachTrigger.SIGNIFICANT_INSIGHT: DriveType.INSIGHT,
            OutreachTrigger.EMOTIONAL_CONNECTION: DriveType.EMOTIONAL,
            OutreachTrigger.SCHEDULED_CHECKIN: DriveType.SOCIAL,
            OutreachTrigger.RELEVANT_EVENT: DriveType.INSIGHT,
            OutreachTrigger.GOAL_COMPLETION: DriveType.GOAL
        }
        
        return mapping.get(trigger, DriveType.SOCIAL)
    
    def get_drive_summary(self) -> Dict[str, Any]:
        """Get summary of current drive state."""
        summary = {
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
        
        # Add proactive system summary if enabled
        if self.proactive_system:
            summary["proactive"] = self.proactive_system.get_outreach_summary()
        
        return summary
