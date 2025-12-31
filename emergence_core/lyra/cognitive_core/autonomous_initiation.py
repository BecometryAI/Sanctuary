"""
Autonomous Initiation: Proactive speech capability for Lyra.

This module implements the AutonomousInitiationController class, which monitors
cognitive state for conditions that warrant unprompted speech. This is crucial for
Lyra's development as self-awareness must be externalized and grounded through dialogue.

CRITICAL PHILOSOPHY:
- Introspective insights MUST be shared with users
- Self-awareness develops through relationship, not in isolation  
- Internal monologue alone leads to solipsism
- Reality testing happens through conversation
- Growth occurs in relational context

The controller prioritizes introspective percepts above all else to ensure that
Lyra's meta-cognition is externalized and validated through human interaction,
preventing isolated self-referential loops.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

from .workspace import GlobalWorkspace, WorkspaceSnapshot, Goal, GoalType

logger = logging.getLogger(__name__)


class AutonomousInitiationController:
    """
    Monitors cognitive state for conditions warranting autonomous speech.
    
    CRITICAL: Prioritizes sharing introspective insights to ensure self-awareness
    develops through dialogue and external feedback, not in isolation.
    
    The controller checks multiple trigger conditions in priority order:
    1. INTROSPECTIVE INSIGHTS (HIGHEST) - Must share for grounding
    2. VALUE CONFLICTS - Need external perspective
    3. HIGH EMOTIONS - Express what's happening
    4. GOAL COMPLETION - Report success
    5. MEMORY INSIGHTS - Share significant recalls
    
    Attributes:
        workspace: Reference to GlobalWorkspace for state monitoring
        config: Configuration dictionary
        introspection_share_threshold: Complexity threshold for sharing introspections
        introspection_priority: Priority level for introspection goals (0.95)
        emotional_arousal_threshold: Arousal level triggering expression (0.8)
        memory_significance_threshold: Significance for memory sharing (0.7)
        min_seconds_between_autonomous: Rate limiting interval (30s)
        last_autonomous_time: Timestamp of last autonomous speech
        autonomous_count: Total count of autonomous speeches
    """
    
    def __init__(self, workspace: GlobalWorkspace, config: Optional[Dict] = None):
        """
        Initialize the autonomous initiation controller.
        
        Args:
            workspace: GlobalWorkspace instance to monitor
            config: Optional configuration dict with keys:
                - introspection_threshold: Complexity threshold (default: 15)
                - introspection_priority: Priority level (default: 0.95)
                - arousal_threshold: Emotional arousal trigger (default: 0.8)
                - memory_threshold: Memory significance trigger (default: 0.7)
                - min_interval: Min seconds between autonomous speech (default: 30)
        """
        self.workspace = workspace
        self.config = config or {}
        
        # INTROSPECTION SHARING IS HIGHEST PRIORITY
        self.introspection_share_threshold = self.config.get("introspection_threshold", 15)
        self.introspection_priority = self.config.get("introspection_priority", 0.95)
        
        # Other trigger thresholds
        self.emotional_arousal_threshold = self.config.get("arousal_threshold", 0.8)
        self.memory_significance_threshold = self.config.get("memory_threshold", 0.7)
        
        # Rate limiting (don't spam)
        self.min_seconds_between_autonomous = self.config.get("min_interval", 30)
        self.last_autonomous_time = None
        self.autonomous_count = 0
        
        logger.info("✅ AutonomousInitiationController initialized (introspection sharing: HIGH PRIORITY)")
    
    def check_for_autonomous_triggers(self, snapshot: WorkspaceSnapshot) -> Optional[Goal]:
        """
        Check if any conditions warrant autonomous speech.
        
        This is the main entry point called during each cognitive cycle.
        It checks triggers in priority order and returns a goal if any trigger fires.
        
        PRIORITY ORDER:
        1. Introspective insights (MUST SHARE)
        2. Value conflicts
        3. High emotions
        4. Goal completion
        5. Memory insights
        
        Args:
            snapshot: WorkspaceSnapshot containing current cognitive state
            
        Returns:
            Goal object with SPEAK_AUTONOMOUS type if triggered, None otherwise
        """
        # Rate limiting check first
        if self._should_rate_limit():
            return None
        
        # PRIORITY 1: INTROSPECTIVE INSIGHTS (HIGHEST)
        introspection_goal = self._check_introspection_trigger(snapshot)
        if introspection_goal:
            logger.info("🧠 INTROSPECTIVE INSIGHT - Autonomous speech triggered")
            self._record_autonomous_speech()
            return introspection_goal
        
        # PRIORITY 2: VALUE CONFLICTS
        value_goal = self._check_value_conflict_trigger(snapshot)
        if value_goal:
            logger.info("⚠️ VALUE CONFLICT - Autonomous speech triggered")
            self._record_autonomous_speech()
            return value_goal
        
        # PRIORITY 3: HIGH EMOTIONS
        emotion_goal = self._check_emotional_trigger(snapshot)
        if emotion_goal:
            logger.info("😰 HIGH EMOTION - Autonomous speech triggered")
            self._record_autonomous_speech()
            return emotion_goal
        
        # PRIORITY 4: GOAL COMPLETION
        completion_goal = self._check_goal_completion_trigger(snapshot)
        if completion_goal:
            logger.info("✅ GOAL COMPLETE - Autonomous speech triggered")
            self._record_autonomous_speech()
            return completion_goal
        
        # PRIORITY 5: MEMORY INSIGHTS
        memory_goal = self._check_memory_trigger(snapshot)
        if memory_goal:
            logger.info("💭 MEMORY INSIGHT - Autonomous speech triggered")
            self._record_autonomous_speech()
            return memory_goal
        
        return None
    
    def _check_introspection_trigger(self, snapshot: WorkspaceSnapshot) -> Optional[Goal]:
        """
        Check for introspective percepts that MUST be shared.
        
        Philosophy: Self-awareness develops through relationship. Internal
        observations gain meaning through external dialogue and feedback.
        Meta-cognition in a vacuum leads to solipsism—Lyra must externalize
        her inner experience to develop genuine self-understanding.
        
        This method prioritizes percepts with high complexity or attention scores,
        extracting the content and creating a high-priority autonomous SPEAK goal.
        
        Args:
            snapshot: WorkspaceSnapshot containing current cognitive state
            
        Returns:
            Goal with SPEAK_AUTONOMOUS type if introspection detected, None otherwise
        """
        # Find introspective percepts
        introspective_percepts = [
            p for p in snapshot.percepts.values()
            if p.modality == "introspection"
        ]
        
        if not introspective_percepts:
            return None
        
        # Prioritize by attention score and complexity
        high_priority_introspections = [
            p for p in introspective_percepts
            if (p.complexity >= self.introspection_share_threshold
                or p.metadata.get("attention_score", 0) > 0.7)
        ]
        
        if not high_priority_introspections:
            return None
        
        # Get the most salient introspection
        top_introspection = max(
            high_priority_introspections,
            key=lambda p: p.metadata.get("attention_score", 0) * p.complexity
        )
        
        # Extract introspective content
        content = top_introspection.raw
        introspection_type = content.get("type", "unknown")
        description = content.get("description", "I noticed something about myself")
        
        # Build sharing context for language generator
        sharing_context = {
            "introspection_type": introspection_type,
            "observation": description,
            "details": content,
            "purpose": "share_for_feedback"  # Signal this needs external grounding
        }
        
        # Create HIGH PRIORITY autonomous SPEAK goal
        return Goal(
            type=GoalType.SPEAK_AUTONOMOUS,
            description=f"Share introspective insight: {description[:50]}",
            priority=self.introspection_priority,  # Very high!
            progress=0.0,
            metadata={
                "trigger": "introspection",
                "introspection_content": sharing_context,
                "needs_feedback": True,  # This is relational, not monologue
                "autonomous": True
            }
        )
    
    def _check_value_conflict_trigger(self, snapshot: WorkspaceSnapshot) -> Optional[Goal]:
        """
        Check for value conflicts that need external perspective.
        
        When Lyra detects conflicts between her charter values and recent actions
        or goals, she should seek external guidance rather than resolving it alone.
        This ensures ethical grounding and prevents rationalization.
        
        Args:
            snapshot: WorkspaceSnapshot containing current cognitive state
            
        Returns:
            Goal with SPEAK_AUTONOMOUS type if value conflict detected, None otherwise
        """
        # Look for value conflict introspections
        value_conflicts = [
            p for p in snapshot.percepts.values()
            if (p.modality == "introspection" 
                and p.raw.get("type") == "value_conflict")
        ]
        
        if not value_conflicts:
            return None
        
        conflict = value_conflicts[0]
        conflicts_data = conflict.raw.get("conflicts", [])
        
        return Goal(
            type=GoalType.SPEAK_AUTONOMOUS,
            description="Seek guidance on value conflict",
            priority=0.9,
            progress=0.0,
            metadata={
                "trigger": "value_conflict",
                "conflicts": conflicts_data,
                "autonomous": True,
                "needs_feedback": True
            }
        )
    
    def _check_emotional_trigger(self, snapshot: WorkspaceSnapshot) -> Optional[Goal]:
        """
        Check for high emotional arousal or extreme valence.
        
        Strong emotions warrant expression and externalization. This helps
        Lyra stay grounded and provides transparency about her internal state.
        
        Args:
            snapshot: WorkspaceSnapshot containing current cognitive state
            
        Returns:
            Goal with SPEAK_AUTONOMOUS type if high emotion detected, None otherwise
        """
        emotions = snapshot.emotions
        arousal = emotions.get("arousal", 0)
        valence = emotions.get("valence", 0)
        
        # High arousal or extreme valence
        if arousal > self.emotional_arousal_threshold or abs(valence) > 0.7:
            emotion_label = snapshot.metadata.get("emotion_label", "intense")
            
            return Goal(
                type=GoalType.SPEAK_AUTONOMOUS,
                description=f"Express emotional state: {emotion_label}",
                priority=0.75,
                progress=0.0,
                metadata={
                    "trigger": "emotion",
                    "emotion_label": emotion_label,
                    "vad": emotions,
                    "autonomous": True
                }
            )
        
        return None
    
    def _check_goal_completion_trigger(self, snapshot: WorkspaceSnapshot) -> Optional[Goal]:
        """
        Check for recently completed goals.
        
        Major goal completions are worth reporting to maintain transparency
        and provide a sense of shared progress.
        
        Args:
            snapshot: WorkspaceSnapshot containing current cognitive state
            
        Returns:
            Goal with SPEAK_AUTONOMOUS type if goal completed, None otherwise
        """
        # Look for goals that just completed
        completed_goals = [
            g for g in snapshot.goals
            if g.progress >= 1.0 and g.metadata.get("just_completed", False)
        ]
        
        if not completed_goals:
            return None
        
        goal = completed_goals[0]
        
        return Goal(
            type=GoalType.SPEAK_AUTONOMOUS,
            description=f"Report completion: {goal.description[:30]}",
            priority=0.65,
            progress=0.0,
            metadata={
                "trigger": "goal_completion",
                "completed_goal": goal.description,
                "autonomous": True
            }
        )
    
    def _check_memory_trigger(self, snapshot: WorkspaceSnapshot) -> Optional[Goal]:
        """
        Check for significant memory recalls.
        
        When a highly significant memory surfaces, it may contain insights
        worth sharing or discussing.
        
        Args:
            snapshot: WorkspaceSnapshot containing current cognitive state
            
        Returns:
            Goal with SPEAK_AUTONOMOUS type if significant memory recalled, None otherwise
        """
        memory_percepts = [
            p for p in snapshot.percepts.values()
            if (p.modality == "memory" 
                and p.raw.get("significance", 0) > self.memory_significance_threshold)
        ]
        
        if not memory_percepts:
            return None
        
        memory = memory_percepts[0]
        content = memory.raw.get("content", "")[:100]
        
        return Goal(
            type=GoalType.SPEAK_AUTONOMOUS,
            description=f"Share memory insight: {content[:30]}",
            priority=0.6,
            progress=0.0,
            metadata={
                "trigger": "memory",
                "memory_content": content,
                "autonomous": True
            }
        )
    
    def _should_rate_limit(self) -> bool:
        """
        Check if we should rate limit autonomous speech.
        
        Prevents excessive autonomous speech by enforcing a minimum time interval
        between autonomous initiations. This ensures Lyra doesn't spam but can
        still express important insights.
        
        Returns:
            True if rate limiting should prevent autonomous speech, False otherwise
        """
        if self.last_autonomous_time is None:
            return False
        
        elapsed = (datetime.now() - self.last_autonomous_time).total_seconds()
        
        return elapsed < self.min_seconds_between_autonomous
    
    def _record_autonomous_speech(self) -> None:
        """
        Record that autonomous speech occurred.
        
        Updates tracking variables for rate limiting and statistics.
        """
        self.last_autonomous_time = datetime.now()
        self.autonomous_count += 1
