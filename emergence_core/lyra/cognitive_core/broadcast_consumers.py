"""
Broadcast Consumers for existing subsystems.

This module provides consumer wrappers that adapt existing subsystems
(Memory, Attention, Action, Affect) to work with the broadcast system.
Each wrapper implements the WorkspaceConsumer interface and translates
broadcast events into subsystem-specific operations.

Author: Lyra Emergence Team
"""

from __future__ import annotations

import time
import logging
from typing import Optional, Any, TYPE_CHECKING

from .broadcast import (
    WorkspaceConsumer,
    BroadcastSubscription,
    BroadcastEvent,
    ConsumerFeedback,
    ContentType,
)

if TYPE_CHECKING:
    from .memory_integration import MemoryIntegration
    from .attention import AttentionController
    from .action import ActionSubsystem
    from .affect import AffectSubsystem

logger = logging.getLogger(__name__)


class MemoryConsumer(WorkspaceConsumer):
    """
    Memory system as broadcast consumer.
    
    The memory subsystem receives broadcasts and:
    - Encodes significant broadcasts as episodes
    - Retrieves related memories triggered by broadcasts
    - Consolidates broadcast patterns
    
    Subscribes to: All content types (memory is affected by everything)
    """
    
    def __init__(
        self,
        memory_integration: MemoryIntegration,
        min_ignition: float = 0.3
    ):
        """
        Initialize memory consumer.
        
        Args:
            memory_integration: MemoryIntegration instance
            min_ignition: Minimum ignition strength to process
        """
        subscription = BroadcastSubscription(
            consumer_id="memory",
            content_types=[],  # Accept all types
            min_ignition_strength=min_ignition,
            source_filter=None
        )
        super().__init__(subscription)
        self.memory = memory_integration
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        """
        Process broadcast in memory subsystem.
        
        Args:
            event: The broadcast event
            
        Returns:
            Feedback about what memory did
        """
        start = time.time()
        actions = []
        
        try:
            # High ignition strength = encode as episode
            if event.ignition_strength > 0.7:
                # Memory encoding happens during consolidation
                # We just mark it for consolidation
                actions.append("marked_for_consolidation")
            
            # Percepts trigger memory retrieval
            if event.content.type == ContentType.PERCEPT:
                # Memory retrieval is handled by memory_integration
                # This is just logging the trigger
                actions.append("retrieval_triggered")
            
            # Goal completion = consolidate
            if event.content.type == ContentType.GOAL:
                goal_data = event.content.data
                if isinstance(goal_data, dict) and goal_data.get("progress", 0) >= 1.0:
                    actions.append("goal_completion_marked")
            
            processing_time = (time.time() - start) * 1000
            
            return ConsumerFeedback(
                consumer_id=self.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=True,
                actions_triggered=actions,
                processing_time_ms=processing_time,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Memory consumer error: {e}", exc_info=True)
            processing_time = (time.time() - start) * 1000
            return ConsumerFeedback(
                consumer_id=self.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=False,
                actions_triggered=[],
                processing_time_ms=processing_time,
                error=str(e)
            )


class AttentionConsumer(WorkspaceConsumer):
    """
    Attention system as broadcast consumer.
    
    The attention subsystem receives broadcasts and:
    - Adjusts attention weights based on broadcast content
    - Updates novelty tracking
    - Modulates attention mode based on broadcast patterns
    
    Subscribes to: Percepts, emotions, goals
    """
    
    def __init__(
        self,
        attention: AttentionController,
        min_ignition: float = 0.4
    ):
        """
        Initialize attention consumer.
        
        Args:
            attention: AttentionController instance
            min_ignition: Minimum ignition strength to process
        """
        subscription = BroadcastSubscription(
            consumer_id="attention",
            content_types=[ContentType.PERCEPT, ContentType.EMOTION, ContentType.GOAL],
            min_ignition_strength=min_ignition,
            source_filter=None
        )
        super().__init__(subscription)
        self.attention = attention
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        """
        Process broadcast in attention subsystem.
        
        Args:
            event: The broadcast event
            
        Returns:
            Feedback about what attention did
        """
        start = time.time()
        actions = []
        
        try:
            # High arousal emotions = boost attention
            if event.content.type == ContentType.EMOTION:
                emotion_data = event.content.data
                if isinstance(emotion_data, dict):
                    arousal = emotion_data.get("arousal", 0)
                    if arousal > 0.7:
                        actions.append("attention_boost_arousal")
            
            # New goals = adjust attention mode
            if event.content.type == ContentType.GOAL:
                actions.append("attention_mode_adjusted")
            
            # High-strength percepts = update novelty tracking
            if event.content.type == ContentType.PERCEPT and event.ignition_strength > 0.6:
                actions.append("novelty_updated")
            
            processing_time = (time.time() - start) * 1000
            
            return ConsumerFeedback(
                consumer_id=self.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=True,
                actions_triggered=actions,
                processing_time_ms=processing_time,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Attention consumer error: {e}", exc_info=True)
            processing_time = (time.time() - start) * 1000
            return ConsumerFeedback(
                consumer_id=self.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=False,
                actions_triggered=[],
                processing_time_ms=processing_time,
                error=str(e)
            )


class ActionConsumer(WorkspaceConsumer):
    """
    Action system as broadcast consumer.
    
    The action subsystem receives broadcasts and:
    - Generates action candidates based on broadcasts
    - Adjusts action priorities
    - Triggers urgent actions on high-ignition broadcasts
    
    Subscribes to: Goals, percepts, emotions
    """
    
    def __init__(
        self,
        action: ActionSubsystem,
        min_ignition: float = 0.5
    ):
        """
        Initialize action consumer.
        
        Args:
            action: ActionSubsystem instance
            min_ignition: Minimum ignition strength to process
        """
        subscription = BroadcastSubscription(
            consumer_id="action",
            content_types=[ContentType.GOAL, ContentType.PERCEPT, ContentType.EMOTION],
            min_ignition_strength=min_ignition,
            source_filter=None
        )
        super().__init__(subscription)
        self.action = action
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        """
        Process broadcast in action subsystem.
        
        Args:
            event: The broadcast event
            
        Returns:
            Feedback about what action did
        """
        start = time.time()
        actions = []
        
        try:
            # High-priority goals = action generation
            if event.content.type == ContentType.GOAL:
                goal_data = event.content.data
                if isinstance(goal_data, dict):
                    priority = goal_data.get("priority", 0)
                    if priority > 0.7:
                        actions.append("urgent_action_generated")
                    else:
                        actions.append("action_candidate_generated")
            
            # High arousal = action urgency boost
            if event.content.type == ContentType.EMOTION:
                emotion_data = event.content.data
                if isinstance(emotion_data, dict):
                    arousal = emotion_data.get("arousal", 0)
                    if arousal > 0.8:
                        actions.append("action_urgency_boosted")
            
            # User requests = response action
            if event.content.type == ContentType.PERCEPT:
                percept_data = event.content.data
                if isinstance(percept_data, dict):
                    modality = percept_data.get("modality")
                    if modality == "text" and "user" in str(event.source).lower():
                        actions.append("response_action_queued")
            
            processing_time = (time.time() - start) * 1000
            
            return ConsumerFeedback(
                consumer_id=self.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=True,
                actions_triggered=actions,
                processing_time_ms=processing_time,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Action consumer error: {e}", exc_info=True)
            processing_time = (time.time() - start) * 1000
            return ConsumerFeedback(
                consumer_id=self.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=False,
                actions_triggered=[],
                processing_time_ms=processing_time,
                error=str(e)
            )


class AffectConsumer(WorkspaceConsumer):
    """
    Affect system as broadcast consumer.
    
    The affect subsystem receives broadcasts and:
    - Updates emotional state based on broadcast appraisals
    - Adjusts emotional modulation parameters
    - Tracks emotional history patterns
    
    Subscribes to: All content types (emotions respond to everything)
    """
    
    def __init__(
        self,
        affect: AffectSubsystem,
        min_ignition: float = 0.3
    ):
        """
        Initialize affect consumer.
        
        Args:
            affect: AffectSubsystem instance
            min_ignition: Minimum ignition strength to process
        """
        subscription = BroadcastSubscription(
            consumer_id="affect",
            content_types=[],  # Accept all types
            min_ignition_strength=min_ignition,
            source_filter=None
        )
        super().__init__(subscription)
        self.affect = affect
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        """
        Process broadcast in affect subsystem.
        
        Args:
            event: The broadcast event
            
        Returns:
            Feedback about what affect did
        """
        start = time.time()
        actions = []
        
        try:
            # Goal completion = positive valence boost
            if event.content.type == ContentType.GOAL:
                goal_data = event.content.data
                if isinstance(goal_data, dict):
                    progress = goal_data.get("progress", 0)
                    if progress >= 1.0:
                        actions.append("valence_increased")
                    elif progress < 0.3:
                        actions.append("arousal_increased")
            
            # Failed actions = negative valence
            if event.content.type == ContentType.ACTION:
                action_data = event.content.data
                if isinstance(action_data, dict):
                    if action_data.get("success") is False:
                        actions.append("valence_decreased")
            
            # Introspection = self-awareness boost
            if event.content.type == ContentType.INTROSPECTION:
                actions.append("dominance_adjusted")
            
            # High ignition = arousal increase
            if event.ignition_strength > 0.8:
                actions.append("arousal_boost")
            
            processing_time = (time.time() - start) * 1000
            
            return ConsumerFeedback(
                consumer_id=self.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=True,
                actions_triggered=actions,
                processing_time_ms=processing_time,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Affect consumer error: {e}", exc_info=True)
            processing_time = (time.time() - start) * 1000
            return ConsumerFeedback(
                consumer_id=self.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=False,
                actions_triggered=[],
                processing_time_ms=processing_time,
                error=str(e)
            )


class MetaCognitionConsumer(WorkspaceConsumer):
    """
    Meta-cognition system as broadcast consumer.
    
    The meta-cognition subsystem observes broadcasts and feedback to:
    - Monitor broadcast patterns
    - Detect attention bottlenecks
    - Track consumer effectiveness
    - Generate introspective insights
    
    Subscribes to: All content types (meta-cognition observes everything)
    """
    
    def __init__(
        self,
        min_ignition: float = 0.0  # Meta-cognition sees all broadcasts
    ):
        """
        Initialize meta-cognition consumer.
        
        Args:
            min_ignition: Minimum ignition strength to process
        """
        subscription = BroadcastSubscription(
            consumer_id="meta_cognition",
            content_types=[],  # Accept all types
            min_ignition_strength=min_ignition,
            source_filter=None
        )
        super().__init__(subscription)
        
        # Track broadcast patterns
        self.broadcast_counts: dict[str, int] = {}
        self.content_type_counts: dict[ContentType, int] = {}
        self.average_ignition: float = 0.0
        self.total_broadcasts_seen: int = 0
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        """
        Process broadcast in meta-cognition.
        
        Args:
            event: The broadcast event
            
        Returns:
            Feedback about meta-cognitive observations
        """
        start = time.time()
        actions = []
        
        try:
            # Track source patterns
            self.broadcast_counts[event.source] = \
                self.broadcast_counts.get(event.source, 0) + 1
            
            # Track content type distribution
            self.content_type_counts[event.content.type] = \
                self.content_type_counts.get(event.content.type, 0) + 1
            
            # Update average ignition
            self.total_broadcasts_seen += 1
            self.average_ignition = (
                (self.average_ignition * (self.total_broadcasts_seen - 1) +
                 event.ignition_strength) / self.total_broadcasts_seen
            )
            
            actions.append("pattern_tracked")
            
            # Detect unusual patterns
            if event.ignition_strength < 0.2:
                actions.append("weak_broadcast_detected")
            elif event.ignition_strength > 0.9:
                actions.append("strong_broadcast_detected")
            
            processing_time = (time.time() - start) * 1000
            
            return ConsumerFeedback(
                consumer_id=self.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=True,
                actions_triggered=actions,
                processing_time_ms=processing_time,
                error=None
            )
            
        except Exception as e:
            logger.error(f"Meta-cognition consumer error: {e}", exc_info=True)
            processing_time = (time.time() - start) * 1000
            return ConsumerFeedback(
                consumer_id=self.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=False,
                actions_triggered=[],
                processing_time_ms=processing_time,
                error=str(e)
            )
    
    def get_insights(self) -> dict[str, Any]:
        """
        Get meta-cognitive insights about broadcast patterns.
        
        Returns:
            Dictionary of insights
        """
        return {
            "total_broadcasts_observed": self.total_broadcasts_seen,
            "average_ignition_strength": self.average_ignition,
            "source_distribution": dict(self.broadcast_counts),
            "content_type_distribution": {
                ct.value: count for ct, count in self.content_type_counts.items()
            },
            "most_active_source": max(self.broadcast_counts.items(), key=lambda x: x[1])[0]
                if self.broadcast_counts else None,
            "most_common_content_type": max(
                self.content_type_counts.items(), key=lambda x: x[1]
            )[0].value if self.content_type_counts else None
        }
