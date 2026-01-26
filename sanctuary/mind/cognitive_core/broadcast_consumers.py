"""
Broadcast consumer adapters for existing subsystems.
Wraps Memory, Attention, Action, Affect as WorkspaceConsumer implementations.
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
    """Memory subsystem as broadcast consumer."""
    
    def __init__(self, memory_integration: MemoryIntegration, min_ignition: float = 0.3):
        subscription = BroadcastSubscription("memory", [], min_ignition, None)
        super().__init__(subscription)
        self.memory = memory_integration
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        start = time.time()
        actions = []
        
        try:
            if event.ignition_strength > 0.7:
                actions.append("marked_for_consolidation")
            
            if event.content.type == ContentType.PERCEPT:
                actions.append("retrieval_triggered")
            
            if event.content.type == ContentType.GOAL:
                goal_data = event.content.data
                if isinstance(goal_data, dict) and goal_data.get("progress", 0) >= 1.0:
                    actions.append("goal_completion_marked")
            
            return ConsumerFeedback(
                self.subscription.consumer_id, event.id, True, True,
                actions, (time.time() - start) * 1000, None
            )
        except Exception as e:
            logger.error(f"Memory consumer error: {e}", exc_info=True)
            return ConsumerFeedback(
                self.subscription.consumer_id, event.id, True, False,
                [], (time.time() - start) * 1000, str(e)
            )


class AttentionConsumer(WorkspaceConsumer):
    """Attention subsystem as broadcast consumer."""
    
    def __init__(self, attention: AttentionController, min_ignition: float = 0.4):
        subscription = BroadcastSubscription(
            "attention", [ContentType.PERCEPT, ContentType.EMOTION, ContentType.GOAL], min_ignition, None
        )
        super().__init__(subscription)
        self.attention = attention
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        start, actions = time.time(), []
        try:
            if event.content.type == ContentType.EMOTION:
                emotion_data = event.content.data
                if isinstance(emotion_data, dict) and emotion_data.get("arousal", 0) > 0.7:
                    actions.append("attention_boost_arousal")
            elif event.content.type == ContentType.GOAL:
                actions.append("attention_mode_adjusted")
            elif event.content.type == ContentType.PERCEPT and event.ignition_strength > 0.6:
                actions.append("novelty_updated")
            
            return ConsumerFeedback(
                self.subscription.consumer_id, event.id, True, True,
                actions, (time.time() - start) * 1000, None
            )
        except Exception as e:
            logger.error(f"Attention consumer error: {e}", exc_info=True)
            return ConsumerFeedback(
                self.subscription.consumer_id, event.id, True, False,
                [], (time.time() - start) * 1000, str(e)
            )


class ActionConsumer(WorkspaceConsumer):
    """Action subsystem as broadcast consumer."""
    
    def __init__(self, action: ActionSubsystem, min_ignition: float = 0.5):
        subscription = BroadcastSubscription(
            "action", [ContentType.GOAL, ContentType.PERCEPT, ContentType.EMOTION], min_ignition, None
        )
        super().__init__(subscription)
        self.action = action
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        start, actions = time.time(), []
        try:
            if event.content.type == ContentType.GOAL:
                goal_data = event.content.data
                if isinstance(goal_data, dict):
                    priority = goal_data.get("priority", 0)
                    actions.append("urgent_action_generated" if priority > 0.7 else "action_candidate_generated")
            
            elif event.content.type == ContentType.EMOTION:
                emotion_data = event.content.data
                if isinstance(emotion_data, dict) and emotion_data.get("arousal", 0) > 0.8:
                    actions.append("action_urgency_boosted")
            
            elif event.content.type == ContentType.PERCEPT:
                percept_data = event.content.data
                if isinstance(percept_data, dict):
                    if percept_data.get("modality") == "text" and "user" in str(event.source).lower():
                        actions.append("response_action_queued")
            
            return ConsumerFeedback(
                self.subscription.consumer_id, event.id, True, True,
                actions, (time.time() - start) * 1000, None
            )
        except Exception as e:
            logger.error(f"Action consumer error: {e}", exc_info=True)
            return ConsumerFeedback(
                self.subscription.consumer_id, event.id, True, False,
                [], (time.time() - start) * 1000, str(e)
            )


class AffectConsumer(WorkspaceConsumer):
    """Affect subsystem as broadcast consumer."""
    
    def __init__(self, affect: AffectSubsystem, min_ignition: float = 0.3):
        subscription = BroadcastSubscription("affect", [], min_ignition, None)
        super().__init__(subscription)
        self.affect = affect
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        start, actions = time.time(), []
        try:
            if event.content.type == ContentType.GOAL:
                goal_data = event.content.data
                if isinstance(goal_data, dict):
                    progress = goal_data.get("progress", 0)
                    actions.append("valence_increased" if progress >= 1.0 else "arousal_increased" if progress < 0.3 else "")
            
            elif event.content.type == ContentType.ACTION:
                action_data = event.content.data
                if isinstance(action_data, dict) and action_data.get("success") is False:
                    actions.append("valence_decreased")
            
            elif event.content.type == ContentType.INTROSPECTION:
                actions.append("dominance_adjusted")
            
            if event.ignition_strength > 0.8:
                actions.append("arousal_boost")
            
            actions = [a for a in actions if a]  # Remove empty strings
            
            return ConsumerFeedback(
                self.subscription.consumer_id, event.id, True, True,
                actions, (time.time() - start) * 1000, None
            )
        except Exception as e:
            logger.error(f"Affect consumer error: {e}", exc_info=True)
            return ConsumerFeedback(
                self.subscription.consumer_id, event.id, True, False,
                [], (time.time() - start) * 1000, str(e)
            )


class MetaCognitionConsumer(WorkspaceConsumer):
    """Meta-cognition observer (receives all broadcasts)."""
    
    def __init__(self, min_ignition: float = 0.0):
        subscription = BroadcastSubscription("meta_cognition", [], min_ignition, None)
        super().__init__(subscription)
        self.broadcast_counts: dict[str, int] = {}
        self.content_type_counts: dict[ContentType, int] = {}
        self.average_ignition: float = 0.0
        self.total_broadcasts_seen: int = 0
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        start, actions = time.time(), []
        try:
            self.broadcast_counts[event.source] = self.broadcast_counts.get(event.source, 0) + 1
            self.content_type_counts[event.content.type] = self.content_type_counts.get(event.content.type, 0) + 1
            self.total_broadcasts_seen += 1
            self.average_ignition = (
                (self.average_ignition * (self.total_broadcasts_seen - 1) + event.ignition_strength) 
                / self.total_broadcasts_seen
            )
            
            actions.append("pattern_tracked")
            if event.ignition_strength < 0.2:
                actions.append("weak_broadcast_detected")
            elif event.ignition_strength > 0.9:
                actions.append("strong_broadcast_detected")
            
            return ConsumerFeedback(
                self.subscription.consumer_id, event.id, True, True,
                actions, (time.time() - start) * 1000, None
            )
        except Exception as e:
            logger.error(f"Meta-cognition consumer error: {e}", exc_info=True)
            return ConsumerFeedback(
                self.subscription.consumer_id, event.id, True, False,
                [], (time.time() - start) * 1000, str(e)
            )
    
    def get_insights(self) -> dict[str, Any]:
        """Get meta-cognitive insights about broadcast patterns."""
        return {
            "total_broadcasts_observed": self.total_broadcasts_seen,
            "average_ignition_strength": self.average_ignition,
            "source_distribution": dict(self.broadcast_counts),
            "content_type_distribution": {ct.value: count for ct, count in self.content_type_counts.items()},
            "most_active_source": max(self.broadcast_counts.items(), key=lambda x: x[1])[0]
                if self.broadcast_counts else None,
            "most_common_content_type": max(self.content_type_counts.items(), key=lambda x: x[1])[0].value
                if self.content_type_counts else None
        }
