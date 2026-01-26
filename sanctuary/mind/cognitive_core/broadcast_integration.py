"""
Broadcast Integration for Cognitive Core.

This module integrates the broadcast system into the existing cognitive architecture.
It provides a BroadcastCoordinator that manages broadcast consumers and coordinates
workspace broadcasts with the existing subsystems.

Author: Lyra Emergence Team
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from .broadcast import (
    GlobalBroadcaster,
    WorkspaceContent,
    ContentType,
    BroadcastMetrics,
)
from .broadcast_consumers import (
    MemoryConsumer,
    AttentionConsumer,
    ActionConsumer,
    AffectConsumer,
    MetaCognitionConsumer,
)

if TYPE_CHECKING:
    from .memory_integration import MemoryIntegration
    from .attention import AttentionController
    from .action import ActionSubsystem
    from .affect import AffectSubsystem
    from .workspace import GlobalWorkspace, Goal, Percept

logger = logging.getLogger(__name__)


class BroadcastCoordinator:
    """
    Coordinates broadcast integration with cognitive subsystems.
    
    This class acts as a bridge between the new broadcast system and
    the existing cognitive architecture. It manages broadcast consumers
    and provides methods to broadcast workspace updates.
    
    Key Features:
    - Registers existing subsystems as broadcast consumers
    - Broadcasts workspace state changes
    - Tracks broadcast metrics
    - Provides meta-cognitive insights
    
    Attributes:
        broadcaster: GlobalBroadcaster instance
        memory_consumer: Memory subsystem consumer
        attention_consumer: Attention subsystem consumer
        action_consumer: Action subsystem consumer
        affect_consumer: Affect subsystem consumer
        meta_consumer: Meta-cognition consumer
    """
    
    def __init__(
        self,
        workspace: 'GlobalWorkspace',
        memory: Optional['MemoryIntegration'] = None,
        attention: Optional['AttentionController'] = None,
        action: Optional['ActionSubsystem'] = None,
        affect: Optional['AffectSubsystem'] = None,
        config: Optional[dict] = None
    ):
        """
        Initialize broadcast coordinator.
        
        Args:
            workspace: GlobalWorkspace instance
            memory: MemoryIntegration instance (optional)
            attention: AttentionController instance (optional)
            action: ActionSubsystem instance (optional)
            affect: AffectSubsystem instance (optional)
            config: Configuration dict
        """
        self.workspace = workspace
        self.config = config or {}
        
        # Initialize broadcaster
        timeout = self.config.get("broadcast_timeout", 0.1)
        max_history = self.config.get("broadcast_history_size", 100)
        enable_metrics = self.config.get("broadcast_metrics", True)
        
        self.broadcaster = GlobalBroadcaster(
            timeout_seconds=timeout,
            max_history=max_history,
            enable_metrics=enable_metrics
        )
        
        # Register subsystems as consumers
        self.memory_consumer = None
        if memory:
            self.memory_consumer = MemoryConsumer(
                memory,
                min_ignition=self.config.get("memory_min_ignition", 0.3)
            )
            self.broadcaster.register_consumer(self.memory_consumer)
            logger.info("✅ Memory registered as broadcast consumer")
        
        self.attention_consumer = None
        if attention:
            self.attention_consumer = AttentionConsumer(
                attention,
                min_ignition=self.config.get("attention_min_ignition", 0.4)
            )
            self.broadcaster.register_consumer(self.attention_consumer)
            logger.info("✅ Attention registered as broadcast consumer")
        
        self.action_consumer = None
        if action:
            self.action_consumer = ActionConsumer(
                action,
                min_ignition=self.config.get("action_min_ignition", 0.5)
            )
            self.broadcaster.register_consumer(self.action_consumer)
            logger.info("✅ Action registered as broadcast consumer")
        
        self.affect_consumer = None
        if affect:
            self.affect_consumer = AffectConsumer(
                affect,
                min_ignition=self.config.get("affect_min_ignition", 0.3)
            )
            self.broadcaster.register_consumer(self.affect_consumer)
            logger.info("✅ Affect registered as broadcast consumer")
        
        # Always create meta-cognition consumer (it observes everything)
        self.meta_consumer = MetaCognitionConsumer(
            min_ignition=0.0  # Meta-cognition sees all broadcasts
        )
        self.broadcaster.register_consumer(self.meta_consumer)
        logger.info("✅ Meta-cognition registered as broadcast consumer")
        
        logger.info(f"BroadcastCoordinator initialized with {len(self.broadcaster.consumers)} consumers")
    
    async def broadcast_percept(
        self,
        percept: 'Percept',
        source: str,
        ignition_strength: float = 1.0
    ):
        """
        Broadcast a percept to all consumers.
        
        Args:
            percept: The percept to broadcast
            source: Source of the percept
            ignition_strength: How strongly this won competition
        """
        content = WorkspaceContent(
            type=ContentType.PERCEPT,
            data=percept.model_dump() if hasattr(percept, 'model_dump') else percept,
            metadata={"source": source}
        )
        
        event = await self.broadcaster.broadcast(content, source, ignition_strength)
        logger.debug(f"Broadcast percept: {event.id} from {source}")
    
    async def broadcast_goal(
        self,
        goal: 'Goal',
        source: str,
        ignition_strength: float = 0.8
    ):
        """
        Broadcast a goal to all consumers.
        
        Args:
            goal: The goal to broadcast
            source: Source of the goal
            ignition_strength: Priority/importance of the goal
        """
        content = WorkspaceContent(
            type=ContentType.GOAL,
            data=goal.model_dump() if hasattr(goal, 'model_dump') else goal,
            metadata={"source": source}
        )
        
        event = await self.broadcaster.broadcast(content, source, ignition_strength)
        logger.debug(f"Broadcast goal: {event.id} from {source}")
    
    async def broadcast_emotion(
        self,
        emotional_state: dict,
        source: str,
        ignition_strength: float = 0.7
    ):
        """
        Broadcast emotional state change to all consumers.
        
        Args:
            emotional_state: Emotional state dict (valence, arousal, dominance)
            source: Source of the emotion update
            ignition_strength: Intensity of the emotional change
        """
        content = WorkspaceContent(
            type=ContentType.EMOTION,
            data=emotional_state,
            metadata={"source": source}
        )
        
        event = await self.broadcaster.broadcast(content, source, ignition_strength)
        logger.debug(f"Broadcast emotion: {event.id} from {source}")
    
    async def broadcast_workspace_state(
        self,
        snapshot,
        source: str = "workspace",
        ignition_strength: float = 0.5
    ):
        """
        Broadcast entire workspace state to all consumers.
        
        This is the main broadcast method - when workspace state changes,
        this broadcasts the new state to all subsystems simultaneously.
        
        Args:
            snapshot: WorkspaceSnapshot of current state
            source: Source of the update
            ignition_strength: Overall salience of the state change
        """
        content = WorkspaceContent(
            type=ContentType.WORKSPACE_STATE,
            data={
                "goals": [g.model_dump() if hasattr(g, 'model_dump') else g for g in snapshot.goals],
                "percepts": snapshot.percepts,
                "emotions": snapshot.emotions,
                "memories": snapshot.memories,
                "cycle_count": snapshot.cycle_count,
            },
            metadata={
                "timestamp": snapshot.timestamp.isoformat(),
                "source": source
            }
        )
        
        event = await self.broadcaster.broadcast(content, source, ignition_strength)
        logger.debug(f"Broadcast workspace state: cycle {snapshot.cycle_count}")
    
    def get_metrics(self) -> BroadcastMetrics:
        """
        Get broadcast metrics for meta-cognition.
        
        Returns:
            BroadcastMetrics with statistics
        """
        return self.broadcaster.get_metrics()
    
    def get_meta_insights(self) -> dict:
        """
        Get meta-cognitive insights about broadcast patterns.
        
        Returns:
            Dictionary of insights from meta-consumer
        """
        return self.meta_consumer.get_insights()
    
    def get_recent_history(self, count: int = 10):
        """
        Get recent broadcast history.
        
        Args:
            count: Number of recent broadcasts to return
            
        Returns:
            List of (event, feedback) tuples
        """
        return self.broadcaster.get_recent_history(count)
