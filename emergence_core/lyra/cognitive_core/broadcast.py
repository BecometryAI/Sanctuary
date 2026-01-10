"""
Global Workspace Theory Broadcast System.

This module implements genuine broadcast dynamics based on Global Workspace Theory (GWT).
In GWT, broadcasting is the functional correlate of consciousness - when information
"ignites" and gets broadcast, that IS the moment of becoming conscious of it.

The broadcast must be explicit and parallel, not implicit state passing.

Key Features:
- Explicit BroadcastEvent model
- Parallel consumption by all subscribed consumers
- Subscription model with filters
- Consumer feedback mechanism
- Broadcast metrics tracking

Author: Lyra Emergence Team
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .workspace import WorkspaceSnapshot

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Types of content that can be broadcast."""
    PERCEPT = "percept"
    GOAL = "goal"
    MEMORY = "memory"
    EMOTION = "emotion"
    ACTION = "action"
    INTROSPECTION = "introspection"
    WORKSPACE_STATE = "workspace_state"


@dataclass
class WorkspaceContent:
    """Content being broadcast through the global workspace."""
    type: ContentType
    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate content type."""
        if isinstance(self.type, str):
            self.type = ContentType(self.type)


@dataclass
class BroadcastEvent:
    """
    Explicit broadcast to all workspace consumers.
    
    This represents the moment of consciousness - when information ignites
    and gets broadcast to all subscribed subsystems simultaneously.
    
    Attributes:
        id: Unique identifier for this broadcast
        timestamp: When the broadcast occurred
        content: What's being broadcast
        source: Which subsystem/process initiated the broadcast
        ignition_strength: How strongly this won competition (0.0-1.0)
        metadata: Additional context about the broadcast
    """
    id: str
    timestamp: datetime
    content: WorkspaceContent
    source: str
    ignition_strength: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def generate_id() -> str:
        """Generate unique broadcast ID."""
        return f"broadcast-{uuid.uuid4().hex[:12]}"


@dataclass
class BroadcastSubscription:
    """
    What a consumer wants to receive.
    
    Consumers can filter broadcasts by:
    - Content types (e.g., only percepts and goals)
    - Minimum ignition strength (ignore weak broadcasts)
    - Source filter (only from certain sources)
    
    Attributes:
        consumer_id: Unique identifier for the consumer
        content_types: List of ContentType to receive (empty = all)
        min_ignition_strength: Minimum strength to process (0.0-1.0)
        source_filter: Optional list of sources to accept (None = all)
    """
    consumer_id: str
    content_types: List[ContentType] = field(default_factory=list)
    min_ignition_strength: float = 0.0
    source_filter: Optional[List[str]] = None
    
    def accepts(self, event: BroadcastEvent) -> bool:
        """
        Check if this subscription accepts the broadcast event.
        
        Args:
            event: The broadcast event to check
            
        Returns:
            True if consumer should receive this event
        """
        # Check ignition strength threshold
        if event.ignition_strength < self.min_ignition_strength:
            return False
        
        # Check content type filter
        if self.content_types and event.content.type not in self.content_types:
            return False
        
        # Check source filter
        if self.source_filter and event.source not in self.source_filter:
            return False
        
        return True


@dataclass
class ConsumerFeedback:
    """
    What a consumer did with a broadcast.
    
    Consumers report back what they did with the broadcast, enabling
    meta-cognition to observe the effects of broadcasting.
    
    Attributes:
        consumer_id: Which consumer this feedback is from
        event_id: Which broadcast event this is about
        received: Whether the consumer received the broadcast
        processed: Whether the consumer processed it
        actions_triggered: List of actions the broadcast caused
        processing_time_ms: How long processing took
        error: Optional error message if processing failed
    """
    consumer_id: str
    event_id: str
    received: bool
    processed: bool
    actions_triggered: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class BroadcastMetrics:
    """
    Metrics tracking what broadcasting actually does.
    
    These metrics enable meta-cognition to understand the effects
    of consciousness (broadcasting) on the system.
    
    Attributes:
        total_broadcasts: Total number of broadcasts
        avg_consumers_per_broadcast: Average number of consumers reached
        avg_actions_triggered: Average actions per broadcast
        broadcast_processing_time_ms: Average time to process broadcasts
        consumer_response_rates: Success rate per consumer
        most_active_sources: Top sources by broadcast count
    """
    total_broadcasts: int = 0
    avg_consumers_per_broadcast: float = 0.0
    avg_actions_triggered: float = 0.0
    broadcast_processing_time_ms: float = 0.0
    consumer_response_rates: Dict[str, float] = field(default_factory=dict)
    most_active_sources: List[tuple[str, int]] = field(default_factory=list)


class WorkspaceConsumer(ABC):
    """
    Abstract base class for broadcast consumers.
    
    All subsystems that want to receive broadcasts must implement
    this interface. Consumers specify what they want via subscriptions
    and process broadcasts in parallel.
    
    Attributes:
        subscription: What this consumer wants to receive
    """
    
    def __init__(self, subscription: BroadcastSubscription):
        """
        Initialize consumer with subscription.
        
        Args:
            subscription: Subscription specifying what to receive
        """
        self.subscription = subscription
    
    def accepts(self, event: BroadcastEvent) -> bool:
        """
        Check if this consumer wants this broadcast.
        
        Args:
            event: The broadcast event
            
        Returns:
            True if consumer should receive this event
        """
        return self.subscription.accepts(event)
    
    @abstractmethod
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        """
        Process the broadcast and return feedback.
        
        This method is called in parallel with all other consumers.
        It should be fast and non-blocking.
        
        Args:
            event: The broadcast event to process
            
        Returns:
            Feedback about what was done with the broadcast
        """
        pass


class GlobalBroadcaster:
    """
    Implements genuine broadcast dynamics for Global Workspace Theory.
    
    The broadcaster maintains a registry of consumers and broadcasts
    workspace content to all subscribed consumers in parallel. This
    implements the core GWT principle: broadcasting is consciousness.
    
    Key Features:
    - Parallel broadcast to all consumers
    - Subscription-based filtering
    - Feedback collection for meta-cognition
    - Metrics tracking
    - Timeout handling for slow consumers
    
    Attributes:
        consumers: Registry of active consumers
        broadcast_history: Recent broadcasts and feedback
        metrics: Aggregate metrics
        config: Configuration parameters
    """
    
    def __init__(
        self,
        timeout_seconds: float = 0.1,
        max_history: int = 100,
        enable_metrics: bool = True
    ):
        """
        Initialize the global broadcaster.
        
        Args:
            timeout_seconds: Timeout for consumer processing
            max_history: Maximum broadcast history to maintain
            enable_metrics: Whether to track metrics
        """
        self.consumers: List[WorkspaceConsumer] = []
        self.broadcast_history: List[tuple[BroadcastEvent, List[ConsumerFeedback]]] = []
        self.timeout_seconds = timeout_seconds
        self.max_history = max_history
        self.enable_metrics = enable_metrics
        
        # Metrics tracking
        self._total_broadcasts = 0
        self._total_consumers_reached = 0
        self._total_actions_triggered = 0
        self._total_processing_time_ms = 0.0
        self._consumer_success_counts: Dict[str, int] = {}
        self._consumer_total_counts: Dict[str, int] = {}
        self._source_counts: Dict[str, int] = {}
        
        logger.info("GlobalBroadcaster initialized")
    
    def register_consumer(self, consumer: WorkspaceConsumer) -> None:
        """
        Register a consumer to receive broadcasts.
        
        Args:
            consumer: The consumer to register
        """
        self.consumers.append(consumer)
        logger.info(
            f"Registered consumer: {consumer.subscription.consumer_id} "
            f"(filters: {[ct.value for ct in consumer.subscription.content_types] or 'all'})"
        )
    
    def unregister_consumer(self, consumer_id: str) -> bool:
        """
        Unregister a consumer.
        
        Args:
            consumer_id: ID of consumer to unregister
            
        Returns:
            True if consumer was found and removed
        """
        initial_count = len(self.consumers)
        self.consumers = [c for c in self.consumers if c.subscription.consumer_id != consumer_id]
        removed = len(self.consumers) < initial_count
        
        if removed:
            logger.info(f"Unregistered consumer: {consumer_id}")
        
        return removed
    
    async def broadcast(
        self,
        content: WorkspaceContent,
        source: str,
        ignition_strength: float = 1.0
    ) -> BroadcastEvent:
        """
        Broadcast content to all subscribed consumers.
        
        This is the moment of consciousness - information becomes
        globally available and all consumers process it in parallel.
        
        Args:
            content: What to broadcast
            source: Where the broadcast originated
            ignition_strength: How strongly this won competition (0.0-1.0)
            
        Returns:
            The broadcast event that was sent
        """
        # Create broadcast event
        event = BroadcastEvent(
            id=BroadcastEvent.generate_id(),
            timestamp=datetime.now(),
            content=content,
            source=source,
            ignition_strength=ignition_strength,
            metadata={}
        )
        
        # Parallel broadcast to all consumers
        feedback = await self._parallel_broadcast(event)
        
        # Store in history
        self.broadcast_history.append((event, feedback))
        if len(self.broadcast_history) > self.max_history:
            self.broadcast_history.pop(0)
        
        # Update metrics
        if self.enable_metrics:
            self._update_metrics(event, feedback)
        
        logger.debug(
            f"Broadcast {event.id}: {event.content.type.value} from {source} "
            f"(strength={ignition_strength:.2f}, consumers={len(feedback)})"
        )
        
        return event
    
    async def _parallel_broadcast(self, event: BroadcastEvent) -> List[ConsumerFeedback]:
        """
        Send to all consumers in parallel.
        
        This implements the core GWT insight: broadcasting happens
        simultaneously to all consumers, not sequentially.
        
        Args:
            event: The broadcast event
            
        Returns:
            List of consumer feedback
        """
        tasks = []
        accepting_consumers = []
        
        # Create tasks for all accepting consumers
        for consumer in self.consumers:
            if consumer.accepts(event):
                accepting_consumers.append(consumer)
                tasks.append(
                    asyncio.create_task(
                        self._consume_with_timeout(consumer, event)
                    )
                )
        
        if not tasks:
            logger.debug(f"No consumers accepted broadcast {event.id}")
            return []
        
        # Wait for all consumers with timeout
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect feedback
        feedback = self._collect_feedback(event, results, accepting_consumers)
        
        return feedback
    
    async def _consume_with_timeout(
        self,
        consumer: WorkspaceConsumer,
        event: BroadcastEvent
    ) -> ConsumerFeedback:
        """
        Call consumer with timeout protection.
        
        Args:
            consumer: The consumer to call
            event: The broadcast event
            
        Returns:
            Consumer feedback
        """
        try:
            return await asyncio.wait_for(
                consumer.receive_broadcast(event),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Consumer {consumer.subscription.consumer_id} timed out "
                f"on broadcast {event.id}"
            )
            return ConsumerFeedback(
                consumer_id=consumer.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=False,
                error="Timeout"
            )
        except Exception as e:
            logger.error(
                f"Consumer {consumer.subscription.consumer_id} error "
                f"on broadcast {event.id}: {e}",
                exc_info=True
            )
            return ConsumerFeedback(
                consumer_id=consumer.subscription.consumer_id,
                event_id=event.id,
                received=True,
                processed=False,
                error=str(e)
            )
    
    def _collect_feedback(
        self,
        event: BroadcastEvent,
        results: List,
        consumers: List[WorkspaceConsumer]
    ) -> List[ConsumerFeedback]:
        """
        Aggregate feedback for meta-cognition.
        
        Args:
            event: The broadcast event
            results: Results from asyncio.gather
            consumers: List of consumers that accepted
            
        Returns:
            List of consumer feedback
        """
        feedback = []
        
        for i, result in enumerate(results):
            if isinstance(result, ConsumerFeedback):
                feedback.append(result)
            elif isinstance(result, Exception):
                # Exception during processing
                consumer_id = consumers[i].subscription.consumer_id if i < len(consumers) else "unknown"
                feedback.append(ConsumerFeedback(
                    consumer_id=consumer_id,
                    event_id=event.id,
                    received=True,
                    processed=False,
                    error=str(result)
                ))
            else:
                # Unexpected result type
                consumer_id = consumers[i].subscription.consumer_id if i < len(consumers) else "unknown"
                logger.warning(f"Unexpected result type from consumer {consumer_id}: {type(result)}")
        
        return feedback
    
    def _update_metrics(self, event: BroadcastEvent, feedback: List[ConsumerFeedback]) -> None:
        """
        Update aggregate metrics.
        
        Args:
            event: The broadcast event
            feedback: Consumer feedback
        """
        self._total_broadcasts += 1
        self._total_consumers_reached += len(feedback)
        
        # Track source
        self._source_counts[event.source] = self._source_counts.get(event.source, 0) + 1
        
        # Process feedback
        for fb in feedback:
            # Track consumer success rate
            self._consumer_total_counts[fb.consumer_id] = \
                self._consumer_total_counts.get(fb.consumer_id, 0) + 1
            
            if fb.processed and not fb.error:
                self._consumer_success_counts[fb.consumer_id] = \
                    self._consumer_success_counts.get(fb.consumer_id, 0) + 1
            
            # Track actions
            self._total_actions_triggered += len(fb.actions_triggered)
            
            # Track processing time
            self._total_processing_time_ms += fb.processing_time_ms
    
    def get_metrics(self) -> BroadcastMetrics:
        """
        Get current broadcast metrics.
        
        Returns:
            BroadcastMetrics with current statistics
        """
        # Calculate averages
        avg_consumers = (
            self._total_consumers_reached / self._total_broadcasts
            if self._total_broadcasts > 0 else 0.0
        )
        
        avg_actions = (
            self._total_actions_triggered / self._total_broadcasts
            if self._total_broadcasts > 0 else 0.0
        )
        
        avg_processing_time = (
            self._total_processing_time_ms / self._total_consumers_reached
            if self._total_consumers_reached > 0 else 0.0
        )
        
        # Calculate consumer response rates
        response_rates = {}
        for consumer_id, total in self._consumer_total_counts.items():
            success = self._consumer_success_counts.get(consumer_id, 0)
            response_rates[consumer_id] = success / total if total > 0 else 0.0
        
        # Get top sources
        most_active_sources = sorted(
            self._source_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return BroadcastMetrics(
            total_broadcasts=self._total_broadcasts,
            avg_consumers_per_broadcast=avg_consumers,
            avg_actions_triggered=avg_actions,
            broadcast_processing_time_ms=avg_processing_time,
            consumer_response_rates=response_rates,
            most_active_sources=most_active_sources
        )
    
    def get_recent_history(self, count: int = 10) -> List[tuple[BroadcastEvent, List[ConsumerFeedback]]]:
        """
        Get recent broadcast history.
        
        Args:
            count: Number of recent broadcasts to return
            
        Returns:
            List of (event, feedback) tuples
        """
        return self.broadcast_history[-count:]
    
    def clear_history(self) -> None:
        """Clear broadcast history."""
        self.broadcast_history.clear()
        logger.info("Broadcast history cleared")
