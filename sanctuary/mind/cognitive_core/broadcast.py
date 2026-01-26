"""
Global Workspace Theory Broadcast System.

Implements parallel broadcast dynamics for GWT where broadcasting is the 
functional correlate of consciousness.
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
    """Broadcast event representing a conscious moment in GWT."""
    id: str
    timestamp: datetime
    content: WorkspaceContent
    source: str
    ignition_strength: float  # 0.0-1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def generate_id() -> str:
        """Generate unique broadcast ID."""
        return f"broadcast-{uuid.uuid4().hex[:12]}"


@dataclass
class BroadcastSubscription:
    """Consumer subscription with filtering criteria."""
    consumer_id: str
    content_types: List[ContentType] = field(default_factory=list)
    min_ignition_strength: float = 0.0
    source_filter: Optional[List[str]] = None
    
    def accepts(self, event: BroadcastEvent) -> bool:
        """Check if subscription accepts this event."""
        return (event.ignition_strength >= self.min_ignition_strength and
                (not self.content_types or event.content.type in self.content_types) and
                (not self.source_filter or event.source in self.source_filter))


@dataclass
class ConsumerFeedback:
    """Consumer response to a broadcast."""
    consumer_id: str
    event_id: str
    received: bool
    processed: bool
    actions_triggered: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class BroadcastMetrics:
    """Aggregate broadcast system metrics."""
    total_broadcasts: int = 0
    avg_consumers_per_broadcast: float = 0.0
    avg_actions_triggered: float = 0.0
    broadcast_processing_time_ms: float = 0.0
    consumer_response_rates: Dict[str, float] = field(default_factory=dict)
    most_active_sources: List[tuple[str, int]] = field(default_factory=list)


class WorkspaceConsumer(ABC):
    """Abstract base for broadcast consumers."""
    
    def __init__(self, subscription: BroadcastSubscription):
        self.subscription = subscription
    
    def accepts(self, event: BroadcastEvent) -> bool:
        """Check if consumer wants this broadcast."""
        return self.subscription.accepts(event)
    
    @abstractmethod
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        """Process broadcast and return feedback. Called in parallel."""
        pass


class GlobalBroadcaster:
    """
    Parallel broadcaster implementing GWT consciousness dynamics.
    Broadcasts to all subscribed consumers simultaneously with timeout protection.
    """
    
    def __init__(
        self,
        timeout_seconds: float = 0.1,
        max_history: int = 100,
        enable_metrics: bool = True
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_history < 0:
            raise ValueError("max_history must be non-negative")
            
        self.consumers: List[WorkspaceConsumer] = []
        self.broadcast_history: List[tuple[BroadcastEvent, List[ConsumerFeedback]]] = []
        self.timeout_seconds = timeout_seconds
        self.max_history = max_history
        self.enable_metrics = enable_metrics
        
        # Metrics (only allocated if enabled)
        if enable_metrics:
            self._total_broadcasts = 0
            self._total_consumers_reached = 0
            self._total_actions_triggered = 0
            self._total_processing_time_ms = 0.0
            self._consumer_success_counts: Dict[str, int] = {}
            self._consumer_total_counts: Dict[str, int] = {}
            self._source_counts: Dict[str, int] = {}
        
        logger.info("GlobalBroadcaster initialized")
    
    def register_consumer(self, consumer: WorkspaceConsumer) -> None:
        """Register consumer to receive broadcasts."""
        self.consumers.append(consumer)
        logger.info(f"Registered: {consumer.subscription.consumer_id}")
    
    def unregister_consumer(self, consumer_id: str) -> bool:
        """Unregister consumer by ID. Returns True if found."""
        initial_len = len(self.consumers)
        self.consumers = [c for c in self.consumers if c.subscription.consumer_id != consumer_id]
        return len(self.consumers) < initial_len
    
    async def broadcast(
        self,
        content: WorkspaceContent,
        source: str,
        ignition_strength: float = 1.0
    ) -> BroadcastEvent:
        """
        Broadcast content to all subscribed consumers in parallel.
        Returns the broadcast event.
        """
        # Validate inputs
        if not 0.0 <= ignition_strength <= 1.0:
            raise ValueError(f"ignition_strength must be in [0,1], got {ignition_strength}")
        
        event = BroadcastEvent(
            id=BroadcastEvent.generate_id(),
            timestamp=datetime.now(),
            content=content,
            source=source,
            ignition_strength=ignition_strength,
            metadata={}
        )
        
        feedback = await self._parallel_broadcast(event)
        
        # Store in history (trim if needed)
        self.broadcast_history.append((event, feedback))
        if len(self.broadcast_history) > self.max_history:
            self.broadcast_history.pop(0)
        
        if self.enable_metrics:
            self._update_metrics(event, feedback)
        
        logger.debug(f"Broadcast {event.id}: {event.content.type.value} (consumers={len(feedback)})")
        return event
    
    async def _parallel_broadcast(self, event: BroadcastEvent) -> List[ConsumerFeedback]:
        """Send to all accepting consumers in parallel with timeout protection."""
        tasks = [
            asyncio.create_task(self._consume_with_timeout(c, event))
            for c in self.consumers if c.accepts(event)
        ]
        
        if not tasks:
            return []
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._collect_feedback(event, results, [c for c in self.consumers if c.accepts(event)])
    
    async def _consume_with_timeout(self, consumer: WorkspaceConsumer, event: BroadcastEvent) -> ConsumerFeedback:
        """Call consumer with timeout protection."""
        try:
            return await asyncio.wait_for(consumer.receive_broadcast(event), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning(f"{consumer.subscription.consumer_id} timed out")
            return ConsumerFeedback(consumer.subscription.consumer_id, event.id, True, False, error="Timeout")
        except Exception as e:
            logger.error(f"{consumer.subscription.consumer_id} error: {e}", exc_info=True)
            return ConsumerFeedback(consumer.subscription.consumer_id, event.id, True, False, error=str(e))
    
    def _collect_feedback(self, event: BroadcastEvent, results: List, consumers: List[WorkspaceConsumer]) -> List[ConsumerFeedback]:
        """Aggregate feedback from consumer results."""
        feedback = []
        for i, result in enumerate(results):
            if isinstance(result, ConsumerFeedback):
                feedback.append(result)
            elif isinstance(result, Exception):
                consumer_id = consumers[i].subscription.consumer_id if i < len(consumers) else "unknown"
                feedback.append(ConsumerFeedback(consumer_id, event.id, True, False, error=str(result)))
        return feedback
    
    def _update_metrics(self, event: BroadcastEvent, feedback: List[ConsumerFeedback]) -> None:
        """Update aggregate metrics efficiently."""
        self._total_broadcasts += 1
        self._total_consumers_reached += len(feedback)
        self._source_counts[event.source] = self._source_counts.get(event.source, 0) + 1
        
        for fb in feedback:
            self._consumer_total_counts[fb.consumer_id] = self._consumer_total_counts.get(fb.consumer_id, 0) + 1
            if fb.processed and not fb.error:
                self._consumer_success_counts[fb.consumer_id] = self._consumer_success_counts.get(fb.consumer_id, 0) + 1
            self._total_actions_triggered += len(fb.actions_triggered)
            self._total_processing_time_ms += fb.processing_time_ms
    
    def get_metrics(self) -> BroadcastMetrics:
        """Get current broadcast metrics."""
        if not self.enable_metrics:
            return BroadcastMetrics()
        
        avg_consumers = self._total_consumers_reached / self._total_broadcasts if self._total_broadcasts > 0 else 0.0
        avg_actions = self._total_actions_triggered / self._total_broadcasts if self._total_broadcasts > 0 else 0.0
        avg_time = self._total_processing_time_ms / self._total_consumers_reached if self._total_consumers_reached > 0 else 0.0
        
        response_rates = {
            cid: self._consumer_success_counts.get(cid, 0) / total
            for cid, total in self._consumer_total_counts.items()
        }
        
        return BroadcastMetrics(
            total_broadcasts=self._total_broadcasts,
            avg_consumers_per_broadcast=avg_consumers,
            avg_actions_triggered=avg_actions,
            broadcast_processing_time_ms=avg_time,
            consumer_response_rates=response_rates,
            most_active_sources=sorted(self._source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
    
    def get_recent_history(self, count: int = 10) -> List[tuple[BroadcastEvent, List[ConsumerFeedback]]]:
        """Get recent broadcast history."""
        return self.broadcast_history[-count:]
    
    def clear_history(self) -> None:
        """Clear broadcast history."""
        self.broadcast_history.clear()
