"""
Tests for the Global Workspace Theory broadcast system.

Tests cover:
- Parallel broadcast execution
- Subscription filtering
- Consumer feedback collection
- Broadcast metrics
- Integration with cognitive loop
"""

import asyncio
import pytest
import time
from datetime import datetime

from mind.cognitive_core.broadcast import (
    BroadcastEvent,
    WorkspaceContent,
    ContentType,
    BroadcastSubscription,
    ConsumerFeedback,
    WorkspaceConsumer,
    GlobalBroadcaster,
)


# Mock consumer for testing
class MockConsumer(WorkspaceConsumer):
    """Mock consumer that tracks calls."""
    
    def __init__(
        self,
        consumer_id: str,
        content_types: list = None,
        min_ignition: float = 0.0,
        delay_ms: float = 0.0,
        should_fail: bool = False
    ):
        subscription = BroadcastSubscription(
            consumer_id=consumer_id,
            content_types=content_types or [],
            min_ignition_strength=min_ignition
        )
        super().__init__(subscription)
        self.received_events = []
        self.delay_ms = delay_ms
        self.should_fail = should_fail
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        """Process broadcast with optional delay or failure."""
        start = time.time()
        
        # Track that we received this
        self.received_events.append(event)
        
        # Simulate processing delay
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000.0)
        
        # Simulate failure
        if self.should_fail:
            raise ValueError("Consumer failed")
        
        processing_time = (time.time() - start) * 1000
        
        return ConsumerFeedback(
            consumer_id=self.subscription.consumer_id,
            event_id=event.id,
            received=True,
            processed=True,
            actions_triggered=["test_action"],
            processing_time_ms=processing_time
        )


class TestBroadcastEvent:
    """Test BroadcastEvent creation and properties."""
    
    def test_create_broadcast_event(self):
        """Test creating a broadcast event."""
        content = WorkspaceContent(
            type=ContentType.PERCEPT,
            data={"test": "data"}
        )
        
        event = BroadcastEvent(
            id="test-123",
            timestamp=datetime.now(),
            content=content,
            source="test_source",
            ignition_strength=0.8
        )
        
        assert event.id == "test-123"
        assert event.content.type == ContentType.PERCEPT
        assert event.source == "test_source"
        assert event.ignition_strength == 0.8
    
    def test_generate_unique_ids(self):
        """Test that generated IDs are unique."""
        id1 = BroadcastEvent.generate_id()
        id2 = BroadcastEvent.generate_id()
        
        assert id1 != id2
        assert id1.startswith("broadcast-")
        assert id2.startswith("broadcast-")


class TestBroadcastSubscription:
    """Test subscription filtering logic."""
    
    def test_accepts_all_by_default(self):
        """Test that empty subscription accepts everything."""
        sub = BroadcastSubscription(
            consumer_id="test",
            content_types=[],
            min_ignition_strength=0.0
        )
        
        event = BroadcastEvent(
            id="test",
            timestamp=datetime.now(),
            content=WorkspaceContent(ContentType.PERCEPT, {}),
            source="test",
            ignition_strength=0.5
        )
        
        assert sub.accepts(event)
    
    def test_filters_by_content_type(self):
        """Test content type filtering."""
        sub = BroadcastSubscription(
            consumer_id="test",
            content_types=[ContentType.PERCEPT, ContentType.GOAL],
            min_ignition_strength=0.0
        )
        
        percept_event = BroadcastEvent(
            id="test1",
            timestamp=datetime.now(),
            content=WorkspaceContent(ContentType.PERCEPT, {}),
            source="test",
            ignition_strength=0.5
        )
        
        emotion_event = BroadcastEvent(
            id="test2",
            timestamp=datetime.now(),
            content=WorkspaceContent(ContentType.EMOTION, {}),
            source="test",
            ignition_strength=0.5
        )
        
        assert sub.accepts(percept_event)
        assert not sub.accepts(emotion_event)
    
    def test_filters_by_ignition_strength(self):
        """Test ignition strength filtering."""
        sub = BroadcastSubscription(
            consumer_id="test",
            content_types=[],
            min_ignition_strength=0.5
        )
        
        weak_event = BroadcastEvent(
            id="test1",
            timestamp=datetime.now(),
            content=WorkspaceContent(ContentType.PERCEPT, {}),
            source="test",
            ignition_strength=0.3
        )
        
        strong_event = BroadcastEvent(
            id="test2",
            timestamp=datetime.now(),
            content=WorkspaceContent(ContentType.PERCEPT, {}),
            source="test",
            ignition_strength=0.7
        )
        
        assert not sub.accepts(weak_event)
        assert sub.accepts(strong_event)
    
    def test_filters_by_source(self):
        """Test source filtering."""
        sub = BroadcastSubscription(
            consumer_id="test",
            content_types=[],
            min_ignition_strength=0.0,
            source_filter=["attention", "memory"]
        )
        
        attention_event = BroadcastEvent(
            id="test1",
            timestamp=datetime.now(),
            content=WorkspaceContent(ContentType.PERCEPT, {}),
            source="attention",
            ignition_strength=0.5
        )
        
        action_event = BroadcastEvent(
            id="test2",
            timestamp=datetime.now(),
            content=WorkspaceContent(ContentType.PERCEPT, {}),
            source="action",
            ignition_strength=0.5
        )
        
        assert sub.accepts(attention_event)
        assert not sub.accepts(action_event)


@pytest.mark.asyncio
class TestGlobalBroadcaster:
    """Test the GlobalBroadcaster."""
    
    async def test_register_and_unregister_consumers(self):
        """Test consumer registration."""
        broadcaster = GlobalBroadcaster()
        consumer = MockConsumer("test1")
        
        broadcaster.register_consumer(consumer)
        assert len(broadcaster.consumers) == 1
        
        removed = broadcaster.unregister_consumer("test1")
        assert removed
        assert len(broadcaster.consumers) == 0
        
        # Try removing non-existent consumer
        removed = broadcaster.unregister_consumer("nonexistent")
        assert not removed
    
    async def test_parallel_broadcast(self):
        """Test that consumers receive broadcasts in parallel."""
        broadcaster = GlobalBroadcaster(timeout_seconds=1.0)
        
        # Create consumers with delays
        consumer1 = MockConsumer("consumer1", delay_ms=50)
        consumer2 = MockConsumer("consumer2", delay_ms=50)
        consumer3 = MockConsumer("consumer3", delay_ms=50)
        
        broadcaster.register_consumer(consumer1)
        broadcaster.register_consumer(consumer2)
        broadcaster.register_consumer(consumer3)
        
        # Broadcast
        content = WorkspaceContent(ContentType.PERCEPT, {"test": "data"})
        
        start = time.time()
        event = await broadcaster.broadcast(content, "test_source", 0.8)
        elapsed = (time.time() - start) * 1000
        
        # If parallel, should take ~50ms, not 150ms (3 * 50ms)
        assert elapsed < 100, f"Took {elapsed}ms, expected < 100ms (parallel execution)"
        
        # All consumers should have received the event
        assert len(consumer1.received_events) == 1
        assert len(consumer2.received_events) == 1
        assert len(consumer3.received_events) == 1
        
        # All should have the same event ID
        assert consumer1.received_events[0].id == event.id
        assert consumer2.received_events[0].id == event.id
        assert consumer3.received_events[0].id == event.id
    
    async def test_subscription_filtering(self):
        """Test that subscription filters work."""
        broadcaster = GlobalBroadcaster()
        
        # Consumer that only accepts percepts
        percept_consumer = MockConsumer(
            "percept_only",
            content_types=[ContentType.PERCEPT]
        )
        
        # Consumer that accepts all
        all_consumer = MockConsumer("accept_all", content_types=[])
        
        broadcaster.register_consumer(percept_consumer)
        broadcaster.register_consumer(all_consumer)
        
        # Broadcast a percept
        percept_content = WorkspaceContent(ContentType.PERCEPT, {"test": "percept"})
        await broadcaster.broadcast(percept_content, "test", 0.8)
        
        # Broadcast an emotion
        emotion_content = WorkspaceContent(ContentType.EMOTION, {"valence": 0.5})
        await broadcaster.broadcast(emotion_content, "test", 0.8)
        
        # Percept consumer should have received only 1 event
        assert len(percept_consumer.received_events) == 1
        assert percept_consumer.received_events[0].content.type == ContentType.PERCEPT
        
        # All consumer should have received both
        assert len(all_consumer.received_events) == 2
    
    async def test_ignition_strength_filtering(self):
        """Test filtering by ignition strength."""
        broadcaster = GlobalBroadcaster()
        
        # Consumer that only accepts strong broadcasts
        strong_consumer = MockConsumer("strong_only", min_ignition=0.7)
        
        broadcaster.register_consumer(strong_consumer)
        
        # Weak broadcast
        weak_content = WorkspaceContent(ContentType.PERCEPT, {"test": "weak"})
        await broadcaster.broadcast(weak_content, "test", 0.3)
        
        # Strong broadcast
        strong_content = WorkspaceContent(ContentType.PERCEPT, {"test": "strong"})
        await broadcaster.broadcast(strong_content, "test", 0.9)
        
        # Should only receive strong broadcast
        assert len(strong_consumer.received_events) == 1
        assert strong_consumer.received_events[0].ignition_strength == 0.9
    
    async def test_consumer_feedback_collection(self):
        """Test that feedback is collected from consumers."""
        broadcaster = GlobalBroadcaster()
        
        consumer1 = MockConsumer("consumer1")
        consumer2 = MockConsumer("consumer2")
        
        broadcaster.register_consumer(consumer1)
        broadcaster.register_consumer(consumer2)
        
        content = WorkspaceContent(ContentType.PERCEPT, {"test": "data"})
        event = await broadcaster.broadcast(content, "test", 0.8)
        
        # Check history
        assert len(broadcaster.broadcast_history) == 1
        stored_event, feedback_list = broadcaster.broadcast_history[0]
        
        assert stored_event.id == event.id
        assert len(feedback_list) == 2
        
        # Check feedback properties
        consumer_ids = {fb.consumer_id for fb in feedback_list}
        assert consumer_ids == {"consumer1", "consumer2"}
        
        for fb in feedback_list:
            assert fb.received
            assert fb.processed
            assert "test_action" in fb.actions_triggered
    
    async def test_consumer_timeout_handling(self):
        """Test handling of slow consumers."""
        broadcaster = GlobalBroadcaster(timeout_seconds=0.05)  # 50ms timeout
        
        # Consumer that takes too long
        slow_consumer = MockConsumer("slow", delay_ms=100)
        
        broadcaster.register_consumer(slow_consumer)
        
        content = WorkspaceContent(ContentType.PERCEPT, {"test": "data"})
        event = await broadcaster.broadcast(content, "test", 0.8)
        
        # Should have feedback even though consumer timed out
        _, feedback_list = broadcaster.broadcast_history[0]
        assert len(feedback_list) == 1
        
        fb = feedback_list[0]
        assert fb.received
        assert not fb.processed  # Timeout means not processed
        assert fb.error == "Timeout"
    
    async def test_consumer_error_handling(self):
        """Test handling of consumer errors."""
        broadcaster = GlobalBroadcaster()
        
        # Consumer that fails
        failing_consumer = MockConsumer("failing", should_fail=True)
        working_consumer = MockConsumer("working")
        
        broadcaster.register_consumer(failing_consumer)
        broadcaster.register_consumer(working_consumer)
        
        content = WorkspaceContent(ContentType.PERCEPT, {"test": "data"})
        event = await broadcaster.broadcast(content, "test", 0.8)
        
        # Should have feedback from both
        _, feedback_list = broadcaster.broadcast_history[0]
        assert len(feedback_list) == 2
        
        # Find the failing consumer's feedback
        failing_fb = next(fb for fb in feedback_list if fb.consumer_id == "failing")
        working_fb = next(fb for fb in feedback_list if fb.consumer_id == "working")
        
        assert failing_fb.received
        assert not failing_fb.processed
        assert failing_fb.error is not None
        
        assert working_fb.received
        assert working_fb.processed
        assert working_fb.error is None
    
    async def test_broadcast_metrics(self):
        """Test broadcast metrics tracking."""
        broadcaster = GlobalBroadcaster(enable_metrics=True)
        
        consumer1 = MockConsumer("consumer1")
        consumer2 = MockConsumer("consumer2")
        
        broadcaster.register_consumer(consumer1)
        broadcaster.register_consumer(consumer2)
        
        # Do several broadcasts
        for i in range(5):
            content = WorkspaceContent(ContentType.PERCEPT, {"count": i})
            await broadcaster.broadcast(content, "test_source", 0.8)
        
        # Get metrics
        metrics = broadcaster.get_metrics()
        
        assert metrics.total_broadcasts == 5
        assert metrics.avg_consumers_per_broadcast == 2.0
        assert metrics.avg_actions_triggered == 2.0  # 2 consumers x 1 action each = 2 per broadcast
        assert "consumer1" in metrics.consumer_response_rates
        assert "consumer2" in metrics.consumer_response_rates
        assert metrics.consumer_response_rates["consumer1"] == 1.0  # 100% success
        assert metrics.consumer_response_rates["consumer2"] == 1.0
        
        # Check source tracking
        assert len(metrics.most_active_sources) > 0
        assert metrics.most_active_sources[0] == ("test_source", 5)
    
    async def test_broadcast_history_limit(self):
        """Test that history is limited."""
        broadcaster = GlobalBroadcaster(max_history=3)
        
        consumer = MockConsumer("consumer1")
        broadcaster.register_consumer(consumer)
        
        # Do more broadcasts than max_history
        for i in range(5):
            content = WorkspaceContent(ContentType.PERCEPT, {"count": i})
            await broadcaster.broadcast(content, "test", 0.8)
        
        # Should only keep last 3
        assert len(broadcaster.broadcast_history) == 3
    
    async def test_get_recent_history(self):
        """Test getting recent broadcast history."""
        broadcaster = GlobalBroadcaster()
        
        consumer = MockConsumer("consumer1")
        broadcaster.register_consumer(consumer)
        
        # Do some broadcasts
        for i in range(5):
            content = WorkspaceContent(ContentType.PERCEPT, {"count": i})
            await broadcaster.broadcast(content, "test", 0.8)
        
        # Get recent history
        recent = broadcaster.get_recent_history(count=3)
        assert len(recent) == 3
        
        # Should be most recent
        event, _ = recent[-1]
        assert event.content.data["count"] == 4


@pytest.mark.asyncio
class TestParallelExecution:
    """Specific tests for parallel execution guarantees."""
    
    async def test_truly_parallel_not_sequential(self):
        """Test that execution is truly parallel, not sequential."""
        broadcaster = GlobalBroadcaster(timeout_seconds=1.0)
        
        # Create 10 consumers, each taking 20ms
        num_consumers = 10
        delay_per_consumer = 20  # ms
        
        for i in range(num_consumers):
            consumer = MockConsumer(f"consumer{i}", delay_ms=delay_per_consumer)
            broadcaster.register_consumer(consumer)
        
        # Broadcast
        content = WorkspaceContent(ContentType.PERCEPT, {"test": "parallel"})
        
        start = time.time()
        await broadcaster.broadcast(content, "test", 0.8)
        elapsed = (time.time() - start) * 1000
        
        # Sequential would take num_consumers * delay_per_consumer = 200ms
        # Parallel should take approximately delay_per_consumer = 20ms
        # Add some margin for overhead
        max_expected = delay_per_consumer * 2.5  # 50ms with overhead
        
        assert elapsed < max_expected, (
            f"Took {elapsed:.1f}ms, expected < {max_expected}ms. "
            f"This suggests sequential execution (would be ~{num_consumers * delay_per_consumer}ms)"
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
