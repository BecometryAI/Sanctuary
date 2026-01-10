"""
Standalone tests for the broadcast system that don't require full dependencies.

These tests verify core broadcast functionality without needing the entire
cognitive architecture to be initialized.
"""

import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import only what we need
from emergence_core.lyra.cognitive_core.broadcast import (
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


async def test_basic_broadcast():
    """Test basic broadcast functionality."""
    print("Testing basic broadcast...")
    
    broadcaster = GlobalBroadcaster()
    consumer = MockConsumer("test_consumer")
    broadcaster.register_consumer(consumer)
    
    content = WorkspaceContent(ContentType.PERCEPT, {"test": "data"})
    event = await broadcaster.broadcast(content, "test_source", 0.8)
    
    assert len(consumer.received_events) == 1
    assert consumer.received_events[0].id == event.id
    print("✅ Basic broadcast works")


async def test_parallel_execution():
    """Test that consumers receive broadcasts in parallel."""
    print("\nTesting parallel execution...")
    
    broadcaster = GlobalBroadcaster(timeout_seconds=1.0)
    
    # Create 3 consumers with 50ms delay each
    consumers = [MockConsumer(f"consumer{i}", delay_ms=50) for i in range(3)]
    for consumer in consumers:
        broadcaster.register_consumer(consumer)
    
    content = WorkspaceContent(ContentType.PERCEPT, {"test": "parallel"})
    
    start = time.time()
    event = await broadcaster.broadcast(content, "test", 0.8)
    elapsed = (time.time() - start) * 1000
    
    # If parallel, should take ~50ms, not 150ms (3 * 50ms)
    assert elapsed < 100, f"Took {elapsed}ms, expected < 100ms (parallel execution)"
    
    # All consumers should have received
    for consumer in consumers:
        assert len(consumer.received_events) == 1
    
    print(f"✅ Parallel execution works ({elapsed:.1f}ms for 3 consumers with 50ms delay each)")


async def test_subscription_filtering():
    """Test subscription filtering."""
    print("\nTesting subscription filtering...")
    
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
    
    # Percept consumer should have received only 1
    assert len(percept_consumer.received_events) == 1
    assert percept_consumer.received_events[0].content.type == ContentType.PERCEPT
    
    # All consumer should have received both
    assert len(all_consumer.received_events) == 2
    
    print("✅ Subscription filtering works")


async def test_ignition_strength_filtering():
    """Test filtering by ignition strength."""
    print("\nTesting ignition strength filtering...")
    
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
    
    print("✅ Ignition strength filtering works")


async def test_consumer_feedback():
    """Test consumer feedback collection."""
    print("\nTesting consumer feedback...")
    
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
    
    # Check feedback
    consumer_ids = {fb.consumer_id for fb in feedback_list}
    assert consumer_ids == {"consumer1", "consumer2"}
    
    for fb in feedback_list:
        assert fb.received
        assert fb.processed
        assert "test_action" in fb.actions_triggered
    
    print("✅ Consumer feedback collection works")


async def test_broadcast_metrics():
    """Test broadcast metrics tracking."""
    print("\nTesting broadcast metrics...")
    
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
    assert metrics.avg_actions_triggered == 1.0
    assert "consumer1" in metrics.consumer_response_rates
    assert metrics.consumer_response_rates["consumer1"] == 1.0
    assert metrics.most_active_sources[0] == ("test_source", 5)
    
    print("✅ Broadcast metrics tracking works")
    print(f"   Total broadcasts: {metrics.total_broadcasts}")
    print(f"   Avg consumers: {metrics.avg_consumers_per_broadcast}")
    print(f"   Avg actions: {metrics.avg_actions_triggered}")


async def test_error_handling():
    """Test error handling."""
    print("\nTesting error handling...")
    
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
    
    # Find feedback
    failing_fb = next(fb for fb in feedback_list if fb.consumer_id == "failing")
    working_fb = next(fb for fb in feedback_list if fb.consumer_id == "working")
    
    assert failing_fb.received
    assert not failing_fb.processed
    assert failing_fb.error is not None
    
    assert working_fb.received
    assert working_fb.processed
    assert working_fb.error is None
    
    print("✅ Error handling works (failing consumer doesn't block others)")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("BROADCAST SYSTEM STANDALONE TESTS")
    print("=" * 60)
    
    try:
        await test_basic_broadcast()
        await test_parallel_execution()
        await test_subscription_filtering()
        await test_ignition_strength_filtering()
        await test_consumer_feedback()
        await test_broadcast_metrics()
        await test_error_handling()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
