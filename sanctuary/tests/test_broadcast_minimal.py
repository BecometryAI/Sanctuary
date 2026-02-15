"""
Minimal test of broadcast system core functionality.
This tests the broadcast module directly without importing the full cognitive stack.
"""

import asyncio
import time
from datetime import datetime

import pytest

pytestmark = pytest.mark.asyncio

from mind.cognitive_core.broadcast import (
    BroadcastEvent,
    WorkspaceContent,
    ContentType,
    BroadcastSubscription,
    ConsumerFeedback,
    WorkspaceConsumer,
    GlobalBroadcaster,
)


class TestConsumer(WorkspaceConsumer):
    """Simple test consumer."""
    
    def __init__(self, name: str, delay_ms: float = 0):
        subscription = BroadcastSubscription(
            consumer_id=name,
            content_types=[],
            min_ignition_strength=0.0
        )
        super().__init__(subscription)
        self.delay_ms = delay_ms
        self.received = []
    
    async def receive_broadcast(self, event: BroadcastEvent) -> ConsumerFeedback:
        start = time.time()
        self.received.append(event)
        
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000.0)
        
        return ConsumerFeedback(
            consumer_id=self.subscription.consumer_id,
            event_id=event.id,
            received=True,
            processed=True,
            actions_triggered=["action_1"],
            processing_time_ms=(time.time() - start) * 1000
        )


async def test_1_basic_broadcast():
    """Test 1: Basic broadcast"""
    print("Test 1: Basic broadcast...", end=" ")
    
    broadcaster = GlobalBroadcaster()
    consumer = TestConsumer("test1")
    broadcaster.register_consumer(consumer)
    
    content = WorkspaceContent(ContentType.PERCEPT, {"data": "test"})
    event = await broadcaster.broadcast(content, "source1", 0.8)
    
    assert len(consumer.received) == 1
    assert consumer.received[0].id == event.id
    print("✅")


async def test_2_parallel_broadcast():
    """Test 2: Parallel broadcast (not sequential)"""
    print("Test 2: Parallel broadcast...", end=" ")
    
    broadcaster = GlobalBroadcaster(timeout_seconds=1.0)
    
    # 3 consumers with 30ms delay each
    consumers = [TestConsumer(f"c{i}", delay_ms=30) for i in range(3)]
    for c in consumers:
        broadcaster.register_consumer(c)
    
    content = WorkspaceContent(ContentType.PERCEPT, {})
    
    start = time.time()
    await broadcaster.broadcast(content, "src", 1.0)
    elapsed_ms = (time.time() - start) * 1000
    
    # Sequential would be 90ms, parallel should be ~30ms
    assert elapsed_ms < 60, f"Took {elapsed_ms:.0f}ms (expected <60ms for parallel)"
    
    for c in consumers:
        assert len(c.received) == 1
    
    print(f"✅ ({elapsed_ms:.0f}ms)")


async def test_3_subscription_filter():
    """Test 3: Subscription filtering"""
    print("Test 3: Subscription filtering...", end=" ")
    
    broadcaster = GlobalBroadcaster()
    
    # Consumer with filter
    consumer1 = TestConsumer("percept_only")
    consumer1.subscription.content_types = [ContentType.PERCEPT]
    broadcaster.register_consumer(consumer1)
    
    # Consumer without filter
    consumer2 = TestConsumer("all")
    broadcaster.register_consumer(consumer2)
    
    # Broadcast percept
    await broadcaster.broadcast(
        WorkspaceContent(ContentType.PERCEPT, {}),
        "src", 1.0
    )
    
    # Broadcast emotion
    await broadcaster.broadcast(
        WorkspaceContent(ContentType.EMOTION, {}),
        "src", 1.0
    )
    
    assert len(consumer1.received) == 1  # Only percept
    assert len(consumer2.received) == 2  # Both
    print("✅")


async def test_4_ignition_filter():
    """Test 4: Ignition strength filtering"""
    print("Test 4: Ignition strength filtering...", end=" ")
    
    broadcaster = GlobalBroadcaster()
    
    consumer = TestConsumer("strong_only")
    consumer.subscription.min_ignition_strength = 0.7
    broadcaster.register_consumer(consumer)
    
    # Weak broadcast
    await broadcaster.broadcast(
        WorkspaceContent(ContentType.PERCEPT, {}),
        "src", 0.3
    )
    
    # Strong broadcast
    await broadcaster.broadcast(
        WorkspaceContent(ContentType.PERCEPT, {}),
        "src", 0.9
    )
    
    assert len(consumer.received) == 1  # Only strong
    assert consumer.received[0].ignition_strength == 0.9
    print("✅")


async def test_5_feedback_collection():
    """Test 5: Consumer feedback collection"""
    print("Test 5: Feedback collection...", end=" ")
    
    broadcaster = GlobalBroadcaster()
    c1 = TestConsumer("c1")
    c2 = TestConsumer("c2")
    broadcaster.register_consumer(c1)
    broadcaster.register_consumer(c2)
    
    content = WorkspaceContent(ContentType.GOAL, {})
    event = await broadcaster.broadcast(content, "src", 0.8)
    
    # Check history
    assert len(broadcaster.broadcast_history) == 1
    stored_event, feedback = broadcaster.broadcast_history[0]
    
    assert stored_event.id == event.id
    assert len(feedback) == 2
    
    ids = {fb.consumer_id for fb in feedback}
    assert ids == {"c1", "c2"}
    print("✅")


async def test_6_metrics():
    """Test 6: Broadcast metrics"""
    print("Test 6: Broadcast metrics...", end=" ")
    
    broadcaster = GlobalBroadcaster(enable_metrics=True)
    c1 = TestConsumer("c1")
    c2 = TestConsumer("c2")
    broadcaster.register_consumer(c1)
    broadcaster.register_consumer(c2)
    
    # Do 3 broadcasts
    for i in range(3):
        content = WorkspaceContent(ContentType.PERCEPT, {"i": i})
        await broadcaster.broadcast(content, "src", 0.8)
    
    metrics = broadcaster.get_metrics()
    
    assert metrics.total_broadcasts == 3
    assert metrics.avg_consumers_per_broadcast == 2.0
    assert metrics.avg_actions_triggered == 2.0  # 2 consumers * 1 action each per broadcast
    assert metrics.consumer_response_rates["c1"] == 1.0
    assert metrics.most_active_sources[0] == ("src", 3)
    print("✅")


async def main():
    print("=" * 60)
    print("BROADCAST SYSTEM CORE TESTS")
    print("=" * 60)
    print()
    
    try:
        await test_1_basic_broadcast()
        await test_2_parallel_broadcast()
        await test_3_subscription_filter()
        await test_4_ignition_filter()
        await test_5_feedback_collection()
        await test_6_metrics()
        
        print()
        print("=" * 60)
        print("✅ ALL 6 TESTS PASSED")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
