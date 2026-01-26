"""
Integration test for broadcast system with cognitive components.

This test demonstrates how the broadcast system integrates with
existing cognitive subsystems without requiring the full stack.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "emergence_core" / "lyra" / "cognitive_core"))

from broadcast import (
    GlobalBroadcaster,
    WorkspaceContent,
    ContentType,
    BroadcastSubscription,
    ConsumerFeedback,
    WorkspaceConsumer,
)


class MockMemorySubsystem:
    """Mock memory subsystem for testing."""
    def __init__(self):
        self.encoded_episodes = []
        self.retrievals = []
    
    def encode_episode(self, content):
        self.encoded_episodes.append(content)
    
    def retrieve_by_cue(self, content):
        # Simulate retrieval
        return self.encoded_episodes[:2] if self.encoded_episodes else []


class MockAttentionSubsystem:
    """Mock attention subsystem for testing."""
    def __init__(self):
        self.attention_adjustments = []
    
    def adjust_weights(self, factor):
        self.attention_adjustments.append(factor)


class MemoryConsumer(WorkspaceConsumer):
    """Memory system as broadcast consumer."""
    
    def __init__(self, memory_subsystem):
        subscription = BroadcastSubscription(
            consumer_id="memory",
            content_types=[],  # Accept all
            min_ignition_strength=0.3
        )
        super().__init__(subscription)
        self.memory = memory_subsystem
    
    async def receive_broadcast(self, event):
        import time
        start = time.time()
        actions = []
        
        # High ignition = encode
        if event.ignition_strength > 0.7:
            self.memory.encode_episode(event.content)
            actions.append("encoded_episode")
        
        # Percepts trigger retrieval
        if event.content.type == ContentType.PERCEPT:
            retrieved = self.memory.retrieve_by_cue(event.content)
            if retrieved:
                actions.append(f"retrieved_{len(retrieved)}_memories")
        
        return ConsumerFeedback(
            consumer_id=self.subscription.consumer_id,
            event_id=event.id,
            received=True,
            processed=True,
            actions_triggered=actions,
            processing_time_ms=(time.time() - start) * 1000
        )


class AttentionConsumer(WorkspaceConsumer):
    """Attention system as broadcast consumer."""
    
    def __init__(self, attention_subsystem):
        subscription = BroadcastSubscription(
            consumer_id="attention",
            content_types=[ContentType.PERCEPT, ContentType.EMOTION],
            min_ignition_strength=0.4
        )
        super().__init__(subscription)
        self.attention = attention_subsystem
    
    async def receive_broadcast(self, event):
        import time
        start = time.time()
        actions = []
        
        # High arousal = boost attention
        if event.content.type == ContentType.EMOTION:
            emotion_data = event.content.data
            if isinstance(emotion_data, dict):
                arousal = emotion_data.get("arousal", 0)
                if arousal > 0.7:
                    self.attention.adjust_weights(arousal)
                    actions.append("attention_boosted")
        
        # High-strength percepts
        if event.content.type == ContentType.PERCEPT and event.ignition_strength > 0.6:
            actions.append("novelty_tracked")
        
        return ConsumerFeedback(
            consumer_id=self.subscription.consumer_id,
            event_id=event.id,
            received=True,
            processed=True,
            actions_triggered=actions,
            processing_time_ms=(time.time() - start) * 1000
        )


async def test_subsystem_integration():
    """Test broadcast system with mock subsystems."""
    print("=" * 60)
    print("BROADCAST SUBSYSTEM INTEGRATION TEST")
    print("=" * 60)
    print()
    
    # Create mock subsystems
    memory_subsystem = MockMemorySubsystem()
    attention_subsystem = MockAttentionSubsystem()
    
    # Create broadcaster
    broadcaster = GlobalBroadcaster(enable_metrics=True)
    
    # Register consumers
    memory_consumer = MemoryConsumer(memory_subsystem)
    attention_consumer = AttentionConsumer(attention_subsystem)
    
    broadcaster.register_consumer(memory_consumer)
    broadcaster.register_consumer(attention_consumer)
    
    print(f"✅ Registered {len(broadcaster.consumers)} consumers")
    print()
    
    # Simulate cognitive cycle broadcasts
    print("Simulating cognitive cycle broadcasts...")
    print()
    
    # 1. Broadcast high-ignition percept (should trigger memory encoding)
    print("1. Broadcasting high-ignition percept...")
    percept_content = WorkspaceContent(
        ContentType.PERCEPT,
        {"text": "Important user question about consciousness"}
    )
    event1 = await broadcaster.broadcast(percept_content, "perception", 0.9)
    print(f"   Event {event1.id[:12]}... broadcasted")
    
    # 2. Broadcast high-arousal emotion (should boost attention)
    print("2. Broadcasting high-arousal emotion...")
    emotion_content = WorkspaceContent(
        ContentType.EMOTION,
        {"valence": 0.3, "arousal": 0.8, "dominance": 0.5}
    )
    event2 = await broadcaster.broadcast(emotion_content, "affect", 0.7)
    print(f"   Event {event2.id[:12]}... broadcasted")
    
    # 3. Broadcast goal (memory consumer sees it, attention consumer filters it out)
    print("3. Broadcasting goal...")
    goal_content = WorkspaceContent(
        ContentType.GOAL,
        {"description": "Respond to user", "priority": 0.9}
    )
    event3 = await broadcaster.broadcast(goal_content, "action", 0.8)
    print(f"   Event {event3.id[:12]}... broadcasted")
    
    # 4. Broadcast weak percept (should be filtered by thresholds)
    print("4. Broadcasting weak percept...")
    weak_content = WorkspaceContent(
        ContentType.PERCEPT,
        {"text": "Minor background update"}
    )
    event4 = await broadcaster.broadcast(weak_content, "perception", 0.2)
    print(f"   Event {event4.id[:12]}... broadcasted")
    
    print()
    
    # Check results
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    
    # Memory subsystem effects
    print("Memory Subsystem:")
    print(f"  - Episodes encoded: {len(memory_subsystem.encoded_episodes)}")
    print(f"  - Retrievals performed: {len(memory_subsystem.retrievals)}")
    print()
    
    # Attention subsystem effects
    print("Attention Subsystem:")
    print(f"  - Attention adjustments: {len(attention_subsystem.attention_adjustments)}")
    print()
    
    # Broadcast metrics
    metrics = broadcaster.get_metrics()
    print("Broadcast Metrics:")
    print(f"  - Total broadcasts: {metrics.total_broadcasts}")
    print(f"  - Avg consumers per broadcast: {metrics.avg_consumers_per_broadcast:.1f}")
    print(f"  - Avg actions triggered: {metrics.avg_actions_triggered:.1f}")
    print(f"  - Consumer response rates:")
    for consumer_id, rate in metrics.consumer_response_rates.items():
        print(f"    * {consumer_id}: {rate*100:.0f}%")
    print()
    
    # Validate expectations
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    print()
    
    checks = []
    
    # Check 1: High-ignition percept should trigger memory encoding
    checks.append((
        "High-ignition percept triggers memory encoding",
        len(memory_subsystem.encoded_episodes) >= 1
    ))
    
    # Check 2: High-arousal emotion should boost attention
    checks.append((
        "High-arousal emotion boosts attention",
        len(attention_subsystem.attention_adjustments) >= 1
    ))
    
    # Check 3: All 4 broadcasts completed
    checks.append((
        "All broadcasts completed",
        metrics.total_broadcasts == 4
    ))
    
    # Check 4: Consumers processed selectively
    checks.append((
        "Subscription filtering works",
        metrics.avg_consumers_per_broadcast < 2.0  # Not all consumers get all broadcasts
    ))
    
    # Check 5: All consumers succeeded
    checks.append((
        "All consumers succeeded",
        all(rate == 1.0 for rate in metrics.consumer_response_rates.values())
    ))
    
    # Print check results
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("✅ ALL INTEGRATION TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(test_subsystem_integration())
    exit(exit_code)
