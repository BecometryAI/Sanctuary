"""
Example: Using the Global Workspace Theory Broadcast System

This example demonstrates the key features of the broadcast system:
1. Creating a broadcaster
2. Registering consumers with different subscriptions
3. Broadcasting various content types
4. Collecting and analyzing feedback
5. Getting metrics
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "emergence_core" / "lyra" / "cognitive_core"))

from broadcast import (
    GlobalBroadcaster,
    WorkspaceContent,
    ContentType,
    BroadcastSubscription,
    ConsumerFeedback,
    WorkspaceConsumer,
)


# Example 1: Create a custom consumer
class NotificationConsumer(WorkspaceConsumer):
    """A consumer that generates notifications for important broadcasts."""
    
    def __init__(self):
        # Subscribe only to high-priority content
        subscription = BroadcastSubscription(
            consumer_id="notifications",
            content_types=[ContentType.GOAL, ContentType.EMOTION],
            min_ignition_strength=0.7,  # Only strong broadcasts
            source_filter=None
        )
        super().__init__(subscription)
        self.notifications = []
    
    async def receive_broadcast(self, event):
        import time
        start = time.time()
        
        # Generate notification
        if event.content.type == ContentType.GOAL:
            notification = f"📌 New high-priority goal: {event.content.data.get('description', 'N/A')}"
        elif event.content.type == ContentType.EMOTION:
            emotion = event.content.data
            if emotion.get('arousal', 0) > 0.8:
                notification = f"⚡ High arousal detected: {emotion}"
            else:
                notification = f"💭 Emotional shift: {emotion}"
        else:
            notification = f"📣 Broadcast from {event.source}"
        
        self.notifications.append(notification)
        
        return ConsumerFeedback(
            consumer_id=self.subscription.consumer_id,
            event_id=event.id,
            received=True,
            processed=True,
            actions_triggered=["notification_generated"],
            processing_time_ms=(time.time() - start) * 1000
        )


# Example 2: Create a logging consumer
class LoggingConsumer(WorkspaceConsumer):
    """A consumer that logs all broadcasts."""
    
    def __init__(self):
        # Subscribe to everything
        subscription = BroadcastSubscription(
            consumer_id="logger",
            content_types=[],  # Empty = accept all
            min_ignition_strength=0.0,  # Accept all strengths
            source_filter=None
        )
        super().__init__(subscription)
        self.log = []
    
    async def receive_broadcast(self, event):
        import time
        start = time.time()
        
        # Log the broadcast
        log_entry = {
            "timestamp": event.timestamp.isoformat(),
            "event_id": event.id,
            "source": event.source,
            "content_type": event.content.type.value,
            "ignition_strength": event.ignition_strength
        }
        self.log.append(log_entry)
        
        return ConsumerFeedback(
            consumer_id=self.subscription.consumer_id,
            event_id=event.id,
            received=True,
            processed=True,
            actions_triggered=["logged"],
            processing_time_ms=(time.time() - start) * 1000
        )


async def main():
    """Run the example."""
    
    print("=" * 70)
    print("BROADCAST SYSTEM EXAMPLE")
    print("=" * 70)
    print()
    
    # Step 1: Create broadcaster
    print("Step 1: Creating broadcaster...")
    broadcaster = GlobalBroadcaster(
        timeout_seconds=0.5,
        max_history=50,
        enable_metrics=True
    )
    print("✅ Broadcaster created")
    print()
    
    # Step 2: Register consumers
    print("Step 2: Registering consumers...")
    notification_consumer = NotificationConsumer()
    logging_consumer = LoggingConsumer()
    
    broadcaster.register_consumer(notification_consumer)
    broadcaster.register_consumer(logging_consumer)
    
    print(f"✅ Registered {len(broadcaster.consumers)} consumers:")
    print(f"   - {notification_consumer.subscription.consumer_id}")
    print(f"   - {logging_consumer.subscription.consumer_id}")
    print()
    
    # Step 3: Broadcast various content
    print("Step 3: Broadcasting content...")
    print()
    
    # Broadcast 1: Low-priority goal (notification consumer will filter it out)
    print("  Broadcasting low-priority goal...")
    goal_content = WorkspaceContent(
        ContentType.GOAL,
        {"description": "Background task", "priority": 0.3}
    )
    await broadcaster.broadcast(goal_content, "planner", 0.4)
    print("  ✓ Low-priority goal broadcast")
    
    # Broadcast 2: High-priority goal (notification consumer will receive it)
    print("  Broadcasting high-priority goal...")
    goal_content = WorkspaceContent(
        ContentType.GOAL,
        {"description": "Respond to urgent user request", "priority": 0.9}
    )
    await broadcaster.broadcast(goal_content, "planner", 0.9)
    print("  ✓ High-priority goal broadcast")
    
    # Broadcast 3: High-arousal emotion (notification consumer will receive it)
    print("  Broadcasting high-arousal emotion...")
    emotion_content = WorkspaceContent(
        ContentType.EMOTION,
        {"valence": -0.5, "arousal": 0.9, "dominance": 0.3}
    )
    await broadcaster.broadcast(emotion_content, "affect", 0.85)
    print("  ✓ High-arousal emotion broadcast")
    
    # Broadcast 4: Percept (notification consumer will filter it out)
    print("  Broadcasting percept...")
    percept_content = WorkspaceContent(
        ContentType.PERCEPT,
        {"text": "Routine sensor data"}
    )
    await broadcaster.broadcast(percept_content, "perception", 0.6)
    print("  ✓ Percept broadcast")
    
    print()
    
    # Step 4: Check notifications
    print("Step 4: Checking notifications...")
    print()
    print("Notifications generated:")
    for notification in notification_consumer.notifications:
        print(f"  {notification}")
    print()
    
    # Step 5: Check logs
    print("Step 5: Checking logs...")
    print()
    print(f"Total events logged: {len(logging_consumer.log)}")
    for entry in logging_consumer.log:
        print(f"  [{entry['timestamp'][:19]}] {entry['content_type']} from {entry['source']} (ignition={entry['ignition_strength']:.2f})")
    print()
    
    # Step 6: Get metrics
    print("Step 6: Getting broadcast metrics...")
    print()
    metrics = broadcaster.get_metrics()
    
    print("Broadcast Metrics:")
    print(f"  Total broadcasts: {metrics.total_broadcasts}")
    print(f"  Avg consumers per broadcast: {metrics.avg_consumers_per_broadcast:.1f}")
    print(f"  Avg actions per broadcast: {metrics.avg_actions_triggered:.1f}")
    print()
    
    print("Consumer Response Rates:")
    for consumer_id, rate in metrics.consumer_response_rates.items():
        print(f"  {consumer_id}: {rate*100:.0f}%")
    print()
    
    print("Most Active Sources:")
    for source, count in metrics.most_active_sources:
        print(f"  {source}: {count} broadcasts")
    print()
    
    # Step 7: Get recent history
    print("Step 7: Getting recent broadcast history...")
    print()
    recent = broadcaster.get_recent_history(count=3)
    print(f"Last {len(recent)} broadcasts:")
    for event, feedback_list in recent:
        print(f"  Event {event.id[:12]}... from {event.source}:")
        print(f"    Content type: {event.content.type.value}")
        print(f"    Ignition: {event.ignition_strength:.2f}")
        print(f"    Consumers: {len(feedback_list)}")
        for fb in feedback_list:
            status = "✓" if fb.processed else "✗"
            print(f"      {status} {fb.consumer_id}: {', '.join(fb.actions_triggered)}")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✅ Demonstrated:")
    print("  1. Custom consumer with selective subscription")
    print("  2. Logging consumer that captures all broadcasts")
    print("  3. Subscription filtering (notifications only for important content)")
    print("  4. Consumer feedback collection")
    print("  5. Broadcast metrics tracking")
    print("  6. Broadcast history retrieval")
    print()
    print("Key Insights:")
    print(f"  - Notification consumer received {len(notification_consumer.notifications)}/4 broadcasts")
    print(f"  - Logger consumer received {len(logging_consumer.log)}/4 broadcasts")
    print(f"  - Subscription filtering reduced unnecessary processing")
    print(f"  - All consumers achieved 100% success rate")
    print()
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
