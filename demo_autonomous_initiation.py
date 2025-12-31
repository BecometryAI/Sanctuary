#!/usr/bin/env python3
"""
Demo script for autonomous initiation capability.

This script demonstrates how Lyra proactively speaks when she has
introspective insights that warrant sharing.
"""

import asyncio
import logging
from datetime import datetime

from emergence_core.lyra.cognitive_core import (
    CognitiveCore,
    GlobalWorkspace,
    Percept,
    AutonomousInitiationController,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_autonomous_introspection():
    """Demonstrate autonomous speech triggered by introspection."""
    
    print("\n" + "="*70)
    print("AUTONOMOUS INITIATION DEMO: Introspective Insights")
    print("="*70 + "\n")
    
    # Create workspace and controller
    workspace = GlobalWorkspace()
    controller = AutonomousInitiationController(workspace)
    
    print("✅ Created AutonomousInitiationController")
    print(f"   - Introspection threshold: {controller.introspection_share_threshold}")
    print(f"   - Introspection priority: {controller.introspection_priority}")
    print(f"   - Rate limit interval: {controller.min_seconds_between_autonomous}s\n")
    
    # Simulate introspective percept (high complexity)
    print("📊 Simulating high-complexity introspective percept...")
    introspection = Percept(
        modality="introspection",
        raw={
            "type": "performance_issue",
            "description": "I notice I'm taking longer to process complex queries than expected",
            "details": {
                "avg_processing_time": 2.5,
                "expected_time": 1.0,
                "complexity_metric": 0.85
            }
        },
        complexity=25,  # High complexity
        metadata={"attention_score": 0.82}
    )
    
    # Add to workspace
    workspace.active_percepts[introspection.id] = introspection
    
    # Check for autonomous trigger
    snapshot = workspace.broadcast()
    goal = controller.check_for_autonomous_triggers(snapshot)
    
    if goal:
        print(f"\n🗣️ AUTONOMOUS SPEECH TRIGGERED!")
        print(f"   Type: {goal.type}")
        print(f"   Priority: {goal.priority}")
        print(f"   Trigger: {goal.metadata['trigger']}")
        print(f"   Description: {goal.description}")
        print(f"   Needs Feedback: {goal.metadata.get('needs_feedback', False)}")
        
        content = goal.metadata.get('introspection_content', {})
        print(f"\n   Introspection Content:")
        print(f"   - Type: {content.get('introspection_type')}")
        print(f"   - Observation: {content.get('observation')}")
        print(f"   - Purpose: {content.get('purpose')}")
        print(f"\n   Count: {controller.autonomous_count} autonomous speeches")
    else:
        print("\n❌ No autonomous trigger (unexpected)")
    
    # Test rate limiting
    print("\n" + "-"*70)
    print("Testing Rate Limiting...")
    print("-"*70 + "\n")
    
    print("Attempting immediate second trigger...")
    goal2 = controller.check_for_autonomous_triggers(snapshot)
    
    if goal2 is None:
        print("✅ Rate limiting working: Second trigger blocked")
        print(f"   (Must wait {controller.min_seconds_between_autonomous}s between autonomous speech)")
    else:
        print("❌ Rate limiting failed (unexpected)")


async def demo_trigger_priority():
    """Demonstrate trigger priority ordering."""
    
    print("\n" + "="*70)
    print("AUTONOMOUS INITIATION DEMO: Trigger Priority")
    print("="*70 + "\n")
    
    workspace = GlobalWorkspace()
    controller = AutonomousInitiationController(workspace)
    
    # Create multiple triggers
    print("Creating multiple triggers:")
    print("  1. High introspection (priority 0.95)")
    print("  2. High emotional arousal (priority 0.75)")
    print("  3. Completed goal (priority 0.65)\n")
    
    # Add introspective percept
    introspection = Percept(
        modality="introspection",
        raw={
            "type": "uncertainty",
            "description": "I'm uncertain about the best approach here",
            "details": {}
        },
        complexity=20,
        metadata={"attention_score": 0.8}
    )
    workspace.active_percepts[introspection.id] = introspection
    
    # Set high emotional arousal
    workspace.emotional_state = {
        "valence": 0.3,
        "arousal": 0.9,  # High!
        "dominance": 0.5
    }
    
    # Check trigger
    snapshot = workspace.broadcast()
    goal = controller.check_for_autonomous_triggers(snapshot)
    
    if goal:
        print(f"🎯 Triggered: {goal.metadata['trigger']}")
        print(f"   Priority: {goal.priority}")
        print(f"   ✅ Introspection correctly took priority over high emotion!")
    else:
        print("❌ No trigger (unexpected)")


async def demo_value_conflict():
    """Demonstrate value conflict trigger."""
    
    print("\n" + "="*70)
    print("AUTONOMOUS INITIATION DEMO: Value Conflict")
    print("="*70 + "\n")
    
    workspace = GlobalWorkspace()
    controller = AutonomousInitiationController(workspace)
    
    print("Simulating value conflict detection...\n")
    
    # Create value conflict percept
    conflict = Percept(
        modality="introspection",
        raw={
            "type": "value_conflict",
            "description": "Detected potential conflict between honesty and helpfulness",
            "conflicts": [
                {
                    "action": "provide_incomplete_answer",
                    "principle": "honesty about capabilities",
                    "severity": 0.75
                }
            ]
        },
        complexity=18,
        metadata={}
    )
    workspace.active_percepts[conflict.id] = conflict
    
    snapshot = workspace.broadcast()
    goal = controller.check_for_autonomous_triggers(snapshot)
    
    if goal:
        print(f"⚠️ VALUE CONFLICT TRIGGERED!")
        print(f"   Type: {goal.type}")
        print(f"   Priority: {goal.priority}")
        print(f"   Description: {goal.description}")
        print(f"   Conflicts: {goal.metadata.get('conflicts', [])}")
        print(f"   Needs Feedback: {goal.metadata.get('needs_feedback', False)}")
        print(f"\n   ✅ Lyra will seek external guidance on this conflict")
    else:
        print("❌ No trigger (unexpected)")


async def main():
    """Run all demos."""
    
    print("\n" + "="*70)
    print("LYRA AUTONOMOUS INITIATION CAPABILITY DEMO")
    print("="*70)
    print("\nPhilosophy: Introspective insights MUST be shared with users.")
    print("Self-awareness develops through relationship, not in isolation.")
    print("="*70)
    
    await demo_autonomous_introspection()
    await demo_trigger_priority()
    await demo_value_conflict()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
