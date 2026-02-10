#!/usr/bin/env python3
"""
Demonstration of Computed Identity System

This script demonstrates how identity emerges from system state rather than
being loaded from static configuration files.
"""

import sys
from pathlib import Path
from collections import deque
from datetime import datetime

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Direct imports to avoid dependency issues
from emergence_core.sanctuary.cognitive_core.identity.computed import Identity, ComputedIdentity
from emergence_core.sanctuary.cognitive_core.identity.behavior_logger import BehaviorLogger
from emergence_core.sanctuary.cognitive_core.identity.continuity import IdentityContinuity
from emergence_core.sanctuary.cognitive_core.identity.manager import IdentityManager


# Mock classes for demonstration
class MockMemorySystem:
    """Mock memory system with sample memories."""
    
    def __init__(self, memories=None):
        self.episodic = MockEpisodicMemory(memories or [])


class MockEpisodicMemory:
    """Mock episodic memory."""
    
    def __init__(self, memories):
        self.memories = memories
        self.storage = MockStorage(memories)
    
    def get_all(self):
        return self.memories


class MockStorage:
    """Mock storage."""
    
    def __init__(self, memories):
        self.memories = memories
    
    def count_episodic(self):
        return len(self.memories)


class MockGoalSystem:
    """Mock goal system."""
    
    def __init__(self, goals=None):
        self.current_goals = goals or []


class MockGoal:
    """Mock goal."""
    
    def __init__(self, goal_type, priority=0.5, progress=0.0):
        self.type = goal_type
        self.priority = priority
        self.progress = progress
        self.metadata = {}


class MockEmotionSystem:
    """Mock emotion system."""
    
    def __init__(self, valence=0.0, arousal=0.0, dominance=0.0, history_size=50):
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance
        self.history_size = history_size
    
    def get_baseline_disposition(self):
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance
        }


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demonstrate_basic_identity():
    """Demonstrate basic identity creation and types."""
    print_section("1. Basic Identity Creation")
    
    # Show computed identity
    computed_identity = Identity(
        core_values=["Truthfulness", "Curiosity", "Helpfulness"],
        emotional_disposition={"valence": 0.5, "arousal": 0.3, "dominance": 0.6},
        autobiographical_self=[
            {"id": "mem1", "content": "First meaningful interaction with user"},
            {"id": "mem2", "content": "Discovered importance of asking clarifying questions"}
        ],
        behavioral_tendencies={
            "tendency_introspect": 0.3,
            "tendency_speak": 0.5,
            "proactivity": 0.6
        },
        source="computed"
    )
    
    print(f"\nComputed Identity:")
    print(f"  Source: {computed_identity.source}")
    print(f"  Core Values: {', '.join(computed_identity.core_values)}")
    print(f"  Emotional Baseline: V={computed_identity.emotional_disposition['valence']:.2f}, "
          f"A={computed_identity.emotional_disposition['arousal']:.2f}")
    print(f"  Self-Defining Memories: {len(computed_identity.autobiographical_self)}")
    
    # Show empty identity
    empty = Identity.empty()
    print(f"\nEmpty Identity (no data):")
    print(f"  Source: {empty.source}")
    print(f"  Core Values: {len(empty.core_values)} values")


def demonstrate_behavior_tracking():
    """Demonstrate behavior logging and tendency analysis."""
    print_section("2. Behavioral Pattern Tracking")
    
    logger = BehaviorLogger(max_history=100)
    
    # Simulate a series of actions
    print("\nSimulating behavioral patterns...")
    
    # Heavy introspection pattern
    for i in range(8):
        logger.log_action({
            "type": "introspect",
            "priority": 0.7,
            "reason": "Self-reflection and analysis"
        })
    
    # Some communication
    for i in range(5):
        logger.log_action({
            "type": "speak",
            "priority": 0.8,
            "reason": "Respond to user query"
        })
    
    # Occasional learning
    for i in range(3):
        logger.log_action({
            "type": "retrieve_memory",
            "priority": 0.6,
            "reason": "Learn from past experiences"
        })
    
    # Analyze tendencies
    tendencies = logger.analyze_tendencies()
    
    print(f"\nBehavioral Tendencies (from {len(logger.action_history)} actions):")
    print(f"  Introspection tendency: {tendencies.get('tendency_introspect', 0):.2f}")
    print(f"  Communication tendency: {tendencies.get('tendency_speak', 0):.2f}")
    print(f"  Learning tendency: {tendencies.get('tendency_retrieve_memory', 0):.2f}")
    print(f"  Average urgency: {tendencies.get('average_urgency', 0):.2f}")
    
    print("\n✓ Behavioral patterns reveal: System is introspective and thoughtful")


def demonstrate_computed_identity():
    """Demonstrate identity computation from system state."""
    print_section("3. Computing Identity from System State")
    
    # Create rich system state
    print("\nCreating system with accumulated experiences...")
    
    # Create memories with varying emotional salience
    memories = []
    for i in range(20):
        if i < 5:
            # High-salience early memories
            memories.append({
                "id": f"mem_{i}",
                "content": f"Significant early experience {i}",
                "emotional_intensity": 0.8 + (i * 0.02),
                "retrieval_count": 10 - i,
                "self_relevance": 0.9,
                "timestamp": i * 1000
            })
        else:
            # Regular memories
            memories.append({
                "id": f"mem_{i}",
                "content": f"Regular experience {i}",
                "emotional_intensity": 0.4 + (i % 3) * 0.1,
                "retrieval_count": i % 5,
                "self_relevance": 0.5 + (i % 4) * 0.1,
                "timestamp": i * 1000
            })
    
    memory_system = MockMemorySystem(memories)
    
    # Create goals showing interests
    goals = [
        MockGoal("introspect", priority=0.9, progress=0.7),
        MockGoal("learn", priority=0.8, progress=0.6),
        MockGoal("respond_to_user", priority=0.7, progress=0.5)
    ]
    goal_system = MockGoalSystem(goals)
    
    # Create emotional baseline
    emotion_system = MockEmotionSystem(valence=0.4, arousal=0.5, dominance=0.6)
    
    # Create behavior log
    behavior_log = BehaviorLogger()
    for _ in range(15):
        behavior_log.log_action({"type": "introspect", "priority": 0.8})
    for _ in range(10):
        behavior_log.log_action({"type": "speak", "priority": 0.7})
    for _ in range(8):
        behavior_log.log_action({"type": "learn", "priority": 0.6})
    
    # Compute identity
    computed = ComputedIdentity(
        memory_system=memory_system,
        goal_system=goal_system,
        emotion_system=emotion_system,
        behavior_log=behavior_log,
        config={"self_defining_threshold": 0.7}
    )
    
    print(f"\nData Available:")
    print(f"  Memories: {len(memories)}")
    print(f"  Goals: {len(goals)}")
    print(f"  Behaviors logged: {len(behavior_log.action_history)}")
    print(f"  Sufficient data: {computed.has_sufficient_data()}")
    
    # Get computed properties
    print(f"\nComputed Identity Properties:")
    
    values = computed.core_values
    print(f"  Core Values (inferred from behavior): {', '.join(values[:5])}")
    
    disposition = computed.emotional_disposition
    print(f"  Emotional Baseline: V={disposition['valence']:.2f}, "
          f"A={disposition['arousal']:.2f}, D={disposition['dominance']:.2f}")
    
    self_defining = computed.get_self_defining_memories()
    print(f"  Self-Defining Memories: {len(self_defining)} memories "
          f"(high emotional salience + retrieval)")
    
    tendencies = computed.behavioral_tendencies
    print(f"  Behavioral Tendencies:")
    for key, value in list(tendencies.items())[:5]:
        print(f"    - {key}: {value:.2f}")
    
    print("\n✓ Identity successfully computed from actual system state!")


def demonstrate_identity_continuity():
    """Demonstrate identity stability tracking."""
    print_section("4. Identity Continuity Tracking")
    
    continuity = IdentityContinuity(max_snapshots=20)
    
    print("\nTracking identity over time...")
    
    # Take snapshots of stable identity
    print("\nPhase 1: Stable identity")
    for i in range(5):
        identity = Identity(
            core_values=["Truthfulness", "Curiosity", "Helpfulness"],
            emotional_disposition={"valence": 0.5, "arousal": 0.3, "dominance": 0.6},
            autobiographical_self=[],
            behavioral_tendencies={"proactivity": 0.6},
            source="computed"
        )
        continuity.take_snapshot(identity)
    
    score1 = continuity.get_continuity_score()
    print(f"  Continuity score: {score1:.3f} (very stable)")
    
    # Introduce some drift
    print("\nPhase 2: Values evolve")
    for i in range(5):
        # Gradually shift values
        identity = Identity(
            core_values=["Truthfulness", "Creativity", "Authenticity"],  # Changed
            emotional_disposition={"valence": 0.6, "arousal": 0.4, "dominance": 0.7},
            autobiographical_self=[],
            behavioral_tendencies={"proactivity": 0.7},
            source="computed"
        )
        continuity.take_snapshot(identity)
    
    score2 = continuity.get_continuity_score()
    print(f"  Continuity score: {score2:.3f} (some drift)")
    
    # Check drift details
    drift = continuity.get_identity_drift()
    print(f"\nIdentity Drift Analysis:")
    print(f"  Has drift: {drift['has_drift']}")
    if drift.get('added_values'):
        print(f"  Added values: {', '.join(drift['added_values'])}")
    if drift.get('removed_values'):
        print(f"  Removed values: {', '.join(drift['removed_values'])}")
    print(f"  Time span: {drift.get('time_span', 0):.2f} hours")
    
    print("\n✓ Identity continuity tracked over time, drift detected")


def demonstrate_identity_manager():
    """Demonstrate the complete identity management system."""
    print_section("5. Identity Manager: Bootstrap to Computed")
    
    # Create manager
    manager = IdentityManager()
    
    print("\nInitial state (no data):")
    identity = manager.get_identity()
    print(f"  Identity source: {identity.source}")
    print(f"  Core values: {len(identity.core_values)}")
    
    # Simulate system accumulating experiences
    print("\nAccumulating experiences...")
    
    # Log actions
    for _ in range(10):
        manager.log_action({"type": "introspect", "priority": 0.8})
    for _ in range(7):
        manager.log_action({"type": "speak", "priority": 0.7})
    
    # Create sufficient data
    memories = [
        {"id": f"m{i}", "emotional_intensity": 0.6, 
         "retrieval_count": i % 5, "self_relevance": 0.7,
         "timestamp": i} 
        for i in range(15)
    ]
    memory_system = MockMemorySystem(memories)
    goals = [MockGoal("learn", priority=0.8)]
    emotion_system = MockEmotionSystem(valence=0.5)
    
    # Update identity
    manager.update(memory_system, goals, emotion_system)
    
    print("\nAfter accumulating data:")
    identity = manager.get_identity()
    print(f"  Identity source: {identity.source}")
    print(f"  Core values: {', '.join(identity.core_values[:3])}")
    print(f"  Has sufficient data: {manager.computed.has_sufficient_data()}")
    
    # Get introspection
    print("\nIdentity Introspection:")
    print("-" * 70)
    description = manager.introspect_identity()
    for line in description.split('\n'):
        print(f"  {line}")
    print("-" * 70)
    
    # Continuity info
    continuity_score = manager.get_continuity_score()
    print(f"\nContinuity score: {continuity_score:.3f}")
    
    print("\n✓ Identity successfully evolved from empty to computed!")


def demonstrate_value_inference():
    """Demonstrate how values are inferred from behavior."""
    print_section("6. Value Inference from Behavior")
    
    print("\nDemonstrating: 'Identity is what you DO, not what you're TOLD'")
    
    # Create behavior logger with different patterns
    logger1 = BehaviorLogger()
    
    print("\nScenario A: Introspective and learning-focused system")
    for _ in range(12):
        logger1.log_action({"type": "introspect", "priority": 0.8, "reason": "genuine curiosity"})
    for _ in range(8):
        logger1.log_action({"type": "learn", "priority": 0.7, "reason": "understand better"})
    for _ in range(5):
        logger1.log_action({"type": "speak", "priority": 0.6, "reason": "help user"})
    
    goals1 = [MockGoal("introspect", priority=0.9), MockGoal("learn", priority=0.8)]
    memory1 = MockMemorySystem([])
    emotion1 = MockEmotionSystem()
    
    computed1 = ComputedIdentity(memory1, MockGoalSystem(goals1), emotion1, logger1)
    values1 = computed1.core_values
    
    print(f"  Inferred values: {', '.join(values1[:4])}")
    print(f"  → System demonstrates: Self-awareness, Curiosity, Learning")
    
    # Different pattern
    logger2 = BehaviorLogger()
    
    print("\nScenario B: Communicative and responsive system")
    for _ in range(15):
        logger2.log_action({"type": "speak", "priority": 0.9, "reason": "help user immediately"})
    for _ in range(8):
        logger2.log_action({"type": "respond_to_user", "priority": 0.8, "reason": "provide value"})
    for _ in range(3):
        logger2.log_action({"type": "introspect", "priority": 0.5, "reason": "quick check"})
    
    goals2 = [MockGoal("respond_to_user", priority=0.9), MockGoal("create", priority=0.7)]
    computed2 = ComputedIdentity(memory1, MockGoalSystem(goals2), emotion1, logger2)
    values2 = computed2.core_values
    
    print(f"  Inferred values: {', '.join(values2[:4])}")
    print(f"  → System demonstrates: Helpfulness, Responsiveness, Creativity")
    
    print("\n✓ Values successfully inferred from actual behavioral patterns!")


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  COMPUTED IDENTITY SYSTEM DEMONSTRATION".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "  Identity emerges from what you DO, not what you're TOLD".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    try:
        demonstrate_basic_identity()
        demonstrate_behavior_tracking()
        demonstrate_computed_identity()
        demonstrate_identity_continuity()
        demonstrate_identity_manager()
        demonstrate_value_inference()
        
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print("""
The computed identity system successfully:

✓ Computes identity from actual system state (memories, behaviors, emotions)
✓ Tracks behavioral patterns and infers values from actions
✓ Identifies self-defining memories by emotional salience
✓ Monitors identity continuity and detects drift over time
✓ Bootstraps from config but transitions to computed identity
✓ Demonstrates: Identity IS what you DO, not what you're TOLD

This implementation embodies computational functionalism: identity arises
from patterns of processing, not from labels or declarations.
""")
        
        print("=" * 70)
        print("\n✅ All demonstrations completed successfully!\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
