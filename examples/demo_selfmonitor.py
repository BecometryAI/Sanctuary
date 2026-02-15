#!/usr/bin/env python3
"""
Demonstration of the SelfMonitor subsystem.

This script shows how the SelfMonitor observes cognitive state and generates
introspective percepts based on various conditions.
"""

import sys
import asyncio
from pathlib import Path

# Add emergence_core to path
sys.path.insert(0, str(Path(__file__).parent / "emergence_core"))

from mind.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType, Percept
from mind.cognitive_core.meta_cognition import SelfMonitor
from mind.cognitive_core.affect import AffectSubsystem, EmotionalState
from mind.cognitive_core.action import ActionSubsystem, Action, ActionType
from datetime import datetime


def demo_value_alignment():
    """Demonstrate value alignment checking."""
    print("\n" + "="*60)
    print("DEMO 1: Value Alignment Checking")
    print("="*60)
    
    workspace = GlobalWorkspace()
    monitor = SelfMonitor(workspace=workspace, config={"monitoring_frequency": 1})
    
    # Create an action that claims a capability
    action = Action(
        type=ActionType.SPEAK,
        metadata={"claimed_capability": True}
    )
    
    # Create snapshot with the action
    snapshot = workspace.broadcast()
    snapshot = snapshot.model_copy(update={
        "metadata": {"recent_actions": [action]}
    })
    
    print("\n📋 Scenario: Action claims capability falsely")
    print(f"   Action type: {action.type}")
    print(f"   Metadata: {action.metadata}")
    
    percepts = monitor.observe(snapshot)
    
    print(f"\n🪞 SelfMonitor generated {len(percepts)} percept(s)")
    for i, percept in enumerate(percepts, 1):
        print(f"\n   Percept {i}:")
        print(f"   - Modality: {percept.modality}")
        print(f"   - Type: {percept.raw['type']}")
        print(f"   - Description: {percept.raw['description']}")
        if 'conflicts' in percept.raw:
            print(f"   - Conflicts: {percept.raw['conflicts']}")
    
    stats = monitor.get_stats()
    print(f"\n📊 Stats: {stats['value_conflicts']} value conflicts detected")


def demo_performance_assessment():
    """Demonstrate performance assessment."""
    print("\n" + "="*60)
    print("DEMO 2: Performance Assessment")
    print("="*60)
    
    workspace = GlobalWorkspace()
    monitor = SelfMonitor(workspace=workspace, config={"monitoring_frequency": 1})
    
    # Create stalled goals
    stalled_goals = [
        Goal(
            type=GoalType.RESPOND_TO_USER,
            description=f"Stalled goal {i}",
            priority=0.8,
            progress=0.05,
            metadata={"age_cycles": 100}
        )
        for i in range(3)
    ]
    
    # Add many percepts to simulate overload
    percepts = {
        f"p{i}": Percept(
            modality="text",
            raw=f"Percept {i}",
            complexity=10
        )
        for i in range(25)
    }
    
    snapshot = workspace.broadcast()
    snapshot = snapshot.model_copy(update={
        "goals": stalled_goals,
        "percepts": {k: v.model_dump() for k, v in percepts.items()}
    })
    
    print("\n📋 Scenario: Multiple stalled goals + workspace overload")
    print(f"   Stalled goals: {len(stalled_goals)}")
    print(f"   Workspace size: {len(percepts)} percepts (threshold: 20)")
    
    percepts_generated = monitor.observe(snapshot)
    
    print(f"\n🪞 SelfMonitor generated {len(percepts_generated)} percept(s)")
    for i, percept in enumerate(percepts_generated, 1):
        print(f"\n   Percept {i}:")
        print(f"   - Type: {percept.raw['type']}")
        print(f"   - Description: {percept.raw['description']}")
        if 'issues' in percept.raw:
            for issue in percept.raw['issues']:
                print(f"     • {issue['type']}: {issue['description']}")
    
    stats = monitor.get_stats()
    print(f"\n📊 Stats: {stats['performance_issues']} performance issues detected")


def demo_uncertainty_detection():
    """Demonstrate uncertainty detection."""
    print("\n" + "="*60)
    print("DEMO 3: Uncertainty Detection")
    print("="*60)
    
    workspace = GlobalWorkspace()
    monitor = SelfMonitor(workspace=workspace, config={"monitoring_frequency": 1})
    
    # Create conflicting goals
    goals = [
        Goal(
            type=GoalType.RESPOND_TO_USER,
            description="avoid speaking to the user",
            priority=0.7
        ),
        Goal(
            type=GoalType.RESPOND_TO_USER,
            description="engage actively with the user",
            priority=0.7
        )
    ]
    
    # Create ambiguous emotional state
    emotions = {
        "valence": 0.5,
        "arousal": 0.5,
        "dominance": 0.5
    }
    
    snapshot = workspace.broadcast()
    snapshot = snapshot.model_copy(update={
        "goals": goals,
        "emotions": emotions
    })
    
    print("\n📋 Scenario: Conflicting goals + ambiguous emotions")
    print(f"   Goal 1: '{goals[0].description}'")
    print(f"   Goal 2: '{goals[1].description}'")
    print(f"   Emotions: V={emotions['valence']}, A={emotions['arousal']}, D={emotions['dominance']}")
    
    percepts = monitor.observe(snapshot)
    
    print(f"\n🪞 SelfMonitor generated {len(percepts)} percept(s)")
    for i, percept in enumerate(percepts, 1):
        print(f"\n   Percept {i}:")
        print(f"   - Type: {percept.raw['type']}")
        print(f"   - Description: {percept.raw['description']}")
        if 'indicators' in percept.raw:
            for indicator in percept.raw['indicators']:
                print(f"     • {indicator['type']}: {indicator.get('description', 'N/A')}")
    
    stats = monitor.get_stats()
    print(f"\n📊 Stats: {stats['uncertainty_detections']} uncertainty states detected")


def demo_emotional_observation():
    """Demonstrate emotional observation."""
    print("\n" + "="*60)
    print("DEMO 4: Emotional Observation")
    print("="*60)
    
    workspace = GlobalWorkspace()
    affect = AffectSubsystem()
    workspace.affect = affect
    
    # Build emotional history with extreme states
    for _ in range(10):
        affect.emotion_history.append(EmotionalState(
            valence=-0.7,
            arousal=0.9,
            dominance=0.2
        ))
    
    monitor = SelfMonitor(workspace=workspace, config={"monitoring_frequency": 1})
    
    emotions = {"valence": -0.7, "arousal": 0.9, "dominance": 0.2}
    
    snapshot = workspace.broadcast()
    snapshot = snapshot.model_copy(update={
        "emotions": emotions
    })
    
    print("\n📋 Scenario: Extreme emotional state")
    print(f"   Valence: {emotions['valence']} (negative)")
    print(f"   Arousal: {emotions['arousal']} (high)")
    print(f"   Dominance: {emotions['dominance']} (low)")
    print(f"   Emotion label: {affect.get_emotion_label()}")
    
    percepts = monitor.observe(snapshot)
    
    print(f"\n🪞 SelfMonitor generated {len(percepts)} percept(s)")
    for i, percept in enumerate(percepts, 1):
        print(f"\n   Percept {i}:")
        print(f"   - Type: {percept.raw['type']}")
        print(f"   - Description: {percept.raw['description']}")
        if 'observations' in percept.raw:
            print(f"   - Observations:")
            for obs in percept.raw['observations']:
                print(f"     • {obs}")
    
    stats = monitor.get_stats()
    print(f"\n📊 Stats: {stats['emotional_observations']} emotional observations")


def demo_pattern_detection():
    """Demonstrate pattern detection."""
    print("\n" + "="*60)
    print("DEMO 5: Pattern Detection")
    print("="*60)
    
    workspace = GlobalWorkspace()
    action_subsystem = ActionSubsystem()
    workspace.action_subsystem = action_subsystem
    
    # Fill action history with repetitive actions
    for _ in range(15):
        action_subsystem.action_history.append(Action(
            type=ActionType.SPEAK,
            reason="test"
        ))
    
    monitor = SelfMonitor(workspace=workspace, config={"monitoring_frequency": 1})
    
    snapshot = workspace.broadcast()
    
    print("\n📋 Scenario: Repetitive action pattern")
    print(f"   Recent actions: 15 consecutive SPEAK actions")
    print(f"   Pattern threshold: >60% of same action type")
    
    percepts = monitor.observe(snapshot)
    
    print(f"\n🪞 SelfMonitor generated {len(percepts)} percept(s)")
    for i, percept in enumerate(percepts, 1):
        print(f"\n   Percept {i}:")
        print(f"   - Type: {percept.raw['type']}")
        print(f"   - Description: {percept.raw['description']}")
        if 'patterns' in percept.raw:
            for pattern in percept.raw['patterns']:
                print(f"     • {pattern['type']}: {pattern.get('action', 'N/A')} " +
                      f"(frequency: {pattern.get('frequency', 0):.1%})")
    
    stats = monitor.get_stats()
    print(f"\n📊 Stats: {stats['pattern_detections']} patterns detected")


def demo_monitoring_frequency():
    """Demonstrate monitoring frequency control."""
    print("\n" + "="*60)
    print("DEMO 6: Monitoring Frequency Control")
    print("="*60)
    
    workspace = GlobalWorkspace()
    monitor = SelfMonitor(workspace=workspace, config={"monitoring_frequency": 3})
    
    snapshot = workspace.broadcast()
    
    print(f"\n📋 Scenario: Monitoring frequency set to every 3 cycles")
    print(f"   Calling observe() 5 times...")
    
    for i in range(1, 6):
        percepts = monitor.observe(snapshot)
        status = "✅ Generated percepts" if len(percepts) > 0 else "⏭️  Skipped (not monitoring cycle)"
        print(f"   Cycle {i}: {status} (cycle_count={monitor.cycle_count})")
    
    stats = monitor.get_stats()
    print(f"\n📊 Final cycle count: {stats['cycle_count']}")
    print(f"   Monitoring frequency: every {stats['monitoring_frequency']} cycles")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print(" 🪞  SELFMONITOR SUBSYSTEM DEMONSTRATION")
    print("="*60)
    print("\nThis demonstration shows the SelfMonitor's meta-cognitive capabilities:")
    print("  1. Value alignment checking")
    print("  2. Performance assessment")
    print("  3. Uncertainty detection")
    print("  4. Emotional observation")
    print("  5. Pattern detection")
    print("  6. Monitoring frequency control")
    
    try:
        demo_value_alignment()
        demo_performance_assessment()
        demo_uncertainty_detection()
        demo_emotional_observation()
        demo_pattern_detection()
        demo_monitoring_frequency()
        
        print("\n" + "="*60)
        print(" ✅  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nThe SelfMonitor subsystem is fully functional and integrated!")
        print("It can now observe cognitive processes and generate introspective percepts.")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
