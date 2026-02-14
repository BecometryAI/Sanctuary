#!/usr/bin/env python3
"""
Demo script showing the AffectSubsystem in action.

This script demonstrates:
1. Emotional state updates based on goals, percepts, and actions
2. Emotion labeling (VAD → categorical emotions)
3. Emotional decay over time
4. Influence on attention and action priorities
"""

import sys
from pathlib import Path

# Add emergence_core to path
sys.path.insert(0, str(Path(__file__).parent / "emergence_core"))

from datetime import datetime
from mind.cognitive_core.affect import AffectSubsystem, EmotionalState
from mind.cognitive_core.workspace import (
    Goal, GoalType, Percept, WorkspaceSnapshot
)
from mind.cognitive_core.action import Action, ActionType


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_emotion_state(affect: AffectSubsystem, description: str = ""):
    """Print current emotional state."""
    state = affect.get_state()
    print(f"{description}")
    print(f"  Valence:   {state['valence']:+.3f} (negative ← 0 → positive)")
    print(f"  Arousal:   {state['arousal']:+.3f} (calm ← 0 → excited)")
    print(f"  Dominance: {state['dominance']:+.3f} (low control ← 0 → high control)")
    print(f"  Emotion:   {state['label']}")
    print()


def demo_basic_emotions():
    """Demonstrate basic emotional states and labeling."""
    print_section("1. Basic Emotional States (VAD Model)")
    
    affect = AffectSubsystem()
    print_emotion_state(affect, "Initial baseline state:")
    
    # Simulate positive progress
    goal = Goal(
        type=GoalType.LEARN,
        description="Learning new concepts",
        progress=0.9,  # High progress
        priority=0.8
    )
    
    snapshot = WorkspaceSnapshot(
        goals=[goal],
        percepts={},
        emotions={},
        memories=[],
        timestamp=datetime.now(),
        cycle_count=1
    )
    
    affect.compute_update(snapshot)
    print_emotion_state(affect, "After high goal progress:")


def demo_goal_based_emotions():
    """Demonstrate goal-based emotional dynamics."""
    print_section("2. Goal-Based Emotional Dynamics")
    
    affect = AffectSubsystem()
    
    # Many high-priority goals = high arousal
    goals = [
        Goal(type=GoalType.RESPOND_TO_USER, description=f"Goal {i}", priority=0.9)
        for i in range(5)
    ]
    
    snapshot = WorkspaceSnapshot(
        goals=goals,
        percepts={},
        emotions={},
        memories=[],
        timestamp=datetime.now(),
        cycle_count=1
    )
    
    affect.compute_update(snapshot)
    print_emotion_state(affect, "With 5 high-priority goals:")


def demo_percept_based_emotions():
    """Demonstrate percept-based emotional responses."""
    print_section("3. Percept-Based Emotional Responses")
    
    affect = AffectSubsystem()
    
    # Negative percept
    percept1 = Percept(
        modality="text",
        raw="This is terrible and I'm very worried about the situation",
        complexity=20
    )
    
    snapshot = WorkspaceSnapshot(
        goals=[],
        percepts={percept1.id: percept1.model_dump()},
        emotions={},
        memories=[],
        timestamp=datetime.now(),
        cycle_count=1
    )
    
    affect.compute_update(snapshot)
    print_emotion_state(affect, "After negative percept:")
    
    # Positive percept
    affect2 = AffectSubsystem()
    percept2 = Percept(
        modality="text",
        raw="This is wonderful and exciting progress!",
        complexity=15
    )
    
    snapshot2 = WorkspaceSnapshot(
        goals=[],
        percepts={percept2.id: percept2.model_dump()},
        emotions={},
        memories=[],
        timestamp=datetime.now(),
        cycle_count=1
    )
    
    affect2.compute_update(snapshot2)
    print_emotion_state(affect2, "After positive percept:")


def demo_emotional_decay():
    """Demonstrate emotional decay toward baseline."""
    print_section("4. Emotional Decay (Regulation)")
    
    config = {"decay_rate": 0.15}  # Faster decay for demo
    affect = AffectSubsystem(config=config)
    
    # Set extreme emotional state
    affect.valence = 0.9
    affect.arousal = 0.9
    affect.dominance = 0.9
    
    print_emotion_state(affect, "Initial extreme state:")
    
    # Run empty updates to show decay
    empty_snapshot = WorkspaceSnapshot(
        goals=[],
        percepts={},
        emotions={},
        memories=[],
        timestamp=datetime.now(),
        cycle_count=1
    )
    
    for i in range(5):
        affect.compute_update(empty_snapshot)
        print_emotion_state(affect, f"After decay cycle {i+1}:")


def demo_emotion_influence():
    """Demonstrate emotional influence on attention and actions."""
    print_section("5. Emotional Influence on Cognition")
    
    # High arousal state
    affect = AffectSubsystem()
    affect.arousal = 0.8
    affect.valence = -0.5
    affect.dominance = 0.3
    
    print_emotion_state(affect, "Current emotional state:")
    
    # Test attention influence
    print("Attention score modulation:")
    urgent_percept = Percept(
        modality="text",
        raw="Urgent crisis situation",
        complexity=35
    )
    base_score = 0.5
    modified_score = affect.influence_attention(base_score, urgent_percept.model_dump())
    print(f"  Base score: {base_score:.3f}")
    print(f"  Modified:   {modified_score:.3f} (boosted by high arousal)")
    print()
    
    # Test action influence
    print("Action priority modulation:")
    introspect_action = Action(type=ActionType.INTROSPECT, priority=0.5)
    base_priority = 0.5
    modified_priority = affect.influence_action(base_priority, introspect_action.model_dump())
    print(f"  Base priority: {base_priority:.3f}")
    print(f"  Modified:      {modified_priority:.3f} (boosted by low dominance)")
    print()


def demo_emotion_labels():
    """Demonstrate various emotion labels from VAD coordinates."""
    print_section("6. Emotion Label Mapping (Russell's Circumplex)")
    
    emotions = [
        ("excited", 0.6, 0.8, 0.7),
        ("anxious", -0.5, 0.8, 0.3),
        ("content", 0.5, 0.2, 0.7),
        ("calm", 0.0, 0.2, 0.5),
        ("depressed", -0.5, 0.2, 0.3),
        ("angry", -0.5, 0.8, 0.7),
        ("neutral", 0.0, 0.5, 0.5),
    ]
    
    for expected_label, v, a, d in emotions:
        affect = AffectSubsystem()
        affect.valence = v
        affect.arousal = a
        affect.dominance = d
        actual_label = affect.get_emotion_label()
        
        match = "✓" if actual_label == expected_label else "✗"
        print(f"{match} Expected: {expected_label:12s} | "
              f"Got: {actual_label:12s} | "
              f"VAD: ({v:+.1f}, {a:+.1f}, {d:+.1f})")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("  AffectSubsystem Demonstration")
    print("  Emotional Dynamics in Cognitive Architecture")
    print("="*60)
    
    demo_basic_emotions()
    demo_goal_based_emotions()
    demo_percept_based_emotions()
    demo_emotional_decay()
    demo_emotion_influence()
    demo_emotion_labels()
    
    print("\n" + "="*60)
    print("  Demo Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
