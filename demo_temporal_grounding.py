#!/usr/bin/env python3
"""
Demo script for the Temporal Grounding system.

This demonstrates the key features of temporal awareness with session tracking:
- Session detection and boundaries
- Time passage effects on emotional state
- Temporal pattern learning and expectations
- Relative time descriptions
"""

import sys
from pathlib import Path

# Add the cognitive_core directory to the path
# Try to find it relative to this script's location
script_dir = Path(__file__).parent.resolve()
cognitive_core_path = script_dir / "emergence_core" / "sanctuary" / "cognitive_core"

if cognitive_core_path.exists():
    sys.path.insert(0, str(cognitive_core_path))
else:
    # Fallback: try parent directory structure
    parent_cognitive_core = script_dir.parent / "emergence_core" / "sanctuary" / "cognitive_core"
    if parent_cognitive_core.exists():
        sys.path.insert(0, str(parent_cognitive_core))
    else:
        raise ImportError("Could not find cognitive_core module. Please run from project root.")

from datetime import datetime, timedelta
from temporal.grounding import TemporalGrounding


def print_section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 60}")
    print(f" {title}")
    print('=' * 60)


def demo_session_awareness():
    """Demonstrate session awareness and boundaries."""
    print_section("1. Session Awareness & Boundaries")
    
    tg = TemporalGrounding()
    
    # First interaction
    print("\n📍 First interaction (new session)")
    context1 = tg.on_interaction()
    print(f"   Session #{context1.session_number}")
    print(f"   Is new session: {context1.is_new_session}")
    print(f"   Time since last: {context1.time_description}")
    
    # Quick follow-up
    print("\n📍 Quick follow-up (5 minutes later)")
    time2 = datetime.now() + timedelta(minutes=5)
    context2 = tg.on_interaction(time2)
    print(f"   Session #{context2.session_number}")
    print(f"   Is new session: {context2.is_new_session}")
    print(f"   Same session: {context1.session_number == context2.session_number}")
    
    # After gap
    print("\n📍 After long gap (2 hours later)")
    time3 = time2 + timedelta(hours=2)
    context3 = tg.on_interaction(time3)
    print(f"   Session #{context3.session_number}")
    print(f"   Is new session: {context3.is_new_session}")
    print(f"   Different session: {context2.session_number != context3.session_number}")
    
    # Get greeting context
    greeting = tg.sessions.get_session_greeting_context()
    print(f"\n💬 Greeting context: {greeting['type']}")
    print(f"   Context hint: {greeting['context']}")


def demo_time_passage_effects():
    """Demonstrate time passage effects on cognitive state."""
    print_section("2. Time Passage Effects on Emotions")
    
    tg = TemporalGrounding()
    
    # Initial strong emotional state
    initial_state = {
        'emotions': {
            'valence': 0.9,    # Very positive
            'arousal': 0.95,   # Very excited
            'dominance': 0.8   # High control
        },
        'goals': [],
        'working_memory': []
    }
    
    print("\n😃 Initial emotional state:")
    print(f"   Valence:   {initial_state['emotions']['valence']:.2f} (very positive)")
    print(f"   Arousal:   {initial_state['emotions']['arousal']:.2f} (very excited)")
    print(f"   Dominance: {initial_state['emotions']['dominance']:.2f} (high control)")
    
    # Simulate 2 hours passing
    print("\n⏰ After 2 hours...")
    tg._last_effect_time = datetime.now() - timedelta(hours=2)
    updated = tg.apply_time_passage_effects(initial_state)
    
    print("\n😌 Emotions decayed toward baseline:")
    print(f"   Valence:   {updated['emotions']['valence']:.2f}")
    print(f"   Arousal:   {updated['emotions']['arousal']:.2f}")
    print(f"   Dominance: {updated['emotions']['dominance']:.2f}")
    
    if updated.get('consolidation_needed'):
        print("\n💭 Memory consolidation triggered!")


def demo_temporal_expectations():
    """Demonstrate temporal pattern learning and expectations."""
    print_section("3. Temporal Pattern Learning & Expectations")
    
    tg = TemporalGrounding()
    
    print("\n📊 Recording daily interaction pattern...")
    base_time = datetime.now() - timedelta(days=4)
    
    for day in range(5):
        event_time = base_time + timedelta(days=day)
        tg.record_event('daily_check_in', event_time)
        print(f"   Day {day + 1}: {event_time.strftime('%Y-%m-%d %H:%M')}")
    
    # Get expectation
    state = tg.get_temporal_state()
    expectations = state['expectations']
    
    if expectations:
        exp = expectations[0]
        print(f"\n🎯 Expectation formed:")
        print(f"   Event: {exp['event_type']}")
        print(f"   Expected: {datetime.fromisoformat(exp['expected_time']).strftime('%Y-%m-%d %H:%M')}")
        print(f"   Confidence: {exp['confidence']:.1%}")
        print(f"   Overdue: {exp['is_overdue']}")
    else:
        print("\n⏳ Insufficient data for expectations yet")


def demo_relative_time():
    """Demonstrate relative time descriptions."""
    print_section("4. Relative Time Descriptions")
    
    tg = TemporalGrounding()
    
    now = datetime.now()
    
    test_times = [
        ("Just now", now - timedelta(seconds=5)),
        ("15 minutes ago", now - timedelta(minutes=15)),
        ("3 hours ago", now - timedelta(hours=3)),
        ("Yesterday", now - timedelta(days=1)),
        ("Last week", now - timedelta(days=7)),
        ("Last month", now - timedelta(days=30)),
    ]
    
    print("\n🕐 Time descriptions:")
    for label, time in test_times:
        desc = tg.describe_time(time)
        print(f"   {label:20} → {desc}")


def demo_session_tracking():
    """Demonstrate session tracking and context."""
    print_section("5. Session Context & History")
    
    tg = TemporalGrounding()
    
    # Session 1
    print("\n📝 Session 1:")
    tg.on_interaction()
    tg.record_topic("temporal grounding")
    tg.record_topic("consciousness")
    tg.record_emotional_state({'valence': 0.7, 'arousal': 0.6})
    
    info = tg.sessions.get_current_session_info()
    print(f"   Topics: {info['topics']}")
    print(f"   Emotional states tracked: {info['emotional_states']}")
    
    # End session
    tg.end_session()
    
    # Session 2 (after gap)
    print("\n📝 Session 2 (after gap):")
    time2 = datetime.now() + timedelta(hours=2)
    tg.on_interaction(time2)
    tg.record_topic("time passage effects")
    
    info = tg.sessions.get_current_session_info()
    print(f"   Topics: {info['topics']}")
    
    # Check history
    print(f"\n📚 Total sessions: {len(tg.awareness.session_history) + 1}")
    print(f"   Archived: {len(tg.awareness.session_history)}")
    print(f"   Current: 1")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print(" TEMPORAL GROUNDING SYSTEM DEMONSTRATION")
    print("=" * 60)
    print("\nThis demonstrates genuine temporal grounding with:")
    print("  • Session awareness and boundaries")
    print("  • Time passage effects on cognitive state")
    print("  • Temporal pattern learning")
    print("  • Human-friendly time descriptions")
    print("  • Session history and context tracking")
    
    try:
        demo_session_awareness()
        demo_time_passage_effects()
        demo_temporal_expectations()
        demo_relative_time()
        demo_session_tracking()
        
        print("\n" + "=" * 60)
        print(" ✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
