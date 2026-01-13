#!/usr/bin/env python3
"""
Demonstration of the Silence-as-Action tracking system.

This script shows how silence decisions are explicitly logged, tracked,
and classified into different types with reasons.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import importlib.util

# Load modules directly without triggering package imports
def load_module_from_file(module_name, file_name, dependencies=None):
    """Load a module directly from file with dependencies."""
    module_path = Path(__file__).parent / "emergence_core" / "lyra" / "cognitive_core" / "communication" / file_name
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Register dependencies if provided
    if dependencies:
        for dep_name, dep_module in dependencies.items():
            setattr(module, dep_name, dep_module)
    
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

print("Loading modules...")

# Load drive module first (no dependencies)
drive = load_module_from_file("comm_drive", "drive.py")

# Load inhibition module (no dependencies)
inhibition = load_module_from_file("comm_inhibition", "inhibition.py")

# Load silence module (no dependencies from communication package)
silence = load_module_from_file("comm_silence", "silence.py")

# Load decision module (depends on drive and silence)
# We need to mock the imports for decision module
sys.modules['lyra.cognitive_core.communication.drive'] = drive
sys.modules['lyra.cognitive_core.communication.silence'] = silence
decision = load_module_from_file("comm_decision", "decision.py")

# Extract the classes we need
CommunicationDriveSystem = drive.CommunicationDriveSystem
CommunicationInhibitionSystem = inhibition.CommunicationInhibitionSystem
CommunicationDecisionLoop = decision.CommunicationDecisionLoop
CommunicationDecision = decision.CommunicationDecision
DecisionResult = decision.DecisionResult
SilenceTracker = silence.SilenceTracker
SilenceType = silence.SilenceType
SilenceAction = silence.SilenceAction

print("✅ Modules loaded successfully!\n")


def demo_silence_types():
    """Demonstrate all silence types."""
    print("=" * 60)
    print("DEMO: Silence Type Categories")
    print("=" * 60)
    
    print("\nAvailable silence types:")
    for st in SilenceType:
        print(f"  • {st.name:25} → {st.value}")
    
    print(f"\n✅ Total: {len(list(SilenceType))} silence categories defined")


def demo_silence_classification():
    """Demonstrate automatic silence classification."""
    print("\n" + "=" * 60)
    print("DEMO: Automatic Silence Classification")
    print("=" * 60)
    
    tracker = SilenceTracker()
    
    test_cases = [
        ("Confidence too low (0.40)", SilenceType.UNCERTAINTY),
        ("Content already expressed in previous response", SilenceType.REDUNDANCY),
        ("Bad timing - only 2.5s since last output", SilenceType.TIMING),
        ("Still processing the complex question", SilenceType.STILL_THINKING),
        ("Respect silence - contemplative moment", SilenceType.RESPECTING_SPACE),
        ("Insufficient drive (0.12)", SilenceType.NOTHING_TO_ADD),
        ("Inhibition (0.85) exceeds drive (0.15)", SilenceType.CHOOSING_DISCRETION),
    ]
    
    print("\nClassifying silence reasons:")
    for reason, expected in test_cases:
        dr = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason=reason,
            confidence=0.7,
            drive_level=0.2,
            inhibition_level=0.5,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )
        
        silence_action = tracker.record_silence(dr)
        match = "✅" if silence_action.silence_type == expected else "❌"
        
        print(f"\n  {match} Reason: \"{reason}\"")
        print(f"     → Classified as: {silence_action.silence_type.value}")
        if silence_action.silence_type != expected:
            print(f"     (Expected: {expected.value})")


def demo_silence_tracking():
    """Demonstrate silence history tracking."""
    print("\n" + "=" * 60)
    print("DEMO: Silence History Tracking")
    print("=" * 60)
    
    tracker = SilenceTracker(config={
        "max_silence_history": 100,
        "silence_pressure_threshold": 60
    })
    
    print(f"\nInitial state:")
    print(f"  Silence history: {len(tracker.silence_history)} entries")
    print(f"  Current silence: {tracker.current_silence is not None}")
    print(f"  Silence streak: {tracker.silence_streak}")
    
    # Simulate multiple silence decisions
    print("\n📝 Recording 5 consecutive silence decisions...")
    for i in range(5):
        dr = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason=f"Silence decision #{i+1}",
            confidence=0.7,
            drive_level=0.2,
            inhibition_level=0.5,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )
        
        silence_action = tracker.record_silence(dr)
        print(f"  {i+1}. {silence_action.silence_type.value} - {silence_action.reason}")
    
    print(f"\n📊 After recording silences:")
    print(f"  Silence history: {len(tracker.silence_history)} entries")
    print(f"  Current silence: {tracker.current_silence is not None}")
    print(f"  Silence streak: {tracker.silence_streak}")
    
    # End silence (breaking silence with speech)
    print("\n🗣️  Breaking silence (speaking)...")
    ended_silence = tracker.end_silence()
    
    if ended_silence:
        print(f"  ✅ Ended silence after {ended_silence.duration:.3f}s")
        print(f"     Type: {ended_silence.silence_type.value}")
    
    print(f"\n📊 After breaking silence:")
    print(f"  Current silence: {tracker.current_silence is not None}")
    print(f"  Silence streak: {tracker.silence_streak}")


def demo_silence_pressure():
    """Demonstrate silence pressure calculation."""
    print("\n" + "=" * 60)
    print("DEMO: Silence Pressure (Drive to Break Silence)")
    print("=" * 60)
    
    # Short pressure threshold for demo
    tracker = SilenceTracker(config={
        "silence_pressure_threshold": 2,  # 2 seconds
        "max_silence_streak": 3
    })
    
    print("\nConfiguration:")
    print(f"  Pressure threshold: 2 seconds")
    print(f"  Max silence streak: 3 cycles")
    
    # Create initial silence
    dr = DecisionResult(
        decision=CommunicationDecision.SILENCE,
        reason="Initial silence",
        confidence=0.7,
        drive_level=0.2,
        inhibition_level=0.5,
        net_pressure=-0.3,
        inhibitions=[],
        urges=[]
    )
    
    silence_action = tracker.record_silence(dr)
    
    # Simulate silence at start
    print(f"\n⏱️  Immediately after silence:")
    pressure = tracker.get_silence_pressure()
    print(f"  Silence pressure: {pressure:.2f}")
    
    # Simulate elapsed time
    print(f"\n⏱️  After 3 seconds (1.5x threshold):")
    silence_action.timestamp = datetime.now() - timedelta(seconds=3)
    pressure = tracker.get_silence_pressure()
    print(f"  Silence pressure: {pressure:.2f}")
    
    # Simulate more elapsed time
    print(f"\n⏱️  After 6 seconds (3x threshold):")
    silence_action.timestamp = datetime.now() - timedelta(seconds=6)
    pressure = tracker.get_silence_pressure()
    print(f"  Silence pressure: {pressure:.2f}")
    
    # Reset and test streak pressure
    print(f"\n📊 Testing streak pressure (consecutive silences):")
    tracker2 = SilenceTracker(config={"max_silence_streak": 3})
    
    for i in range(1, 4):
        dr = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason=f"Silence {i}",
            confidence=0.7,
            drive_level=0.2,
            inhibition_level=0.5,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )
        tracker2.record_silence(dr)
        pressure = tracker2.get_silence_pressure()
        print(f"  After {i} silence(s): pressure = {pressure:.2f}")


def demo_silence_integration():
    """Demonstrate integration with decision loop."""
    print("\n" + "=" * 60)
    print("DEMO: Integration with Communication Decision Loop")
    print("=" * 60)
    
    # Create communication systems
    drives = CommunicationDriveSystem()
    inhibitions = CommunicationInhibitionSystem()
    decision_loop = CommunicationDecisionLoop(drives, inhibitions)
    
    print("\n✅ Decision loop initialized with silence tracker")
    print(f"   Has silence_tracker: {hasattr(decision_loop, 'silence_tracker')}")
    
    # Check initial state
    summary = decision_loop.get_decision_summary()
    print(f"\n📊 Initial decision summary includes silence tracking:")
    print(f"   Contains 'silence_tracking' key: {'silence_tracking' in summary}")
    
    if 'silence_tracking' in summary:
        st = summary['silence_tracking']
        print(f"   Total silences: {st['total_silences']}")
        print(f"   Current silence: {st['current_silence']}")
        print(f"   Silence pressure: {st['silence_pressure']:.2f}")
    
    print("\n✅ Silence tracking fully integrated into decision loop!")


def demo_silence_summary():
    """Demonstrate silence summary statistics."""
    print("\n" + "=" * 60)
    print("DEMO: Silence Summary Statistics")
    print("=" * 60)
    
    tracker = SilenceTracker()
    
    # Add various types of silence
    silence_types = [
        SilenceType.NOTHING_TO_ADD,
        SilenceType.UNCERTAINTY,
        SilenceType.TIMING,
        SilenceType.NOTHING_TO_ADD,
        SilenceType.RESPECTING_SPACE
    ]
    
    for st in silence_types:
        dr = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason=f"Test silence: {st.value}",
            confidence=0.7,
            drive_level=0.2,
            inhibition_level=0.5,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )
        tracker.record_silence(dr, silence_type=st)
    
    summary = tracker.get_silence_summary()
    
    print(f"\n📊 Summary after {summary['total_silences']} silences:")
    print(f"   Total silences: {summary['total_silences']}")
    print(f"   Recent silences (5 min): {summary['recent_silences']}")
    print(f"   Current silence active: {summary['current_silence']}")
    print(f"   Silence streak: {summary['silence_streak']}")
    print(f"   Silence pressure: {summary['silence_pressure']:.2f}")
    
    print(f"\n   Breakdown by type:")
    for st_name, count in summary['silence_by_type'].items():
        if count > 0:
            print(f"     • {st_name:20} : {count}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("SILENCE-AS-ACTION DEMONSTRATION")
    print("=" * 60)
    print("\nShowing how silence decisions are explicitly logged,")
    print("tracked, and categorized with typed reasons.\n")
    
    try:
        demo_silence_types()
        demo_silence_classification()
        demo_silence_tracking()
        demo_silence_pressure()
        demo_silence_integration()
        demo_silence_summary()
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nKey Features Demonstrated:")
        print("  ✓ 7 distinct silence type categories")
        print("  ✓ Automatic classification from decision context")
        print("  ✓ Complete silence history tracking")
        print("  ✓ Silence duration measurement")
        print("  ✓ Pressure to break silence (duration + streak)")
        print("  ✓ Integration with communication decision loop")
        print("  ✓ Comprehensive silence statistics")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
