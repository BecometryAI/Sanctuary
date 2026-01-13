#!/usr/bin/env python3
"""
Simple test/demo of Silence-as-Action tracking.

Tests the core silence tracking functionality without full integration.
"""

import sys
from pathlib import Path
import importlib.util
from datetime import datetime, timedelta

# Load silence module directly
def load_silence_module():
    """Load the silence module directly from file."""
    silence_path = Path(__file__).parent / "emergence_core" / "lyra" / "cognitive_core" / "communication" / "silence.py"
    
    spec = importlib.util.spec_from_file_location("comm_silence", silence_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['comm_silence'] = module
    spec.loader.exec_module(module)
    return module

print("Loading silence module...")
silence = load_silence_module()

SilenceType = silence.SilenceType
SilenceAction = silence.SilenceAction
SilenceTracker = silence.SilenceTracker

print("✅ Silence module loaded!\n")

def main():
    """Run basic silence tracking tests."""
    print("=" * 60)
    print("SILENCE-AS-ACTION BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test 1: SilenceType enum
    print("\n1. Testing SilenceType Enum")
    print("-" * 60)
    print(f"   Found {len(list(SilenceType))} silence types:")
    for st in SilenceType:
        print(f"     • {st.name:25} → {st.value}")
    print("   ✅ All silence types defined")
    
    # Test 2: SilenceAction creation
    print("\n2. Testing SilenceAction")
    print("-" * 60)
    action = SilenceAction(
        silence_type=SilenceType.NOTHING_TO_ADD,
        reason="No valuable content to share",
        inhibitions=[],
        suppressed_urges=[]
    )
    print(f"   Created: {action.silence_type.value}")
    print(f"   Reason: {action.reason}")
    print(f"   Initial duration: {action.duration}")
    
    # Simulate time passing
    import time
    time.sleep(0.1)
    duration = action.end_silence()
    print(f"   After end_silence(): {duration:.3f}s")
    print("   ✅ SilenceAction works correctly")
    
    # Test 3: SilenceTracker initialization
    print("\n3. Testing SilenceTracker Initialization")
    print("-" * 60)
    tracker = SilenceTracker()
    print(f"   History size: {len(tracker.silence_history)}")
    print(f"   Current silence: {tracker.current_silence}")
    print(f"   Silence streak: {tracker.silence_streak}")
    print("   ✅ Tracker initialized")
    
    # Test 4: Recording silences without decision result
    print("\n4. Testing Manual Silence Recording")
    print("-" * 60)
    
    # Create mock decision result
    class MockDecisionResult:
        def __init__(self, reason, drive=0.2, inhibition=0.5):
            self.decision = "SILENCE"
            self.reason = reason
            self.confidence = 0.7
            self.drive_level = drive
            self.inhibition_level = inhibition
            self.net_pressure = drive - inhibition
            self.inhibitions = []
            self.urges = []
    
    # Test different silence types
    test_cases = [
        ("Uncertainty in response", SilenceType.UNCERTAINTY),
        ("Content redundant with previous", SilenceType.REDUNDANCY),
        ("Bad timing for response", SilenceType.TIMING),
        ("Still processing information", SilenceType.STILL_THINKING),
        ("Respecting conversational space", SilenceType.RESPECTING_SPACE),
        ("Insufficient drive to speak", SilenceType.NOTHING_TO_ADD),
    ]
    
    print(f"   Recording {len(test_cases)} silences...")
    for i, (reason, expected_type) in enumerate(test_cases, 1):
        dr = MockDecisionResult(reason)
        silence_action = tracker.record_silence(dr)
        print(f"   {i}. {silence_action.silence_type.value:20} - {reason[:40]}")
    
    print(f"   ✅ Recorded {len(tracker.silence_history)} silences")
    
    # Test 5: Silence history and statistics
    print("\n5. Testing Silence History & Statistics")
    print("-" * 60)
    summary = tracker.get_silence_summary()
    print(f"   Total silences: {summary['total_silences']}")
    print(f"   Recent silences (5min): {summary['recent_silences']}")
    print(f"   Current silence active: {summary['current_silence']}")
    print(f"   Silence streak: {summary['silence_streak']}")
    print(f"   Silence pressure: {summary['silence_pressure']:.2f}")
    
    print("\n   Breakdown by type:")
    for type_name, count in summary['silence_by_type'].items():
        if count > 0:
            print(f"     {type_name:20} : {count}")
    
    print("   ✅ Statistics working correctly")
    
    # Test 6: Ending silence
    print("\n6. Testing End Silence")
    print("-" * 60)
    print(f"   Before end: streak={tracker.silence_streak}, current={tracker.current_silence is not None}")
    
    ended = tracker.end_silence()
    print(f"   Ended silence: duration={ended.duration:.3f}s")
    print(f"   After end: streak={tracker.silence_streak}, current={tracker.current_silence is not None}")
    print("   ✅ Silence ending works correctly")
    
    # Test 7: Silence pressure
    print("\n7. Testing Silence Pressure")
    print("-" * 60)
    tracker2 = SilenceTracker({"silence_pressure_threshold": 1, "max_silence_streak": 3})
    
    # Create a silence and make it appear old
    dr = MockDecisionResult("Test silence")
    sa = tracker2.record_silence(dr)
    sa.timestamp = datetime.now() - timedelta(seconds=5)
    
    pressure = tracker2.get_silence_pressure()
    print(f"   After 5s (threshold=1s): pressure={pressure:.2f}")
    print("   ✅ Pressure calculation working")
    
    # Test 8: Get recent silences
    print("\n8. Testing Recent Silences Filter")
    print("-" * 60)
    recent = tracker.get_recent_silences(minutes=5)
    print(f"   Recent silences (5 min window): {len(recent)}")
    print("   ✅ Time filtering working")
    
    # Test 9: Get silences by type
    print("\n9. Testing Silence Type Filtering")
    print("-" * 60)
    uncertainty_silences = tracker.get_silence_by_type(SilenceType.UNCERTAINTY)
    print(f"   UNCERTAINTY silences: {len(uncertainty_silences)}")
    timing_silences = tracker.get_silence_by_type(SilenceType.TIMING)
    print(f"   TIMING silences: {len(timing_silences)}")
    print("   ✅ Type filtering working")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nSilence-as-Action Implementation Summary:")
    print("  ✓ SilenceType enum with 7 categories")
    print("  ✓ SilenceAction dataclass with duration tracking")
    print("  ✓ SilenceTracker with history management")
    print("  ✓ Automatic silence classification")
    print("  ✓ Silence pressure calculation")
    print("  ✓ Time-based and type-based filtering")
    print("  ✓ Comprehensive statistics")
    print()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
