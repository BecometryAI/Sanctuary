#!/usr/bin/env python3
"""
Direct test of communication drive module without dependency chain.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import importlib.util

# Load the drive module directly without package imports
def load_drive_module():
    """Load the drive module directly from file."""
    drive_path = Path(__file__).parent / "emergence_core" / "lyra" / "cognitive_core" / "communication" / "drive.py"
    
    spec = importlib.util.spec_from_file_location("communication_drive", drive_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['communication_drive'] = module  # Register it
    spec.loader.exec_module(module)
    return module

# Load the module
drive = load_drive_module()

CommunicationDriveSystem = drive.CommunicationDriveSystem
CommunicationUrge = drive.CommunicationUrge
DriveType = drive.DriveType

def test_all():
    """Run all tests."""
    print("\n=== Testing Communication Drive System ===\n")
    
    # Test 1: Drive types
    expected = {"insight", "question", "emotional", "social", "goal", "correction", "acknowledgment"}
    actual = {dt.value for dt in DriveType}
    assert actual == expected
    print("✓ Drive types defined correctly")
    
    # Test 2: Urge creation
    urge = CommunicationUrge(
        drive_type=DriveType.INSIGHT,
        intensity=0.8,
        content="Test insight",
        reason="Testing"
    )
    assert urge.drive_type == DriveType.INSIGHT
    assert urge.intensity == 0.8
    print("✓ Urge creation works")
    
    # Test 3: Urge decay
    urge = CommunicationUrge(
        drive_type=DriveType.EMOTIONAL,
        intensity=1.0,
        decay_rate=0.5
    )
    urge.created_at = datetime.now() - timedelta(minutes=1)
    current = urge.get_current_intensity()
    assert 0.4 < current < 0.6, f"Unexpected decay: {current}"
    print("✓ Urge decay works")
    
    # Test 4: System initialization
    system = CommunicationDriveSystem()
    assert system.active_urges == []
    assert system.last_input_time is None
    print("✓ System initialization works")
    
    # Test 5: Emotional drive
    system = CommunicationDriveSystem(config={"emotional_threshold": 0.5})
    emotional_state = {
        "valence": 0.0,
        "arousal": 0.8,
        "dominance": 0.5
    }
    urges = system._compute_emotional_drive(emotional_state)
    assert len(urges) > 0
    assert any(u.drive_type == DriveType.EMOTIONAL for u in urges)
    print("✓ Emotional drive computation works")
    
    # Test 6: Social drive
    system = CommunicationDriveSystem(config={"social_silence_minutes": 5})
    system.last_input_time = datetime.now() - timedelta(minutes=10)
    urges = system._compute_social_drive()
    assert len(urges) > 0
    assert urges[0].drive_type == DriveType.SOCIAL
    print("✓ Social drive computation works")
    
    # Test 7: Total drive
    system = CommunicationDriveSystem()
    system.active_urges = [
        CommunicationUrge(DriveType.INSIGHT, 0.8, priority=0.7),
        CommunicationUrge(DriveType.EMOTIONAL, 0.6, priority=0.6)
    ]
    total = system.get_total_drive()
    assert 0.0 < total <= 1.0
    print(f"✓ Total drive aggregation works (total={total:.3f})")
    
    # Test 8: Strongest urge
    system = CommunicationDriveSystem()
    weak = CommunicationUrge(DriveType.SOCIAL, 0.2, priority=0.3)
    strong = CommunicationUrge(DriveType.INSIGHT, 0.9, priority=0.8)
    system.active_urges = [weak, strong]
    strongest = system.get_strongest_urge()
    assert strongest == strong
    print("✓ Strongest urge selection works")
    
    # Test 9: Input/output recording
    system = CommunicationDriveSystem()
    system.record_input()
    assert system.last_input_time is not None
    system.record_output()
    assert system.last_output_time is not None
    print("✓ Input/output recording works")
    
    # Test 10: Drive summary
    system = CommunicationDriveSystem()
    system.active_urges = [
        CommunicationUrge(DriveType.INSIGHT, 0.7),
        CommunicationUrge(DriveType.EMOTIONAL, 0.5)
    ]
    summary = system.get_drive_summary()
    assert "total_drive" in summary
    assert "active_urges" in summary
    assert summary["active_urges"] == 2
    print("✓ Drive summary generation works")
    
    print("\n=== All tests passed! ✓ ===\n")

if __name__ == "__main__":
    try:
        test_all()
    except Exception as e:
        print(f"\n✗ Test failed: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
