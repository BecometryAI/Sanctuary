#!/usr/bin/env python3
"""
Minimal standalone test for communication drive system.
Tests the module in isolation without requiring full dependencies.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add emergence_core to path
sys.path.insert(0, str(Path(__file__).parent / "emergence_core"))

# Now import the drive module directly
from lyra.cognitive_core.communication.drive import (
    CommunicationDriveSystem,
    CommunicationUrge,
    DriveType
)

def test_drive_types():
    """Test that all drive types are defined."""
    expected = {"insight", "question", "emotional", "social", "goal", "correction", "acknowledgment"}
    actual = {dt.value for dt in DriveType}
    assert actual == expected, f"Drive types mismatch: {actual} vs {expected}"
    print("✓ Drive types defined correctly")

def test_urge_creation():
    """Test creating a communication urge."""
    urge = CommunicationUrge(
        drive_type=DriveType.INSIGHT,
        intensity=0.8,
        content="Test insight",
        reason="Testing"
    )
    assert urge.drive_type == DriveType.INSIGHT
    assert urge.intensity == 0.8
    assert urge.content == "Test insight"
    print("✓ Urge creation works")

def test_urge_decay():
    """Test urge decay over time."""
    urge = CommunicationUrge(
        drive_type=DriveType.EMOTIONAL,
        intensity=1.0,
        decay_rate=0.5
    )
    # Set to 1 minute ago
    urge.created_at = datetime.now() - timedelta(minutes=1)
    current = urge.get_current_intensity()
    # Should decay from 1.0 by 50% per minute
    assert 0.4 < current < 0.6, f"Unexpected decay: {current}"
    print("✓ Urge decay works")

def test_system_initialization():
    """Test initializing the drive system."""
    system = CommunicationDriveSystem()
    assert system.active_urges == []
    assert system.last_input_time is None
    assert system.last_output_time is None
    print("✓ System initialization works")

def test_emotional_drive():
    """Test emotional drive computation."""
    system = CommunicationDriveSystem(config={"emotional_threshold": 0.5})
    
    # High arousal
    emotional_state = {
        "valence": 0.0,
        "arousal": 0.8,
        "dominance": 0.5
    }
    
    urges = system._compute_emotional_drive(emotional_state)
    assert len(urges) > 0, "Should generate urge for high arousal"
    assert any(u.drive_type == DriveType.EMOTIONAL for u in urges)
    print("✓ Emotional drive computation works")

def test_social_drive():
    """Test social drive after silence."""
    system = CommunicationDriveSystem(config={"social_silence_minutes": 5})
    
    # Set last input to 10 minutes ago
    system.last_input_time = datetime.now() - timedelta(minutes=10)
    
    urges = system._compute_social_drive()
    assert len(urges) > 0, "Should generate social urge after silence"
    assert urges[0].drive_type == DriveType.SOCIAL
    print("✓ Social drive computation works")

def test_total_drive():
    """Test total drive aggregation."""
    system = CommunicationDriveSystem()
    
    # Add some urges
    system.active_urges = [
        CommunicationUrge(DriveType.INSIGHT, 0.8, priority=0.7),
        CommunicationUrge(DriveType.EMOTIONAL, 0.6, priority=0.6)
    ]
    
    total = system.get_total_drive()
    assert 0.0 < total <= 1.0, f"Total drive out of range: {total}"
    print(f"✓ Total drive aggregation works (total={total:.3f})")

def test_strongest_urge():
    """Test getting strongest urge."""
    system = CommunicationDriveSystem()
    
    weak = CommunicationUrge(DriveType.SOCIAL, 0.2, priority=0.3)
    strong = CommunicationUrge(DriveType.INSIGHT, 0.9, priority=0.8)
    
    system.active_urges = [weak, strong]
    
    strongest = system.get_strongest_urge()
    assert strongest == strong, "Should return strongest urge"
    print("✓ Strongest urge selection works")

def test_record_io():
    """Test recording input/output."""
    system = CommunicationDriveSystem()
    
    system.record_input()
    assert system.last_input_time is not None
    
    system.record_output()
    assert system.last_output_time is not None
    print("✓ Input/output recording works")

def test_drive_summary():
    """Test drive summary generation."""
    system = CommunicationDriveSystem()
    system.active_urges = [
        CommunicationUrge(DriveType.INSIGHT, 0.7),
        CommunicationUrge(DriveType.EMOTIONAL, 0.5)
    ]
    
    summary = system.get_drive_summary()
    assert "total_drive" in summary
    assert "active_urges" in summary
    assert summary["active_urges"] == 2
    assert "urges_by_type" in summary
    print("✓ Drive summary generation works")

def main():
    """Run all tests."""
    print("\n=== Testing Communication Drive System ===\n")
    
    tests = [
        test_drive_types,
        test_urge_creation,
        test_urge_decay,
        test_system_initialization,
        test_emotional_drive,
        test_social_drive,
        test_total_drive,
        test_strongest_urge,
        test_record_io,
        test_drive_summary,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
    
    if failed > 0:
        sys.exit(1)
    
    print("All tests passed! ✓\n")

if __name__ == "__main__":
    main()
