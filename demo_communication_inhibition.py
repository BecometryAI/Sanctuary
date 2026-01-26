#!/usr/bin/env python3
"""
Demonstration of the Communication Inhibition System.

This script shows how the communication inhibition system computes reasons
not to communicate, providing the counterbalancing force to communication drives.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import importlib.util

# Load modules directly without package imports
def load_module(name, file_path):
    """Load a module directly from file."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load the inhibition and drive modules
base_path = Path(__file__).parent / "emergence_core" / "lyra" / "cognitive_core" / "communication"
inhibition = load_module("inhibition", base_path / "inhibition.py")
drive = load_module("drive", base_path / "drive.py")

# Extract classes
CommunicationInhibitionSystem = inhibition.CommunicationInhibitionSystem
InhibitionFactor = inhibition.InhibitionFactor
InhibitionType = inhibition.InhibitionType
CommunicationDriveSystem = drive.CommunicationDriveSystem
CommunicationUrge = drive.CommunicationUrge
DriveType = drive.DriveType


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_urge(urge):
    """Print urge details."""
    print(f"  • {urge.drive_type.value.upper()}: intensity={urge.get_current_intensity():.2f}, "
          f"priority={urge.priority:.2f}")
    if urge.reason:
        print(f"    Reason: {urge.reason}")


def print_inhibition(inhibition):
    """Print inhibition details."""
    print(f"  • {inhibition.inhibition_type.value.upper()}: strength={inhibition.get_current_strength():.2f}, "
          f"priority={inhibition.priority:.2f}")
    if inhibition.reason:
        print(f"    Reason: {inhibition.reason}")
    if inhibition.duration:
        print(f"    Duration: {inhibition.duration.total_seconds():.1f}s")


def demo_basic_inhibitions():
    """Demonstrate basic inhibition types."""
    print_section("Demo 1: Basic Inhibition Types")
    
    system = CommunicationInhibitionSystem()
    
    # Low value content
    print("\n1. LOW VALUE INHIBITION (content value = 0.15):")
    inhibitions = system._compute_low_value_inhibition(content_value=0.15)
    for inh in inhibitions:
        print_inhibition(inh)
    
    # Bad timing (just spoke)
    print("\n2. BAD TIMING INHIBITION (spoke 2 seconds ago):")
    system.last_output_time = datetime.now() - timedelta(seconds=2)
    inhibitions = system._compute_bad_timing_inhibition()
    for inh in inhibitions:
        print_inhibition(inh)
    
    # High uncertainty
    print("\n3. UNCERTAINTY INHIBITION (confidence = 0.25):")
    inhibitions = system._compute_uncertainty_inhibition(confidence=0.25)
    for inh in inhibitions:
        print_inhibition(inh)
    
    # Respect silence
    print("\n4. RESPECT SILENCE INHIBITION (low arousal, weak urges):")
    emotional_state = {"valence": 0.1, "arousal": 0.2, "dominance": 0.5}
    weak_urge = CommunicationUrge(DriveType.SOCIAL, 0.15, priority=0.3)
    inhibitions = system._compute_respect_silence_inhibition(emotional_state, [weak_urge])
    for inh in inhibitions:
        print_inhibition(inh)


def demo_redundancy_detection():
    """Demonstrate redundancy detection."""
    print_section("Demo 2: Redundancy Detection")
    
    system = CommunicationInhibitionSystem(config={
        "redundancy_similarity_threshold": 0.6
    })
    
    # Simulate previous output
    system.record_output("Hello world, this is a test message about cognitive systems")
    print("\nPrevious output: 'Hello world, this is a test message about cognitive systems'")
    print(f"Keywords tracked: {system.recent_outputs[0]['keywords']}")
    
    # Try very similar content
    print("\nAttempting similar content: 'Hello world, this is a test about cognitive systems'")
    
    # Create mock workspace with very similar content
    class MockPercept:
        def __init__(self, content):
            self.raw = content
    
    class MockWorkspace:
        def __init__(self, content):
            self.percepts = {'p1': MockPercept(content)}
    
    workspace = MockWorkspace("Hello world, this is a test about cognitive systems")
    inhibitions = system._compute_redundancy_inhibition(workspace)
    
    if inhibitions:
        print("\n✓ REDUNDANCY DETECTED:")
        for inh in inhibitions:
            print_inhibition(inh)
    else:
        print("\n✗ No redundancy detected")


def demo_drive_vs_inhibition():
    """Demonstrate drive vs inhibition decision-making."""
    print_section("Demo 3: Drive vs Inhibition Decision")
    
    drive_system = CommunicationDriveSystem()
    inhibition_system = CommunicationInhibitionSystem()
    
    # Scenario 1: Strong urge, weak inhibition
    print("\nScenario 1: Strong insight urge, minimal inhibition")
    drive_system.active_urges = [
        CommunicationUrge(DriveType.INSIGHT, 0.9, priority=0.8, 
                         reason="Important realization about consciousness")
    ]
    inhibition_system.active_inhibitions = [
        InhibitionFactor(InhibitionType.LOW_VALUE, 0.2, priority=0.5)
    ]
    
    total_drive = drive_system.get_total_drive()
    total_inhibition = inhibition_system.get_total_inhibition()
    should_inhibit = inhibition_system.should_inhibit(drive_system.active_urges, threshold=0.5)
    
    print(f"  Total drive: {total_drive:.3f}")
    print(f"  Total inhibition: {total_inhibition:.3f}")
    print(f"  Decision: {'INHIBIT (stay silent)' if should_inhibit else 'SPEAK'}")
    print(f"  Strongest urge: {drive_system.get_strongest_urge().drive_type.value}")
    
    # Scenario 2: Weak urge, strong inhibition
    print("\nScenario 2: Weak social urge, strong uncertainty")
    drive_system.active_urges = [
        CommunicationUrge(DriveType.SOCIAL, 0.3, priority=0.4, 
                         reason="Silence for 15 minutes")
    ]
    inhibition_system.active_inhibitions = [
        InhibitionFactor(InhibitionType.UNCERTAINTY, 0.9, priority=0.7,
                        reason="Very low confidence in response"),
        InhibitionFactor(InhibitionType.LOW_VALUE, 0.6, priority=0.6,
                        reason="Content not valuable enough")
    ]
    
    total_drive = drive_system.get_total_drive()
    total_inhibition = inhibition_system.get_total_inhibition()
    should_inhibit = inhibition_system.should_inhibit(drive_system.active_urges, threshold=0.5)
    
    print(f"  Total drive: {total_drive:.3f}")
    print(f"  Total inhibition: {total_inhibition:.3f}")
    print(f"  Decision: {'INHIBIT (stay silent)' if should_inhibit else 'SPEAK'}")
    print(f"  Strongest inhibition: {inhibition_system.get_strongest_inhibition().inhibition_type.value}")
    
    # Scenario 3: Balanced forces
    print("\nScenario 3: Balanced drive and inhibition")
    drive_system.active_urges = [
        CommunicationUrge(DriveType.EMOTIONAL, 0.6, priority=0.6,
                         reason="Moderate emotional expression need"),
        CommunicationUrge(DriveType.QUESTION, 0.5, priority=0.5,
                         reason="Clarification needed")
    ]
    inhibition_system.active_inhibitions = [
        InhibitionFactor(InhibitionType.BAD_TIMING, 0.5, priority=0.7,
                        reason="Spoke 3 seconds ago"),
        InhibitionFactor(InhibitionType.STILL_PROCESSING, 0.4, priority=0.5,
                        reason="Still thinking")
    ]
    
    total_drive = drive_system.get_total_drive()
    total_inhibition = inhibition_system.get_total_inhibition()
    should_inhibit = inhibition_system.should_inhibit(drive_system.active_urges, threshold=0.5)
    
    print(f"  Total drive: {total_drive:.3f}")
    print(f"  Total inhibition: {total_inhibition:.3f}")
    print(f"  Decision: {'INHIBIT (stay silent)' if should_inhibit else 'SPEAK'}")


def demo_output_frequency_limiting():
    """Demonstrate output frequency limiting."""
    print_section("Demo 4: Output Frequency Limiting")
    
    system = CommunicationInhibitionSystem(config={
        "max_output_frequency_per_minute": 3,
        "recent_output_window_minutes": 1
    })
    
    print("\nConfiguration: Max 3 outputs per minute")
    print("\nSimulating rapid outputs...")
    
    now = datetime.now()
    for i in range(5):
        system.recent_outputs.append({
            'timestamp': now - timedelta(seconds=i * 10),
            'keywords': set(),
            'content_preview': f"Output {i}"
        })
        print(f"  Output {i+1} at t-{i*10}s")
    
    inhibitions = system._compute_recent_output_inhibition()
    
    print(f"\nOutputs in last minute: {len(system.recent_outputs)}")
    if inhibitions:
        print("\n✓ FREQUENCY LIMIT TRIGGERED:")
        for inh in inhibitions:
            print_inhibition(inh)
    else:
        print("\n✗ No frequency inhibition")


def demo_inhibition_expiration():
    """Demonstrate inhibition expiration."""
    print_section("Demo 5: Inhibition Expiration")
    
    system = CommunicationInhibitionSystem()
    
    # Create inhibitions with different durations
    system.active_inhibitions = [
        InhibitionFactor(InhibitionType.BAD_TIMING, 0.8, priority=0.7,
                        duration=timedelta(seconds=5),
                        reason="Temporary timing constraint"),
        InhibitionFactor(InhibitionType.UNCERTAINTY, 0.6, priority=0.6,
                        reason="Persistent uncertainty (no expiration)")
    ]
    
    # Set first one to be created 6 seconds ago (expired)
    system.active_inhibitions[0].created_at = datetime.now() - timedelta(seconds=6)
    
    print("\nActive inhibitions before cleanup:")
    for inh in system.active_inhibitions:
        status = "EXPIRED" if inh.is_expired() else "ACTIVE"
        print(f"  • {inh.inhibition_type.value} - {status}")
        if inh.duration:
            print(f"    Duration: {inh.duration.total_seconds():.1f}s")
    
    system._cleanup_expired_inhibitions()
    
    print("\nActive inhibitions after cleanup:")
    for inh in system.active_inhibitions:
        print(f"  • {inh.inhibition_type.value} - ACTIVE")


def demo_summary():
    """Show summary of inhibition state."""
    print_section("Demo 6: Inhibition Summary")
    
    system = CommunicationInhibitionSystem()
    
    # Add various inhibitions
    system.active_inhibitions = [
        InhibitionFactor(InhibitionType.LOW_VALUE, 0.7, priority=0.7),
        InhibitionFactor(InhibitionType.UNCERTAINTY, 0.6, priority=0.6),
        InhibitionFactor(InhibitionType.BAD_TIMING, 0.5, priority=0.8)
    ]
    
    # Add some outputs
    system.record_output("First output")
    system.record_output("Second output")
    
    summary = system.get_inhibition_summary()
    
    print("\nInhibition System State:")
    print(f"  Total inhibition: {summary['total_inhibition']:.3f}")
    print(f"  Active inhibitions: {summary['active_inhibitions']}")
    print(f"  Recent outputs: {summary['recent_output_count']}")
    if summary['time_since_output']:
        print(f"  Time since last output: {summary['time_since_output']:.1f}s")
    
    print("\nInhibitions by type:")
    for inh_type, count in summary['inhibitions_by_type'].items():
        if count > 0:
            print(f"  • {inh_type}: {count}")
    
    strongest = summary['strongest_inhibition']
    if strongest:
        print(f"\nStrongest inhibition: {strongest.inhibition_type.value}")
        print(f"  Strength: {strongest.get_current_strength():.3f}")
        print(f"  Reason: {strongest.reason}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  COMMUNICATION INHIBITION SYSTEM DEMONSTRATION")
    print("  Counterbalancing force for autonomous communication agency")
    print("=" * 70)
    
    demo_basic_inhibitions()
    demo_redundancy_detection()
    demo_drive_vs_inhibition()
    demo_output_frequency_limiting()
    demo_inhibition_expiration()
    demo_summary()
    
    print("\n" + "=" * 70)
    print("  Demo complete! The inhibition system provides selective silence.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
