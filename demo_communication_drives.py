#!/usr/bin/env python3
"""
Demonstration of the Communication Drive System.

This script shows how the communication drive system computes internal
urges to communicate based on various factors.
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

def demo_emotional_drive():
    """Demonstrate emotional drive computation."""
    print("\n=== Emotional Drive Demo ===\n")
    
    system = CommunicationDriveSystem(config={"emotional_threshold": 0.6})
    
    # High arousal state (excitement/anxiety)
    emotional_state = {
        "valence": 0.3,
        "arousal": 0.85,  # Very high arousal
        "dominance": 0.6
    }
    
    urges = system._compute_emotional_drive(emotional_state)
    
    print(f"Emotional State: valence={emotional_state['valence']:.2f}, "
          f"arousal={emotional_state['arousal']:.2f}")
    print(f"Generated {len(urges)} emotional drive(s):")
    for urge in urges:
        print(f"  - Intensity: {urge.intensity:.2f}")
        print(f"    Reason: {urge.reason}")
        print(f"    Priority: {urge.priority:.2f}")

def demo_social_drive():
    """Demonstrate social drive computation."""
    print("\n=== Social Drive Demo ===\n")
    
    system = CommunicationDriveSystem(config={"social_silence_minutes": 10})
    
    # Simulate 20 minutes of silence
    system.last_input_time = datetime.now() - timedelta(minutes=20)
    
    urges = system._compute_social_drive()
    
    print(f"Time since last input: 20 minutes")
    print(f"Social silence threshold: 10 minutes")
    print(f"Generated {len(urges)} social drive(s):")
    for urge in urges:
        print(f"  - Intensity: {urge.intensity:.2f}")
        print(f"    Reason: {urge.reason}")
        print(f"    Priority: {urge.priority:.2f}")

def demo_goal_drive():
    """Demonstrate goal-driven communication."""
    print("\n=== Goal Drive Demo ===\n")
    
    system = CommunicationDriveSystem()
    
    # Mock goal that requires response
    class MockGoal:
        def __init__(self):
            self.type = "RESPOND_TO_USER"
            self.description = "Answer user's question about consciousness"
            self.priority = 0.9
    
    goals = [MockGoal()]
    urges = system._compute_goal_drive(goals)
    
    print(f"Active goal: {goals[0].description}")
    print(f"Goal type: {goals[0].type}")
    print(f"Generated {len(urges)} goal drive(s):")
    for urge in urges:
        print(f"  - Intensity: {urge.intensity:.2f}")
        print(f"    Content: {urge.content}")
        print(f"    Reason: {urge.reason}")
        print(f"    Priority: {urge.priority:.2f}")

def demo_urge_decay():
    """Demonstrate urge decay over time."""
    print("\n=== Urge Decay Demo ===\n")
    
    urge = CommunicationUrge(
        drive_type=DriveType.INSIGHT,
        intensity=1.0,
        content="Important realization",
        decay_rate=0.2  # 20% decay per minute
    )
    
    print(f"Initial urge: {urge.content}")
    print(f"Initial intensity: {urge.intensity:.2f}")
    print(f"Decay rate: {urge.decay_rate:.2f} per minute\n")
    
    # Simulate time passing
    for minutes in [1, 2, 3, 5]:
        urge.created_at = datetime.now() - timedelta(minutes=minutes)
        current = urge.get_current_intensity()
        expired = urge.is_expired()
        print(f"After {minutes} minute(s): intensity={current:.2f}, "
              f"expired={expired}")

def demo_total_drive_computation():
    """Demonstrate total drive aggregation."""
    print("\n=== Total Drive Computation Demo ===\n")
    
    system = CommunicationDriveSystem()
    
    # Add multiple urges
    system.active_urges = [
        CommunicationUrge(DriveType.INSIGHT, 0.9, priority=0.8),
        CommunicationUrge(DriveType.EMOTIONAL, 0.7, priority=0.6),
        CommunicationUrge(DriveType.SOCIAL, 0.4, priority=0.4),
        CommunicationUrge(DriveType.GOAL, 0.8, priority=0.75),
    ]
    
    print(f"Active urges:")
    for urge in system.active_urges:
        print(f"  - {urge.drive_type.value}: "
              f"intensity={urge.intensity:.2f}, priority={urge.priority:.2f}")
    
    total = system.get_total_drive()
    strongest = system.get_strongest_urge()
    
    print(f"\nTotal drive intensity: {total:.3f}")
    print(f"Strongest urge: {strongest.drive_type.value} "
          f"(intensity={strongest.intensity:.2f}, priority={strongest.priority:.2f})")

def demo_drive_summary():
    """Demonstrate drive summary generation."""
    print("\n=== Drive Summary Demo ===\n")
    
    system = CommunicationDriveSystem()
    
    # Add various urges
    system.active_urges = [
        CommunicationUrge(DriveType.INSIGHT, 0.8, priority=0.7),
        CommunicationUrge(DriveType.INSIGHT, 0.6, priority=0.6),
        CommunicationUrge(DriveType.EMOTIONAL, 0.7, priority=0.65),
        CommunicationUrge(DriveType.SOCIAL, 0.3, priority=0.4),
    ]
    
    # Record some I/O times
    system.last_input_time = datetime.now() - timedelta(minutes=5)
    system.last_output_time = datetime.now() - timedelta(minutes=2)
    
    summary = system.get_drive_summary()
    
    print(f"Total drive: {summary['total_drive']:.3f}")
    print(f"Active urges: {summary['active_urges']}")
    print(f"Strongest urge: {summary['strongest_urge'].drive_type.value if summary['strongest_urge'] else None}")
    print(f"\nUrges by type:")
    for drive_type, count in summary['urges_by_type'].items():
        if count > 0:
            print(f"  - {drive_type}: {count}")
    print(f"\nTime since last input: {summary['time_since_input']:.1f} seconds")
    print(f"Time since last output: {summary['time_since_output']:.1f} seconds")

def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("Communication Drive System Demonstration")
    print("="*60)
    
    demos = [
        demo_emotional_drive,
        demo_social_drive,
        demo_goal_drive,
        demo_urge_decay,
        demo_total_drive_computation,
        demo_drive_summary,
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"\n✗ Demo failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Demonstration Complete")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
