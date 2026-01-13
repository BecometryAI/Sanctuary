"""Example: Communication Decision Loop usage."""

from lyra.cognitive_core.communication import (
    CommunicationDecisionLoop,
    CommunicationDriveSystem,
    CommunicationInhibitionSystem,
    CommunicationDecision
)
from unittest.mock import MagicMock


def cognitive_cycle_example(workspace, emotional_state, goals, memories):
    """Integrate decision loop in cognitive cycle."""
    # Initialize systems
    drives = CommunicationDriveSystem()
    inhibitions = CommunicationInhibitionSystem()
    decision_loop = CommunicationDecisionLoop(drives, inhibitions)
    
    # Compute drives and inhibitions
    drives.compute_drives(workspace, emotional_state, goals, memories)
    inhibitions.compute_inhibitions(workspace, drives.active_urges, 0.8, 0.7, emotional_state)
    
    # Make decision
    result = decision_loop.evaluate(workspace, emotional_state, goals, memories)
    
    print(f"Decision: {result.decision.value.upper()}")
    print(f"Reason: {result.reason}")
    print(f"Drive: {result.drive_level:.2f}, Inhibition: {result.inhibition_level:.2f}")
    
    # Act on decision
    if result.decision == CommunicationDecision.SPEAK:
        print(f"Speaking: {result.urge.content if result.urge else 'output'}")
        drives.record_output()
        inhibitions.record_output()
    elif result.decision == CommunicationDecision.DEFER:
        print(f"Deferred until: {result.defer_until}")
    
    return result


if __name__ == "__main__":
    print("=== Communication Decision Loop Example ===\n")
    
    # Cycle 1: Low activity → SILENCE
    print("Cycle 1: Low activity")
    workspace = MagicMock(percepts={})
    emotions = {"valence": 0.0, "arousal": 0.2, "dominance": 0.5}
    cognitive_cycle_example(workspace, emotions, [], [])
    
    # Cycle 2: High emotion → SPEAK
    print("\nCycle 2: High emotion")
    emotions = {"valence": 0.9, "arousal": 0.8, "dominance": 0.7}
    cognitive_cycle_example(workspace, emotions, [], [])
    
    print("\n=== Example Complete ===")
