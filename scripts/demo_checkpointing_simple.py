#!/usr/bin/env python3
"""
Simplified demo script showing workspace checkpointing without running the cognitive loop.

This script demonstrates core checkpoint functionality without triggering
the full cognitive loop, to avoid unrelated bugs in the codebase.

Usage:
    python scripts/demo_checkpointing_simple.py
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emergence_core.sanctuary.cognitive_core.checkpoint import CheckpointManager
from emergence_core.sanctuary.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType, Percept


def demo_checkpoint_basics():
    """Demo basic checkpoint save/load functionality."""
    print("\n" + "="*60)
    print("Workspace Checkpointing Demo")
    print("="*60)
    
    # Setup
    checkpoint_dir = Path("data/checkpoints_demo_simple")
    manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=10,
        compression=True,
    )
    
    # Demo 1: Create and save workspace
    print("\n1. Creating initial workspace state...")
    workspace1 = GlobalWorkspace(capacity=10)
    
    goal1 = Goal(type=GoalType.LEARN, description="Study consciousness", priority=0.8)
    goal2 = Goal(type=GoalType.RESPOND_TO_USER, description="Answer questions", priority=0.9)
    workspace1.add_goal(goal1)
    workspace1.add_goal(goal2)
    
    percept = Percept(modality="text", raw="Hello world", complexity=2)
    workspace1.active_percepts[percept.id] = percept
    
    workspace1.emotional_state["valence"] = 0.7
    workspace1.cycle_count = 42
    
    print(f"   Goals: {len(workspace1.current_goals)}")
    print(f"   Percepts: {len(workspace1.active_percepts)}")
    print(f"   Valence: {workspace1.emotional_state['valence']}")
    print(f"   Cycle count: {workspace1.cycle_count}")
    
    # Demo 2: Save checkpoint
    print("\n2. Saving checkpoint...")
    path = manager.save_checkpoint(
        workspace1,
        metadata={"user_label": "Initial state for demo"}
    )
    print(f"   ✅ Saved to: {path.name}")
    print(f"   Size: {path.stat().st_size / 1024:.1f} KB")
    
    # Demo 3: Load checkpoint
    print("\n3. Loading checkpoint...")
    workspace2 = manager.load_checkpoint(path)
    
    print(f"   ✅ Loaded successfully")
    print(f"   Goals: {len(workspace2.current_goals)}")
    print(f"   Percepts: {len(workspace2.active_percepts)}")
    print(f"   Valence: {workspace2.emotional_state['valence']}")
    print(f"   Cycle count: {workspace2.cycle_count}")
    
    # Verify state matches
    assert len(workspace2.current_goals) == 2
    assert len(workspace2.active_percepts) == 1
    assert workspace2.emotional_state["valence"] == 0.7
    assert workspace2.cycle_count == 42
    print("   ✅ All state verified!")
    
    # Demo 4: Create multiple checkpoints
    print("\n4. Creating multiple checkpoints...")
    for i in range(3):
        ws = GlobalWorkspace()
        goal = Goal(type=GoalType.CREATE, description=f"Task {i+1}")
        ws.add_goal(goal)
        manager.save_checkpoint(ws, metadata={"user_label": f"Checkpoint {i+1}"})
        time.sleep(0.05)  # Ensure different timestamps
    
    print(f"   ✅ Created 3 additional checkpoints")
    
    # Demo 5: List checkpoints
    print("\n5. Listing all checkpoints...")
    checkpoints = manager.list_checkpoints()
    print(f"   Total: {len(checkpoints)}")
    for i, cp in enumerate(checkpoints[:5], 1):
        label = cp.metadata.get("user_label", "N/A")
        size_kb = cp.size_bytes / 1024
        print(f"   {i}. {cp.timestamp.strftime('%H:%M:%S')} - {label} ({size_kb:.1f} KB)")
    
    # Demo 6: Get latest checkpoint
    print("\n6. Getting latest checkpoint...")
    latest = manager.get_latest_checkpoint()
    if latest:
        print(f"   ✅ Latest: {latest.name}")
        ws_latest = manager.load_checkpoint(latest)
        print(f"   Goals in latest: {[g.description for g in ws_latest.current_goals]}")
    
    # Demo 7: Delete a checkpoint
    print("\n7. Deleting oldest checkpoint...")
    oldest = checkpoints[-1]
    success = manager.delete_checkpoint(oldest.checkpoint_id)
    if success:
        print(f"   ✅ Deleted: {oldest.checkpoint_id[:16]}...")
        remaining = manager.list_checkpoints()
        print(f"   Remaining: {len(remaining)} checkpoints")
    
    # Demo 8: Test compression
    print("\n8. Testing compression...")
    ws_large = GlobalWorkspace()
    for i in range(50):
        goal = Goal(type=GoalType.LEARN, description=f"Large goal {i}")
        ws_large.add_goal(goal)
    
    path_compressed = manager.save_checkpoint(ws_large)
    print(f"   ✅ Compressed checkpoint: {path_compressed.stat().st_size / 1024:.1f} KB")
    print(f"   Compression: {'.gz' in path_compressed.name}")
    
    # Verify it loads correctly
    ws_loaded = manager.load_checkpoint(path_compressed)
    assert len(ws_loaded.current_goals) == 50
    print(f"   ✅ Loaded with all 50 goals intact")
    
    print("\n" + "="*60)
    print("All demos completed successfully! ✅")
    print("="*60)
    print(f"\nCheckpoint files saved to: {checkpoint_dir}")
    print("You can inspect these files to see the checkpoint format.\n")


if __name__ == "__main__":
    try:
        demo_checkpoint_basics()
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
