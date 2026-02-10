#!/usr/bin/env python3
"""
Demo script showing workspace checkpointing usage patterns.

This script demonstrates:
1. Manual checkpoint save/restore
2. Automatic periodic checkpointing
3. Checkpoint listing and management
4. Session recovery after interruption
5. Experimentation with state rollback

Usage:
    python scripts/demo_checkpointing.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emergence_core.sanctuary.cognitive_core.core import CognitiveCore
from emergence_core.sanctuary.cognitive_core.workspace import Goal, GoalType


async def demo_basic_save_restore():
    """Demo 1: Basic save and restore."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Save and Restore")
    print("="*60)
    
    # Create cognitive core with checkpointing enabled
    config = {
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": "data/checkpoints_demo/",
            "max_checkpoints": 10,
        }
    }
    
    core = CognitiveCore(config=config)
    
    print("\n1. Creating initial workspace state...")
    goal1 = Goal(type=GoalType.LEARN, description="Study consciousness", priority=0.8)
    goal2 = Goal(type=GoalType.RESPOND_TO_USER, description="Answer questions", priority=0.9)
    core.workspace.add_goal(goal1)
    core.workspace.add_goal(goal2)
    core.workspace.emotional_state["valence"] = 0.7
    
    print(f"   Goals: {len(core.workspace.current_goals)}")
    print(f"   Valence: {core.workspace.emotional_state['valence']}")
    
    print("\n2. Saving checkpoint...")
    path = core.save_state(label="Initial state for demo")
    print(f"   ✅ Saved to: {path.name}")
    
    print("\n3. Modifying state...")
    core.workspace.clear()
    print(f"   Goals: {len(core.workspace.current_goals)}")
    print(f"   Valence: {core.workspace.emotional_state['valence']}")
    
    print("\n4. Restoring from checkpoint...")
    success = core.restore_state(path)
    if success:
        print(f"   ✅ Restored successfully")
        print(f"   Goals: {len(core.workspace.current_goals)}")
        print(f"   Valence: {core.workspace.emotional_state['valence']}")
    else:
        print("   ❌ Restore failed")


async def demo_auto_save():
    """Demo 2: Automatic periodic checkpointing."""
    print("\n" + "="*60)
    print("DEMO 2: Automatic Periodic Checkpointing")
    print("="*60)
    
    config = {
        "checkpointing": {
            "enabled": True,
            "auto_save": True,
            "auto_save_interval": 2.0,  # 2 seconds for demo
            "checkpoint_dir": "data/checkpoints_demo/",
        }
    }
    
    core = CognitiveCore(config=config)
    
    print("\n1. Starting cognitive core with auto-save...")
    
    # Start core (this will start auto-save in background)
    asyncio.create_task(core.start())
    
    # Wait for core to start
    await asyncio.sleep(0.5)
    
    print(f"   ✅ Auto-save enabled (interval: 2 seconds)")
    
    print("\n2. Simulating activity...")
    for i in range(3):
        await asyncio.sleep(1.5)
        goal = Goal(type=GoalType.INTROSPECT, description=f"Reflection {i+1}")
        core.workspace.add_goal(goal)
        print(f"   Added goal: {goal.description}")
    
    print("\n3. Waiting for auto-saves...")
    await asyncio.sleep(3)
    
    print("\n4. Stopping core...")
    await core.stop()
    
    print("\n5. Listing auto-saved checkpoints...")
    checkpoints = core.checkpoint_manager.list_checkpoints()
    auto_saves = [cp for cp in checkpoints if cp.metadata.get("auto_save", False)]
    print(f"   Found {len(auto_saves)} auto-save checkpoints")
    for i, cp in enumerate(auto_saves[:3], 1):
        print(f"   {i}. {cp.timestamp.strftime('%H:%M:%S')} - {cp.size_bytes/1024:.1f} KB")


async def demo_checkpoint_management():
    """Demo 3: Listing and managing checkpoints."""
    print("\n" + "="*60)
    print("DEMO 3: Checkpoint Management")
    print("="*60)
    
    config = {
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": "data/checkpoints_demo/",
        }
    }
    
    core = CognitiveCore(config=config)
    
    print("\n1. Creating multiple labeled checkpoints...")
    labels = ["Before experiment", "Midpoint", "After completion"]
    for label in labels:
        goal = Goal(type=GoalType.CREATE, description=f"Task for {label}")
        core.workspace.add_goal(goal)
        path = core.save_state(label=label)
        print(f"   ✅ Saved: {label}")
        await asyncio.sleep(0.1)  # Ensure different timestamps
    
    print("\n2. Listing all checkpoints...")
    checkpoints = core.checkpoint_manager.list_checkpoints()
    print(f"   Total checkpoints: {len(checkpoints)}")
    for i, cp in enumerate(checkpoints[:5], 1):
        label = cp.metadata.get("user_label", "N/A")
        auto = " [auto]" if cp.metadata.get("auto_save") else ""
        print(f"   {i}. {cp.timestamp.strftime('%Y-%m-%d %H:%M:%S')}{auto}")
        print(f"      Label: {label}")
        print(f"      Size: {cp.size_bytes/1024:.1f} KB")
        print(f"      ID: {cp.checkpoint_id[:16]}...")
    
    print("\n3. Getting latest checkpoint...")
    latest = core.checkpoint_manager.get_latest_checkpoint()
    if latest:
        print(f"   ✅ Latest: {latest.name}")
    
    print("\n4. Deleting a checkpoint...")
    if len(checkpoints) > 0:
        to_delete = checkpoints[-1]  # Delete oldest
        success = core.checkpoint_manager.delete_checkpoint(to_delete.checkpoint_id)
        if success:
            print(f"   ✅ Deleted: {to_delete.checkpoint_id[:16]}...")
            remaining = core.checkpoint_manager.list_checkpoints()
            print(f"   Remaining checkpoints: {len(remaining)}")


async def demo_session_recovery():
    """Demo 4: Session recovery simulation."""
    print("\n" + "="*60)
    print("DEMO 4: Session Recovery Simulation")
    print("="*60)
    
    config = {
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": "data/checkpoints_demo/",
        }
    }
    
    print("\n1. Session 1: Creating workspace state...")
    core1 = CognitiveCore(config=config)
    
    # Build up some state
    for i in range(3):
        goal = Goal(type=GoalType.LEARN, description=f"Topic {i+1}", priority=0.5 + i*0.1)
        core1.workspace.add_goal(goal)
    
    core1.workspace.emotional_state["valence"] = 0.6
    core1.workspace.cycle_count = 42
    
    print(f"   Goals: {len(core1.workspace.current_goals)}")
    print(f"   Cycle count: {core1.workspace.cycle_count}")
    print(f"   Valence: {core1.workspace.emotional_state['valence']}")
    
    print("\n2. Saving state before 'shutdown'...")
    path = core1.save_state(label="Before shutdown")
    print(f"   ✅ Saved: {path.name}")
    
    print("\n3. Simulating session end...")
    del core1
    print("   Session 1 ended")
    
    print("\n4. Session 2: Starting new instance...")
    core2 = CognitiveCore(config=config)
    
    print("\n5. Restoring from previous session...")
    latest = core2.checkpoint_manager.get_latest_checkpoint()
    if latest:
        success = core2.restore_state(latest)
        if success:
            print(f"   ✅ Restored from: {latest.name}")
            print(f"   Goals: {len(core2.workspace.current_goals)}")
            print(f"   Cycle count: {core2.workspace.cycle_count}")
            print(f"   Valence: {core2.workspace.emotional_state['valence']}")
            print("\n   Session continuity achieved! 🎉")


async def demo_experiment_rollback():
    """Demo 5: Experiment with state rollback."""
    print("\n" + "="*60)
    print("DEMO 5: Experiment with State Rollback")
    print("="*60)
    
    config = {
        "checkpointing": {
            "enabled": True,
            "checkpoint_dir": "data/checkpoints_demo/",
        }
    }
    
    core = CognitiveCore(config=config)
    
    print("\n1. Setting up baseline state...")
    goal = Goal(type=GoalType.RESPOND_TO_USER, description="Be helpful")
    core.workspace.add_goal(goal)
    core.workspace.emotional_state["valence"] = 0.5
    
    print(f"   Baseline valence: {core.workspace.emotional_state['valence']}")
    
    print("\n2. Saving checkpoint before experiment...")
    before_path = core.save_state(label="Before risky experiment")
    print(f"   ✅ Saved: {before_path.name}")
    
    print("\n3. Running 'risky experiment' (modifying state)...")
    core.workspace.emotional_state["valence"] = -0.8  # Negative emotion
    risky_goal = Goal(type=GoalType.CREATE, description="Risky action", priority=1.0)
    core.workspace.add_goal(risky_goal)
    
    print(f"   Experimental valence: {core.workspace.emotional_state['valence']}")
    print(f"   Goals: {[g.description for g in core.workspace.current_goals]}")
    
    print("\n4. Experiment didn't go well, rolling back...")
    success = core.restore_state(before_path)
    if success:
        print(f"   ✅ Rolled back to safe state")
        print(f"   Restored valence: {core.workspace.emotional_state['valence']}")
        print(f"   Goals: {[g.description for g in core.workspace.current_goals]}")
        print("\n   State safely restored! 🎉")


async def main():
    """Run all demos."""
    print("\n" + "🔸"*30)
    print("Workspace Checkpointing Demo")
    print("🔸"*30)
    
    try:
        await demo_basic_save_restore()
        await demo_auto_save()
        await demo_checkpoint_management()
        await demo_session_recovery()
        await demo_experiment_rollback()
        
        print("\n" + "="*60)
        print("All demos completed successfully! ✅")
        print("="*60)
        print("\nCheckpoint files saved to: data/checkpoints_demo/")
        print("You can inspect these files to see the checkpoint format.")
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
