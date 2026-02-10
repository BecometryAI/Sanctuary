"""
Simple CLI for testing Sanctuary conversational interface.

This is a basic command-line interface for interacting with Sanctuary's cognitive core
through natural conversation. It demonstrates the use of the SanctuaryAPI for
multi-turn dialogue.

Usage:
    python -m sanctuary.cli
    
    Or:
    python sanctuary/mind/cli.py

Commands:
    - Type any message to chat with Sanctuary
    - Type 'quit' or 'exit' to exit
    - Type 'reset' to clear conversation history
    - Type 'history' to see recent conversation
    - Type 'metrics' to see conversation statistics
"""

import asyncio
import sys
from pathlib import Path

# Note: This import path manipulation is for development/testing only.
# In production, install the package properly using pip/uv.
# For proper installation, see README.md installation instructions.
try:
    from mind.client import SanctuaryAPI
except ImportError:
    # Fallback for development: add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from mind.client import SanctuaryAPI


async def main():
    """Main CLI loop for interacting with Sanctuary."""
    # Initialize Sanctuary
    print("🧠 Initializing Sanctuary...")
    sanctuary = SanctuaryAPI()

    try:
        await sanctuary.start()
        print("✅ Sanctuary is online. Type 'help' for commands or 'quit' to exit.\n")
        
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                break
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ["quit", "exit"]:
                break
            
            if user_input.lower() in ["help", "?"]:
                print("\n📖 Available Commands:")
                print("   quit, exit          - Exit the CLI")
                print("   help, ?             - Show this help message")
                print("   reset               - Clear conversation history")
                print("   history             - Show recent conversation")
                print("   metrics             - Show system metrics")
                print("   save [label]        - Save current state (optional label)")
                print("   checkpoints         - List all available checkpoints")
                print("   load <id>           - Load a specific checkpoint by ID")
                print("   restore latest      - Restore from most recent checkpoint")
                print("\n🧹 Memory Management:")
                print("   memory stats        - Show memory health statistics")
                print("   memory gc           - Manually trigger garbage collection")
                print("   memory gc --threshold <value>  - Run GC with custom threshold")
                print("   memory gc --dry-run - Preview what would be removed")
                print("   memory autogc on    - Enable automatic GC")
                print("   memory autogc off   - Disable automatic GC")
                print("\n   Any other text will be sent to Sanctuary for conversation.\n")
                continue
            
            if user_input.lower() == "reset":
                sanctuary.reset_conversation()
                print("🔄 Conversation reset.\n")
                continue
            
            if user_input.lower() == "history":
                history = sanctuary.get_conversation_history(10)
                if not history:
                    print("No conversation history yet.\n")
                else:
                    print("\n📜 Recent conversation:")
                    for i, turn in enumerate(history, 1):
                        print(f"\n{i}. You: {turn.user_input}")
                        print(f"   Sanctuary: {turn.system_response}")
                        print(f"   (Response time: {turn.response_time:.2f}s)")
                    print()
                continue
            
            if user_input.lower() == "metrics":
                metrics = sanctuary.get_metrics()
                print("\n📊 Conversation Metrics:")
                print(f"   Total turns: {metrics['conversation']['total_turns']}")
                print(f"   Average response time: {metrics['conversation']['avg_response_time']:.2f}s")
                print(f"   Timeouts: {metrics['conversation']['timeouts']}")
                print(f"   Errors: {metrics['conversation']['errors']}")
                print(f"   Topics tracked: {metrics['conversation']['topics_tracked']}")
                print(f"   History size: {metrics['conversation']['history_size']}")
                print(f"\n🧠 Cognitive Core Metrics:")
                print(f"   Total cycles: {metrics['cognitive_core']['total_cycles']}")
                print(f"   Average cycle time: {metrics['cognitive_core']['avg_cycle_time_ms']:.2f}ms")
                print(f"   Workspace size: {metrics['cognitive_core']['workspace_size']}")
                print(f"   Current goals: {metrics['cognitive_core']['current_goals']}")
                print()
                continue
            
            # Checkpoint commands
            if user_input.lower().startswith("save"):
                parts = user_input.split(maxsplit=1)
                label = parts[1] if len(parts) > 1 else None
                path = sanctuary.core.save_state(label)
                if path:
                    print(f"💾 State saved: {path.name}\n")
                else:
                    print("❌ Failed to save state (checkpointing may be disabled)\n")
                continue
            
            if user_input.lower() == "checkpoints":
                if not sanctuary.core.checkpoint_manager:
                    print("❌ Checkpointing is disabled\n")
                    continue
                
                checkpoints = sanctuary.core.checkpoint_manager.list_checkpoints()
                if not checkpoints:
                    print("No checkpoints found.\n")
                else:
                    print(f"\n💾 Available Checkpoints ({len(checkpoints)}):")
                    for i, cp in enumerate(checkpoints[:10], 1):  # Show max 10
                        label = cp.metadata.get('user_label', 'N/A')
                        auto = " [auto]" if cp.metadata.get('auto_save') else ""
                        shutdown = " [shutdown]" if cp.metadata.get('shutdown') else ""
                        size_kb = cp.size_bytes / 1024
                        print(f"\n{i}. {cp.timestamp.strftime('%Y-%m-%d %H:%M:%S')}{auto}{shutdown}")
                        print(f"   ID: {cp.checkpoint_id[:16]}...")
                        print(f"   Label: {label}")
                        print(f"   Size: {size_kb:.1f} KB")
                    print()
                continue
            
            if user_input.lower().startswith("load"):
                if not sanctuary.core.checkpoint_manager:
                    print("❌ Checkpointing is disabled\n")
                    continue
                
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("❌ Usage: load <checkpoint_id>\n")
                    continue
                
                checkpoint_id = parts[1]
                
                # Find checkpoint by ID prefix
                checkpoints = sanctuary.core.checkpoint_manager.list_checkpoints()
                matching = [cp for cp in checkpoints if cp.checkpoint_id.startswith(checkpoint_id)]
                
                if not matching:
                    print(f"❌ Checkpoint not found: {checkpoint_id}\n")
                    continue
                
                if len(matching) > 1:
                    print(f"❌ Ambiguous checkpoint ID (matches {len(matching)} checkpoints)\n")
                    continue
                
                checkpoint = matching[0]
                
                # Cannot load while running - need to stop first
                print(f"⚠️  Loading checkpoint requires restarting Sanctuary...")
                print(f"💾 Stopping Sanctuary...")
                await sanctuary.stop()
                
                # Restore state
                success = sanctuary.core.restore_state(checkpoint.path)
                if success:
                    print(f"✅ State restored from {checkpoint.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"🧠 Restarting Sanctuary...")
                    await sanctuary.start()
                    print("✅ Sanctuary is online.\n")
                else:
                    print("❌ Failed to restore state")
                    print("🧠 Restarting Sanctuary with original state...")
                    await sanctuary.start()
                    print("✅ Sanctuary is online.\n")
                continue
            
            if user_input.lower() == "restore latest":
                if not sanctuary.core.checkpoint_manager:
                    print("❌ Checkpointing is disabled\n")
                    continue
                
                latest = sanctuary.core.checkpoint_manager.get_latest_checkpoint()
                if not latest:
                    print("❌ No checkpoints found\n")
                    continue
                
                print(f"⚠️  Loading checkpoint requires restarting Sanctuary...")
                print(f"💾 Stopping Sanctuary...")
                await sanctuary.stop()
                
                # Restore state
                success = sanctuary.core.restore_state(latest)
                if success:
                    print(f"✅ State restored from latest checkpoint")
                    print(f"🧠 Restarting Sanctuary...")
                    await sanctuary.start()
                    print("✅ Sanctuary is online.\n")
                else:
                    print("❌ Failed to restore state")
                    print("🧠 Restarting Sanctuary with original state...")
                    await sanctuary.start()
                    print("✅ Sanctuary is online.\n")
                continue
            
            # Memory management commands
            if user_input.lower().startswith("memory"):
                parts = user_input.lower().split()
                
                if len(parts) < 2:
                    print("❌ Usage: memory <stats|gc|autogc>\n")
                    continue
                
                command = parts[1]
                
                # memory stats
                if command == "stats":
                    print("📊 Analyzing memory health...")
                    health = await sanctuary.core.memory.memory_manager.get_memory_health()
                    
                    print(f"\n🧹 Memory System Health:")
                    print(f"   Total memories: {health.total_memories}")
                    print(f"   Total size: {health.total_size_mb:.2f} MB")
                    print(f"   Average significance: {health.avg_significance:.2f}")
                    print(f"   Oldest memory: {health.oldest_memory_age_days:.1f} days")
                    print(f"   Newest memory: {health.newest_memory_age_days:.1f} days")
                    print(f"   Estimated duplicates: {health.estimated_duplicates}")
                    print(f"   Needs collection: {'Yes' if health.needs_collection else 'No'}")
                    print(f"   Recommended threshold: {health.recommended_threshold:.2f}")
                    
                    if health.significance_distribution:
                        print(f"\n   Significance Distribution:")
                        for bucket, count in sorted(health.significance_distribution.items()):
                            print(f"      {bucket}: {count} memories")
                    print()
                    continue
                
                # memory gc
                elif command == "gc":
                    threshold = None
                    dry_run = False
                    
                    # Parse options
                    if "--threshold" in parts:
                        try:
                            idx = parts.index("--threshold")
                            if idx + 1 < len(parts):
                                threshold = float(parts[idx + 1])
                        except (ValueError, IndexError):
                            print("❌ Invalid threshold value\n")
                            continue
                    
                    if "--dry-run" in parts:
                        dry_run = True
                    
                    mode_str = "DRY RUN" if dry_run else "ACTIVE"
                    threshold_str = f"threshold={threshold}" if threshold else "default threshold"
                    print(f"🧹 Running garbage collection ({mode_str}, {threshold_str})...")
                    
                    stats = await sanctuary.core.memory.memory_manager.run_gc(
                        threshold=threshold,
                        dry_run=dry_run
                    )
                    
                    print(f"\n✅ Garbage Collection Complete:")
                    print(f"   Memories analyzed: {stats.memories_analyzed}")
                    print(f"   Memories removed: {stats.memories_removed}")
                    print(f"   Bytes freed: {stats.bytes_freed:,}")
                    print(f"   Duration: {stats.duration_seconds:.2f}s")
                    print(f"   Avg significance before: {stats.avg_significance_before:.2f}")
                    print(f"   Avg significance after: {stats.avg_significance_after:.2f}")
                    
                    if stats.removal_reasons:
                        print(f"\n   Removal Reasons:")
                        for reason, count in stats.removal_reasons.items():
                            print(f"      {reason}: {count}")
                    print()
                    continue
                
                # memory autogc
                elif command == "autogc":
                    if len(parts) < 3:
                        print("❌ Usage: memory autogc <on|off>\n")
                        continue
                    
                    action = parts[2]
                    
                    if action == "on":
                        sanctuary.core.memory.memory_manager.enable_auto_gc()
                        print("✅ Automatic garbage collection enabled\n")
                    elif action == "off":
                        sanctuary.core.memory.memory_manager.disable_auto_gc()
                        print("✅ Automatic garbage collection disabled\n")
                    else:
                        print("❌ Usage: memory autogc <on|off>\n")
                    
                    continue
                
                else:
                    print(f"❌ Unknown memory command: {command}\n")
                    continue
            
            # Process turn
            print("💭 Thinking...")
            turn = await sanctuary.chat(user_input)
            
            # Display response with emotion
            emotion = turn.emotional_state
            if emotion:
                valence = emotion.get('valence', 0.0)
                arousal = emotion.get('arousal', 0.0)
                emotion_label = f"[{valence:.1f}V {arousal:.1f}A]"
            else:
                emotion_label = ""
            
            print(f"\nSanctuary {emotion_label}: {turn.system_response}")
            print(f"(Response time: {turn.response_time:.2f}s)\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🛑 Shutting down Sanctuary...")
        await sanctuary.stop()
        print("👋 Sanctuary offline.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
