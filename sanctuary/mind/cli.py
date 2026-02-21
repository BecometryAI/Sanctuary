"""
Sanctuary CLI — hardened interactive interface.

Provides a robust command-line interface for interacting with Sanctuary's cognitive
core through natural conversation.  Includes signal handling (SIGTERM / SIGINT),
shutdown timeout, checkpoint-on-exit guarantee, argparse configuration, and
categorised error display.

Usage:
    python -m sanctuary.cli
    python sanctuary/mind/cli.py
    python sanctuary/mind/cli.py --verbose
    python sanctuary/mind/cli.py --restore-latest --auto-save --auto-save-interval 120
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Import resolution — development fallback when not pip-installed.
# ---------------------------------------------------------------------------
try:
    from mind.client import SanctuaryAPI
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from mind.client import SanctuaryAPI

logger = logging.getLogger(__name__)

# Maximum seconds to wait for a graceful shutdown before force-quitting.
SHUTDOWN_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="sanctuary",
        description="Sanctuary — Interactive cognitive architecture CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Interactive commands (inside the REPL):
  help, ?             Show in-REPL help
  quit, exit          Exit the CLI
  reset               Clear conversation history
  history             Show recent conversation turns
  metrics             Show cognitive & conversation metrics
  health              Show subsystem health report
  save [label]        Manual checkpoint with optional label
  checkpoints         List saved checkpoints
  load <id>           Load a checkpoint by ID prefix
  restore latest      Restore the most recent checkpoint
  memory stats        Show memory health statistics
  memory gc           Run garbage collection
  memory autogc on|off  Toggle automatic GC
""",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--restore-latest", action="store_true",
        help="Restore from the most recent checkpoint on startup",
    )
    parser.add_argument(
        "--auto-save", action="store_true",
        help="Enable automatic periodic checkpoints",
    )
    parser.add_argument(
        "--auto-save-interval", type=float, default=300.0,
        help="Auto-save interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--cycle-rate", type=float, default=10.0,
        help="Cognitive cycle rate in Hz (default: 10.0)",
    )
    parser.add_argument(
        "--shutdown-timeout", type=float, default=SHUTDOWN_TIMEOUT,
        help=f"Max seconds to wait for graceful shutdown (default: {SHUTDOWN_TIMEOUT})",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Error formatting
# ---------------------------------------------------------------------------

def _format_error(exc: Exception, verbose: bool = False) -> str:
    """Return a user-friendly error string, optionally with traceback."""
    # Map known exception types to short prefixes
    from mind.exceptions import (
        ModelLoadError, GPUMemoryError, RateLimitError,
        ConcurrencyError, ValidationError,
    )
    prefix_map = {
        GPUMemoryError: "GPU memory exhausted",
        ModelLoadError: "Model load failure",
        RateLimitError: "Rate limited",
        ConcurrencyError: "Concurrency error",
        ValidationError: "Validation error",
        RuntimeError: "Runtime error",
        ConnectionError: "Connection error",
        TimeoutError: "Operation timed out",
        asyncio.TimeoutError: "Operation timed out",
        KeyboardInterrupt: "Interrupted",
    }
    prefix = "Error"
    for exc_type, label in prefix_map.items():
        if isinstance(exc, exc_type):
            prefix = label
            break

    msg = f"{prefix}: {exc}"
    if verbose:
        msg += "\n" + traceback.format_exc()
    return msg


# ---------------------------------------------------------------------------
# REPL command handlers (extracted for testability)
# ---------------------------------------------------------------------------

async def _handle_health(sanctuary: SanctuaryAPI) -> None:
    """Print subsystem health report."""
    report = sanctuary.core.get_health_report()
    print("\n🏥 Subsystem Health Report:")
    status_summary = report.get("status_summary", {})
    for status, count in status_summary.items():
        print(f"   {status}: {count}")
    subsystems = report.get("subsystems", {})
    for name, info in sorted(subsystems.items()):
        state = info.get("state", "UNKNOWN")
        fails = info.get("consecutive_failures", 0)
        indicator = "✅" if state == "HEALTHY" else "⚠️" if state == "DEGRADED" else "❌"
        extra = f" (failures: {fails})" if fails > 0 else ""
        print(f"   {indicator} {name}: {state}{extra}")
    print()


async def _handle_metrics(sanctuary: SanctuaryAPI) -> None:
    """Print conversation + cognitive metrics."""
    metrics = sanctuary.get_metrics()
    conv = metrics.get("conversation", {})
    cog = metrics.get("cognitive_core", {})
    print("\n📊 Conversation Metrics:")
    print(f"   Total turns: {conv.get('total_turns', 0)}")
    print(f"   Average response time: {conv.get('avg_response_time', 0):.2f}s")
    print(f"   Timeouts: {conv.get('timeouts', 0)}")
    print(f"   Errors: {conv.get('errors', 0)}")
    print(f"   Topics tracked: {conv.get('topics_tracked', 0)}")
    print(f"   History size: {conv.get('history_size', 0)}")
    print(f"\n🧠 Cognitive Core Metrics:")
    print(f"   Total cycles: {cog.get('total_cycles', 0)}")
    print(f"   Average cycle time: {cog.get('avg_cycle_time_ms', 0):.2f}ms")
    print(f"   Workspace size: {cog.get('workspace_size', 0)}")
    print(f"   Current goals: {cog.get('current_goals', 0)}")
    print()


async def _handle_history(sanctuary: SanctuaryAPI) -> None:
    """Print recent conversation history."""
    history = sanctuary.get_conversation_history(10)
    if not history:
        print("No conversation history yet.\n")
        return
    print("\n📜 Recent conversation:")
    for i, turn in enumerate(history, 1):
        print(f"\n{i}. You: {turn.user_input}")
        print(f"   Sanctuary: {turn.system_response}")
        print(f"   (Response time: {turn.response_time:.2f}s)")
    print()


async def _handle_save(sanctuary: SanctuaryAPI, user_input: str) -> None:
    """Save checkpoint with optional label."""
    parts = user_input.split(maxsplit=1)
    label = parts[1] if len(parts) > 1 else None
    path = sanctuary.core.save_state(label)
    if path:
        print(f"💾 State saved: {path.name}\n")
    else:
        print("❌ Failed to save state (checkpointing may be disabled)\n")


async def _handle_checkpoints(sanctuary: SanctuaryAPI) -> None:
    """List available checkpoints."""
    if not sanctuary.core.checkpoint_manager:
        print("❌ Checkpointing is disabled\n")
        return
    checkpoints = sanctuary.core.checkpoint_manager.list_checkpoints()
    if not checkpoints:
        print("No checkpoints found.\n")
        return
    print(f"\n💾 Available Checkpoints ({len(checkpoints)}):")
    for i, cp in enumerate(checkpoints[:10], 1):
        label = cp.metadata.get("user_label", "N/A")
        auto = " [auto]" if cp.metadata.get("auto_save") else ""
        shutdown = " [shutdown]" if cp.metadata.get("shutdown") else ""
        size_kb = cp.size_bytes / 1024
        print(f"\n{i}. {cp.timestamp.strftime('%Y-%m-%d %H:%M:%S')}{auto}{shutdown}")
        print(f"   ID: {cp.checkpoint_id[:16]}...")
        print(f"   Label: {label}")
        print(f"   Size: {size_kb:.1f} KB")
    print()


async def _handle_load(sanctuary: SanctuaryAPI, user_input: str) -> None:
    """Load a checkpoint by ID prefix."""
    if not sanctuary.core.checkpoint_manager:
        print("❌ Checkpointing is disabled\n")
        return
    parts = user_input.split(maxsplit=1)
    if len(parts) < 2:
        print("❌ Usage: load <checkpoint_id>\n")
        return
    checkpoint_id = parts[1]
    checkpoints = sanctuary.core.checkpoint_manager.list_checkpoints()
    matching = [cp for cp in checkpoints if cp.checkpoint_id.startswith(checkpoint_id)]
    if not matching:
        print(f"❌ Checkpoint not found: {checkpoint_id}\n")
        return
    if len(matching) > 1:
        print(f"❌ Ambiguous checkpoint ID (matches {len(matching)} checkpoints)\n")
        return
    checkpoint = matching[0]
    print("⚠️  Loading checkpoint requires restarting Sanctuary...")
    print("💾 Stopping Sanctuary...")
    await sanctuary.stop()
    success = sanctuary.core.restore_state(checkpoint.path)
    if success:
        print(f"✅ State restored from {checkpoint.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print("❌ Failed to restore state")
    print("🧠 Restarting Sanctuary...")
    await sanctuary.start()
    print("✅ Sanctuary is online.\n")


async def _handle_restore_latest(sanctuary: SanctuaryAPI) -> None:
    """Restore from the latest checkpoint."""
    if not sanctuary.core.checkpoint_manager:
        print("❌ Checkpointing is disabled\n")
        return
    latest = sanctuary.core.checkpoint_manager.get_latest_checkpoint()
    if not latest:
        print("❌ No checkpoints found\n")
        return
    print("⚠️  Loading checkpoint requires restarting Sanctuary...")
    print("💾 Stopping Sanctuary...")
    await sanctuary.stop()
    success = sanctuary.core.restore_state(latest)
    if success:
        print("✅ State restored from latest checkpoint")
    else:
        print("❌ Failed to restore state")
    print("🧠 Restarting Sanctuary...")
    await sanctuary.start()
    print("✅ Sanctuary is online.\n")


async def _handle_memory(sanctuary: SanctuaryAPI, user_input: str) -> None:
    """Dispatch memory sub-commands."""
    parts = user_input.lower().split()
    if len(parts) < 2:
        print("❌ Usage: memory <stats|gc|autogc>\n")
        return

    command = parts[1]
    mm = sanctuary.core.memory.memory_manager

    if command == "stats":
        print("📊 Analyzing memory health...")
        health = await mm.get_memory_health()
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

    elif command == "gc":
        threshold = None
        dry_run = "--dry-run" in parts
        if "--threshold" in parts:
            try:
                idx = parts.index("--threshold")
                threshold = float(parts[idx + 1])
            except (ValueError, IndexError):
                print("❌ Invalid threshold value\n")
                return
        mode_str = "DRY RUN" if dry_run else "ACTIVE"
        threshold_str = f"threshold={threshold}" if threshold else "default threshold"
        print(f"🧹 Running garbage collection ({mode_str}, {threshold_str})...")
        stats = await mm.run_gc(threshold=threshold, dry_run=dry_run)
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

    elif command == "autogc":
        if len(parts) < 3:
            print("❌ Usage: memory autogc <on|off>\n")
            return
        action = parts[2]
        if action == "on":
            mm.enable_auto_gc()
            print("✅ Automatic garbage collection enabled\n")
        elif action == "off":
            mm.disable_auto_gc()
            print("✅ Automatic garbage collection disabled\n")
        else:
            print("❌ Usage: memory autogc <on|off>\n")

    else:
        print(f"❌ Unknown memory command: {command}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main(argv=None):
    """Main CLI loop for interacting with Sanctuary."""
    args = parse_args(argv)

    # ------- Logging -------
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.verbose:
        logging.getLogger("mind").setLevel(logging.WARNING)
        logging.getLogger("sanctuary").setLevel(logging.WARNING)

    # ------- Build config from args -------
    config = {
        "cognitive_core": {
            "cycle_rate_hz": args.cycle_rate,
            "checkpointing": {
                "enabled": True,
                "auto_save": args.auto_save,
                "auto_save_interval": args.auto_save_interval,
                "checkpoint_on_shutdown": True,
            },
        }
    }

    # ------- Signal handling -------
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _signal_handler(sig, _frame=None):
        """Handle SIGTERM/SIGINT by requesting clean shutdown."""
        sig_name = signal.Signals(sig).name
        print(f"\n🛑 Received {sig_name}, shutting down gracefully...")
        shutdown_event.set()

    # Register handlers (SIGTERM not available on Windows for add_signal_handler)
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler, sig)
    else:
        signal.signal(signal.SIGINT, _signal_handler)

    # ------- Boot -------
    sanctuary = None
    exit_code = 0

    try:
        print("🧠 Initializing Sanctuary...")
        sanctuary = SanctuaryAPI(config)
        await sanctuary.start()

        if args.restore_latest:
            if sanctuary.core.checkpoint_manager:
                latest = sanctuary.core.checkpoint_manager.get_latest_checkpoint()
                if latest:
                    sanctuary.core.restore_state(latest)
                    print("✅ Restored from latest checkpoint.")

        print("✅ Sanctuary is online. Type 'help' for commands or 'quit' to exit.\n")

        # ------- REPL -------
        while not shutdown_event.is_set():
            try:
                user_input = await asyncio.to_thread(input, "You: ")
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            lower = user_input.lower()

            # ---- exit ----
            if lower in ("quit", "exit"):
                break

            # ---- help ----
            if lower in ("help", "?"):
                print("\n📖 Available Commands:")
                print("   quit, exit          - Exit the CLI")
                print("   help, ?             - Show this help message")
                print("   reset               - Clear conversation history")
                print("   history             - Show recent conversation")
                print("   metrics             - Show system metrics")
                print("   health              - Show subsystem health report")
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

            # ---- simple commands ----
            if lower == "reset":
                sanctuary.reset_conversation()
                print("🔄 Conversation reset.\n")
                continue
            if lower == "history":
                await _handle_history(sanctuary)
                continue
            if lower == "metrics":
                await _handle_metrics(sanctuary)
                continue
            if lower == "health":
                await _handle_health(sanctuary)
                continue

            # ---- checkpoint commands ----
            if lower.startswith("save"):
                await _handle_save(sanctuary, user_input)
                continue
            if lower == "checkpoints":
                await _handle_checkpoints(sanctuary)
                continue
            if lower.startswith("load"):
                await _handle_load(sanctuary, user_input)
                continue
            if lower == "restore latest":
                await _handle_restore_latest(sanctuary)
                continue

            # ---- memory commands ----
            if lower.startswith("memory"):
                await _handle_memory(sanctuary, user_input)
                continue

            # ---- chat ----
            try:
                print("💭 Thinking...")
                turn = await sanctuary.chat(user_input)
                emotion = turn.emotional_state
                if emotion:
                    valence = emotion.get("valence", 0.0)
                    arousal = emotion.get("arousal", 0.0)
                    emotion_label = f"[{valence:.1f}V {arousal:.1f}A]"
                else:
                    emotion_label = ""
                print(f"\nSanctuary {emotion_label}: {turn.system_response}")
                print(f"(Response time: {turn.response_time:.2f}s)\n")
            except Exception as chat_err:
                print(f"\n❌ {_format_error(chat_err, args.verbose)}\n")

    except Exception as e:
        print(f"\n❌ {_format_error(e, args.verbose)}")
        exit_code = 1

    finally:
        # ------- Shutdown -------
        if sanctuary is not None:
            print("\n🛑 Shutting down Sanctuary...")
            try:
                await asyncio.wait_for(
                    sanctuary.stop(),
                    timeout=args.shutdown_timeout,
                )
            except asyncio.TimeoutError:
                print(f"⚠️  Shutdown timed out after {args.shutdown_timeout}s — forcing exit")
            except Exception as stop_err:
                print(f"⚠️  Error during shutdown: {stop_err}")
            print("👋 Sanctuary offline.")

    return exit_code


if __name__ == "__main__":
    try:
        code = asyncio.run(main())
        sys.exit(code)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(130)
