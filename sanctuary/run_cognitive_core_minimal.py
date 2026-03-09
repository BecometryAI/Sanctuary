#!/usr/bin/env python3
"""
Minimal Cognitive Core CLI - Single Cycle Demonstration

This script demonstrates the cognitive core running a single cycle.
It initializes the system, executes one cognitive loop iteration,
prints the workspace state, and exits cleanly.

Purpose: Prove the cognitive architecture is functional
Usage: python sanctuary/run_cognitive_core_minimal.py
Expected: Completes in < 5 seconds with workspace state output
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mind.cognitive_core.core import CognitiveCore
from mind.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType

# Configuration constants
CYCLE_COMPLETION_MULTIPLIER = 1.5  # Wait 1.5 cycles to ensure at least one completes


async def main():
    """Run a single cognitive cycle and display results."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 70)
    print("MINIMAL COGNITIVE CORE TEST - SINGLE CYCLE EXECUTION")
    print("=" * 70)
    print()
    
    # Step 1: Initialize workspace
    print("Step 1: Initializing GlobalWorkspace...")
    workspace = GlobalWorkspace()
    print(f"✅ Workspace initialized")
    print()
    
    # Step 2: Initialize cognitive core with minimal config
    print("Step 2: Initializing CognitiveCore...")
    config = {
        "cycle_rate_hz": 10,
        "attention_budget": 100,
        "max_queue_size": 100,
        "log_interval_cycles": 1,  # Log every cycle for visibility
        "checkpointing": {"enabled": False},  # Disable for minimal test
        "input_llm": {"use_real_model": False},  # Use mock for speed
        "output_llm": {"use_real_model": False},  # Use mock for speed
    }
    
    core = CognitiveCore(workspace=workspace, config=config)
    print(f"✅ CognitiveCore initialized")
    print(f"   - Cycle rate: {config['cycle_rate_hz']} Hz")
    print(f"   - Attention budget: {config['attention_budget']}")
    print()
    
    # Step 3: Add a test goal to workspace
    print("Step 3: Adding test goal to workspace...")
    test_goal = Goal(
        type=GoalType.RESPOND_TO_USER,
        description="Test goal: Verify cognitive cycle execution",
        priority=0.5,
        metadata={"test": True, "created_at": datetime.now().isoformat()}
    )
    workspace.add_goal(test_goal)
    print(f"✅ Test goal added: {test_goal.description}")
    print()
    
    # Step 4: Start the cognitive core (initializes queues)
    print("Step 4: Starting CognitiveCore (initializes async context)...")
    start_task = asyncio.create_task(core.start())
    await asyncio.sleep(0.1)  # Give it a moment to initialize
    print(f"✅ CognitiveCore started (running in background)")
    print()
    
    # Step 5: Run ONE cognitive cycle by waiting for cycle duration
    print("Step 5: Executing ONE cognitive cycle...")
    cycle_duration = 1.0 / config["cycle_rate_hz"]
    # Wait for cycle completion using multiplier
    await asyncio.sleep(cycle_duration * CYCLE_COMPLETION_MULTIPLIER)
    print(f"✅ Cognitive cycle completed")
    print()
    
    # Step 6: Query workspace state
    print("Step 6: Querying workspace state...")
    snapshot = core.query_state()
    print(f"✅ Workspace snapshot retrieved")
    print()
    
    # Step 7: Display workspace state
    print("=" * 70)
    print("WORKSPACE STATE AFTER ONE CYCLE")
    print("=" * 70)
    print()
    print(f"Goals ({len(snapshot.goals)}):")
    for i, goal in enumerate(snapshot.goals, 1):
        print(f"  {i}. [{goal.type.name}] {goal.description} (priority: {goal.priority})")
    print()
    
    print(f"Active Percepts ({len(snapshot.percepts)}):")
    for i, (percept_id, percept_data) in enumerate(list(snapshot.percepts.items())[:5], 1):  # Show first 5
        modality = getattr(percept_data, 'modality', 'unknown')
        print(f"  {i}. [{modality}] {percept_id[:8]}...")
    if len(snapshot.percepts) > 5:
        print(f"  ... and {len(snapshot.percepts) - 5} more")
    print()
    
    print(f"Emotional State:")
    for key, value in snapshot.emotions.items():
        print(f"  {key}: {value:.2f}")
    print()
    
    print(f"Retrieved Memories: {len(snapshot.memories)}")
    print()
    
    # Step 8: Display metrics
    print("=" * 70)
    print("COGNITIVE CORE METRICS")
    print("=" * 70)
    print()
    metrics = core.get_metrics()
    print(f"Total Cycles: {metrics['total_cycles']}")
    print(f"Average Cycle Time: {metrics['avg_cycle_time_ms']:.2f} ms")
    print(f"Target Cycle Time: {metrics['target_cycle_time_ms']:.0f} ms")
    print(f"Cycle Rate: {metrics['cycle_rate_hz']} Hz")
    print(f"Percepts Processed: {metrics['percepts_processed']}")
    print(f"Attention Selections: {metrics['attention_selections']}")
    print(f"Workspace Size: {metrics['workspace_size']}")
    print(f"Current Goals: {metrics['current_goals']}")
    print()
    print("Timing Enforcement:")
    print(f"  Slow Cycles (>100ms): {metrics['slow_cycles']} ({metrics['slow_cycle_percentage']:.1f}%)")
    print(f"  Critical Cycles (>200ms): {metrics['critical_cycles']} ({metrics['critical_cycle_percentage']:.1f}%)")
    print(f"  Slowest Cycle: {metrics['slowest_cycle_ms']:.2f} ms")
    print()
    
    # Display performance breakdown
    print("=" * 70)
    print("PERFORMANCE BREAKDOWN")
    print("=" * 70)
    print()
    
    breakdown = core.get_performance_breakdown()
    if breakdown:
        print("Per-Subsystem Timing (averages over last cycles):")
        print()
        for subsystem, stats in sorted(breakdown.items(), key=lambda x: -x[1]['avg_ms']):
            print(f"  {subsystem}:")
            print(f"    Average: {stats['avg_ms']:.2f}ms")
            print(f"    Min: {stats['min_ms']:.2f}ms")
            print(f"    Max: {stats['max_ms']:.2f}ms")
            print(f"    P95: {stats['p95_ms']:.2f}ms")
            print()
    else:
        print("  No performance data available yet")
    print()

    # Display subsystem health report
    print("=" * 70)
    print("SUBSYSTEM HEALTH REPORT")
    print("=" * 70)
    print()

    health_report = core.get_health_report()
    print(f"Overall Status: {health_report['overall_status'].upper()}")
    print(f"Total Subsystems Tracked: {health_report['total_subsystems']}")
    print(f"  Healthy:  {health_report['healthy_count']}")
    print(f"  Degraded: {health_report['degraded_count']}")
    print(f"  Failed:   {health_report['failed_count']}")
    print()

    if health_report['subsystems']:
        print("Per-Subsystem Health:")
        print()
        for name, info in sorted(health_report['subsystems'].items()):
            status = info['status'].upper()
            failures = info['consecutive_failures']
            rate = info['failure_rate']
            marker = "✅" if status == "HEALTHY" else ("⚠️" if status == "DEGRADED" else "❌")
            print(f"  {marker} {name}: {status}")
            if failures > 0 or rate > 0:
                print(f"      consecutive_failures={failures}, failure_rate={rate:.1%}")
            if info.get('last_error'):
                print(f"      last_error: {info['last_error'][:80]}")
        print()

    # Step 9: Stop the cognitive core
    print("Step 9: Stopping CognitiveCore...")
    await core.stop()
    
    # Cancel the start task
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass
    
    print(f"✅ CognitiveCore stopped cleanly")
    print()
    
    # Step 10: Verification
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    print()
    
    success = True
    checks = []
    
    # Check 1: At least one cycle executed
    if metrics['total_cycles'] >= 1:
        checks.append("✅ At least one cognitive cycle executed")
    else:
        checks.append("❌ No cognitive cycles executed")
        success = False
    
    # Check 2: Workspace has content
    if len(snapshot.goals) > 0:
        checks.append("✅ Workspace contains goals")
    else:
        checks.append("❌ Workspace has no goals")
        success = False
    
    # Check 3: Emotional state exists
    if snapshot.emotions:
        checks.append("✅ Emotional state computed")
    else:
        checks.append("❌ No emotional state")
        success = False
    
    # Check 4: Cycle time reasonable
    if 0 < metrics['avg_cycle_time_ms'] < 1000:
        checks.append(f"✅ Cycle time reasonable ({metrics['avg_cycle_time_ms']:.1f}ms)")
    else:
        checks.append(f"❌ Cycle time unreasonable ({metrics['avg_cycle_time_ms']:.1f}ms)")
        success = False
    
    for check in checks:
        print(check)
    print()
    
    if success:
        print("🎉 SUCCESS: Minimal cognitive core test passed!")
        print()
        return 0
    else:
        print("❌ FAILURE: Minimal cognitive core test failed!")
        print()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
