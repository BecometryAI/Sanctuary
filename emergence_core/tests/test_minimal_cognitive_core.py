"""
Unit test for minimal cognitive core execution.

This test verifies that the cognitive core can initialize,
run a single cycle, and produce valid output.
"""

import pytest
import asyncio
from pathlib import Path

from lyra.cognitive_core.core import CognitiveCore
from lyra.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType

# Configuration constants  
CYCLE_COMPLETION_MULTIPLIER = 1.5  # Wait 1.5 cycles to ensure at least one completes


@pytest.mark.asyncio
async def test_single_cycle_execution():
    """Test that cognitive core can execute a single cycle."""
    # Initialize workspace and core
    workspace = GlobalWorkspace()
    config = {
        "cycle_rate_hz": 10,
        "attention_budget": 100,
        "checkpointing": {"enabled": False},
        "input_llm": {"use_real_model": False},
        "output_llm": {"use_real_model": False},
    }
    core = CognitiveCore(workspace=workspace, config=config)
    
    # Add test goal
    test_goal = Goal(
        type=GoalType.RESPOND_TO_USER,
        description="Test goal",
        priority=0.5,
        metadata={"test": True}
    )
    workspace.add_goal(test_goal)
    
    # Start core
    start_task = asyncio.create_task(core.start())
    await asyncio.sleep(0.1)  # Let it initialize
    
    # Wait for cycles to complete
    cycle_duration = 1.0 / config["cycle_rate_hz"]
    # Use same multiplier as main script for consistency
    await asyncio.sleep(cycle_duration * CYCLE_COMPLETION_MULTIPLIER)
    
    # Query state
    snapshot = core.query_state()
    metrics = core.get_metrics()
    
    # Stop core
    await core.stop()
    start_task.cancel()
    try:
        await start_task
    except asyncio.CancelledError:
        pass
    
    # Assertions
    assert metrics['total_cycles'] >= 1, "At least one cycle should execute"
    assert len(snapshot.goals) > 0, "Workspace should have goals"
    assert snapshot.emotions is not None, "Emotional state should exist"
    assert 0 < metrics['avg_cycle_time_ms'] < 1000, "Cycle time should be reasonable"


@pytest.mark.asyncio
async def test_core_initialization():
    """Test that cognitive core initializes without errors."""
    workspace = GlobalWorkspace()
    config = {
        "checkpointing": {"enabled": False},
        "input_llm": {"use_real_model": False},
        "output_llm": {"use_real_model": False},
    }
    
    # Should not raise
    core = CognitiveCore(workspace=workspace, config=config)
    
    # Basic checks
    assert core.workspace is not None
    assert core.attention is not None
    assert core.perception is not None
    assert core.action is not None
    assert core.affect is not None
    assert core.meta_cognition is not None


@pytest.mark.asyncio
async def test_workspace_state_query():
    """Test that workspace state can be queried."""
    workspace = GlobalWorkspace()
    config = {
        "checkpointing": {"enabled": False},
        "input_llm": {"use_real_model": False},
        "output_llm": {"use_real_model": False},
    }
    core = CognitiveCore(workspace=workspace, config=config)
    
    # Query should work even before starting
    snapshot = core.query_state()
    
    assert snapshot is not None
    assert hasattr(snapshot, 'goals')
    assert hasattr(snapshot, 'percepts')
    assert hasattr(snapshot, 'emotions')
