"""
Unit tests for cognitive cycle timing enforcement.

Verifies that the 10 Hz timing target is actively monitored
and warnings are generated when cycles exceed thresholds.
"""

import pytest
import asyncio
import logging
from unittest.mock import patch

from lyra.cognitive_core.core import CognitiveCore
from lyra.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType


@pytest.mark.asyncio
async def test_slow_cycle_warning():
    """Test that slow cycles trigger warnings."""
    workspace = GlobalWorkspace()
    config = {
        "cycle_rate_hz": 10,
        "log_interval_cycles": 1,
        "timing": {
            "warn_threshold_ms": 100,
            "critical_threshold_ms": 200,
            "track_slow_cycles": True,
        },
        "checkpointing": {"enabled": False},
        "input_llm": {"use_real_model": False},
        "output_llm": {"use_real_model": False},
    }
    
    core = CognitiveCore(workspace=workspace, config=config)
    
    # Add test goal
    test_goal = Goal(
        type=GoalType.RESPOND,
        description="Test goal",
        priority=5,
        context={"test": True}
    )
    workspace.add_goal(test_goal)
    
    # Patch sleep to simulate slow cycle
    with patch('asyncio.sleep', return_value=None):
        # Patch time.time to simulate 150ms cycle
        with patch('time.time', side_effect=[0.0, 0.15]):  # 150ms cycle
            # Start core
            start_task = asyncio.create_task(core.start())
            await asyncio.sleep(0.1)
            
            # Wait for one cycle
            await asyncio.sleep(0.2)
            
            # Stop core
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                # Cancellation is expected when stopping the core; ignore this error.
                pass
    
    # Check metrics
    metrics = core.get_metrics()
    assert metrics['slow_cycles'] > 0, "Slow cycle should be detected"
    assert metrics['slowest_cycle_ms'] >= 150, "Slowest cycle should be recorded"


@pytest.mark.asyncio
async def test_critical_cycle_warning():
    """Test that critical slow cycles trigger critical warnings."""
    workspace = GlobalWorkspace()
    config = {
        "cycle_rate_hz": 10,
        "log_interval_cycles": 1,
        "timing": {
            "warn_threshold_ms": 100,
            "critical_threshold_ms": 200,
            "track_slow_cycles": True,
        },
        "checkpointing": {"enabled": False},
        "input_llm": {"use_real_model": False},
        "output_llm": {"use_real_model": False},
    }
    
    core = CognitiveCore(workspace=workspace, config=config)
    
    # Patch to simulate 250ms cycle (critical)
    with patch('asyncio.sleep', return_value=None):
        with patch('time.time', side_effect=[0.0, 0.25]):  # 250ms cycle
            start_task = asyncio.create_task(core.start())
            await asyncio.sleep(0.1)
            await asyncio.sleep(0.2)
            
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                # Cancellation is expected when stopping the core; ignore this error.
                pass
    
    metrics = core.get_metrics()
    assert metrics['critical_cycles'] > 0, "Critical cycle should be detected"


@pytest.mark.asyncio
async def test_metrics_include_timing_stats():
    """Test that get_metrics() includes timing enforcement stats."""
    workspace = GlobalWorkspace()
    config = {
        "checkpointing": {"enabled": False},
        "input_llm": {"use_real_model": False},
        "output_llm": {"use_real_model": False},
    }
    core = CognitiveCore(workspace=workspace, config=config)
    
    metrics = core.get_metrics()
    
    # Verify new timing metrics exist
    assert 'slow_cycles' in metrics
    assert 'slow_cycle_percentage' in metrics
    assert 'critical_cycles' in metrics
    assert 'critical_cycle_percentage' in metrics
    assert 'slowest_cycle_ms' in metrics
