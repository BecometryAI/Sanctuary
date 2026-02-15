"""
Unit tests for cognitive cycle timing enforcement.

Verifies that the 10 Hz timing target is actively monitored
and warnings are generated when cycles exceed thresholds.
"""

import pytest
import asyncio
import logging

from mind.cognitive_core.core import CognitiveCore
from mind.cognitive_core.core.timing import TimingManager
from mind.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType


@pytest.mark.asyncio
async def test_slow_cycle_warning():
    """Test that slow cycles trigger warnings."""
    config = {
        "cycle_rate_hz": 10,
        "log_interval_cycles": 100,
        "timing": {
            "warn_threshold_ms": 100,
            "critical_threshold_ms": 200,
            "track_slow_cycles": True,
        },
    }

    timing = TimingManager(config)

    # Simulate a 150ms cycle (above warn threshold, below critical)
    timing.check_cycle_timing(0.15, 1)  # 150ms
    timing.update_metrics(0.15)

    assert timing.metrics['slow_cycles'] == 1, "Slow cycle should be detected"
    assert timing.metrics['critical_cycles'] == 0, "Should not be critical"
    assert timing.metrics['slowest_cycle_ms'] >= 150, "Slowest cycle should be recorded"


@pytest.mark.asyncio
async def test_critical_cycle_warning():
    """Test that critical slow cycles trigger critical warnings."""
    config = {
        "cycle_rate_hz": 10,
        "log_interval_cycles": 100,
        "timing": {
            "warn_threshold_ms": 100,
            "critical_threshold_ms": 200,
            "track_slow_cycles": True,
        },
    }

    timing = TimingManager(config)

    # Simulate a 250ms cycle (above critical threshold)
    timing.check_cycle_timing(0.25, 1)  # 250ms
    timing.update_metrics(0.25)

    assert timing.metrics['critical_cycles'] == 1, "Critical cycle should be detected"
    assert timing.metrics['slowest_cycle_ms'] >= 250, "Slowest cycle should be recorded"


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
