"""
Unit tests for the refactored cognitive core modules.

Tests cover efficiency, robustness, edge cases, and input validation.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from collections import deque

# Test TimingManager
def test_timing_manager_initialization():
    """Test TimingManager initializes correctly with valid config."""
    from emergence_core.sanctuary.cognitive_core.core.timing import TimingManager
    
    config = {
        "cycle_rate_hz": 10,
        "timing": {
            "warn_threshold_ms": 100,
            "critical_threshold_ms": 200
        },
        "log_interval_cycles": 100
    }
    
    manager = TimingManager(config)
    assert manager.cycle_duration == 0.1
    assert manager.warn_threshold_ms == 100
    assert manager.critical_threshold_ms == 200
    assert manager.metrics['total_cycles'] == 0


def test_timing_manager_invalid_cycle_rate():
    """Test TimingManager rejects invalid cycle rate."""
    from emergence_core.sanctuary.cognitive_core.core.timing import TimingManager
    
    config = {"cycle_rate_hz": 0}
    
    with pytest.raises(ValueError, match="cycle_rate_hz must be positive"):
        TimingManager(config)


def test_timing_manager_invalid_thresholds():
    """Test TimingManager rejects invalid threshold configuration."""
    from emergence_core.sanctuary.cognitive_core.core.timing import TimingManager
    
    config = {
        "cycle_rate_hz": 10,
        "timing": {
            "warn_threshold_ms": 200,
            "critical_threshold_ms": 100  # Less than warn
        }
    }
    
    with pytest.raises(ValueError, match="critical_threshold_ms.*must be greater"):
        TimingManager(config)


def test_timing_manager_check_cycle_timing():
    """Test cycle timing threshold checking."""
    from emergence_core.sanctuary.cognitive_core.core.timing import TimingManager
    
    config = {
        "cycle_rate_hz": 10,
        "timing": {
            "warn_threshold_ms": 100,
            "critical_threshold_ms": 200
        }
    }
    
    manager = TimingManager(config)
    
    # Normal cycle - no warnings
    manager.check_cycle_timing(0.05, 1)  # 50ms
    assert manager.metrics['slow_cycles'] == 0
    assert manager.metrics['critical_cycles'] == 0
    
    # Slow cycle - warning
    manager.check_cycle_timing(0.15, 2)  # 150ms
    assert manager.metrics['slow_cycles'] == 1
    assert manager.metrics['critical_cycles'] == 0
    
    # Critical cycle
    manager.check_cycle_timing(0.25, 3)  # 250ms
    assert manager.metrics['slow_cycles'] == 1
    assert manager.metrics['critical_cycles'] == 1


def test_timing_manager_percentile_edge_cases():
    """Test percentile calculation handles edge cases."""
    from emergence_core.sanctuary.cognitive_core.core.timing import TimingManager
    
    config = {"cycle_rate_hz": 10}
    manager = TimingManager(config)
    
    # Add timing data
    manager.metrics['subsystem_timings'] = {
        'test': deque([1.0], maxlen=100)  # Single element
    }
    
    # Should not crash with single element
    breakdown = manager.get_performance_breakdown()
    assert 'test' in breakdown
    assert breakdown['test']['p50_ms'] == 1.0
    assert breakdown['test']['p95_ms'] == 1.0
    assert breakdown['test']['p99_ms'] == 1.0


# Test StateManager
def test_state_manager_initialization():
    """Test StateManager initializes correctly."""
    from emergence_core.sanctuary.cognitive_core.core.state_manager import StateManager
    
    config = {"max_queue_size": 100}
    manager = StateManager(None, config)
    
    assert manager.workspace is not None
    assert manager.running is False
    assert manager.input_queue is None  # Not initialized yet
    assert manager.output_queue is None


def test_state_manager_invalid_queue_size():
    """Test StateManager rejects invalid queue size."""
    from emergence_core.sanctuary.cognitive_core.core.state_manager import StateManager
    
    config = {"max_queue_size": -1}
    
    with pytest.raises(ValueError, match="max_queue_size must be positive"):
        StateManager(None, config)


def test_state_manager_inject_input_before_init():
    """Test inject_input raises error before queue initialization."""
    from emergence_core.sanctuary.cognitive_core.core.state_manager import StateManager
    
    config = {"max_queue_size": 100}
    manager = StateManager(None, config)
    
    with pytest.raises(RuntimeError, match="queues must be initialized"):
        manager.inject_input("test", "text")


@pytest.mark.asyncio
async def test_state_manager_queue_initialization():
    """Test queue initialization in async context."""
    from emergence_core.sanctuary.cognitive_core.core.state_manager import StateManager
    
    config = {"max_queue_size": 10}
    manager = StateManager(None, config)
    
    manager.initialize_queues()
    
    assert manager.input_queue is not None
    assert manager.output_queue is not None
    assert manager.input_queue.maxsize == 10


@pytest.mark.asyncio
async def test_state_manager_gather_percepts_empty():
    """Test gather_percepts handles empty queue gracefully."""
    from emergence_core.sanctuary.cognitive_core.core.state_manager import StateManager
    
    config = {"max_queue_size": 100}
    manager = StateManager(None, config)
    manager.initialize_queues()
    
    mock_perception = Mock()
    percepts = await manager.gather_percepts(mock_perception)
    
    assert percepts == []
    mock_perception.encode.assert_not_called()


# Test ActionExecutor
@pytest.mark.asyncio
async def test_action_executor_execute_invalid_response():
    """Test SPEAK action handles invalid response gracefully."""
    from emergence_core.sanctuary.cognitive_core.core.action_executor import ActionExecutor
    from emergence_core.sanctuary.cognitive_core.core.state_manager import StateManager
    
    # Setup mocks
    config = {"max_queue_size": 100}
    state = StateManager(None, config)
    state.initialize_queues()
    
    subsystems = Mock()
    subsystems.language_output = Mock()
    subsystems.language_output.generate = AsyncMock(return_value=None)  # Invalid response
    
    state.workspace = Mock()
    state.workspace.broadcast = Mock(return_value=Mock(emotions={}))
    
    executor = ActionExecutor(subsystems, state)
    
    # Create mock action
    action = Mock()
    action.metadata = {"responding_to": "test"}
    
    # Should not raise, should log warning and use fallback
    await executor.execute_speak(action)
    
    # Check fallback was used
    output = await state.output_queue.get()
    assert output['text'] == "..."


# Test CycleExecutor error handling
@pytest.mark.asyncio
async def test_cycle_executor_handles_perception_failure():
    """Test cycle executor continues despite perception failure."""
    from emergence_core.sanctuary.cognitive_core.core.cycle_executor import CycleExecutor
    from emergence_core.sanctuary.cognitive_core.core.state_manager import StateManager
    
    config = {"max_queue_size": 100}
    state = StateManager(None, config)
    state.initialize_queues()
    
    # Mock subsystems
    subsystems = Mock()
    subsystems.perception = Mock()
    subsystems.attention = Mock()
    subsystems.attention.select_for_broadcast = Mock(return_value=[])
    subsystems.affect = Mock()
    subsystems.affect.compute_update = Mock(return_value={})
    subsystems.action = Mock()
    subsystems.action.decide = Mock(return_value=[])
    subsystems.meta_cognition = Mock()
    subsystems.meta_cognition.observe = Mock(return_value=[])
    subsystems.autonomous = Mock()
    subsystems.autonomous.check_for_autonomous_triggers = Mock(return_value=None)
    subsystems.memory = Mock()
    subsystems.memory.consolidate = AsyncMock()
    
    # Mock action executor
    action_executor = Mock()
    action_executor.execute = AsyncMock()
    action_executor.execute_tool = AsyncMock()
    action_executor.extract_outcome = Mock(return_value={})
    
    # Make perception fail
    state.gather_percepts = AsyncMock(side_effect=Exception("Perception failed"))
    
    executor = CycleExecutor(subsystems, state, action_executor)
    
    # Execute cycle - should not raise despite perception failure
    timings = await executor.execute_cycle()
    
    # Check that we got timing data and cycle continued
    assert 'perception' in timings
    assert 'attention' in timings
    assert timings['perception'] == 0.0  # Failed step


def test_config_validation_comprehensive():
    """Test comprehensive config validation across modules."""
    from emergence_core.sanctuary.cognitive_core.core.timing import TimingManager
    from emergence_core.sanctuary.cognitive_core.core.state_manager import StateManager
    
    # Test various invalid configs
    invalid_configs = [
        ({"cycle_rate_hz": -1}, TimingManager, "cycle_rate_hz must be positive"),
        ({"cycle_rate_hz": 0}, TimingManager, "cycle_rate_hz must be positive"),
        ({"max_queue_size": 0}, StateManager, "max_queue_size must be positive"),
        ({"log_interval_cycles": -1}, TimingManager, "log_interval_cycles must be positive"),
    ]
    
    for config, cls, expected_msg in invalid_configs:
        # Fill in required defaults
        full_config = {"cycle_rate_hz": 10, "max_queue_size": 100, "log_interval_cycles": 100}
        full_config.update(config)
        
        with pytest.raises(ValueError, match=expected_msg):
            if cls == StateManager:
                cls(None, full_config)
            else:
                cls(full_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
