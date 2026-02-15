"""
Unit tests for the refactored cognitive core modules.

Tests cover efficiency, robustness, edge cases, and input validation
for TimingManager, StateManager, ActionExecutor, and CycleExecutor.
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from collections import deque


# ---------------------------------------------------------------------------
# TimingManager
# ---------------------------------------------------------------------------

def _make_timing_config(**overrides):
    """Helper to build a valid TimingManager config with overrides."""
    config = {
        "cycle_rate_hz": 10,
        "timing": {
            "warn_threshold_ms": 100,
            "critical_threshold_ms": 200,
        },
        "log_interval_cycles": 100,
    }
    config.update(overrides)
    return config


def _make_timing_manager(**overrides):
    from mind.cognitive_core.core.timing import TimingManager
    return TimingManager(_make_timing_config(**overrides))


class TestTimingManagerInit:
    """TimingManager initialization and validation."""

    def test_valid_config(self):
        tm = _make_timing_manager()
        assert tm.cycle_duration == pytest.approx(0.1)
        assert tm.warn_threshold_ms == 100
        assert tm.critical_threshold_ms == 200
        assert tm.metrics["total_cycles"] == 0

    def test_rejects_zero_cycle_rate(self):
        from mind.cognitive_core.core.timing import TimingManager
        with pytest.raises(ValueError, match="cycle_rate_hz must be positive"):
            TimingManager({"cycle_rate_hz": 0})

    def test_rejects_negative_cycle_rate(self):
        from mind.cognitive_core.core.timing import TimingManager
        with pytest.raises(ValueError, match="cycle_rate_hz must be positive"):
            TimingManager({"cycle_rate_hz": -5})

    def test_rejects_critical_less_than_warn(self):
        from mind.cognitive_core.core.timing import TimingManager
        with pytest.raises(ValueError, match="critical_threshold_ms.*must be greater"):
            TimingManager({
                "cycle_rate_hz": 10,
                "timing": {"warn_threshold_ms": 200, "critical_threshold_ms": 100},
            })

    def test_rejects_negative_log_interval(self):
        from mind.cognitive_core.core.timing import TimingManager
        with pytest.raises(ValueError, match="log_interval_cycles must be positive"):
            TimingManager({
                "cycle_rate_hz": 10,
                "log_interval_cycles": -1,
            })


class TestTimingManagerBehavior:
    """TimingManager runtime behavior."""

    def test_check_cycle_timing_normal(self):
        tm = _make_timing_manager()
        tm.check_cycle_timing(0.05, 1)  # 50ms — under warn
        assert tm.metrics["slow_cycles"] == 0
        assert tm.metrics["critical_cycles"] == 0

    def test_check_cycle_timing_slow(self):
        tm = _make_timing_manager()
        tm.check_cycle_timing(0.15, 1)  # 150ms — above warn, below critical
        assert tm.metrics["slow_cycles"] == 1
        assert tm.metrics["critical_cycles"] == 0

    def test_check_cycle_timing_critical(self):
        tm = _make_timing_manager()
        tm.check_cycle_timing(0.25, 1)  # 250ms — above critical
        assert tm.metrics["slow_cycles"] == 0  # critical doesn't count as slow
        assert tm.metrics["critical_cycles"] == 1

    def test_calculate_sleep_time_normal(self):
        tm = _make_timing_manager()  # 10Hz → 100ms period
        sleep = tm.calculate_sleep_time(0.05)  # took 50ms
        assert sleep == pytest.approx(0.05, abs=0.01)

    def test_calculate_sleep_time_overrun(self):
        tm = _make_timing_manager()
        sleep = tm.calculate_sleep_time(0.15)  # took 150ms, period is 100ms
        assert sleep == 0.0

    def test_update_metrics_tracks_cycle_count(self):
        tm = _make_timing_manager()
        tm.update_metrics(0.05)
        tm.update_metrics(0.06)
        assert tm.metrics["total_cycles"] == 2

    def test_update_metrics_tracks_slowest(self):
        tm = _make_timing_manager()
        tm.update_metrics(0.05)
        tm.update_metrics(0.12)
        tm.update_metrics(0.03)
        assert tm.metrics["slowest_cycle_ms"] == pytest.approx(120.0, abs=1.0)

    def test_get_metrics_summary(self):
        tm = _make_timing_manager()
        tm.update_metrics(0.05)
        tm.check_cycle_timing(0.05, 1)
        summary = tm.get_metrics_summary()
        assert "total_cycles" in summary
        assert summary["total_cycles"] == 1

    def test_performance_breakdown_empty(self):
        tm = _make_timing_manager()
        breakdown = tm.get_performance_breakdown()
        assert isinstance(breakdown, dict)

    def test_performance_breakdown_single_element(self):
        tm = _make_timing_manager()
        tm.metrics["subsystem_timings"] = {
            "test_sub": deque([1.0], maxlen=100),
        }
        breakdown = tm.get_performance_breakdown()
        assert "test_sub" in breakdown
        assert breakdown["test_sub"]["p50_ms"] == 1.0

    def test_performance_breakdown_multiple_elements(self):
        tm = _make_timing_manager()
        tm.metrics["subsystem_timings"] = {
            "attention": deque([10.0, 20.0, 30.0, 40.0, 50.0], maxlen=100),
        }
        breakdown = tm.get_performance_breakdown()
        stats = breakdown["attention"]
        assert stats["avg_ms"] == pytest.approx(30.0)
        assert stats["min_ms"] == pytest.approx(10.0)
        assert stats["max_ms"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------

def _make_state_manager(queue_size=100, workspace=None):
    from mind.cognitive_core.core.state_manager import StateManager
    return StateManager(workspace, {"max_queue_size": queue_size})


class TestStateManagerInit:
    """StateManager initialization and validation."""

    def test_valid_config(self):
        sm = _make_state_manager()
        assert sm.running is False
        assert sm.input_queue is None
        assert sm.output_queue is None

    def test_rejects_zero_queue_size(self):
        from mind.cognitive_core.core.state_manager import StateManager
        with pytest.raises(ValueError, match="max_queue_size must be positive"):
            StateManager(None, {"max_queue_size": 0})

    def test_rejects_negative_queue_size(self):
        from mind.cognitive_core.core.state_manager import StateManager
        with pytest.raises(ValueError, match="max_queue_size must be positive"):
            StateManager(None, {"max_queue_size": -1})

    def test_creates_default_workspace_when_none(self):
        sm = _make_state_manager(workspace=None)
        assert sm.workspace is not None


class TestStateManagerQueues:
    """StateManager queue lifecycle."""

    def test_initialize_queues(self):
        sm = _make_state_manager(queue_size=10)
        sm.initialize_queues()
        assert sm.input_queue is not None
        assert sm.output_queue is not None
        assert sm.input_queue.maxsize == 10

    def test_inject_input_before_init_raises(self):
        sm = _make_state_manager()
        with pytest.raises(RuntimeError, match="queues must be initialized"):
            sm.inject_input("test", "text")

    def test_inject_input_after_init(self):
        sm = _make_state_manager()
        sm.initialize_queues()
        sm.inject_input("hello", "text")
        assert not sm.input_queue.empty()

    @pytest.mark.asyncio
    async def test_gather_percepts_empty_queue(self):
        sm = _make_state_manager()
        sm.initialize_queues()
        mock_perception = Mock()
        percepts = await sm.gather_percepts(mock_perception)
        assert percepts == []
        mock_perception.encode.assert_not_called()

    @pytest.mark.asyncio
    async def test_gather_percepts_with_input(self):
        sm = _make_state_manager()
        sm.initialize_queues()
        sm.inject_input("test input", "text")

        mock_percept = Mock()
        mock_perception = Mock()
        mock_perception.encode = AsyncMock(return_value=mock_percept)

        percepts = await sm.gather_percepts(mock_perception)
        assert len(percepts) == 1
        assert percepts[0] is mock_percept

    @pytest.mark.asyncio
    async def test_gather_percepts_includes_pending_tool_percepts(self):
        sm = _make_state_manager()
        sm.initialize_queues()

        tool_percept = Mock()
        sm.add_pending_tool_percept(tool_percept)

        mock_perception = Mock()
        percepts = await sm.gather_percepts(mock_perception)
        assert tool_percept in percepts

    @pytest.mark.asyncio
    async def test_get_response_timeout_returns_none(self):
        sm = _make_state_manager()
        sm.initialize_queues()
        result = await sm.get_response(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_response_before_init_raises(self):
        sm = _make_state_manager()
        with pytest.raises(RuntimeError):
            await sm.get_response()

    @pytest.mark.asyncio
    async def test_queue_output_and_get_response(self):
        sm = _make_state_manager()
        sm.initialize_queues()
        await sm.queue_output({"text": "hello"})
        result = await sm.get_response(timeout=1.0)
        assert result == {"text": "hello"}

    def test_query_state_returns_snapshot(self):
        sm = _make_state_manager()
        sm.initialize_queues()
        snapshot = sm.query_state()
        assert snapshot is not None


# ---------------------------------------------------------------------------
# ActionExecutor
# ---------------------------------------------------------------------------

class TestActionExecutor:
    """ActionExecutor routing and error handling."""

    def _make_executor(self):
        from mind.cognitive_core.core.action_executor import ActionExecutor
        from mind.cognitive_core.core.state_manager import StateManager

        config = {"max_queue_size": 100}
        state = StateManager(None, config)
        state.initialize_queues()

        subsystems = Mock()
        subsystems.language_output = Mock()
        subsystems.language_output.generate = AsyncMock(return_value="test response")
        subsystems.temporal_grounding = None

        state.workspace = Mock()
        state.workspace.broadcast = Mock(return_value=Mock(emotions={}))

        return ActionExecutor(subsystems, state), state, subsystems

    @pytest.mark.asyncio
    async def test_speak_with_valid_response(self):
        executor, state, _ = self._make_executor()
        action = Mock()
        action.metadata = {"responding_to": "test"}

        await executor.execute_speak(action)
        output = await state.output_queue.get()
        assert output["text"] == "test response"

    @pytest.mark.asyncio
    async def test_speak_with_none_response_falls_back(self):
        executor, state, subsystems = self._make_executor()
        subsystems.language_output.generate = AsyncMock(return_value=None)

        action = Mock()
        action.metadata = {"responding_to": "test"}

        await executor.execute_speak(action)
        output = await state.output_queue.get()
        assert output["text"] == "..."

    @pytest.mark.asyncio
    async def test_speak_with_non_string_response_falls_back(self):
        executor, state, subsystems = self._make_executor()
        subsystems.language_output.generate = AsyncMock(return_value=12345)

        action = Mock()
        action.metadata = {"responding_to": "test"}

        await executor.execute_speak(action)
        output = await state.output_queue.get()
        assert output["text"] == "..."

    @pytest.mark.asyncio
    async def test_execute_tool_missing_tool_name(self):
        executor, state, _ = self._make_executor()
        action = Mock()
        action.metadata = {}

        result = await executor.execute_tool(action)
        assert result is None

    def test_extract_outcome_static(self):
        from mind.cognitive_core.core.action_executor import ActionExecutor
        action = Mock()
        action.type = "SPEAK"
        action.metadata = {"responding_to": "test"}
        outcome = ActionExecutor.extract_outcome(action)
        assert isinstance(outcome, dict)


# ---------------------------------------------------------------------------
# CycleExecutor
# ---------------------------------------------------------------------------

class TestCycleExecutor:
    """CycleExecutor error isolation and step execution."""

    def _make_cycle_executor(self):
        from mind.cognitive_core.core.cycle_executor import CycleExecutor
        from mind.cognitive_core.core.state_manager import StateManager

        config = {"max_queue_size": 100}
        state = StateManager(None, config)
        state.initialize_queues()

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
        subsystems.iwmt_core = None
        subsystems.temporal_grounding = None

        action_executor = Mock()
        action_executor.execute = AsyncMock()
        action_executor.execute_tool = AsyncMock(return_value=None)
        action_executor.extract_outcome = Mock(return_value={})

        return CycleExecutor(subsystems, state, action_executor), state, subsystems

    @pytest.mark.asyncio
    async def test_handles_perception_failure(self):
        executor, state, _ = self._make_cycle_executor()
        state.gather_percepts = AsyncMock(side_effect=Exception("Perception failed"))

        timings = await executor.execute_cycle()
        assert "perception" in timings
        assert "attention" in timings

    @pytest.mark.asyncio
    async def test_returns_timing_dict(self):
        executor, state, _ = self._make_cycle_executor()
        state.gather_percepts = AsyncMock(return_value=[])

        timings = await executor.execute_cycle()
        assert isinstance(timings, dict)
        assert "perception" in timings

    @pytest.mark.asyncio
    async def test_empty_percepts_still_runs_attention(self):
        executor, state, subsystems = self._make_cycle_executor()
        state.gather_percepts = AsyncMock(return_value=[])

        await executor.execute_cycle()
        # Attention should still be called even with no percepts
        subsystems.attention.select_for_broadcast.assert_called()


# ---------------------------------------------------------------------------
# Config validation across modules
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """Cross-module config validation."""

    @pytest.mark.parametrize("config,cls_name,msg", [
        ({"cycle_rate_hz": -1}, "TimingManager", "cycle_rate_hz must be positive"),
        ({"cycle_rate_hz": 0}, "TimingManager", "cycle_rate_hz must be positive"),
        ({"max_queue_size": 0}, "StateManager", "max_queue_size must be positive"),
    ])
    def test_invalid_configs_raise(self, config, cls_name, msg):
        from mind.cognitive_core.core.timing import TimingManager
        from mind.cognitive_core.core.state_manager import StateManager

        cls = {"TimingManager": TimingManager, "StateManager": StateManager}[cls_name]

        full_config = {"cycle_rate_hz": 10, "max_queue_size": 100, "log_interval_cycles": 100}
        full_config.update(config)

        with pytest.raises(ValueError, match=msg):
            if cls == StateManager:
                cls(None, full_config)
            else:
                cls(full_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
