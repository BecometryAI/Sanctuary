"""
Tests for CLI hardening features.

Covers:
  - argparse configuration
  - _format_error error categorisation
  - Shutdown timeout in LifecycleManager
  - SanctuaryAPI.start() no longer uses fire-and-forget
  - Signal-driven shutdown event
"""

import asyncio
import signal
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. argparse
# ---------------------------------------------------------------------------

class TestParseArgs:
    """Verify CLI argument parsing produces correct defaults and overrides."""

    def test_defaults(self):
        from mind.cli import parse_args
        args = parse_args([])
        assert args.verbose is False
        assert args.restore_latest is False
        assert args.auto_save is False
        assert args.auto_save_interval == 300.0
        assert args.cycle_rate == 10.0
        assert args.shutdown_timeout == 30.0

    def test_verbose_short(self):
        from mind.cli import parse_args
        args = parse_args(["-v"])
        assert args.verbose is True

    def test_verbose_long(self):
        from mind.cli import parse_args
        args = parse_args(["--verbose"])
        assert args.verbose is True

    def test_restore_latest(self):
        from mind.cli import parse_args
        args = parse_args(["--restore-latest"])
        assert args.restore_latest is True

    def test_auto_save_with_interval(self):
        from mind.cli import parse_args
        args = parse_args(["--auto-save", "--auto-save-interval", "120"])
        assert args.auto_save is True
        assert args.auto_save_interval == 120.0

    def test_cycle_rate(self):
        from mind.cli import parse_args
        args = parse_args(["--cycle-rate", "5.0"])
        assert args.cycle_rate == 5.0

    def test_shutdown_timeout(self):
        from mind.cli import parse_args
        args = parse_args(["--shutdown-timeout", "15"])
        assert args.shutdown_timeout == 15.0


# ---------------------------------------------------------------------------
# 2. _format_error
# ---------------------------------------------------------------------------

class TestFormatError:
    """Verify error categorisation produces the right prefix."""

    def test_runtime_error(self):
        from mind.cli import _format_error
        msg = _format_error(RuntimeError("boom"))
        assert msg.startswith("Runtime error:")
        assert "boom" in msg

    def test_connection_error(self):
        from mind.cli import _format_error
        msg = _format_error(ConnectionError("refused"))
        assert msg.startswith("Connection error:")

    def test_timeout_error(self):
        from mind.cli import _format_error
        msg = _format_error(TimeoutError("slow"))
        assert msg.startswith("Operation timed out:")

    def test_asyncio_timeout(self):
        from mind.cli import _format_error
        msg = _format_error(asyncio.TimeoutError())
        assert msg.startswith("Operation timed out:")

    def test_generic_exception(self):
        from mind.cli import _format_error
        msg = _format_error(ValueError("bad"))
        assert msg.startswith("Error:")

    def test_verbose_includes_traceback(self):
        from mind.cli import _format_error
        try:
            raise ValueError("trace-test")
        except ValueError as e:
            msg = _format_error(e, verbose=True)
        assert "Traceback" in msg or "trace-test" in msg

    def test_gpu_memory_error(self):
        from mind.cli import _format_error
        from mind.exceptions import GPUMemoryError
        msg = _format_error(GPUMemoryError("OOM"))
        assert msg.startswith("GPU memory exhausted:")

    def test_model_load_error(self):
        from mind.cli import _format_error
        from mind.exceptions import ModelLoadError
        msg = _format_error(ModelLoadError("missing weights"))
        assert msg.startswith("Model load failure:")

    def test_rate_limit_error(self):
        from mind.cli import _format_error
        from mind.exceptions import RateLimitError
        msg = _format_error(RateLimitError("429"))
        assert msg.startswith("Rate limited:")


# ---------------------------------------------------------------------------
# 3. LifecycleManager shutdown timeout
# ---------------------------------------------------------------------------

class TestLifecycleShutdownTimeout:
    """Verify that LifecycleManager.stop() respects the timeout parameter."""

    @pytest.mark.asyncio
    async def test_stop_returns_on_timeout(self):
        """If _shutdown_sequence hangs, stop() should not block forever."""
        from mind.cognitive_core.core.lifecycle import LifecycleManager

        # Build minimal fakes
        state = MagicMock()
        state.running = True
        state.active_task = None
        state.idle_task = None

        subsystems = MagicMock()
        subsystems.memory.memory_manager.disable_auto_gc = MagicMock()
        subsystems.checkpoint_manager = None

        timing = MagicMock()
        timing.metrics = {"cycle_times": [], "total_cycles": 0}

        lm = LifecycleManager(subsystems, state, timing, {})

        # Patch _shutdown_sequence to hang indefinitely
        async def _hang():
            await asyncio.sleep(9999)

        lm._shutdown_sequence = _hang

        # stop() with a very short timeout should return quickly
        await lm.stop(timeout=0.1)
        # If we got here without hanging, the timeout worked.
        assert state.running is False


# ---------------------------------------------------------------------------
# 4. SanctuaryAPI.start() race condition fix
# ---------------------------------------------------------------------------

class TestSanctuaryAPIStartAwait:
    """Verify that SanctuaryAPI.start() awaits core.start() directly."""

    @pytest.mark.asyncio
    async def test_start_awaits_core(self):
        """start() should call core.start() with await, not fire-and-forget."""
        from mind.client import SanctuaryAPI

        api = SanctuaryAPI.__new__(SanctuaryAPI)
        api._running = False
        api.core = AsyncMock()
        api.core.start = AsyncMock()
        api.conversation = MagicMock()

        await api.start()

        # core.start() should have been awaited exactly once
        api.core.start.assert_awaited_once()
        assert api._running is True


# ---------------------------------------------------------------------------
# 5. Signal-driven shutdown event
# ---------------------------------------------------------------------------

class TestSignalShutdownEvent:
    """Verify the shutdown_event pattern works with the REPL loop."""

    @pytest.mark.asyncio
    async def test_shutdown_event_breaks_repl(self):
        """When shutdown_event is set, the REPL while-loop should exit."""
        shutdown_event = asyncio.Event()

        iterations = 0

        async def fake_repl():
            nonlocal iterations
            while not shutdown_event.is_set():
                iterations += 1
                if iterations >= 3:
                    shutdown_event.set()
                await asyncio.sleep(0)

        await fake_repl()
        assert iterations == 3
        assert shutdown_event.is_set()


# ---------------------------------------------------------------------------
# 6. CognitiveCore._started flag
# ---------------------------------------------------------------------------

class TestCognitiveCoreSsartedFlag:
    """Verify the _started flag is set after core.start()."""

    @pytest.mark.asyncio
    async def test_started_flag_set(self):
        """CognitiveCore._started should be True after start()."""
        from mind.cognitive_core.core import CognitiveCore

        core = CognitiveCore.__new__(CognitiveCore)
        core._started = False
        core.lifecycle = AsyncMock()
        core.loop = AsyncMock()

        # Mock the loop task
        async def _noop():
            pass
        core.loop.run = _noop

        await core.start()
        assert core._started is True
