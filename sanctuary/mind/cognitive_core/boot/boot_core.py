"""
Boot-mode CognitiveCore for Phase 1 integration testing.

Simplified CognitiveCore using BootCoordinator (mock/stub subsystems).
Allows cognitive loop to instantiate and cycle without heavy ML deps.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any

from ..workspace import GlobalWorkspace, WorkspaceSnapshot
from .boot_coordinator import BootCoordinator
from ..core.state_manager import StateManager
from ..core.timing import TimingManager
from ..core.cycle_executor import CycleExecutor
from ..core.cognitive_loop import CognitiveLoop
from ..core.action_executor import ActionExecutor

logger = logging.getLogger(__name__)

BOOT_CONFIG = {
    "cycle_rate_hz": 10,
    "attention_budget": 100,
    "max_queue_size": 100,
    "log_interval_cycles": 10,
    "timing": {
        "warn_threshold_ms": 200,
        "critical_threshold_ms": 500,
        "track_slow_cycles": True,
    },
    "checkpointing": {"enabled": False},
    "perception": {"mock_mode": True, "mock_embedding_dim": 384},
    "affect": {},
    "action": {},
}


class BootCognitiveCore:
    """
    Minimal CognitiveCore for boot testing.
    Same public API as CognitiveCore but with BootCoordinator.
    """

    def __init__(self, workspace=None, config=None):
        self.config = {**BOOT_CONFIG, **(config or {})}

        self.state = StateManager(workspace, self.config)
        self.state.initialize_queues()  # Initialize immediately for boot

        self.subsystems = BootCoordinator(self.state.workspace, self.config)
        self.timing = TimingManager(self.config)
        self.action_executor = ActionExecutor(self.subsystems, self.state)
        self.cycle_executor = CycleExecutor(
            self.subsystems, self.state, self.action_executor
        )
        self.loop = CognitiveLoop(
            self.subsystems, self.state, self.timing, self.cycle_executor
        )
        self._loop_task: Optional[asyncio.Task] = None

        logger.info(f"\U0001f680 BootCognitiveCore initialized: {self.config['cycle_rate_hz']}Hz")

    async def start(self):
        self.state.running = True
        self._loop_task = asyncio.create_task(self.loop.run())
        logger.info("\U0001f680 Boot cognitive loop started")

    async def stop(self):
        self.state.running = False
        if self.state.idle_task:
            self.state.idle_task.cancel()
        if self.state.active_task:
            self.state.active_task.cancel()
        if self._loop_task:
            try:
                await asyncio.wait_for(self._loop_task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                if self._loop_task and not self._loop_task.done():
                    self._loop_task.cancel()
            self._loop_task = None
        logger.info("\U0001f680 Boot cognitive loop stopped")

    def inject_input(self, raw_input, modality="text"):
        self.state.inject_input(raw_input, modality)

    def query_state(self):
        return self.state.query_state()

    async def get_response(self, timeout=5.0):
        return await self.state.get_response(timeout)

    def get_metrics(self):
        metrics = self.timing.get_metrics_summary()
        metrics["boot_mode"] = True
        metrics["workspace_size"] = len(self.state.workspace.active_percepts)
        metrics["current_goals"] = len(self.state.workspace.current_goals)
        return metrics

    @property
    def workspace(self):
        return self.state.workspace

    @property
    def running(self):
        return self.state.running
