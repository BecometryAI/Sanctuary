"""Async subsystem processor — running subsystems in parallel rather than sequentially.

Instead of processing subsystems one-by-one in the cognitive cycle, this module
enables parallel execution of independent subsystems. This reduces cycle
latency when multiple subsystems need to run.

Key design: subsystems declare their dependencies, and the processor runs
independent subsystems concurrently while respecting dependency ordering.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


@dataclass
class SubsystemTask:
    """A subsystem task that can be run as part of the cognitive cycle."""

    name: str
    fn: Callable[..., Coroutine]  # async function to run
    depends_on: list[str] = field(default_factory=list)  # Names of dependencies
    timeout_ms: float = 1000.0  # Max time for this subsystem
    critical: bool = False  # If True, failure stops the cycle


@dataclass
class SubsystemResult:
    """Result of running a subsystem."""

    name: str
    success: bool = True
    duration_ms: float = 0.0
    result: Any = None
    error: Optional[str] = None


@dataclass
class AsyncProcessorConfig:
    """Configuration for async subsystem processing."""

    max_concurrency: int = 10
    default_timeout_ms: float = 1000.0
    fail_fast: bool = False  # Stop all tasks on first failure


class AsyncSubsystemProcessor:
    """Runs subsystems in parallel with dependency ordering.

    Usage::

        processor = AsyncSubsystemProcessor()

        # Register subsystems
        processor.register("attention", attention_fn)
        processor.register("affect", affect_fn)
        processor.register("goals", goals_fn, depends_on=["attention"])

        # Run all (attention and affect run in parallel; goals waits for attention)
        results = await processor.run_all()
    """

    def __init__(self, config: Optional[AsyncProcessorConfig] = None):
        self.config = config or AsyncProcessorConfig()
        self._tasks: dict[str, SubsystemTask] = {}
        self._execution_history: list[dict] = []

    def register(
        self,
        name: str,
        fn: Callable[..., Coroutine],
        depends_on: list[str] | None = None,
        timeout_ms: float = 0,
        critical: bool = False,
    ) -> None:
        """Register a subsystem for parallel execution."""
        self._tasks[name] = SubsystemTask(
            name=name,
            fn=fn,
            depends_on=depends_on or [],
            timeout_ms=timeout_ms or self.config.default_timeout_ms,
            critical=critical,
        )

    def unregister(self, name: str) -> bool:
        """Remove a registered subsystem."""
        if name in self._tasks:
            del self._tasks[name]
            return True
        return False

    async def run_all(self, **kwargs) -> list[SubsystemResult]:
        """Run all registered subsystems respecting dependency order.

        Independent subsystems run concurrently; dependent ones wait.
        """
        if not self._tasks:
            return []

        results: dict[str, SubsystemResult] = {}
        completed: set[str] = set()
        execution_order: list[str] = []

        # Build execution order (topological sort)
        order = self._topological_sort()

        # Group into levels (tasks at same level can run concurrently)
        levels = self._group_into_levels(order)

        start_time = time.perf_counter()

        for level in levels:
            # Run this level's tasks concurrently
            level_results = await asyncio.gather(
                *[
                    self._run_task(self._tasks[name], kwargs)
                    for name in level
                    if name in self._tasks
                ],
                return_exceptions=True,
            )

            for name, result in zip(level, level_results):
                if isinstance(result, Exception):
                    sr = SubsystemResult(
                        name=name, success=False, error=str(result),
                    )
                else:
                    sr = result
                results[name] = sr
                completed.add(name)
                execution_order.append(name)

                task = self._tasks.get(name)
                if not sr.success and task is not None and task.critical:
                    if self.config.fail_fast:
                        break

        total_ms = (time.perf_counter() - start_time) * 1000
        self._execution_history.append({
            "order": execution_order,
            "total_ms": total_ms,
            "success_count": sum(1 for r in results.values() if r.success),
            "failure_count": sum(1 for r in results.values() if not r.success),
        })

        return list(results.values())

    async def run_one(self, name: str, **kwargs) -> Optional[SubsystemResult]:
        """Run a single subsystem by name."""
        if name not in self._tasks:
            return None
        return await self._run_task(self._tasks[name], kwargs)

    def get_execution_order(self) -> list[list[str]]:
        """Get the planned execution order (grouped by parallelism level)."""
        order = self._topological_sort()
        return self._group_into_levels(order)

    def get_stats(self) -> dict:
        """Get processor statistics."""
        history = self._execution_history
        return {
            "registered_subsystems": len(self._tasks),
            "total_executions": len(history),
            "avg_total_ms": (
                sum(h["total_ms"] for h in history) / len(history)
                if history else 0.0
            ),
        }

    # -- Internal --

    async def _run_task(
        self, task: SubsystemTask, kwargs: dict
    ) -> SubsystemResult:
        """Run a single subsystem task with timeout."""
        start = time.perf_counter()
        try:
            timeout = task.timeout_ms / 1000.0
            result = await asyncio.wait_for(task.fn(**kwargs), timeout=timeout)
            elapsed = (time.perf_counter() - start) * 1000
            return SubsystemResult(
                name=task.name, success=True, duration_ms=elapsed, result=result,
            )
        except asyncio.TimeoutError:
            elapsed = (time.perf_counter() - start) * 1000
            return SubsystemResult(
                name=task.name, success=False, duration_ms=elapsed,
                error=f"Timeout after {task.timeout_ms}ms",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return SubsystemResult(
                name=task.name, success=False, duration_ms=elapsed,
                error=str(e),
            )

    def _topological_sort(self) -> list[str]:
        """Topological sort of tasks by dependencies."""
        visited: set[str] = set()
        order: list[str] = []

        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            task = self._tasks.get(name)
            if task:
                for dep in task.depends_on:
                    if dep in self._tasks:
                        visit(dep)
            order.append(name)

        for name in self._tasks:
            visit(name)

        return order

    def _group_into_levels(self, order: list[str]) -> list[list[str]]:
        """Group topologically sorted tasks into parallelism levels."""
        levels: list[list[str]] = []
        placed: set[str] = set()

        remaining = list(order)
        while remaining:
            level = []
            for name in remaining:
                task = self._tasks.get(name)
                if task is None:
                    continue
                deps = set(task.depends_on) & set(self._tasks.keys())
                if deps.issubset(placed):
                    level.append(name)
            if not level:
                # Circular dependency or missing — force remaining
                level = remaining[:]
            for name in level:
                remaining.remove(name)
                placed.add(name)
            levels.append(level)

        return levels
