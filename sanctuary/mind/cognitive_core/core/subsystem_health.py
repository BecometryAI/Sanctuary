"""
Subsystem health tracking and fault isolation.

Implements a supervisor pattern for cognitive subsystems. Each subsystem's
health is tracked independently — if one subsystem enters a failure loop,
it is temporarily disabled (circuit breaker opens) while the rest of the
cognitive loop continues operating normally.

Health states:
    HEALTHY     → Normal operation, no recent failures
    DEGRADED    → Experiencing intermittent failures, still attempting
    FAILED      → Circuit breaker open, subsystem skipped until recovery timeout
    RECOVERING  → Testing whether a failed subsystem has recovered
"""

from __future__ import annotations

import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class SubsystemStatus(Enum):
    """Health status of a cognitive subsystem."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class SubsystemHealthState:
    """
    Tracks the health of a single cognitive subsystem.

    Attributes:
        name: Subsystem identifier (e.g., 'perception', 'affect')
        status: Current health status
        consecutive_failures: Number of failures in a row without a success
        total_failures: Lifetime failure count
        total_successes: Lifetime success count
        last_error: String representation of most recent error
        last_error_time: Timestamp of most recent error
        last_success_time: Timestamp of most recent success
        disabled_until: When FAILED, timestamp at which recovery will be attempted
        is_critical: If True, subsystem is always attempted regardless of status
    """
    name: str
    status: SubsystemStatus = SubsystemStatus.HEALTHY
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_error: Optional[str] = None
    last_error_time: float = 0.0
    last_success_time: float = 0.0
    disabled_until: float = 0.0
    is_critical: bool = False

    @property
    def failure_rate(self) -> float:
        """Fraction of total executions that failed."""
        total = self.total_failures + self.total_successes
        if total == 0:
            return 0.0
        return self.total_failures / total

    @property
    def is_healthy(self) -> bool:
        return self.status == SubsystemStatus.HEALTHY

    def to_dict(self) -> Dict:
        """Serialize health state for reporting."""
        return {
            "name": self.name,
            "status": self.status.value,
            "consecutive_failures": self.consecutive_failures,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "failure_rate": round(self.failure_rate, 4),
            "last_error": self.last_error,
            "is_critical": self.is_critical,
        }


# Subsystems that must always be attempted even when FAILED.
# A workspace update failure is catastrophic, so we never skip it.
CRITICAL_SUBSYSTEMS = frozenset({"workspace_update"})


class SubsystemSupervisor:
    """
    Monitors subsystem health and controls fault isolation.

    The supervisor wraps around the CycleExecutor's per-step error handling.
    For each cognitive step:
        1. Check should_execute() before attempting the step
        2. Call record_success() if the step completed without error
        3. Call record_failure() if the step raised an exception

    When a subsystem accumulates too many consecutive failures, its circuit
    breaker opens and the step is skipped until a recovery timeout elapses.
    At that point, one recovery attempt is made. If it succeeds, the
    subsystem returns to HEALTHY. If it fails, the backoff doubles.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the subsystem supervisor.

        Config keys:
            failure_threshold (int): Consecutive failures before FAILED. Default 5.
            degraded_threshold (int): Consecutive failures before DEGRADED. Default 2.
            recovery_timeout (float): Seconds to wait before attempting recovery. Default 30.0.
            max_recovery_timeout (float): Maximum backoff cap in seconds. Default 300.0.
        """
        config = config or {}
        self.failure_threshold: int = config.get("failure_threshold", 5)
        self.degraded_threshold: int = config.get("degraded_threshold", 2)
        self.recovery_timeout: float = config.get("recovery_timeout", 30.0)
        self.max_recovery_timeout: float = config.get("max_recovery_timeout", 300.0)

        self._health: Dict[str, SubsystemHealthState] = {}
        self._recovery_backoff: Dict[str, float] = {}

        logger.info(
            f"SubsystemSupervisor initialized: "
            f"failure_threshold={self.failure_threshold}, "
            f"degraded_threshold={self.degraded_threshold}, "
            f"recovery_timeout={self.recovery_timeout}s"
        )

    def _get_or_create(self, name: str) -> SubsystemHealthState:
        """Get health state for a subsystem, creating it if needed."""
        if name not in self._health:
            self._health[name] = SubsystemHealthState(
                name=name,
                is_critical=(name in CRITICAL_SUBSYSTEMS),
            )
        return self._health[name]

    def should_execute(self, name: str) -> bool:
        """
        Check whether a subsystem should be attempted this cycle.

        Returns True for HEALTHY, DEGRADED, and RECOVERING subsystems.
        Returns False for FAILED subsystems unless the recovery timeout
        has elapsed, in which case it transitions to RECOVERING.

        Critical subsystems always return True.
        """
        health = self._get_or_create(name)

        if health.is_critical:
            return True

        if health.status in (SubsystemStatus.HEALTHY, SubsystemStatus.DEGRADED):
            return True

        if health.status == SubsystemStatus.FAILED:
            if time.time() >= health.disabled_until:
                health.status = SubsystemStatus.RECOVERING
                logger.info(f"Subsystem '{name}' entering recovery test")
                return True
            return False

        if health.status == SubsystemStatus.RECOVERING:
            return True

        return True

    def record_success(self, name: str) -> None:
        """Record a successful execution of a subsystem step."""
        health = self._get_or_create(name)
        health.consecutive_failures = 0
        health.total_successes += 1
        health.last_success_time = time.time()

        if health.status in (SubsystemStatus.DEGRADED, SubsystemStatus.RECOVERING):
            old_status = health.status
            health.status = SubsystemStatus.HEALTHY
            # Reset backoff on successful recovery
            self._recovery_backoff.pop(name, None)
            logger.info(f"Subsystem '{name}' recovered: {old_status.value} -> HEALTHY")

    def record_failure(self, name: str, error: Exception) -> None:
        """
        Record a failed execution of a subsystem step.

        Transitions:
            HEALTHY/DEGRADED + N consecutive failures >= degraded_threshold -> DEGRADED
            HEALTHY/DEGRADED + N consecutive failures >= failure_threshold  -> FAILED
            RECOVERING + failure -> FAILED (with doubled backoff)
        """
        health = self._get_or_create(name)
        health.consecutive_failures += 1
        health.total_failures += 1
        health.last_error = f"{type(error).__name__}: {error}"
        health.last_error_time = time.time()

        if health.status == SubsystemStatus.RECOVERING:
            # Recovery attempt failed — increase backoff
            current_backoff = self._recovery_backoff.get(name, self.recovery_timeout)
            new_backoff = min(current_backoff * 2, self.max_recovery_timeout)
            self._recovery_backoff[name] = new_backoff

            health.status = SubsystemStatus.FAILED
            health.disabled_until = time.time() + new_backoff
            logger.warning(
                f"Subsystem '{name}' recovery FAILED — "
                f"disabled for {new_backoff:.0f}s (backoff doubled)"
            )

        elif health.consecutive_failures >= self.failure_threshold:
            if health.status != SubsystemStatus.FAILED:
                backoff = self._recovery_backoff.get(name, self.recovery_timeout)
                health.status = SubsystemStatus.FAILED
                health.disabled_until = time.time() + backoff
                logger.error(
                    f"Subsystem '{name}' -> FAILED after "
                    f"{health.consecutive_failures} consecutive failures. "
                    f"Disabled for {backoff:.0f}s. Last error: {health.last_error}"
                )

        elif health.consecutive_failures >= self.degraded_threshold:
            if health.status == SubsystemStatus.HEALTHY:
                health.status = SubsystemStatus.DEGRADED
                logger.warning(
                    f"Subsystem '{name}' -> DEGRADED "
                    f"({health.consecutive_failures} consecutive failures)"
                )

    def get_health(self, name: str) -> SubsystemHealthState:
        """Get the health state for a specific subsystem."""
        return self._get_or_create(name)

    def get_all_health(self) -> Dict[str, SubsystemHealthState]:
        """Get health states for all tracked subsystems."""
        return dict(self._health)

    def get_system_report(self) -> Dict:
        """
        Generate a system-wide health report.

        Returns:
            Dict with overall_status, failed/degraded counts, and per-subsystem details.
        """
        failed_count = 0
        degraded_count = 0
        subsystems = {}

        for name, health in self._health.items():
            subsystems[name] = health.to_dict()
            if health.status == SubsystemStatus.FAILED:
                failed_count += 1
            elif health.status == SubsystemStatus.DEGRADED:
                degraded_count += 1

        total = len(self._health)
        if failed_count > 0 and failed_count >= total / 2:
            overall = "critical"
        elif failed_count > 0:
            overall = "degraded"
        elif degraded_count > 0:
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "overall_status": overall,
            "total_subsystems": total,
            "healthy_count": total - failed_count - degraded_count,
            "degraded_count": degraded_count,
            "failed_count": failed_count,
            "subsystems": subsystems,
        }

    def get_failed_subsystems(self) -> List[str]:
        """Return names of all currently FAILED subsystems."""
        return [
            name for name, health in self._health.items()
            if health.status == SubsystemStatus.FAILED
        ]

    def get_degraded_subsystems(self) -> List[str]:
        """Return names of all currently DEGRADED subsystems."""
        return [
            name for name, health in self._health.items()
            if health.status == SubsystemStatus.DEGRADED
        ]

    def reset(self, name: str) -> None:
        """Manually reset a subsystem to HEALTHY (e.g., after a fix or restart)."""
        health = self._get_or_create(name)
        health.status = SubsystemStatus.HEALTHY
        health.consecutive_failures = 0
        health.disabled_until = 0.0
        self._recovery_backoff.pop(name, None)
        logger.info(f"Subsystem '{name}' manually reset to HEALTHY")

    def reset_all(self) -> None:
        """Reset all subsystems to HEALTHY."""
        for name in list(self._health.keys()):
            self.reset(name)
