"""
Bottleneck Detector: Monitors cognitive load and detects processing bottlenecks.

Enables self-preserving behavior by detecting overload conditions and routing
signals to inhibition systems and introspection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Deque
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)


class BottleneckType(Enum):
    """Types of cognitive bottlenecks."""
    WORKSPACE_OVERLOAD = "workspace_overload"
    SUBSYSTEM_SLOWDOWN = "subsystem_slowdown"
    GOAL_RESOURCE_EXHAUSTION = "goal_resource_exhaustion"
    MEMORY_LAG = "memory_lag"
    CYCLE_OVERRUN = "cycle_overrun"


# Mapping from bottleneck type to introspective description
_INTROSPECTION_TEMPLATES = {
    BottleneckType.WORKSPACE_OVERLOAD: "My attention is stretched thin across too many things",
    BottleneckType.GOAL_RESOURCE_EXHAUSTION: "I have more goals than I can effectively pursue",
    BottleneckType.MEMORY_LAG: "I'm having trouble consolidating recent experiences",
    BottleneckType.CYCLE_OVERRUN: "My thinking is taking longer than usual",
    BottleneckType.SUBSYSTEM_SLOWDOWN: "Some of my cognitive processes are running slowly",
}


@dataclass
class BottleneckSignal:
    """A detected bottleneck condition."""
    bottleneck_type: BottleneckType
    severity: float  # 0.0 to 1.0
    source: str
    description: str
    detected_at: datetime = field(default_factory=datetime.now)
    consecutive_cycles: int = 1
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.severity = max(0.0, min(1.0, self.severity))


@dataclass
class BottleneckState:
    """Overall bottleneck state of the cognitive system."""
    is_bottlenecked: bool = False
    overall_load: float = 0.0
    active_bottlenecks: List[BottleneckSignal] = field(default_factory=list)
    recommendation: str = "normal_operation"
    last_updated: datetime = field(default_factory=datetime.now)

    def get_severity(self) -> float:
        """Get maximum severity among active bottlenecks."""
        if not self.active_bottlenecks:
            return 0.0
        return max(b.severity for b in self.active_bottlenecks)


class BottleneckDetector:
    """Detects cognitive processing bottlenecks and generates appropriate signals."""

    # Default thresholds
    DEFAULT_WORKSPACE_THRESHOLD = 20
    DEFAULT_SLOWDOWN_FACTOR = 2.0
    DEFAULT_RESOURCE_THRESHOLD = 0.9
    DEFAULT_CYCLE_TARGET_MS = 100
    DEFAULT_PERSISTENCE_CYCLES = 3
    DEFAULT_MEMORY_LAG_MS = 500
    DEFAULT_BASELINE_MIN_SAMPLES = 10

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # Thresholds
        self.workspace_threshold = config.get("workspace_overload_threshold", self.DEFAULT_WORKSPACE_THRESHOLD)
        self.slowdown_factor = config.get("subsystem_slowdown_factor", self.DEFAULT_SLOWDOWN_FACTOR)
        self.resource_threshold = min(0.99, config.get("resource_exhaustion_threshold", self.DEFAULT_RESOURCE_THRESHOLD))
        self.cycle_target_ms = max(1, config.get("cycle_duration_target_ms", self.DEFAULT_CYCLE_TARGET_MS))
        self.persistence_threshold = config.get("consecutive_cycles_for_bottleneck", self.DEFAULT_PERSISTENCE_CYCLES)
        self.memory_lag_ms = config.get("memory_lag_threshold_ms", self.DEFAULT_MEMORY_LAG_MS)

        # History tracking
        self._timing_history: Deque[Dict[str, float]] = deque(maxlen=100)
        self._load_history: Deque[float] = deque(maxlen=100)
        self._counters: Dict[str, int] = {}
        self._baselines: Dict[str, float] = {}
        self._baseline_dirty = True  # Track if baselines need recomputation

        self._current_state = BottleneckState()
        self._last_warning_time = 0.0  # Rate-limit warning logs

        logger.info("BottleneckDetector initialized")

    def update(
        self,
        subsystem_timings: Dict[str, float],
        workspace_percept_count: int,
        goal_resource_utilization: float,
        goal_queue_depth: int = 0,
        waiting_goals: int = 0
    ) -> BottleneckState:
        """Update bottleneck detection with current metrics."""
        # Input validation
        workspace_percept_count = max(0, workspace_percept_count)
        goal_resource_utilization = max(0.0, min(1.0, goal_resource_utilization))
        subsystem_timings = {k: max(0.0, v) for k, v in subsystem_timings.items()}

        # Update history
        self._timing_history.append(subsystem_timings)
        self._baseline_dirty = True

        # Detect bottlenecks
        bottlenecks = self._detect_all(
            subsystem_timings, workspace_percept_count,
            goal_resource_utilization, waiting_goals
        )

        # Compute load
        overall_load = self._compute_load(
            workspace_percept_count, goal_resource_utilization, subsystem_timings
        )
        self._load_history.append(overall_load)

        # Evaluate persistence and generate recommendation
        is_bottlenecked = any(
            b.consecutive_cycles >= self.persistence_threshold for b in bottlenecks
        )
        recommendation = self._recommend(bottlenecks, overall_load)

        self._current_state = BottleneckState(
            is_bottlenecked=is_bottlenecked,
            overall_load=overall_load,
            active_bottlenecks=bottlenecks,
            recommendation=recommendation,
            last_updated=datetime.now()
        )

        if is_bottlenecked:
            import time as _time
            now = _time.time()
            if now - self._last_warning_time >= 30.0:
                logger.warning(
                    f"Bottleneck: load={overall_load:.0%}, count={len(bottlenecks)}, action={recommendation}"
                )
                self._last_warning_time = now

        return self._current_state

    def _detect_all(
        self,
        timings: Dict[str, float],
        percept_count: int,
        utilization: float,
        waiting_goals: int
    ) -> List[BottleneckSignal]:
        """Run all detection checks and return signals."""
        signals = []

        # Workspace overload
        if percept_count > self.workspace_threshold:
            excess = percept_count - self.workspace_threshold
            severity = min(1.0, excess / max(1, self.workspace_threshold))
            signals.append(self._make_signal(
                "workspace_overload", BottleneckType.WORKSPACE_OVERLOAD, severity,
                "workspace", f"Workspace: {percept_count} percepts (limit: {self.workspace_threshold})"
            ))
        else:
            self._counters.pop("workspace_overload", None)

        # Resource exhaustion
        if utilization >= self.resource_threshold:
            denom = max(0.01, 1.0 - self.resource_threshold)
            severity = min(1.0, (utilization - self.resource_threshold) / denom)
            if waiting_goals > 0:
                severity = min(1.0, severity + 0.2)
            signals.append(self._make_signal(
                "resource_exhaustion", BottleneckType.GOAL_RESOURCE_EXHAUSTION, severity,
                "goal_competition", f"Resources: {utilization:.0%}, {waiting_goals} goals waiting"
            ))
        else:
            self._counters.pop("resource_exhaustion", None)

        # Memory lag
        memory_time = timings.get("memory_consolidation", 0)
        if memory_time > self.memory_lag_ms:
            severity = min(1.0, (memory_time - self.memory_lag_ms) / max(1, self.memory_lag_ms))
            signals.append(self._make_signal(
                "memory_lag", BottleneckType.MEMORY_LAG, severity,
                "memory", f"Memory: {memory_time:.0f}ms (limit: {self.memory_lag_ms}ms)"
            ))
        else:
            self._counters.pop("memory_lag", None)

        # Cycle overrun
        total_time = sum(timings.values()) if timings else 0
        if total_time > self.cycle_target_ms:
            ratio = total_time / self.cycle_target_ms
            severity = min(1.0, (ratio - 1.0) / 2.0)
            signals.append(self._make_signal(
                "cycle_overrun", BottleneckType.CYCLE_OVERRUN, severity,
                "cycle_executor", f"Cycle: {total_time:.0f}ms (target: {self.cycle_target_ms}ms)"
            ))
        else:
            self._counters.pop("cycle_overrun", None)

        # Subsystem slowdowns
        signals.extend(self._detect_slowdowns(timings))

        return signals

    def _detect_slowdowns(self, timings: Dict[str, float]) -> List[BottleneckSignal]:
        """Detect subsystems running slower than baseline."""
        self._update_baselines()
        signals = []

        for subsystem, duration in timings.items():
            baseline = self._baselines.get(subsystem)
            if baseline is None or baseline < 1.0:
                continue

            ratio = duration / baseline
            key = f"slowdown_{subsystem}"

            if ratio >= self.slowdown_factor:
                severity = min(1.0, (ratio - 1.0) / (self.slowdown_factor * 2))
                signals.append(self._make_signal(
                    key, BottleneckType.SUBSYSTEM_SLOWDOWN, severity,
                    subsystem, f"{subsystem}: {ratio:.1f}x slower than baseline"
                ))
            else:
                self._counters.pop(key, None)

        return signals

    def _make_signal(
        self, key: str, btype: BottleneckType, severity: float, source: str, desc: str
    ) -> BottleneckSignal:
        """Create a signal and increment its counter."""
        self._counters[key] = self._counters.get(key, 0) + 1
        return BottleneckSignal(
            bottleneck_type=btype, severity=severity, source=source,
            description=desc, consecutive_cycles=self._counters[key]
        )

    def _update_baselines(self) -> None:
        """Update baseline timings from history (lazy, only when dirty)."""
        if not self._baseline_dirty or len(self._timing_history) < self.DEFAULT_BASELINE_MIN_SAMPLES:
            return

        # Collect all subsystem names
        subsystems = set()
        for t in self._timing_history:
            subsystems.update(t.keys())

        # Compute median for each
        for sub in subsystems:
            values = sorted(t.get(sub, 0) for t in self._timing_history if sub in t)
            if values:
                self._baselines[sub] = values[len(values) // 2]

        self._baseline_dirty = False

    def _compute_load(
        self, percept_count: int, utilization: float, timings: Dict[str, float]
    ) -> float:
        """Compute overall cognitive load (0.0 to 1.0)."""
        ws_denom = max(1, self.workspace_threshold) * 1.5
        workspace_load = min(1.0, percept_count / ws_denom)
        timing_load = min(1.0, sum(timings.values()) / (self.cycle_target_ms * 2)) if timings else 0.0

        # Weighted: resources 50%, workspace 25%, timing 25%
        return min(1.0, utilization * 0.5 + workspace_load * 0.25 + timing_load * 0.25)

    def _recommend(self, bottlenecks: List[BottleneckSignal], load: float) -> str:
        """Generate recommendation based on bottleneck state."""
        if not bottlenecks:
            return "normal_operation"

        worst = max(bottlenecks, key=lambda b: b.severity)

        recommendations = {
            BottleneckType.GOAL_RESOURCE_EXHAUSTION: (
                "pause_low_priority_goals" if worst.severity > 0.7 else "reduce_goal_parallelism"
            ),
            BottleneckType.WORKSPACE_OVERLOAD: "increase_attention_selectivity",
            BottleneckType.MEMORY_LAG: "defer_memory_consolidation",
            BottleneckType.CYCLE_OVERRUN: "skip_optional_processing",
            BottleneckType.SUBSYSTEM_SLOWDOWN: f"investigate_{worst.source}",
        }

        return recommendations.get(worst.bottleneck_type, "reduce_processing_load" if load > 0.8 else "monitor")

    # Public accessors
    def get_state(self) -> BottleneckState:
        return self._current_state

    def is_overloaded(self) -> bool:
        return self._current_state.is_bottlenecked

    def get_load(self) -> float:
        return self._current_state.overall_load

    def get_average_load(self, cycles: int = 10) -> float:
        if not self._load_history:
            return 0.0
        # Slice deque directly without full list conversion
        n = min(cycles, len(self._load_history))
        return sum(list(self._load_history)[-n:]) / n

    def should_inhibit_communication(self) -> bool:
        """Check if communication should be inhibited due to bottleneck."""
        if not self._current_state.is_bottlenecked:
            return False
        return self._current_state.overall_load > 0.8 or self._current_state.get_severity() > 0.7

    def get_introspection_text(self) -> Optional[str]:
        """Generate first-person description of bottleneck state."""
        if not self._current_state.is_bottlenecked or not self._current_state.active_bottlenecks:
            return None

        lines = ["I notice I'm experiencing processing constraints:"]
        for b in self._current_state.active_bottlenecks:
            lines.append(f"- {_INTROSPECTION_TEMPLATES.get(b.bottleneck_type, 'Unknown constraint')}")

        lines.append(f"\nCognitive load: {self._current_state.overall_load:.0%}")
        if self._current_state.recommendation != "normal_operation":
            lines.append(f"Adapting: {self._current_state.recommendation.replace('_', ' ')}")

        return "\n".join(lines)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "is_bottlenecked": self._current_state.is_bottlenecked,
            "overall_load": self._current_state.overall_load,
            "average_load": self.get_average_load(),
            "bottleneck_count": len(self._current_state.active_bottlenecks),
            "bottleneck_types": [b.bottleneck_type.value for b in self._current_state.active_bottlenecks],
            "max_severity": self._current_state.get_severity(),
            "recommendation": self._current_state.recommendation,
            "should_inhibit": self.should_inhibit_communication(),
        }
