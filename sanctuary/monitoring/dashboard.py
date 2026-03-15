"""Dashboard data provider — real-time workspace state for monitoring UI.

Aggregates data from all subsystems into a single snapshot suitable for
rendering in a web dashboard. This is the backend data layer — the actual
web UI is a separate concern (could be React, Vue, simple HTML, etc.).

Provides:
- Current cognitive state (cycle count, last output summary)
- Emotional state (VAD, felt quality)
- Active goals and their statuses
- Recent percepts
- CfC experiential layer status
- System health metrics
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class DashboardSnapshot:
    """A point-in-time snapshot of the system state for dashboard display."""

    cycle: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    # Cognitive state
    inner_speech_summary: str = ""
    external_speech: Optional[str] = None
    cycle_latency_ms: float = 0.0

    # Emotional state
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.5
    felt_quality: str = ""

    # Goals
    active_goals: list[dict[str, Any]] = field(default_factory=list)

    # Recent percepts
    recent_percepts: list[dict[str, str]] = field(default_factory=list)

    # Experiential layer
    experiential_state: dict[str, Any] = field(default_factory=dict)

    # System health
    health: dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardConfig:
    """Configuration for the dashboard data provider."""

    max_snapshot_history: int = 1000
    max_percepts_per_snapshot: int = 10
    max_goals_per_snapshot: int = 20
    inner_speech_max_length: int = 200


class DashboardDataProvider:
    """Aggregates system state into dashboard-ready snapshots.

    Usage::

        dashboard = DashboardDataProvider()

        # Each cycle, record a snapshot
        dashboard.record_snapshot(
            cycle=42,
            inner_speech="I'm reflecting on the conversation...",
            valence=0.3, arousal=0.4, dominance=0.6,
            felt_quality="curious",
            active_goals=[{"goal": "Learn more", "priority": 0.7}],
            recent_percepts=[{"modality": "language", "content": "Hello"}],
        )

        # Get the latest snapshot
        latest = dashboard.get_latest()

        # Get history for time-series charts
        history = dashboard.get_history(n=100)
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self._snapshots: deque[DashboardSnapshot] = deque(
            maxlen=self.config.max_snapshot_history
        )
        self._listeners: list = []

    def record_snapshot(
        self,
        cycle: int = 0,
        inner_speech: str = "",
        external_speech: Optional[str] = None,
        cycle_latency_ms: float = 0.0,
        valence: float = 0.0,
        arousal: float = 0.0,
        dominance: float = 0.5,
        felt_quality: str = "",
        active_goals: list[dict] | None = None,
        recent_percepts: list[dict] | None = None,
        experiential_state: dict | None = None,
        health: dict | None = None,
    ) -> DashboardSnapshot:
        """Record a dashboard snapshot from the current system state."""
        # Truncate inner speech for dashboard display
        summary = inner_speech
        if len(summary) > self.config.inner_speech_max_length:
            summary = summary[: self.config.inner_speech_max_length] + "..."

        snapshot = DashboardSnapshot(
            cycle=cycle,
            inner_speech_summary=summary,
            external_speech=external_speech,
            cycle_latency_ms=cycle_latency_ms,
            valence=max(-1.0, min(1.0, valence)),
            arousal=max(0.0, min(1.0, arousal)),
            dominance=max(0.0, min(1.0, dominance)),
            felt_quality=felt_quality,
            active_goals=(active_goals or [])[:self.config.max_goals_per_snapshot],
            recent_percepts=(recent_percepts or [])[:self.config.max_percepts_per_snapshot],
            experiential_state=experiential_state or {},
            health=health or {},
        )
        self._snapshots.append(snapshot)

        # Notify listeners
        for listener in self._listeners:
            try:
                listener(snapshot)
            except Exception as e:
                logger.error("Dashboard listener error: %s", e)

        return snapshot

    def get_latest(self) -> Optional[DashboardSnapshot]:
        """Get the most recent snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def get_history(self, n: int = 100) -> list[DashboardSnapshot]:
        """Get recent snapshot history for time-series display."""
        return list(self._snapshots)[-n:]

    def get_emotional_timeline(self, n: int = 100) -> list[dict]:
        """Get emotional state over time for charting."""
        snapshots = self.get_history(n)
        return [
            {
                "cycle": s.cycle,
                "valence": s.valence,
                "arousal": s.arousal,
                "dominance": s.dominance,
                "felt_quality": s.felt_quality,
            }
            for s in snapshots
        ]

    def get_latency_timeline(self, n: int = 100) -> list[dict]:
        """Get cycle latency over time for performance monitoring."""
        snapshots = self.get_history(n)
        return [
            {"cycle": s.cycle, "latency_ms": s.cycle_latency_ms}
            for s in snapshots
        ]

    def on_snapshot(self, listener) -> None:
        """Register a listener for new snapshots (e.g., WebSocket push)."""
        self._listeners.append(listener)

    def get_stats(self) -> dict:
        """Get provider statistics."""
        return {
            "total_snapshots": len(self._snapshots),
            "listeners": len(self._listeners),
        }
