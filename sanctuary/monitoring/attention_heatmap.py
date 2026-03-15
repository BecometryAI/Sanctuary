"""Attention heatmap tracker — visualizing what content receives attention over time.

Records attention allocation across different content types and topics,
enabling visualization of what the system focuses on. Useful for:
- Understanding attention biases
- Debugging attention allocation issues
- Observing how attention shifts with emotional state
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AttentionEvent:
    """A single attention allocation event."""

    target: str  # What received attention
    category: str = ""  # "percept", "memory", "goal", "internal"
    salience: float = 0.0  # 0 to 1
    cycle: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HeatmapCell:
    """Accumulated attention for one target over a time window."""

    target: str
    total_salience: float = 0.0
    event_count: int = 0
    avg_salience: float = 0.0
    peak_salience: float = 0.0


@dataclass
class AttentionHeatmapConfig:
    """Configuration for attention heatmap tracking."""

    max_events: int = 5000
    window_size: int = 100  # Cycles per heatmap window
    max_targets: int = 50  # Top N targets to track


class AttentionHeatmapTracker:
    """Tracks attention allocation for heatmap visualization.

    Usage::

        tracker = AttentionHeatmapTracker()

        # Record attention events
        tracker.record(target="user message", category="percept", salience=0.8, cycle=1)
        tracker.record(target="goal: learn", category="goal", salience=0.6, cycle=1)

        # Get heatmap data
        heatmap = tracker.get_heatmap(window_start=0, window_end=100)
    """

    def __init__(self, config: Optional[AttentionHeatmapConfig] = None):
        self.config = config or AttentionHeatmapConfig()
        self._events: deque[AttentionEvent] = deque(
            maxlen=self.config.max_events
        )
        self._category_totals: dict[str, float] = {}

    def record(
        self,
        target: str,
        category: str = "",
        salience: float = 0.0,
        cycle: int = 0,
    ) -> None:
        """Record an attention allocation event."""
        event = AttentionEvent(
            target=target,
            category=category,
            salience=max(0.0, min(1.0, salience)),
            cycle=cycle,
        )
        self._events.append(event)

        # Update category totals
        self._category_totals[category] = (
            self._category_totals.get(category, 0.0) + salience
        )

    def get_heatmap(
        self, window_start: int = 0, window_end: int = 0
    ) -> list[HeatmapCell]:
        """Get attention heatmap for a cycle window.

        Returns sorted list of HeatmapCells (highest attention first).
        """
        # Filter events in window
        if window_end <= window_start:
            window_end = window_start + self.config.window_size

        events_in_window = [
            e for e in self._events
            if window_start <= e.cycle < window_end
        ]

        # Aggregate by target
        targets: dict[str, list[float]] = {}
        for event in events_in_window:
            if event.target not in targets:
                targets[event.target] = []
            targets[event.target].append(event.salience)

        # Build cells
        cells = []
        for target, saliences in targets.items():
            cell = HeatmapCell(
                target=target,
                total_salience=sum(saliences),
                event_count=len(saliences),
                avg_salience=sum(saliences) / len(saliences),
                peak_salience=max(saliences),
            )
            cells.append(cell)

        # Sort by total salience, limit to top N
        cells.sort(key=lambda c: c.total_salience, reverse=True)
        return cells[: self.config.max_targets]

    def get_category_distribution(self) -> dict[str, float]:
        """Get attention distribution across categories."""
        total = sum(self._category_totals.values()) or 1.0
        return {
            cat: val / total
            for cat, val in sorted(
                self._category_totals.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    def get_attention_over_time(
        self, target: str, n_windows: int = 10
    ) -> list[dict]:
        """Get attention for a specific target over time windows."""
        if not self._events:
            return []

        min_cycle = self._events[0].cycle
        max_cycle = self._events[-1].cycle
        window = max(1, (max_cycle - min_cycle) // max(1, n_windows))

        timeline = []
        for i in range(n_windows):
            start = min_cycle + i * window
            end = start + window
            events = [
                e for e in self._events
                if start <= e.cycle < end and e.target == target
            ]
            avg = sum(e.salience for e in events) / len(events) if events else 0.0
            timeline.append({
                "window_start": start,
                "window_end": end,
                "avg_salience": avg,
                "event_count": len(events),
            })
        return timeline

    def get_stats(self) -> dict:
        """Get tracker statistics."""
        return {
            "total_events": len(self._events),
            "unique_targets": len(set(e.target for e in self._events)),
            "categories": list(self._category_totals.keys()),
        }
