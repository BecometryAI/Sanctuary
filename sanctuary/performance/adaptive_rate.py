"""Adaptive cycle rate — auto-adjusting cognitive loop speed based on system load.

When the system is busy (lots of input, high cognitive load), the cycle rate
can increase. When idle, it can slow down to conserve resources. This creates
natural rhythm variation rather than rigid fixed-rate processing.

Factors that influence cycle rate:
- Input queue depth (more percepts → faster processing)
- Recent cycle latency (if cycles are slow, back off)
- Emotional arousal (high arousal → faster processing)
- Sleep state (during sleep → slower rate)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveRateConfig:
    """Configuration for adaptive cycle rate."""

    base_rate_hz: float = 10.0  # Default cycle rate
    min_rate_hz: float = 1.0  # Minimum (idle/sleep)
    max_rate_hz: float = 30.0  # Maximum (high activity)
    # Factor weights
    input_weight: float = 0.3
    latency_weight: float = 0.3
    arousal_weight: float = 0.2
    load_weight: float = 0.2
    # Smoothing
    ema_alpha: float = 0.2  # Exponential moving average for rate changes
    # Thresholds
    high_input_threshold: int = 5  # Percepts per cycle
    high_latency_threshold_ms: float = 80.0  # Slow cycle


class AdaptiveCycleRate:
    """Dynamically adjusts cognitive loop frequency.

    Usage::

        rate = AdaptiveCycleRate()

        # Each cycle, update and get the delay
        delay = rate.compute_delay(
            input_queue_depth=3,
            last_cycle_ms=45.0,
            arousal=0.6,
            cpu_load=0.4,
        )
        await asyncio.sleep(delay)
    """

    def __init__(self, config: Optional[AdaptiveRateConfig] = None):
        self.config = config or AdaptiveRateConfig()
        self._current_rate: float = self.config.base_rate_hz
        self._rate_history: deque[float] = deque(maxlen=200)
        self._total_adjustments: int = 0

    @property
    def current_rate_hz(self) -> float:
        return self._current_rate

    @property
    def current_delay_s(self) -> float:
        if self._current_rate <= 0:
            return 1.0
        return 1.0 / self._current_rate

    def compute_delay(
        self,
        input_queue_depth: int = 0,
        last_cycle_ms: float = 0.0,
        arousal: float = 0.0,
        cpu_load: float = 0.0,
    ) -> float:
        """Compute the delay until next cycle based on current conditions.

        Returns delay in seconds.
        """
        target_rate = self.config.base_rate_hz

        # Input pressure: more input → faster
        input_factor = min(2.0, 1.0 + input_queue_depth / max(1, self.config.high_input_threshold))
        target_rate *= (1.0 + (input_factor - 1.0) * self.config.input_weight)

        # Latency pressure: slow cycles → back off
        if last_cycle_ms > self.config.high_latency_threshold_ms:
            latency_factor = self.config.high_latency_threshold_ms / max(1.0, last_cycle_ms)
            target_rate *= (1.0 - (1.0 - latency_factor) * self.config.latency_weight)

        # Arousal: high arousal → faster
        arousal_factor = 1.0 + arousal * 0.5
        target_rate *= (1.0 + (arousal_factor - 1.0) * self.config.arousal_weight)

        # CPU load: high load → back off
        if cpu_load > 0.7:
            load_factor = 1.0 - (cpu_load - 0.7) * 2.0
            target_rate *= max(0.5, 1.0 - (1.0 - load_factor) * self.config.load_weight)

        # Clamp to limits
        target_rate = max(self.config.min_rate_hz, min(self.config.max_rate_hz, target_rate))

        # Smooth with EMA
        alpha = self.config.ema_alpha
        self._current_rate = alpha * target_rate + (1.0 - alpha) * self._current_rate
        self._current_rate = max(
            self.config.min_rate_hz,
            min(self.config.max_rate_hz, self._current_rate),
        )
        self._rate_history.append(self._current_rate)
        self._total_adjustments += 1

        return self.current_delay_s

    def set_idle(self) -> float:
        """Set to idle rate (minimum). Returns delay."""
        self._current_rate = self.config.min_rate_hz
        self._rate_history.append(self._current_rate)
        return self.current_delay_s

    def set_active(self) -> float:
        """Set to base rate. Returns delay."""
        self._current_rate = self.config.base_rate_hz
        self._rate_history.append(self._current_rate)
        return self.current_delay_s

    def get_rate_timeline(self, last_n: int = 50) -> list[float]:
        """Get recent rate history."""
        return list(self._rate_history)[-last_n:]

    def get_stats(self) -> dict:
        """Get rate statistics."""
        history = list(self._rate_history)
        return {
            "current_rate_hz": round(self._current_rate, 2),
            "current_delay_s": round(self.current_delay_s, 4),
            "avg_rate_hz": (
                round(sum(history) / len(history), 2) if history else 0.0
            ),
            "min_rate_hz": round(min(history), 2) if history else 0.0,
            "max_rate_hz": round(max(history), 2) if history else 0.0,
            "total_adjustments": self._total_adjustments,
        }
