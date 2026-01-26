"""
Timing and performance tracking for the cognitive core.

Handles rate limiting, cycle timing enforcement, and performance metrics.
"""

from __future__ import annotations

import time
import logging
from typing import Dict, Any, List
from collections import deque
from statistics import mean

logger = logging.getLogger(__name__)


class TimingManager:
    """
    Manages timing, rate limiting, and performance metrics for the cognitive loop.
    
    Responsibilities:
    - Track cycle times and performance metrics
    - Enforce timing thresholds (warn/critical)
    - Maintain ~10Hz cycle rate
    - Monitor subsystem performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize timing manager with validated configuration.
        
        Args:
            config: Configuration dict with:
                - cycle_rate_hz: Target cycle frequency (default: 10)
                - timing.warn_threshold_ms: Warn if cycle exceeds this
                - timing.critical_threshold_ms: Critical warning threshold
                - log_interval_cycles: How often to log metrics
                
        Raises:
            ValueError: If configuration values are invalid
        """
        self.config = config
        
        # Validate and set cycle rate
        cycle_rate_hz = config.get("cycle_rate_hz", 10)
        if cycle_rate_hz <= 0:
            raise ValueError(f"cycle_rate_hz must be positive, got {cycle_rate_hz}")
        self.cycle_duration = 1.0 / cycle_rate_hz
        
        # Performance metrics
        self.metrics: Dict[str, Any] = {
            'total_cycles': 0,
            'cycle_times': deque(maxlen=100),
            'attention_selections': 0,
            'percepts_processed': 0,
            'slow_cycles': 0,
            'critical_cycles': 0,
            'slowest_cycle_ms': 0.0,
        }
        
        # Validate and set timing thresholds
        timing_config = config.get("timing", {})
        self.warn_threshold_ms = timing_config.get("warn_threshold_ms", 100)
        self.critical_threshold_ms = timing_config.get("critical_threshold_ms", 200)
        
        if self.warn_threshold_ms <= 0:
            raise ValueError(f"warn_threshold_ms must be positive, got {self.warn_threshold_ms}")
        if self.critical_threshold_ms <= self.warn_threshold_ms:
            raise ValueError(
                f"critical_threshold_ms ({self.critical_threshold_ms}) must be greater than "
                f"warn_threshold_ms ({self.warn_threshold_ms})"
            )
        
        # Validate and set log interval
        self.log_interval = config.get("log_interval_cycles", 100)
        if self.log_interval <= 0:
            raise ValueError(f"log_interval_cycles must be positive, got {self.log_interval}")
    
    def check_cycle_timing(self, cycle_time: float, cycle_number: int) -> None:
        """
        Check if cycle time exceeds thresholds and log warnings.
        
        Args:
            cycle_time: Time taken for cycle in seconds
            cycle_number: Current cycle number for logging
        """
        cycle_time_ms = cycle_time * 1000
        
        if cycle_time_ms > self.critical_threshold_ms:
            logger.warning(
                f"⚠️ CRITICAL: Cognitive cycle exceeded {self.critical_threshold_ms}ms "
                f"(actual: {cycle_time_ms:.1f}ms, target: {self.cycle_duration*1000:.0f}ms). "
                f"Cycle {cycle_number}. "
                f"System performance degraded."
            )
            self.metrics['critical_cycles'] += 1
        elif cycle_time_ms > self.warn_threshold_ms:
            logger.warning(
                f"⚠️  Cognitive cycle exceeded target "
                f"(actual: {cycle_time_ms:.1f}ms, target: {self.warn_threshold_ms:.0f}ms). "
                f"Cycle {cycle_number}."
            )
            self.metrics['slow_cycles'] += 1
    
    def update_metrics(self, cycle_time: float, subsystem_timings: Dict[str, float] = None) -> None:
        """
        Update performance metrics.
        
        Args:
            cycle_time: Time taken for the current cycle (seconds)
            subsystem_timings: Optional dict of subsystem name -> timing in ms
        """
        self.metrics['total_cycles'] += 1
        self.metrics['cycle_times'].append(cycle_time)
        
        # Track slowest cycle
        cycle_time_ms = cycle_time * 1000
        if cycle_time_ms > self.metrics['slowest_cycle_ms']:
            self.metrics['slowest_cycle_ms'] = cycle_time_ms
        
        # Track per-subsystem timings
        if subsystem_timings:
            if 'subsystem_timings' not in self.metrics:
                self.metrics['subsystem_timings'] = {
                    name: deque(maxlen=100) for name in subsystem_timings.keys()
                }
            
            for subsystem, timing in subsystem_timings.items():
                if subsystem not in self.metrics['subsystem_timings']:
                    self.metrics['subsystem_timings'][subsystem] = deque(maxlen=100)
                self.metrics['subsystem_timings'][subsystem].append(timing)
        
        # Log every N cycles
        if self.metrics['total_cycles'] % self.log_interval == 0:
            self._log_performance_summary(subsystem_timings)
    
    def _log_performance_summary(self, subsystem_timings: Dict[str, float] = None) -> None:
        """Log performance summary."""
        avg_time = mean(self.metrics['cycle_times'])
        
        # Calculate subsystem averages if available
        if 'subsystem_timings' in self.metrics and subsystem_timings:
            subsystem_avgs = {
                name: sum(timings) / len(timings) if timings else 0
                for name, timings in self.metrics['subsystem_timings'].items()
            }
            
            # Sort by time descending to show bottlenecks first
            sorted_subsystems = sorted(subsystem_avgs.items(), key=lambda x: -x[1])
            subsystem_breakdown = "\n".join(
                f"    - {name}: {avg:.1f}ms" 
                for name, avg in sorted_subsystems
            )
            
            logger.info(
                f"📊 Performance Summary (last {self.log_interval} cycles):\n"
                f"  Total Cycles: {self.metrics['total_cycles']}\n"
                f"  Avg Cycle Time: {avg_time*1000:.1f}ms\n"
                f"  Subsystem Breakdown:\n{subsystem_breakdown}"
            )
        else:
            logger.info(
                f"📊 Cycle {self.metrics['total_cycles']}: "
                f"avg_time={avg_time*1000:.1f}ms, "
                f"target={self.cycle_duration*1000:.0f}ms"
            )
    
    def calculate_sleep_time(self, cycle_time: float) -> float:
        """
        Calculate how long to sleep to maintain target cycle rate.
        
        Args:
            cycle_time: Time taken for current cycle (seconds)
            
        Returns:
            Sleep time in seconds (0 if cycle overran)
        """
        sleep_time = max(0, self.cycle_duration - cycle_time)
        if sleep_time == 0:
            logger.debug(f"Cycle overran by {abs(self.cycle_duration - cycle_time)*1000:.1f}ms, skipping sleep")
        return sleep_time
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of timing metrics.
        
        Returns:
            Dict with timing statistics
        """
        avg_cycle_time = mean(self.metrics['cycle_times']) if self.metrics['cycle_times'] else 0.0
        
        # Calculate timing statistics
        slow_cycle_pct = 0.0
        critical_cycle_pct = 0.0
        if self.metrics['total_cycles'] > 0:
            slow_cycle_pct = (self.metrics['slow_cycles'] / self.metrics['total_cycles']) * 100
            critical_cycle_pct = (self.metrics['critical_cycles'] / self.metrics['total_cycles']) * 100
        
        return {
            'total_cycles': self.metrics['total_cycles'],
            'avg_cycle_time_ms': avg_cycle_time * 1000,
            'target_cycle_time_ms': self.cycle_duration * 1000,
            'cycle_rate_hz': self.config.get('cycle_rate_hz', 10),
            'slow_cycles': self.metrics['slow_cycles'],
            'slow_cycle_percentage': slow_cycle_pct,
            'critical_cycles': self.metrics['critical_cycles'],
            'critical_cycle_percentage': critical_cycle_pct,
            'slowest_cycle_ms': self.metrics['slowest_cycle_ms'],
        }
    
    def get_performance_breakdown(self) -> Dict[str, Any]:
        """
        Get detailed performance breakdown by subsystem.
        
        Returns:
            Dict mapping subsystem names to timing statistics (all values in ms)
        """
        if 'subsystem_timings' not in self.metrics:
            return {}
        
        breakdown = {}
        for subsystem, timings in self.metrics['subsystem_timings'].items():
            if not timings:
                continue
            
            timings_list = list(timings)
            timings_sorted = sorted(timings_list)
            n = len(timings_sorted)
            
            # Calculate percentiles safely
            p50_idx = max(0, n // 2 - 1) if n > 1 else 0
            p95_idx = max(0, int(n * 0.95) - 1) if n > 1 else 0
            p99_idx = max(0, int(n * 0.99) - 1) if n > 1 else 0
            
            breakdown[subsystem] = {
                'avg_ms': sum(timings_list) / n,
                'min_ms': min(timings_list),
                'max_ms': max(timings_list),
                'p50_ms': timings_sorted[p50_idx] if n > 0 else 0,
                'p95_ms': timings_sorted[p95_idx] if n > 0 else 0,
                'p99_ms': timings_sorted[p99_idx] if n > 0 else 0,
            }
        
        return breakdown
