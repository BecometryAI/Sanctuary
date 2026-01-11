"""
Meta-Cognitive Monitor: Observes and tracks cognitive processing patterns.

This module implements the MetaCognitiveMonitor class that observes
cognitive processes and identifies patterns in processing behavior.
"""

from __future__ import annotations

import logging
from typing import List, Dict

from .processing_observation import (
    ProcessingObservation,
    ProcessingContext,
    ProcessStats
)
from .pattern_detection import PatternDetector, CognitivePattern

logger = logging.getLogger(__name__)


class MetaCognitiveMonitor:
    """
    Monitors cognitive processing and identifies patterns.
    
    Provides context managers for observing cognitive processes,
    records observations, and uses pattern detection to identify
    success conditions, failure modes, and efficiency factors.
    
    Usage:
        monitor = MetaCognitiveMonitor()
        
        with monitor.observe('reasoning') as ctx:
            ctx.input_complexity = 0.8
            result = perform_reasoning()
            ctx.output_quality = assess_quality(result)
        
        # Later, analyze patterns
        patterns = monitor.get_identified_patterns()
        stats = monitor.get_process_statistics('reasoning')
    """
    
    def __init__(self, min_observations_for_patterns: int = 3):
        """
        Initialize meta-cognitive monitor.
        
        Args:
            min_observations_for_patterns: Minimum observations needed to detect patterns
        """
        self.observations: List[ProcessingObservation] = []
        self.pattern_detector = PatternDetector(min_observations=min_observations_for_patterns)
        
        # Statistics cache
        self._stats_cache: Dict[str, ProcessStats] = {}
        self._stats_dirty = False
        
        logger.info("✅ MetaCognitiveMonitor initialized")
    
    def observe(self, process_type: str) -> ProcessingContext:
        """
        Context manager to observe a cognitive process.
        
        Args:
            process_type: Type of process to observe (e.g., 'reasoning', 'memory_retrieval')
            
        Returns:
            ProcessingContext that will record the observation
            
        Example:
            with monitor.observe('goal_selection') as ctx:
                ctx.input_complexity = 0.6
                goals = select_goals()
                ctx.output_quality = 0.9
        """
        return ProcessingContext(self, process_type)
    
    def record_observation(self, obs: ProcessingObservation):
        """
        Record a processing observation.
        
        Args:
            obs: Processing observation to record
        """
        self.observations.append(obs)
        self._update_statistics(obs)
        self.pattern_detector.update(obs)
        
        if not obs.success:
            logger.warning(
                f"Process '{obs.process_type}' failed: {obs.error}"
            )
        else:
            logger.debug(
                f"Process '{obs.process_type}' completed in {obs.duration_ms:.1f}ms "
                f"(quality: {obs.output_quality:.2f})"
            )
        
        # Keep only recent observations to prevent unbounded growth
        max_observations = 10000
        if len(self.observations) > max_observations:
            self.observations = self.observations[-max_observations:]
    
    def _update_statistics(self, obs: ProcessingObservation):
        """
        Update statistics for a process type.
        
        Args:
            obs: New observation to incorporate into statistics
        """
        self._stats_dirty = True
        # Statistics will be recomputed on next access
    
    def get_process_statistics(self, process_type: str) -> ProcessStats:
        """
        Get statistics for a process type.
        
        Args:
            process_type: Type of process to get statistics for
            
        Returns:
            ProcessStats with aggregated metrics for this process type
        """
        # Recompute if cache is dirty or type not cached
        if self._stats_dirty or process_type not in self._stats_cache:
            self._recompute_statistics()
        
        return self._stats_cache.get(
            process_type,
            ProcessStats(
                total_executions=0,
                success_rate=0.0,
                avg_duration_ms=0.0,
                avg_quality=0.0
            )
        )
    
    def _recompute_statistics(self):
        """Recompute statistics for all process types."""
        # Group observations by type
        by_type: Dict[str, List[ProcessingObservation]] = {}
        for obs in self.observations:
            if obs.process_type not in by_type:
                by_type[obs.process_type] = []
            by_type[obs.process_type].append(obs)
        
        # Compute stats for each type
        self._stats_cache = {}
        for process_type, observations in by_type.items():
            if not observations:
                continue
            
            total = len(observations)
            successes = sum(1 for o in observations if o.success)
            success_rate = successes / total if total > 0 else 0.0
            
            avg_duration = (
                sum(o.duration_ms for o in observations) / total
                if total > 0 else 0.0
            )
            
            # Only compute avg quality for successful observations
            successful_obs = [o for o in observations if o.success]
            avg_quality = (
                sum(o.output_quality for o in successful_obs) / len(successful_obs)
                if successful_obs else 0.0
            )
            
            self._stats_cache[process_type] = ProcessStats(
                total_executions=total,
                success_rate=success_rate,
                avg_duration_ms=avg_duration,
                avg_quality=avg_quality
            )
        
        self._stats_dirty = False
    
    def get_identified_patterns(self) -> List[CognitivePattern]:
        """
        Get patterns identified in processing.
        
        Returns:
            List of detected cognitive patterns across all processes
        """
        return self.pattern_detector.get_patterns()
    
    def get_patterns_for_process(self, process_type: str) -> List[CognitivePattern]:
        """
        Get patterns for a specific process type.
        
        Args:
            process_type: Type of process to get patterns for
            
        Returns:
            List of patterns related to this process type
        """
        return self.pattern_detector.get_patterns_for_type(process_type)
    
    def get_all_process_types(self) -> List[str]:
        """
        Get list of all observed process types.
        
        Returns:
            List of process type names that have been observed
        """
        if self._stats_dirty:
            self._recompute_statistics()
        return list(self._stats_cache.keys())
    
    def get_recent_observations(
        self,
        process_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ProcessingObservation]:
        """
        Get recent observations, optionally filtered by process type.
        
        Args:
            process_type: Optional process type to filter by
            limit: Maximum number of observations to return
            
        Returns:
            List of recent processing observations
        """
        observations = self.observations
        
        if process_type:
            observations = [o for o in observations if o.process_type == process_type]
        
        return observations[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all monitoring data.
        
        Returns:
            Dictionary containing overall statistics and patterns
        """
        if self._stats_dirty:
            self._recompute_statistics()
        
        all_patterns = self.get_identified_patterns()
        
        return {
            "total_observations": len(self.observations),
            "process_types": len(self._stats_cache),
            "identified_patterns": len(all_patterns),
            "patterns_by_type": {
                "success_condition": len([p for p in all_patterns if p.pattern_type == 'success_condition']),
                "failure_mode": len([p for p in all_patterns if p.pattern_type == 'failure_mode']),
                "efficiency_factor": len([p for p in all_patterns if p.pattern_type == 'efficiency_factor'])
            },
            "statistics_by_process": {
                ptype: {
                    "executions": stats.total_executions,
                    "success_rate": stats.success_rate,
                    "avg_duration_ms": stats.avg_duration_ms,
                    "avg_quality": stats.avg_quality
                }
                for ptype, stats in self._stats_cache.items()
            }
        }
