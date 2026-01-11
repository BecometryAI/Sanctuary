"""
Processing Monitor: Observes and tracks cognitive processing patterns.

This module implements meta-cognitive monitoring capabilities that observe
and learn from cognitive processing patterns. It tracks the performance of
different cognitive processes and identifies patterns in successes and failures.
"""

from __future__ import annotations

import time
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CognitiveResources:
    """Resources used by a cognitive process."""
    attention_units: float = 0.0
    memory_accesses: int = 0
    computation_cycles: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingObservation:
    """Observation of a cognitive processing episode."""
    id: str
    timestamp: datetime
    process_type: str  # 'reasoning', 'memory_retrieval', 'goal_selection', etc.
    duration_ms: float
    success: bool
    input_complexity: float  # How complex was the input?
    output_quality: float  # Self-assessed quality
    resources_used: CognitiveResources
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessStats:
    """Statistics for a process type."""
    total_executions: int
    success_rate: float
    avg_duration_ms: float
    avg_quality: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitivePattern:
    """A pattern identified in cognitive processing."""
    pattern_type: str  # 'success_condition', 'failure_mode', 'efficiency_factor'
    description: str
    confidence: float
    supporting_observations: List[str]  # Observation IDs
    actionable: bool  # Can we do something about this?
    suggested_adaptation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessingContext:
    """Context manager for observing a cognitive process."""
    
    def __init__(self, monitor: MetaCognitiveMonitor, process_type: str):
        self.monitor = monitor
        self.process_type = process_type
        self.start_time: Optional[float] = None
        self.observation: Optional[ProcessingObservation] = None
        self.input_complexity: float = 0.5
        self.output_quality: float = 0.5
        self.resources: CognitiveResources = CognitiveResources()
        self.metadata: Dict[str, Any] = {}
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is None:
            return
            
        duration = (time.time() - self.start_time) * 1000
        self.observation = ProcessingObservation(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            process_type=self.process_type,
            duration_ms=duration,
            success=exc_type is None,
            input_complexity=self.input_complexity,
            output_quality=self.output_quality,
            resources_used=self.resources,
            error=str(exc_val) if exc_val else None,
            metadata=self.metadata
        )
        self.monitor.record_observation(self.observation)
    
    def set_complexity(self, complexity: float):
        """Set the input complexity (0.0-1.0)."""
        if not isinstance(complexity, (int, float)):
            raise TypeError(f"Complexity must be numeric, got {type(complexity)}")
        self.input_complexity = max(0.0, min(1.0, float(complexity)))
    
    def set_quality(self, quality: float):
        """Set the output quality (0.0-1.0)."""
        if not isinstance(quality, (int, float)):
            raise TypeError(f"Quality must be numeric, got {type(quality)}")
        self.output_quality = max(0.0, min(1.0, float(quality)))
    
    def set_resources(self, resources: CognitiveResources):
        """Set the resources used."""
        self.resources = resources
    
    def add_metadata(self, key: str, value: Any):
        """Add metadata to the observation."""
        self.metadata[key] = value


class PatternDetector:
    """Detects patterns in cognitive processing."""
    
    # Pattern detection thresholds
    MIN_SAMPLES_FOR_PATTERN = 3
    SUCCESS_COMPLEXITY_THRESHOLD = 0.7  # Successes must be this much simpler
    FAILURE_COMPLEXITY_THRESHOLD = 1.3  # Failures must be this much more complex
    HIGH_QUALITY_THRESHOLD = 0.7
    HIGH_QUALITY_FREQUENCY = 0.6
    SLOW_PROCESS_MS = 500
    HIGH_RESOURCE_ATTENTION = 10
    HIGH_RESOURCE_MEMORY = 20
    HIGH_RESOURCE_FREQUENCY = 0.5
    
    def __init__(self):
        self.observations_by_type: Dict[str, List[ProcessingObservation]] = defaultdict(list)
        self.min_samples_for_pattern = self.MIN_SAMPLES_FOR_PATTERN
    
    def update(self, obs: ProcessingObservation):
        """Update with a new observation."""
        self.observations_by_type[obs.process_type].append(obs)
        
        # Keep only recent observations (last 100 per type)
        if len(self.observations_by_type[obs.process_type]) > 100:
            self.observations_by_type[obs.process_type] = \
                self.observations_by_type[obs.process_type][-100:]
    
    def get_patterns(self) -> List[CognitivePattern]:
        """Get all identified patterns."""
        patterns = []
        
        for process_type, observations in self.observations_by_type.items():
            if len(observations) < self.min_samples_for_pattern:
                continue
                
            patterns.extend(self._detect_success_conditions(process_type, observations))
            patterns.extend(self._detect_failure_modes(process_type, observations))
            patterns.extend(self._detect_efficiency_factors(process_type, observations))
        
        return patterns
    
    def _detect_success_conditions(self, process_type: str,
                                   observations: List[ProcessingObservation]) -> List[CognitivePattern]:
        """Find conditions associated with success."""
        successes = [o for o in observations if o.success]
        if len(successes) < self.min_samples_for_pattern:
            return []
        
        patterns = []
        failures = [o for o in observations if not o.success]
        
        # Compute averages once
        avg_success_complexity = sum(s.input_complexity for s in successes) / len(successes)
        avg_failure_complexity = sum(f.input_complexity for f in failures) / len(failures) if failures else 1.0
        
        # Check if low complexity correlates with success
        if avg_success_complexity < avg_failure_complexity * self.SUCCESS_COMPLEXITY_THRESHOLD:
            patterns.append(CognitivePattern(
                pattern_type='success_condition',
                description=f"{process_type} succeeds more often on simpler inputs",
                confidence=min(0.9, len(successes) / 10),
                supporting_observations=[s.id for s in successes],
                actionable=True,
                suggested_adaptation="Prioritize simpler inputs or break complex ones into chunks"
            ))
        
        # Check if high quality output indicates good processing
        high_quality_successes = [s for s in successes if s.output_quality > self.HIGH_QUALITY_THRESHOLD]
        if len(high_quality_successes) > len(successes) * self.HIGH_QUALITY_FREQUENCY:
            patterns.append(CognitivePattern(
                pattern_type='success_condition',
                description=f"{process_type} consistently produces high-quality output when successful",
                confidence=min(0.9, len(high_quality_successes) / 10),
                supporting_observations=[s.id for s in high_quality_successes],
                actionable=False,
                suggested_adaptation=None
            ))
        
        return patterns
    
    def _detect_failure_modes(self, process_type: str,
                             observations: List[ProcessingObservation]) -> List[CognitivePattern]:
        """Find conditions associated with failure."""
        failures = [o for o in observations if not o.success]
        if len(failures) < self.min_samples_for_pattern:
            return []
        
        patterns = []
        
        # Check if high complexity correlates with failure
        avg_failure_complexity = sum(f.input_complexity for f in failures) / len(failures)
        successes = [o for o in observations if o.success]
        avg_success_complexity = sum(o.input_complexity for o in successes) / max(1, len(successes))
        
        if avg_failure_complexity > avg_success_complexity * self.FAILURE_COMPLEXITY_THRESHOLD:
            patterns.append(CognitivePattern(
                pattern_type='failure_mode',
                description=f"{process_type} tends to fail on high-complexity inputs",
                confidence=min(0.9, len(failures) / 10),
                supporting_observations=[f.id for f in failures],
                actionable=True,
                suggested_adaptation="Break complex inputs into smaller chunks"
            ))
        
        # Check for timeout/long duration failures
        long_failures = [f for f in failures if f.duration_ms > 1000]
        if len(long_failures) > len(failures) * 0.5:
            patterns.append(CognitivePattern(
                pattern_type='failure_mode',
                description=f"{process_type} failures often involve long processing times",
                confidence=min(0.9, len(long_failures) / 10),
                supporting_observations=[f.id for f in long_failures],
                actionable=True,
                suggested_adaptation="Implement timeout and early-exit strategies"
            ))
        
        return patterns
    
    def _detect_efficiency_factors(self, process_type: str,
                                   observations: List[ProcessingObservation]) -> List[CognitivePattern]:
        """Find factors affecting efficiency."""
        if len(observations) < self.min_samples_for_pattern:
            return []
        
        patterns = []
        
        # Check average duration
        avg_duration = sum(o.duration_ms for o in observations) / len(observations)
        if avg_duration > self.SLOW_PROCESS_MS:
            patterns.append(CognitivePattern(
                pattern_type='efficiency_factor',
                description=f"{process_type} is relatively slow (avg {avg_duration:.0f}ms)",
                confidence=0.8,
                supporting_observations=[o.id for o in observations],
                actionable=True,
                suggested_adaptation="Consider optimization or caching strategies"
            ))
        
        # Check for resource usage patterns
        high_resource_obs = [o for o in observations 
                            if o.resources_used.attention_units > self.HIGH_RESOURCE_ATTENTION 
                            or o.resources_used.memory_accesses > self.HIGH_RESOURCE_MEMORY]
        if len(high_resource_obs) > len(observations) * self.HIGH_RESOURCE_FREQUENCY:
            patterns.append(CognitivePattern(
                pattern_type='efficiency_factor',
                description=f"{process_type} uses significant resources",
                confidence=min(0.9, len(high_resource_obs) / 10),
                supporting_observations=[o.id for o in high_resource_obs],
                actionable=True,
                suggested_adaptation="Monitor resource allocation and consider limiting"
            ))
        
        return patterns


class MetaCognitiveMonitor:
    """Monitors cognitive processing and identifies patterns."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.observations: List[ProcessingObservation] = []
        self.pattern_detector = PatternDetector()
        self.config = config or {}
        self.stats_by_type: Dict[str, ProcessStats] = {}
        self.max_observations = self.config.get("max_observations", 1000)
        
        logger.info("✅ MetaCognitiveMonitor initialized")
    
    def observe(self, process_type: str) -> ProcessingContext:
        """Context manager to observe a cognitive process."""
        return ProcessingContext(self, process_type)
    
    def record_observation(self, obs: ProcessingObservation):
        """Record a processing observation."""
        self.observations.append(obs)
        
        # Limit memory usage
        if len(self.observations) > self.max_observations:
            self.observations = self.observations[-self.max_observations:]
        
        self._update_statistics(obs)
        self.pattern_detector.update(obs)
        
        logger.debug(f"📊 Recorded observation: {obs.process_type} "
                    f"(success={obs.success}, duration={obs.duration_ms:.1f}ms)")
    
    def _update_statistics(self, obs: ProcessingObservation):
        """Update statistics for a process type."""
        process_type = obs.process_type
        relevant = [o for o in self.observations if o.process_type == process_type]
        
        if not relevant:
            return
        
        self.stats_by_type[process_type] = ProcessStats(
            total_executions=len(relevant),
            success_rate=sum(1 for o in relevant if o.success) / len(relevant),
            avg_duration_ms=sum(o.duration_ms for o in relevant) / len(relevant),
            avg_quality=sum(o.output_quality for o in relevant) / len(relevant)
        )
    
    def get_process_statistics(self, process_type: str) -> ProcessStats:
        """Get statistics for a process type."""
        if process_type in self.stats_by_type:
            return self.stats_by_type[process_type]
        
        # Compute on demand if not cached
        relevant = [o for o in self.observations if o.process_type == process_type]
        if not relevant:
            return ProcessStats(
                total_executions=0,
                success_rate=0.0,
                avg_duration_ms=0.0,
                avg_quality=0.0
            )
        
        return ProcessStats(
            total_executions=len(relevant),
            success_rate=sum(1 for o in relevant if o.success) / len(relevant),
            avg_duration_ms=sum(o.duration_ms for o in relevant) / len(relevant),
            avg_quality=sum(o.output_quality for o in relevant) / len(relevant)
        )
    
    def get_identified_patterns(self) -> List[CognitivePattern]:
        """Get patterns identified in processing."""
        return self.pattern_detector.get_patterns()
    
    def get_all_statistics(self) -> Dict[str, ProcessStats]:
        """Get statistics for all process types."""
        return self.stats_by_type.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of monitoring data."""
        patterns = self.get_identified_patterns()
        return {
            "total_observations": len(self.observations),
            "process_types": list(self.stats_by_type.keys()),
            "patterns_identified": len(patterns),
            "actionable_patterns": sum(1 for p in patterns if p.actionable),
            "statistics": {pt: {
                "executions": stats.total_executions,
                "success_rate": stats.success_rate,
                "avg_duration_ms": stats.avg_duration_ms,
                "avg_quality": stats.avg_quality
            } for pt, stats in self.stats_by_type.items()}
        }
