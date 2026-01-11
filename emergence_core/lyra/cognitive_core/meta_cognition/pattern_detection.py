"""
Pattern Detection: Identify patterns in cognitive processing.

This module analyzes processing observations to detect success conditions,
failure modes, and efficiency factors that can inform adaptive strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import defaultdict

from .processing_observation import ProcessingObservation

logger = logging.getLogger(__name__)


@dataclass
class CognitivePattern:
    """
    A pattern identified in cognitive processing.
    
    Represents a discovered relationship between context/conditions and
    outcomes that can inform future processing strategies.
    
    Attributes:
        pattern_type: Category ('success_condition', 'failure_mode', 'efficiency_factor')
        description: Human-readable description of the pattern
        confidence: How confident we are in this pattern (0.0-1.0)
        supporting_observations: IDs of observations that support this pattern
        actionable: Whether we can do something about this pattern
        suggested_adaptation: Optional recommendation for adapting behavior
    """
    pattern_type: str
    description: str
    confidence: float
    supporting_observations: List[str]
    actionable: bool
    suggested_adaptation: Optional[str] = None


class PatternDetector:
    """
    Detects patterns in cognitive processing observations.
    
    Analyzes processing history to identify:
    - Success conditions: What makes processes succeed
    - Failure modes: Common failure patterns
    - Efficiency factors: What makes processes faster/better
    """
    
    # Pattern detection thresholds
    MODERATE_COMPLEXITY_LOW = 0.3
    MODERATE_COMPLEXITY_HIGH = 0.7
    HIGH_COMPLEXITY_FAILURE_RATIO = 1.3
    
    def __init__(self, min_observations: int = 3):
        """
        Initialize pattern detector.
        
        Args:
            min_observations: Minimum observations needed to detect a pattern
        """
        self.observations_by_type: Dict[str, List[ProcessingObservation]] = defaultdict(list)
        self.min_observations = min_observations
        self._cached_patterns: Optional[List[CognitivePattern]] = None
        self._cache_dirty = False
    
    def update(self, obs: ProcessingObservation):
        """
        Add a new observation to analyze.
        
        Args:
            obs: Processing observation to add
        """
        self.observations_by_type[obs.process_type].append(obs)
        self._cache_dirty = True
        
        # Keep only recent observations to prevent unbounded growth
        max_per_type = 1000
        if len(self.observations_by_type[obs.process_type]) > max_per_type:
            self.observations_by_type[obs.process_type] = \
                self.observations_by_type[obs.process_type][-max_per_type:]
    
    def get_patterns(self) -> List[CognitivePattern]:
        """
        Get all detected patterns.
        
        Returns:
            List of cognitive patterns found across all process types
        """
        # Use cached patterns if available
        if not self._cache_dirty and self._cached_patterns is not None:
            return self._cached_patterns
        
        patterns = []
        
        for process_type, observations in self.observations_by_type.items():
            if len(observations) < self.min_observations:
                continue
            
            patterns.extend(self._detect_success_conditions(process_type, observations))
            patterns.extend(self._detect_failure_modes(process_type, observations))
            patterns.extend(self._detect_efficiency_factors(process_type, observations))
        
        self._cached_patterns = patterns
        self._cache_dirty = False
        
        return patterns
    
    def _detect_success_conditions(
        self,
        process_type: str,
        observations: List[ProcessingObservation]
    ) -> List[CognitivePattern]:
        """
        Find conditions associated with success.
        
        Args:
            process_type: Type of process to analyze
            observations: Observations of this process type
            
        Returns:
            List of detected success condition patterns
        """
        successes = [o for o in observations if o.success]
        failures = [o for o in observations if not o.success]
        
        if len(successes) < self.min_observations:
            return []
        
        patterns = []
        
        # Check if moderate complexity correlates with success
        if successes and failures:
            avg_success_complexity = sum(s.input_complexity for s in successes) / len(successes)
            avg_failure_complexity = sum(f.input_complexity for f in failures) / len(failures)
            
            # Success on moderate complexity inputs
            if (self.MODERATE_COMPLEXITY_LOW <= avg_success_complexity <= self.MODERATE_COMPLEXITY_HIGH 
                and avg_failure_complexity > self.MODERATE_COMPLEXITY_HIGH):
                patterns.append(CognitivePattern(
                    pattern_type='success_condition',
                    description=f"{process_type} succeeds best with moderate complexity inputs",
                    confidence=min(0.9, len(successes) / 10),
                    supporting_observations=[s.id for s in successes[:10]],
                    actionable=True,
                    suggested_adaptation="Prefer moderate complexity inputs when possible"
                ))
        
        # Check if high quality outputs correlate with sufficient resources
        high_quality = [s for s in successes if s.output_quality > 0.7]
        if len(high_quality) >= self.min_observations:
            avg_resources = sum(
                hq.resources_used.total() for hq in high_quality
            ) / len(high_quality)
            
            if avg_resources > 2.0:  # Above average resource usage
                patterns.append(CognitivePattern(
                    pattern_type='success_condition',
                    description=f"{process_type} produces high quality with adequate resources",
                    confidence=min(0.8, len(high_quality) / 10),
                    supporting_observations=[hq.id for hq in high_quality[:10]],
                    actionable=True,
                    suggested_adaptation="Allocate more resources for quality-critical tasks"
                ))
        
        return patterns
    
    def _detect_failure_modes(
        self,
        process_type: str,
        observations: List[ProcessingObservation]
    ) -> List[CognitivePattern]:
        """
        Find conditions associated with failure.
        
        Args:
            process_type: Type of process to analyze
            observations: Observations of this process type
            
        Returns:
            List of detected failure mode patterns
        """
        failures = [o for o in observations if not o.success]
        
        if len(failures) < self.min_observations:
            return []
        
        patterns = []
        successes = [o for o in observations if o.success]
        
        # Check if high complexity correlates with failure
        if failures and successes:
            avg_failure_complexity = sum(f.input_complexity for f in failures) / len(failures)
            avg_success_complexity = sum(
                s.input_complexity for s in successes
            ) / len(successes) if successes else 0.5
            
            if avg_failure_complexity > avg_success_complexity * self.HIGH_COMPLEXITY_FAILURE_RATIO:
                patterns.append(CognitivePattern(
                    pattern_type='failure_mode',
                    description=f"{process_type} tends to fail on high-complexity inputs",
                    confidence=min(0.9, len(failures) / 10),
                    supporting_observations=[f.id for f in failures[:10]],
                    actionable=True,
                    suggested_adaptation="Break complex inputs into smaller chunks"
                ))
        
        # Check if resource starvation correlates with failure
        low_resource_failures = [
            f for f in failures if f.resources_used.total() < 1.0
        ]
        if len(low_resource_failures) >= self.min_observations:
            ratio = len(low_resource_failures) / len(failures)
            if ratio > 0.6:
                patterns.append(CognitivePattern(
                    pattern_type='failure_mode',
                    description=f"{process_type} often fails with insufficient resources",
                    confidence=min(0.85, len(low_resource_failures) / 10),
                    supporting_observations=[f.id for f in low_resource_failures[:10]],
                    actionable=True,
                    suggested_adaptation="Ensure adequate resource allocation before execution"
                ))
        
        return patterns
    
    def _detect_efficiency_factors(
        self,
        process_type: str,
        observations: List[ProcessingObservation]
    ) -> List[CognitivePattern]:
        """
        Find factors that affect processing efficiency.
        
        Args:
            process_type: Type of process to analyze
            observations: Observations of this process type
            
        Returns:
            List of detected efficiency factor patterns
        """
        if len(observations) < self.min_observations:
            return []
        
        patterns = []
        successes = [o for o in observations if o.success]
        
        if len(successes) < self.min_observations:
            return []
        
        # Analyze duration patterns
        avg_duration = sum(s.duration_ms for s in successes) / len(successes)
        fast_successes = [s for s in successes if s.duration_ms < avg_duration * 0.7]
        slow_successes = [s for s in successes if s.duration_ms > avg_duration * 1.5]
        
        # Check if lower complexity leads to faster processing
        if len(fast_successes) >= self.min_observations and len(slow_successes) >= self.min_observations:
            avg_fast_complexity = sum(
                fs.input_complexity for fs in fast_successes
            ) / len(fast_successes)
            avg_slow_complexity = sum(
                ss.input_complexity for ss in slow_successes
            ) / len(slow_successes)
            
            if avg_fast_complexity < avg_slow_complexity * 0.8:
                patterns.append(CognitivePattern(
                    pattern_type='efficiency_factor',
                    description=f"{process_type} processes faster with lower complexity",
                    confidence=min(0.8, len(fast_successes) / 10),
                    supporting_observations=[fs.id for fs in fast_successes[:10]],
                    actionable=True,
                    suggested_adaptation="Simplify inputs when speed is important"
                ))
        
        # Check quality vs resource trade-off
        high_quality = [s for s in successes if s.output_quality > 0.75]
        if len(high_quality) >= self.min_observations:
            avg_hq_resources = sum(
                hq.resources_used.total() for hq in high_quality
            ) / len(high_quality)
            avg_resources = sum(
                s.resources_used.total() for s in successes
            ) / len(successes)
            
            if avg_hq_resources > avg_resources * 1.2:
                patterns.append(CognitivePattern(
                    pattern_type='efficiency_factor',
                    description=f"{process_type} quality improves with resource investment",
                    confidence=min(0.75, len(high_quality) / 10),
                    supporting_observations=[hq.id for hq in high_quality[:10]],
                    actionable=True,
                    suggested_adaptation="Invest more resources for quality-critical outputs"
                ))
        
        return patterns
    
    def get_patterns_for_type(self, process_type: str) -> List[CognitivePattern]:
        """
        Get patterns for a specific process type.
        
        Args:
            process_type: Type of process to get patterns for
            
        Returns:
            List of patterns related to this process type
        """
        all_patterns = self.get_patterns()
        return [p for p in all_patterns if process_type in p.description]
    
    def clear_cache(self):
        """Clear the pattern cache, forcing recomputation on next access."""
        self._cached_patterns = None
        self._cache_dirty = True
