"""
Processing Observation: Track cognitive processing episodes.

This module defines data structures for observing and recording cognitive
processing patterns, enabling meta-cognitive monitoring of system performance.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any

from ..goals.resources import CognitiveResources


@dataclass
class ProcessingObservation:
    """
    Observation of a cognitive processing episode.
    
    Attributes:
        id: Unique identifier
        timestamp: Recording time
        process_type: Process category
        duration_ms: Execution time in milliseconds
        success: True if completed without error
        input_complexity: Input complexity (0.0-1.0)
        output_quality: Output quality (0.0-1.0)
        resources_used: Consumed resources
        error: Error message if failed
        metadata: Additional data
    """
    id: str
    timestamp: datetime
    process_type: str
    duration_ms: float
    success: bool
    input_complexity: float
    output_quality: float
    resources_used: CognitiveResources
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate observation data."""
        if not 0 <= self.input_complexity <= 1:
            raise ValueError(f"input_complexity must be 0-1, got {self.input_complexity}")
        if not 0 <= self.output_quality <= 1:
            raise ValueError(f"output_quality must be 0-1, got {self.output_quality}")
        if self.duration_ms < 0:
            raise ValueError(f"duration_ms cannot be negative, got {self.duration_ms}")


@dataclass
class ProcessStats:
    """
    Statistical summary of a process type.
    
    Attributes:
        total_executions: Execution count
        success_rate: Success fraction (0.0-1.0)
        avg_duration_ms: Average time in ms
        avg_quality: Average quality (0.0-1.0)
    """
    total_executions: int
    success_rate: float
    avg_duration_ms: float
    avg_quality: float
    
    def __post_init__(self):
        """Validate statistics."""
        if self.total_executions < 0:
            raise ValueError(f"total_executions cannot be negative")
        if not 0 <= self.success_rate <= 1:
            raise ValueError(f"success_rate must be 0-1, got {self.success_rate}")
        if self.avg_duration_ms < 0:
            raise ValueError(f"avg_duration_ms cannot be negative")


class ProcessingContext:
    """Context manager for observing a cognitive process."""
    
    def __init__(self, monitor: 'MetaCognitiveMonitor', process_type: str):
        """
        Initialize processing context.
        
        Args:
            monitor: MetaCognitiveMonitor instance
            process_type: Process category
        """
        self.monitor = monitor
        self.process_type = process_type
        self.start_time: Optional[float] = None
        self.observation: Optional[ProcessingObservation] = None
        
        # Defaults (can be overridden)
        self._input_complexity: float = 0.5
        self._output_quality: float = 0.5
        self.resources: CognitiveResources = CognitiveResources()
        self.metadata: Dict[str, Any] = {}
    
    @property
    def input_complexity(self) -> float:
        return self._input_complexity
    
    @input_complexity.setter
    def input_complexity(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"input_complexity must be 0-1, got {value}")
        self._input_complexity = value
    
    @property
    def output_quality(self) -> float:
        return self._output_quality
    
    @output_quality.setter
    def output_quality(self, value: float):
        if not 0 <= value <= 1:
            raise ValueError(f"output_quality must be 0-1, got {value}")
        self._output_quality = value
    
    def __enter__(self):
        """Start timing the cognitive process."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Record observation when process completes."""
        if self.start_time is None:
            return False  # __enter__ was never called
            
        duration = (time.time() - self.start_time) * 1000
        
        self.observation = ProcessingObservation(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            process_type=self.process_type,
            duration_ms=duration,
            success=exc_type is None,
            input_complexity=self._input_complexity,
            output_quality=self._output_quality,
            resources_used=self.resources,
            error=str(exc_val) if exc_val else None,
            metadata=self.metadata
        )
        
        self.monitor.record_observation(self.observation)
        return False  # Don't suppress exceptions


def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())
