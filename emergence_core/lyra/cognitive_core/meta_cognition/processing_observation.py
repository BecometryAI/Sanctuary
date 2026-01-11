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
    
    Records metrics about a single cognitive process execution, including
    duration, success, complexity, quality, and resource usage. These
    observations enable pattern detection and meta-cognitive learning.
    
    Attributes:
        id: Unique identifier for this observation
        timestamp: When the observation was recorded
        process_type: Type of process ('reasoning', 'memory_retrieval', 'goal_selection', etc.)
        duration_ms: How long the process took in milliseconds
        success: Whether the process completed successfully
        input_complexity: Subjective complexity of the input (0.0-1.0)
        output_quality: Self-assessed quality of output (0.0-1.0)
        resources_used: Cognitive resources consumed
        error: Error message if process failed
        metadata: Additional process-specific information
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


@dataclass
class ProcessStats:
    """
    Statistical summary of a process type.
    
    Aggregates observations to provide overall performance metrics
    for a specific type of cognitive process.
    
    Attributes:
        total_executions: Number of times this process has run
        success_rate: Fraction of executions that succeeded (0.0-1.0)
        avg_duration_ms: Average execution time in milliseconds
        avg_quality: Average self-assessed output quality (0.0-1.0)
    """
    total_executions: int
    success_rate: float
    avg_duration_ms: float
    avg_quality: float


class ProcessingContext:
    """
    Context manager for observing a cognitive process.
    
    Usage:
        with monitor.observe('reasoning') as ctx:
            ctx.input_complexity = 0.7
            # ... do cognitive work ...
            ctx.output_quality = 0.8
            ctx.resources = CognitiveResources(attention_budget=0.5)
    
    The context manager automatically records timing, success/failure,
    and creates a ProcessingObservation when the context exits.
    """
    
    def __init__(self, monitor: 'MetaCognitiveMonitor', process_type: str):
        """
        Initialize processing context.
        
        Args:
            monitor: MetaCognitiveMonitor that will receive the observation
            process_type: Type of process being observed
        """
        self.monitor = monitor
        self.process_type = process_type
        self.start_time: Optional[float] = None
        self.observation: Optional[ProcessingObservation] = None
        
        # Default values that can be overridden
        self.input_complexity: float = 0.5
        self.output_quality: float = 0.5
        self.resources: CognitiveResources = CognitiveResources()
        self.metadata: Dict[str, Any] = {}
    
    def __enter__(self):
        """Start timing the cognitive process."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Record observation when process completes.
        
        Args:
            exc_type: Exception type if process failed
            exc_val: Exception value if process failed
            exc_tb: Exception traceback if process failed
        """
        duration = (time.time() - self.start_time) * 1000  # Convert to ms
        
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
        
        # Don't suppress exceptions
        return False


def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())
