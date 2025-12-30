"""
Meta-Cognition: Self-monitoring and introspection.

This module implements the SelfMonitor class, which observes and reports on internal
cognitive state. It generates introspective percepts that allow the system to reason
about its own processing, creating a foundation for meta-cognitive awareness.

The meta-cognition subsystem is responsible for:
- Monitoring internal cognitive processes and states
- Detecting anomalies or inefficiencies in processing
- Generating introspective reports for the workspace
- Supporting higher-order reasoning about cognition
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MonitoringLevel(Enum):
    """
    Granularity of self-monitoring.

    MINIMAL: Only critical state and errors
    NORMAL: Standard operational monitoring
    DETAILED: Fine-grained tracking of all subsystems
    INTROSPECTIVE: Deep analysis including reasoning traces
    """
    MINIMAL = "minimal"
    NORMAL = "normal"
    DETAILED = "detailed"
    INTROSPECTIVE = "introspective"


@dataclass
class IntrospectiveReport:
    """
    Report on internal cognitive state and processing.

    An introspective report is a structured observation of the system's own
    cognitive processes. These reports can be treated as percepts that enter
    the GlobalWorkspace, allowing the system to reason about itself.

    Attributes:
        timestamp: When the report was generated
        monitoring_level: Granularity of monitoring that produced this report
        subsystem_states: Status of each cognitive subsystem
        processing_metrics: Performance and efficiency measurements
        anomalies: Detected issues or unexpected patterns
        cognitive_load: Current resource utilization (0.0-1.0)
        coherence_score: Measure of internal consistency (0.0-1.0)
        insights: High-level observations about cognitive state
        metadata: Additional contextual information
    """
    timestamp: datetime
    monitoring_level: MonitoringLevel
    subsystem_states: Dict[str, str]
    processing_metrics: Dict[str, float]
    anomalies: List[str]
    cognitive_load: float
    coherence_score: float
    insights: List[str]
    metadata: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class SelfMonitor:
    """
    Observes and reports on internal cognitive state.

    The SelfMonitor implements meta-cognition by treating the cognitive system
    itself as an object of observation. It generates introspective percepts that
    can enter the GlobalWorkspace, enabling the system to reason about its own
    processing and maintain self-awareness.

    Key Responsibilities:
    - Monitor all cognitive subsystems (workspace, attention, perception, etc.)
    - Track processing metrics (speed, efficiency, resource usage)
    - Detect anomalies and inefficiencies in cognitive processing
    - Generate introspective reports for workspace inclusion
    - Maintain meta-cognitive history for pattern analysis
    - Support debugging and transparency through observability

    Integration Points:
    - GlobalWorkspace: Introspective reports can enter as special percepts
    - AttentionController: Monitor attention allocation patterns
    - PerceptionSubsystem: Observe perceptual processing and confidence
    - ActionSubsystem: Track action selection and outcomes
    - AffectSubsystem: Monitor emotional dynamics and regulation
    - CognitiveCore: Reports are generated in the main cognitive loop

    Meta-Cognitive Capabilities:
    1. Process Monitoring: Track what the system is doing
       - Which subsystems are active
       - What information is in the workspace
       - What actions are being considered

    2. Performance Monitoring: Assess how well it's working
       - Processing speed and efficiency
       - Resource utilization and bottlenecks
       - Goal progress and achievement

    3. Anomaly Detection: Identify issues and unexpected patterns
       - Unusual attention patterns
       - Emotional instability
       - Goal conflicts or deadlocks
       - Degraded performance

    4. Introspective Reporting: Generate natural language descriptions
       - "I am currently focused on goal X"
       - "My attention keeps shifting, suggesting distraction"
       - "I notice increased emotional arousal without clear cause"

    This enables higher-order reasoning like:
    - "I'm stuck in a loop, I should try a different approach"
    - "My emotional state is biasing my perception, I should recalibrate"
    - "I'm operating at high cognitive load, I should simplify"

    Attributes:
        monitoring_level: Granularity of monitoring
        report_history: Recent introspective reports
        anomaly_thresholds: Criteria for detecting issues
    """

    def __init__(
        self,
        monitoring_level: MonitoringLevel = MonitoringLevel.NORMAL,
        history_size: int = 1000,
        anomaly_detection: bool = True,
    ) -> None:
        """
        Initialize the self-monitor.

        Args:
            monitoring_level: Granularity of self-monitoring
            history_size: Number of reports to maintain in history
            anomaly_detection: Whether to actively detect processing anomalies
        """
        # Placeholder implementation - will be fully implemented in Phase 2
        self.monitoring_level = monitoring_level
        self.history_size = history_size
        self.anomaly_detection = anomaly_detection
        self.report_history: List[IntrospectiveReport] = []
    
    def observe(self, snapshot: Any) -> List[Any]:
        """
        Placeholder: will be implemented in Phase 2.
        
        Observes workspace state and generates introspective percepts.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            List of meta-cognitive percepts (empty for now)
        """
        return []  # No meta-percepts yet
