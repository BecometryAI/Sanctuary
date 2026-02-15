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

import logging
from typing import Optional, Dict, Any, List
from collections import deque
from pathlib import Path

from ..workspace import GlobalWorkspace, WorkspaceSnapshot, Percept

# Import shared data classes
from ._shared import PredictionRecord, AccuracySnapshot

# Import component modules
from .introspection import IntrospectiveJournal
from .monitor import Monitor
from .conflict_detector import ConflictDetector
from .confidence_estimator import ConfidenceEstimator
from .regulator import Regulator
from .metrics import MetricsReporter

logger = logging.getLogger(__name__)


class SelfMonitor:
    """
    Observes and reports on internal cognitive state.

    The SelfMonitor implements meta-cognition by treating the cognitive system
    itself as an object of observation. It generates introspective percepts that
    can enter the GlobalWorkspace, enabling the system to reason about its own
    processing and maintain self-awareness.

    Key Responsibilities:
    - Monitor cognitive state and generate self-reflective percepts
    - Check value alignment against charter principles
    - Assess performance and detect inefficiencies
    - Identify uncertainty and ambiguous states
    - Observe emotional trajectory and patterns
    - Detect behavioral patterns and loops
    
    This is a facade class that composes specialized components for different
    aspects of meta-cognition.
    """

    def __init__(
        self,
        workspace: Optional[GlobalWorkspace] = None,
        config: Optional[Dict] = None,
        identity: Optional[Any] = None,
        identity_manager: Optional[Any] = None
    ):
        """
        Initialize the self-monitor.

        Args:
            workspace: GlobalWorkspace instance to observe
            config: Optional configuration dict
            identity: Optional IdentityLoader instance with charter and protocols
            identity_manager: Optional IdentityManager for computed identity
        """
        self.workspace = workspace
        self.config = config or {}
        self.identity = identity
        self.identity_manager = identity_manager
        
        # Load identity files
        charter_text = ""
        protocols_text = ""
        if self.identity and self.identity.charter:
            charter_text = self.identity.charter.full_text
        else:
            charter_text = Monitor.load_charter()
            
        if self.identity and self.identity.protocols:
            protocols_text = Monitor.format_protocols(self.identity.protocols)
        else:
            protocols_text = Monitor.load_protocols()
        
        # Initialize self-model
        self.self_model = {
            "capabilities": {},
            "limitations": {},
            "preferences": {},
            "behavioral_traits": {},
            "values_hierarchy": []
        }
        
        # Shared tracking structures
        self.prediction_history = deque(maxlen=500)
        self.behavioral_log = deque(maxlen=1000)
        self.observation_history = deque(maxlen=100)
        
        # Stats
        self.stats = {
            "total_observations": 0,
            "value_conflicts": 0,
            "performance_issues": 0,
            "uncertainty_detections": 0,
            "emotional_observations": 0,
            "pattern_detections": 0,
            "self_model_updates": 0,
            "predictions_made": 0,
            "behavioral_inconsistencies": 0,
            "predictions_validated": 0,
            "accuracy_snapshots_taken": 0,
            "self_model_refinements": 0
        }
        
        # Initialize component modules
        self.monitor = Monitor(
            workspace=workspace,
            charter_text=charter_text,
            protocols_text=protocols_text,
            identity=identity,
            config=config
        )
        
        self.conflict_detector = ConflictDetector(
            self_model=self.self_model,
            behavioral_log=self.behavioral_log,
            config=config
        )
        
        self.confidence_estimator = ConfidenceEstimator(
            self_model=self.self_model,
            prediction_history=self.prediction_history,
            config=config
        )
        
        self.regulator = Regulator(
            self_model=self.self_model,
            prediction_records=self.confidence_estimator.prediction_records,
            config=config
        )
        
        self.metrics_reporter = MetricsReporter(
            workspace=workspace,
            stats=self.stats,
            observation_history=self.observation_history,
            prediction_records=self.confidence_estimator.prediction_records,
            pending_validations=self.confidence_estimator.pending_validations,
            accuracy_by_category=self.confidence_estimator.accuracy_by_category,
            calibration_bins=self.confidence_estimator.calibration_bins,
            self_model_version=self.regulator.self_model_version,
            config=config
        )
        
        # Sync stats references
        self._sync_stats()
        
        # Tracking from config
        self.monitoring_frequency = self.config.get("monitoring_frequency", 10)
        self.cycle_count = 0
        self.self_model_version = 0

        # Config-driven thresholds (exposed for test access)
        self.self_model_update_frequency = self.config.get("self_model_update_frequency", 10)
        self.prediction_confidence_threshold = self.config.get("prediction_confidence_threshold", 0.6)
        self.inconsistency_severity_threshold = self.config.get("inconsistency_severity_threshold", 0.5)
        self._update_call_count = 0

        # Feature flags
        self.enable_existential_questions = self.config.get("enable_existential_questions", True)
        self.enable_capability_tracking = self.config.get("enable_capability_tracking", True)
        self.enable_value_alignment_tracking = self.config.get("enable_value_alignment_tracking", True)

        # Identity text (exposed for backward compatibility)
        self.charter_text = charter_text
        self.protocols_text = protocols_text

        # Daily snapshots tracking
        self.daily_snapshots: Dict[str, Any] = {}

        logger.info("✅ SelfMonitor initialized with modular architecture")

    # Properties to expose internal component attributes
    @property
    def prediction_records(self) -> Dict[str, PredictionRecord]:
        """Access prediction records from confidence estimator."""
        return self.confidence_estimator.prediction_records

    @property
    def accuracy_history(self) -> deque:
        """Access accuracy history from metrics reporter."""
        return self.metrics_reporter.accuracy_history

    @property
    def pending_validations(self) -> deque:
        """Access pending validations from confidence estimator."""
        return self.confidence_estimator.pending_validations

    @property
    def accuracy_by_category(self) -> Dict[str, Any]:
        """Access accuracy by category from confidence estimator."""
        return self.confidence_estimator.accuracy_by_category

    @property
    def calibration_bins(self) -> Dict[str, Any]:
        """Access calibration bins from confidence estimator."""
        return self.confidence_estimator.calibration_bins

    def _sync_stats(self):
        """Sync stats between components."""
        # Monitor stats
        self.monitor.stats = self.stats
        # Merge regulator stats into shared stats
        for key, value in self.regulator.stats.items():
            if key in self.stats:
                self.stats[key] = value
        # Merge conflict detector stats
        for key, value in self.conflict_detector.stats.items():
            if key in self.stats:
                self.stats[key] = value

    # Backward-compatible delegation to Monitor private methods
    def _check_value_alignment(self, snapshot: WorkspaceSnapshot):
        """Delegate to Monitor component."""
        return self.monitor._check_value_alignment(snapshot)

    def _assess_performance(self, snapshot: WorkspaceSnapshot):
        """Delegate to Monitor component."""
        return self.monitor._assess_performance(snapshot)

    def _detect_uncertainty(self, snapshot: WorkspaceSnapshot):
        """Delegate to Monitor component."""
        return self.monitor._detect_uncertainty(snapshot)

    def _detect_goal_conflicts(self, goals):
        """Delegate to Monitor component."""
        return self.monitor._detect_goal_conflicts(goals)

    def _observe_emotions(self, snapshot: WorkspaceSnapshot):
        """Delegate to Monitor component."""
        return self.monitor._observe_emotions(snapshot)

    def _detect_patterns(self, snapshot: WorkspaceSnapshot):
        """Delegate to Monitor component."""
        return self.monitor._detect_patterns(snapshot)

    def _verify_capability(self, action_type) -> bool:
        """Delegate to ConflictDetector component."""
        return self.conflict_detector._verify_capability(action_type)
    
    def observe(self, snapshot: WorkspaceSnapshot) -> List[Percept]:
        """
        Generate meta-cognitive percepts.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            List of meta-cognitive percepts
        """
        self.cycle_count += 1
        self.monitor.cycle_count = self.cycle_count
        
        # Delegate to monitor component
        percepts = self.monitor.observe(snapshot)
        
        # Update shared tracking
        self.observation_history.extend(percepts)
        self._sync_stats()
        
        return percepts
    
    # Prediction and confidence methods
    def predict_behavior(self, hypothetical_state: WorkspaceSnapshot) -> Dict[str, Any]:
        """Predict what I would do in a given state."""
        result = self.confidence_estimator.predict_behavior(hypothetical_state)
        self.stats["predictions_made"] += 1
        return result
    
    def record_prediction(
        self,
        category: str,
        predicted_state: Dict[str, Any],
        confidence: float,
        context: Dict[str, Any]
    ) -> str:
        """Record a new prediction for future validation."""
        prediction_id = self.confidence_estimator.record_prediction(
            category, predicted_state, confidence, context
        )
        self._sync_stats()
        return prediction_id
    
    def validate_prediction(
        self,
        prediction_id: str,
        actual_state: Dict[str, Any]
    ) -> Optional[PredictionRecord]:
        """Validate a prediction against actual outcome."""
        result = self.confidence_estimator.validate_prediction(prediction_id, actual_state)
        self._sync_stats()
        return result
    
    def auto_validate_predictions(self, snapshot: WorkspaceSnapshot) -> List[PredictionRecord]:
        """Automatically validate pending predictions."""
        return self.confidence_estimator.auto_validate_predictions(snapshot)
    
    def measure_prediction_accuracy(self) -> Dict[str, float]:
        """Calculate accuracy of recent self-predictions."""
        return self.confidence_estimator.measure_prediction_accuracy()
    
    def calculate_confidence_calibration(self) -> Dict[str, Any]:
        """Analyze confidence calibration quality."""
        return self.confidence_estimator.calculate_confidence_calibration()
    
    def detect_systematic_biases(self) -> Dict[str, Any]:
        """Identify systematic prediction errors."""
        return self.confidence_estimator.detect_systematic_biases()
    
    # Regulation methods
    def update_self_model(self, snapshot: WorkspaceSnapshot, actual_outcome: Dict) -> None:
        """Update internal self-model based on observed behavior."""
        self._update_call_count += 1

        # Always log the behavior
        self.behavioral_log.append({
            "snapshot": {
                "emotions": snapshot.emotions if isinstance(snapshot.emotions, dict)
                    else {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                "goal_count": len(snapshot.goals),
            },
            "outcome": actual_outcome,
        })

        # Respect update frequency gating for capability/limitation updates
        if self._update_call_count % self.self_model_update_frequency == 0:
            self.regulator.update_self_model(snapshot, actual_outcome)

        self._sync_stats()
    
    def refine_self_model_from_errors(self, prediction_records: List[PredictionRecord]) -> None:
        """Automatically refine self-model based on prediction errors."""
        self.regulator.refine_self_model_from_errors(prediction_records)
        self.self_model_version = self.regulator.self_model_version
        self._sync_stats()
    
    def adjust_capability_confidence(
        self,
        capability: str,
        prediction_error: float,
        error_context: Dict
    ) -> None:
        """Adjust confidence in a specific capability based on error."""
        self.regulator.adjust_capability_confidence(capability, prediction_error, error_context)
    
    def update_limitation_boundaries(
        self,
        capability: str,
        success: bool,
        difficulty: float,
        context: Dict
    ) -> None:
        """Update understanding of capability boundaries."""
        self.regulator.update_limitation_boundaries(capability, success, difficulty, context)
    
    def identify_capability_gaps(self) -> List[Dict[str, Any]]:
        """Identify areas where self-model needs more data."""
        return self.regulator.identify_capability_gaps()
    
    # Conflict detection methods
    def analyze_behavioral_consistency(self, snapshot: WorkspaceSnapshot) -> Optional[Percept]:
        """Check if current behavior aligns with past patterns and stated values."""
        result = self.conflict_detector.analyze_behavioral_consistency(
            snapshot,
            self.monitor._compute_embedding
        )
        self._sync_stats()
        return result

    def detect_value_action_misalignment(self, snapshot: WorkspaceSnapshot) -> List[Dict]:
        """Identify when actions don't match stated values."""
        return self.conflict_detector.detect_value_action_misalignment(snapshot)

    def assess_capability_claims(self, snapshot: WorkspaceSnapshot) -> Optional[Percept]:
        """Compare claimed capabilities with actual performance."""
        result = self.conflict_detector.assess_capability_claims(
            snapshot,
            self.monitor._compute_embedding
        )
        self._sync_stats()
        return result
    
    # Metrics and reporting methods
    def get_meta_cognitive_health(self) -> Dict[str, Any]:
        """Comprehensive meta-cognitive health report."""
        return self.metrics_reporter.get_meta_cognitive_health(
            self.measure_prediction_accuracy,
            self.self_model
        )
    
    def generate_meta_cognitive_report(self) -> str:
        """Generate human-readable meta-cognitive status report."""
        return self.metrics_reporter.generate_meta_cognitive_report(
            self.get_meta_cognitive_health
        )
    
    def generate_prediction_summary(self, prediction_records: List[PredictionRecord]) -> Dict:
        """Summarize a set of predictions."""
        return self.metrics_reporter.generate_prediction_summary(prediction_records)

    def generate_accuracy_report(self, format: str = "text") -> str:
        """
        Generate a formatted accuracy report.

        Args:
            format: Output format - "text", "markdown", or "json"

        Returns:
            Formatted report string
        """
        import json as json_module
        from datetime import datetime as dt

        metrics = self.get_accuracy_metrics()
        overall = metrics.get("overall", {})
        by_category = metrics.get("by_category", {})
        calibration = metrics.get("calibration", {})

        if format == "json":
            return json_module.dumps(metrics, indent=2, default=str)

        # Build report data
        accuracy = overall.get("accuracy", 0.0)
        total = overall.get("total_predictions", 0)
        validated = overall.get("validated_predictions", 0)
        pending = overall.get("pending_validations", 0)
        cal_score = calibration.get("calibration_score", 0.0)
        overconf = calibration.get("overconfidence", 0.0)
        underconf = calibration.get("underconfidence", 0.0)

        if format == "markdown":
            lines = [
                "# SELF-MODEL ACCURACY REPORT",
                f"Generated: {dt.now().isoformat()}",
                "",
                "## Overall Metrics",
                f"- **Accuracy**: {accuracy:.1%}",
                f"- **Total Predictions**: {total}",
                f"- **Validated**: {validated}",
                f"- **Pending**: {pending}",
                "",
                "## Calibration",
                f"- **Calibration Score**: {cal_score:.2f}",
                f"- **Overconfidence**: {overconf:.2f}",
                f"- **Underconfidence**: {underconf:.2f}",
                "",
                "## Accuracy by Category",
            ]
            for cat, data in by_category.items():
                cat_acc = data.get("accuracy", 0.0)
                cat_count = data.get("count", 0)
                lines.append(f"- **{cat}**: {cat_acc:.1%} ({cat_count} predictions)")
        else:
            # Text format
            lines = [
                "=" * 50,
                "SELF-MODEL ACCURACY REPORT",
                f"Generated: {dt.now().isoformat()}",
                "=" * 50,
                "",
                "OVERALL METRICS",
                "-" * 30,
                f"Accuracy:           {accuracy:.1%}",
                f"Total Predictions:  {total}",
                f"Validated:          {validated}",
                f"Pending:            {pending}",
                "",
                "CALIBRATION",
                "-" * 30,
                f"Calibration Score:  {cal_score:.2f}",
                f"Overconfidence:     {overconf:.2f}",
                f"Underconfidence:    {underconf:.2f}",
                "",
                "ACCURACY BY CATEGORY",
                "-" * 30,
            ]
            for cat, data in by_category.items():
                cat_acc = data.get("accuracy", 0.0)
                cat_count = data.get("count", 0)
                lines.append(f"{cat:20s} {cat_acc:.1%} ({cat_count} predictions)")

        return "\n".join(lines)

    def record_accuracy_snapshot(self) -> AccuracySnapshot:
        """Capture current accuracy state."""
        return self.metrics_reporter.record_accuracy_snapshot(
            lambda: self.get_accuracy_metrics()
        )
    
    def get_accuracy_trend(self, days: int = 7) -> Dict[str, Any]:
        """Analyze accuracy trends over time."""
        return self.metrics_reporter.get_accuracy_trend(days)
    
    def get_stats(self) -> Dict[str, Any]:
        """Return meta-cognitive statistics."""
        return self.metrics_reporter.get_stats(self.monitoring_frequency, self.cycle_count)
    
    # Placeholder for methods that need full implementation
    def get_accuracy_metrics(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get detailed accuracy metrics.
        
        NOTE: This is a simplified implementation. Full accuracy metrics with
        temporal trends and detailed calibration analysis are available through
        the individual component methods (confidence_estimator, metrics_reporter).
        
        TODO: Consider implementing _calculate_temporal_trends if detailed
        temporal analysis is needed at the facade level.
        """
        records = list(self.confidence_estimator.prediction_records.values())
        if time_window:
            from datetime import datetime
            cutoff_time = datetime.now().timestamp() - time_window
            records = [r for r in records if r.timestamp.timestamp() >= cutoff_time]
        
        validated_records = [r for r in records if r.correct is not None]
        
        if not validated_records:
            return self._empty_accuracy_metrics()
        
        # Overall accuracy
        correct_count = sum(1 for r in validated_records if r.correct)
        overall_accuracy = correct_count / len(validated_records)
        
        # Accuracy by category
        by_category = {}
        for category in self.confidence_estimator.accuracy_by_category.keys():
            cat_records = [r for r in validated_records if r.category == category]
            if cat_records:
                cat_correct = sum(1 for r in cat_records if r.correct)
                by_category[category] = {
                    "accuracy": cat_correct / len(cat_records),
                    "count": len(cat_records)
                }
            else:
                by_category[category] = {"accuracy": 0.0, "count": 0}
        
        # Simplified metrics
        calibration = self.calculate_confidence_calibration()
        
        return {
            "overall": {
                "accuracy": overall_accuracy,
                "total_predictions": len(records),
                "validated_predictions": len(validated_records),
                "pending_validations": len(self.confidence_estimator.pending_validations)
            },
            "by_category": by_category,
            "by_confidence_level": {},  # Simplified
            "calibration": calibration,
            "temporal_trends": {"trend_direction": "stable"},  # Simplified
            "error_patterns": self.detect_systematic_biases()
        }
    
    def _empty_accuracy_metrics(self) -> Dict[str, Any]:
        """Return empty accuracy metrics structure."""
        return {
            "overall": {
                "accuracy": 0.0,
                "total_predictions": 0,
                "validated_predictions": 0,
                "pending_validations": len(self.confidence_estimator.pending_validations)
            },
            "by_category": {
                "action": {"accuracy": 0.0, "count": 0},
                "emotion": {"accuracy": 0.0, "count": 0},
                "capability": {"accuracy": 0.0, "count": 0},
                "goal_priority": {"accuracy": 0.0, "count": 0},
                "value_alignment": {"accuracy": 0.0, "count": 0}
            },
            "by_confidence_level": {},
            "calibration": {
                "calibration_score": 0.0,
                "overconfidence": 0.0,
                "underconfidence": 0.0,
                "calibration_curve": []
            },
            "temporal_trends": {
                "recent_accuracy": 0.0,
                "weekly_accuracy": 0.0,
                "trend_direction": "stable"
            },
            "error_patterns": {
                "common_errors": [],
                "error_contexts": [],
                "systematic_biases": []
            }
        }
    
    def introspect_identity(self) -> str:
        """
        Generate identity description from computed identity.
        
        This method uses the computed identity system to generate a
        self-description based on actual system state rather than
        static configuration.
        
        Returns:
            Human-readable identity description
        """
        if self.identity_manager:
            # Use computed identity if available
            return self.identity_manager.introspect_identity()
        elif self.identity and self.identity.charter:
            # Fallback to charter-based identity
            charter = self.identity.charter
            lines = ["Based on my charter and protocols:"]
            
            if charter.core_values:
                values_str = ", ".join(charter.core_values)
                lines.append(f"- Core values: {values_str}")
            
            if charter.purpose_statement:
                lines.append(f"- Purpose: {charter.purpose_statement}")
            
            lines.append("- Identity source: configuration (static)")
            
            return "\n".join(lines)
        else:
            return "Identity information not available"
    
    def get_computed_identity_percept(self) -> Optional[Percept]:
        """
        Generate an introspective percept about computed identity.
        
        Returns:
            Percept containing identity introspection or None if no identity manager
        """
        if not self.identity_manager:
            return None
        
        try:
            # Get identity introspection from manager
            description = self.identity_manager.introspect_identity()
            
            # Get continuity information
            continuity_score = self.identity_manager.get_continuity_score()
            
            percept_text = f"""Meta-Cognitive Observation: Identity State

{description}

Identity Continuity Score: {continuity_score:.2f}

This identity is computed from actual system state, not loaded from configuration.
It reflects what I actually DO, not what I was TOLD to be.
"""
            
            # Create introspection percept
            from ..workspace import Percept
            return Percept(
                modality="introspection",
                raw=percept_text,
                complexity=5,
                metadata={
                    "type": "identity_introspection",
                    "source": "computed_identity"
                }
            )
        except Exception as e:
            logger.error(f"Error generating identity introspection: {e}")
            return None


# Import new meta-cognitive components
from .processing_monitor import (
    MetaCognitiveMonitor,
    ProcessingObservation,
    ProcessingContext,
    CognitivePattern,
    ProcessStats,
    CognitiveResources,
)
from .action_learning import (
    ActionOutcomeLearner,
    ActionOutcome,
    ActionReliability,
    OutcomePrediction,
    ActionModel,
)
from .attention_history import (
    AttentionHistory,
    AttentionAllocation,
    AttentionOutcome,
    AttentionPattern,
)
from .system import (
    MetaCognitiveSystem,
    SelfAssessment,
)
from .bottleneck_detector import (
    BottleneckDetector,
    BottleneckSignal,
    BottleneckState,
    BottleneckType,
)

# Export main classes for backward compatibility
__all__ = [
    # Existing classes
    "SelfMonitor",
    "IntrospectiveJournal",
    "PredictionRecord",
    "AccuracySnapshot",
    # New meta-cognitive components
    "MetaCognitiveMonitor",
    "ProcessingObservation",
    "ProcessingContext",
    "CognitivePattern",
    "ProcessStats",
    "CognitiveResources",
    "ActionOutcomeLearner",
    "ActionOutcome",
    "ActionReliability",
    "OutcomePrediction",
    "ActionModel",
    "AttentionHistory",
    "AttentionAllocation",
    "AttentionOutcome",
    "AttentionPattern",
    "MetaCognitiveSystem",
    "SelfAssessment",
    # Bottleneck detection
    "BottleneckDetector",
    "BottleneckSignal",
    "BottleneckState",
    "BottleneckType",
]
