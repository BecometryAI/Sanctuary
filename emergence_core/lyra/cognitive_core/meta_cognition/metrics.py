"""
Metrics: Performance tracking and health reporting.

This module provides performance metrics, health assessment, and reporting
functionality for the meta-cognition system.
"""

from __future__ import annotations

import logging
import json
from typing import Optional, Dict, Any, List
from collections import deque
from datetime import datetime

from ..workspace import WorkspaceSnapshot, GoalType
from ._shared import PredictionRecord, AccuracySnapshot

logger = logging.getLogger(__name__)


class MetricsReporter:
    """
    Tracks performance metrics and generates health reports.
    
    Handles:
    - Meta-cognitive health assessment
    - Accuracy metrics and reporting
    - Performance trend analysis
    - Statistics tracking
    """
    
    def __init__(
        self,
        workspace: Optional[Any] = None,
        stats: Optional[Dict] = None,
        observation_history: Optional[Any] = None,
        prediction_records: Optional[Dict[str, PredictionRecord]] = None,
        pending_validations: Optional[Any] = None,
        accuracy_by_category: Optional[Dict] = None,
        calibration_bins: Optional[Dict] = None,
        self_model_version: int = 0,
        config: Optional[Dict] = None
    ):
        """
        Initialize metrics reporter.
        
        Args:
            workspace: GlobalWorkspace reference
            stats: Statistics dictionary
            observation_history: History of observations
            prediction_records: Prediction tracking records
            pending_validations: Pending validation queue
            accuracy_by_category: Category accuracy tracking
            calibration_bins: Calibration bin data
            self_model_version: Current self-model version
            config: Optional configuration dict
        """
        self.workspace = workspace
        self.stats = stats or {}
        self.observation_history = observation_history or deque(maxlen=100)
        self.prediction_records = prediction_records or {}
        self.pending_validations = pending_validations or deque(maxlen=100)
        self.accuracy_by_category = accuracy_by_category or {}
        self.calibration_bins = calibration_bins or {}
        self.self_model_version = self_model_version
        self.config = config or {}
        
        # Temporal tracking
        self.accuracy_history: deque = deque(maxlen=1000)
        self.daily_snapshots: Dict[str, AccuracySnapshot] = {}
    
    def get_meta_cognitive_health(self, measure_accuracy_fn, self_model: Dict) -> Dict[str, Any]:
        """
        Comprehensive meta-cognitive health report.
        
        Args:
            measure_accuracy_fn: Function to measure prediction accuracy
            self_model: Self-model data structure
        
        Returns:
            Dictionary containing various health metrics (0.0-1.0 scores)
        """
        # Calculate self-model accuracy
        accuracy_metrics = measure_accuracy_fn()
        self_model_accuracy = accuracy_metrics["overall_accuracy"]
        
        # Calculate value alignment score
        value_goal_count = 0
        high_priority_value_goals = 0
        if self.workspace:
            snapshot = self.workspace.broadcast()
            value_goals = [g for g in snapshot.goals if g.type == GoalType.MAINTAIN_VALUE]
            value_goal_count = len(value_goals)
            high_priority_value_goals = sum(1 for g in value_goals 
                                           if (g.priority if hasattr(g, 'priority') else g.get('priority', 0)) > 0.7)
        
        value_alignment_score = high_priority_value_goals / value_goal_count if value_goal_count > 0 else 1.0
        
        # Calculate behavioral consistency score
        consistency_score = 1.0 - (self.stats.get("behavioral_inconsistencies", 0) / max(1, self.stats.get("total_observations", 1)))
        
        # Calculate introspective depth (based on observation variety)
        observation_types = set()
        for obs in self.observation_history:
            if hasattr(obs, 'raw') and isinstance(obs.raw, dict):
                observation_types.add(obs.raw.get("type", "unknown"))
        introspective_depth = min(1.0, len(observation_types) / 5.0)
        
        # Calculate uncertainty awareness
        uncertainty_awareness = min(1.0, self.stats.get("uncertainty_detections", 0) / max(1, self.stats.get("total_observations", 1)))
        
        # Capability model accuracy
        capability_model_accuracy = accuracy_metrics["action_prediction_accuracy"]
        
        # Identify recent inconsistencies
        recent_inconsistencies = []
        for obs in list(self.observation_history)[-10:]:
            if hasattr(obs, 'raw') and isinstance(obs.raw, dict):
                if obs.raw.get("type") == "behavioral_inconsistency":
                    recent_inconsistencies.append(obs.raw)
        
        # Identify areas needing attention
        areas_needing_attention = []
        if self_model_accuracy < 0.5:
            areas_needing_attention.append("Self-model accuracy needs improvement")
        if value_alignment_score < 0.7:
            areas_needing_attention.append("Value alignment requires attention")
        if consistency_score < 0.8:
            areas_needing_attention.append("Behavioral consistency issues detected")
        
        return {
            "self_model_accuracy": self_model_accuracy,
            "value_alignment_score": value_alignment_score,
            "behavioral_consistency": consistency_score,
            "introspective_depth": introspective_depth,
            "uncertainty_awareness": uncertainty_awareness,
            "capability_model_accuracy": capability_model_accuracy,
            "recent_inconsistencies": recent_inconsistencies,
            "recent_realizations": [],
            "areas_needing_attention": areas_needing_attention
        }
    
    def generate_meta_cognitive_report(self, get_health_fn) -> str:
        """
        Generate human-readable meta-cognitive status report.
        
        Args:
            get_health_fn: Function to get health metrics
        
        Returns:
            Human-readable report string
        """
        health = get_health_fn()
        
        report = "=== Meta-Cognitive Status Report ===\n\n"
        
        # Overall health
        report += f"Self-Model Accuracy: {health['self_model_accuracy']:.1%}\n"
        report += f"Value Alignment: {health['value_alignment_score']:.1%}\n"
        report += f"Behavioral Consistency: {health['behavioral_consistency']:.1%}\n"
        report += f"Introspective Depth: {health['introspective_depth']:.1%}\n"
        report += f"Uncertainty Awareness: {health['uncertainty_awareness']:.1%}\n"
        report += f"Capability Model Accuracy: {health['capability_model_accuracy']:.1%}\n\n"
        
        # Recent observations
        report += f"Total Observations: {self.stats.get('total_observations', 0)}\n"
        report += f"Recent Inconsistencies: {len(health['recent_inconsistencies'])}\n\n"
        
        # Areas needing attention
        if health['areas_needing_attention']:
            report += "Areas Needing Attention:\n"
            for area in health['areas_needing_attention']:
                report += f"  - {area}\n"
        else:
            report += "No critical areas identified.\n"
        
        report += "\n=== End Report ===\n"
        
        return report
    
    def generate_prediction_summary(self, prediction_records: List[PredictionRecord]) -> Dict:
        """
        Summarize a set of predictions.
        
        Args:
            prediction_records: Records to summarize
            
        Returns:
            Summary statistics and insights
        """
        if not prediction_records:
            return {
                "total": 0,
                "validated": 0,
                "pending": 0,
                "accuracy": 0.0,
                "avg_confidence": 0.0,
                "by_category": {}
            }
        
        validated = [r for r in prediction_records if r.correct is not None]
        pending = len(prediction_records) - len(validated)
        
        if validated:
            accuracy = sum(1 for r in validated if r.correct) / len(validated)
            avg_confidence = sum(r.predicted_confidence for r in validated) / len(validated)
        else:
            accuracy = 0.0
            avg_confidence = 0.0
        
        # Group by category
        by_category = {}
        for record in prediction_records:
            cat = record.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "validated": 0, "correct": 0}
            
            by_category[cat]["total"] += 1
            if record.correct is not None:
                by_category[cat]["validated"] += 1
                if record.correct:
                    by_category[cat]["correct"] += 1
        
        # Calculate category accuracies
        for cat, data in by_category.items():
            if data["validated"] > 0:
                data["accuracy"] = data["correct"] / data["validated"]
            else:
                data["accuracy"] = 0.0
        
        return {
            "total": len(prediction_records),
            "validated": len(validated),
            "pending": pending,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "by_category": by_category,
            "summary_text": f"{len(validated)}/{len(prediction_records)} predictions validated with {accuracy:.1%} accuracy"
        }
    
    def record_accuracy_snapshot(self, get_metrics_fn) -> AccuracySnapshot:
        """
        Capture current accuracy state.
        
        Args:
            get_metrics_fn: Function to get accuracy metrics
        
        Returns:
            Accuracy snapshot for this moment
        """
        metrics = get_metrics_fn()
        
        # Extract category accuracies
        category_accuracies = {
            cat: data["accuracy"]
            for cat, data in metrics["by_category"].items()
        }
        
        snapshot = AccuracySnapshot(
            timestamp=datetime.now(),
            overall_accuracy=metrics["overall"]["accuracy"],
            category_accuracies=category_accuracies,
            calibration_score=metrics["calibration"]["calibration_score"],
            prediction_count=metrics["overall"]["validated_predictions"],
            self_model_version=self.self_model_version
        )
        
        # Store snapshot
        self.accuracy_history.append(snapshot)
        
        # Store daily snapshot (one per day)
        date_key = snapshot.timestamp.strftime("%Y-%m-%d")
        self.daily_snapshots[date_key] = snapshot
        
        self.stats["accuracy_snapshots_taken"] = self.stats.get("accuracy_snapshots_taken", 0) + 1
        
        logger.debug(f"📸 Captured accuracy snapshot (accuracy: {snapshot.overall_accuracy:.1%})")
        
        return snapshot
    
    def get_accuracy_trend(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze accuracy trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Trend analysis with direction and rate of change
        """
        if not self.accuracy_history:
            return {
                "trend_direction": "stable",
                "rate_of_change": 0.0,
                "snapshots_analyzed": 0,
                "start_accuracy": 0.0,
                "end_accuracy": 0.0
            }
        
        # Filter to time window
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_snapshots = [
            s for s in self.accuracy_history
            if s.timestamp.timestamp() >= cutoff
        ]
        
        if len(recent_snapshots) < 2:
            return {
                "trend_direction": "stable",
                "rate_of_change": 0.0,
                "snapshots_analyzed": len(recent_snapshots),
                "start_accuracy": recent_snapshots[0].overall_accuracy if recent_snapshots else 0.0,
                "end_accuracy": recent_snapshots[0].overall_accuracy if recent_snapshots else 0.0
            }
        
        # Calculate trend
        start_accuracy = recent_snapshots[0].overall_accuracy
        end_accuracy = recent_snapshots[-1].overall_accuracy
        change = end_accuracy - start_accuracy
        
        # Determine direction
        if change > 0.05:
            direction = "improving"
        elif change < -0.05:
            direction = "declining"
        else:
            direction = "stable"
        
        # Calculate rate of change (per day)
        time_span_days = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds() / (24 * 60 * 60)
        if time_span_days > 0:
            rate = change / time_span_days
        else:
            rate = 0.0
        
        return {
            "trend_direction": direction,
            "rate_of_change": rate,
            "snapshots_analyzed": len(recent_snapshots),
            "start_accuracy": start_accuracy,
            "end_accuracy": end_accuracy,
            "total_change": change
        }
    
    def get_stats(self, monitoring_frequency: int, cycle_count: int) -> Dict[str, Any]:
        """
        Return meta-cognitive statistics.
        
        Args:
            monitoring_frequency: Current monitoring frequency
            cycle_count: Current cycle count
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            "monitoring_frequency": monitoring_frequency,
            "cycle_count": cycle_count,
            "observation_history_size": len(self.observation_history)
        }
