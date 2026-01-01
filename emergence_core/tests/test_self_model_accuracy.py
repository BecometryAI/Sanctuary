"""
Test suite for Phase 4.3: Self-Model Accuracy Tracking & Refinement

Comprehensive tests for prediction tracking, accuracy metrics,
self-model refinement, reporting, and temporal tracking.
"""
import pytest
from datetime import datetime, timedelta
from collections import deque
import json

# Test that the required dataclasses and SelfMonitor exist and have correct signatures


def test_imports():
    """Test that Phase 4.3 additions can be imported."""
    try:
        # Import should work
        from emergence_core.lyra.cognitive_core.meta_cognition import (
            SelfMonitor, 
            PredictionRecord, 
            AccuracySnapshot
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import Phase 4.3 classes: {e}")


def test_prediction_record_structure():
    """Test PredictionRecord dataclass structure."""
    from emergence_core.lyra.cognitive_core.meta_cognition import PredictionRecord
    from dataclasses import fields
    
    # Check that it's a dataclass with expected fields
    field_names = [f.name for f in fields(PredictionRecord)]
    
    required_fields = [
        'id', 'timestamp', 'category', 'predicted_state', 
        'predicted_confidence', 'actual_state', 'correct', 
        'error_magnitude', 'context', 'validated_at', 'self_model_version'
    ]
    
    for field in required_fields:
        assert field in field_names, f"Missing required field: {field}"


def test_accuracy_snapshot_structure():
    """Test AccuracySnapshot dataclass structure."""
    from emergence_core.lyra.cognitive_core.meta_cognition import AccuracySnapshot
    from dataclasses import fields
    
    # Check that it's a dataclass with expected fields
    field_names = [f.name for f in fields(AccuracySnapshot)]
    
    required_fields = [
        'timestamp', 'overall_accuracy', 'category_accuracies',
        'calibration_score', 'prediction_count', 'self_model_version'
    ]
    
    for field in required_fields:
        assert field in field_names, f"Missing required field: {field}"


def test_self_monitor_has_new_methods():
    """Test that SelfMonitor has all Phase 4.3 methods."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    required_methods = [
        # Prediction tracking
        'record_prediction',
        'validate_prediction',
        'auto_validate_predictions',
        
        # Accuracy metrics
        'get_accuracy_metrics',
        'calculate_confidence_calibration',
        'detect_systematic_biases',
        
        # Self-model refinement
        'refine_self_model_from_errors',
        'adjust_capability_confidence',
        'update_limitation_boundaries',
        'identify_capability_gaps',
        
        # Reporting
        'generate_accuracy_report',
        'generate_prediction_summary',
        
        # Temporal tracking
        'record_accuracy_snapshot',
        'get_accuracy_trend'
    ]
    
    for method_name in required_methods:
        assert hasattr(SelfMonitor, method_name), f"Missing method: {method_name}"
        assert callable(getattr(SelfMonitor, method_name)), f"Method not callable: {method_name}"


def test_self_monitor_initialization():
    """Test that SelfMonitor initializes with Phase 4.3 attributes."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    config = {
        "prediction_tracking": {
            "enabled": True,
            "max_pending_validations": 100
        },
        "self_model_refinement": {
            "auto_refine": True
        }
    }
    
    monitor = SelfMonitor(workspace=None, config=config)
    
    # Check Phase 4.3 attributes
    assert hasattr(monitor, 'prediction_records')
    assert hasattr(monitor, 'pending_validations')
    assert hasattr(monitor, 'self_model_version')
    assert hasattr(monitor, 'accuracy_by_category')
    assert hasattr(monitor, 'calibration_bins')
    assert hasattr(monitor, 'accuracy_history')
    assert hasattr(monitor, 'daily_snapshots')
    
    # Check they're initialized
    assert isinstance(monitor.prediction_records, dict)
    assert isinstance(monitor.pending_validations, deque)
    assert isinstance(monitor.accuracy_by_category, dict)
    assert isinstance(monitor.calibration_bins, dict)
    assert isinstance(monitor.accuracy_history, deque)
    assert isinstance(monitor.daily_snapshots, dict)


def test_record_prediction_basic():
    """Test basic prediction recording functionality."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={
        "prediction_tracking": {"enabled": True}
    })
    
    prediction_id = monitor.record_prediction(
        category="action",
        predicted_state={"action": "test"},
        confidence=0.85,
        context={"test": "context"}
    )
    
    assert prediction_id != ""
    assert prediction_id in monitor.prediction_records
    
    record = monitor.prediction_records[prediction_id]
    assert record.category == "action"
    assert record.predicted_confidence == 0.85


def test_validate_prediction_basic():
    """Test basic prediction validation."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={
        "prediction_tracking": {"enabled": True}
    })
    
    # Record prediction
    prediction_id = monitor.record_prediction(
        category="action",
        predicted_state={"action": "test"},
        confidence=0.9,
        context={}
    )
    
    # Validate
    validated = monitor.validate_prediction(
        prediction_id,
        {"action": "test"}
    )
    
    assert validated is not None
    assert validated.correct is True
    assert validated.validated_at is not None


def test_get_accuracy_metrics_structure():
    """Test that get_accuracy_metrics returns proper structure."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={})
    metrics = monitor.get_accuracy_metrics()
    
    # Check structure
    assert "overall" in metrics
    assert "by_category" in metrics
    assert "by_confidence_level" in metrics
    assert "calibration" in metrics
    assert "temporal_trends" in metrics
    assert "error_patterns" in metrics
    
    # Check overall structure
    assert "accuracy" in metrics["overall"]
    assert "total_predictions" in metrics["overall"]
    assert "validated_predictions" in metrics["overall"]
    assert "pending_validations" in metrics["overall"]
    
    # Check calibration structure
    assert "calibration_score" in metrics["calibration"]
    assert "overconfidence" in metrics["calibration"]
    assert "underconfidence" in metrics["calibration"]
    assert "calibration_curve" in metrics["calibration"]


def test_generate_report_formats():
    """Test that report generation works for all formats."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={})
    
    # Add some predictions
    for i in range(3):
        pid = monitor.record_prediction(
            category="action",
            predicted_state={"action": "test"},
            confidence=0.8,
            context={}
        )
        monitor.validate_prediction(pid, {"action": "test"})
    
    # Test text format
    text_report = monitor.generate_accuracy_report(format="text")
    assert isinstance(text_report, str)
    assert "SELF-MODEL ACCURACY REPORT" in text_report
    
    # Test markdown format
    md_report = monitor.generate_accuracy_report(format="markdown")
    assert isinstance(md_report, str)
    assert "# SELF-MODEL ACCURACY REPORT" in md_report
    
    # Test JSON format
    json_report = monitor.generate_accuracy_report(format="json")
    assert isinstance(json_report, str)
    data = json.loads(json_report)
    assert "overall" in data


def test_accuracy_snapshot_creation():
    """Test accuracy snapshot creation."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={})
    
    # Add predictions
    for i in range(5):
        pid = monitor.record_prediction(
            category="action",
            predicted_state={"action": "test"},
            confidence=0.8,
            context={}
        )
        monitor.validate_prediction(pid, {"action": "test"})
    
    # Create snapshot
    snapshot = monitor.record_accuracy_snapshot()
    
    assert snapshot is not None
    assert hasattr(snapshot, 'overall_accuracy')
    assert hasattr(snapshot, 'timestamp')
    assert len(monitor.accuracy_history) == 1
    assert monitor.stats["accuracy_snapshots_taken"] == 1


def test_self_model_refinement():
    """Test self-model refinement from errors."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={
        "self_model_refinement": {
            "auto_refine": True,
            "require_min_samples": 2  # Lower for testing
        }
    })
    
    # Initialize capability
    monitor.self_model["capabilities"]["test_action"] = {
        "attempts": 5,
        "successes": 5,
        "confidence": 0.9
    }
    
    initial_confidence = monitor.self_model["capabilities"]["test_action"]["confidence"]
    
    # Create error records
    error_records = []
    for i in range(5):
        pid = monitor.record_prediction(
            category="action",
            predicted_state={"action": "test_action"},
            confidence=0.9,
            context={}
        )
        record = monitor.validate_prediction(pid, {"action": "wrong_action"})
        error_records.append(record)
    
    # Refine
    monitor.refine_self_model_from_errors(error_records)
    
    # Confidence should decrease
    new_confidence = monitor.self_model["capabilities"]["test_action"]["confidence"]
    assert new_confidence < initial_confidence


def test_capability_gap_identification():
    """Test identifying capability gaps."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={})
    
    # Add capability with few attempts
    monitor.self_model["capabilities"]["sparse_capability"] = {
        "attempts": 2,
        "successes": 1,
        "confidence": 0.5
    }
    
    gaps = monitor.identify_capability_gaps()
    
    assert len(gaps) > 0
    assert any(g.get("reason") == "insufficient_data" for g in gaps)


def test_confidence_calibration():
    """Test confidence calibration calculation."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={})
    
    # Add predictions with known calibration
    for i in range(10):
        pid = monitor.record_prediction(
            category="action",
            predicted_state={"action": "test"},
            confidence=0.8,
            context={}
        )
        # Make 8/10 correct for perfect calibration
        if i < 8:
            monitor.validate_prediction(pid, {"action": "test"})
        else:
            monitor.validate_prediction(pid, {"action": "wrong"})
    
    calibration = monitor.calculate_confidence_calibration()
    
    assert "calibration_score" in calibration
    assert "overconfidence" in calibration
    assert "underconfidence" in calibration
    assert "calibration_curve" in calibration
    
    # Should be well calibrated
    assert calibration["calibration_score"] > 0.7


def test_systematic_bias_detection():
    """Test detection of systematic biases."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={})
    
    # Create systematic errors in emotion category
    for i in range(10):
        pid = monitor.record_prediction(
            category="emotion",
            predicted_state={
                "emotional_prediction": {
                    "valence": 0.8,
                    "arousal": 0.5,
                    "dominance": 0.5
                }
            },
            confidence=0.9,
            context={}
        )
        # All wrong
        monitor.validate_prediction(pid, {
            "emotions": {"valence": 0.2, "arousal": 0.5, "dominance": 0.5}
        })
    
    biases = monitor.detect_systematic_biases()
    
    assert "systematic_biases" in biases
    assert "common_errors" in biases
    assert "error_contexts" in biases


def test_accuracy_trend_analysis():
    """Test temporal trend analysis."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={})
    
    # Create first snapshot with low accuracy
    for i in range(5):
        pid = monitor.record_prediction(
            category="action",
            predicted_state={"action": "test"},
            confidence=0.8,
            context={}
        )
        # Only 2/5 correct
        if i < 2:
            monitor.validate_prediction(pid, {"action": "test"})
        else:
            monitor.validate_prediction(pid, {"action": "wrong"})
    
    monitor.record_accuracy_snapshot()
    
    # Create second snapshot with high accuracy
    for i in range(5):
        pid = monitor.record_prediction(
            category="action",
            predicted_state={"action": "test"},
            confidence=0.8,
            context={}
        )
        # 4/5 correct
        if i < 4:
            monitor.validate_prediction(pid, {"action": "test"})
        else:
            monitor.validate_prediction(pid, {"action": "wrong"})
    
    monitor.record_accuracy_snapshot()
    
    # Analyze trend
    trend = monitor.get_accuracy_trend(days=7)
    
    assert "trend_direction" in trend
    assert "rate_of_change" in trend
    assert "snapshots_analyzed" in trend
    assert trend["snapshots_analyzed"] == 2


def test_prediction_summary():
    """Test prediction summary generation."""
    from emergence_core.lyra.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor(workspace=None, config={})
    
    # Create mix of predictions
    records = []
    for i in range(10):
        pid = monitor.record_prediction(
            category="action",
            predicted_state={"action": "test"},
            confidence=0.8,
            context={}
        )
        record = monitor.prediction_records[pid]
        
        # Validate first 7
        if i < 7:
            monitor.validate_prediction(pid, {"action": "test" if i < 5 else "wrong"})
        
        records.append(record)
    
    summary = monitor.generate_prediction_summary(records)
    
    assert summary["total"] == 10
    assert summary["validated"] == 7
    assert summary["pending"] == 3
    assert "accuracy" in summary
    assert "by_category" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
