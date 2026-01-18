"""
Unit tests for precision-weighted attention.

Tests cover:
- PrecisionWeighting computation
- Emotional state modulation
- Prediction error boosting
- Salience weighting
"""

import pytest

from lyra.cognitive_core.precision_weighting import (
    PrecisionWeighting,
    PrecisionWeights
)


class TestPrecisionWeights:
    """Test PrecisionWeights dataclass."""
    
    def test_creation(self):
        """Test creating precision weights."""
        weights = PrecisionWeights(
            percept_id="test_1",
            precision=0.7,
            base_precision=0.5,
            emotional_modulation=-0.1,
            prediction_error_boost=0.3
        )
        assert weights.percept_id == "test_1"
        assert weights.precision == 0.7


class TestPrecisionWeighting:
    """Test PrecisionWeighting class."""
    
    def test_initialization(self):
        """Test PrecisionWeighting initializes correctly."""
        pw = PrecisionWeighting()
        assert pw.base_precision == 0.5
        assert pw.arousal_dampening == 0.5
        assert pw.prediction_error_boost == 0.3
    
    def test_compute_precision_neutral(self):
        """Test precision computation with neutral emotional state."""
        pw = PrecisionWeighting()
        percept = {"content": "test"}
        emotional_state = {"arousal": 0.0, "valence": 0.0}
        
        precision = pw.compute_precision(percept, emotional_state)
        assert precision == pytest.approx(pw.base_precision, abs=0.01)
    
    def test_compute_precision_high_arousal(self):
        """Test that high arousal reduces precision."""
        pw = PrecisionWeighting()
        percept = {"content": "test"}
        
        low_arousal = {"arousal": 0.0, "valence": 0.0}
        high_arousal = {"arousal": 1.0, "valence": 0.0}
        
        precision_low = pw.compute_precision(percept, low_arousal)
        precision_high = pw.compute_precision(percept, high_arousal)
        
        # High arousal should reduce precision
        assert precision_high < precision_low
    
    def test_compute_precision_prediction_error(self):
        """Test that prediction error boosts precision."""
        pw = PrecisionWeighting()
        percept = {"content": "test"}
        emotional_state = {"arousal": 0.0, "valence": 0.0}
        
        precision_no_error = pw.compute_precision(percept, emotional_state, None)
        precision_with_error = pw.compute_precision(percept, emotional_state, 0.8)
        
        # Prediction error should boost precision
        assert precision_with_error > precision_no_error
    
    def test_precision_bounded(self):
        """Test that precision is bounded to [0, 1]."""
        pw = PrecisionWeighting()
        percept = {"content": "test"}
        
        # Extreme arousal
        extreme_state = {"arousal": 10.0, "valence": 0.0}
        precision = pw.compute_precision(percept, extreme_state)
        assert 0.0 <= precision <= 1.0
        
        # Large prediction error
        precision = pw.compute_precision(percept, {"arousal": 0.0, "valence": 0.0}, 10.0)
        assert 0.0 <= precision <= 1.0
    
    def test_apply_precision_weighting(self):
        """Test applying precision weights to salience scores."""
        pw = PrecisionWeighting()
        
        salience_scores = {
            "percept_1": 0.8,
            "percept_2": 0.6,
            "percept_3": 0.4
        }
        
        precisions = {
            "percept_1": 0.9,  # High precision
            "percept_2": 0.5,  # Medium precision
            "percept_3": 0.2   # Low precision
        }
        
        weighted = pw.apply_precision_weighting(salience_scores, precisions)
        
        # Check that weighting was applied correctly
        assert weighted["percept_1"] == pytest.approx(0.8 * 0.9, abs=0.01)
        assert weighted["percept_2"] == pytest.approx(0.6 * 0.5, abs=0.01)
        assert weighted["percept_3"] == pytest.approx(0.4 * 0.2, abs=0.01)
    
    def test_apply_precision_weighting_missing(self):
        """Test precision weighting with missing precision values."""
        pw = PrecisionWeighting()
        
        salience_scores = {
            "percept_1": 0.8,
            "percept_2": 0.6
        }
        
        precisions = {
            "percept_1": 0.9
            # percept_2 missing - should use base_precision
        }
        
        weighted = pw.apply_precision_weighting(salience_scores, precisions)
        
        assert weighted["percept_1"] == pytest.approx(0.8 * 0.9, abs=0.01)
        assert weighted["percept_2"] == pytest.approx(0.6 * pw.base_precision, abs=0.01)
    
    def test_get_precision_summary(self):
        """Test precision summary generation."""
        pw = PrecisionWeighting()
        percept = {"content": "test"}
        emotional_state = {"arousal": 0.3, "valence": 0.0}
        
        # Compute some precisions
        pw.compute_precision(percept, emotional_state)
        pw.compute_precision(percept, emotional_state, 0.5)
        
        summary = pw.get_precision_summary()
        assert summary["total_computations"] == 2
        assert "average_precision" in summary
        assert "recent_precisions" in summary
        assert len(summary["recent_precisions"]) == 2
    
    def test_history_bounded(self):
        """Test that precision history is kept bounded."""
        pw = PrecisionWeighting()
        percept = {"content": "test"}
        emotional_state = {"arousal": 0.0, "valence": 0.0}
        
        # Add many precision computations
        for i in range(150):
            pw.compute_precision(percept, emotional_state)
        
        # History should be bounded to 100
        assert len(pw.precision_history) == 100
