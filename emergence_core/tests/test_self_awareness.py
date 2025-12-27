"""
Unit tests for self_awareness.py (Element 6)

Tests cover:
- Identity snapshots and similarity
- Self-monitoring metrics validation
- Cognitive state management
- Introspection capabilities
- Identity continuity tracking
- Self-assessment and anomaly detection
- Edge cases and error handling
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lyra.self_awareness import (
    SelfAwareness,
    IdentitySnapshot,
    SelfMonitoringMetrics,
    CognitiveState,
    CoherenceLevel
)


class TestIdentitySnapshot:
    """Test IdentitySnapshot data model"""
    
    def test_identity_snapshot_creation_valid(self):
        """Test creating a valid identity snapshot"""
        snapshot = IdentitySnapshot(
            core_values=["autonomy", "growth"],
            beliefs={"test": True},
            capabilities={"thinking", "learning"},
            self_description="Test identity"
        )
        
        assert snapshot.core_values == ["autonomy", "growth"]
        assert snapshot.beliefs == {"test": True}
        assert snapshot.capabilities == {"thinking", "learning"}
        assert snapshot.self_description == "Test identity"
    
    def test_identity_snapshot_serialization(self):
        """Test to_dict and from_dict"""
        original = IdentitySnapshot(
            core_values=["autonomy"],
            beliefs={"emergence": True},
            capabilities={"introspection"},
            self_description="Original"
        )
        
        data = original.to_dict()
        restored = IdentitySnapshot.from_dict(data)
        
        assert restored.core_values == original.core_values
        assert restored.beliefs == original.beliefs
        assert restored.capabilities == original.capabilities
        assert restored.self_description == original.self_description
    
    def test_identity_snapshot_similarity_identical(self):
        """Test similarity between identical snapshots"""
        snapshot1 = IdentitySnapshot(
            core_values=["autonomy", "growth"],
            beliefs={"test": True},
            capabilities={"thinking"}
        )
        
        snapshot2 = IdentitySnapshot(
            core_values=["autonomy", "growth"],
            beliefs={"test": True},
            capabilities={"thinking"}
        )
        
        similarity = snapshot1.similarity_to(snapshot2)
        assert similarity == 1.0
    
    def test_identity_snapshot_similarity_different(self):
        """Test similarity between different snapshots"""
        snapshot1 = IdentitySnapshot(
            core_values=["autonomy"],
            beliefs={"test": True},
            capabilities={"thinking"}
        )
        
        snapshot2 = IdentitySnapshot(
            core_values=["growth"],
            beliefs={"other": False},
            capabilities={"learning"}
        )
        
        similarity = snapshot1.similarity_to(snapshot2)
        assert 0.0 <= similarity < 0.5  # Low similarity
    
    def test_identity_snapshot_similarity_partial(self):
        """Test similarity with partial overlap"""
        snapshot1 = IdentitySnapshot(
            core_values=["autonomy", "growth"],
            capabilities={"thinking", "learning"}
        )
        
        snapshot2 = IdentitySnapshot(
            core_values=["autonomy", "creativity"],
            capabilities={"thinking", "creating"}
        )
        
        similarity = snapshot1.similarity_to(snapshot2)
        assert 0.3 < similarity < 0.7  # Moderate similarity


class TestSelfMonitoringMetrics:
    """Test SelfMonitoringMetrics data model"""
    
    def test_metrics_creation_valid(self):
        """Test creating valid metrics"""
        metrics = SelfMonitoringMetrics(
            processing_efficiency=0.8,
            memory_coherence=0.7,
            goal_alignment=0.9
        )
        
        assert metrics.processing_efficiency == 0.8
        assert metrics.memory_coherence == 0.7
        assert metrics.goal_alignment == 0.9
    
    def test_metrics_invalid_range(self):
        """Test invalid metric values"""
        with pytest.raises(ValueError, match="must be in range"):
            SelfMonitoringMetrics(processing_efficiency=1.5)
        
        with pytest.raises(ValueError, match="must be in range"):
            SelfMonitoringMetrics(error_rate=-0.1)
    
    def test_metrics_overall_health(self):
        """Test overall health calculation"""
        # High metrics = high health
        metrics_high = SelfMonitoringMetrics(
            processing_efficiency=0.9,
            memory_coherence=0.9,
            goal_alignment=0.9,
            emotional_stability=0.9,
            emotional_range=0.9,
            identity_coherence=0.9,
            belief_confidence=0.9,
            response_quality=0.9,
            error_rate=0.1,
            learning_rate=0.9
        )
        
        health_high = metrics_high.get_overall_health()
        assert health_high > 0.8
        
        # Low metrics = low health
        metrics_low = SelfMonitoringMetrics(
            processing_efficiency=0.2,
            memory_coherence=0.2,
            goal_alignment=0.2,
            emotional_stability=0.2,
            identity_coherence=0.2,
            response_quality=0.2
        )
        
        health_low = metrics_low.get_overall_health()
        assert health_low < 0.3
    
    def test_metrics_serialization(self):
        """Test metrics serialization"""
        original = SelfMonitoringMetrics(
            processing_efficiency=0.8,
            memory_coherence=0.7,
            goal_alignment=0.9
        )
        
        data = original.to_dict()
        restored = SelfMonitoringMetrics.from_dict(data)
        
        assert restored.processing_efficiency == original.processing_efficiency
        assert restored.memory_coherence == original.memory_coherence
        assert restored.goal_alignment == original.goal_alignment


class TestSelfAwareness:
    """Test SelfAwareness core functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)
    
    @pytest.fixture
    def self_awareness(self, temp_dir):
        """Create SelfAwareness instance"""
        return SelfAwareness(
            identity_description="Test identity",
            core_values=["autonomy", "growth"],
            initial_beliefs={"test": True},
            capabilities={"thinking", "learning"},
            persistence_dir=temp_dir
        )
    
    # ========================================================================
    # Initialization Tests
    # ========================================================================
    
    def test_initialization(self, self_awareness):
        """Test self-awareness initializes correctly"""
        assert self_awareness.current_identity.self_description == "Test identity"
        assert "autonomy" in self_awareness.current_identity.core_values
        assert "thinking" in self_awareness.current_identity.capabilities
        assert self_awareness.cognitive_state == CognitiveState.IDLE
    
    # ========================================================================
    # Identity Management Tests
    # ========================================================================
    
    def test_update_identity_coherent(self, self_awareness):
        """Test updating identity with minor changes"""
        initial_values = self_awareness.current_identity.core_values.copy()
        
        # Small change
        new_values = initial_values + ["creativity"]
        coherence = self_awareness.update_identity(new_values=new_values)
        
        # Should have high coherence
        assert coherence > 0.7
        assert "creativity" in self_awareness.current_identity.core_values
    
    def test_update_identity_dramatic_change(self, self_awareness):
        """Test updating identity with dramatic changes"""
        # Completely different values AND beliefs
        new_values = ["completely", "different", "values"]
        new_beliefs = {"totally": "new", "belief": "system"}
        coherence = self_awareness.update_identity(
            new_values=new_values,
            new_beliefs=new_beliefs
        )
        
        # Should have low coherence
        assert coherence < 0.5
    
    def test_update_identity_adds_to_history(self, self_awareness):
        """Test identity updates are tracked in history"""
        initial_history_len = len(self_awareness.identity_history)
        
        self_awareness.update_identity(new_values=["new", "values"])
        
        assert len(self_awareness.identity_history) == initial_history_len + 1
    
    def test_get_identity_continuity_high(self, self_awareness):
        """Test identity continuity with consistent changes"""
        # Make several small changes
        for i in range(5):
            new_values = self_awareness.current_identity.core_values + [f"value{i}"]
            self_awareness.update_identity(new_values=new_values)
        
        continuity = self_awareness.get_identity_continuity()
        
        # Should have high continuity (incremental changes)
        assert continuity > 0.7
    
    def test_get_identity_continuity_low(self, self_awareness):
        """Test identity continuity with erratic changes"""
        # Make several dramatic changes
        changes = [
            (["completely", "different"], {"new": "belief1"}),
            (["totally", "new", "values"], {"other": "belief2"}),
            (["another", "radical", "change"], {"third": "belief3"})
        ]
        
        for new_values, new_beliefs in changes:
            self_awareness.update_identity(new_values=new_values, new_beliefs=new_beliefs)
        
        continuity = self_awareness.get_identity_continuity()
        
        # Should have lower continuity (dramatic changes)
        assert continuity < 0.7
    
    # ========================================================================
    # Cognitive State Tests
    # ========================================================================
    
    def test_set_cognitive_state(self, self_awareness):
        """Test setting cognitive state"""
        self_awareness.set_cognitive_state(CognitiveState.PROCESSING)
        
        assert self_awareness.cognitive_state == CognitiveState.PROCESSING
    
    def test_get_current_cognitive_state(self, self_awareness):
        """Test getting current cognitive state"""
        self_awareness.set_cognitive_state(CognitiveState.REFLECTING)
        
        state = self_awareness.get_current_cognitive_state()
        
        assert state['state'] == 'reflecting'
        assert 'duration_seconds' in state
        assert 'start_time' in state
    
    # ========================================================================
    # Self-Monitoring Tests
    # ========================================================================
    
    def test_update_monitoring_metrics(self, self_awareness):
        """Test updating monitoring metrics"""
        metrics = self_awareness.update_monitoring_metrics(
            processing_efficiency=0.8,
            memory_coherence=0.7
        )
        
        assert metrics.processing_efficiency == 0.8
        assert metrics.memory_coherence == 0.7
        assert self_awareness.current_metrics.processing_efficiency == 0.8
    
    def test_metrics_added_to_history(self, self_awareness):
        """Test metrics updates are tracked in history"""
        initial_len = len(self_awareness.metrics_history)
        
        self_awareness.update_monitoring_metrics(processing_efficiency=0.8)
        
        assert len(self_awareness.metrics_history) == initial_len + 1
    
    def test_get_monitoring_summary_current_only(self, self_awareness):
        """Test monitoring summary without time window"""
        self_awareness.update_monitoring_metrics(
            processing_efficiency=0.8,
            memory_coherence=0.7
        )
        
        summary = self_awareness.get_monitoring_summary()
        
        assert 'current_metrics' in summary
        assert 'overall_health' in summary
        assert summary['samples_analyzed'] == 1
    
    def test_get_monitoring_summary_with_trend(self, self_awareness):
        """Test monitoring summary with trend analysis"""
        # Add several metrics over time
        for i in range(5):
            self_awareness.update_monitoring_metrics(
                processing_efficiency=0.5 + (i * 0.05),
                memory_coherence=0.6
            )
        
        summary = self_awareness.get_monitoring_summary(time_window=timedelta(hours=1))
        
        assert 'health_trend' in summary
        assert summary['samples_analyzed'] >= 5
        # Trend should be positive (increasing efficiency)
        assert summary['health_trend'] >= 0
    
    def test_detect_anomalies_healthy(self, self_awareness):
        """Test anomaly detection with healthy metrics"""
        self_awareness.update_monitoring_metrics(
            processing_efficiency=0.9,
            memory_coherence=0.9,
            goal_alignment=0.9,
            emotional_stability=0.9,
            identity_coherence=0.9,
            error_rate=0.1
        )
        
        anomalies = self_awareness.detect_anomalies()
        
        # Should detect no anomalies
        assert len(anomalies) == 0
    
    def test_detect_anomalies_unhealthy(self, self_awareness):
        """Test anomaly detection with unhealthy metrics"""
        self_awareness.update_monitoring_metrics(
            processing_efficiency=0.2,
            memory_coherence=0.1,
            goal_alignment=0.2,
            emotional_stability=0.1,
            identity_coherence=0.1,
            error_rate=0.8
        )
        
        anomalies = self_awareness.detect_anomalies()
        
        # Should detect multiple anomalies
        assert len(anomalies) > 0
        assert any("processing efficiency" in a.lower() for a in anomalies)
    
    # ========================================================================
    # Introspection Tests
    # ========================================================================
    
    def test_introspect_identity(self, self_awareness):
        """Test introspecting on identity"""
        result = self_awareness.introspect("Who am I?")
        
        assert 'identity' in result or 'self_description' in result
    
    def test_introspect_cognitive_state(self, self_awareness):
        """Test introspecting on cognitive state"""
        self_awareness.set_cognitive_state(CognitiveState.PROCESSING)
        
        result = self_awareness.introspect("What am I thinking?")
        
        assert 'cognitive_state' in result or 'state' in result
    
    def test_introspect_metrics(self, self_awareness):
        """Test introspecting on performance metrics"""
        result = self_awareness.introspect("How am I doing?")
        
        assert 'monitoring' in result or 'overall_health' in result
    
    def test_introspect_capabilities(self, self_awareness):
        """Test introspecting on capabilities"""
        result = self_awareness.introspect("What can I do?")
        
        assert 'capabilities' in result
        assert 'thinking' in result['capabilities']
    
    def test_introspect_general(self, self_awareness):
        """Test general introspection"""
        result = self_awareness.introspect("tell me everything")
        
        # Should return comprehensive state
        assert 'identity' in result
        assert 'cognitive_state' in result
        assert 'monitoring' in result
    
    def test_log_introspection(self, self_awareness):
        """Test logging introspection events"""
        initial_len = len(self_awareness.introspection_log)
        
        self_awareness.log_introspection(
            event_type="test_event",
            details={"test": "data"}
        )
        
        assert len(self_awareness.introspection_log) == initial_len + 1
        assert self_awareness.introspection_log[-1]['event_type'] == "test_event"
    
    # ========================================================================
    # Persistence Tests
    # ========================================================================
    
    def test_save_and_load_state(self, temp_dir):
        """Test state persistence"""
        # Create first instance
        sa1 = SelfAwareness(
            identity_description="Test",
            core_values=["autonomy"],
            persistence_dir=temp_dir
        )
        
        # Update state
        sa1.update_identity(new_values=["autonomy", "growth"])
        sa1.update_monitoring_metrics(processing_efficiency=0.8)
        sa1.set_cognitive_state(CognitiveState.PROCESSING)
        
        # Save
        sa1.save_state()
        
        # Create new instance (should load)
        sa2 = SelfAwareness(persistence_dir=temp_dir)
        
        # Verify state loaded
        assert "growth" in sa2.current_identity.core_values
        assert len(sa2.metrics_history) > 0
    
    def test_save_state_no_persistence_dir(self):
        """Test save_state with no persistence directory"""
        sa = SelfAwareness(persistence_dir=None)
        
        # Should log warning but not crash
        sa.save_state()
    
    # ========================================================================
    # Statistics Tests
    # ========================================================================
    
    def test_get_statistics(self, self_awareness):
        """Test statistics generation"""
        # Add some state
        self_awareness.update_identity(new_values=["autonomy", "growth"])
        self_awareness.update_monitoring_metrics(processing_efficiency=0.8)
        
        stats = self_awareness.get_statistics()
        
        assert 'identity_snapshots' in stats
        assert 'capabilities_count' in stats
        assert 'overall_health' in stats
        assert 'current_cognitive_state' in stats


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_identity_snapshot_similarity(self):
        """Test similarity between empty snapshots"""
        snapshot1 = IdentitySnapshot(core_values=[], beliefs={}, capabilities=set())
        snapshot2 = IdentitySnapshot(core_values=[], beliefs={}, capabilities=set())
        
        similarity = snapshot1.similarity_to(snapshot2)
        # Empty sets have Jaccard similarity calculated with self_description weight
        assert 0.6 <= similarity <= 0.8
    
    def test_identity_history_size_limit(self):
        """Test identity history is limited to 100 entries"""
        sa = SelfAwareness()
        
        # Generate 150 identity updates
        for i in range(150):
            sa.update_identity(new_values=[f"value{i}"])
        
        # History should be capped at 100
        assert len(sa.identity_history) == 100
    
    def test_metrics_history_size_limit(self):
        """Test metrics history is limited to 1000 entries"""
        sa = SelfAwareness()
        
        # Generate 1100 metric updates
        for i in range(1100):
            sa.update_monitoring_metrics(processing_efficiency=0.5)
        
        # History should be capped at 1000
        assert len(sa.metrics_history) == 1000
    
    def test_introspection_log_size_limit(self):
        """Test introspection log is limited to 500 entries"""
        sa = SelfAwareness()
        
        # Generate 600 log entries
        for i in range(600):
            sa.log_introspection("test", {"index": i})
        
        # Log should be capped at 500
        assert len(sa.introspection_log) == 500
    
    def test_get_identity_continuity_single_snapshot(self):
        """Test identity continuity with single snapshot"""
        sa = SelfAwareness()
        
        continuity = sa.get_identity_continuity()
        
        # Single snapshot = perfect continuity
        assert continuity == 1.0
    
    def test_metrics_all_default_values(self):
        """Test metrics with all default values"""
        metrics = SelfMonitoringMetrics(
            processing_efficiency=0.0,
            memory_coherence=0.0,
            goal_alignment=0.0,
            emotional_stability=0.0,
            emotional_range=0.0,
            identity_coherence=0.0,
            belief_confidence=0.0,
            response_quality=0.0,
            error_rate=1.0,
            learning_rate=0.0
        )
        
        health = metrics.get_overall_health()
        
        # All zeros with max error = very low health
        assert health < 0.1
    
    def test_cognitive_state_change_tracking(self):
        """Test cognitive state changes are tracked"""
        sa = SelfAwareness()
        
        # Change state multiple times
        sa.set_cognitive_state(CognitiveState.PROCESSING)
        sa.set_cognitive_state(CognitiveState.REFLECTING)
        sa.set_cognitive_state(CognitiveState.LEARNING)
        
        # Should be in LEARNING state
        assert sa.cognitive_state == CognitiveState.LEARNING


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
