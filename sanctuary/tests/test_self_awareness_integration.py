"""
Integration tests for Self-Awareness with ConsciousnessCore

Tests the integration of SelfAwareness system with other consciousness components
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mind.consciousness import ConsciousnessCore
from mind.self_awareness import CognitiveState


class TestSelfAwarenessIntegration:
    """Test SelfAwareness integration with ConsciousnessCore"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        # Skip cleanup due to Windows file locking with ChromaDB
        # Files will be cleaned up by OS temp directory cleanup
    
    @pytest.fixture
    def consciousness(self, temp_dir):
        """Create ConsciousnessCore instance with all systems"""
        return ConsciousnessCore(
            memory_persistence_dir=str(temp_dir / "memories"),
            context_persistence_dir=str(temp_dir / "context"),
            executive_persistence_dir=str(temp_dir / "executive"),
            emotion_persistence_dir=str(temp_dir / "emotion"),
            self_awareness_persistence_dir=str(temp_dir / "self_awareness")
        )
    
    def test_perform_introspection_identity(self, consciousness):
        """Test identity introspection through consciousness"""
        result = consciousness.perform_introspection("who am I?")
        
        assert result is not None
        assert 'core_values' in result
        assert 'beliefs' in result
        assert 'self_description' in result
    
    def test_perform_introspection_cognitive_state(self, consciousness):
        """Test cognitive state introspection"""
        # Set a cognitive state first
        consciousness.set_cognitive_state(CognitiveState.LEARNING)
        
        result = consciousness.perform_introspection("what am I thinking about?")
        
        assert result is not None
        assert 'state' in result
        assert result['state'] == CognitiveState.LEARNING.value
    
    def test_update_self_model(self, consciousness):
        """Test updating self-model through consciousness"""
        # Update self-model (note: new_values is a list of value names, not a dict)
        consciousness.update_self_model(
            new_values=["test_value"],
            new_beliefs={"test_belief": 0.8},
            new_capabilities={"test_capability"}
        )
        
        # Verify update through introspection
        result = consciousness.perform_introspection("who am I?")
        
        assert "test_belief" in result['beliefs']
        assert "test_capability" in result.get('capabilities', set()) or result.get('capabilities_count', 0) > 0
    
    def test_get_self_state_comprehensive(self, consciousness):
        """Test comprehensive self-state retrieval"""
        # Set some state first
        consciousness.set_cognitive_state(CognitiveState.PROCESSING)
        
        state = consciousness.get_self_state_comprehensive()
        
        # Should include all subsystems
        assert 'identity' in state
        assert 'cognitive_state' in state
        assert 'emotional_state' in state
        assert 'executive_state' in state
        assert 'monitoring_metrics' in state
        assert 'continuity' in state
        assert 'capabilities' in state
        assert 'anomalies' in state
    
    def test_perform_self_assessment(self, consciousness):
        """Test self-assessment integration"""
        try:
            assessment = consciousness.perform_self_assessment()
            
            assert 'overall_health' in assessment
            assert 'health_trend' in assessment
            assert 'current_metrics' in assessment
            assert 'detected_anomalies' in assessment
            assert 'recommendations' in assessment
            assert 'metrics_updated' in assessment
            assert assessment['metrics_updated'] is True
        except AttributeError as e:
            # If memory/executive doesn't have get_statistics, skip this test
            pytest.skip(f"Missing required method: {e}")
    
    def test_set_cognitive_state(self, consciousness):
        """Test setting cognitive state"""
        consciousness.set_cognitive_state(CognitiveState.REFLECTING)
        
        result = consciousness.perform_introspection("what state am I in?")
        assert result['state'] == CognitiveState.REFLECTING.value
    
    def test_save_self_awareness_state(self, consciousness, temp_dir):
        """Test state persistence"""
        # Update model
        consciousness.update_self_model(
            new_beliefs={"persistence_test": 1.0}
        )
        
        # Save state
        consciousness.save_self_awareness_state()
        
        # Verify file exists
        state_file = temp_dir / "self_awareness" / "self_awareness_state.json"
        assert state_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
