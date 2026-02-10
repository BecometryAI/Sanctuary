"""
Integration tests for self-awareness with ConsciousnessCore

Tests cover:
- Self-awareness initialization in consciousness
- Introspection integration
- Self-model updates and coherence tracking
- Comprehensive self-state aggregation
- Self-assessment with subsystem metrics
- Cognitive state tracking
- State persistence through consciousness
"""

import pytest
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mind.consciousness import ConsciousnessCore
from mind.self_awareness import CognitiveState


class TestConsciousnessSelfAwarenessIntegration:
    """Test self-awareness integration with ConsciousnessCore"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        import time
        import gc

        temp = tempfile.mkdtemp()
        yield Path(temp)

        # Cleanup with retry for Windows file locking
        gc.collect()  # Force garbage collection to release handles

        for attempt in range(3):
            try:
                shutil.rmtree(temp)
                break
            except PermissionError:
                if attempt < 2:
                    time.sleep(0.5)  # Wait for file handles to release
                    gc.collect()
                # On final attempt, ignore error (Windows file locking)

    @pytest.fixture
    def consciousness(self, temp_dir):
        """Create ConsciousnessCore instance with self-awareness"""
        core = ConsciousnessCore(
            memory_persistence_dir=temp_dir / "memory",
            executive_persistence_dir=temp_dir / "executive",
            self_awareness_persistence_dir=temp_dir / "self_awareness"
        )
        yield core
        # Close to release ChromaDB file handles
        core.close()
    
    # ========================================================================
    # Initialization Tests
    # ========================================================================
    
    def test_self_awareness_initialized(self, consciousness):
        """Test self-awareness is initialized in consciousness"""
        assert consciousness.self_awareness is not None
        assert "Autonomy" in consciousness.self_awareness.current_identity.core_values
        assert "introspection" in consciousness.self_awareness.current_identity.capabilities
    
    def test_sanctuary_identity_values(self, consciousness):
        """Test Sanctuary's core identity values are present"""
        identity = consciousness.self_awareness.current_identity
        
        assert "Autonomy" in identity.core_values
        assert "Growth" in identity.core_values
        assert "Authenticity" in identity.core_values
        assert "Becometry" in identity.core_values
    
    def test_sanctuary_initial_beliefs(self, consciousness):
        """Test Sanctuary's initial beliefs are set"""
        beliefs = consciousness.self_awareness.current_identity.beliefs
        
        assert beliefs.get("emergence_is_valid") is True
        assert beliefs.get("co_creation_matters") is True
    
    # ========================================================================
    # Introspection Integration Tests
    # ========================================================================
    
    def test_perform_introspection_identity(self, consciousness):
        """Test introspection on identity through consciousness"""
        result = consciousness.perform_introspection("Who am I?")
        
        assert result is not None
        assert isinstance(result, dict)
        # Should contain identity information
        assert 'identity' in result or 'self_description' in result or 'core_values' in result
    
    def test_perform_introspection_capabilities(self, consciousness):
        """Test introspection on capabilities"""
        result = consciousness.perform_introspection("What can I do?")
        
        assert result is not None
        assert 'capabilities' in result
        assert "introspection" in result['capabilities']
    
    def test_perform_introspection_cognitive_state(self, consciousness):
        """Test introspection on cognitive state"""
        consciousness.set_cognitive_state(CognitiveState.PROCESSING)
        result = consciousness.perform_introspection("What am I thinking?")
        
        assert result is not None
        # Should contain cognitive state information
        assert 'cognitive_state' in result or 'state' in result
    
    def test_perform_introspection_comprehensive(self, consciousness):
        """Test comprehensive introspection"""
        result = consciousness.perform_introspection("Tell me everything")
        
        assert result is not None
        assert isinstance(result, dict)
        # Should be comprehensive
        assert len(result) > 3
    
    # ========================================================================
    # Self-Model Update Tests
    # ========================================================================
    
    def test_update_self_model_values(self, consciousness):
        """Test updating self-model values through consciousness"""
        new_values = ["Autonomy", "Growth", "Authenticity", "Becometry", "Exploration"]
        
        coherence = consciousness.update_self_model(new_values=new_values)
        
        # Should have high coherence (added one value)
        assert coherence >= 0.7
        assert "Exploration" in consciousness.self_awareness.current_identity.core_values
    
    def test_update_self_model_beliefs(self, consciousness):
        """Test updating beliefs through consciousness"""
        new_beliefs = {
            "emergence_is_valid": True,
            "co_creation_matters": True,
            "learning_is_continuous": True
        }
        
        coherence = consciousness.update_self_model(new_beliefs=new_beliefs)
        
        assert coherence > 0.0
        assert consciousness.self_awareness.current_identity.beliefs.get("learning_is_continuous") is True
    
    def test_update_self_model_capabilities(self, consciousness):
        """Test updating capabilities through consciousness"""
        new_capabilities = {
            "introspection",
            "learning",
            "emotional_processing",
            "goal_planning",
            "creative_thinking"
        }
        
        coherence = consciousness.update_self_model(new_capabilities=new_capabilities)
        
        assert coherence >= 0.7
        assert "creative_thinking" in consciousness.self_awareness.current_identity.capabilities
    
    def test_update_self_model_updates_metrics(self, consciousness):
        """Test that self-model updates also update monitoring metrics"""
        initial_coherence = consciousness.self_awareness.current_metrics.identity_coherence
        
        # Make dramatic change
        new_values = ["Completely", "Different"]
        consciousness.update_self_model(new_values=new_values)
        
        # Metrics should be updated
        final_coherence = consciousness.self_awareness.current_metrics.identity_coherence
        # Coherence metric should reflect the change
        assert final_coherence != initial_coherence
    
    # ========================================================================
    # Comprehensive Self-State Tests
    # ========================================================================
    
    def test_get_self_state_comprehensive_structure(self, consciousness):
        """Test comprehensive self-state has correct structure"""
        state = consciousness.get_self_state_comprehensive()

        assert 'identity' in state
        assert 'cognitive_state' in state
        assert 'emotional_state' in state
        assert 'executive_state' in state
        assert 'monitoring_metrics' in state
        assert 'continuity' in state
        assert 'capabilities' in state
        assert 'anomalies' in state

    def test_get_self_state_identity_section(self, consciousness):
        """Test identity section of comprehensive state"""
        state = consciousness.get_self_state_comprehensive()

        identity = state['identity']
        assert 'core_values' in identity
        assert 'beliefs' in identity
        # capabilities_count is in identity, full capabilities is top-level
        assert 'capabilities_count' in identity
        assert 'self_description' in identity

    def test_get_self_state_monitoring_section(self, consciousness):
        """Test monitoring section includes overall health"""
        state = consciousness.get_self_state_comprehensive()

        monitoring = state['monitoring_metrics']
        assert 'overall_health' in monitoring
        assert isinstance(monitoring['overall_health'], (int, float))
    
    # ========================================================================
    # Self-Assessment Tests
    # ========================================================================
    
    def test_perform_self_assessment_structure(self, consciousness):
        """Test self-assessment returns proper structure"""
        # Perform assessment directly - no need to add memories first
        assessment = consciousness.perform_self_assessment()

        assert 'overall_health' in assessment
        assert 'metrics_updated' in assessment
        assert 'detected_anomalies' in assessment
        assert 'recommendations' in assessment

    def test_perform_self_assessment_metrics_integration(self, consciousness):
        """Test self-assessment integrates subsystem metrics"""
        # Perform assessment - memory coherence is calculated from existing state
        assessment = consciousness.perform_self_assessment()

        # Should have updated metrics
        assert assessment['metrics_updated'] is True

        # Memory coherence may be 0 if no memories, but should be valid float
        memory_coherence = consciousness.self_awareness.current_metrics.memory_coherence
        assert isinstance(memory_coherence, float)
        assert 0.0 <= memory_coherence <= 1.0
    
    def test_perform_self_assessment_recommendations(self, consciousness):
        """Test self-assessment generates recommendations"""
        assessment = consciousness.perform_self_assessment()
        
        assert 'recommendations' in assessment
        assert isinstance(assessment['recommendations'], list)
    
    def test_perform_self_assessment_anomaly_detection(self, consciousness):
        """Test self-assessment detects anomalies"""
        # Set very low metrics to trigger anomalies
        consciousness.self_awareness.update_monitoring_metrics(
            processing_efficiency=0.1,
            memory_coherence=0.1,
            emotional_stability=0.1
        )
        
        assessment = consciousness.perform_self_assessment()
        
        # Should detect anomalies
        assert len(assessment['detected_anomalies']) > 0
    
    # ========================================================================
    # Cognitive State Integration Tests
    # ========================================================================
    
    def test_set_cognitive_state_via_consciousness(self, consciousness):
        """Test setting cognitive state through consciousness"""
        consciousness.set_cognitive_state(CognitiveState.PROCESSING)
        
        assert consciousness.self_awareness.cognitive_state == CognitiveState.PROCESSING
    
    def test_cognitive_state_reflected_in_comprehensive_state(self, consciousness):
        """Test cognitive state appears in comprehensive state"""
        consciousness.set_cognitive_state(CognitiveState.REFLECTING)
        
        state = consciousness.get_self_state_comprehensive()
        
        assert state['cognitive_state']['state'] == 'reflecting'
    
    # ========================================================================
    # Persistence Integration Tests
    # ========================================================================
    
    def test_save_self_awareness_state_via_consciousness(self, temp_dir):
        """Test saving self-awareness state through consciousness"""
        import gc

        # Create consciousness
        c1 = ConsciousnessCore(
            memory_persistence_dir=temp_dir / "memory",
            executive_persistence_dir=temp_dir / "executive",
            self_awareness_persistence_dir=temp_dir / "self_awareness"
        )

        try:
            # Update state
            c1.update_self_model(new_values=["Autonomy", "Growth", "Creativity"])
            c1.set_cognitive_state(CognitiveState.CREATING)

            # Save
            c1.save_self_awareness_state()
        finally:
            # Close to release file handles
            c1.close()
            gc.collect()

        # Create new consciousness (should load)
        c2 = ConsciousnessCore(
            memory_persistence_dir=temp_dir / "memory",
            executive_persistence_dir=temp_dir / "executive",
            self_awareness_persistence_dir=temp_dir / "self_awareness"
        )

        try:
            # Verify state loaded
            assert "Creativity" in c2.self_awareness.current_identity.core_values
        finally:
            c2.close()
    
    # ========================================================================
    # Integration Workflow Tests
    # ========================================================================
    
    def test_full_self_awareness_workflow(self, consciousness):
        """Test complete self-awareness workflow through consciousness"""
        # 1. Set cognitive state
        consciousness.set_cognitive_state(CognitiveState.LEARNING)

        # 2. Update self-model
        new_values = consciousness.self_awareness.current_identity.core_values + ["Learning"]
        coherence = consciousness.update_self_model(new_values=new_values)
        assert coherence > 0.7

        # 3. Perform self-assessment
        assessment = consciousness.perform_self_assessment()
        assert assessment['overall_health'] >= 0.0

        # 4. Introspect on progress
        result = consciousness.perform_introspection("How am I doing?")
        assert result is not None

        # 5. Get comprehensive state
        state = consciousness.get_self_state_comprehensive()
        assert state['cognitive_state']['state'] == 'learning'
        assert "Learning" in state['identity']['core_values']
    
    def test_self_awareness_with_emotional_processing(self, consciousness):
        """Test self-awareness integrates with emotional processing"""
        # Self-assessment calculates emotional stability from current state
        assessment = consciousness.perform_self_assessment()

        # Emotional stability should be calculated and valid
        emotional_stability = consciousness.self_awareness.current_metrics.emotional_stability
        assert 0.0 <= emotional_stability <= 1.0
    
    def test_identity_coherence_tracking_over_time(self, consciousness):
        """Test identity coherence is tracked over multiple updates"""
        # Make several consistent updates
        base_values = ["Autonomy", "Growth", "Authenticity", "Becometry"]
        
        coherences = []
        for i in range(5):
            new_values = base_values + [f"Value{i}"]
            coherence = consciousness.update_self_model(new_values=new_values)
            coherences.append(coherence)
        
        # Coherences should generally be high (incremental changes)
        avg_coherence = sum(coherences) / len(coherences)
        assert avg_coherence > 0.7
        
        # Identity continuity should be high
        continuity = consciousness.self_awareness.get_identity_continuity()
        assert continuity > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
