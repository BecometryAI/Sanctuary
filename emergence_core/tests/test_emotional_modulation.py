"""
Unit tests for emotional_modulation.py

Tests verify that emotions functionally modulate processing parameters
BEFORE any LLM invocation, making emotions causally efficacious.

Tests cover:
- Arousal modulation of processing speed/thoroughness
- Valence modulation of action selection
- Dominance modulation of decision thresholds
- Metrics tracking of modulation effects
- Ablation testing (modulation on vs off)
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for standalone testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lyra.cognitive_core.emotional_modulation import (
    EmotionalModulation,
    ProcessingParams,
    ModulationMetrics
)


class TestProcessingParams:
    """Test ProcessingParams dataclass."""
    
    def test_processing_params_creation(self):
        """Test creating ProcessingParams."""
        params = ProcessingParams(
            attention_iterations=8,
            ignition_threshold=0.55,
            memory_retrieval_limit=4,
            processing_timeout=1.8,
            decision_threshold=0.65,
            action_bias_strength=0.2
        )
        
        assert params.attention_iterations == 8
        assert params.ignition_threshold == 0.55
        assert params.memory_retrieval_limit == 4
        assert params.processing_timeout == 1.8
        assert params.decision_threshold == 0.65
        assert params.action_bias_strength == 0.2
    
    def test_processing_params_to_dict(self):
        """Test converting ProcessingParams to dict."""
        params = ProcessingParams(
            attention_iterations=7,
            arousal_level=0.6,
            valence_level=0.3,
            dominance_level=0.7
        )
        
        params_dict = params.to_dict()
        assert params_dict['attention_iterations'] == 7
        assert params_dict['arousal_level'] == 0.6
        assert params_dict['valence_level'] == 0.3
        assert params_dict['dominance_level'] == 0.7
        assert 'timestamp' in params_dict


class TestModulationMetrics:
    """Test ModulationMetrics dataclass."""
    
    def test_metrics_initialization(self):
        """Test metrics start at zero."""
        metrics = ModulationMetrics()
        
        assert metrics.total_modulations == 0
        assert metrics.high_arousal_fast_processing == 0
        assert metrics.low_arousal_slow_processing == 0
        assert len(metrics.arousal_attention_correlations) == 0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dict."""
        metrics = ModulationMetrics()
        metrics.total_modulations = 10
        metrics.high_arousal_fast_processing = 5
        
        metrics_dict = metrics.to_dict()
        assert metrics_dict['total_modulations'] == 10
        assert metrics_dict['arousal_effects']['high_arousal_fast'] == 5


class TestEmotionalModulationInitialization:
    """Test EmotionalModulation initialization."""
    
    def test_initialization_enabled(self):
        """Test initialization with modulation enabled."""
        modulation = EmotionalModulation(enabled=True)
        
        assert modulation.enabled is True
        assert isinstance(modulation.metrics, ModulationMetrics)
        assert isinstance(modulation.baseline_params, ProcessingParams)
    
    def test_initialization_disabled(self):
        """Test initialization with modulation disabled (for ablation)."""
        modulation = EmotionalModulation(enabled=False)
        
        assert modulation.enabled is False
    
    def test_baseline_params_sensible(self):
        """Test baseline parameters are sensible defaults."""
        modulation = EmotionalModulation()
        
        baseline = modulation.baseline_params
        assert 5 <= baseline.attention_iterations <= 10
        assert 0.4 <= baseline.ignition_threshold <= 0.6
        assert 2 <= baseline.memory_retrieval_limit <= 5
        assert 1.0 <= baseline.processing_timeout <= 2.0
        assert 0.5 <= baseline.decision_threshold <= 0.7


class TestArousalModulation:
    """Test arousal modulation of processing parameters."""
    
    def test_high_arousal_fast_processing(self):
        """High arousal should produce faster, less thorough processing."""
        modulation = EmotionalModulation()
        
        # High arousal (0.9)
        params = modulation.modulate_processing(
            arousal=0.9,
            valence=0.0,
            dominance=0.5
        )
        
        # Should have fewer iterations (snap decisions)
        assert params.attention_iterations < 7
        
        # Should have lower threshold (react to more stimuli)
        assert params.ignition_threshold < 0.5
        
        # Should have fewer memories (less deliberation)
        assert params.memory_retrieval_limit < 4
        
        # Should have shorter timeout (quick response)
        assert params.processing_timeout < 1.5
        
        # Metrics should track this
        assert modulation.metrics.high_arousal_fast_processing > 0
    
    def test_low_arousal_slow_processing(self):
        """Low arousal should produce slower, more deliberate processing."""
        modulation = EmotionalModulation()
        
        # Low arousal (0.1)
        params = modulation.modulate_processing(
            arousal=0.1,
            valence=0.0,
            dominance=0.5
        )
        
        # Should have more iterations (careful analysis)
        assert params.attention_iterations > 7
        
        # Should have higher threshold (more selective)
        assert params.ignition_threshold > 0.5
        
        # Should have more memories (thorough consideration)
        assert params.memory_retrieval_limit > 3
        
        # Should have longer timeout (take time)
        assert params.processing_timeout > 1.5
        
        # Metrics should track this
        assert modulation.metrics.low_arousal_slow_processing > 0
    
    def test_arousal_correlation_tracking(self):
        """Test that arousal-attention correlations are tracked."""
        modulation = EmotionalModulation()
        
        # Multiple modulations with different arousal levels
        for arousal in [0.2, 0.5, 0.8]:
            modulation.modulate_processing(arousal, 0.0, 0.5)
        
        # Should have correlation data
        assert len(modulation.metrics.arousal_attention_correlations) == 3
        
        # Check correlation data structure
        arousal_val, iterations, threshold = modulation.metrics.arousal_attention_correlations[0]
        assert 0.0 <= arousal_val <= 1.0
        assert 5 <= iterations <= 10
        assert 0.4 <= threshold <= 0.6
    
    def test_arousal_range_bounds(self):
        """Test that arousal modulation respects parameter bounds."""
        modulation = EmotionalModulation()
        
        # Extreme arousal values
        params_high = modulation.modulate_processing(1.0, 0.0, 0.5)
        params_low = modulation.modulate_processing(0.0, 0.0, 0.5)
        
        # Check bounds
        assert 5 <= params_high.attention_iterations <= 10
        assert 5 <= params_low.attention_iterations <= 10
        
        assert 0.4 <= params_high.ignition_threshold <= 0.6
        assert 0.4 <= params_low.ignition_threshold <= 0.6
        
        assert 2 <= params_high.memory_retrieval_limit <= 5
        assert 2 <= params_low.memory_retrieval_limit <= 5
        
        assert 1.0 <= params_high.processing_timeout <= 2.0
        assert 1.0 <= params_low.processing_timeout <= 2.0


class TestValenceModulation:
    """Test valence modulation of action selection."""
    
    def test_positive_valence_approach_bias(self):
        """Positive valence should boost approach actions."""
        modulation = EmotionalModulation()
        
        # Create mock actions (dicts)
        actions = [
            {'type': 'speak', 'priority': 0.5},
            {'type': 'explore', 'priority': 0.5},
            {'type': 'wait', 'priority': 0.5},
            {'type': 'withdraw', 'priority': 0.5}
        ]
        
        # Apply positive valence bias
        biased_actions = modulation.bias_action_selection(actions, valence=0.8)
        
        # Approach actions should be boosted
        speak_action = next(a for a in biased_actions if a['type'] == 'speak')
        assert speak_action['priority'] > 0.5
        
        explore_action = next(a for a in biased_actions if a['type'] == 'explore')
        assert explore_action['priority'] > 0.5
        
        # Avoidance actions should be reduced
        wait_action = next(a for a in biased_actions if a['type'] == 'wait')
        assert wait_action['priority'] < 0.5
        
        withdraw_action = next(a for a in biased_actions if a['type'] == 'withdraw')
        assert withdraw_action['priority'] < 0.5
    
    def test_negative_valence_avoidance_bias(self):
        """Negative valence should boost avoidance actions."""
        modulation = EmotionalModulation()
        
        # Create mock actions
        actions = [
            {'type': 'speak', 'priority': 0.5},
            {'type': 'create', 'priority': 0.5},
            {'type': 'wait', 'priority': 0.5},
            {'type': 'introspect', 'priority': 0.5}
        ]
        
        # Apply negative valence bias
        biased_actions = modulation.bias_action_selection(actions, valence=-0.8)
        
        # Approach actions should be reduced
        speak_action = next(a for a in biased_actions if a['type'] == 'speak')
        assert speak_action['priority'] < 0.5
        
        create_action = next(a for a in biased_actions if a['type'] == 'create')
        assert create_action['priority'] < 0.5
        
        # Avoidance actions should be boosted
        wait_action = next(a for a in biased_actions if a['type'] == 'wait')
        assert wait_action['priority'] > 0.5
        
        introspect_action = next(a for a in biased_actions if a['type'] == 'introspect')
        assert introspect_action['priority'] > 0.5
    
    def test_neutral_valence_no_bias(self):
        """Neutral valence should not significantly bias actions."""
        modulation = EmotionalModulation()
        
        actions = [
            {'type': 'speak', 'priority': 0.5},
            {'type': 'wait', 'priority': 0.5}
        ]
        
        # Apply neutral valence
        biased_actions = modulation.bias_action_selection(actions, valence=0.1)
        
        # Priorities should be unchanged or minimally changed
        for action in biased_actions:
            assert abs(action['priority'] - 0.5) < 0.1
    
    def test_action_objects_with_attributes(self):
        """Test biasing works with action objects (not just dicts)."""
        modulation = EmotionalModulation()
        
        # Create mock action objects using simple class
        class MockAction:
            def __init__(self, action_type, priority):
                self.type = action_type
                self.priority = priority
        
        actions = [
            MockAction('speak', 0.5),
            MockAction('wait', 0.5)
        ]
        
        # Apply positive valence
        biased_actions = modulation.bias_action_selection(actions, valence=0.6)
        
        # Check biasing worked on objects
        assert biased_actions[0].priority > 0.5  # speak boosted
        assert biased_actions[1].priority < 0.5  # wait reduced
    
    def test_valence_correlation_tracking(self):
        """Test that valence-action correlations are tracked."""
        modulation = EmotionalModulation()
        
        actions = [{'type': 'speak', 'priority': 0.5}]
        
        # Multiple bias applications
        for valence in [-0.5, 0.0, 0.5]:
            modulation.bias_action_selection(actions, valence)
            # Also call modulate_processing to update metrics
            modulation.modulate_processing(0.5, valence, 0.5)
        
        # Should have correlation data
        assert len(modulation.metrics.valence_action_correlations) == 3


class TestDominanceModulation:
    """Test dominance modulation of decision thresholds."""
    
    def test_high_dominance_assertive(self):
        """High dominance should lower decision threshold (more assertive)."""
        modulation = EmotionalModulation()
        
        # High dominance (0.9)
        params = modulation.modulate_processing(
            arousal=0.5,
            valence=0.0,
            dominance=0.9
        )
        
        # Should have lower decision threshold
        assert params.decision_threshold < 0.7
        assert params.decision_threshold >= 0.5
        
        # Metrics should track this
        assert modulation.metrics.high_dominance_assertive > 0
    
    def test_low_dominance_cautious(self):
        """Low dominance should raise decision threshold (more cautious)."""
        modulation = EmotionalModulation()
        
        # Low dominance (0.1)
        params = modulation.modulate_processing(
            arousal=0.5,
            valence=0.0,
            dominance=0.1
        )
        
        # Should have higher decision threshold
        assert params.decision_threshold > 0.65
        assert params.decision_threshold <= 0.7
        
        # Metrics should track this
        assert modulation.metrics.low_dominance_cautious > 0
    
    def test_dominance_correlation_tracking(self):
        """Test that dominance-threshold correlations are tracked."""
        modulation = EmotionalModulation()
        
        # Multiple modulations with different dominance levels
        for dominance in [0.2, 0.5, 0.8]:
            modulation.modulate_processing(0.5, 0.0, dominance)
        
        # Should have correlation data
        assert len(modulation.metrics.dominance_threshold_correlations) == 3
        
        # Check correlation data structure
        dom_val, threshold = modulation.metrics.dominance_threshold_correlations[0]
        assert 0.0 <= dom_val <= 1.0
        assert 0.5 <= threshold <= 0.7


class TestAblationTesting:
    """Test ablation (modulation on vs off) to verify functional effects."""
    
    def test_disabled_returns_baseline(self):
        """When disabled, should return baseline parameters."""
        modulation = EmotionalModulation(enabled=False)
        
        # Try to modulate with extreme values
        params = modulation.modulate_processing(
            arousal=1.0,
            valence=1.0,
            dominance=1.0
        )
        
        # Should match baseline exactly
        baseline = modulation.baseline_params
        assert params.attention_iterations == baseline.attention_iterations
        assert params.ignition_threshold == baseline.ignition_threshold
        assert params.memory_retrieval_limit == baseline.memory_retrieval_limit
        assert params.processing_timeout == baseline.processing_timeout
        assert params.decision_threshold == baseline.decision_threshold
    
    def test_enabled_vs_disabled_different(self):
        """Enabled and disabled modulation should produce different results."""
        enabled_mod = EmotionalModulation(enabled=True)
        disabled_mod = EmotionalModulation(enabled=False)
        
        arousal, valence, dominance = 0.9, 0.8, 0.2
        
        enabled_params = enabled_mod.modulate_processing(arousal, valence, dominance)
        disabled_params = disabled_mod.modulate_processing(arousal, valence, dominance)
        
        # Should be different
        assert enabled_params.attention_iterations != disabled_params.attention_iterations
        assert enabled_params.ignition_threshold != disabled_params.ignition_threshold
        assert enabled_params.decision_threshold != disabled_params.decision_threshold
    
    def test_set_enabled_toggle(self):
        """Test toggling enabled state."""
        modulation = EmotionalModulation(enabled=True)
        
        # Get modulated params
        params1 = modulation.modulate_processing(0.9, 0.0, 0.5)
        assert params1.attention_iterations != modulation.baseline_params.attention_iterations
        
        # Disable and try again
        modulation.set_enabled(False)
        params2 = modulation.modulate_processing(0.9, 0.0, 0.5)
        assert params2.attention_iterations == modulation.baseline_params.attention_iterations
        
        # Re-enable
        modulation.set_enabled(True)
        params3 = modulation.modulate_processing(0.9, 0.0, 0.5)
        assert params3.attention_iterations != modulation.baseline_params.attention_iterations


class TestMetricsTracking:
    """Test metrics tracking verifies functional modulation."""
    
    def test_metrics_accumulate(self):
        """Test that metrics accumulate over multiple modulations."""
        modulation = EmotionalModulation()
        
        # Multiple modulations
        for _ in range(10):
            modulation.modulate_processing(0.8, 0.5, 0.7)
        
        # Total should accumulate
        assert modulation.metrics.total_modulations == 10
        assert modulation.metrics.high_arousal_fast_processing > 0
        assert modulation.metrics.positive_valence_approach_bias > 0
        assert modulation.metrics.high_dominance_assertive > 0
    
    def test_get_metrics(self):
        """Test getting metrics as dict."""
        modulation = EmotionalModulation()
        
        # Do some modulations
        modulation.modulate_processing(0.9, 0.7, 0.8)
        modulation.modulate_processing(0.2, -0.6, 0.3)
        
        metrics_dict = modulation.get_metrics()
        
        assert 'total_modulations' in metrics_dict
        assert 'arousal_effects' in metrics_dict
        assert 'valence_effects' in metrics_dict
        assert 'dominance_effects' in metrics_dict
        
        assert metrics_dict['total_modulations'] == 2
    
    def test_reset_metrics(self):
        """Test resetting metrics."""
        modulation = EmotionalModulation()
        
        # Do some modulations
        modulation.modulate_processing(0.8, 0.5, 0.7)
        assert modulation.metrics.total_modulations > 0
        
        # Reset
        modulation.reset_metrics()
        assert modulation.metrics.total_modulations == 0
        assert len(modulation.metrics.arousal_attention_correlations) == 0
    
    def test_correlation_list_bounded(self):
        """Test that correlation lists stay bounded (last 100)."""
        modulation = EmotionalModulation()
        
        # Do 150 modulations
        for _ in range(150):
            modulation.modulate_processing(0.5, 0.0, 0.5)
        
        # Lists should be bounded to 100
        assert len(modulation.metrics.arousal_attention_correlations) == 100
        assert len(modulation.metrics.valence_action_correlations) == 100
        assert len(modulation.metrics.dominance_threshold_correlations) == 100


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_fight_or_flight_scenario(self):
        """Test fight-or-flight emotional state (high arousal, negative valence, low dominance)."""
        modulation = EmotionalModulation()
        
        # Fearful state: high arousal, negative valence, low dominance
        params = modulation.modulate_processing(
            arousal=0.9,    # Very aroused (alert)
            valence=-0.7,   # Negative (fearful)
            dominance=0.2   # Low control (submissive)
        )
        
        # Should produce fast, defensive processing
        assert params.attention_iterations < 7  # Quick decisions
        assert params.ignition_threshold < 0.5  # React to everything
        assert params.decision_threshold > 0.65  # Cautious (high threshold from low dominance)
        
        # Test action biasing
        actions = [
            {'type': 'speak', 'priority': 0.5},
            {'type': 'wait', 'priority': 0.5},
            {'type': 'introspect', 'priority': 0.5}
        ]
        biased = modulation.bias_action_selection(actions, valence=-0.7)
        
        # Should prefer defensive actions
        wait_action = next(a for a in biased if a['type'] == 'wait')
        assert wait_action['priority'] > 0.5
    
    def test_confident_joy_scenario(self):
        """Test joyful, confident emotional state."""
        modulation = EmotionalModulation()
        
        # Joyful state: medium arousal, high positive valence, high dominance
        params = modulation.modulate_processing(
            arousal=0.6,    # Moderately aroused
            valence=0.8,    # Very positive
            dominance=0.9   # High control (dominant)
        )
        
        # Should produce moderately fast, assertive processing
        assert params.attention_iterations <= 7
        assert params.decision_threshold < 0.6  # Assertive (low threshold from high dominance)
        
        # Test action biasing
        actions = [
            {'type': 'speak', 'priority': 0.5},
            {'type': 'create', 'priority': 0.5},
            {'type': 'wait', 'priority': 0.5}
        ]
        biased = modulation.bias_action_selection(actions, valence=0.8)
        
        # Should prefer approach actions
        speak_action = next(a for a in biased if a['type'] == 'speak')
        assert speak_action['priority'] > 0.5
        
        create_action = next(a for a in biased if a['type'] == 'create')
        assert create_action['priority'] > 0.5
    
    def test_calm_deliberation_scenario(self):
        """Test calm, deliberate emotional state."""
        modulation = EmotionalModulation()
        
        # Calm state: low arousal, neutral valence, moderate dominance
        params = modulation.modulate_processing(
            arousal=0.2,    # Very calm
            valence=0.1,    # Slightly positive
            dominance=0.5   # Moderate control
        )
        
        # Should produce slow, thorough processing
        assert params.attention_iterations > 7  # Careful analysis
        assert params.ignition_threshold > 0.5  # Selective attention
        assert params.memory_retrieval_limit > 3  # Thorough consideration
        assert params.processing_timeout > 1.5  # Take time to think


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
