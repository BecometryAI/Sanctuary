"""
Integration tests for emotional modulation across cognitive subsystems.

Tests verify that emotional modulation:
1. Affects attention parameters (arousal → iterations, threshold)
2. Affects action selection (valence → approach/avoidance bias)
3. Affects decision thresholds (dominance → confidence)
4. Produces measurably different behavior when enabled vs disabled
"""

import sys
from pathlib import Path

# Add paths for standalone testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lyra" / "cognitive_core"))

import emotional_modulation
from datetime import datetime


class MockAction:
    """Mock action for testing."""
    def __init__(self, action_type, priority=0.5):
        self.type = action_type
        self.priority = priority


def test_affect_integration():
    """Test that AffectSubsystem properly integrates emotional modulation."""
    print("\n=== Testing Affect Integration ===")
    
    # We can't fully import affect.py due to dependencies, but we can test
    # that the emotional_modulation module works as expected
    modulation = emotional_modulation.EmotionalModulation(enabled=True)
    
    # Simulate high arousal state (fearful)
    params = modulation.modulate_processing(
        arousal=0.9,
        valence=-0.6,
        dominance=0.2
    )
    
    # Should produce fast, cautious processing
    assert params.attention_iterations < 7, "High arousal should reduce iterations"
    assert params.ignition_threshold < 0.5, "High arousal should lower threshold"
    assert params.decision_threshold > 0.65, "Low dominance should raise decision threshold"
    
    print(f"✓ High arousal (fear): iterations={params.attention_iterations}, "
          f"threshold={params.ignition_threshold:.2f}, "
          f"decision={params.decision_threshold:.2f}")
    
    # Simulate calm, confident state
    params = modulation.modulate_processing(
        arousal=0.2,
        valence=0.5,
        dominance=0.8
    )
    
    # Should produce slow, assertive processing
    assert params.attention_iterations > 7, "Low arousal should increase iterations"
    assert params.ignition_threshold > 0.5, "Low arousal should raise threshold"
    assert params.decision_threshold < 0.6, "High dominance should lower decision threshold"
    
    print(f"✓ Low arousal (calm): iterations={params.attention_iterations}, "
          f"threshold={params.ignition_threshold:.2f}, "
          f"decision={params.decision_threshold:.2f}")


def test_action_biasing():
    """Test valence-based action biasing."""
    print("\n=== Testing Action Biasing ===")
    
    modulation = emotional_modulation.EmotionalModulation(enabled=True)
    
    # Create mock actions
    actions = [
        MockAction('speak', 0.5),
        MockAction('create', 0.5),
        MockAction('wait', 0.5),
        MockAction('introspect', 0.5)
    ]
    
    # Test positive valence (should boost approach actions)
    biased_positive = modulation.bias_action_selection(actions, valence=0.8)
    
    speak_action = next(a for a in biased_positive if a.type == 'speak')
    wait_action = next(a for a in biased_positive if a.type == 'wait')
    
    assert speak_action.priority > 0.5, "Positive valence should boost speak"
    assert wait_action.priority < 0.5, "Positive valence should reduce wait"
    
    print(f"✓ Positive valence: speak={speak_action.priority:.2f}, wait={wait_action.priority:.2f}")
    
    # Reset actions
    for action in actions:
        action.priority = 0.5
    
    # Test negative valence (should boost avoidance actions)
    biased_negative = modulation.bias_action_selection(actions, valence=-0.8)
    
    speak_action = next(a for a in biased_negative if a.type == 'speak')
    wait_action = next(a for a in biased_negative if a.type == 'wait')
    
    assert speak_action.priority < 0.5, "Negative valence should reduce speak"
    assert wait_action.priority > 0.5, "Negative valence should boost wait"
    
    print(f"✓ Negative valence: speak={speak_action.priority:.2f}, wait={wait_action.priority:.2f}")


def test_ablation_behavior_difference():
    """Test that enabling/disabling modulation produces different behavior."""
    print("\n=== Testing Ablation (Enabled vs Disabled) ===")
    
    enabled_mod = emotional_modulation.EmotionalModulation(enabled=True)
    disabled_mod = emotional_modulation.EmotionalModulation(enabled=False)
    
    # High emotional state
    arousal, valence, dominance = 0.9, 0.8, 0.2
    
    enabled_params = enabled_mod.modulate_processing(arousal, valence, dominance)
    disabled_params = disabled_mod.modulate_processing(arousal, valence, dominance)
    
    # Should be different when enabled
    assert enabled_params.attention_iterations != disabled_params.attention_iterations, \
        "Enabled should differ from disabled"
    assert enabled_params.ignition_threshold != disabled_params.ignition_threshold, \
        "Enabled should differ from disabled"
    assert enabled_params.decision_threshold != disabled_params.decision_threshold, \
        "Enabled should differ from disabled"
    
    print(f"✓ Enabled: iterations={enabled_params.attention_iterations}, "
          f"threshold={enabled_params.ignition_threshold:.2f}")
    print(f"✓ Disabled: iterations={disabled_params.attention_iterations}, "
          f"threshold={disabled_params.ignition_threshold:.2f}")
    print("✓ Ablation test passed: enabled modulation produces different behavior")


def test_metrics_correlations():
    """Test that metrics track correlations between emotions and parameters."""
    print("\n=== Testing Metrics Correlations ===")
    
    modulation = emotional_modulation.EmotionalModulation(enabled=True)
    
    # Generate diverse emotional states
    test_states = [
        (0.9, 0.7, 0.8),   # High arousal, positive, dominant
        (0.2, -0.6, 0.3),  # Low arousal, negative, submissive
        (0.8, -0.5, 0.4),  # High arousal, negative, moderate
        (0.1, 0.8, 0.9),   # Low arousal, positive, dominant (changed from 0.3)
        (0.5, 0.0, 0.5),   # Neutral
    ]
    
    for arousal, valence, dominance in test_states:
        modulation.modulate_processing(arousal, valence, dominance)
    
    metrics = modulation.get_metrics()
    
    assert metrics['total_modulations'] == 5
    assert metrics['arousal_effects']['correlations_count'] == 5
    assert metrics['valence_effects']['correlations_count'] == 5
    assert metrics['dominance_effects']['correlations_count'] == 5
    
    # Check that high/low arousal states were tracked
    assert metrics['arousal_effects']['high_arousal_fast'] >= 2, \
        "Should track high arousal states"
    assert metrics['arousal_effects']['low_arousal_slow'] >= 1, \
        "Should track low arousal states"
    
    print(f"✓ Metrics tracking: {metrics['total_modulations']} modulations")
    print(f"  - High arousal fast: {metrics['arousal_effects']['high_arousal_fast']}")
    print(f"  - Low arousal slow: {metrics['arousal_effects']['low_arousal_slow']}")
    print(f"  - Positive approach: {metrics['valence_effects']['positive_approach']}")
    print(f"  - Negative avoidance: {metrics['valence_effects']['negative_avoidance']}")


def test_realistic_scenarios():
    """Test realistic emotional scenarios."""
    print("\n=== Testing Realistic Scenarios ===")
    
    modulation = emotional_modulation.EmotionalModulation(enabled=True)
    
    # Scenario 1: Panic (fight-or-flight)
    print("\n1. Panic scenario (high arousal, negative valence, low dominance)")
    params = modulation.modulate_processing(arousal=0.95, valence=-0.8, dominance=0.15)
    assert params.attention_iterations <= 6, "Panic should produce very fast processing"
    assert params.decision_threshold >= 0.66, "Panic should require high confidence (low dominance)"
    print(f"   → Fast processing: {params.attention_iterations} iterations")
    print(f"   → Cautious: {params.decision_threshold:.2f} threshold")
    
    # Scenario 2: Flow state (moderate arousal, positive valence, high dominance)
    print("\n2. Flow scenario (moderate arousal, positive valence, high dominance)")
    params = modulation.modulate_processing(arousal=0.6, valence=0.7, dominance=0.85)
    assert params.attention_iterations <= 8, "Flow should be moderately fast"
    assert params.decision_threshold <= 0.60, "Flow should be assertive"
    print(f"   → Moderate speed: {params.attention_iterations} iterations")
    print(f"   → Assertive: {params.decision_threshold:.2f} threshold")
    
    # Scenario 3: Deep contemplation (low arousal, neutral valence, moderate dominance)
    print("\n3. Contemplation scenario (low arousal, neutral, moderate dominance)")
    params = modulation.modulate_processing(arousal=0.15, valence=0.1, dominance=0.5)
    assert params.attention_iterations >= 9, "Contemplation should be slow/thorough"
    assert params.ignition_threshold >= 0.57, "Contemplation should be selective"
    print(f"   → Thorough: {params.attention_iterations} iterations")
    print(f"   → Selective: {params.ignition_threshold:.2f} threshold")
    
    # Scenario 4: Joyful confidence (high arousal, high valence, high dominance)
    print("\n4. Joyful confidence (high arousal, positive valence, high dominance)")
    params = modulation.modulate_processing(arousal=0.8, valence=0.9, dominance=0.9)
    assert params.attention_iterations <= 7, "Joy should be moderately fast"
    assert params.decision_threshold <= 0.55, "Confidence should lower threshold"
    print(f"   → Energetic: {params.attention_iterations} iterations")
    print(f"   → Bold: {params.decision_threshold:.2f} threshold")


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("EMOTIONAL MODULATION INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        test_affect_integration()
        test_action_biasing()
        test_ablation_behavior_difference()
        test_metrics_correlations()
        test_realistic_scenarios()
        
        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
        print("\nEmotional modulation successfully:")
        print("  ✓ Modulates attention parameters based on arousal")
        print("  ✓ Biases action selection based on valence")
        print("  ✓ Adjusts decision thresholds based on dominance")
        print("  ✓ Produces measurably different behavior when enabled")
        print("  ✓ Tracks correlations between emotions and parameters")
        return 0
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
