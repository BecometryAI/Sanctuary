#!/usr/bin/env python3
"""
Standalone test runner for emotional_modulation.py
Tests the module directly without importing the full lyra package.
"""

import sys
from pathlib import Path

# Add the cognitive_core directory to the path
cognitive_core_path = Path(__file__).parent.parent / "lyra" / "cognitive_core"
sys.path.insert(0, str(cognitive_core_path))

# Now import the module directly
import emotional_modulation

def test_basic_functionality():
    """Test basic emotional modulation functionality."""
    print("Testing EmotionalModulation class...")
    
    # Test initialization
    modulation = emotional_modulation.EmotionalModulation(enabled=True)
    assert modulation.enabled is True
    print("✓ Initialization successful")
    
    # Test input validation
    try:
        modulation.modulate_processing(arousal=1.5, valence=0.0, dominance=0.5)
        assert False, "Should have raised ValueError for invalid arousal"
    except ValueError as e:
        assert "Arousal must be in" in str(e)
        print("✓ Input validation: arousal range checked")
    
    try:
        modulation.modulate_processing(arousal=0.5, valence=2.0, dominance=0.5)
        assert False, "Should have raised ValueError for invalid valence"
    except ValueError as e:
        assert "Valence must be in" in str(e)
        print("✓ Input validation: valence range checked")
    
    try:
        modulation.modulate_processing(arousal=0.5, valence=0.0, dominance=1.5)
        assert False, "Should have raised ValueError for invalid dominance"
    except ValueError as e:
        assert "Dominance must be in" in str(e)
        print("✓ Input validation: dominance range checked")
    
    # Test high arousal produces fast processing
    params = modulation.modulate_processing(arousal=0.9, valence=0.0, dominance=0.5)
    assert params.attention_iterations < 7, f"Expected <7, got {params.attention_iterations}"
    assert params.ignition_threshold < 0.5, f"Expected <0.5, got {params.ignition_threshold}"
    print(f"✓ High arousal modulation: iterations={params.attention_iterations}, threshold={params.ignition_threshold:.2f}")
    
    # Test low arousal produces slow processing
    params = modulation.modulate_processing(arousal=0.1, valence=0.0, dominance=0.5)
    assert params.attention_iterations > 7, f"Expected >7, got {params.attention_iterations}"
    assert params.ignition_threshold > 0.5, f"Expected >0.5, got {params.ignition_threshold}"
    print(f"✓ Low arousal modulation: iterations={params.attention_iterations}, threshold={params.ignition_threshold:.2f}")
    
    # Test dominance modulation
    params_high_dom = modulation.modulate_processing(arousal=0.5, valence=0.0, dominance=0.9)
    assert params_high_dom.decision_threshold < 0.7, f"Expected <0.7, got {params_high_dom.decision_threshold}"
    print(f"✓ High dominance: decision_threshold={params_high_dom.decision_threshold:.2f}")
    
    params_low_dom = modulation.modulate_processing(arousal=0.5, valence=0.0, dominance=0.1)
    assert params_low_dom.decision_threshold > 0.65, f"Expected >0.65, got {params_low_dom.decision_threshold}"
    print(f"✓ Low dominance: decision_threshold={params_low_dom.decision_threshold:.2f}")
    
    # Test action biasing with positive valence
    actions = [
        {'type': 'speak', 'priority': 0.5},
        {'type': 'wait', 'priority': 0.5}
    ]
    biased = modulation.bias_action_selection(actions, valence=0.8)
    speak_priority = next(a for a in biased if a['type'] == 'speak')['priority']
    wait_priority = next(a for a in biased if a['type'] == 'wait')['priority']
    assert speak_priority > 0.5, f"Expected speak >0.5, got {speak_priority}"
    assert wait_priority < 0.5, f"Expected wait <0.5, got {wait_priority}"
    print(f"✓ Positive valence biasing: speak={speak_priority:.2f}, wait={wait_priority:.2f}")
    
    # Test action biasing with negative valence
    actions = [
        {'type': 'speak', 'priority': 0.5},
        {'type': 'wait', 'priority': 0.5}
    ]
    biased = modulation.bias_action_selection(actions, valence=-0.8)
    speak_priority = next(a for a in biased if a['type'] == 'speak')['priority']
    wait_priority = next(a for a in biased if a['type'] == 'wait')['priority']
    assert speak_priority < 0.5, f"Expected speak <0.5, got {speak_priority}"
    assert wait_priority > 0.5, f"Expected wait >0.5, got {wait_priority}"
    print(f"✓ Negative valence biasing: speak={speak_priority:.2f}, wait={wait_priority:.2f}")
    
    # Test ablation (disabled state)
    modulation_disabled = emotional_modulation.EmotionalModulation(enabled=False)
    params_baseline = modulation_disabled.modulate_processing(arousal=1.0, valence=1.0, dominance=1.0)
    baseline = modulation_disabled.baseline_params
    assert params_baseline.attention_iterations == baseline.attention_iterations
    assert params_baseline.ignition_threshold == baseline.ignition_threshold
    print("✓ Ablation test: disabled modulation returns baseline")
    
    # Test metrics tracking
    assert modulation.metrics.total_modulations > 0
    assert modulation.metrics.high_arousal_fast_processing > 0
    assert modulation.metrics.low_arousal_slow_processing > 0
    print(f"✓ Metrics tracking: {modulation.metrics.total_modulations} modulations recorded")
    
    metrics_dict = modulation.get_metrics()
    assert 'total_modulations' in metrics_dict
    assert 'arousal_effects' in metrics_dict
    print("✓ Metrics export successful")
    
    print("\n✅ All tests passed!")
    return True

if __name__ == '__main__':
    try:
        test_basic_functionality()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
