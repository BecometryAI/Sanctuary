#!/usr/bin/env python3
"""
Standalone test for enhanced emotional appraisal rules.
"""

import sys
from pathlib import Path

# Direct import
sys.path.insert(0, str(Path("emergence_core/lyra/cognitive_core")))

import importlib.util

# Import affect module directly
affect_path = Path("emergence_core/lyra/cognitive_core/affect.py")
spec = importlib.util.spec_from_file_location("affect", affect_path)
affect_module = importlib.util.module_from_spec(spec)

# Need to set up sys.modules to avoid dataclass issues
sys.modules['affect'] = affect_module
spec.loader.exec_module(affect_module)

AffectSubsystem = affect_module.AffectSubsystem
EmotionalState = affect_module.EmotionalState
EmotionCategory = affect_module.EmotionCategory


def test_emotion_categories():
    """Test emotion category enum."""
    print("=" * 70)
    print("TEST: Emotion Categories")
    print("=" * 70)
    
    categories = list(EmotionCategory)
    print(f"\n✅ Defined {len(categories)} emotion categories:")
    for cat in categories:
        print(f"  - {cat.value}")
    
    assert len(categories) == 8
    assert EmotionCategory.JOY in categories
    assert EmotionCategory.SADNESS in categories
    assert EmotionCategory.ANGER in categories
    assert EmotionCategory.FEAR in categories
    assert EmotionCategory.SURPRISE in categories
    assert EmotionCategory.DISGUST in categories
    
    return True


def test_vad_to_category_mapping():
    """Test VAD to emotion category mapping."""
    print("\n" + "=" * 70)
    print("TEST: VAD to Category Mapping")
    print("=" * 70)
    
    affect = AffectSubsystem()
    
    test_cases = [
        # (valence, arousal, dominance, expected_emotion)
        (0.8, 0.7, 0.7, EmotionCategory.JOY),
        (-0.6, 0.2, 0.3, EmotionCategory.SADNESS),
        (-0.7, 0.8, 0.8, EmotionCategory.ANGER),
        (-0.7, 0.8, 0.2, EmotionCategory.FEAR),
        (0.0, 0.9, 0.5, EmotionCategory.SURPRISE),
    ]
    
    passed = 0
    for v, a, d, expected in test_cases:
        affect.valence = v
        affect.arousal = a
        affect.dominance = d
        
        categories = affect.get_emotion_categories()
        label = affect.get_emotion_label()
        
        if expected in categories:
            print(f"  ✅ VAD({v:.1f}, {a:.1f}, {d:.1f}) → {expected.value}")
            passed += 1
        else:
            print(f"  ❌ VAD({v:.1f}, {a:.1f}, {d:.1f}) → Expected {expected.value}, got {label}")
    
    print(f"\n  Passed {passed}/{len(test_cases)} tests")
    return passed == len(test_cases)


def test_goal_achievement_joy():
    """Test that goal achievement generates joy."""
    print("\n" + "=" * 70)
    print("TEST: Goal Achievement → Joy")
    print("=" * 70)
    
    # Create mock goal (completed)
    class MockGoal:
        def __init__(self):
            self.progress = 1.0
            self.priority = 0.9
            self.metadata = {}
    
    affect = AffectSubsystem()
    
    # Start at neutral
    initial_valence = affect.valence
    initial_arousal = affect.arousal
    initial_dominance = affect.dominance
    
    print(f"\n  Initial state: V={initial_valence:.2f}, A={initial_arousal:.2f}, D={initial_dominance:.2f}")
    
    # Apply goal completion
    goals = [MockGoal()]
    deltas = affect._update_from_goals(goals)
    
    print(f"  Goal deltas: V={deltas['valence']:.2f}, A={deltas['arousal']:.2f}, D={deltas['dominance']:.2f}")
    
    # Check that valence, arousal, and dominance increased (joy)
    assert deltas["valence"] > 0, "Goal completion should increase valence (joy)"
    assert deltas["arousal"] > 0, "Goal completion should increase arousal (excitement)"
    assert deltas["dominance"] > 0, "Goal completion should increase dominance (agency)"
    
    print(f"  ✅ Goal achievement generates joy response")
    
    return True


def test_goal_failure_sadness():
    """Test that goal failure generates sadness."""
    print("\n" + "=" * 70)
    print("TEST: Goal Failure → Sadness")
    print("=" * 70)
    
    # Create mock goal (failed)
    class MockGoal:
        def __init__(self):
            self.progress = 0.1
            self.priority = 0.8
            self.metadata = {"failed": True}
    
    affect = AffectSubsystem()
    
    # Start at neutral
    initial_valence = affect.valence
    
    print(f"\n  Initial state: V={initial_valence:.2f}")
    
    # Apply goal failure
    goals = [MockGoal()]
    deltas = affect._update_from_goals(goals)
    
    print(f"  Goal deltas: V={deltas['valence']:.2f}, A={deltas['arousal']:.2f}, D={deltas['dominance']:.2f}")
    
    # Check that valence decreased (sadness)
    assert deltas["valence"] < 0, "Goal failure should decrease valence (sadness)"
    assert deltas["arousal"] < 0, "Goal failure should decrease arousal (low energy)"
    assert deltas["dominance"] < 0, "Goal failure should decrease dominance (helplessness)"
    
    print(f"  ✅ Goal failure generates sadness response")
    
    return True


def test_surprise_detection():
    """Test surprise detection from novelty."""
    print("\n" + "=" * 70)
    print("TEST: Novelty → Surprise")
    print("=" * 70)
    
    affect = AffectSubsystem()
    
    # Create percept with high novelty
    percepts = {
        "p1": {
            "raw": "Something completely unexpected just happened!",
            "modality": "text",
            "complexity": 10,
            "metadata": {"novelty": 0.9}
        }
    }
    
    initial_arousal = affect.arousal
    
    print(f"\n  Initial arousal: {initial_arousal:.2f}")
    
    # Apply percept
    deltas = affect._update_from_percepts(percepts)
    
    print(f"  Percept deltas: V={deltas['valence']:.2f}, A={deltas['arousal']:.2f}, D={deltas['dominance']:.2f}")
    
    # Check that arousal increased significantly (surprise)
    assert deltas["arousal"] > 0.3, "High novelty should trigger surprise (high arousal)"
    
    print(f"  ✅ Novelty generates surprise response")
    
    return True


def test_social_feedback():
    """Test emotional response to social feedback."""
    print("\n" + "=" * 70)
    print("TEST: Social Feedback")
    print("=" * 70)
    
    affect = AffectSubsystem()
    
    # Positive feedback
    praise_percepts = {
        "p1": {
            "raw": "Great job! You did really well!",
            "modality": "text",
            "complexity": 5,
            "metadata": {}
        }
    }
    
    deltas_praise = affect._update_from_percepts(praise_percepts)
    
    print(f"\n  Praise deltas: V={deltas_praise['valence']:.2f}, A={deltas_praise['arousal']:.2f}, D={deltas_praise['dominance']:.2f}")
    
    assert deltas_praise["valence"] > 0, "Praise should increase valence"
    assert deltas_praise["dominance"] > 0, "Praise should increase dominance"
    
    print(f"  ✅ Positive feedback generates positive emotion")
    
    # Negative feedback
    criticism_percepts = {
        "p1": {
            "raw": "That was wrong, you failed at this task.",
            "modality": "text",
            "complexity": 5,
            "metadata": {}
        }
    }
    
    deltas_criticism = affect._update_from_percepts(criticism_percepts)
    
    print(f"  Criticism deltas: V={deltas_criticism['valence']:.2f}, A={deltas_criticism['arousal']:.2f}, D={deltas_criticism['dominance']:.2f}")
    
    assert deltas_criticism["valence"] < 0, "Criticism should decrease valence"
    
    print(f"  ✅ Negative feedback generates negative emotion")
    
    return True


def test_value_alignment():
    """Test emotional response to value alignment."""
    print("\n" + "=" * 70)
    print("TEST: Value Alignment")
    print("=" * 70)
    
    affect = AffectSubsystem()
    
    # Value-aligned percept
    aligned_percepts = {
        "p1": {
            "raw": "This action aligns with core values",
            "modality": "introspection",
            "complexity": 5,
            "metadata": {"value_aligned": True}
        }
    }
    
    deltas = affect._update_from_percepts(aligned_percepts)
    
    print(f"\n  Value alignment deltas: V={deltas['valence']:.2f}, A={deltas['arousal']:.2f}, D={deltas['dominance']:.2f}")
    
    assert deltas["valence"] > 0, "Value alignment should increase valence"
    assert deltas["dominance"] > 0, "Value alignment should increase dominance"
    
    print(f"  ✅ Value alignment generates positive emotion")
    
    # Value conflict
    conflict_percepts = {
        "p1": {
            "raw": {"type": "value_conflict", "details": "Action conflicts with honesty value"},
            "modality": "introspection",
            "complexity": 5,
            "metadata": {}
        }
    }
    
    deltas_conflict = affect._update_from_percepts(conflict_percepts)
    
    print(f"  Value conflict deltas: V={deltas_conflict['valence']:.2f}, A={deltas_conflict['arousal']:.2f}, D={deltas_conflict['dominance']:.2f}")
    
    assert deltas_conflict["valence"] < 0, "Value conflict should decrease valence"
    
    print(f"  ✅ Value conflict generates negative emotion")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("ENHANCED EMOTIONAL APPRAISAL TEST SUITE")
    print("=" * 70)
    
    tests = [
        ("Emotion Categories", test_emotion_categories),
        ("VAD to Category Mapping", test_vad_to_category_mapping),
        ("Goal Achievement → Joy", test_goal_achievement_joy),
        ("Goal Failure → Sadness", test_goal_failure_sadness),
        ("Novelty → Surprise", test_surprise_detection),
        ("Social Feedback", test_social_feedback),
        ("Value Alignment", test_value_alignment),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
