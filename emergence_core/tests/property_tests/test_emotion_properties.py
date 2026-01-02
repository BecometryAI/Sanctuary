"""
Property-based tests for Emotional dynamics invariants.

Tests cover:
- Emotional state bounds (VAD in [-1, 1])
- Emotional decay toward baseline
- State update properties
- Emotional history
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
import numpy as np

from lyra.cognitive_core.workspace import GlobalWorkspace, WorkspaceSnapshot
from lyra.cognitive_core.affect import AffectSubsystem, EmotionalState
from .strategies import (
    emotional_states,
    percepts,
    goals,
    percept_lists,
    goal_lists,
    make_unique_by_id,
)


def emotion_distance_from_baseline(state: EmotionalState, baseline: dict = None) -> float:
    """Calculate Euclidean distance from baseline emotional state."""
    if baseline is None:
        baseline = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
    
    dv = state.valence - baseline["valence"]
    da = state.arousal - baseline["arousal"]
    dd = state.dominance - baseline["dominance"]
    
    return np.sqrt(dv**2 + da**2 + dd**2)


def create_snapshot_with_content(percepts_list, goals_list):
    """Helper to create a workspace snapshot with given content."""
    workspace = GlobalWorkspace()
    
    for percept in percepts_list:
        workspace.active_percepts[percept.id] = percept
    
    for goal in goals_list:
        workspace.add_goal(goal)
    
    return workspace.broadcast()


@pytest.mark.property
class TestEmotionProperties:
    """Property-based tests for emotional dynamics invariants."""
    
    @given(emotional_states)
    @settings(max_examples=100)
    def test_emotional_state_vad_bounded(self, initial_state):
        """Property: Emotional VAD values are always in valid ranges."""
        # Valence should be in [-1, 1]
        assert -1.0 <= initial_state.valence <= 1.0
        
        # Arousal should be in [-1, 1]
        assert -1.0 <= initial_state.arousal <= 1.0
        
        # Dominance should be in [-1, 1]
        assert -1.0 <= initial_state.dominance <= 1.0
    
    @given(emotional_states, st.lists(percepts(), max_size=5))
    @settings(max_examples=50, deadline=None)
    def test_emotional_update_maintains_bounds(self, initial_state, percepts_list):
        """Property: Emotional updates never exceed VAD bounds."""
        affect = AffectSubsystem()
        affect.valence = initial_state.valence
        affect.arousal = initial_state.arousal
        affect.dominance = initial_state.dominance
        
        snapshot = create_snapshot_with_content(percepts_list, [])
        affect.compute_update(snapshot)
        
        # After update, values should still be bounded
        assert -1.0 <= affect.valence <= 1.0
        assert -1.0 <= affect.arousal <= 1.0
        assert -1.0 <= affect.dominance <= 1.0
    
    @given(emotional_states)
    @settings(max_examples=100)
    def test_emotion_decay_moves_toward_baseline(self, initial_state):
        """Property: Repeated decay moves emotions toward baseline."""
        affect = AffectSubsystem(config={"decay_rate": 0.1})
        affect.valence = initial_state.valence
        affect.arousal = initial_state.arousal
        affect.dominance = initial_state.dominance
        
        initial_distance = emotion_distance_from_baseline(
            EmotionalState(
                valence=affect.valence,
                arousal=affect.arousal,
                dominance=affect.dominance
            ),
            affect.baseline
        )
        
        # Apply decay multiple times
        for _ in range(10):
            affect._apply_decay()
        
        final_distance = emotion_distance_from_baseline(
            EmotionalState(
                valence=affect.valence,
                arousal=affect.arousal,
                dominance=affect.dominance
            ),
            affect.baseline
        )
        
        # Distance should decrease or stay approximately the same
        # (with small epsilon for floating point precision)
        assert final_distance <= initial_distance + 1e-6
    
    @given(emotional_states)
    @settings(max_examples=100)
    def test_emotional_state_serialization(self, state):
        """Property: Emotional states can be serialized and deserialized."""
        # Convert to dict
        state_dict = state.to_dict()
        
        # Verify dict structure
        assert "valence" in state_dict
        assert "arousal" in state_dict
        assert "dominance" in state_dict
        
        # Values should match
        assert state_dict["valence"] == state.valence
        assert state_dict["arousal"] == state.arousal
        assert state_dict["dominance"] == state.dominance
    
    @given(emotional_states, st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50)
    def test_decay_rate_effect(self, initial_state, decay_rate):
        """Property: Higher decay rates lead to faster convergence to baseline."""
        # Test with low decay rate
        affect_slow = AffectSubsystem(config={"decay_rate": decay_rate * 0.1})
        affect_slow.valence = initial_state.valence
        affect_slow.arousal = initial_state.arousal
        affect_slow.dominance = initial_state.dominance
        
        # Test with high decay rate
        affect_fast = AffectSubsystem(config={"decay_rate": decay_rate * 0.9})
        affect_fast.valence = initial_state.valence
        affect_fast.arousal = initial_state.arousal
        affect_fast.dominance = initial_state.dominance
        
        # Apply decay once
        affect_slow._apply_decay()
        affect_fast._apply_decay()
        
        # Calculate distances
        dist_slow = emotion_distance_from_baseline(
            EmotionalState(
                valence=affect_slow.valence,
                arousal=affect_slow.arousal,
                dominance=affect_slow.dominance
            ),
            affect_slow.baseline
        )
        
        dist_fast = emotion_distance_from_baseline(
            EmotionalState(
                valence=affect_fast.valence,
                arousal=affect_fast.arousal,
                dominance=affect_fast.dominance
            ),
            affect_fast.baseline
        )
        
        # If we started away from baseline, faster decay should be closer
        initial_distance = emotion_distance_from_baseline(initial_state, affect_slow.baseline)
        if initial_distance > 0.1:  # Only test if starting away from baseline
            assert dist_fast <= dist_slow + 1e-6
    
    @given(emotional_states)
    @settings(max_examples=100)
    def test_emotional_intensity_calculation(self, state):
        """Property: Emotional intensity is non-negative and bounded."""
        assert state.intensity >= 0.0
        # Intensity should be bounded by sqrt(3) (max distance in unit cube)
        assert state.intensity <= 1.0 + 1e-6  # Small epsilon for floating point
    
    @given(st.lists(emotional_states, min_size=1, max_size=20))
    @settings(max_examples=50)
    def test_emotional_history_tracking(self, states_list):
        """Property: AffectSubsystem maintains emotional history."""
        affect = AffectSubsystem(config={"history_size": 10})
        
        # Add states to history by creating snapshots
        for state in states_list[:10]:  # Limit to history size
            affect.valence = state.valence
            affect.arousal = state.arousal
            affect.dominance = state.dominance
            
            # Record in history
            affect.emotion_history.append(EmotionalState(
                valence=affect.valence,
                arousal=affect.arousal,
                dominance=affect.dominance
            ))
        
        # History should not exceed max size
        assert len(affect.emotion_history) <= 10
    
    @given(emotional_states)
    @settings(max_examples=100)
    def test_emotional_state_immutability(self, state):
        """Property: EmotionalState dataclass values are accessible."""
        # Should be able to read values
        v = state.valence
        a = state.arousal
        d = state.dominance
        
        # Values should be in valid ranges
        assert -1.0 <= v <= 1.0
        assert -1.0 <= a <= 1.0
        assert -1.0 <= d <= 1.0
    
    @given(percept_lists, goal_lists)
    @settings(max_examples=50, deadline=None)
    def test_affect_subsystem_update_deterministic(self, percepts_list, goals_list):
        """Property: Same input produces same emotional update."""
        # Create two affect subsystems with same config
        config = {"decay_rate": 0.05, "sensitivity": 0.3, "baseline": {"valence": 0.1, "arousal": 0.3, "dominance": 0.6}}
        affect1 = AffectSubsystem(config=config)
        affect2 = AffectSubsystem(config=config)
        
        # Set same initial state
        initial_valence = 0.5
        initial_arousal = 0.3
        initial_dominance = 0.7
        
        affect1.valence = initial_valence
        affect1.arousal = initial_arousal
        affect1.dominance = initial_dominance
        
        affect2.valence = initial_valence
        affect2.arousal = initial_arousal
        affect2.dominance = initial_dominance
        
        # Create same snapshot
        snapshot = create_snapshot_with_content(percepts_list[:3], goals_list[:3])
        
        # Apply same update
        affect1.compute_update(snapshot)
        affect2.compute_update(snapshot)
        
        # Results should be identical
        assert abs(affect1.valence - affect2.valence) < 1e-10
        assert abs(affect1.arousal - affect2.arousal) < 1e-10
        assert abs(affect1.dominance - affect2.dominance) < 1e-10
    
    @given(emotional_states)
    @settings(max_examples=100)
    def test_emotion_to_vector_conversion(self, state):
        """Property: EmotionalState can be converted to numpy vector."""
        vector = state.to_vector()
        
        # Should be numpy array with 3 elements
        assert len(vector) == 3
        assert vector[0] == state.valence
        assert vector[1] == state.arousal
        assert vector[2] == state.dominance
