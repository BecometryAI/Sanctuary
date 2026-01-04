"""
Property-based tests for emotional state.

Tests validate:
- VAD value bounds
- Emotional state structure
"""

import pytest
from hypothesis import given, settings
from .strategies import emotional_states


@pytest.mark.property
class TestEmotionProperties:
    
    @given(emotional_states())
    @settings(max_examples=50)
    def test_vad_bounds_enforced(self, emotion):
        """Property: VAD values always in [-1, 1] range."""
        assert -1.0 <= emotion['valence'] <= 1.0
        assert -1.0 <= emotion['arousal'] <= 1.0
        assert -1.0 <= emotion['dominance'] <= 1.0
    
    @given(emotional_states())
    @settings(max_examples=50)
    def test_emotional_state_has_required_keys(self, emotion):
        """Property: Emotional state has all required VAD keys."""
        assert 'valence' in emotion
        assert 'arousal' in emotion
        assert 'dominance' in emotion
    
    @given(emotional_states())
    @settings(max_examples=50)
    def test_emotional_state_values_are_floats(self, emotion):
        """Property: All VAD values are floats."""
        assert isinstance(emotion['valence'], float)
        assert isinstance(emotion['arousal'], float)
        assert isinstance(emotion['dominance'], float)
    
    @given(emotional_states())
    @settings(max_examples=50)
    def test_emotional_state_no_nan_or_inf(self, emotion):
        """Property: VAD values are not NaN or infinite."""
        import math
        
        assert not math.isnan(emotion['valence'])
        assert not math.isnan(emotion['arousal'])
        assert not math.isnan(emotion['dominance'])
        
        assert not math.isinf(emotion['valence'])
        assert not math.isinf(emotion['arousal'])
        assert not math.isinf(emotion['dominance'])
