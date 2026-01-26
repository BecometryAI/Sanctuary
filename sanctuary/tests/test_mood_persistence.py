"""
Unit tests for Mood Persistence in AffectSubsystem.

Tests cover:
- Emotional momentum (resistance to change)
- Mood-congruent processing bias
- Refractory periods
- Emotion-specific decay rates
- Mood vs transient emotion tracking
- Smooth emotion transitions
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch
import time

from mind.cognitive_core.affect import AffectSubsystem, EmotionalState
from mind.cognitive_core.workspace import WorkspaceSnapshot, Goal
from mind.cognitive_core.emotional_attention import EMOTION_REGISTRY


class TestMoodPersistenceInit:
    """Test mood persistence initialization."""

    def test_default_config(self):
        """Test default mood persistence configuration."""
        affect = AffectSubsystem()

        assert affect._momentum_enabled is True
        assert affect._refractory_enabled is True
        assert affect._mood_congruence_enabled is True
        assert affect._current_emotion == "calm"
        assert affect._is_mood is False

    def test_custom_config(self):
        """Test custom mood persistence configuration."""
        config = {
            "momentum_enabled": False,
            "refractory_enabled": False,
            "mood_congruence_enabled": False,
            "momentum_strength": 0.9,
            "mood_threshold_duration": 60.0
        }
        affect = AffectSubsystem(config=config)

        assert affect._momentum_enabled is False
        assert affect._refractory_enabled is False
        assert affect._mood_congruence_enabled is False
        assert affect._momentum_strength == 0.9
        assert affect._mood_threshold_duration == 60.0


class TestEmotionalMomentum:
    """Test emotional momentum (resistance to change)."""

    def test_momentum_dampens_changes(self):
        """Test that momentum reduces emotional change magnitude."""
        # High momentum config
        affect_high = AffectSubsystem(config={
            "momentum_enabled": True,
            "momentum_strength": 0.9
        })
        affect_high.set_mood("calm", 0.5)

        # No momentum config
        affect_none = AffectSubsystem(config={
            "momentum_enabled": False
        })
        affect_none.set_mood("calm", 0.5)

        # Apply same delta to both
        delta = {"valence": 0.5, "arousal": 0.3, "dominance": 0.2}

        dampened = affect_high._apply_momentum(delta)
        undampened = affect_none._apply_momentum(delta)

        # High momentum should dampen more
        assert dampened["valence"] < undampened["valence"]
        assert dampened["arousal"] < undampened["arousal"]

    def test_mood_has_extra_momentum(self):
        """Test that moods (persistent emotions) have extra resistance."""
        affect = AffectSubsystem(config={"momentum_strength": 0.5})

        # Not a mood
        affect._is_mood = False
        delta = {"valence": 0.5, "arousal": 0.3, "dominance": 0.2}
        dampened_transient = affect._apply_momentum(delta.copy())

        # Is a mood
        affect._is_mood = True
        dampened_mood = affect._apply_momentum(delta.copy())

        # Mood should be more dampened
        assert dampened_mood["valence"] < dampened_transient["valence"]

    def test_emotion_specific_momentum(self):
        """Test that different emotions have different momentum."""
        affect = AffectSubsystem()

        # Love has high momentum (0.9)
        affect._current_emotion = "love"
        delta = {"valence": 0.5, "arousal": 0.3, "dominance": 0.2}
        love_dampened = affect._apply_momentum(delta.copy())

        # Surprise has low momentum (0.1)
        affect._current_emotion = "surprise"
        surprise_dampened = affect._apply_momentum(delta.copy())

        # Love should be more resistant to change
        assert love_dampened["valence"] < surprise_dampened["valence"]


class TestMoodCongruentBias:
    """Test mood-congruent processing bias."""

    def test_positive_mood_amplifies_positive(self):
        """Test that positive mood amplifies positive events."""
        affect = AffectSubsystem(config={
            "mood_congruence_enabled": True,
            "mood_congruence_strength": 0.5
        })
        affect.valence = 0.5  # Positive mood

        positive_delta = {"valence": 0.3, "arousal": 0.0, "dominance": 0.0}
        biased = affect._apply_mood_congruent_bias(positive_delta)

        # Positive should be amplified
        assert biased["valence"] > positive_delta["valence"]

    def test_positive_mood_dampens_negative(self):
        """Test that positive mood dampens negative events."""
        affect = AffectSubsystem(config={
            "mood_congruence_enabled": True,
            "mood_congruence_strength": 0.5
        })
        affect.valence = 0.5  # Positive mood

        negative_delta = {"valence": -0.3, "arousal": 0.0, "dominance": 0.0}
        biased = affect._apply_mood_congruent_bias(negative_delta)

        # Negative should be dampened (less negative)
        assert biased["valence"] > negative_delta["valence"]

    def test_negative_mood_amplifies_negative(self):
        """Test that negative mood amplifies negative events."""
        affect = AffectSubsystem(config={
            "mood_congruence_enabled": True,
            "mood_congruence_strength": 0.5
        })
        affect.valence = -0.5  # Negative mood

        negative_delta = {"valence": -0.3, "arousal": 0.0, "dominance": 0.0}
        biased = affect._apply_mood_congruent_bias(negative_delta)

        # Negative should be amplified (more negative)
        assert biased["valence"] < negative_delta["valence"]

    def test_high_arousal_amplifies_arousing(self):
        """Test that high arousal amplifies arousing events."""
        affect = AffectSubsystem(config={
            "mood_congruence_enabled": True,
            "mood_congruence_strength": 0.5
        })
        affect.arousal = 0.8  # High arousal

        arousing_delta = {"valence": 0.0, "arousal": 0.3, "dominance": 0.0}
        biased = affect._apply_mood_congruent_bias(arousing_delta)

        # Arousal should be amplified
        assert biased["arousal"] > arousing_delta["arousal"]


class TestRefractoryPeriods:
    """Test refractory period functionality."""

    def test_same_emotion_always_allowed(self):
        """Test that transitioning to same emotion is always allowed."""
        affect = AffectSubsystem(config={"refractory_enabled": True})
        affect._current_emotion = "joy"

        assert affect._can_transition_to("joy") is True

    def test_refractory_blocks_return(self):
        """Test that refractory period blocks return to previous emotion."""
        affect = AffectSubsystem(config={"refractory_enabled": True})
        affect._current_emotion = "joy"

        # Set refractory for fear
        affect._refractory_until["fear"] = datetime.now() + timedelta(seconds=10)

        # Should not be able to transition to fear
        assert affect._can_transition_to("fear") is False

    def test_refractory_expires(self):
        """Test that refractory period expires."""
        affect = AffectSubsystem(config={"refractory_enabled": True})

        # Set expired refractory
        affect._refractory_until["fear"] = datetime.now() - timedelta(seconds=1)

        # Should now be allowed
        assert affect._can_transition_to("fear") is True
        # Should have been cleaned up
        assert "fear" not in affect._refractory_until

    def test_emotion_change_sets_refractory(self):
        """Test that changing emotion sets refractory for old emotion."""
        affect = AffectSubsystem(config={"refractory_enabled": True})
        affect._current_emotion = "joy"

        # Update tracking to new emotion
        affect._update_emotion_tracking("fear")

        # Joy should now have refractory period
        assert "joy" in affect._refractory_until


class TestEmotionSpecificDecay:
    """Test emotion-specific decay rates."""

    def test_different_emotions_decay_differently(self):
        """Test that emotions with different decay rates decay differently."""
        # Fear has high decay rate (0.7)
        affect_fear = AffectSubsystem()
        affect_fear._current_emotion = "fear"
        affect_fear.valence = -0.8
        affect_fear.arousal = 0.9
        affect_fear.dominance = 0.2
        initial_fear = affect_fear.valence

        # Love has low decay rate (0.05)
        affect_love = AffectSubsystem()
        affect_love._current_emotion = "love"
        affect_love.valence = 0.8
        affect_love.arousal = 0.5
        affect_love.dominance = 0.6
        initial_love = affect_love.valence

        # Apply decay
        affect_fear._apply_decay_with_profiles()
        affect_love._apply_decay_with_profiles()

        # Fear should have decayed more toward baseline
        fear_change = abs(affect_fear.valence - initial_fear)
        love_change = abs(affect_love.valence - initial_love)

        assert fear_change > love_change

    def test_mood_decays_slower(self):
        """Test that moods decay slower than transient emotions."""
        affect_transient = AffectSubsystem()
        affect_transient._current_emotion = "joy"
        affect_transient._is_mood = False
        affect_transient.valence = 0.8
        initial_transient = affect_transient.valence

        affect_mood = AffectSubsystem()
        affect_mood._current_emotion = "joy"
        affect_mood._is_mood = True
        affect_mood.valence = 0.8
        initial_mood = affect_mood.valence

        # Apply decay
        affect_transient._apply_decay_with_profiles()
        affect_mood._apply_decay_with_profiles()

        # Mood should have decayed less
        transient_change = abs(affect_transient.valence - initial_transient)
        mood_change = abs(affect_mood.valence - initial_mood)

        assert mood_change < transient_change


class TestMoodTracking:
    """Test mood vs transient emotion tracking."""

    def test_initial_state_is_not_mood(self):
        """Test that initial emotional state is not a mood."""
        affect = AffectSubsystem()
        assert affect._is_mood is False

    def test_persistent_emotion_becomes_mood(self):
        """Test that persistent emotion becomes a mood."""
        affect = AffectSubsystem(config={"mood_threshold_duration": 0.1})  # 100ms

        # Update with same emotion
        affect._update_emotion_tracking("joy")

        # Wait for threshold
        time.sleep(0.15)

        # Update again
        affect._update_emotion_tracking("joy")

        # Should now be a mood
        assert affect._is_mood is True

    def test_emotion_change_resets_mood(self):
        """Test that emotion change resets mood status."""
        affect = AffectSubsystem()
        affect._is_mood = True
        affect._current_emotion = "joy"

        # Change emotion
        affect._update_emotion_tracking("fear")

        # Should no longer be a mood
        assert affect._is_mood is False

    def test_get_mood_state(self):
        """Test get_mood_state returns correct info."""
        affect = AffectSubsystem()
        affect._current_emotion = "joy"
        affect._is_mood = True
        affect._emotion_intensity = 0.7

        state = affect.get_mood_state()

        assert state["current_emotion"] == "joy"
        assert state["is_mood"] is True
        assert state["emotion_intensity"] == 0.7
        assert "emotion_duration_seconds" in state


class TestSetMood:
    """Test direct mood setting."""

    def test_set_valid_mood(self):
        """Test setting a valid mood."""
        affect = AffectSubsystem()
        affect.set_mood("joy", 0.8)

        assert affect._current_emotion == "joy"
        assert affect._is_mood is True
        assert affect._emotion_intensity == 0.8
        assert affect.valence > 0  # Joy has positive valence

    def test_set_invalid_mood_defaults(self):
        """Test setting invalid mood defaults to calm."""
        affect = AffectSubsystem()
        affect.set_mood("nonexistent", 0.5)

        assert affect._current_emotion == "calm"

    def test_set_mood_intensity_scales(self):
        """Test that intensity scales VAD values."""
        affect_low = AffectSubsystem()
        affect_low.set_mood("fear", 0.3)

        affect_high = AffectSubsystem()
        affect_high.set_mood("fear", 0.9)

        # High intensity should have more extreme values
        assert abs(affect_high.valence) > abs(affect_low.valence)
        assert abs(affect_high.arousal) > abs(affect_low.arousal)


class TestSmoothTransitions:
    """Test smooth emotion transitions."""

    def test_transitions_are_gradual(self):
        """Test that emotion changes are gradual, not instant."""
        affect = AffectSubsystem(config={"transition_rate": 0.3})
        affect.valence = 0.0
        affect.arousal = 0.0
        affect.dominance = 0.5

        # Large delta
        delta = {"valence": 1.0, "arousal": 1.0, "dominance": 0.0}
        affect._apply_deltas_with_smoothing(delta)

        # Should not have jumped to full values
        assert affect.valence < 1.0
        assert affect.arousal < 1.0


class TestIntegration:
    """Integration tests for mood persistence."""

    def test_compute_update_uses_mood_persistence(self):
        """Test that compute_update uses all mood persistence features."""
        affect = AffectSubsystem(config={
            "momentum_enabled": True,
            "refractory_enabled": True,
            "mood_congruence_enabled": True
        })

        # Create a minimal snapshot
        snapshot = WorkspaceSnapshot(
            percepts={},
            goals=[],
            broadcasts=[],
            actions=[],
            emotions={},
            memories=[],
            cycle_count=1,
            timestamp=datetime.now()
        )

        # Should not raise any errors
        result = affect.compute_update(snapshot)

        assert "valence" in result
        assert "arousal" in result
        assert "dominance" in result

    def test_get_state_includes_mood_info(self):
        """Test that get_state includes mood information."""
        affect = AffectSubsystem()
        affect._is_mood = True

        state = affect.get_state()

        assert "is_mood" in state
        assert state["is_mood"] is True
        assert "intensity" in state
        assert "emotion_duration" in state

    def test_full_mood_lifecycle(self):
        """Test full lifecycle: transient → mood → change → transient."""
        affect = AffectSubsystem(config={"mood_threshold_duration": 0.1})

        # Start transient
        assert affect._is_mood is False

        # Set emotion and wait
        affect._update_emotion_tracking("joy")
        time.sleep(0.15)
        affect._update_emotion_tracking("joy")

        # Now a mood
        assert affect._is_mood is True

        # Change emotion
        affect._update_emotion_tracking("sadness")

        # Back to transient
        assert affect._is_mood is False
        assert affect._current_emotion == "sadness"


class TestTargetEmotionDetection:
    """Test target emotion detection."""

    def test_detects_correct_target(self):
        """Test that target emotion is correctly detected."""
        affect = AffectSubsystem()
        affect.valence = 0.0
        affect.arousal = 0.0
        affect.dominance = 0.5

        # Delta toward fear (negative valence, high arousal, low dominance)
        delta = {"valence": -0.7, "arousal": 0.8, "dominance": -0.3}
        target = affect._detect_target_emotion(delta)

        assert target == "fear"

    def test_detects_joy_target(self):
        """Test detection of joy as target."""
        affect = AffectSubsystem()
        affect.valence = 0.0
        affect.arousal = 0.3
        affect.dominance = 0.5

        # Delta toward joy (positive valence, high arousal, high dominance)
        delta = {"valence": 0.8, "arousal": 0.4, "dominance": 0.2}
        target = affect._detect_target_emotion(delta)

        assert target == "joy"
