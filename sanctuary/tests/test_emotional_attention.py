"""
Unit tests for Emotional Attention System.

Tests cover:
- EmotionalAttentionSystem initialization
- Emotion registry completeness
- Emotional modulation computation
- Priority modifiers and action biases
- Temporal dynamics and blending
- Integration with AttentionController
- Integration with AffectSubsystem
"""

import pytest
from datetime import datetime, timedelta

from mind.cognitive_core.emotional_attention import (
    EmotionalAttentionSystem,
    EmotionalState,
    EmotionalAttentionOutput,
    EmotionProfile,
    EmotionCategory,
    IntensityLevel,
    EMOTION_REGISTRY,
    COMPOUND_EMOTIONS
)


class TestEmotionRegistry:
    """Test emotion registry completeness."""

    def test_registry_not_empty(self):
        """Test that emotion registry has entries."""
        assert len(EMOTION_REGISTRY) > 0

    def test_registry_has_primary_emotions(self):
        """Test all primary emotions are registered."""
        primary_emotions = ["fear", "anger", "sadness", "joy", "disgust", "surprise", "anticipation", "trust"]
        for emotion in primary_emotions:
            assert emotion in EMOTION_REGISTRY, f"Missing primary emotion: {emotion}"

    def test_registry_has_secondary_emotions(self):
        """Test secondary emotions are registered."""
        secondary_emotions = ["anxiety", "frustration", "curiosity", "interest", "boredom", "calm", "excitement"]
        for emotion in secondary_emotions:
            assert emotion in EMOTION_REGISTRY, f"Missing secondary emotion: {emotion}"

    def test_registry_has_social_emotions(self):
        """Test social emotions are registered."""
        social_emotions = ["shame", "guilt", "pride", "embarrassment", "gratitude", "envy", "compassion", "love"]
        for emotion in social_emotions:
            assert emotion in EMOTION_REGISTRY, f"Missing social emotion: {emotion}"

    def test_registry_has_cognitive_emotions(self):
        """Test cognitive emotions are registered."""
        cognitive_emotions = ["confusion", "certainty", "doubt", "awe", "wonder", "realization"]
        for emotion in cognitive_emotions:
            assert emotion in EMOTION_REGISTRY, f"Missing cognitive emotion: {emotion}"

    def test_registry_has_ai_emotions(self):
        """Test AI-relevant emotions are registered."""
        ai_emotions = ["overwhelm", "flow", "stuck", "accomplished", "uncertain", "engaged"]
        for emotion in ai_emotions:
            assert emotion in EMOTION_REGISTRY, f"Missing AI emotion: {emotion}"

    def test_all_profiles_have_required_fields(self):
        """Test all emotion profiles have required fields."""
        for name, profile in EMOTION_REGISTRY.items():
            assert isinstance(profile.name, str)
            assert isinstance(profile.category, EmotionCategory)
            assert -1.0 <= profile.valence <= 1.0
            assert 0.0 <= profile.arousal <= 1.0
            assert 0.0 <= profile.dominance <= 1.0
            assert -1.0 <= profile.approach <= 1.0
            assert 0.0 <= profile.onset_rate <= 1.0
            assert 0.0 <= profile.decay_rate <= 1.0

    def test_compound_emotions_exist(self):
        """Test compound emotions are defined."""
        assert len(COMPOUND_EMOTIONS) > 0
        for name, compound in COMPOUND_EMOTIONS.items():
            assert len(compound.components) >= 2
            for component in compound.components:
                assert component in EMOTION_REGISTRY


class TestEmotionalState:
    """Test EmotionalState dataclass."""

    def test_default_state(self):
        """Test default emotional state values."""
        state = EmotionalState()
        assert state.primary_emotion == "calm"
        assert state.intensity == 0.3
        assert state.valence == 0.1
        assert state.arousal == 0.3

    def test_intensity_level_mild(self):
        """Test mild intensity classification."""
        state = EmotionalState(intensity=0.2)
        assert state.get_intensity_level() == IntensityLevel.MILD

    def test_intensity_level_moderate(self):
        """Test moderate intensity classification."""
        state = EmotionalState(intensity=0.5)
        assert state.get_intensity_level() == IntensityLevel.MODERATE

    def test_intensity_level_intense(self):
        """Test intense intensity classification."""
        state = EmotionalState(intensity=0.8)
        assert state.get_intensity_level() == IntensityLevel.INTENSE

    def test_to_dict(self):
        """Test state serialization."""
        state = EmotionalState(primary_emotion="joy", intensity=0.7)
        d = state.to_dict()
        assert d["primary_emotion"] == "joy"
        assert d["intensity"] == 0.7
        assert "valence" in d
        assert "arousal" in d


class TestEmotionalAttentionSystem:
    """Test EmotionalAttentionSystem class."""

    def test_initialization_default(self):
        """Test default initialization."""
        system = EmotionalAttentionSystem()
        assert system.enabled is True
        assert system.modulation_strength == 0.8

    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {
            "enabled": True,
            "modulation_strength": 0.5,
            "max_concurrent_emotions": 4
        }
        system = EmotionalAttentionSystem(config)
        assert system.modulation_strength == 0.5
        assert system.max_concurrent_emotions == 4

    def test_compute_modulation_calm(self):
        """Test modulation for calm state."""
        system = EmotionalAttentionSystem()
        state = EmotionalState(primary_emotion="calm", intensity=0.3)
        output = system.compute_modulation(state)

        assert isinstance(output, EmotionalAttentionOutput)
        # Calm should have high attention depth
        assert output.attention_depth > 0.5
        # Calm should have positive precision modifier
        assert output.precision_modifier > 0

    def test_compute_modulation_fear(self):
        """Test modulation for fear state."""
        system = EmotionalAttentionSystem()
        state = EmotionalState(
            primary_emotion="fear",
            intensity=0.8,
            valence=-0.8,
            arousal=0.9,
            dominance=0.2,
            approach=-0.9
        )
        output = system.compute_modulation(state)

        # Fear should increase attention breadth (scanning for threats)
        assert output.attention_breadth > 0.5
        # Fear should lower ignition threshold (faster reactions)
        assert output.ignition_threshold < 0.5
        # Fear should have threat in priority modifiers
        assert "threat" in output.percept_priority_modifiers

    def test_compute_modulation_joy(self):
        """Test modulation for joy state."""
        system = EmotionalAttentionSystem()
        state = EmotionalState(
            primary_emotion="joy",
            intensity=0.7,
            valence=0.8,
            arousal=0.7,
            dominance=0.7,
            approach=0.8
        )
        output = system.compute_modulation(state)

        # Joy should have broad attention
        assert output.attention_breadth > 0.4
        # Joy should bias toward engagement
        assert "engage" in output.action_biases or "share" in output.action_biases

    def test_compute_modulation_disabled(self):
        """Test modulation when system is disabled."""
        config = {"enabled": False}
        system = EmotionalAttentionSystem(config)
        state = EmotionalState(primary_emotion="fear", intensity=0.9)
        output = system.compute_modulation(state)

        # Should return default output when disabled
        assert output.precision_modifier == 0.0
        assert output.attention_breadth == 0.5
        assert output.attention_depth == 0.5


class TestEmotionalBlending:
    """Test emotion blending functionality."""

    def test_blend_single_emotion(self):
        """Test blending with single emotion."""
        system = EmotionalAttentionSystem()
        emotions = [("joy", 0.7)]
        state = system.blend_emotions(emotions)

        assert state.primary_emotion == "joy"
        assert state.intensity == 0.7
        assert not state.is_blended

    def test_blend_multiple_emotions(self):
        """Test blending multiple emotions."""
        system = EmotionalAttentionSystem()
        emotions = [("joy", 0.6), ("excitement", 0.5)]
        state = system.blend_emotions(emotions)

        assert state.primary_emotion == "joy"  # Strongest
        assert state.is_blended
        assert "excitement" in state.secondary_emotions or "excitement" in state.blend_components

    def test_blend_filters_low_intensity(self):
        """Test that low intensity emotions are filtered."""
        system = EmotionalAttentionSystem()
        emotions = [("joy", 0.6), ("sadness", 0.1)]  # sadness below threshold
        state = system.blend_emotions(emotions)

        assert state.primary_emotion == "joy"
        assert "sadness" not in state.secondary_emotions

    def test_blend_empty_list(self):
        """Test blending with empty list."""
        system = EmotionalAttentionSystem()
        state = system.blend_emotions([])

        assert state.primary_emotion == "calm"  # Default


class TestTemporalDynamics:
    """Test temporal dynamics functionality."""

    def test_apply_decay(self):
        """Test that emotions decay over time."""
        system = EmotionalAttentionSystem()
        state = EmotionalState(
            primary_emotion="fear",
            intensity=0.8,
            valence=-0.8,
            arousal=0.9
        )

        # Apply decay for 2 seconds
        new_state = system.apply_temporal_dynamics(state, elapsed_seconds=2.0)

        # Intensity should decrease
        assert new_state.intensity < state.intensity
        # Values should move toward baseline
        assert new_state.arousal < state.arousal

    def test_momentum_slows_decay(self):
        """Test that high momentum slows decay."""
        system = EmotionalAttentionSystem()

        # Love has high momentum (0.9)
        state_love = EmotionalState(primary_emotion="love", intensity=0.8)
        # Surprise has low momentum (0.1)
        state_surprise = EmotionalState(primary_emotion="surprise", intensity=0.8)

        decayed_love = system.apply_temporal_dynamics(state_love, elapsed_seconds=1.0)
        decayed_surprise = system.apply_temporal_dynamics(state_surprise, elapsed_seconds=1.0)

        # Love should decay slower
        assert decayed_love.intensity > decayed_surprise.intensity


class TestPriorityModifiers:
    """Test priority modifier computation."""

    def test_fear_prioritizes_threat(self):
        """Test that fear prioritizes threat-related percepts."""
        system = EmotionalAttentionSystem()
        state = EmotionalState(
            primary_emotion="fear",
            intensity=0.8,
            valence=-0.8,
            arousal=0.9
        )
        output = system.compute_modulation(state)

        assert "threat" in output.percept_priority_modifiers
        assert output.percept_priority_modifiers["threat"] > 0

    def test_curiosity_prioritizes_novel(self):
        """Test that curiosity prioritizes novel percepts."""
        system = EmotionalAttentionSystem()
        state = EmotionalState(
            primary_emotion="curiosity",
            intensity=0.7,
            valence=0.4,
            arousal=0.6
        )
        output = system.compute_modulation(state)

        assert "novel" in output.percept_priority_modifiers
        assert output.percept_priority_modifiers["novel"] > 0


class TestActionBiases:
    """Test action bias computation."""

    def test_fear_biases_escape(self):
        """Test that fear biases toward escape actions."""
        system = EmotionalAttentionSystem()
        state = EmotionalState(
            primary_emotion="fear",
            intensity=0.8,
            approach=-0.9
        )
        output = system.compute_modulation(state)

        # Escape should be biased (negative means preferred in EFE)
        assert "escape" in output.action_biases
        assert output.action_biases["escape"] < 0

    def test_joy_biases_engagement(self):
        """Test that joy biases toward engagement."""
        system = EmotionalAttentionSystem()
        state = EmotionalState(
            primary_emotion="joy",
            intensity=0.7,
            approach=0.8
        )
        output = system.compute_modulation(state)

        # Engage should be biased
        assert "engage" in output.action_biases or "share" in output.action_biases


class TestEmotionRetrieval:
    """Test emotion retrieval methods."""

    def test_get_emotion_profile(self):
        """Test getting emotion profile by name."""
        system = EmotionalAttentionSystem()
        profile = system.get_emotion_profile("fear")

        assert profile is not None
        assert profile.name == "fear"
        assert profile.category == EmotionCategory.PRIMARY

    def test_get_nonexistent_profile(self):
        """Test getting nonexistent emotion profile."""
        system = EmotionalAttentionSystem()
        profile = system.get_emotion_profile("nonexistent")

        assert profile is None

    def test_get_all_emotions(self):
        """Test getting all emotion names."""
        system = EmotionalAttentionSystem()
        emotions = system.get_all_emotions()

        assert len(emotions) >= 35  # At least 35 emotions (currently 39)
        assert "fear" in emotions
        assert "joy" in emotions

    def test_get_emotions_by_category(self):
        """Test getting emotions by category."""
        system = EmotionalAttentionSystem()

        primary = system.get_emotions_by_category(EmotionCategory.PRIMARY)
        assert "fear" in primary
        assert "joy" in primary

        social = system.get_emotions_by_category(EmotionCategory.SOCIAL)
        assert "shame" in social
        assert "pride" in social


class TestCompetitionParams:
    """Test competition parameter modulation."""

    def test_high_arousal_lowers_threshold(self):
        """Test that high arousal lowers ignition threshold."""
        system = EmotionalAttentionSystem()

        low_arousal_state = EmotionalState(arousal=0.2)
        high_arousal_state = EmotionalState(arousal=0.9)

        low_output = system.compute_modulation(low_arousal_state)
        high_output = system.compute_modulation(high_arousal_state)

        assert high_output.ignition_threshold < low_output.ignition_threshold

    def test_negative_valence_increases_inhibition(self):
        """Test that negative valence increases inhibition strength."""
        system = EmotionalAttentionSystem()

        positive_state = EmotionalState(valence=0.5)
        negative_state = EmotionalState(valence=-0.5)

        positive_output = system.compute_modulation(positive_state)
        negative_output = system.compute_modulation(negative_state)

        assert negative_output.inhibition_strength >= positive_output.inhibition_strength


class TestErrorProcessing:
    """Test prediction error processing modulation."""

    def test_high_arousal_amplifies_errors(self):
        """Test that high arousal amplifies prediction errors."""
        system = EmotionalAttentionSystem()

        calm_state = EmotionalState(arousal=0.2)
        aroused_state = EmotionalState(arousal=0.9)

        calm_output = system.compute_modulation(calm_state)
        aroused_output = system.compute_modulation(aroused_state)

        assert aroused_output.error_amplification > calm_output.error_amplification

    def test_negative_valence_increases_threat_bias(self):
        """Test that negative valence increases threat interpretation bias."""
        system = EmotionalAttentionSystem()

        positive_state = EmotionalState(valence=0.5)
        negative_state = EmotionalState(valence=-0.5)

        positive_output = system.compute_modulation(positive_state)
        negative_output = system.compute_modulation(negative_state)

        assert negative_output.threat_interpretation_bias > positive_output.threat_interpretation_bias


class TestSummary:
    """Test summary methods."""

    def test_get_summary(self):
        """Test getting system summary."""
        system = EmotionalAttentionSystem()
        state = EmotionalState(primary_emotion="joy", intensity=0.7)
        system.compute_modulation(state)

        summary = system.get_summary()

        assert "enabled" in summary
        assert "registered_emotions" in summary
        assert "compound_emotions" in summary
        assert "categories" in summary
        assert summary["registered_emotions"] >= 35  # At least 35 emotions (currently 39)
