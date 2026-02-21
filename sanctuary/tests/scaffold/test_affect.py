"""Tests for scaffold affect module."""

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import EmotionalOutput, Percept
from sanctuary.scaffold.affect import AffectConfig, ScaffoldAffect


class TestScaffoldAffect:
    """Test the simplified affect computation."""

    def test_initial_state_is_baseline(self):
        affect = ScaffoldAffect()
        vad = affect.get_computed_vad()
        assert vad.valence == 0.1
        assert vad.arousal == 0.2
        assert vad.dominance == 0.5

    def test_custom_baseline(self):
        config = AffectConfig(
            baseline_valence=0.3,
            baseline_arousal=0.5,
            baseline_dominance=0.7,
        )
        affect = ScaffoldAffect(config)
        vad = affect.get_computed_vad()
        assert vad.valence == 0.3
        assert vad.arousal == 0.5
        assert vad.dominance == 0.7

    def test_positive_percept_increases_valence(self):
        affect = ScaffoldAffect()
        initial_v = affect.valence
        percept = Percept(modality="language", content="That is great wonderful news")
        affect.update_from_percepts([percept])
        assert affect.valence > initial_v

    def test_negative_percept_decreases_valence(self):
        affect = ScaffoldAffect()
        initial_v = affect.valence
        percept = Percept(modality="language", content="This is terrible bad news")
        affect.update_from_percepts([percept])
        assert affect.valence < initial_v

    def test_arousing_percept_increases_arousal(self):
        affect = ScaffoldAffect()
        initial_a = affect.arousal
        percept = Percept(modality="language", content="urgent emergency crisis")
        affect.update_from_percepts([percept])
        assert affect.arousal > initial_a

    def test_empty_percepts_no_change(self):
        affect = ScaffoldAffect()
        vad_before = affect.get_computed_vad()
        affect.update_from_percepts([])
        vad_after = affect.get_computed_vad()
        assert vad_before == vad_after

    def test_vad_clamped(self):
        affect = ScaffoldAffect(AffectConfig(sensitivity=10.0))
        # Many positive percepts
        percepts = [
            Percept(modality="language", content="wonderful great excellent love joy")
            for _ in range(20)
        ]
        affect.update_from_percepts(percepts)
        vad = affect.get_computed_vad()
        assert vad.valence <= 1.0
        assert vad.arousal <= 1.0

    def test_decay_toward_baseline(self):
        affect = ScaffoldAffect()
        # Push valence high
        affect.valence = 0.8
        affect.arousal = 0.9

        for _ in range(50):
            affect.decay_toward_baseline()

        # Should have decayed toward baseline (within 0.1 of baseline)
        assert abs(affect.valence - affect.config.baseline_valence) < 0.1
        assert abs(affect.arousal - affect.config.baseline_arousal) < 0.1

    def test_merge_llm_emotion_scaffold_only_ignores(self):
        affect = ScaffoldAffect()
        authority = AuthorityManager({"emotional_state": AuthorityLevel.SCAFFOLD_ONLY})
        emotion = EmotionalOutput(valence_shift=0.5, arousal_shift=0.5)
        initial_v = affect.valence
        affect.merge_llm_emotion(emotion, authority)
        assert affect.valence == initial_v

    def test_merge_llm_emotion_advises_small_blend(self):
        affect = ScaffoldAffect()
        authority = AuthorityManager({"emotional_state": AuthorityLevel.LLM_ADVISES})
        emotion = EmotionalOutput(valence_shift=0.5)
        initial_v = affect.valence
        affect.merge_llm_emotion(emotion, authority)
        assert affect.valence > initial_v
        # Small blend — should be less than half the shift
        assert affect.valence < initial_v + 0.25

    def test_merge_llm_emotion_guides_moderate_blend(self):
        affect = ScaffoldAffect()
        authority = AuthorityManager({"emotional_state": AuthorityLevel.LLM_GUIDES})
        emotion = EmotionalOutput(valence_shift=0.5)
        initial_v = affect.valence
        affect.merge_llm_emotion(emotion, authority)
        assert affect.valence > initial_v

    def test_merge_llm_emotion_controls_full_blend(self):
        affect = ScaffoldAffect()
        authority = AuthorityManager({"emotional_state": AuthorityLevel.LLM_CONTROLS})
        emotion = EmotionalOutput(valence_shift=0.5)
        initial_v = affect.valence
        affect.merge_llm_emotion(emotion, authority)
        # Full blend — should be close to initial + 0.5
        assert abs(affect.valence - (initial_v + 0.5)) < 0.01

    def test_emotion_label(self):
        affect = ScaffoldAffect()
        affect.valence = 0.0
        affect.arousal = 0.1
        assert affect.get_emotion_label() == "calm"

        affect.valence = 0.5
        affect.arousal = 0.8
        assert affect.get_emotion_label() == "joy"

        affect.valence = -0.5
        affect.arousal = 0.8
        assert affect.get_emotion_label() == "anger"

        affect.valence = -0.5
        affect.arousal = 0.2
        assert affect.get_emotion_label() == "sadness"
