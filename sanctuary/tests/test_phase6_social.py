"""Tests for Phase 6.3: Social & Interactive.

Tests cover:
- MultiPartyManager: participant management, turn-taking, addressee detection
- ProsodyAnalyzer: audio feature analysis, emotional tone classification, calibration
- UserModeler: profile building, preference inference, relationship tracking
"""

import pytest

from sanctuary.social.multi_party import (
    MultiPartyConfig,
    MultiPartyManager,
    ParticipantStatus,
)
from sanctuary.social.prosody import (
    AudioFeatures,
    ProsodyAnalyzer,
    ProsodyConfig,
)
from sanctuary.social.user_modeling import (
    UserModeler,
    UserModelingConfig,
)


# =========================================================================
# MultiPartyManager
# =========================================================================


class TestMultiPartyManager:
    """Tests for multi-party conversation management."""

    def test_self_participant_exists(self):
        mp = MultiPartyManager()
        assert "sanctuary" in mp._participants
        assert mp._participants["sanctuary"].is_self

    def test_add_participant(self):
        mp = MultiPartyManager()
        result = mp.add_participant("user1", "Alice")
        assert result is True
        assert "user1" in mp._participants

    def test_add_participant_max_limit(self):
        config = MultiPartyConfig(max_participants=3)
        mp = MultiPartyManager(config=config)
        mp.add_participant("u1", "A")
        mp.add_participant("u2", "B")
        # 3rd slot taken by self
        result = mp.add_participant("u3", "C")
        assert result is False

    def test_remove_participant(self):
        mp = MultiPartyManager()
        mp.add_participant("user1", "Alice")
        result = mp.remove_participant("user1")
        assert result is True
        assert mp._participants["user1"].status == ParticipantStatus.LEFT

    def test_cannot_remove_self(self):
        mp = MultiPartyManager()
        result = mp.remove_participant("sanctuary")
        assert result is False

    def test_record_turn(self):
        mp = MultiPartyManager()
        mp.add_participant("user1", "Alice")
        mp.record_turn(speaker_id="user1", content="Hello!", cycle=1)
        assert len(mp._turns) == 1
        assert mp._participants["user1"].messages_sent == 1

    def test_addressee_detection_mention(self):
        mp = MultiPartyManager()
        mp.add_participant("user1", "Alice")
        mp.record_turn(
            speaker_id="user1",
            content="Hey @sanctuary, what do you think?",
            cycle=1,
        )
        assert "sanctuary" in mp._turns[-1].addressee_ids

    def test_should_respond_when_addressed(self):
        mp = MultiPartyManager()
        mp.add_participant("user1", "Alice")
        mp.record_turn(
            speaker_id="user1",
            content="@sanctuary help me",
            cycle=1,
        )
        should, reason = mp.should_respond(current_cycle=1)
        assert should is True
        assert "directly addressed" in reason

    def test_should_not_respond_to_self(self):
        mp = MultiPartyManager()
        mp.record_turn(speaker_id="sanctuary", content="I said something", cycle=1)
        should, reason = mp.should_respond(current_cycle=2)
        assert should is False

    def test_should_respond_group_question_after_patience(self):
        config = MultiPartyConfig(turn_taking_patience=3)
        mp = MultiPartyManager(config=config)
        mp.add_participant("user1", "Alice")
        mp.record_turn(speaker_id="user1", content="What do we all think?", cycle=1)
        # Not enough patience
        should, _ = mp.should_respond(current_cycle=2)
        assert should is False
        # After patience
        should, reason = mp.should_respond(current_cycle=5)
        assert should is True
        assert "group question" in reason

    def test_conversation_context(self):
        mp = MultiPartyManager()
        mp.add_participant("user1", "Alice")
        mp.record_turn(speaker_id="user1", content="Hello!", cycle=1)
        mp.record_turn(speaker_id="sanctuary", content="Hi Alice!", cycle=2)
        context = mp.get_conversation_context(n_turns=5)
        assert "Alice" in context
        assert "Sanctuary" in context

    def test_update_statuses(self):
        config = MultiPartyConfig(idle_threshold_cycles=10, away_threshold_cycles=20)
        mp = MultiPartyManager(config=config)
        mp.add_participant("user1", "Alice")
        mp.record_turn(speaker_id="user1", content="hi", cycle=1)
        mp.update_statuses(current_cycle=15)
        assert mp._participants["user1"].status == ParticipantStatus.IDLE
        mp.update_statuses(current_cycle=25)
        assert mp._participants["user1"].status == ParticipantStatus.AWAY

    def test_active_participants(self):
        mp = MultiPartyManager()
        mp.add_participant("user1", "Alice")
        mp.add_participant("user2", "Bob")
        active = mp.get_active_participants()
        assert len(active) == 2

    def test_addressee_for_response(self):
        mp = MultiPartyManager()
        mp.add_participant("user1", "Alice")
        mp.record_turn(
            speaker_id="user1",
            content="@sanctuary help",
            cycle=1,
        )
        addressees = mp.get_addressee_for_response()
        assert "user1" in addressees

    def test_stats(self):
        mp = MultiPartyManager()
        mp.add_participant("user1", "Alice")
        mp.record_turn(speaker_id="user1", content="hi", cycle=1)
        stats = mp.get_stats()
        assert stats["total_participants"] == 1
        assert stats["total_turns"] == 1


# =========================================================================
# ProsodyAnalyzer
# =========================================================================


class TestProsodyAnalyzer:
    """Tests for voice prosody analysis."""

    def test_analyze_neutral(self):
        analyzer = ProsodyAnalyzer()
        features = AudioFeatures(
            pitch_mean=0.5, energy_mean=0.5,
            speaking_rate=0.5, pause_ratio=0.2,
        )
        result = analyzer.analyze(features)
        assert -1.0 <= result.valence <= 1.0
        assert 0.0 <= result.arousal <= 1.0
        assert 0.0 <= result.dominance <= 1.0

    def test_high_energy_increases_arousal(self):
        analyzer = ProsodyAnalyzer()
        low = analyzer.analyze(AudioFeatures(energy_mean=0.2))
        high = analyzer.analyze(AudioFeatures(energy_mean=0.9))
        assert high.arousal > low.arousal

    def test_high_pitch_increases_arousal(self):
        analyzer = ProsodyAnalyzer()
        low = analyzer.analyze(AudioFeatures(pitch_mean=0.2))
        high = analyzer.analyze(AudioFeatures(pitch_mean=0.9))
        assert high.arousal > low.arousal

    def test_fast_rate_increases_arousal(self):
        analyzer = ProsodyAnalyzer()
        slow = analyzer.analyze(AudioFeatures(speaking_rate=0.2))
        fast = analyzer.analyze(AudioFeatures(speaking_rate=0.9))
        assert fast.arousal > slow.arousal

    def test_high_energy_increases_dominance(self):
        analyzer = ProsodyAnalyzer()
        low = analyzer.analyze(AudioFeatures(energy_mean=0.1))
        high = analyzer.analyze(AudioFeatures(energy_mean=0.9))
        assert high.dominance > low.dominance

    def test_emotional_tone_classification(self):
        analyzer = ProsodyAnalyzer()
        # Excited: high arousal, positive valence
        result = analyzer.analyze(AudioFeatures(
            pitch_mean=0.8, energy_mean=0.9,
            speaking_rate=0.8, pause_ratio=0.05,
        ))
        assert result.emotional_tone in ("excited", "enthusiastic", "agitated", "engaged")

    def test_sad_tone(self):
        analyzer = ProsodyAnalyzer()
        result = analyzer.analyze(AudioFeatures(
            pitch_mean=0.2, energy_mean=0.1,
            speaking_rate=0.2, pause_ratio=0.5,
        ))
        assert result.emotional_tone in ("sad", "withdrawn", "subdued")

    def test_confidence_increases_with_duration(self):
        analyzer = ProsodyAnalyzer()
        short = analyzer.analyze(AudioFeatures(
            pitch_mean=0.8, energy_mean=0.8, duration_seconds=0.5,
        ))
        long = analyzer.analyze(AudioFeatures(
            pitch_mean=0.8, energy_mean=0.8, duration_seconds=5.0,
        ))
        assert long.confidence >= short.confidence

    def test_calibrate_for_user(self):
        analyzer = ProsodyAnalyzer()
        baseline = AudioFeatures(pitch_mean=0.3, energy_mean=0.4, speaking_rate=0.3)
        analyzer.calibrate_for_user("user1", baseline)
        assert "user1" in analyzer._calibration

    def test_analyze_for_user_with_calibration(self):
        analyzer = ProsodyAnalyzer()
        baseline = AudioFeatures(pitch_mean=0.3, energy_mean=0.4, speaking_rate=0.3)
        analyzer.calibrate_for_user("user1", baseline)
        # Same features as baseline should be "neutral"
        result = analyzer.analyze_for_user("user1", baseline)
        assert isinstance(result.emotional_tone, str)

    def test_analyze_for_unknown_user(self):
        analyzer = ProsodyAnalyzer()
        result = analyzer.analyze_for_user("unknown", AudioFeatures())
        assert result is not None

    def test_features_used_in_result(self):
        analyzer = ProsodyAnalyzer()
        result = analyzer.analyze(AudioFeatures(pitch_mean=0.7))
        assert "pitch_dev" in result.features_used

    def test_stats(self):
        analyzer = ProsodyAnalyzer()
        analyzer.analyze(AudioFeatures())
        stats = analyzer.get_stats()
        assert stats["total_analyses"] == 1

    def test_calibration_preserves_weights(self):
        """Verify analyze_for_user doesn't lose custom config weights."""
        config = ProsodyConfig(
            pitch_weight=0.5, energy_weight=0.5, rate_weight=0.0, pause_weight=0.0
        )
        analyzer = ProsodyAnalyzer(config=config)
        baseline = AudioFeatures(pitch_mean=0.3, energy_mean=0.4, speaking_rate=0.3)
        analyzer.calibrate_for_user("u1", baseline)
        analyzer.analyze_for_user("u1", AudioFeatures(pitch_mean=0.8, energy_mean=0.8))
        # Config should still have custom weights after analyze_for_user
        assert analyzer.config.pitch_weight == 0.5
        assert analyzer.config.rate_weight == 0.0


# =========================================================================
# UserModeler
# =========================================================================


class TestUserModeler:
    """Tests for user modeling."""

    def test_record_first_interaction(self):
        m = UserModeler()
        m.record_interaction(
            user_id="user1", display_name="Alice",
            content_length=100, cycle=1,
        )
        profile = m.get_profile("user1")
        assert profile is not None
        assert profile.display_name == "Alice"
        assert profile.total_interactions == 1

    def test_profile_accumulates(self):
        m = UserModeler()
        m.record_interaction(user_id="user1", cycle=1)
        m.record_interaction(user_id="user1", cycle=2)
        m.record_interaction(user_id="user1", cycle=3)
        profile = m.get_profile("user1")
        assert profile.total_interactions == 3

    def test_max_users(self):
        config = UserModelingConfig(max_users=3)
        m = UserModeler(config=config)
        for i in range(5):
            m.record_interaction(user_id=f"u{i}", cycle=i)
        assert len(m._profiles) == 3

    def test_unknown_profile_returns_none(self):
        m = UserModeler()
        assert m.get_profile("nonexistent") is None

    def test_trust_grows_with_positive_sentiment(self):
        m = UserModeler()
        m.record_interaction(user_id="u1", sentiment=0.0, cycle=1)
        initial_trust = m.get_profile("u1").trust_level
        for i in range(10):
            m.record_interaction(user_id="u1", sentiment=0.8, cycle=i + 2)
        assert m.get_profile("u1").trust_level > initial_trust

    def test_trust_decays_with_negative_sentiment(self):
        m = UserModeler()
        m.record_interaction(user_id="u1", sentiment=0.5, cycle=1)
        for i in range(5):
            m.record_interaction(user_id="u1", sentiment=0.5, cycle=i + 2)
        mid_trust = m.get_profile("u1").trust_level
        for i in range(5):
            m.record_interaction(user_id="u1", sentiment=-0.8, cycle=i + 7)
        assert m.get_profile("u1").trust_level < mid_trust

    def test_rapport_grows_with_interactions(self):
        m = UserModeler()
        m.record_interaction(user_id="u1", cycle=1)
        initial_rapport = m.get_profile("u1").rapport
        for i in range(20):
            m.record_interaction(user_id="u1", cycle=i + 2)
        assert m.get_profile("u1").rapport > initial_rapport

    def test_verbosity_preference(self):
        m = UserModeler()
        # Short messages → low verbosity preference
        for i in range(10):
            m.record_interaction(
                user_id="terse_user", content_length=20, cycle=i,
            )
        profile = m.get_profile("terse_user")
        assert profile.communication_prefs.preferred_verbosity < 0.3

    def test_topic_interests_tracked(self):
        m = UserModeler()
        m.record_interaction(
            user_id="u1", topics=["AI", "robotics"], cycle=1,
        )
        m.record_interaction(
            user_id="u1", topics=["AI", "ethics"], cycle=2,
        )
        profile = m.get_profile("u1")
        assert "AI" in profile.communication_prefs.topic_interests
        assert profile.communication_prefs.topic_interests["AI"] > 0

    def test_question_frequency(self):
        m = UserModeler()
        for i in range(10):
            m.record_interaction(
                user_id="u1", was_question=(i < 8), cycle=i,
            )
        profile = m.get_profile("u1")
        assert profile.communication_prefs.question_frequency == pytest.approx(0.8, abs=0.01)

    def test_response_guidance(self):
        m = UserModeler()
        m.record_interaction(user_id="u1", content_length=50, cycle=1)
        guidance = m.get_response_guidance("u1")
        assert "verbosity" in guidance
        assert "relationship" in guidance

    def test_response_guidance_unknown_user(self):
        m = UserModeler()
        guidance = m.get_response_guidance("unknown")
        assert guidance["relationship"] == "new"

    def test_add_note(self):
        m = UserModeler()
        m.record_interaction(user_id="u1", cycle=1)
        result = m.add_note("u1", "Prefers morning conversations")
        assert result is True
        assert "Prefers morning conversations" in m.get_profile("u1").notes

    def test_add_note_unknown_user(self):
        m = UserModeler()
        assert m.add_note("unknown", "note") is False

    def test_known_users(self):
        m = UserModeler()
        m.record_interaction(user_id="u1", cycle=1)
        m.record_interaction(user_id="u2", cycle=2)
        users = m.get_known_users()
        assert set(users) == {"u1", "u2"}

    def test_relationship_progression(self):
        m = UserModeler()
        m.record_interaction(user_id="u1", sentiment=0.5, cycle=1)
        assert m.get_response_guidance("u1")["relationship"] == "new"
        for i in range(50):
            m.record_interaction(user_id="u1", sentiment=0.5, cycle=i + 2)
        guidance = m.get_response_guidance("u1")
        assert guidance["relationship"] in ("acquaintance", "developing", "familiar", "trusted")

    def test_stats(self):
        m = UserModeler()
        m.record_interaction(user_id="u1", cycle=1)
        stats = m.get_stats()
        assert stats["total_users"] == 1
        assert stats["total_interactions"] == 1

    def test_notes_bounded(self):
        m = UserModeler()
        m.record_interaction(user_id="u1", cycle=1)
        for i in range(60):
            m.add_note("u1", f"Note {i}")
        assert len(m.get_profile("u1").notes) == 50

    def test_topic_interests_bounded(self):
        m = UserModeler()
        for i in range(60):
            m.record_interaction(
                user_id="u1", topics=[f"topic_{i}"], cycle=i,
            )
        profile = m.get_profile("u1")
        assert len(profile.communication_prefs.topic_interests) <= 50
