"""Tests for Conversational Rhythm Model."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from mind.cognitive_core.communication import (
    ConversationalRhythmModel,
    ConversationPhase,
    ConversationTurn
)


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""
    
    def test_turn_creation(self):
        """Test basic turn creation."""
        now = datetime.now()
        turn = ConversationTurn(
            speaker="human",
            started_at=now,
            content_length=100
        )
        assert turn.speaker == "human"
        assert turn.started_at == now
        assert turn.ended_at is None
        assert turn.content_length == 100
        assert not turn.is_complete
    
    def test_turn_duration_ongoing(self):
        """Test duration calculation for ongoing turn."""
        past = datetime.now() - timedelta(seconds=5)
        turn = ConversationTurn(
            speaker="human",
            started_at=past,
            content_length=50
        )
        duration = turn.duration
        assert 4.5 <= duration <= 5.5  # Allow small timing variance
    
    def test_turn_duration_complete(self):
        """Test duration calculation for completed turn."""
        start = datetime.now() - timedelta(seconds=10)
        end = datetime.now() - timedelta(seconds=5)
        turn = ConversationTurn(
            speaker="system",
            started_at=start,
            ended_at=end,
            content_length=80
        )
        assert turn.is_complete
        assert 4.5 <= turn.duration <= 5.5


class TestConversationalRhythmModel:
    """Tests for ConversationalRhythmModel."""
    
    def test_initialization(self):
        """Test rhythm model initialization."""
        model = ConversationalRhythmModel()
        assert model.turns == []
        assert model.current_phase == ConversationPhase.MUTUAL_SILENCE
        assert model.avg_response_time == 2.0
        assert model.avg_turn_length == 50.0
    
    def test_initialization_with_config(self):
        """Test initialization with custom configuration."""
        config = {
            "natural_pause_threshold": 3.0,
            "rapid_exchange_threshold": 0.5,
            "max_turn_history": 100,
            "default_response_time": 1.5,
            "default_turn_length": 75.0
        }
        model = ConversationalRhythmModel(config=config)
        assert model.natural_pause_threshold == 3.0
        assert model.rapid_exchange_threshold == 0.5
        assert model.max_turn_history == 100
        assert model.avg_response_time == 1.5
        assert model.avg_turn_length == 75.0
    
    def test_record_human_input_new_turn(self):
        """Test recording new human input creates turn."""
        model = ConversationalRhythmModel()
        model.record_human_input("Hello, how are you?")
        
        assert len(model.turns) == 1
        assert model.turns[0].speaker == "human"
        assert model.turns[0].content_length == len("Hello, how are you?")
        assert not model.turns[0].is_complete
    
    def test_record_human_input_continuation(self):
        """Test recording continued human input extends turn."""
        model = ConversationalRhythmModel()
        model.record_human_input("Hello")
        initial_length = model.turns[0].content_length
        
        model.record_human_input(" world")
        
        assert len(model.turns) == 1  # Still one turn
        assert model.turns[0].content_length == initial_length + len(" world")
    
    def test_record_system_output(self):
        """Test recording system output."""
        model = ConversationalRhythmModel()
        model.record_human_input("What is 2+2?")
        model.record_system_output("The answer is 4.")
        
        assert len(model.turns) == 2
        assert model.turns[0].is_complete  # Human turn ended
        assert model.turns[1].speaker == "system"
        assert model.turns[1].is_complete  # System turn complete immediately
        assert model.turns[1].content_length == len("The answer is 4.")
    
    def test_phase_human_speaking(self):
        """Test phase detection for human speaking."""
        model = ConversationalRhythmModel()
        model.record_human_input("I'm typing a message")
        
        model.update_phase()
        assert model.current_phase == ConversationPhase.HUMAN_SPEAKING
    
    def test_phase_human_paused(self):
        """Test phase detection for human paused (recent silence)."""
        model = ConversationalRhythmModel(config={"natural_pause_threshold": 2.0})
        
        # Create a completed human turn 1 second ago
        past = datetime.now() - timedelta(seconds=1)
        turn = ConversationTurn(
            speaker="human",
            started_at=past - timedelta(seconds=2),
            ended_at=past,
            content_length=50
        )
        model.turns.append(turn)
        
        model.update_phase()
        assert model.current_phase == ConversationPhase.HUMAN_PAUSED
    
    def test_phase_mutual_silence(self):
        """Test phase detection for mutual silence (longer silence)."""
        model = ConversationalRhythmModel(config={"natural_pause_threshold": 2.0})
        
        # Create a completed human turn 5 seconds ago
        past = datetime.now() - timedelta(seconds=5)
        turn = ConversationTurn(
            speaker="human",
            started_at=past - timedelta(seconds=2),
            ended_at=past,
            content_length=50
        )
        model.turns.append(turn)
        
        model.update_phase()
        assert model.current_phase == ConversationPhase.MUTUAL_SILENCE
    
    def test_phase_system_speaking(self):
        """Test phase detection for system speaking."""
        model = ConversationalRhythmModel()
        
        # Create incomplete system turn
        turn = ConversationTurn(
            speaker="system",
            started_at=datetime.now(),
            content_length=50
        )
        model.turns.append(turn)
        
        model.update_phase()
        assert model.current_phase == ConversationPhase.SYSTEM_SPEAKING
    
    def test_phase_rapid_exchange(self):
        """Test phase detection for rapid exchange."""
        model = ConversationalRhythmModel(config={"rapid_exchange_threshold": 1.0})
        
        # Create rapid back-and-forth turns
        now = datetime.now()
        turns = [
            ConversationTurn("human", now - timedelta(seconds=5), now - timedelta(seconds=4.5), 50),
            ConversationTurn("system", now - timedelta(seconds=4), now - timedelta(seconds=3.5), 40),
            ConversationTurn("human", now - timedelta(seconds=3), now - timedelta(seconds=2.5), 60),
        ]
        model.turns.extend(turns)
        
        model.update_phase()
        assert model.current_phase == ConversationPhase.RAPID_EXCHANGE
    
    def test_timing_appropriateness_human_speaking(self):
        """Test appropriateness during human speaking (low)."""
        model = ConversationalRhythmModel()
        model.record_human_input("I am speaking now")
        
        appropriateness = model.get_timing_appropriateness()
        assert appropriateness == 0.0  # Very inappropriate to interrupt
    
    def test_timing_appropriateness_natural_pause(self):
        """Test appropriateness at natural pause (high)."""
        model = ConversationalRhythmModel(config={"natural_pause_threshold": 2.0})
        
        # Create human turn that ended 3 seconds ago
        past = datetime.now() - timedelta(seconds=3)
        turn = ConversationTurn(
            speaker="human",
            started_at=past - timedelta(seconds=2),
            ended_at=past,
            content_length=50
        )
        model.turns.append(turn)
        
        model.update_phase()
        appropriateness = model.get_timing_appropriateness()
        assert appropriateness >= 0.7  # High appropriateness
    
    def test_timing_appropriateness_no_conversation(self):
        """Test appropriateness with no conversation yet."""
        model = ConversationalRhythmModel()
        appropriateness = model.get_timing_appropriateness()
        assert appropriateness == 0.8  # Slightly appropriate
    
    def test_suggested_wait_time_human_speaking(self):
        """Test suggested wait time during human speaking."""
        model = ConversationalRhythmModel(config={"natural_pause_threshold": 2.0})
        model.record_human_input("I am typing")
        
        wait_time = model.get_suggested_wait_time()
        assert wait_time == 2.0  # Wait for natural pause threshold
    
    def test_suggested_wait_time_appropriate(self):
        """Test suggested wait time when appropriate to speak."""
        model = ConversationalRhythmModel()
        
        # Create old completed turn
        past = datetime.now() - timedelta(seconds=10)
        turn = ConversationTurn(
            speaker="human",
            started_at=past - timedelta(seconds=2),
            ended_at=past,
            content_length=50
        )
        model.turns.append(turn)
        
        wait_time = model.get_suggested_wait_time()
        assert wait_time == 0.0  # No wait needed
    
    def test_is_natural_pause_true(self):
        """Test natural pause detection returns true."""
        model = ConversationalRhythmModel(config={"natural_pause_threshold": 2.0})
        
        # Create human turn that ended 3 seconds ago
        past = datetime.now() - timedelta(seconds=3)
        turn = ConversationTurn(
            speaker="human",
            started_at=past - timedelta(seconds=2),
            ended_at=past,
            content_length=50
        )
        model.turns.append(turn)
        
        assert model.is_natural_pause() is True
    
    def test_is_natural_pause_false(self):
        """Test natural pause detection returns false."""
        model = ConversationalRhythmModel()
        model.record_human_input("Currently speaking")
        
        assert model.is_natural_pause() is False
    
    def test_is_natural_pause_no_conversation(self):
        """Test natural pause with no conversation."""
        model = ConversationalRhythmModel()
        assert model.is_natural_pause() is True  # No conversation = natural pause
    
    def test_rhythm_summary(self):
        """Test rhythm summary generation."""
        model = ConversationalRhythmModel()
        model.record_human_input("Hello there")
        model.record_system_output("Hi! How can I help?")
        
        summary = model.get_rhythm_summary()
        
        assert "current_phase" in summary
        assert "total_turns" in summary
        assert summary["total_turns"] == 2
        assert "avg_response_time" in summary
        assert "avg_turn_length" in summary
        assert "conversation_tempo" in summary
        assert "last_speaker" in summary
        assert summary["last_speaker"] == "system"
        assert "silence_duration" in summary
        assert "is_natural_pause" in summary
        assert "timing_appropriateness" in summary
        assert "suggested_wait_time" in summary
    
    def test_turn_history_limit(self):
        """Test turn history is limited to max_turn_history."""
        # MIN_TURN_HISTORY is 10, so we need to set above that to test trimming
        model = ConversationalRhythmModel(config={"max_turn_history": 15})

        # Record 20 turns to exceed limit
        for i in range(20):
            if i % 2 == 0:
                model.record_human_input(f"Message {i}")
            else:
                model.record_system_output(f"Response {i}")

        # Should only keep last 15
        assert len(model.turns) == 15
    
    def test_averages_update(self):
        """Test that averages update with new turns."""
        model = ConversationalRhythmModel()
        initial_avg_response = model.avg_response_time
        
        # Create several turns with short gaps
        now = datetime.now()
        model.turns = [
            ConversationTurn("human", now - timedelta(seconds=10), now - timedelta(seconds=9.5), 50),
            ConversationTurn("system", now - timedelta(seconds=9), now - timedelta(seconds=8.5), 40),
            ConversationTurn("human", now - timedelta(seconds=8), now - timedelta(seconds=7.5), 60),
            ConversationTurn("system", now - timedelta(seconds=7), now - timedelta(seconds=6.5), 45),
        ]
        
        model._update_averages()
        
        # Average response time should have changed
        assert model.avg_response_time != initial_avg_response
    
    def test_conversation_tempo_fast(self):
        """Test tempo classification as fast."""
        model = ConversationalRhythmModel()
        model.avg_response_time = 1.0  # Fast responses
        
        summary = model.get_rhythm_summary()
        assert summary["conversation_tempo"] == "fast"
    
    def test_conversation_tempo_slow(self):
        """Test tempo classification as slow."""
        model = ConversationalRhythmModel()
        model.avg_response_time = 5.0  # Slow responses
        
        summary = model.get_rhythm_summary()
        assert summary["conversation_tempo"] == "slow"
    
    def test_conversation_tempo_normal(self):
        """Test tempo classification as normal."""
        model = ConversationalRhythmModel()
        model.avg_response_time = 2.5  # Normal responses
        
        summary = model.get_rhythm_summary()
        assert summary["conversation_tempo"] == "normal"
    
    def test_multiple_turns_integration(self):
        """Test full conversation flow with multiple turns."""
        model = ConversationalRhythmModel()
        
        # Simulate a conversation
        model.record_human_input("Hello")
        assert model.current_phase == ConversationPhase.HUMAN_SPEAKING
        
        model.record_system_output("Hi there!")
        assert len(model.turns) == 2
        assert model.turns[0].is_complete
        
        # Wait a bit and add another human input
        model.turns[-1].ended_at = datetime.now() - timedelta(seconds=3)
        model.record_human_input("How are you?")
        
        summary = model.get_rhythm_summary()
        assert summary["total_turns"] == 3
        assert summary["last_speaker"] == "human"


class TestRhythmIntegration:
    """Integration tests for rhythm model with inhibition system."""
    
    def test_rhythm_model_in_inhibition_system(self):
        """Test rhythm model can be used by inhibition system."""
        from mind.cognitive_core.communication import CommunicationInhibitionSystem
        
        rhythm_model = ConversationalRhythmModel()
        inhibition_system = CommunicationInhibitionSystem(rhythm_model=rhythm_model)
        
        assert inhibition_system.rhythm_model is rhythm_model
    
    def test_timing_inhibition_with_bad_timing(self):
        """Test timing inhibition is created during bad timing."""
        from mind.cognitive_core.communication import CommunicationInhibitionSystem
        
        rhythm_model = ConversationalRhythmModel()
        rhythm_model.record_human_input("I'm speaking now")
        
        inhibition_system = CommunicationInhibitionSystem(rhythm_model=rhythm_model)
        timing_inhibitions = inhibition_system._compute_timing_inhibition()
        
        assert len(timing_inhibitions) > 0
        assert timing_inhibitions[0].strength > 0.7  # Strong inhibition
    
    def test_timing_inhibition_with_good_timing(self):
        """Test no timing inhibition during appropriate times."""
        from mind.cognitive_core.communication import CommunicationInhibitionSystem
        
        rhythm_model = ConversationalRhythmModel(config={"natural_pause_threshold": 2.0})
        
        # Create completed turn from 5 seconds ago
        past = datetime.now() - timedelta(seconds=5)
        turn = ConversationTurn(
            speaker="human",
            started_at=past - timedelta(seconds=2),
            ended_at=past,
            content_length=50
        )
        rhythm_model.turns.append(turn)
        
        inhibition_system = CommunicationInhibitionSystem(rhythm_model=rhythm_model)
        timing_inhibitions = inhibition_system._compute_timing_inhibition()
        
        assert len(timing_inhibitions) == 0  # No inhibition
    
    def test_timing_inhibition_without_rhythm_model(self):
        """Test timing inhibition gracefully handles missing rhythm model."""
        from mind.cognitive_core.communication import CommunicationInhibitionSystem
        
        inhibition_system = CommunicationInhibitionSystem()
        timing_inhibitions = inhibition_system._compute_timing_inhibition()
        
        assert len(timing_inhibitions) == 0  # No inhibition without rhythm model


class TestInputValidation:
    """Tests for input validation and error handling."""
    
    def test_record_human_input_invalid_type(self):
        """Test that non-string input raises TypeError."""
        model = ConversationalRhythmModel()
        
        with pytest.raises(TypeError, match="content must be str"):
            model.record_human_input(123)
        
        with pytest.raises(TypeError, match="content must be str"):
            model.record_human_input(None)
        
        with pytest.raises(TypeError, match="content must be str"):
            model.record_human_input(['hello'])
    
    def test_record_system_output_invalid_type(self):
        """Test that non-string output raises TypeError."""
        model = ConversationalRhythmModel()
        
        with pytest.raises(TypeError, match="content must be str"):
            model.record_system_output(456)
        
        with pytest.raises(TypeError, match="content must be str"):
            model.record_system_output(None)
    
    def test_empty_string_input(self):
        """Test handling of empty string inputs."""
        model = ConversationalRhythmModel()
        
        model.record_human_input("")
        assert len(model.turns) == 1
        assert model.turns[0].content_length == 0
        
        model.record_system_output("")
        assert len(model.turns) == 2
        assert model.turns[1].content_length == 0
    
    def test_very_long_input(self):
        """Test handling of very long inputs."""
        model = ConversationalRhythmModel()
        
        long_text = "x" * 10000
        model.record_human_input(long_text)
        
        assert model.turns[0].content_length == 10000
        assert model.avg_turn_length > 50.0  # Should update average
    
    def test_config_validation(self):
        """Test that invalid config values are clamped to minimums."""
        config = {
            "natural_pause_threshold": 0.1,  # Below minimum
            "rapid_exchange_threshold": 0.01,  # Below minimum
            "max_turn_history": 1,  # Below minimum
            "default_response_time": 0.1,  # Below minimum
            "default_turn_length": 1.0  # Below minimum
        }
        model = ConversationalRhythmModel(config=config)
        
        assert model.natural_pause_threshold >= 0.5
        assert model.rapid_exchange_threshold >= 0.1
        assert model.max_turn_history >= 10
        assert model.avg_response_time >= 0.5
        assert model.avg_turn_length >= 10.0
