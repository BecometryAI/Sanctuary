"""Tests for Communication Decision Loop."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from lyra.cognitive_core.communication import (
    CommunicationDecisionLoop,
    CommunicationDecision,
    DecisionResult,
    DeferredCommunication,
    CommunicationDriveSystem,
    CommunicationInhibitionSystem,
    CommunicationUrge,
    DriveType
)


class TestDeferredCommunication:
    """Tests for DeferredCommunication dataclass."""
    
    def test_deferred_creation(self):
        """Test basic deferred communication creation."""
        urge = CommunicationUrge(
            drive_type=DriveType.INSIGHT,
            intensity=0.7,
            content="Important thought"
        )
        
        defer_until = datetime.now() + timedelta(seconds=30)
        deferred = DeferredCommunication(
            urge=urge,
            reason="Bad timing",
            defer_until=defer_until
        )
        
        assert deferred.urge == urge
        assert deferred.reason == "Bad timing"
        assert deferred.attempts == 0
        assert not deferred.is_ready()
    
    def test_deferred_ready(self):
        """Test deferred becomes ready after time passes."""
        urge = CommunicationUrge(
            drive_type=DriveType.QUESTION,
            intensity=0.6
        )
        
        # Set to 1 second ago
        defer_until = datetime.now() - timedelta(seconds=1)
        deferred = DeferredCommunication(
            urge=urge,
            reason="Test",
            defer_until=defer_until
        )
        
        assert deferred.is_ready()
    
    def test_increment_attempts(self):
        """Test incrementing attempt counter."""
        urge = CommunicationUrge(
            drive_type=DriveType.SOCIAL,
            intensity=0.5
        )
        
        deferred = DeferredCommunication(
            urge=urge,
            reason="Test"
        )
        
        deferred.increment_attempts()
        assert deferred.attempts == 1
        
        deferred.increment_attempts()
        assert deferred.attempts == 2


class TestDecisionResult:
    """Tests for DecisionResult dataclass."""
    
    def test_result_creation(self):
        """Test decision result creation."""
        result = DecisionResult(
            decision=CommunicationDecision.SPEAK,
            reason="Strong drive",
            confidence=0.9,
            drive_level=0.8,
            inhibition_level=0.2,
            net_pressure=0.6
        )
        
        assert result.decision == CommunicationDecision.SPEAK
        assert result.confidence == 0.9
        assert result.net_pressure == 0.6
        assert result.urge is None


class TestCommunicationDecisionLoop:
    """Tests for CommunicationDecisionLoop."""
    
    @pytest.fixture
    def drive_system(self):
        """Create drive system for testing."""
        return CommunicationDriveSystem()
    
    @pytest.fixture
    def inhibition_system(self):
        """Create inhibition system for testing."""
        return CommunicationInhibitionSystem()
    
    @pytest.fixture
    def decision_loop(self, drive_system, inhibition_system):
        """Create decision loop for testing."""
        return CommunicationDecisionLoop(
            drive_system=drive_system,
            inhibition_system=inhibition_system
        )
    
    def test_initialization(self, decision_loop):
        """Test decision loop initialization."""
        assert decision_loop.deferred_queue == []
        assert decision_loop.decision_history == []
        assert decision_loop.speak_threshold == 0.3
        assert decision_loop.silence_threshold == -0.2
    
    def test_decision_speak_high_drive(self, decision_loop):
        """Test SPEAK decision when drive is high."""
        # Set up high drive, low inhibition
        decision_loop.drives.active_urges = [
            CommunicationUrge(
                drive_type=DriveType.INSIGHT,
                intensity=0.9,
                priority=0.8
            )
        ]
        
        result = decision_loop.evaluate(
            workspace_state=MagicMock(),
            emotional_state={},
            goals=[],
            memories=[]
        )
        
        assert result.decision == CommunicationDecision.SPEAK
        assert result.drive_level > 0.5
        assert "exceeds inhibition" in result.reason.lower()
    
    def test_decision_silence_high_inhibition(self, decision_loop):
        """Test SILENCE decision when inhibition is high."""
        # Set up low drive, high inhibition
        decision_loop.inhibitions.active_inhibitions = [
            MagicMock(
                get_current_strength=lambda: 0.8,
                priority=0.7
            )
        ]
        
        result = decision_loop.evaluate(
            workspace_state=MagicMock(),
            emotional_state={},
            goals=[],
            memories=[]
        )
        
        assert result.decision == CommunicationDecision.SILENCE
        assert result.inhibition_level > 0.5
    
    def test_decision_silence_insufficient_drive(self, decision_loop):
        """Test SILENCE decision when drive is too low."""
        # No urges, no inhibitions
        result = decision_loop.evaluate(
            workspace_state=MagicMock(),
            emotional_state={},
            goals=[],
            memories=[]
        )
        
        assert result.decision == CommunicationDecision.SILENCE
        assert "insufficient" in result.reason.lower()
    
    def test_decision_defer_ambiguous(self, decision_loop):
        """Test DEFER decision when both drive and inhibition are moderate."""
        # Set up moderate drive and moderate inhibition
        decision_loop.drives.active_urges = [
            CommunicationUrge(
                drive_type=DriveType.EMOTIONAL,
                intensity=0.6,
                priority=0.6
            )
        ]
        
        decision_loop.inhibitions.active_inhibitions = [
            MagicMock(
                get_current_strength=lambda: 0.5,
                priority=0.6
            )
        ]
        
        result = decision_loop.evaluate(
            workspace_state=MagicMock(),
            emotional_state={},
            goals=[],
            memories=[]
        )
        
        assert result.decision == CommunicationDecision.DEFER
        assert "defer" in result.reason.lower()
        assert result.defer_until is not None
    
    def test_defer_communication(self, decision_loop):
        """Test deferring a communication."""
        urge = CommunicationUrge(
            drive_type=DriveType.GOAL,
            intensity=0.7
        )
        
        decision_loop.defer_communication(
            urge=urge,
            reason="Timing not right",
            defer_seconds=30
        )
        
        assert len(decision_loop.deferred_queue) == 1
        assert decision_loop.deferred_queue[0].urge == urge
        assert decision_loop.deferred_queue[0].reason == "Timing not right"
    
    def test_check_deferred_queue_empty(self, decision_loop):
        """Test checking empty deferred queue."""
        result = decision_loop.check_deferred_queue()
        assert result is None
    
    def test_check_deferred_queue_not_ready(self, decision_loop):
        """Test checking queue when items not ready."""
        urge = CommunicationUrge(
            drive_type=DriveType.SOCIAL,
            intensity=0.6
        )
        
        # Defer for future
        defer_until = datetime.now() + timedelta(seconds=60)
        decision_loop.deferred_queue.append(
            DeferredCommunication(
                urge=urge,
                reason="Test",
                defer_until=defer_until
            )
        )
        
        result = decision_loop.check_deferred_queue()
        assert result is None
    
    def test_check_deferred_queue_ready(self, decision_loop):
        """Test checking queue when item is ready."""
        urge = CommunicationUrge(
            drive_type=DriveType.INSIGHT,
            intensity=0.8
        )
        
        # Defer to past
        defer_until = datetime.now() - timedelta(seconds=1)
        decision_loop.deferred_queue.append(
            DeferredCommunication(
                urge=urge,
                reason="Test",
                defer_until=defer_until
            )
        )
        
        result = decision_loop.check_deferred_queue()
        assert result is not None
        assert result.urge == urge
        assert result.attempts == 1
    
    def test_deferred_max_attempts(self, decision_loop):
        """Test dropping deferred item after max attempts."""
        decision_loop.max_defer_attempts = 2
        
        urge = CommunicationUrge(
            drive_type=DriveType.QUESTION,
            intensity=0.5
        )
        
        defer_until = datetime.now() - timedelta(seconds=1)
        deferred = DeferredCommunication(
            urge=urge,
            reason="Test",
            defer_until=defer_until
        )
        deferred.attempts = 2  # Already at max
        decision_loop.deferred_queue.append(deferred)
        
        result = decision_loop.check_deferred_queue()
        assert result is None
        assert len(decision_loop.deferred_queue) == 0
    
    def test_deferred_priority_from_queue(self, decision_loop):
        """Test deferred queue returns highest priority item."""
        # Create two deferred items
        urge1 = CommunicationUrge(
            drive_type=DriveType.SOCIAL,
            intensity=0.5,
            priority=0.4
        )
        urge2 = CommunicationUrge(
            drive_type=DriveType.INSIGHT,
            intensity=0.7,
            priority=0.8
        )
        
        defer_until = datetime.now() - timedelta(seconds=1)
        
        decision_loop.deferred_queue.extend([
            DeferredCommunication(urge=urge1, reason="Test1", defer_until=defer_until),
            DeferredCommunication(urge=urge2, reason="Test2", defer_until=defer_until)
        ])
        
        result = decision_loop.check_deferred_queue()
        assert result is not None
        assert result.urge == urge2  # Higher priority
    
    def test_deferred_queue_max_size(self, decision_loop):
        """Test deferred queue maintains size limit."""
        decision_loop.max_deferred = 3
        
        # Add 4 items
        for i in range(4):
            urge = CommunicationUrge(
                drive_type=DriveType.SOCIAL,
                intensity=0.5
            )
            decision_loop.defer_communication(
                urge=urge,
                reason=f"Test {i}",
                defer_seconds=30
            )
        
        assert len(decision_loop.deferred_queue) == 3
    
    def test_evaluate_with_ready_deferred(self, decision_loop):
        """Test evaluation prioritizes ready deferred items."""
        urge = CommunicationUrge(
            drive_type=DriveType.GOAL,
            intensity=0.8
        )
        
        # Add ready deferred item
        defer_until = datetime.now() - timedelta(seconds=1)
        decision_loop.deferred_queue.append(
            DeferredCommunication(
                urge=urge,
                reason="Was deferred",
                defer_until=defer_until
            )
        )
        
        result = decision_loop.evaluate(
            workspace_state=MagicMock(),
            emotional_state={},
            goals=[],
            memories=[]
        )
        
        assert result.decision == CommunicationDecision.SPEAK
        assert "deferred" in result.reason.lower()
        assert len(decision_loop.deferred_queue) == 0  # Removed from queue
    
    def test_decision_history_logged(self, decision_loop):
        """Test decisions are logged to history."""
        result = decision_loop.evaluate(
            workspace_state=MagicMock(),
            emotional_state={},
            goals=[],
            memories=[]
        )
        
        assert len(decision_loop.decision_history) == 1
        assert decision_loop.decision_history[0] == result
    
    def test_decision_history_size_limit(self, decision_loop):
        """Test decision history maintains size limit."""
        decision_loop.history_size = 5
        
        # Generate 10 decisions
        for _ in range(10):
            decision_loop.evaluate(
                workspace_state=MagicMock(),
                emotional_state={},
                goals=[],
                memories=[]
            )
        
        assert len(decision_loop.decision_history) == 5
    
    def test_get_decision_summary(self, decision_loop):
        """Test getting decision summary."""
        # Make some decisions
        for _ in range(3):
            decision_loop.evaluate(
                workspace_state=MagicMock(),
                emotional_state={},
                goals=[],
                memories=[]
            )
        
        summary = decision_loop.get_decision_summary()
        
        assert "deferred_queue_size" in summary
        assert "decision_history_size" in summary
        assert "last_decision" in summary
        assert "recent_decisions" in summary
        assert summary["decision_history_size"] == 3
    
    def test_custom_thresholds(self):
        """Test custom decision thresholds."""
        drive_system = CommunicationDriveSystem()
        inhibition_system = CommunicationInhibitionSystem()
        
        # Custom thresholds
        config = {
            "speak_threshold": 0.5,
            "silence_threshold": -0.3,
            "defer_min_drive": 0.4,
            "defer_min_inhibition": 0.4
        }
        
        decision_loop = CommunicationDecisionLoop(
            drive_system=drive_system,
            inhibition_system=inhibition_system,
            config=config
        )
        
        assert decision_loop.speak_threshold == 0.5
        assert decision_loop.silence_threshold == -0.3
        assert decision_loop.defer_min_drive == 0.4
    
    def test_net_pressure_calculation(self, decision_loop):
        """Test net pressure is correctly calculated."""
        # Set up drive and inhibition
        decision_loop.drives.active_urges = [
            CommunicationUrge(
                drive_type=DriveType.INSIGHT,
                intensity=0.8,
                priority=0.7
            )
        ]
        
        decision_loop.inhibitions.active_inhibitions = [
            MagicMock(
                get_current_strength=lambda: 0.3,
                priority=0.5
            )
        ]
        
        result = decision_loop.evaluate(
            workspace_state=MagicMock(),
            emotional_state={},
            goals=[],
            memories=[]
        )
        
        # net_pressure should be drive - inhibition
        assert result.net_pressure == result.drive_level - result.inhibition_level
        assert result.net_pressure > 0


class TestIntegration:
    """Integration tests with drive and inhibition systems."""
    
    def test_full_cycle_speak(self):
        """Test full cycle resulting in SPEAK."""
        drive_system = CommunicationDriveSystem()
        inhibition_system = CommunicationInhibitionSystem()
        decision_loop = CommunicationDecisionLoop(
            drive_system=drive_system,
            inhibition_system=inhibition_system
        )
        
        # Create high drive scenario
        emotional_state = {"valence": 0.9, "arousal": 0.8, "dominance": 0.7}
        drive_system.compute_drives(
            workspace_state=MagicMock(percepts={}),
            emotional_state=emotional_state,
            goals=[],
            memories=[]
        )
        
        # Low inhibition
        inhibition_system.compute_inhibitions(
            workspace_state=MagicMock(percepts={}),
            urges=drive_system.active_urges,
            confidence=0.9,
            content_value=0.8
        )
        
        result = decision_loop.evaluate(
            workspace_state=MagicMock(percepts={}),
            emotional_state=emotional_state,
            goals=[],
            memories=[]
        )
        
        assert result.decision == CommunicationDecision.SPEAK
    
    def test_full_cycle_silence(self):
        """Test full cycle resulting in SILENCE."""
        drive_system = CommunicationDriveSystem()
        inhibition_system = CommunicationInhibitionSystem()
        decision_loop = CommunicationDecisionLoop(
            drive_system=drive_system,
            inhibition_system=inhibition_system
        )
        
        # Low drive
        emotional_state = {"valence": 0.0, "arousal": 0.1, "dominance": 0.5}
        drive_system.compute_drives(
            workspace_state=MagicMock(percepts={}),
            emotional_state=emotional_state,
            goals=[],
            memories=[]
        )
        
        # High inhibition
        inhibition_system.compute_inhibitions(
            workspace_state=MagicMock(percepts={}),
            urges=drive_system.active_urges,
            confidence=0.3,  # Low confidence
            content_value=0.2  # Low value
        )
        
        result = decision_loop.evaluate(
            workspace_state=MagicMock(percepts={}),
            emotional_state=emotional_state,
            goals=[],
            memories=[]
        )
        
        assert result.decision == CommunicationDecision.SILENCE
