"""Tests for Silence-as-Action tracking."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from mind.cognitive_core.communication import (
    SilenceTracker,
    SilenceAction,
    SilenceType,
    CommunicationDecisionLoop,
    CommunicationDecision,
    DecisionResult,
    CommunicationDriveSystem,
    CommunicationInhibitionSystem,
    CommunicationUrge,
    InhibitionFactor,
    DriveType,
    InhibitionType
)


class TestSilenceType:
    """Tests for SilenceType enum."""
    
    def test_all_silence_types_exist(self):
        """Test all expected silence types are defined."""
        expected_types = [
            "NOTHING_TO_ADD",
            "RESPECTING_SPACE",
            "STILL_THINKING",
            "CHOOSING_DISCRETION",
            "UNCERTAINTY",
            "TIMING",
            "REDUNDANCY"
        ]
        
        for type_name in expected_types:
            assert hasattr(SilenceType, type_name)
    
    def test_silence_type_values(self):
        """Test silence type enum values."""
        assert SilenceType.NOTHING_TO_ADD.value == "nothing_to_add"
        assert SilenceType.RESPECTING_SPACE.value == "respecting_space"
        assert SilenceType.UNCERTAINTY.value == "uncertainty"


class TestSilenceAction:
    """Tests for SilenceAction dataclass."""
    
    def test_silence_action_creation(self):
        """Test basic silence action creation."""
        inhibitions = [
            InhibitionFactor(
                inhibition_type=InhibitionType.LOW_VALUE,
                strength=0.7,
                reason="Content not valuable"
            )
        ]
        urges = [
            CommunicationUrge(
                drive_type=DriveType.INSIGHT,
                intensity=0.3,
                reason="Minor thought"
            )
        ]
        
        action = SilenceAction(
            silence_type=SilenceType.NOTHING_TO_ADD,
            reason="No valuable content to share",
            inhibitions=inhibitions,
            suppressed_urges=urges
        )
        
        assert action.silence_type == SilenceType.NOTHING_TO_ADD
        assert action.reason == "No valuable content to share"
        assert len(action.inhibitions) == 1
        assert len(action.suppressed_urges) == 1
        assert action.duration is None
    
    def test_end_silence(self):
        """Test ending a silence period."""
        action = SilenceAction(
            silence_type=SilenceType.STILL_THINKING,
            reason="Processing",
            inhibitions=[],
            suppressed_urges=[]
        )
        
        # Simulate some time passing
        action.timestamp = datetime.now() - timedelta(seconds=5)
        
        duration = action.end_silence()
        
        assert action.duration is not None
        assert duration > 4.0  # At least 5 seconds
        assert duration == action.duration
    
    def test_end_silence_twice(self):
        """Test ending silence twice returns same duration."""
        action = SilenceAction(
            silence_type=SilenceType.TIMING,
            reason="Bad timing",
            inhibitions=[],
            suppressed_urges=[]
        )
        
        action.timestamp = datetime.now() - timedelta(seconds=3)
        duration1 = action.end_silence()
        duration2 = action.end_silence()
        
        assert duration1 == duration2


class TestSilenceTracker:
    """Tests for SilenceTracker class."""
    
    def test_tracker_initialization(self):
        """Test silence tracker initialization."""
        tracker = SilenceTracker()
        
        assert len(tracker.silence_history) == 0
        assert tracker.current_silence is None
        assert tracker.silence_streak == 0
    
    def test_tracker_with_config(self):
        """Test tracker with custom configuration."""
        config = {
            "max_silence_history": 50,
            "silence_pressure_threshold": 120,
            "max_silence_streak": 10
        }
        
        tracker = SilenceTracker(config)
        
        assert tracker.max_history == 50
        assert tracker.pressure_threshold_seconds == 120
        assert tracker.max_silence_streak == 10
    
    def test_record_silence(self):
        """Test recording a silence decision."""
        tracker = SilenceTracker()
        
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Inhibition (0.80) exceeds drive (0.20)",
            confidence=0.8,
            drive_level=0.2,
            inhibition_level=0.8,
            net_pressure=-0.6,
            inhibitions=[],
            urges=[]
        )
        
        silence_action = tracker.record_silence(decision_result)
        
        assert silence_action.silence_type == SilenceType.CHOOSING_DISCRETION
        assert len(tracker.silence_history) == 1
        assert tracker.current_silence == silence_action
        assert tracker.silence_streak == 1
    
    def test_record_multiple_silences(self):
        """Test recording multiple silence decisions."""
        tracker = SilenceTracker()
        
        for i in range(5):
            decision_result = DecisionResult(
                decision=CommunicationDecision.SILENCE,
                reason=f"Silence reason {i}",
                confidence=0.7,
                drive_level=0.1,
                inhibition_level=0.5,
                net_pressure=-0.4,
                inhibitions=[],
                urges=[]
            )
            
            tracker.record_silence(decision_result)
        
        assert len(tracker.silence_history) == 5
        assert tracker.silence_streak == 5
    
    def test_end_silence(self):
        """Test ending a silence period."""
        tracker = SilenceTracker()
        
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Insufficient drive",
            confidence=0.7,
            drive_level=0.1,
            inhibition_level=0.3,
            net_pressure=-0.2,
            inhibitions=[],
            urges=[]
        )
        
        tracker.record_silence(decision_result)
        assert tracker.current_silence is not None
        assert tracker.silence_streak == 1
        
        ended_silence = tracker.end_silence()
        
        assert ended_silence is not None
        assert ended_silence.duration is not None
        assert tracker.current_silence is None
        assert tracker.silence_streak == 0
    
    def test_end_silence_when_none(self):
        """Test ending silence when no active silence."""
        tracker = SilenceTracker()
        
        result = tracker.end_silence()
        
        assert result is None
    
    def test_classify_uncertainty_silence(self):
        """Test classification of uncertainty silence."""
        tracker = SilenceTracker()
        
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Confidence too low (0.40)",
            confidence=0.7,
            drive_level=0.5,
            inhibition_level=0.8,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )
        
        silence_action = tracker.record_silence(decision_result)
        
        assert silence_action.silence_type == SilenceType.UNCERTAINTY
    
    def test_classify_redundancy_silence(self):
        """Test classification of redundancy silence."""
        tracker = SilenceTracker()
        
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Content already expressed",
            confidence=0.7,
            drive_level=0.3,
            inhibition_level=0.6,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )
        
        silence_action = tracker.record_silence(decision_result)
        
        assert silence_action.silence_type == SilenceType.REDUNDANCY
    
    def test_classify_timing_silence(self):
        """Test classification of timing silence."""
        tracker = SilenceTracker()
        
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Bad timing - only 2s since last output",
            confidence=0.7,
            drive_level=0.4,
            inhibition_level=0.7,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )
        
        silence_action = tracker.record_silence(decision_result)
        
        assert silence_action.silence_type == SilenceType.TIMING
    
    def test_classify_still_thinking_silence(self):
        """Test classification of still thinking silence."""
        tracker = SilenceTracker()
        
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Still processing the information",
            confidence=0.7,
            drive_level=0.3,
            inhibition_level=0.5,
            net_pressure=-0.2,
            inhibitions=[],
            urges=[]
        )
        
        silence_action = tracker.record_silence(decision_result)
        
        assert silence_action.silence_type == SilenceType.STILL_THINKING
    
    def test_classify_respecting_space_silence(self):
        """Test classification of respecting space silence."""
        tracker = SilenceTracker()
        
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Respect silence - low arousal",
            confidence=0.7,
            drive_level=0.2,
            inhibition_level=0.4,
            net_pressure=-0.2,
            inhibitions=[],
            urges=[]
        )
        
        silence_action = tracker.record_silence(decision_result)
        
        assert silence_action.silence_type == SilenceType.RESPECTING_SPACE
    
    def test_classify_nothing_to_add_silence(self):
        """Test classification of nothing to add silence (default)."""
        tracker = SilenceTracker()
        
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Insufficient drive (0.15)",
            confidence=0.7,
            drive_level=0.15,
            inhibition_level=0.3,
            net_pressure=-0.15,
            inhibitions=[],
            urges=[]
        )
        
        silence_action = tracker.record_silence(decision_result)
        
        assert silence_action.silence_type == SilenceType.NOTHING_TO_ADD
    
    def test_silence_pressure_no_silence(self):
        """Test silence pressure when no active silence."""
        tracker = SilenceTracker()
        
        pressure = tracker.get_silence_pressure()
        
        assert pressure == 0.0
    
    def test_silence_pressure_short_duration(self):
        """Test silence pressure with short duration."""
        tracker = SilenceTracker({"silence_pressure_threshold": 60})
        
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Test silence",
            confidence=0.7,
            drive_level=0.2,
            inhibition_level=0.5,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )
        
        tracker.record_silence(decision_result)
        # Immediately after silence, pressure should be low
        pressure = tracker.get_silence_pressure()
        
        assert 0.0 <= pressure < 0.2
    
    def test_silence_pressure_long_duration(self):
        """Test silence pressure increases with duration."""
        # Min threshold is 10s; use that and simulate enough time to build pressure
        tracker = SilenceTracker({"silence_pressure_threshold": 10})

        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Test silence",
            confidence=0.7,
            drive_level=0.2,
            inhibition_level=0.5,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )

        silence_action = tracker.record_silence(decision_result)
        # Simulate time passing well beyond threshold (max_duration = 10*3 = 30s)
        silence_action.timestamp = datetime.now() - timedelta(seconds=30)

        pressure = tracker.get_silence_pressure()

        # After 30 seconds with 10s threshold (max_duration=30s), duration_pressure=1.0
        # streak_pressure = 1/5 = 0.2, total = 1.0*0.6 + 0.2*0.4 = 0.68
        assert pressure > 0.5
    
    def test_silence_pressure_streak(self):
        """Test silence pressure increases with streak."""
        tracker = SilenceTracker({"max_silence_streak": 3})
        
        for i in range(3):
            decision_result = DecisionResult(
                decision=CommunicationDecision.SILENCE,
                reason=f"Silence {i}",
                confidence=0.7,
                drive_level=0.2,
                inhibition_level=0.5,
                net_pressure=-0.3,
                inhibitions=[],
                urges=[]
            )
            tracker.record_silence(decision_result)
        
        pressure = tracker.get_silence_pressure()
        
        # With streak at max, pressure should be significant
        assert pressure > 0.3
    
    def test_get_recent_silences(self):
        """Test retrieving recent silence actions."""
        tracker = SilenceTracker()
        
        # Add some silences
        for i in range(5):
            decision_result = DecisionResult(
                decision=CommunicationDecision.SILENCE,
                reason=f"Silence {i}",
                confidence=0.7,
                drive_level=0.2,
                inhibition_level=0.5,
                net_pressure=-0.3,
                inhibitions=[],
                urges=[]
            )
            tracker.record_silence(decision_result)
        
        recent = tracker.get_recent_silences(minutes=5)
        
        assert len(recent) == 5
    
    def test_get_recent_silences_time_filter(self):
        """Test recent silences filters by time."""
        tracker = SilenceTracker()
        
        # Add old silence
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Old silence",
            confidence=0.7,
            drive_level=0.2,
            inhibition_level=0.5,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )
        old_silence = tracker.record_silence(decision_result)
        old_silence.timestamp = datetime.now() - timedelta(minutes=10)
        
        # Add recent silence
        decision_result = DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason="Recent silence",
            confidence=0.7,
            drive_level=0.2,
            inhibition_level=0.5,
            net_pressure=-0.3,
            inhibitions=[],
            urges=[]
        )
        tracker.record_silence(decision_result)
        
        recent = tracker.get_recent_silences(minutes=5)
        
        # Should only get the recent one
        assert len(recent) == 1
        assert recent[0].reason == "Recent silence"
    
    def test_get_silence_by_type(self):
        """Test filtering silences by type."""
        tracker = SilenceTracker()
        
        # Add different types of silence
        types_to_add = [
            SilenceType.UNCERTAINTY,
            SilenceType.TIMING,
            SilenceType.UNCERTAINTY,
            SilenceType.NOTHING_TO_ADD
        ]
        
        for silence_type in types_to_add:
            decision_result = DecisionResult(
                decision=CommunicationDecision.SILENCE,
                reason="Test",
                confidence=0.7,
                drive_level=0.2,
                inhibition_level=0.5,
                net_pressure=-0.3,
                inhibitions=[],
                urges=[]
            )
            tracker.record_silence(decision_result, silence_type=silence_type)
        
        uncertainty_silences = tracker.get_silence_by_type(SilenceType.UNCERTAINTY)
        timing_silences = tracker.get_silence_by_type(SilenceType.TIMING)
        
        assert len(uncertainty_silences) == 2
        assert len(timing_silences) == 1
    
    def test_get_silence_summary(self):
        """Test getting silence summary statistics."""
        tracker = SilenceTracker()
        
        # Add some silences
        for i in range(3):
            decision_result = DecisionResult(
                decision=CommunicationDecision.SILENCE,
                reason=f"Silence {i}",
                confidence=0.7,
                drive_level=0.2,
                inhibition_level=0.5,
                net_pressure=-0.3,
                inhibitions=[],
                urges=[]
            )
            tracker.record_silence(decision_result)
        
        summary = tracker.get_silence_summary()
        
        assert summary["total_silences"] == 3
        assert summary["current_silence"] is True
        assert summary["silence_streak"] == 3
        assert "silence_by_type" in summary
        assert "silence_pressure" in summary
    
    def test_silence_history_limit(self):
        """Test silence history respects max limit."""
        tracker = SilenceTracker({"max_silence_history": 10})
        
        # Add more than the limit
        for i in range(15):
            decision_result = DecisionResult(
                decision=CommunicationDecision.SILENCE,
                reason=f"Silence {i}",
                confidence=0.7,
                drive_level=0.2,
                inhibition_level=0.5,
                net_pressure=-0.3,
                inhibitions=[],
                urges=[]
            )
            tracker.record_silence(decision_result)
        
        assert len(tracker.silence_history) == 10


class TestDecisionLoopIntegration:
    """Tests for SilenceTracker integration with CommunicationDecisionLoop."""
    
    def test_decision_loop_has_silence_tracker(self):
        """Test decision loop initializes with silence tracker."""
        drives = CommunicationDriveSystem()
        inhibitions = CommunicationInhibitionSystem()
        loop = CommunicationDecisionLoop(drives, inhibitions)
        
        assert hasattr(loop, 'silence_tracker')
        assert isinstance(loop.silence_tracker, SilenceTracker)
    
    def test_silence_decision_records_silence(self):
        """Test silence decision records in silence tracker."""
        drives = CommunicationDriveSystem()
        inhibitions = CommunicationInhibitionSystem()
        loop = CommunicationDecisionLoop(drives, inhibitions)
        
        workspace = MagicMock()
        workspace.percepts = {}
        emotions = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        goals = []
        memories = []
        
        result = loop.evaluate(workspace, emotions, goals, memories)
        
        if result.decision == CommunicationDecision.SILENCE:
            assert len(loop.silence_tracker.silence_history) > 0
            assert loop.silence_tracker.current_silence is not None
    
    def test_speak_decision_ends_silence(self):
        """Test speak decision ends current silence."""
        drives = CommunicationDriveSystem()
        inhibitions = CommunicationInhibitionSystem()
        loop = CommunicationDecisionLoop(drives, inhibitions)
        
        # First, create a silence
        workspace = MagicMock()
        workspace.percepts = {}
        emotions = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        
        result1 = loop.evaluate(workspace, emotions, [], [])
        
        if result1.decision == CommunicationDecision.SILENCE:
            assert loop.silence_tracker.current_silence is not None
            
            # Now create conditions for speaking
            drives.active_urges.append(
                CommunicationUrge(
                    drive_type=DriveType.INSIGHT,
                    intensity=0.9,
                    priority=0.8
                )
            )
            
            result2 = loop.evaluate(workspace, emotions, [], [])
            
            if result2.decision == CommunicationDecision.SPEAK:
                # Silence should be ended
                assert loop.silence_tracker.current_silence is None
                assert loop.silence_tracker.silence_streak == 0
    
    def test_decision_summary_includes_silence_tracking(self):
        """Test decision summary includes silence tracking info."""
        drives = CommunicationDriveSystem()
        inhibitions = CommunicationInhibitionSystem()
        loop = CommunicationDecisionLoop(drives, inhibitions)
        
        summary = loop.get_decision_summary()
        
        assert "silence_tracking" in summary
        assert "total_silences" in summary["silence_tracking"]
        assert "silence_pressure" in summary["silence_tracking"]
