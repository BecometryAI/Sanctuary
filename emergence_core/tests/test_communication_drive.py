"""Tests for Communication Drive System."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from lyra.cognitive_core.communication import (
    CommunicationDriveSystem,
    CommunicationUrge,
    DriveType
)


class TestCommunicationUrge:
    """Tests for CommunicationUrge dataclass."""
    
    def test_urge_creation(self):
        """Test basic urge creation."""
        urge = CommunicationUrge(
            drive_type=DriveType.INSIGHT,
            intensity=0.8,
            content="Important realization",
            reason="High salience observation"
        )
        assert urge.drive_type == DriveType.INSIGHT
        assert urge.intensity == 0.8
        assert urge.content == "Important realization"
    
    def test_urge_decay(self):
        """Test urge intensity decays over time."""
        urge = CommunicationUrge(
            drive_type=DriveType.EMOTIONAL,
            intensity=1.0,
            decay_rate=0.5  # 50% per minute
        )
        # Manually set created_at to 2 minutes ago
        urge.created_at = datetime.now() - timedelta(minutes=2)
        
        # Should have decayed significantly
        current = urge.get_current_intensity()
        assert current < 0.5
    
    def test_urge_expiration(self):
        """Test urge expiration detection."""
        urge = CommunicationUrge(
            drive_type=DriveType.SOCIAL,
            intensity=0.3,
            decay_rate=0.5
        )
        urge.created_at = datetime.now() - timedelta(minutes=10)
        
        assert urge.is_expired(threshold=0.05)


class TestCommunicationDriveSystem:
    """Tests for CommunicationDriveSystem."""
    
    def test_initialization(self):
        """Test drive system initialization."""
        system = CommunicationDriveSystem()
        assert system.active_urges == []
        assert system.last_input_time is None
    
    def test_emotional_drive_high_arousal(self):
        """Test emotional drive from high arousal."""
        system = CommunicationDriveSystem(config={"emotional_threshold": 0.5})
        
        emotional_state = {
            "valence": 0.0,
            "arousal": 0.8,  # High arousal
            "dominance": 0.5
        }
        
        urges = system._compute_emotional_drive(emotional_state)
        
        assert len(urges) > 0
        assert any(u.drive_type == DriveType.EMOTIONAL for u in urges)
        assert any(u.intensity >= 0.8 for u in urges)
    
    def test_emotional_drive_extreme_valence(self):
        """Test emotional drive from extreme valence."""
        system = CommunicationDriveSystem(config={"emotional_threshold": 0.5})
        
        emotional_state = {
            "valence": -0.9,  # Very negative
            "arousal": 0.3,
            "dominance": 0.5
        }
        
        urges = system._compute_emotional_drive(emotional_state)
        
        assert len(urges) > 0
        emotional_urges = [u for u in urges if u.drive_type == DriveType.EMOTIONAL]
        assert len(emotional_urges) > 0
    
    def test_social_drive_after_silence(self):
        """Test social drive increases with silence duration."""
        system = CommunicationDriveSystem(config={"social_silence_minutes": 5})
        
        # Set last input to 10 minutes ago
        system.last_input_time = datetime.now() - timedelta(minutes=10)
        
        urges = system._compute_social_drive()
        
        assert len(urges) > 0
        social_urges = [u for u in urges if u.drive_type == DriveType.SOCIAL]
        assert len(social_urges) > 0
        assert social_urges[0].intensity > 0
    
    def test_no_social_drive_recent_interaction(self):
        """Test no social drive with recent interaction."""
        system = CommunicationDriveSystem(config={"social_silence_minutes": 30})
        
        # Set last input to 1 minute ago
        system.last_input_time = datetime.now() - timedelta(minutes=1)
        
        urges = system._compute_social_drive()
        
        assert len(urges) == 0
    
    def test_goal_drive_response_goal(self):
        """Test goal drive from response goals."""
        system = CommunicationDriveSystem()
        
        mock_goal = MagicMock()
        mock_goal.type = "RESPOND_TO_USER"
        mock_goal.description = "Answer user question"
        mock_goal.priority = 0.9
        
        urges = system._compute_goal_drive([mock_goal])
        
        assert len(urges) > 0
        goal_urges = [u for u in urges if u.drive_type == DriveType.GOAL]
        assert len(goal_urges) > 0
    
    def test_total_drive_computation(self):
        """Test total drive combines urges correctly."""
        system = CommunicationDriveSystem()
        
        # Add some urges manually
        system.active_urges = [
            CommunicationUrge(DriveType.INSIGHT, 0.8, priority=0.7),
            CommunicationUrge(DriveType.EMOTIONAL, 0.6, priority=0.6),
            CommunicationUrge(DriveType.SOCIAL, 0.3, priority=0.4)
        ]
        
        total = system.get_total_drive()
        
        assert 0 < total <= 1.0
    
    def test_strongest_urge(self):
        """Test getting strongest urge."""
        system = CommunicationDriveSystem()
        
        weak_urge = CommunicationUrge(DriveType.SOCIAL, 0.2, priority=0.4)
        strong_urge = CommunicationUrge(DriveType.INSIGHT, 0.9, priority=0.8)
        
        system.active_urges = [weak_urge, strong_urge]
        
        strongest = system.get_strongest_urge()
        
        assert strongest == strong_urge
    
    def test_urge_cleanup(self):
        """Test expired urges are cleaned up."""
        system = CommunicationDriveSystem()
        
        # Add an expired urge
        expired_urge = CommunicationUrge(DriveType.EMOTIONAL, 0.1, decay_rate=1.0)
        expired_urge.created_at = datetime.now() - timedelta(minutes=5)
        
        # Add a fresh urge
        fresh_urge = CommunicationUrge(DriveType.INSIGHT, 0.8)
        
        system.active_urges = [expired_urge, fresh_urge]
        system._cleanup_expired_urges()
        
        assert expired_urge not in system.active_urges
        assert fresh_urge in system.active_urges
    
    def test_record_input_output(self):
        """Test recording input/output times."""
        system = CommunicationDriveSystem()
        
        assert system.last_input_time is None
        assert system.last_output_time is None
        
        system.record_input()
        assert system.last_input_time is not None
        
        system.record_output()
        assert system.last_output_time is not None
    
    def test_drive_summary(self):
        """Test drive summary generation."""
        system = CommunicationDriveSystem()
        system.active_urges = [
            CommunicationUrge(DriveType.INSIGHT, 0.7),
            CommunicationUrge(DriveType.EMOTIONAL, 0.5)
        ]
        
        summary = system.get_drive_summary()
        
        assert "total_drive" in summary
        assert "active_urges" in summary
        assert summary["active_urges"] == 2
        assert "urges_by_type" in summary


class TestDriveIntegration:
    """Integration tests for drive computation."""
    
    def test_full_drive_computation(self):
        """Test computing all drives from mock state."""
        system = CommunicationDriveSystem()
        
        # Mock workspace state
        workspace_state = MagicMock()
        workspace_state.percepts = {}
        
        emotional_state = {"valence": 0.5, "arousal": 0.7, "dominance": 0.5}
        
        mock_goal = MagicMock()
        mock_goal.type = "RESPOND_TO_USER"
        mock_goal.description = "Help user"
        mock_goal.priority = 0.8
        
        urges = system.compute_drives(
            workspace_state=workspace_state,
            emotional_state=emotional_state,
            goals=[mock_goal],
            memories=[]
        )
        
        # Should have generated some urges
        assert len(system.active_urges) > 0
        assert system.get_total_drive() > 0


class TestEdgeCases:
    """Tests for edge cases and robustness."""
    
    def test_empty_inputs(self):
        """Test with empty inputs."""
        system = CommunicationDriveSystem()
        workspace_state = MagicMock()
        workspace_state.percepts = {}
        
        urges = system.compute_drives(
            workspace_state=workspace_state,
            emotional_state={},
            goals=[],
            memories=[]
        )
        
        # Should handle gracefully without errors
        assert isinstance(urges, list)
    
    def test_none_workspace_state(self):
        """Test with workspace state missing percepts."""
        system = CommunicationDriveSystem()
        workspace_state = MagicMock()
        del workspace_state.percepts  # No percepts attribute
        
        urges = system.compute_drives(
            workspace_state=workspace_state,
            emotional_state={"valence": 0.5, "arousal": 0.3},
            goals=[],
            memories=[]
        )
        
        # Should not crash
        assert isinstance(urges, list)
    
    def test_invalid_config_values(self):
        """Test that invalid config values are clamped."""
        system = CommunicationDriveSystem(config={
            "insight_threshold": 1.5,  # Should clamp to 1.0
            "emotional_threshold": -0.2,  # Should clamp to 0.0
            "social_silence_minutes": -5,  # Should clamp to 1
            "max_urges": 0  # Should clamp to 1
        })
        
        assert system.insight_threshold == 1.0
        assert system.emotional_threshold == 0.0
        assert system.social_silence_minutes == 1
        assert system.max_urges == 1
    
    def test_many_urges_limited(self):
        """Test that excess urges are limited to max_urges."""
        system = CommunicationDriveSystem(config={"max_urges": 3})
        
        # Add many urges
        for i in range(10):
            system.active_urges.append(
                CommunicationUrge(DriveType.INSIGHT, 0.5 + i * 0.05, priority=0.5)
            )
        
        system._limit_active_urges()
        
        # Should keep only strongest 3
        assert len(system.active_urges) == 3
        # Should be sorted by weighted intensity
        assert all(
            system.active_urges[i].intensity >= system.active_urges[i+1].intensity
            for i in range(len(system.active_urges) - 1)
        )
    
    def test_zero_intensity_urge(self):
        """Test handling of zero-intensity urge."""
        urge = CommunicationUrge(DriveType.SOCIAL, 0.0)
        assert urge.get_current_intensity() == 0.0
        assert urge.is_expired()
    
    def test_negative_decay_produces_zero(self):
        """Test that negative decay results produce zero intensity."""
        urge = CommunicationUrge(DriveType.INSIGHT, 0.5, decay_rate=1.0)
        urge.created_at = datetime.now() - timedelta(minutes=10)
        
        # Should produce negative but clamp to 0.0
        assert urge.get_current_intensity() == 0.0
    
    def test_missing_goal_attributes(self):
        """Test goals with missing attributes."""
        system = CommunicationDriveSystem()
        
        # Goal with minimal attributes
        mock_goal = MagicMock()
        mock_goal.type = None
        mock_goal.description = None
        
        # Should not crash
        urges = system._compute_goal_drive([mock_goal])
        assert isinstance(urges, list)
