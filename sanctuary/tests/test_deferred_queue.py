"""Tests for Deferred Communication Queue."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from mind.cognitive_core.communication import (
    DeferredQueue,
    DeferredCommunication,
    DeferralReason,
    CommunicationUrge,
    DriveType
)


class TestDeferralReason:
    """Tests for DeferralReason enum."""
    
    def test_all_reasons_exist(self):
        """Test all expected deferral reasons are defined."""
        expected_reasons = [
            "bad_timing",
            "wait_for_response",
            "topic_change",
            "processing",
            "courtesy",
            "custom"
        ]
        
        actual_reasons = [r.value for r in DeferralReason]
        assert set(actual_reasons) == set(expected_reasons)


class TestDeferredCommunication:
    """Tests for DeferredCommunication dataclass."""
    
    def test_deferred_creation(self):
        """Test basic deferred communication creation."""
        urge = CommunicationUrge(
            drive_type=DriveType.INSIGHT,
            intensity=0.7,
            content="Important thought",
            priority=0.6
        )
        
        deferred = DeferredCommunication(
            urge=urge,
            reason=DeferralReason.BAD_TIMING,
            release_condition="Wait 30 seconds",
            priority=0.6
        )
        
        assert deferred.urge == urge
        assert deferred.reason == DeferralReason.BAD_TIMING
        assert deferred.attempts == 0
        assert deferred.priority == 0.6
        assert deferred.max_age_seconds == 300.0
    
    def test_is_ready_with_time(self):
        """Test time-based readiness check."""
        urge = CommunicationUrge(
            drive_type=DriveType.QUESTION,
            intensity=0.6
        )
        
        # Not ready yet (future release time)
        release_after = datetime.now() + timedelta(seconds=30)
        deferred = DeferredCommunication(
            urge=urge,
            reason=DeferralReason.BAD_TIMING,
            release_after=release_after
        )
        assert not deferred.is_ready()
        
        # Ready (past release time)
        release_after = datetime.now() - timedelta(seconds=1)
        deferred = DeferredCommunication(
            urge=urge,
            reason=DeferralReason.BAD_TIMING,
            release_after=release_after
        )
        assert deferred.is_ready()
    
    def test_is_ready_no_condition(self):
        """Test readiness with no explicit condition."""
        urge = CommunicationUrge(
            drive_type=DriveType.SOCIAL,
            intensity=0.5
        )
        
        deferred = DeferredCommunication(
            urge=urge,
            reason=DeferralReason.COURTESY,
            release_after=None
        )
        
        # No release condition = ready immediately
        assert deferred.is_ready()
    
    def test_is_expired(self):
        """Test expiration check."""
        urge = CommunicationUrge(
            drive_type=DriveType.EMOTIONAL,
            intensity=0.7
        )
        
        # Not expired (just created)
        deferred = DeferredCommunication(
            urge=urge,
            reason=DeferralReason.PROCESSING,
            max_age_seconds=60.0
        )
        assert not deferred.is_expired()
        
        # Expired (old creation time)
        deferred = DeferredCommunication(
            urge=urge,
            reason=DeferralReason.PROCESSING,
            deferred_at=datetime.now() - timedelta(seconds=61),
            max_age_seconds=60.0
        )
        assert deferred.is_expired()
    
    def test_expired_not_ready(self):
        """Test that expired items are not ready."""
        urge = CommunicationUrge(
            drive_type=DriveType.INSIGHT,
            intensity=0.8
        )
        
        deferred = DeferredCommunication(
            urge=urge,
            reason=DeferralReason.BAD_TIMING,
            deferred_at=datetime.now() - timedelta(seconds=400),
            release_after=datetime.now() - timedelta(seconds=1),
            max_age_seconds=300.0
        )
        
        # Should be ready based on time, but expired
        assert deferred.is_expired()
        assert not deferred.is_ready()
    
    def test_increment_attempts(self):
        """Test incrementing attempt counter."""
        urge = CommunicationUrge(
            drive_type=DriveType.GOAL,
            intensity=0.6
        )
        
        deferred = DeferredCommunication(
            urge=urge,
            reason=DeferralReason.WAIT_FOR_RESPONSE
        )
        
        assert deferred.attempts == 0
        deferred.increment_attempts()
        assert deferred.attempts == 1
        deferred.increment_attempts()
        assert deferred.attempts == 2
    
    def test_get_age_seconds(self):
        """Test age calculation."""
        urge = CommunicationUrge(
            drive_type=DriveType.ACKNOWLEDGMENT,
            intensity=0.5
        )
        
        deferred = DeferredCommunication(
            urge=urge,
            reason=DeferralReason.COURTESY,
            deferred_at=datetime.now() - timedelta(seconds=10)
        )
        
        age = deferred.get_age_seconds()
        assert age >= 10.0
        assert age < 11.0  # Allow small margin


class TestDeferredQueue:
    """Tests for DeferredQueue."""
    
    def test_queue_creation(self):
        """Test basic queue creation."""
        queue = DeferredQueue()
        
        assert len(queue.queue) == 0
        assert len(queue.released_history) == 0
        assert len(queue.expired_history) == 0
        assert queue.max_queue_size == 20
        assert queue.default_defer_seconds == 30
    
    def test_queue_creation_with_config(self):
        """Test queue creation with custom config."""
        config = {
            "max_queue_size": 10,
            "max_history_size": 25,
            "default_defer_seconds": 60,
            "max_defer_attempts": 5
        }
        
        queue = DeferredQueue(config)
        
        assert queue.max_queue_size == 10
        assert queue.max_history_size == 25
        assert queue.default_defer_seconds == 60
        assert queue.max_defer_attempts == 5
    
    def test_defer_basic(self):
        """Test deferring a communication."""
        queue = DeferredQueue()
        
        urge = CommunicationUrge(
            drive_type=DriveType.INSIGHT,
            intensity=0.7,
            priority=0.6
        )
        
        deferred = queue.defer(
            urge=urge,
            reason=DeferralReason.BAD_TIMING,
            release_seconds=30
        )
        
        assert len(queue.queue) == 1
        assert deferred in queue.queue
        assert deferred.reason == DeferralReason.BAD_TIMING
        assert deferred.priority == 0.6
    
    def test_defer_with_custom_priority(self):
        """Test deferring with custom priority."""
        queue = DeferredQueue()
        
        urge = CommunicationUrge(
            drive_type=DriveType.QUESTION,
            intensity=0.5,
            priority=0.4
        )
        
        deferred = queue.defer(
            urge=urge,
            reason=DeferralReason.PROCESSING,
            release_seconds=20,
            priority=0.9
        )
        
        assert deferred.priority == 0.9  # Custom priority used
    
    def test_defer_default_release_time(self):
        """Test deferring with default release time."""
        queue = DeferredQueue()
        
        urge = CommunicationUrge(
            drive_type=DriveType.SOCIAL,
            intensity=0.6
        )
        
        deferred = queue.defer(
            urge=urge,
            reason=DeferralReason.COURTESY
        )
        
        # Should use default_defer_seconds (30)
        expected_release = datetime.now() + timedelta(seconds=30)
        assert deferred.release_after is not None
        assert abs((deferred.release_after - expected_release).total_seconds()) < 1.0
    
    def test_defer_queue_size_limit(self):
        """Test queue size limit enforcement."""
        queue = DeferredQueue({"max_queue_size": 3})
        
        urges = [
            CommunicationUrge(drive_type=DriveType.INSIGHT, intensity=0.5, priority=0.3),
            CommunicationUrge(drive_type=DriveType.QUESTION, intensity=0.6, priority=0.5),
            CommunicationUrge(drive_type=DriveType.EMOTIONAL, intensity=0.7, priority=0.7),
            CommunicationUrge(drive_type=DriveType.GOAL, intensity=0.8, priority=0.9)
        ]
        
        for urge in urges:
            queue.defer(urge=urge, reason=DeferralReason.BAD_TIMING)
        
        # Should only keep 3 items, removing lowest priority
        assert len(queue.queue) == 3
        # Highest priority should be kept
        priorities = [d.priority for d in queue.queue]
        assert 0.9 in priorities
        assert 0.7 in priorities
    
    def test_check_ready_empty(self):
        """Test check_ready with empty queue."""
        queue = DeferredQueue()
        
        result = queue.check_ready()
        assert result is None
    
    def test_check_ready_none_ready(self):
        """Test check_ready when no items are ready."""
        queue = DeferredQueue()
        
        urge = CommunicationUrge(
            drive_type=DriveType.INSIGHT,
            intensity=0.7
        )
        
        # Defer for future time
        queue.defer(
            urge=urge,
            reason=DeferralReason.BAD_TIMING,
            release_seconds=60
        )
        
        result = queue.check_ready()
        assert result is None
    
    def test_check_ready_returns_highest_priority(self):
        """Test check_ready returns highest priority ready item."""
        queue = DeferredQueue()
        
        # Create multiple ready items with different priorities
        urge1 = CommunicationUrge(
            drive_type=DriveType.INSIGHT,
            intensity=0.5,
            priority=0.3
        )
        urge2 = CommunicationUrge(
            drive_type=DriveType.GOAL,
            intensity=0.8,
            priority=0.7
        )
        urge3 = CommunicationUrge(
            drive_type=DriveType.EMOTIONAL,
            intensity=0.6,
            priority=0.5
        )
        
        # Defer all to past (ready immediately)
        queue.defer(urge=urge1, reason=DeferralReason.BAD_TIMING, release_seconds=0)
        queue.defer(urge=urge2, reason=DeferralReason.PROCESSING, release_seconds=0)
        queue.defer(urge=urge3, reason=DeferralReason.COURTESY, release_seconds=0)
        
        result = queue.check_ready()
        
        # Should return urge2 (highest weighted priority: 0.8 * 0.7)
        assert result is not None
        assert result.urge == urge2
        assert result.attempts == 1
    
    def test_check_ready_increments_attempts(self):
        """Test check_ready increments attempt counter."""
        queue = DeferredQueue()
        
        urge = CommunicationUrge(
            drive_type=DriveType.QUESTION,
            intensity=0.6
        )
        
        queue.defer(
            urge=urge,
            reason=DeferralReason.BAD_TIMING,
            release_seconds=0
        )
        
        # First check
        result1 = queue.check_ready()
        assert result1.attempts == 1
        
        # Second check (still in queue)
        result2 = queue.check_ready()
        assert result2.attempts == 2
    
    def test_check_ready_max_attempts(self):
        """Test check_ready with max attempts reached."""
        queue = DeferredQueue({"max_defer_attempts": 2})
        
        urge = CommunicationUrge(
            drive_type=DriveType.INSIGHT,
            intensity=0.7
        )
        
        queue.defer(
            urge=urge,
            reason=DeferralReason.BAD_TIMING,
            release_seconds=0
        )
        
        # First attempt
        result1 = queue.check_ready()
        assert result1 is not None
        assert result1.attempts == 1
        assert len(queue.queue) == 1
        
        # Second attempt (reaches max)
        result2 = queue.check_ready()
        assert result2 is not None
        assert result2.attempts == 2
        
        # Should be removed from queue and added to history
        assert len(queue.queue) == 0
        assert len(queue.released_history) == 1
        
        # Third attempt should return None
        result3 = queue.check_ready()
        assert result3 is None
    
    def test_cleanup_expired(self):
        """Test cleanup of expired items."""
        queue = DeferredQueue()
        
        # Add non-expired item
        urge1 = CommunicationUrge(drive_type=DriveType.INSIGHT, intensity=0.6)
        queue.defer(urge=urge1, reason=DeferralReason.BAD_TIMING, max_age_seconds=60)
        
        # Add expired item (by setting old deferred_at time)
        urge2 = CommunicationUrge(drive_type=DriveType.QUESTION, intensity=0.7)
        deferred2 = DeferredCommunication(
            urge=urge2,
            reason=DeferralReason.PROCESSING,
            deferred_at=datetime.now() - timedelta(seconds=400),
            max_age_seconds=300.0
        )
        queue.queue.append(deferred2)
        
        assert len(queue.queue) == 2
        
        # Cleanup
        expired = queue.cleanup_expired()
        
        assert len(expired) == 1
        assert expired[0].urge == urge2
        assert len(queue.queue) == 1
        assert len(queue.expired_history) == 1
    
    def test_remove(self):
        """Test removing specific item."""
        queue = DeferredQueue()
        
        urge = CommunicationUrge(
            drive_type=DriveType.SOCIAL,
            intensity=0.5
        )
        
        deferred = queue.defer(
            urge=urge,
            reason=DeferralReason.COURTESY
        )
        
        assert len(queue.queue) == 1
        
        # Remove it
        removed = queue.remove(deferred)
        assert removed is True
        assert len(queue.queue) == 0
        
        # Try to remove again
        removed = queue.remove(deferred)
        assert removed is False
    
    def test_clear(self):
        """Test clearing queue."""
        queue = DeferredQueue()
        
        # Add multiple items
        for i in range(5):
            urge = CommunicationUrge(
                drive_type=DriveType.INSIGHT,
                intensity=0.6
            )
            queue.defer(urge=urge, reason=DeferralReason.BAD_TIMING)
        
        assert len(queue.queue) == 5
        
        count = queue.clear()
        
        assert count == 5
        assert len(queue.queue) == 0
    
    def test_get_queue_summary(self):
        """Test queue summary generation."""
        queue = DeferredQueue()
        
        # Add items with different reasons
        urge1 = CommunicationUrge(drive_type=DriveType.INSIGHT, intensity=0.7, priority=0.6)
        urge2 = CommunicationUrge(drive_type=DriveType.QUESTION, intensity=0.5, priority=0.4)
        urge3 = CommunicationUrge(drive_type=DriveType.EMOTIONAL, intensity=0.8, priority=0.7)
        
        queue.defer(urge=urge1, reason=DeferralReason.BAD_TIMING, release_seconds=0)
        queue.defer(urge=urge2, reason=DeferralReason.PROCESSING, release_seconds=30)
        queue.defer(urge=urge3, reason=DeferralReason.BAD_TIMING, release_seconds=0)
        
        summary = queue.get_queue_summary()
        
        assert summary["queue_size"] == 3
        assert summary["ready_count"] == 2  # Two with release_seconds=0
        assert summary["released_count"] == 0
        assert summary["expired_count"] == 0
        assert summary["reasons"][DeferralReason.BAD_TIMING.value] == 2
        assert summary["reasons"][DeferralReason.PROCESSING.value] == 1
        assert "average_priority" in summary
        
    def test_priority_ordering(self):
        """Test that priority ordering works correctly."""
        queue = DeferredQueue({"max_defer_attempts": 1})
        
        # Add items with different priorities and intensities
        urge_low = CommunicationUrge(
            drive_type=DriveType.SOCIAL,
            intensity=0.3,
            priority=0.2
        )
        urge_med = CommunicationUrge(
            drive_type=DriveType.QUESTION,
            intensity=0.6,
            priority=0.5
        )
        urge_high = CommunicationUrge(
            drive_type=DriveType.GOAL,
            intensity=0.9,
            priority=0.8
        )
        
        # Defer in random order, all ready
        queue.defer(urge=urge_med, reason=DeferralReason.BAD_TIMING, release_seconds=0)
        queue.defer(urge=urge_low, reason=DeferralReason.COURTESY, release_seconds=0)
        queue.defer(urge=urge_high, reason=DeferralReason.PROCESSING, release_seconds=0)
        
        # Should return high priority first
        result1 = queue.check_ready()
        assert result1.urge == urge_high
        
        # Then medium
        result2 = queue.check_ready()
        assert result2.urge == urge_med
        
        # Then low
        result3 = queue.check_ready()
        assert result3.urge == urge_low


class TestDeferredQueueIntegration:
    """Integration tests with decision loop."""
    
    def test_queue_with_decision_loop(self):
        """Test deferred queue integrated with decision loop."""
        from mind.cognitive_core.communication import (
            CommunicationDecisionLoop,
            CommunicationDriveSystem,
            CommunicationInhibitionSystem
        )
        
        # Create systems
        drives = CommunicationDriveSystem()
        inhibitions = CommunicationInhibitionSystem()
        decision_loop = CommunicationDecisionLoop(drives, inhibitions)
        
        # Queue should be initialized
        assert decision_loop.deferred_queue is not None
        assert isinstance(decision_loop.deferred_queue, DeferredQueue)
        
        # Summary should work
        summary = decision_loop.get_decision_summary()
        assert "deferred_queue" in summary
        assert summary["deferred_queue"]["queue_size"] == 0
