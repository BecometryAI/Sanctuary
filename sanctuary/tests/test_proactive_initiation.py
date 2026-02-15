"""Tests for Proactive Session Initiation System."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from mind.cognitive_core.communication import (
    ProactiveInitiationSystem,
    OutreachOpportunity,
    OutreachTrigger,
    CommunicationDriveSystem,
    DriveType
)


class TestOutreachTrigger:
    """Tests for OutreachTrigger enum."""
    
    def test_all_trigger_types(self):
        """Test all trigger types are defined."""
        assert OutreachTrigger.TIME_ELAPSED.value == "time_elapsed"
        assert OutreachTrigger.SIGNIFICANT_INSIGHT.value == "significant_insight"
        assert OutreachTrigger.EMOTIONAL_CONNECTION.value == "emotional_connection"
        assert OutreachTrigger.SCHEDULED_CHECKIN.value == "scheduled_checkin"
        assert OutreachTrigger.RELEVANT_EVENT.value == "relevant_event"
        assert OutreachTrigger.GOAL_COMPLETION.value == "goal_completion"


class TestOutreachOpportunity:
    """Tests for OutreachOpportunity dataclass."""
    
    def test_opportunity_creation(self):
        """Test basic opportunity creation."""
        opp = OutreachOpportunity(
            trigger=OutreachTrigger.TIME_ELAPSED,
            urgency=0.5,
            reason="It's been 3 days",
            suggested_content="Let's catch up!"
        )
        assert opp.trigger == OutreachTrigger.TIME_ELAPSED
        assert opp.urgency == 0.5
        assert opp.reason == "It's been 3 days"
        assert opp.suggested_content == "Let's catch up!"
    
    def test_appropriate_times_default(self):
        """Test default appropriate times."""
        opp = OutreachOpportunity(
            trigger=OutreachTrigger.SCHEDULED_CHECKIN,
            urgency=0.3,
            reason="Daily check-in"
        )
        assert len(opp.appropriate_times) > 0
        assert "morning" in opp.appropriate_times or "afternoon" in opp.appropriate_times
    
    def test_is_appropriate_now_morning(self):
        """Test time appropriateness checking for morning."""
        opp = OutreachOpportunity(
            trigger=OutreachTrigger.TIME_ELAPSED,
            urgency=0.5,
            reason="Check-in",
            appropriate_times=["morning"]
        )
        
        # Create a mock datetime for testing
        current_hour = datetime.now().hour
        
        # Check if we can determine appropriateness
        # (actual result depends on current time)
        result = opp.is_appropriate_now()
        assert isinstance(result, bool)
    
    def test_is_appropriate_now_no_restrictions(self):
        """Test appropriateness with no time restrictions."""
        opp = OutreachOpportunity(
            trigger=OutreachTrigger.RELEVANT_EVENT,
            urgency=0.7,
            reason="Important event",
            appropriate_times=[]
        )
        assert opp.is_appropriate_now() is True


class TestProactiveInitiationSystem:
    """Tests for ProactiveInitiationSystem."""
    
    def test_initialization(self):
        """Test system initialization."""
        system = ProactiveInitiationSystem()
        assert system.last_interaction is None
        assert system.pending_opportunities == []
        assert system.outreach_history == []
        assert system.scheduled_checkins == []
    
    def test_initialization_with_config(self):
        """Test system initialization with custom config."""
        config = {
            "time_elapsed_threshold": 1440,  # 1 day
            "insight_urgency": 0.7,
            "max_pending": 3
        }
        system = ProactiveInitiationSystem(config)
        assert system.time_elapsed_threshold == 1440
        assert system.insight_urgency == 0.7
        assert system.max_pending == 3
    
    def test_record_interaction(self):
        """Test recording interactions."""
        system = ProactiveInitiationSystem()
        assert system.last_interaction is None
        
        system.record_interaction()
        assert system.last_interaction is not None
        assert isinstance(system.last_interaction, datetime)
    
    def test_get_time_since_interaction(self):
        """Test getting time since last interaction."""
        system = ProactiveInitiationSystem()
        
        # No interaction yet
        assert system.get_time_since_interaction() is None
        
        # Record interaction
        system.record_interaction()
        time_delta = system.get_time_since_interaction()
        assert isinstance(time_delta, timedelta)
        assert time_delta.total_seconds() < 1  # Should be very recent
    
    def test_time_elapsed_trigger(self):
        """Test time elapsed trigger."""
        system = ProactiveInitiationSystem(config={
            "time_elapsed_threshold": 5  # 5 minutes for testing
        })
        
        # Set last interaction to 10 minutes ago
        system.last_interaction = datetime.now() - timedelta(minutes=10)
        
        # Check for opportunities
        opportunities = system._check_time_elapsed()
        
        assert len(opportunities) == 1
        assert opportunities[0].trigger == OutreachTrigger.TIME_ELAPSED
        assert opportunities[0].urgency > 0.3
    
    def test_time_elapsed_no_trigger_when_recent(self):
        """Test no time elapsed trigger when interaction is recent."""
        system = ProactiveInitiationSystem(config={
            "time_elapsed_threshold": 60  # 1 hour
        })
        
        # Set last interaction to 30 minutes ago
        system.last_interaction = datetime.now() - timedelta(minutes=30)
        
        opportunities = system._check_time_elapsed()
        assert len(opportunities) == 0
    
    def test_time_elapsed_no_duplicate(self):
        """Test no duplicate time elapsed opportunities."""
        system = ProactiveInitiationSystem(config={
            "time_elapsed_threshold": 5
        })
        
        system.last_interaction = datetime.now() - timedelta(minutes=10)
        
        # First check creates opportunity
        opportunities = system._check_time_elapsed()
        assert len(opportunities) == 1
        
        # Add to pending
        system.pending_opportunities.extend(opportunities)
        
        # Second check should not create duplicate
        opportunities = system._check_time_elapsed()
        assert len(opportunities) == 0
    
    def test_significant_insight_from_percept(self):
        """Test significant insight detection from workspace percepts."""
        system = ProactiveInitiationSystem()
        
        # Mock workspace with high-salience introspective percept
        workspace = MagicMock()
        percept = MagicMock()
        percept.source = "introspection"
        percept.salience = 0.85
        percept.content = "Important realization"
        workspace.percepts = {"p1": percept}
        
        opportunities = system._check_significant_insights(workspace, [])
        
        assert len(opportunities) >= 1
        insight_opps = [o for o in opportunities if o.trigger == OutreachTrigger.SIGNIFICANT_INSIGHT]
        assert len(insight_opps) >= 1
        assert insight_opps[0].urgency == system.insight_urgency
    
    def test_significant_insight_from_memory(self):
        """Test significant insight from memory connection."""
        system = ProactiveInitiationSystem()
        
        # Mock memory with high significance
        memory = MagicMock()
        memory.significance = 0.85
        memory.summary = "meaningful memory"
        
        workspace = MagicMock()
        workspace.percepts = {}
        
        opportunities = system._check_significant_insights(workspace, [memory])
        
        assert len(opportunities) >= 1
        assert opportunities[0].trigger == OutreachTrigger.SIGNIFICANT_INSIGHT
    
    def test_emotional_connection_trigger(self):
        """Test emotional connection trigger."""
        system = ProactiveInitiationSystem()
        
        # Set last interaction to 25 hours ago
        system.last_interaction = datetime.now() - timedelta(hours=25)
        
        # Mock workspace with strong emotions
        workspace = MagicMock()
        workspace.emotional_state = {
            "valence": 0.7,
            "arousal": 0.3
        }
        
        opportunities = system._check_emotional_connection(workspace)
        
        assert len(opportunities) >= 1
        assert opportunities[0].trigger == OutreachTrigger.EMOTIONAL_CONNECTION
    
    def test_scheduled_checkin_trigger(self):
        """Test scheduled check-in trigger."""
        system = ProactiveInitiationSystem()
        
        # Schedule a check-in for 1 minute ago
        past_time = datetime.now() - timedelta(minutes=1)
        system.schedule_checkin(
            time=past_time,
            reason="Daily check-in",
            message="Time for our daily check-in!"
        )
        
        assert len(system.scheduled_checkins) == 1
        
        # Check for opportunities
        opportunities = system._check_scheduled_checkins()
        
        assert len(opportunities) >= 1
        assert opportunities[0].trigger == OutreachTrigger.SCHEDULED_CHECKIN
        assert opportunities[0].reason == "Daily check-in"
        
        # Scheduled checkin should be removed after processing
        assert len(system.scheduled_checkins) == 0
    
    def test_scheduled_checkin_future(self):
        """Test scheduled check-in doesn't trigger if in future."""
        system = ProactiveInitiationSystem()
        
        # Schedule a check-in for 1 hour from now
        future_time = datetime.now() + timedelta(hours=1)
        system.schedule_checkin(
            time=future_time,
            reason="Future check-in"
        )
        
        opportunities = system._check_scheduled_checkins()
        assert len(opportunities) == 0
        assert len(system.scheduled_checkins) == 1  # Still pending
    
    def test_relevant_event_trigger(self):
        """Test relevant event trigger."""
        system = ProactiveInitiationSystem()
        
        # Mock workspace with high-salience non-introspective percept
        workspace = MagicMock()
        percept = MagicMock()
        percept.source = "external"
        percept.salience = 0.8
        percept.content = "Important event"
        workspace.percepts = {"p1": percept}
        
        opportunities = system._check_relevant_events(workspace)
        
        assert len(opportunities) >= 1
        assert opportunities[0].trigger == OutreachTrigger.RELEVANT_EVENT
    
    def test_goal_completion_trigger(self):
        """Test goal completion trigger."""
        system = ProactiveInitiationSystem()
        
        # Mock completed high-priority goal
        goal = MagicMock()
        goal.status = "completed"
        goal.description = "Important task"
        goal.priority = 0.7
        
        opportunities = system._check_goal_completions([goal])
        
        assert len(opportunities) >= 1
        assert opportunities[0].trigger == OutreachTrigger.GOAL_COMPLETION
    
    def test_goal_completion_ignores_low_priority(self):
        """Test goal completion doesn't trigger for low-priority goals."""
        system = ProactiveInitiationSystem()
        
        # Mock completed low-priority goal
        goal = MagicMock()
        goal.status = "completed"
        goal.description = "Minor task"
        goal.priority = 0.3
        
        opportunities = system._check_goal_completions([goal])
        assert len(opportunities) == 0
    
    def test_check_for_opportunities_combines_all(self):
        """Test check_for_opportunities combines all trigger types."""
        system = ProactiveInitiationSystem(config={
            "time_elapsed_threshold": 5
        })
        
        # Set up conditions for multiple triggers
        system.last_interaction = datetime.now() - timedelta(minutes=10)
        
        workspace = MagicMock()
        workspace.percepts = {}
        workspace.emotional_state = {"valence": 0.0, "arousal": 0.0}
        
        opportunities = system.check_for_opportunities(workspace, [], [])
        
        # Should get at least time elapsed
        assert len(opportunities) >= 1
        assert len(system.pending_opportunities) >= 1
    
    def test_limit_pending_opportunities(self):
        """Test pending opportunities are limited."""
        system = ProactiveInitiationSystem(config={"max_pending": 3})
        
        # Add many opportunities
        for i in range(10):
            system.pending_opportunities.append(OutreachOpportunity(
                trigger=OutreachTrigger.TIME_ELAPSED,
                urgency=0.1 + (i * 0.05),
                reason=f"Reason {i}"
            ))
        
        system._limit_pending_opportunities()
        
        # Should keep only max_pending
        assert len(system.pending_opportunities) == 3
        
        # Should keep highest urgency (top 3: 0.55, 0.50, 0.45)
        urgencies = [opp.urgency for opp in system.pending_opportunities]
        assert all(u >= 0.45 for u in urgencies)
    
    def test_should_initiate_now_with_appropriate(self):
        """Test should_initiate_now with appropriate opportunity."""
        system = ProactiveInitiationSystem()
        
        # Add high-urgency opportunity with no time restrictions
        opp = OutreachOpportunity(
            trigger=OutreachTrigger.SIGNIFICANT_INSIGHT,
            urgency=0.7,
            reason="Important insight",
            appropriate_times=[]  # Always appropriate
        )
        system.pending_opportunities.append(opp)
        
        should_initiate, selected_opp = system.should_initiate_now()
        
        assert should_initiate is True
        assert selected_opp is not None
        assert selected_opp.urgency >= 0.3
    
    def test_should_initiate_now_no_opportunities(self):
        """Test should_initiate_now with no opportunities."""
        system = ProactiveInitiationSystem()
        
        should_initiate, selected_opp = system.should_initiate_now()
        
        assert should_initiate is False
        assert selected_opp is None
    
    def test_should_initiate_now_low_urgency(self):
        """Test should_initiate_now doesn't trigger for low urgency."""
        system = ProactiveInitiationSystem()
        
        # Add low-urgency opportunity
        opp = OutreachOpportunity(
            trigger=OutreachTrigger.EMOTIONAL_CONNECTION,
            urgency=0.2,  # Below threshold
            reason="Low urgency",
            appropriate_times=[]
        )
        system.pending_opportunities.append(opp)
        
        should_initiate, selected_opp = system.should_initiate_now()
        
        assert should_initiate is False
    
    def test_record_outreach(self):
        """Test recording outreach attempt."""
        system = ProactiveInitiationSystem()
        
        opp = OutreachOpportunity(
            trigger=OutreachTrigger.TIME_ELAPSED,
            urgency=0.5,
            reason="Check-in"
        )
        system.pending_opportunities.append(opp)
        
        assert len(system.pending_opportunities) == 1
        assert len(system.outreach_history) == 0
        
        system.record_outreach(opp, success=True)
        
        # Should remove from pending
        assert len(system.pending_opportunities) == 0
        
        # Should add to history
        assert len(system.outreach_history) == 1
        assert system.outreach_history[0]["success"] is True
        assert system.outreach_history[0]["trigger"] == "time_elapsed"
    
    def test_get_outreach_summary(self):
        """Test outreach summary generation."""
        system = ProactiveInitiationSystem()
        
        system.record_interaction()
        
        opp = OutreachOpportunity(
            trigger=OutreachTrigger.SIGNIFICANT_INSIGHT,
            urgency=0.6,
            reason="Test"
        )
        system.pending_opportunities.append(opp)
        
        summary = system.get_outreach_summary()
        
        assert "last_interaction" in summary
        assert "time_since_interaction" in summary
        assert "pending_opportunities" in summary
        assert summary["pending_opportunities"] == 1
        assert "opportunities_by_trigger" in summary
        assert "should_initiate" in summary


class TestDriveSystemIntegration:
    """Tests for integration with CommunicationDriveSystem."""
    
    def test_drive_system_creates_proactive(self):
        """Test drive system initializes proactive system by default."""
        system = CommunicationDriveSystem()
        assert system.proactive_system is not None
    
    def test_drive_system_disable_proactive(self):
        """Test drive system can disable proactive system."""
        system = CommunicationDriveSystem(config={"enable_proactive": False})
        assert system.proactive_system is None
    
    def test_drive_system_proactive_config(self):
        """Test drive system passes config to proactive system."""
        config = {
            "enable_proactive": True,
            "proactive_config": {
                "time_elapsed_threshold": 1440,
                "insight_urgency": 0.8
            }
        }
        system = CommunicationDriveSystem(config=config)
        assert system.proactive_system is not None
        assert system.proactive_system.time_elapsed_threshold == 1440
        assert system.proactive_system.insight_urgency == 0.8
    
    def test_record_input_updates_proactive(self):
        """Test recording input updates proactive system."""
        system = CommunicationDriveSystem()
        
        assert system.proactive_system.last_interaction is None
        
        system.record_input()
        
        assert system.proactive_system.last_interaction is not None
    
    def test_record_output_updates_proactive(self):
        """Test recording output updates proactive system."""
        system = CommunicationDriveSystem()
        
        assert system.proactive_system.last_interaction is None
        
        system.record_output()
        
        assert system.proactive_system.last_interaction is not None
    
    def test_compute_drives_includes_proactive(self):
        """Test compute_drives includes proactive opportunities."""
        system = CommunicationDriveSystem(config={
            "enable_proactive": True,
            "proactive_config": {
                "time_elapsed_threshold": 5  # 5 minutes
            }
        })
        
        # Set last interaction to trigger time-based outreach
        system.proactive_system.last_interaction = datetime.now() - timedelta(minutes=10)
        
        # Mock workspace and state
        workspace = MagicMock()
        workspace.percepts = {}
        workspace.emotional_state = {"valence": 0.0, "arousal": 0.0}
        emotional_state = {"valence": 0.0, "arousal": 0.0}
        
        urges = system.compute_drives(workspace, emotional_state, [], [])
        
        # Should include proactive urges
        proactive_urges = [u for u in urges if "Proactive:" in u.reason]
        assert len(proactive_urges) >= 1
    
    def test_proactive_urges_map_to_drive_types(self):
        """Test proactive opportunities map correctly to drive types."""
        system = CommunicationDriveSystem(config={
            "enable_proactive": True,
            "proactive_config": {"time_elapsed_threshold": 5}
        })
        
        system.proactive_system.last_interaction = datetime.now() - timedelta(minutes=10)
        
        workspace = MagicMock()
        workspace.percepts = {}
        workspace.emotional_state = {"valence": 0.0, "arousal": 0.0}
        
        urges = system.compute_drives(workspace, {"valence": 0.0, "arousal": 0.0}, [], [])
        
        # Time elapsed should map to SOCIAL drive
        social_urges = [u for u in urges if u.drive_type == DriveType.SOCIAL and "Proactive:" in u.reason]
        assert len(social_urges) >= 1
    
    def test_drive_summary_includes_proactive(self):
        """Test drive summary includes proactive information."""
        system = CommunicationDriveSystem()
        system.record_input()
        
        summary = system.get_drive_summary()
        
        assert "proactive" in summary
        assert "last_interaction" in summary["proactive"]
        assert "pending_opportunities" in summary["proactive"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
