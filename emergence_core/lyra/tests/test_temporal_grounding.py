"""
Tests for the temporal grounding system.

This test module validates all components of the temporal grounding implementation:
- TemporalContext and TemporalAwareness with session tracking
- Session detection and management
- Time passage effects on cognitive state
- Temporal pattern learning and expectations
- Relative time descriptions
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from emergence_core.lyra.cognitive_core.temporal import (
    TemporalAwareness,
    TemporalContext,
    Session,
    SessionManager,
    TimePassageEffects,
    TemporalExpectations,
    TemporalExpectation,
    RelativeTime,
    TemporalGrounding
)


class TestTemporalContext:
    """
    Tests for TemporalContext dataclass.
    
    Validates:
    - Context creation with all required fields
    - Time description generation for various durations
    - Session description formatting
    """
    
    def test_temporal_context_creation(self):
        """Test creating a temporal context."""
        now = datetime.now()
        session_start = now - timedelta(minutes=30)
        last_interaction = now - timedelta(minutes=5)
        
        context = TemporalContext(
            current_time=now,
            session_start=session_start,
            last_interaction=last_interaction,
            elapsed_since_last=now - last_interaction,
            session_duration=now - session_start,
            is_new_session=False,
            session_number=1
        )
        
        assert context.current_time == now
        assert context.session_number == 1
        assert not context.is_new_session
    
    def test_time_description_moments_ago(self):
        """Test time description for very recent interactions."""
        now = datetime.now()
        last = now - timedelta(minutes=2)
        
        context = TemporalContext(
            current_time=now,
            session_start=now,
            last_interaction=last,
            elapsed_since_last=now - last,
            session_duration=timedelta(0),
            is_new_session=True,
            session_number=1
        )
        
        assert context.time_description == "moments ago"
    
    def test_time_description_minutes(self):
        """Test time description for minutes."""
        now = datetime.now()
        last = now - timedelta(minutes=15)
        
        context = TemporalContext(
            current_time=now,
            session_start=now,
            last_interaction=last,
            elapsed_since_last=now - last,
            session_duration=timedelta(0),
            is_new_session=True,
            session_number=1
        )
        
        assert "minutes ago" in context.time_description
    
    def test_session_description(self):
        """Test session duration description."""
        now = datetime.now()
        session_start = now - timedelta(minutes=45)
        
        context = TemporalContext(
            current_time=now,
            session_start=session_start,
            last_interaction=now,
            elapsed_since_last=timedelta(0),
            session_duration=now - session_start,
            is_new_session=False,
            session_number=1
        )
        
        assert "minutes" in context.session_description


class TestTemporalAwareness:
    """Tests for enhanced TemporalAwareness with session tracking."""
    
    def test_initialization(self):
        """Test temporal awareness initialization."""
        ta = TemporalAwareness()
        
        assert ta.session_gap_threshold == timedelta(hours=1)
        assert ta.current_session is None
        assert len(ta.session_history) == 0
        assert ta._session_counter == 0
    
    def test_custom_session_threshold(self):
        """Test custom session gap threshold."""
        custom_gap = timedelta(minutes=30)
        ta = TemporalAwareness(session_gap_threshold=custom_gap)
        
        assert ta.session_gap_threshold == custom_gap
    
    def test_first_interaction_creates_session(self):
        """Test that first interaction creates a new session."""
        ta = TemporalAwareness()
        
        context = ta.update()
        
        assert context.is_new_session
        assert context.session_number == 1
        assert ta.current_session is not None
        assert ta.current_session.interaction_count == 1
    
    def test_quick_interactions_same_session(self):
        """Test that quick interactions stay in same session."""
        ta = TemporalAwareness()
        
        # First interaction
        time1 = datetime.now()
        context1 = ta.update(time1)
        
        # Second interaction 5 minutes later
        time2 = time1 + timedelta(minutes=5)
        context2 = ta.update(time2)
        
        assert context1.is_new_session
        assert not context2.is_new_session
        assert context1.session_number == context2.session_number
        assert ta.current_session.interaction_count == 2
    
    def test_gap_creates_new_session(self):
        """Test that long gap creates new session."""
        ta = TemporalAwareness(session_gap_threshold=timedelta(minutes=30))
        
        # First interaction
        time1 = datetime.now()
        context1 = ta.update(time1)
        
        # Second interaction 1 hour later (exceeds threshold)
        time2 = time1 + timedelta(hours=1)
        context2 = ta.update(time2)
        
        assert context1.is_new_session
        assert context2.is_new_session
        assert context2.session_number == 2
        assert len(ta.session_history) == 1
    
    def test_get_last_session(self):
        """Test retrieving last completed session."""
        ta = TemporalAwareness(session_gap_threshold=timedelta(minutes=10))
        
        # First session
        time1 = datetime.now()
        ta.update(time1)
        
        # Second session (after gap)
        time2 = time1 + timedelta(minutes=15)
        ta.update(time2)
        
        last_session = ta.get_last_session()
        
        assert last_session is not None
        assert last_session.interaction_count == 1
    
    def test_get_context(self):
        """Test getting temporal context dictionary."""
        ta = TemporalAwareness()
        ta.update()
        
        context_dict = ta.get_context()
        
        assert "session_id" in context_dict
        assert "session_number" in context_dict
        assert "interaction_count" in context_dict
        assert context_dict["session_number"] == 1
        assert ta.session_count == 1  # Use public property instead of private attribute


class TestSessionManager:
    """Tests for SessionManager."""
    
    def test_initialization(self):
        """Test session manager initialization."""
        ta = TemporalAwareness()
        sm = SessionManager(ta)
        
        assert sm.temporal == ta
        assert sm.memory is None
    
    def test_session_start_first_time(self):
        """Test handling first session start."""
        ta = TemporalAwareness()
        sm = SessionManager(ta)
        
        context = ta.update()
        sm.on_session_start(ta.current_session)
        
        # Verify session started without errors and has expected state
        assert ta.current_session is not None
        assert ta.current_session.interaction_count >= 1
    
    def test_greeting_context_first_meeting(self):
        """Test greeting context for first meeting."""
        ta = TemporalAwareness()
        sm = SessionManager(ta)
        
        greeting = sm.get_session_greeting_context()
        
        assert greeting["type"] == "first_meeting"
        assert "first" in greeting["context"].lower()
    
    def test_greeting_context_continuation(self):
        """Test greeting context for continuation."""
        ta = TemporalAwareness(session_gap_threshold=timedelta(hours=1))
        sm = SessionManager(ta)
        
        # First session
        time1 = datetime.now()
        ta.update(time1)
        
        # End session and start new one (after gap)
        time2 = time1 + timedelta(hours=2)
        ta.update(time2)
        
        # Get greeting context
        greeting = sm.get_session_greeting_context()
        
        # Should recognize it's a resumption (not first meeting)
        assert greeting["type"] in ["continuation", "same_day", "recent"]
        assert greeting["type"] != "first_meeting"
    
    def test_record_topic(self):
        """Test recording topics in session."""
        ta = TemporalAwareness()
        sm = SessionManager(ta)
        
        ta.update()
        sm.record_topic("consciousness")
        sm.record_topic("emotions")
        
        assert "consciousness" in ta.current_session.topics
        assert "emotions" in ta.current_session.topics
        assert len(ta.current_session.topics) == 2
    
    def test_record_emotional_state(self):
        """Test recording emotional states."""
        ta = TemporalAwareness()
        sm = SessionManager(ta)
        
        ta.update()
        
        emotion_state = {"valence": 0.5, "arousal": 0.3}
        sm.record_emotional_state(emotion_state)
        
        assert len(ta.current_session.emotional_arc) == 1
        assert ta.current_session.emotional_arc[0] == emotion_state


class TestTimePassageEffects:
    """Tests for TimePassageEffects."""
    
    def test_initialization(self):
        """Test time passage effects initialization."""
        tpe = TimePassageEffects()
        
        assert tpe.emotion_decay_rate == 0.9
        assert "valence" in tpe.emotion_baseline
    
    def test_emotion_decay(self):
        """Test emotional decay toward baseline."""
        tpe = TimePassageEffects()
        
        emotions = {"valence": 0.8, "arousal": 0.9, "dominance": 0.7}
        elapsed = timedelta(hours=2)
        
        decayed = tpe._decay_emotions(emotions, elapsed)
        
        # Emotions should move toward baseline
        assert abs(decayed["valence"]) < abs(emotions["valence"])
        assert abs(decayed["arousal"]) < abs(emotions["arousal"])
    
    def test_apply_effects(self):
        """Test applying all time passage effects."""
        tpe = TimePassageEffects()
        
        state = {
            "emotions": {"valence": 0.7, "arousal": 0.8, "dominance": 0.6},
            "working_memory": [{"item": "test", "salience": 0.9}],
            "goals": []
        }
        
        elapsed = timedelta(hours=2)
        updated = tpe.apply(elapsed, state)
        
        assert "emotions" in updated
        assert "consolidation_needed" in updated
        assert updated["consolidation_needed"]
    
    def test_goal_urgency_update(self):
        """Test updating goal urgency based on deadlines."""
        tpe = TimePassageEffects()
        
        # Goal with deadline in 12 hours
        deadline = datetime.now() + timedelta(hours=12)
        goals = [
            {
                "description": "test goal",
                "urgency": 0.5,
                "deadline": deadline.isoformat()
            }
        ]
        
        elapsed = timedelta(hours=1)
        updated_goals = tpe._update_urgencies(goals, elapsed)
        
        # Urgency should increase as deadline approaches
        assert updated_goals[0]["urgency"] >= 0.5


class TestTemporalExpectations:
    """Tests for TemporalExpectations."""
    
    def test_initialization(self):
        """Test temporal expectations initialization."""
        te = TemporalExpectations(min_observations=3)
        
        assert te.min_observations == 3
        assert len(te.patterns) == 0
    
    def test_record_event(self):
        """Test recording events."""
        te = TemporalExpectations()
        
        te.record_event("user_interaction")
        te.record_event("user_interaction")
        
        assert "user_interaction" in te.patterns
        assert len(te.patterns["user_interaction"]) == 2
    
    def test_insufficient_observations(self):
        """Test that expectations require minimum observations."""
        te = TemporalExpectations(min_observations=3)
        
        te.record_event("test_event")
        te.record_event("test_event")
        
        expectation = te.get_expectation("test_event")
        
        assert expectation is None
    
    def test_expectation_formation(self):
        """Test forming expectations from patterns."""
        te = TemporalExpectations(min_observations=3)
        
        base_time = datetime.now()
        
        # Record events at regular intervals
        te.record_event("daily_check", base_time)
        te.record_event("daily_check", base_time + timedelta(days=1))
        te.record_event("daily_check", base_time + timedelta(days=2))
        te.record_event("daily_check", base_time + timedelta(days=3))
        
        expectation = te.get_expectation("daily_check")
        
        assert expectation is not None
        assert expectation.event_type == "daily_check"
        assert expectation.confidence > 0
    
    def test_overdue_expectations(self):
        """Test detecting overdue expectations."""
        te = TemporalExpectations(min_observations=3)
        
        # Create pattern in the past
        base_time = datetime.now() - timedelta(days=10)
        
        te.record_event("overdue_event", base_time)
        te.record_event("overdue_event", base_time + timedelta(days=1))
        te.record_event("overdue_event", base_time + timedelta(days=2))
        
        overdue = te.get_overdue_expectations()
        
        # Should have one overdue expectation
        assert len(overdue) > 0


class TestRelativeTime:
    """Tests for RelativeTime utilities."""
    
    def test_describe_just_now(self):
        """Test describing very recent time."""
        now = datetime.now()
        recent = now - timedelta(seconds=5)
        
        description = RelativeTime.describe(recent, now)
        
        assert description == "just now"
    
    def test_describe_minutes_ago(self):
        """Test describing minutes ago."""
        now = datetime.now()
        past = now - timedelta(minutes=30)
        
        description = RelativeTime.describe(past, now)
        
        assert "minutes ago" in description
    
    def test_describe_hours_ago(self):
        """Test describing hours ago."""
        now = datetime.now()
        past = now - timedelta(hours=3)
        
        description = RelativeTime.describe(past, now)
        
        assert "hours ago" in description
    
    def test_describe_days_ago(self):
        """Test describing days ago."""
        now = datetime.now()
        past = now - timedelta(days=2)
        
        description = RelativeTime.describe(past, now)
        
        assert "days ago" in description
    
    def test_describe_future(self):
        """Test describing future time."""
        now = datetime.now()
        future = now + timedelta(hours=2)
        
        description = RelativeTime.describe(future, now)
        
        assert "in 2 hours" in description
    
    def test_is_recent(self):
        """Test checking if time is recent."""
        now = datetime.now()
        recent = now - timedelta(minutes=30)
        old = now - timedelta(hours=3)
        
        assert RelativeTime.is_recent(recent)
        assert not RelativeTime.is_recent(old)
    
    def test_is_today(self):
        """Test checking if time is today."""
        now = datetime.now()
        earlier_today = now - timedelta(hours=2)
        yesterday = now - timedelta(days=1)
        
        assert RelativeTime.is_today(earlier_today)
        assert not RelativeTime.is_today(yesterday)
    
    def test_categorize_recency(self):
        """Test categorizing recency."""
        now = datetime.now()
        
        just_now = now - timedelta(minutes=2)
        recent = now - timedelta(minutes=30)
        today = now - timedelta(hours=5)
        
        assert RelativeTime.categorize_recency(just_now) == "now"
        assert RelativeTime.categorize_recency(recent) == "recent"
        assert RelativeTime.categorize_recency(today) == "today"


class TestTemporalGrounding:
    """Tests for integrated TemporalGrounding system."""
    
    def test_initialization(self):
        """Test temporal grounding initialization."""
        tg = TemporalGrounding()
        
        assert tg.awareness is not None
        assert tg.sessions is not None
        assert tg.effects is not None
        assert tg.expectations is not None
    
    def test_on_interaction(self):
        """Test interaction handling."""
        tg = TemporalGrounding()
        
        context = tg.on_interaction()
        
        assert isinstance(context, TemporalContext)
        assert context.is_new_session
    
    def test_get_temporal_state(self):
        """Test getting complete temporal state."""
        tg = TemporalGrounding()
        
        tg.on_interaction()
        state = tg.get_temporal_state()
        
        assert "context" in state
        assert "session" in state
        assert "greeting_context" in state
        assert "expectations" in state
    
    def test_record_topic(self):
        """Test recording topics through grounding."""
        tg = TemporalGrounding()
        
        tg.on_interaction()
        tg.record_topic("test_topic")
        
        session_info = tg.sessions.get_current_session_info()
        assert "test_topic" in session_info["topics"]
    
    def test_apply_time_passage_effects(self):
        """Test applying time passage effects."""
        tg = TemporalGrounding()
        
        state = {
            "emotions": {"valence": 0.8, "arousal": 0.7, "dominance": 0.6},
            "goals": []
        }
        
        # Simulate time passage
        tg._last_effect_time = datetime.now() - timedelta(hours=2)
        updated = tg.apply_time_passage_effects(state)
        
        assert "emotions" in updated
    
    def test_describe_time(self):
        """Test describing timestamps."""
        tg = TemporalGrounding()
        
        past = datetime.now() - timedelta(hours=2)
        description = tg.describe_time(past)
        
        assert "hours ago" in description
    
    def test_end_session(self):
        """Test ending a session."""
        tg = TemporalGrounding()
        
        tg.on_interaction()
        tg.end_session()
        
        # Should archive current session
        assert len(tg.awareness.session_history) == 1
