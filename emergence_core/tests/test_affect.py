"""
Unit tests for AffectSubsystem class.

Tests cover:
- Initialization and baseline state
- Goal-based emotional updates
- Percept-based emotional updates
- Action-based emotional updates
- Emotional decay
- Emotion labeling
- Attention influence
- Action influence
- State serialization
"""

import pytest
from datetime import datetime

from lyra.cognitive_core.affect import (
    AffectSubsystem,
    EmotionalState,
)
from lyra.cognitive_core.workspace import (
    Goal,
    GoalType,
    Percept,
    WorkspaceSnapshot,
)
from lyra.cognitive_core.action import (
    Action,
    ActionType,
)


class TestEmotionalState:
    """Test EmotionalState dataclass."""
    
    def test_emotional_state_creation(self):
        """Test creating an EmotionalState."""
        state = EmotionalState(
            valence=0.5,
            arousal=0.7,
            dominance=0.6
        )
        
        assert state.valence == 0.5
        assert state.arousal == 0.7
        assert state.dominance == 0.6
        assert isinstance(state.timestamp, datetime)
        assert state.intensity > 0.0
    
    def test_emotional_state_to_dict(self):
        """Test converting EmotionalState to dict."""
        state = EmotionalState(
            valence=0.3,
            arousal=0.4,
            dominance=0.5
        )
        
        state_dict = state.to_dict()
        assert state_dict["valence"] == 0.3
        assert state_dict["arousal"] == 0.4
        assert state_dict["dominance"] == 0.5


class TestAffectSubsystemInitialization:
    """Test AffectSubsystem initialization."""
    
    def test_initialization_with_defaults(self):
        """Test creating AffectSubsystem with default config."""
        affect = AffectSubsystem()
        
        # Check baseline values
        assert "valence" in affect.baseline
        assert "arousal" in affect.baseline
        assert "dominance" in affect.baseline
        
        # Check current state matches baseline
        assert affect.valence == affect.baseline["valence"]
        assert affect.arousal == affect.baseline["arousal"]
        assert affect.dominance == affect.baseline["dominance"]
        
        # Check parameters
        assert 0.0 < affect.decay_rate < 1.0
        assert 0.0 < affect.sensitivity < 1.0
        assert len(affect.emotion_history) == 0
    
    def test_initialization_with_custom_config(self):
        """Test creating AffectSubsystem with custom config."""
        config = {
            "baseline": {
                "valence": 0.2,
                "arousal": 0.5,
                "dominance": 0.7
            },
            "decay_rate": 0.1,
            "sensitivity": 0.5,
            "history_size": 50
        }
        
        affect = AffectSubsystem(config=config)
        
        assert affect.baseline["valence"] == 0.2
        assert affect.baseline["arousal"] == 0.5
        assert affect.baseline["dominance"] == 0.7
        assert affect.decay_rate == 0.1
        assert affect.sensitivity == 0.5
        assert affect.emotion_history.maxlen == 50


class TestGoalBasedEmotions:
    """Test emotional updates from goal states."""
    
    def test_goal_progress_increases_valence(self):
        """Test that high goal progress increases valence."""
        affect = AffectSubsystem()
        initial_valence = affect.valence
        
        # Create goals with high progress
        goals = [
            Goal(
                type=GoalType.RESPOND_TO_USER,
                description="Test goal",
                progress=0.9,
                priority=0.8
            )
        ]
        
        snapshot = WorkspaceSnapshot(
            goals=goals,
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        affect.compute_update(snapshot)
        
        # Valence should increase due to high progress
        assert affect.valence > initial_valence
    
    def test_many_goals_increase_arousal(self):
        """Test that multiple goals increase arousal."""
        affect = AffectSubsystem()
        initial_arousal = affect.arousal
        
        # Create many goals
        goals = [
            Goal(type=GoalType.RESPOND_TO_USER, description=f"Goal {i}", priority=0.6)
            for i in range(5)
        ]
        
        snapshot = WorkspaceSnapshot(
            goals=goals,
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        affect.compute_update(snapshot)
        
        # Arousal should increase due to many goals
        assert affect.arousal > initial_arousal
    
    def test_high_priority_goals_boost_dominance(self):
        """Test that high-priority goals increase dominance."""
        affect = AffectSubsystem()
        initial_dominance = affect.dominance
        
        # Create high-priority goal
        goals = [
            Goal(
                type=GoalType.COMMIT_MEMORY,
                description="Important goal",
                priority=0.9
            )
        ]
        
        snapshot = WorkspaceSnapshot(
            goals=goals,
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        affect.compute_update(snapshot)
        
        # Dominance should increase
        assert affect.dominance >= initial_dominance
    
    def test_completed_goals_boost_emotions(self):
        """Test that completed goals boost valence and dominance."""
        affect = AffectSubsystem()
        initial_valence = affect.valence
        initial_dominance = affect.dominance
        
        # Create completed goal
        goals = [
            Goal(
                type=GoalType.LEARN,
                description="Completed goal",
                progress=1.0,
                priority=0.8
            )
        ]
        
        snapshot = WorkspaceSnapshot(
            goals=goals,
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        affect.compute_update(snapshot)
        
        # Both should increase
        assert affect.valence > initial_valence
        assert affect.dominance > initial_dominance


class TestPerceptBasedEmotions:
    """Test emotional updates from percepts."""
    
    def test_positive_keywords_increase_valence(self):
        """Test that positive keywords increase valence."""
        affect = AffectSubsystem()
        initial_valence = affect.valence
        
        # Create percept with positive keywords
        percept = Percept(
            modality="text",
            raw="I am so happy and excited about this wonderful progress!",
            complexity=10
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={percept.id: percept.model_dump()},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        affect.compute_update(snapshot)
        
        # Valence should increase
        assert affect.valence > initial_valence
    
    def test_negative_keywords_decrease_valence(self):
        """Test that negative keywords decrease valence."""
        affect = AffectSubsystem()
        initial_valence = affect.valence
        
        # Create percept with negative keywords
        percept = Percept(
            modality="text",
            raw="I am sad and worried about this terrible situation",
            complexity=10
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={percept.id: percept.model_dump()},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        affect.compute_update(snapshot)
        
        # Valence should decrease
        assert affect.valence < initial_valence
    
    def test_urgent_keywords_increase_arousal(self):
        """Test that urgent keywords increase arousal."""
        affect = AffectSubsystem()
        initial_arousal = affect.arousal
        
        # Create percept with urgent keywords
        percept = Percept(
            modality="text",
            raw="Urgent crisis requires immediate attention!",
            complexity=20
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={percept.id: percept.model_dump()},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        affect.compute_update(snapshot)
        
        # Arousal should increase
        assert affect.arousal > initial_arousal
    
    def test_high_complexity_increases_arousal(self):
        """Test that high complexity percepts increase arousal."""
        affect = AffectSubsystem()
        initial_arousal = affect.arousal
        
        # Create high complexity percept
        percept = Percept(
            modality="text",
            raw="Complex information",
            complexity=40
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={percept.id: percept.model_dump()},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        affect.compute_update(snapshot)
        
        # Arousal should increase
        assert affect.arousal > initial_arousal
    
    def test_introspection_with_value_conflict(self):
        """Test that value conflict introspections affect emotions."""
        affect = AffectSubsystem()
        initial_valence = affect.valence
        
        # Create introspective percept with value conflict
        percept = Percept(
            modality="introspection",
            raw={"type": "value_conflict", "description": "Conflicting values detected"},
            complexity=25
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={percept.id: percept.model_dump()},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        affect.compute_update(snapshot)
        
        # Valence should decrease, arousal should increase
        assert affect.valence < initial_valence


class TestActionBasedEmotions:
    """Test emotional updates from actions."""
    
    def test_speak_action_boosts_arousal_and_dominance(self):
        """Test that SPEAK actions increase arousal and dominance."""
        # Use no decay and high sensitivity to clearly see the effect
        config = {"decay_rate": 0.0, "sensitivity": 1.0}
        affect = AffectSubsystem(config=config)
        
        # Add a goal to avoid negative goal effects
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Test goal",
            priority=0.5,
            progress=0.5
        )
        
        # Create multiple SPEAK actions for stronger effect
        actions = [
            Action(type=ActionType.SPEAK, priority=0.8),
            Action(type=ActionType.SPEAK, priority=0.8),
        ]
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1,
            metadata={"recent_actions": actions}
        )
        
        # Get baseline with just goals
        affect_baseline = AffectSubsystem(config=config)
        snapshot_baseline = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1,
            metadata={"recent_actions": []}
        )
        affect_baseline.compute_update(snapshot_baseline)
        
        # Now test with actions
        affect.compute_update(snapshot)
        
        # With actions should have higher arousal and dominance than baseline
        assert affect.arousal > affect_baseline.arousal
        assert affect.dominance > affect_baseline.dominance
    
    def test_commit_memory_positive_effect(self):
        """Test that COMMIT_MEMORY actions have positive effect."""
        # Use no decay and high sensitivity to clearly see the effect
        config = {"decay_rate": 0.0, "sensitivity": 1.0}
        affect = AffectSubsystem(config=config)
        
        # Add a goal to avoid negative goal effects
        goal = Goal(
            type=GoalType.COMMIT_MEMORY,
            description="Test goal",
            priority=0.5,
            progress=0.5
        )
        
        # Create multiple COMMIT_MEMORY actions for stronger effect
        actions = [
            Action(type=ActionType.COMMIT_MEMORY, priority=0.6),
            Action(type=ActionType.COMMIT_MEMORY, priority=0.6),
        ]
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1,
            metadata={"recent_actions": actions}
        )
        
        # Get baseline with just goals
        affect_baseline = AffectSubsystem(config=config)
        snapshot_baseline = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1,
            metadata={"recent_actions": []}
        )
        affect_baseline.compute_update(snapshot_baseline)
        
        # Now test with actions
        affect.compute_update(snapshot)
        
        # With actions should have higher valence and dominance than baseline
        assert affect.valence > affect_baseline.valence
        assert affect.dominance > affect_baseline.dominance
    
    def test_blocked_actions_decrease_dominance(self):
        """Test that blocked actions decrease dominance."""
        affect = AffectSubsystem()
        initial_dominance = affect.dominance
        
        # Create blocked action
        action = Action(
            type=ActionType.TOOL_CALL,
            priority=0.7,
            metadata={"blocked": True}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1,
            metadata={"recent_actions": [action]}
        )
        
        affect.compute_update(snapshot)
        
        # Dominance should decrease
        assert affect.dominance < initial_dominance


class TestEmotionalDecay:
    """Test emotional decay mechanism."""
    
    def test_decay_returns_to_baseline(self):
        """Test that extreme emotions gradually return to baseline."""
        config = {
            "baseline": {"valence": 0.1, "arousal": 0.3, "dominance": 0.6},
            "decay_rate": 0.2,  # Faster decay for testing
            "sensitivity": 0.3
        }
        affect = AffectSubsystem(config=config)
        
        # Set extreme emotional state
        affect.valence = 0.9
        affect.arousal = 0.9
        affect.dominance = 0.9
        
        # Create empty snapshot
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        # Run several updates to allow decay
        for _ in range(10):
            affect.compute_update(snapshot)
        
        # Should be closer to baseline
        assert abs(affect.valence - 0.1) < 0.5
        assert abs(affect.arousal - 0.3) < 0.4
        assert abs(affect.dominance - 0.6) < 0.4


class TestEmotionLabeling:
    """Test emotion label generation."""
    
    def test_excited_label(self):
        """Test 'excited' label for high arousal, positive valence, high dominance."""
        affect = AffectSubsystem()
        affect.valence = 0.5
        affect.arousal = 0.8
        affect.dominance = 0.7
        
        label = affect.get_emotion_label()
        assert label == "excited"
    
    def test_anxious_label(self):
        """Test 'anxious' label for high arousal, negative valence, low dominance."""
        affect = AffectSubsystem()
        affect.valence = -0.5
        affect.arousal = 0.8
        affect.dominance = 0.3
        
        label = affect.get_emotion_label()
        assert label == "anxious"
    
    def test_content_label(self):
        """Test 'content' label for low arousal, positive valence, high dominance."""
        affect = AffectSubsystem()
        affect.valence = 0.5
        affect.arousal = 0.2
        affect.dominance = 0.7
        
        label = affect.get_emotion_label()
        assert label == "content"
    
    def test_calm_label(self):
        """Test 'calm' label for low arousal, neutral valence."""
        affect = AffectSubsystem()
        affect.valence = 0.0
        affect.arousal = 0.2
        affect.dominance = 0.5
        
        label = affect.get_emotion_label()
        assert label == "calm"
    
    def test_neutral_label(self):
        """Test 'neutral' label for mid arousal, neutral valence."""
        affect = AffectSubsystem()
        affect.valence = 0.0
        affect.arousal = 0.5
        affect.dominance = 0.5
        
        label = affect.get_emotion_label()
        assert label == "neutral"


class TestAttentionInfluence:
    """Test attention score modification."""
    
    def test_high_arousal_boosts_urgent_percepts(self):
        """Test that high arousal boosts attention to urgent percepts."""
        affect = AffectSubsystem()
        affect.arousal = 0.8
        
        percept = Percept(
            modality="text",
            raw="Urgent matter needs attention",
            complexity=20
        )
        
        base_score = 0.5
        modified_score = affect.influence_attention(base_score, percept.model_dump())
        
        # Score should be boosted
        assert modified_score > base_score
    
    def test_high_arousal_boosts_complex_percepts(self):
        """Test that high arousal boosts attention to complex percepts."""
        affect = AffectSubsystem()
        affect.arousal = 0.8
        
        percept = Percept(
            modality="text",
            raw="Complex information",
            complexity=35
        )
        
        base_score = 0.5
        modified_score = affect.influence_attention(base_score, percept.model_dump())
        
        # Score should be boosted
        assert modified_score > base_score
    
    def test_negative_valence_boosts_introspection(self):
        """Test that negative valence boosts introspective percepts."""
        affect = AffectSubsystem()
        affect.valence = -0.5
        
        percept = Percept(
            modality="introspection",
            raw="Internal reflection",
            complexity=15
        )
        
        base_score = 0.5
        modified_score = affect.influence_attention(base_score, percept.model_dump())
        
        # Score should be boosted
        assert modified_score > base_score
    
    def test_low_dominance_boosts_supportive_percepts(self):
        """Test that low dominance boosts supportive percepts."""
        affect = AffectSubsystem()
        affect.dominance = 0.2
        
        percept = Percept(
            modality="text",
            raw="Here to help and support you",
            complexity=10
        )
        
        base_score = 0.5
        modified_score = affect.influence_attention(base_score, percept.model_dump())
        
        # Score should be boosted
        assert modified_score > base_score


class TestActionInfluence:
    """Test action priority modification."""
    
    def test_high_arousal_boosts_speak_actions(self):
        """Test that high arousal boosts SPEAK actions."""
        affect = AffectSubsystem()
        affect.arousal = 0.8
        
        action = Action(type=ActionType.SPEAK, priority=0.5)
        
        modified_priority = affect.influence_action(0.5, action.model_dump())
        
        # Priority should be boosted
        assert modified_priority > 0.5
    
    def test_high_arousal_boosts_tool_call_actions(self):
        """Test that high arousal boosts TOOL_CALL actions."""
        affect = AffectSubsystem()
        affect.arousal = 0.8
        
        action = Action(type=ActionType.TOOL_CALL, priority=0.5)
        
        modified_priority = affect.influence_action(0.5, action.model_dump())
        
        # Priority should be boosted
        assert modified_priority > 0.5
    
    def test_low_dominance_boosts_introspect_actions(self):
        """Test that low dominance boosts INTROSPECT actions."""
        affect = AffectSubsystem()
        affect.dominance = 0.3
        
        action = Action(type=ActionType.INTROSPECT, priority=0.5)
        
        modified_priority = affect.influence_action(0.5, action.model_dump())
        
        # Priority should be boosted
        assert modified_priority > 0.5
    
    def test_negative_valence_boosts_wait_actions(self):
        """Test that negative valence boosts WAIT actions."""
        affect = AffectSubsystem()
        affect.valence = -0.5
        
        action = Action(type=ActionType.WAIT, priority=0.5)
        
        modified_priority = affect.influence_action(0.5, action.model_dump())
        
        # Priority should be boosted
        assert modified_priority > 0.5


class TestStateSerialization:
    """Test state serialization methods."""
    
    def test_get_state(self):
        """Test get_state method returns complete state."""
        affect = AffectSubsystem()
        affect.valence = 0.3
        affect.arousal = 0.5
        affect.dominance = 0.6
        
        state = affect.get_state()
        
        assert "valence" in state
        assert "arousal" in state
        assert "dominance" in state
        assert "label" in state
        assert "history_size" in state
        assert "baseline" in state
        
        assert state["valence"] == 0.3
        assert state["arousal"] == 0.5
        assert state["dominance"] == 0.6
        assert isinstance(state["label"], str)


class TestEmotionHistory:
    """Test emotion history tracking."""
    
    def test_history_tracks_states(self):
        """Test that emotion history tracks emotional states."""
        affect = AffectSubsystem()
        
        # Create snapshot
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        initial_size = len(affect.emotion_history)
        
        # Run several updates
        for _ in range(5):
            affect.compute_update(snapshot)
        
        # History should grow
        assert len(affect.emotion_history) == initial_size + 5
    
    def test_history_respects_maxlen(self):
        """Test that history respects maximum length."""
        config = {"history_size": 10}
        affect = AffectSubsystem(config=config)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        # Run more updates than maxlen
        for _ in range(20):
            affect.compute_update(snapshot)
        
        # History should not exceed maxlen
        assert len(affect.emotion_history) == 10
