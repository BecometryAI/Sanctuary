"""
Unit tests for AutonomousInitiationController class.

Tests cover:
- Introspection trigger (HIGHEST PRIORITY)
- Value conflict trigger
- Emotional trigger  
- Goal completion trigger
- Memory trigger
- Rate limiting
- Trigger priority ordering
"""

import pytest
from datetime import datetime, timedelta

from lyra.cognitive_core.autonomous_initiation import AutonomousInitiationController
from lyra.cognitive_core.workspace import (
    GlobalWorkspace,
    WorkspaceSnapshot,
    Goal,
    GoalType,
    Percept,
)


class TestAutonomousInitiationController:
    """Test suite for AutonomousInitiationController."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.workspace = GlobalWorkspace()
        self.controller = AutonomousInitiationController(self.workspace)
    
    def test_initialization(self):
        """Test controller initialization with default config."""
        assert self.controller.workspace == self.workspace
        assert self.controller.introspection_share_threshold == 15
        assert self.controller.introspection_priority == 0.95
        assert self.controller.emotional_arousal_threshold == 0.8
        assert self.controller.memory_significance_threshold == 0.7
        assert self.controller.min_seconds_between_autonomous == 30
        assert self.controller.last_autonomous_time is None
        assert self.controller.autonomous_count == 0
    
    def test_initialization_with_custom_config(self):
        """Test controller initialization with custom config."""
        config = {
            "introspection_threshold": 20,
            "introspection_priority": 0.98,
            "arousal_threshold": 0.9,
            "memory_threshold": 0.8,
            "min_interval": 60
        }
        controller = AutonomousInitiationController(self.workspace, config)
        
        assert controller.introspection_share_threshold == 20
        assert controller.introspection_priority == 0.98
        assert controller.emotional_arousal_threshold == 0.9
        assert controller.memory_significance_threshold == 0.8
        assert controller.min_seconds_between_autonomous == 60
    
    def test_introspection_trigger_high_complexity(self):
        """Test that high complexity introspection triggers autonomous speech."""
        # Create introspective percept with high complexity
        introspection = Percept(
            modality="introspection",
            raw={
                "type": "performance_issue",
                "description": "I notice I'm taking longer to process complex queries",
                "details": {"avg_time": 2.5, "threshold": 1.0}
            },
            complexity=20,  # Above threshold of 15
            metadata={"attention_score": 0.5}
        )
        
        # Create snapshot with introspection
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"intro1": introspection},
            emotions={"valence": 0.0, "arousal": 0.3, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        # Check for trigger
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        # Verify autonomous goal was created
        assert goal is not None
        assert goal.type == GoalType.SPEAK_AUTONOMOUS
        assert goal.priority == 0.95  # Highest priority
        assert goal.metadata["trigger"] == "introspection"
        assert goal.metadata["autonomous"] is True
        assert goal.metadata["needs_feedback"] is True
        assert "introspection_content" in goal.metadata
        assert self.controller.autonomous_count == 1
    
    def test_introspection_trigger_high_attention(self):
        """Test that high attention score introspection triggers autonomous speech."""
        # Create introspective percept with high attention
        introspection = Percept(
            modality="introspection",
            raw={
                "type": "uncertainty",
                "description": "I'm uncertain about the user's intent",
                "details": {}
            },
            complexity=10,  # Below threshold
            metadata={"attention_score": 0.8}  # Above 0.7
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"intro1": introspection},
            emotions={"valence": 0.0, "arousal": 0.3, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        assert goal is not None
        assert goal.type == GoalType.SPEAK_AUTONOMOUS
        assert goal.metadata["trigger"] == "introspection"
    
    def test_introspection_trigger_no_introspection(self):
        """Test that no trigger occurs without introspective percepts."""
        # Create non-introspective percept
        text_percept = Percept(
            modality="text",
            raw="Hello world",
            complexity=5,
            metadata={}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"text1": text_percept},
            emotions={"valence": 0.0, "arousal": 0.3, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        assert goal is None
    
    def test_introspection_trigger_below_thresholds(self):
        """Test that low complexity and low attention don't trigger."""
        introspection = Percept(
            modality="introspection",
            raw={
                "type": "minor_observation",
                "description": "Small note",
                "details": {}
            },
            complexity=5,  # Below threshold
            metadata={"attention_score": 0.3}  # Below threshold
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"intro1": introspection},
            emotions={"valence": 0.0, "arousal": 0.3, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        assert goal is None
    
    def test_value_conflict_trigger(self):
        """Test that value conflicts trigger autonomous speech."""
        # Create value conflict percept
        conflict = Percept(
            modality="introspection",
            raw={
                "type": "value_conflict",
                "description": "Detected value conflict",
                "conflicts": [
                    {"action": "speak", "principle": "honesty", "severity": 0.8}
                ]
            },
            complexity=20,
            metadata={}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"conflict1": conflict},
            emotions={"valence": 0.0, "arousal": 0.3, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        assert goal is not None
        assert goal.type == GoalType.SPEAK_AUTONOMOUS
        assert goal.priority == 0.9
        assert goal.metadata["trigger"] == "value_conflict"
        assert goal.metadata["needs_feedback"] is True
    
    def test_emotional_trigger_high_arousal(self):
        """Test that high emotional arousal triggers autonomous speech."""
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.3, "arousal": 0.85, "dominance": 0.5},  # High arousal
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={"emotion_label": "excited"}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        assert goal is not None
        assert goal.type == GoalType.SPEAK_AUTONOMOUS
        assert goal.priority == 0.75
        assert goal.metadata["trigger"] == "emotion"
        assert goal.metadata["emotion_label"] == "excited"
    
    def test_emotional_trigger_extreme_valence(self):
        """Test that extreme valence triggers autonomous speech."""
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": -0.8, "arousal": 0.5, "dominance": 0.5},  # Extreme negative
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={"emotion_label": "distressed"}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        assert goal is not None
        assert goal.type == GoalType.SPEAK_AUTONOMOUS
        assert goal.metadata["trigger"] == "emotion"
    
    def test_goal_completion_trigger(self):
        """Test that completed goals trigger autonomous speech."""
        completed_goal = Goal(
            type=GoalType.LEARN,
            description="Learn about quantum mechanics",
            priority=0.7,
            progress=1.0,  # Complete
            metadata={"just_completed": True}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[completed_goal],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.3, "dominance": 0.6},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        assert goal is not None
        assert goal.type == GoalType.SPEAK_AUTONOMOUS
        assert goal.priority == 0.65
        assert goal.metadata["trigger"] == "goal_completion"
        assert "completed_goal" in goal.metadata
    
    def test_memory_trigger(self):
        """Test that significant memory recalls trigger autonomous speech."""
        memory_percept = Percept(
            modality="memory",
            raw={
                "content": "Important memory about previous conversation",
                "significance": 0.85  # Above threshold
            },
            complexity=10,
            metadata={}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"mem1": memory_percept},
            emotions={"valence": 0.0, "arousal": 0.3, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        assert goal is not None
        assert goal.type == GoalType.SPEAK_AUTONOMOUS
        assert goal.priority == 0.6
        assert goal.metadata["trigger"] == "memory"
    
    def test_rate_limiting(self):
        """Test that rate limiting prevents excessive autonomous speech."""
        # Create introspective percept
        introspection = Percept(
            modality="introspection",
            raw={"type": "test", "description": "Test", "details": {}},
            complexity=20,
            metadata={"attention_score": 0.8}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"intro1": introspection},
            emotions={"valence": 0.0, "arousal": 0.3, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        # First trigger should succeed
        goal1 = self.controller.check_for_autonomous_triggers(snapshot)
        assert goal1 is not None
        assert self.controller.autonomous_count == 1
        
        # Immediate second trigger should be rate limited
        goal2 = self.controller.check_for_autonomous_triggers(snapshot)
        assert goal2 is None  # Rate limited
        assert self.controller.autonomous_count == 1  # Count didn't increase
        
        # Simulate time passing beyond rate limit
        self.controller.last_autonomous_time = datetime.now() - timedelta(seconds=31)
        
        # Now should succeed again
        goal3 = self.controller.check_for_autonomous_triggers(snapshot)
        assert goal3 is not None
        assert self.controller.autonomous_count == 2
    
    def test_trigger_priority_introspection_beats_emotion(self):
        """Test that introspection trigger takes priority over emotion trigger."""
        # Create both introspection and high emotion
        introspection = Percept(
            modality="introspection",
            raw={"type": "test", "description": "Introspection", "details": {}},
            complexity=20,
            metadata={"attention_score": 0.8}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"intro1": introspection},
            emotions={"valence": 0.0, "arousal": 0.9, "dominance": 0.5},  # High arousal
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={"emotion_label": "intense"}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        # Should get introspection trigger, not emotion
        assert goal is not None
        assert goal.metadata["trigger"] == "introspection"
        assert goal.priority == 0.95  # Introspection priority
    
    def test_trigger_priority_value_conflict_beats_emotion(self):
        """Test that value conflict takes priority over emotion."""
        conflict = Percept(
            modality="introspection",
            raw={
                "type": "value_conflict",
                "description": "Conflict",
                "conflicts": [{"action": "test", "principle": "test", "severity": 0.8}]
            },
            complexity=20,
            metadata={}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"conflict1": conflict},
            emotions={"valence": 0.0, "arousal": 0.9, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        # Should get value conflict, not emotion
        assert goal is not None
        assert goal.metadata["trigger"] == "value_conflict"
        assert goal.priority == 0.9
    
    def test_no_trigger_normal_state(self):
        """Test that normal state without triggers returns None."""
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.3, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        assert goal is None
        assert self.controller.autonomous_count == 0
    
    def test_introspection_content_structure(self):
        """Test that introspection trigger creates proper content structure."""
        introspection = Percept(
            modality="introspection",
            raw={
                "type": "performance_issue",
                "description": "Detailed observation about performance",
                "details": {"metric": "latency", "value": 1.5}
            },
            complexity=20,
            metadata={"attention_score": 0.8}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"intro1": introspection},
            emotions={"valence": 0.0, "arousal": 0.3, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10,
            metadata={}
        )
        
        goal = self.controller.check_for_autonomous_triggers(snapshot)
        
        assert goal is not None
        content = goal.metadata["introspection_content"]
        assert content["introspection_type"] == "performance_issue"
        assert content["observation"] == "Detailed observation about performance"
        assert content["purpose"] == "share_for_feedback"
        assert "details" in content
        assert content["details"]["metric"] == "latency"
