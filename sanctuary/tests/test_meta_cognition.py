"""
Unit tests for meta_cognition SelfMonitor.

Tests cover:
- Initialization and configuration
- Value alignment checking
- Performance assessment
- Uncertainty detection
- Emotional observation
- Pattern detection
- Monitoring frequency control
"""

import pytest
from datetime import datetime
from pathlib import Path
from collections import deque

from mind.cognitive_core.meta_cognition import SelfMonitor
from mind.cognitive_core.workspace import (
    GlobalWorkspace, WorkspaceSnapshot, Percept, Goal, GoalType
)
from mind.cognitive_core.action import Action, ActionType
from mind.cognitive_core.affect import AffectSubsystem, EmotionalState


class TestSelfMonitorInitialization:
    """Test SelfMonitor initialization"""
    
    def test_initialization_default(self):
        """Test creating SelfMonitor with default parameters"""
        monitor = SelfMonitor()
        assert monitor is not None
        assert isinstance(monitor, SelfMonitor)
        assert monitor.observation_history is not None
        assert monitor.monitoring_frequency == 10  # Default
        assert monitor.cycle_count == 0
        assert "total_observations" in monitor.stats
    
    def test_initialization_with_workspace(self):
        """Test creating SelfMonitor with workspace reference"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        assert monitor.workspace == workspace
    
    def test_initialization_with_config(self):
        """Test creating SelfMonitor with custom config"""
        config = {"monitoring_frequency": 5}
        monitor = SelfMonitor(config=config)
        assert monitor.monitoring_frequency == 5
    
    def test_charter_loading(self):
        """Test that charter is loaded if file exists"""
        monitor = SelfMonitor()
        # Charter should be loaded if file exists
        if Path("data/identity/charter.md").exists():
            assert len(monitor.charter_text) > 0
    
    def test_protocols_loading(self):
        """Test that protocols are loaded if file exists"""
        monitor = SelfMonitor()
        # Protocols should be loaded if file exists
        if Path("data/identity/protocols.md").exists():
            assert len(monitor.protocols_text) > 0


class TestValueConflictDetection:
    """Test value alignment checking"""
    
    def test_no_conflicts_with_empty_actions(self):
        """Test that no conflicts are detected with no actions"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": []}
        )
        
        percept = monitor._check_value_alignment(snapshot)
        assert percept is None
    
    def test_conflict_with_claimed_capability(self):
        """Test that conflict is detected when claiming false capability"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        # Create action with claimed capability
        action = Action(
            type=ActionType.SPEAK,
            metadata={"claimed_capability": True}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={"recent_actions": [action]}
        )
        
        percept = monitor._check_value_alignment(snapshot)
        assert percept is not None
        assert percept.modality == "introspection"
        assert "value_conflict" in percept.raw["type"]


class TestPerformanceAssessment:
    """Test performance assessment"""
    
    def test_no_issues_with_healthy_state(self):
        """Test that no issues detected with healthy workspace state"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        percept = monitor._assess_performance(snapshot)
        assert percept is None
    
    def test_stalled_goals_detection(self):
        """Test detection of stalled goals"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        # Create stalled goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Test goal",
            priority=0.8,
            progress=0.05,  # Low progress
            metadata={"age_cycles": 100}  # Old goal
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        percept = monitor._assess_performance(snapshot)
        assert percept is not None
        assert percept.modality == "introspection"
        assert "performance_issue" in percept.raw["type"]
    
    def test_workspace_overload_detection(self):
        """Test detection of workspace overload"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        # Create many percepts
        percepts = {
            f"p{i}": {"modality": "text", "raw": f"percept {i}"}
            for i in range(25)  # More than 20
        }
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts=percepts,
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        percept = monitor._assess_performance(snapshot)
        assert percept is not None
        assert "workspace_overload" in str(percept.raw)


class TestUncertaintyDetection:
    """Test uncertainty detection"""
    
    def test_no_uncertainty_with_clear_state(self):
        """Test no uncertainty with clear workspace state"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.8, "arousal": 0.5, "dominance": 0.7},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        percept = monitor._detect_uncertainty(snapshot)
        assert percept is None
    
    def test_emotional_ambiguity_detection(self):
        """Test detection of emotionally ambiguous state"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        # Mid-range emotions indicate ambiguity
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        percept = monitor._detect_uncertainty(snapshot)
        assert percept is not None
        assert "uncertainty" in percept.raw["type"]
    
    def test_goal_conflict_detection(self):
        """Test detection of conflicting goals"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        # Create conflicting goals
        goal1 = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="avoid speaking",
            priority=0.8
        )
        goal2 = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="engage with user",
            priority=0.8
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[goal1, goal2],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        conflicts = monitor._detect_goal_conflicts(snapshot.goals)
        assert conflicts is True


class TestEmotionalObservation:
    """Test emotional observation"""
    
    def test_no_observation_without_affect_subsystem(self):
        """Test no observation when affect subsystem unavailable"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        percept = monitor._observe_emotions(snapshot)
        # Should return None because workspace doesn't have affect subsystem
        assert percept is None
    
    def test_extreme_emotional_state_detection(self):
        """Test detection of extreme emotional states"""
        workspace = GlobalWorkspace()
        affect = AffectSubsystem()
        workspace.affect = affect
        
        # Build up emotion history
        for _ in range(10):
            affect.emotion_history.append(EmotionalState(
                valence=-0.7,
                arousal=0.9,
                dominance=0.2
            ))
        
        monitor = SelfMonitor(workspace=workspace)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": -0.7, "arousal": 0.9, "dominance": 0.2},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        percept = monitor._observe_emotions(snapshot)
        assert percept is not None
        assert "emotional_observation" in percept.raw["type"]


class TestPatternDetection:
    """Test pattern detection"""
    
    def test_no_patterns_without_action_subsystem(self):
        """Test no patterns detected without action subsystem"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        percept = monitor._detect_patterns(snapshot)
        assert percept is None
    
    def test_repetitive_action_detection(self):
        """Test detection of repetitive actions"""
        from mind.cognitive_core.action import ActionSubsystem
        
        workspace = GlobalWorkspace()
        action_subsystem = ActionSubsystem()
        workspace.action_subsystem = action_subsystem
        
        # Fill action history with repetitive actions
        for _ in range(15):
            action_subsystem.action_history.append(Action(
                type=ActionType.SPEAK,
                reason="test"
            ))
        
        monitor = SelfMonitor(workspace=workspace)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        percept = monitor._detect_patterns(snapshot)
        assert percept is not None
        assert "pattern_detected" in percept.raw["type"]


class TestMonitoringFrequency:
    """Test monitoring frequency control"""
    
    def test_frequency_control(self):
        """Test that introspections only generated at specified frequency"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace, config={"monitoring_frequency": 5})
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        # First 4 calls should return empty
        for i in range(4):
            percepts = monitor.observe(snapshot)
            assert len(percepts) == 0
            assert monitor.cycle_count == i + 1
        
        # 5th call should potentially generate percepts
        percepts = monitor.observe(snapshot)
        assert monitor.cycle_count == 5
        # May or may not generate percepts depending on state
    
    def test_observe_increments_cycle_count(self):
        """Test that observe increments cycle count"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        assert monitor.cycle_count == 0
        monitor.observe(snapshot)
        assert monitor.cycle_count == 1
        monitor.observe(snapshot)
        assert monitor.cycle_count == 2


class TestGetStats:
    """Test statistics reporting"""
    
    def test_get_stats_returns_dict(self):
        """Test that get_stats returns proper dictionary"""
        monitor = SelfMonitor()
        stats = monitor.get_stats()
        
        assert isinstance(stats, dict)
        assert "total_observations" in stats
        assert "monitoring_frequency" in stats
        assert "cycle_count" in stats
    
    def test_stats_tracking(self):
        """Test that stats are tracked correctly"""
        workspace = GlobalWorkspace()
        affect = AffectSubsystem()
        workspace.affect = affect
        
        # Build up emotion history for observation
        for _ in range(10):
            affect.emotion_history.append(EmotionalState(
                valence=-0.7,
                arousal=0.9,
                dominance=0.2
            ))
        
        monitor = SelfMonitor(workspace=workspace, config={"monitoring_frequency": 1})
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": -0.7, "arousal": 0.9, "dominance": 0.2},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )
        
        # Generate some observations
        percepts = monitor.observe(snapshot)
        
        stats = monitor.get_stats()
        assert stats["total_observations"] == len(percepts)
        if len(percepts) > 0:
            assert stats["emotional_observations"] >= 1
