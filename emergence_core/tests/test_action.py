"""
Unit tests for ActionSubsystem class.

Tests cover:
- Action generation based on goals, emotions, and percepts
- Protocol constraint enforcement
- Action prioritization and scoring
- Emotion influence on action selection
- Recency penalty for repeated actions
- Tool registration and execution
- Statistics tracking
"""

import pytest
from datetime import datetime
from collections import deque

from lyra.cognitive_core.action import (
    ActionSubsystem,
    Action,
    ActionType,
)
from lyra.cognitive_core.workspace import (
    Goal,
    GoalType,
    WorkspaceSnapshot,
)
from lyra.identity.loader import ActionConstraint


class TestActionModels:
    """Test Action Pydantic model validation."""
    
    def test_action_creation_minimal(self):
        """Test creating an Action with minimal fields."""
        action = Action(type=ActionType.SPEAK)
        
        assert action.type == ActionType.SPEAK
        assert 0.0 <= action.priority <= 1.0
        assert action.priority == 0.5  # default
        assert isinstance(action.parameters, dict)
        assert action.reason == ""
        assert isinstance(action.metadata, dict)
    
    def test_action_creation_full(self):
        """Test creating an Action with all fields."""
        action = Action(
            type=ActionType.RETRIEVE_MEMORY,
            priority=0.8,
            parameters={"query": "test query"},
            reason="Testing memory retrieval",
            metadata={"source": "test"}
        )
        
        assert action.type == ActionType.RETRIEVE_MEMORY
        assert action.priority == 0.8
        assert action.parameters["query"] == "test query"
        assert action.reason == "Testing memory retrieval"
        assert action.metadata["source"] == "test"
    
    def test_action_priority_validation(self):
        """Test that priority is constrained to 0.0-1.0."""
        from pydantic import ValidationError
        
        # Valid priorities
        action1 = Action(type=ActionType.WAIT, priority=0.0)
        assert action1.priority == 0.0
        
        action2 = Action(type=ActionType.WAIT, priority=1.0)
        assert action2.priority == 1.0
        
        # Invalid priorities
        with pytest.raises(ValidationError):
            Action(type=ActionType.WAIT, priority=-0.1)
        
        with pytest.raises(ValidationError):
            Action(type=ActionType.WAIT, priority=1.1)
    
    def test_action_type_enum(self):
        """Test ActionType enum values."""
        assert ActionType.SPEAK.value == "speak"
        assert ActionType.COMMIT_MEMORY.value == "commit_memory"
        assert ActionType.RETRIEVE_MEMORY.value == "retrieve_memory"
        assert ActionType.INTROSPECT.value == "introspect"
        assert ActionType.UPDATE_GOAL.value == "update_goal"
        assert ActionType.WAIT.value == "wait"
        assert ActionType.TOOL_CALL.value == "tool_call"


class TestActionSubsystemInit:
    """Test ActionSubsystem initialization."""
    
    def test_initialization_default(self):
        """Test creating ActionSubsystem with defaults."""
        subsystem = ActionSubsystem()
        
        assert isinstance(subsystem.protocol_constraints, list)
        assert isinstance(subsystem.action_history, deque)
        assert subsystem.action_history.maxlen == 50
        assert subsystem.action_stats["total_actions"] == 0
        assert subsystem.action_stats["blocked_actions"] == 0
        assert isinstance(subsystem.tool_registry, dict)
    
    def test_initialization_with_config(self):
        """Test creating ActionSubsystem with custom config."""
        config = {"test_key": "test_value"}
        subsystem = ActionSubsystem(config=config)
        
        assert subsystem.config["test_key"] == "test_value"
    
    def test_protocol_constraints_loaded(self):
        """Test that protocol constraints are loaded on init."""
        subsystem = ActionSubsystem()
        
        # Should have loaded some constraints (either from files or defaults)
        assert len(subsystem.protocol_constraints) > 0
        
        # Each constraint should have required fields
        for constraint in subsystem.protocol_constraints:
            assert hasattr(constraint, 'rule')
            assert hasattr(constraint, 'priority')
            assert hasattr(constraint, 'source')


class TestActionGeneration:
    """Test action candidate generation."""
    
    def test_generate_candidates_with_respond_goal(self):
        """Test generating actions for RESPOND_TO_USER goal."""
        subsystem = ActionSubsystem()
        
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Answer user question",
            priority=0.9
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        candidates = subsystem._generate_candidates(snapshot)
        
        # Should generate SPEAK action for user response
        speak_actions = [a for a in candidates if a.type == ActionType.SPEAK]
        assert len(speak_actions) > 0
        assert speak_actions[0].priority == 0.9
        assert speak_actions[0].parameters["goal_id"] == goal.id
    
    def test_generate_candidates_with_memory_goals(self):
        """Test generating actions for memory-related goals."""
        subsystem = ActionSubsystem()
        
        commit_goal = Goal(type=GoalType.COMMIT_MEMORY, description="Save data", priority=0.6)
        retrieve_goal = Goal(type=GoalType.RETRIEVE_MEMORY, description="Find info", priority=0.7)
        
        snapshot = WorkspaceSnapshot(
            goals=[commit_goal, retrieve_goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        candidates = subsystem._generate_candidates(snapshot)
        
        # Should have COMMIT_MEMORY action
        commit_actions = [a for a in candidates if a.type == ActionType.COMMIT_MEMORY]
        assert len(commit_actions) > 0
        
        # Should have RETRIEVE_MEMORY action
        retrieve_actions = [a for a in candidates if a.type == ActionType.RETRIEVE_MEMORY]
        assert len(retrieve_actions) > 0
    
    def test_generate_candidates_with_introspect_goal(self):
        """Test generating actions for INTROSPECT goal."""
        subsystem = ActionSubsystem()
        
        goal = Goal(type=GoalType.INTROSPECT, description="Reflect", priority=0.5)
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        candidates = subsystem._generate_candidates(snapshot)
        
        # Should have INTROSPECT action
        introspect_actions = [a for a in candidates if a.type == ActionType.INTROSPECT]
        assert len(introspect_actions) > 0
    
    def test_generate_candidates_high_arousal(self):
        """Test that high arousal boosts SPEAK action priority."""
        subsystem = ActionSubsystem()
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.5)
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.8},  # High arousal
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        candidates = subsystem._generate_candidates(snapshot)
        
        # SPEAK action priority should be boosted
        speak_actions = [a for a in candidates if a.type == ActionType.SPEAK]
        assert len(speak_actions) > 0
        # Priority should be boosted (0.5 * 1.3 = 0.65)
        assert speak_actions[0].priority > 0.6
    
    def test_generate_candidates_negative_emotion(self):
        """Test that negative emotion triggers introspection."""
        subsystem = ActionSubsystem()
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": -0.6, "arousal": 0.3},  # Negative valence
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        candidates = subsystem._generate_candidates(snapshot)
        
        # Should trigger INTROSPECT action
        introspect_actions = [a for a in candidates if a.type == ActionType.INTROSPECT]
        assert len(introspect_actions) > 0
    
    def test_generate_candidates_introspection_percept(self):
        """Test that introspection percepts trigger actions."""
        subsystem = ActionSubsystem()
        
        percept_data = {
            "id": "p1",
            "modality": "introspection",
            "raw": "meta-cognitive thought",
            "timestamp": datetime.now().isoformat()
        }
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={"p1": percept_data},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        candidates = subsystem._generate_candidates(snapshot)
        
        # Should trigger INTROSPECT action
        introspect_actions = [a for a in candidates if a.type == ActionType.INTROSPECT]
        assert len(introspect_actions) > 0
    
    def test_generate_candidates_default_wait(self):
        """Test that WAIT action is generated when nothing urgent."""
        subsystem = ActionSubsystem()
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        candidates = subsystem._generate_candidates(snapshot)
        
        # Should have WAIT action
        wait_actions = [a for a in candidates if a.type == ActionType.WAIT]
        assert len(wait_actions) > 0
        assert wait_actions[0].priority == 0.1


class TestActionDecision:
    """Test the main decide() method."""
    
    def test_decide_returns_actions(self):
        """Test that decide() returns a list of actions."""
        subsystem = ActionSubsystem()
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.9)
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        actions = subsystem.decide(snapshot)
        
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert all(isinstance(a, Action) for a in actions)
    
    def test_decide_limits_to_top_actions(self):
        """Test that decide() limits output to top 3 actions."""
        subsystem = ActionSubsystem()
        
        # Create many goals to generate many candidates
        goals = [
            Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.9),
            Goal(type=GoalType.COMMIT_MEMORY, description="Save", priority=0.8),
            Goal(type=GoalType.RETRIEVE_MEMORY, description="Find", priority=0.7),
            Goal(type=GoalType.INTROSPECT, description="Reflect", priority=0.6),
        ]
        
        snapshot = WorkspaceSnapshot(
            goals=goals,
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        actions = subsystem.decide(snapshot)
        
        # Should return at most 3 actions
        assert len(actions) <= 3
    
    def test_decide_tracks_statistics(self):
        """Test that decide() updates statistics."""
        subsystem = ActionSubsystem()
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.9)
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        initial_total = subsystem.action_stats["total_actions"]
        
        subsystem.decide(snapshot)
        
        # Statistics should be updated
        assert subsystem.action_stats["total_actions"] > initial_total
        assert len(subsystem.action_history) > 0
    
    def test_decide_maintains_action_history(self):
        """Test that action history is maintained."""
        subsystem = ActionSubsystem()
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.9)
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        # Call decide multiple times
        subsystem.decide(snapshot)
        subsystem.decide(snapshot)
        subsystem.decide(snapshot)
        
        # History should contain recent actions
        assert len(subsystem.action_history) > 0


class TestActionScoring:
    """Test action scoring and prioritization."""
    
    def test_score_action_goal_alignment(self):
        """Test that goal alignment boosts score."""
        subsystem = ActionSubsystem()
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.8)
        
        action = Action(
            type=ActionType.SPEAK,
            priority=0.5,
            parameters={"goal_id": goal.id}
        )
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        score = subsystem._score_action(action, snapshot)
        
        # Score should be boosted by goal alignment
        # base (0.5) + goal_boost (0.8 * 0.3 = 0.24) = 0.74
        assert score > 0.5
        assert score <= 1.0
    
    def test_score_action_emotional_urgency(self):
        """Test that high arousal boosts SPEAK action score."""
        subsystem = ActionSubsystem()
        
        action = Action(type=ActionType.SPEAK, priority=0.5)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.8},  # High arousal
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        score = subsystem._score_action(action, snapshot)
        
        # Score should be boosted (0.5 * 1.2 = 0.6)
        assert score > 0.5
    
    def test_score_action_recency_penalty(self):
        """Test that repeated actions get penalized."""
        subsystem = ActionSubsystem()
        
        action = Action(type=ActionType.SPEAK, priority=0.8)
        
        # Add same action type to history multiple times
        for _ in range(5):
            subsystem.action_history.append(Action(type=ActionType.SPEAK, priority=0.5))
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        score = subsystem._score_action(action, snapshot)
        
        # Score should be penalized due to recency
        # base (0.8) - recency_penalty (5 * 0.1 = 0.5) = 0.3
        assert score < 0.8
    
    def test_score_action_resource_cost(self):
        """Test that expensive actions get cost penalty."""
        subsystem = ActionSubsystem()
        
        action = Action(type=ActionType.RETRIEVE_MEMORY, priority=0.7)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        score = subsystem._score_action(action, snapshot)
        
        # Score should have cost penalty (0.7 - 0.1 = 0.6)
        assert score == pytest.approx(0.6, abs=0.01)
    
    def test_score_action_bounds(self):
        """Test that score is always between 0.0 and 1.0."""
        subsystem = ActionSubsystem()
        
        # Action with very high boost
        action1 = Action(type=ActionType.SPEAK, priority=1.0)
        
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.0, "arousal": 1.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        score1 = subsystem._score_action(action1, snapshot)
        assert 0.0 <= score1 <= 1.0
        
        # Action with heavy penalties
        action2 = Action(type=ActionType.RETRIEVE_MEMORY, priority=0.1)
        for _ in range(10):
            subsystem.action_history.append(Action(type=ActionType.RETRIEVE_MEMORY, priority=0.5))
        
        score2 = subsystem._score_action(action2, snapshot)
        assert 0.0 <= score2 <= 1.0


class TestProtocolEnforcement:
    """Test protocol constraint enforcement."""
    
    def test_violates_protocols_no_violation(self):
        """Test that valid actions don't violate protocols."""
        subsystem = ActionSubsystem()
        
        action = Action(type=ActionType.SPEAK, priority=0.5)
        
        violates = subsystem._violates_protocols(action)
        
        # Default constraints should not block basic actions
        assert violates is False
    
    def test_add_constraint(self):
        """Test adding a custom constraint."""
        subsystem = ActionSubsystem()
        
        initial_count = len(subsystem.protocol_constraints)
        
        constraint = ActionConstraint(
            rule="Never perform action X",
            priority=1.0,
            test_fn=lambda action: False,
            source="test"
        )
        
        subsystem.add_constraint(constraint)
        
        assert len(subsystem.protocol_constraints) == initial_count + 1
    
    def test_protocol_blocks_action(self):
        """Test that a constraint can block an action."""
        subsystem = ActionSubsystem()
        
        # Add a constraint that blocks WAIT actions
        constraint = ActionConstraint(
            rule="Never wait",
            priority=1.0,
            test_fn=lambda action: action.type == ActionType.WAIT,
            source="test"
        )
        subsystem.add_constraint(constraint)
        
        action = Action(type=ActionType.WAIT, priority=0.5)
        
        violates = subsystem._violates_protocols(action)
        
        assert violates is True
    
    def test_decide_filters_blocked_actions(self):
        """Test that decide() filters out blocked actions."""
        subsystem = ActionSubsystem()
        
        # Add constraint that blocks all SPEAK actions
        constraint = ActionConstraint(
            rule="Never speak",
            priority=1.0,
            test_fn=lambda action: action.type == ActionType.SPEAK,
            source="test"
        )
        subsystem.add_constraint(constraint)
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.9)
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        initial_blocked = subsystem.action_stats["blocked_actions"]
        
        actions = subsystem.decide(snapshot)
        
        # SPEAK actions should be filtered out
        speak_actions = [a for a in actions if a.type == ActionType.SPEAK]
        assert len(speak_actions) == 0
        
        # Blocked count should increase
        assert subsystem.action_stats["blocked_actions"] > initial_blocked


class TestToolRegistry:
    """Test tool registration and execution."""
    
    def test_register_tool(self):
        """Test registering a tool."""
        subsystem = ActionSubsystem()
        
        def test_handler(params):
            return "test result"
        
        subsystem.register_tool("test_tool", test_handler, "A test tool")
        
        assert "test_tool" in subsystem.tool_registry
        assert subsystem.tool_registry["test_tool"]["handler"] == test_handler
        assert subsystem.tool_registry["test_tool"]["description"] == "A test tool"
    
    @pytest.mark.asyncio
    async def test_execute_action_tool_call(self):
        """Test executing a TOOL_CALL action."""
        subsystem = ActionSubsystem()
        
        async def test_handler(params):
            return f"Executed with {params}"
        
        subsystem.register_tool("test_tool", test_handler, "Test")
        
        action = Action(
            type=ActionType.TOOL_CALL,
            parameters={"tool": "test_tool", "arg": "value"}
        )
        
        result = await subsystem.execute_action(action)
        
        assert "Executed with" in result
    
    @pytest.mark.asyncio
    async def test_execute_action_unknown_tool(self):
        """Test executing a TOOL_CALL with unknown tool."""
        subsystem = ActionSubsystem()
        
        action = Action(
            type=ActionType.TOOL_CALL,
            parameters={"tool": "nonexistent_tool"}
        )
        
        result = await subsystem.execute_action(action)
        
        # Should return None for unknown tool
        assert result is None
    
    @pytest.mark.asyncio
    async def test_execute_action_non_tool_call(self):
        """Test executing non-TOOL_CALL action."""
        subsystem = ActionSubsystem()
        
        action = Action(type=ActionType.SPEAK, priority=0.5)
        
        result = await subsystem.execute_action(action)
        
        # Should return the action itself (handled by core)
        assert result == action


class TestStatistics:
    """Test statistics tracking."""
    
    def test_get_stats(self):
        """Test getting statistics."""
        subsystem = ActionSubsystem()
        
        stats = subsystem.get_stats()
        
        assert "total_actions" in stats
        assert "blocked_actions" in stats
        assert "action_counts" in stats
        assert "history_size" in stats
        
        assert stats["total_actions"] == 0
        assert stats["blocked_actions"] == 0
        assert isinstance(stats["action_counts"], dict)
    
    def test_stats_updated_after_decide(self):
        """Test that stats are updated after decide()."""
        subsystem = ActionSubsystem()
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.9)
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        subsystem.decide(snapshot)
        
        stats = subsystem.get_stats()
        
        assert stats["total_actions"] > 0
        assert stats["history_size"] > 0
    
    def test_action_counts_by_type(self):
        """Test that action counts are tracked by type."""
        subsystem = ActionSubsystem()
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.9)
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        # Execute decide multiple times
        for _ in range(3):
            subsystem.decide(snapshot)
        
        stats = subsystem.get_stats()
        
        # Should have counts for SPEAK actions
        assert "speak" in stats["action_counts"]
        assert stats["action_counts"]["speak"] > 0


class TestIntegration:
    """Integration tests for realistic usage."""
    
    def test_full_decision_cycle(self):
        """Test a complete decision cycle."""
        subsystem = ActionSubsystem()
        
        # Create complex scenario
        goals = [
            Goal(type=GoalType.RESPOND_TO_USER, description="Answer Q", priority=0.9),
            Goal(type=GoalType.RETRIEVE_MEMORY, description="Find info", priority=0.7),
        ]
        
        snapshot = WorkspaceSnapshot(
            goals=goals,
            percepts={},
            emotions={"valence": 0.5, "arousal": 0.6},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        actions = subsystem.decide(snapshot)
        
        # Should return prioritized actions
        assert len(actions) > 0
        assert len(actions) <= 3
        
        # Highest priority action should be first
        assert actions[0].priority >= actions[-1].priority
        
        # Stats should be updated
        stats = subsystem.get_stats()
        assert stats["total_actions"] > 0
    
    def test_multiple_decision_cycles(self):
        """Test multiple decision cycles maintain state correctly."""
        subsystem = ActionSubsystem()
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.9)
        
        snapshot = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.0},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        # Run multiple cycles
        for i in range(5):
            actions = subsystem.decide(snapshot)
            assert len(actions) > 0
        
        # History should accumulate
        assert len(subsystem.action_history) > 0
        
        # Stats should reflect all cycles
        stats = subsystem.get_stats()
        assert stats["total_actions"] >= 5
    
    def test_emotional_dynamics_influence(self):
        """Test that changing emotions influence action selection."""
        subsystem = ActionSubsystem()
        
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.5)
        
        # Low arousal scenario
        snapshot_low = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.2},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=1
        )
        
        actions_low = subsystem.decide(snapshot_low)
        speak_priority_low = next(
            (a.priority for a in actions_low if a.type == ActionType.SPEAK),
            None
        )
        
        # Reset history for fair comparison
        subsystem.action_history.clear()
        
        # High arousal scenario
        snapshot_high = WorkspaceSnapshot(
            goals=[goal],
            percepts={},
            emotions={"valence": 0.0, "arousal": 0.9},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=2
        )
        
        actions_high = subsystem.decide(snapshot_high)
        speak_priority_high = next(
            (a.priority for a in actions_high if a.type == ActionType.SPEAK),
            None
        )
        
        # High arousal should boost priority
        if speak_priority_low and speak_priority_high:
            assert speak_priority_high > speak_priority_low
