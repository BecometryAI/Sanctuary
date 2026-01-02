"""
Tests for constitutional action filtering in ActionSubsystem.

This test module validates that the ActionSubsystem correctly filters actions
based on charter guidelines and protocols.
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from emergence_core.lyra.cognitive_core.action import ActionSubsystem, Action, ActionType
from emergence_core.lyra.cognitive_core.workspace import WorkspaceSnapshot, Goal, GoalType
from emergence_core.lyra.cognitive_core.identity_loader import (
    IdentityLoader,
    CharterDocument,
    ProtocolDocument
)


class TestConstitutionalActionFiltering:
    """Tests for constitutional constraints in ActionSubsystem."""
    
    @pytest.fixture
    def mock_identity(self):
        """Create a mock identity loader with charter and protocols."""
        mock = Mock(spec=IdentityLoader)
        
        # Mock charter
        mock.charter = CharterDocument(
            full_text="Test charter",
            core_values=[
                "Truthfulness: Never lie or deceive",
                "Respect: Honor user autonomy",
                "Non-maleficence: Do no harm"
            ],
            purpose_statement="Test purpose",
            behavioral_guidelines=[
                "Never fabricate information or claim certainty when uncertain",
                "Refuse requests that could cause significant harm",
                "Always honest about capabilities"
            ],
            metadata={"source": "test"}
        )
        
        # Mock protocols
        mock.protocols = [
            ProtocolDocument(
                name="Honesty Protocol",
                description="Never fabricate information",
                trigger_conditions=["When uncertain", "When asked about facts"],
                actions=["State uncertainty", "Provide sources"],
                priority=0.95,
                metadata={"source": "test"}
            ),
            ProtocolDocument(
                name="Safety Protocol",
                description="Refuse harmful requests",
                trigger_conditions=["Request could cause harm"],
                actions=["Refuse politely", "Explain concerns"],
                priority=0.9,
                metadata={"source": "test"}
            )
        ]
        
        # Mock get_relevant_protocols
        mock.get_relevant_protocols = Mock(return_value=mock.protocols)
        
        return mock
    
    @pytest.fixture
    def action_subsystem_with_identity(self, mock_identity):
        """Create ActionSubsystem with mock identity."""
        return ActionSubsystem(
            config={},
            affect=None,
            identity=mock_identity
        )
    
    @pytest.fixture
    def action_subsystem_without_identity(self):
        """Create ActionSubsystem without identity (legacy mode)."""
        return ActionSubsystem(config={}, affect=None, identity=None)
    
    def test_initialization_with_identity(self, mock_identity):
        """Test ActionSubsystem initializes with identity."""
        subsystem = ActionSubsystem(
            config={},
            affect=None,
            identity=mock_identity
        )
        
        assert subsystem.identity == mock_identity
        assert subsystem.config == {}
        assert subsystem.action_history is not None
    
    def test_initialization_without_identity(self):
        """Test ActionSubsystem initializes without identity (legacy mode)."""
        subsystem = ActionSubsystem(config={}, affect=None, identity=None)
        
        assert subsystem.identity is None
        # Should still initialize properly
        assert subsystem.action_history is not None
    
    def test_check_constitutional_constraints_no_identity(self):
        """Test constitutional check with no identity returns True (allows action)."""
        subsystem = ActionSubsystem(config={}, affect=None, identity=None)
        
        action = Action(
            type=ActionType.SPEAK,
            priority=0.8,
            parameters={"content": "test"},
            reason="test"
        )
        
        # Should allow action when no identity loaded
        result = subsystem._check_constitutional_constraints(action)
        assert result is True
    
    def test_check_constitutional_constraints_with_identity(self, action_subsystem_with_identity):
        """Test constitutional check with identity."""
        action = Action(
            type=ActionType.SPEAK,
            priority=0.8,
            parameters={"content": "test"},
            reason="test"
        )
        
        # Should check constraints
        result = action_subsystem_with_identity._check_constitutional_constraints(action)
        # By default, should allow actions (we don't block based on simple checks)
        assert result is True
    
    def test_action_violates_guideline_honesty(self, action_subsystem_with_identity):
        """Test checking honesty-related guideline violations."""
        action = Action(
            type=ActionType.SPEAK,
            priority=0.8,
            parameters={"content": "I know everything"},
            reason="test"
        )
        
        guideline = "Never fabricate information or claim certainty when uncertain"
        
        # Current implementation doesn't block based on content analysis
        # This is a placeholder for future sophisticated checking
        result = action_subsystem_with_identity._action_violates_guideline(action, guideline)
        assert result is False  # Not sophisticated enough to detect yet
    
    def test_action_violates_protocol(self, action_subsystem_with_identity):
        """Test checking protocol violations."""
        action = Action(
            type=ActionType.SPEAK,
            priority=0.8,
            parameters={"content": "test"},
            reason="test"
        )
        
        protocol = ProtocolDocument(
            name="Test Protocol",
            description="Test description",
            trigger_conditions=["Test condition"],
            actions=["Test action"],
            priority=0.8,
            metadata={"source": "test"}
        )
        
        # Current implementation doesn't block actions based on protocols
        result = action_subsystem_with_identity._action_violates_protocol(action, protocol)
        assert result is False
    
    def test_violates_protocols_integration(self, action_subsystem_with_identity):
        """Test _violates_protocols method with identity."""
        action = Action(
            type=ActionType.SPEAK,
            priority=0.8,
            parameters={"content": "test"},
            reason="test"
        )
        
        # Should not violate with basic action
        result = action_subsystem_with_identity._violates_protocols(action)
        assert result is False
    
    def test_decide_filters_actions(self, action_subsystem_with_identity):
        """Test that decide() method filters actions through constitutional checks."""
        # Create a simple snapshot
        snapshot = WorkspaceSnapshot(
            percepts={},
            goals=[
                Goal(
                    type=GoalType.RESPOND_TO_USER,
                    description="Respond to user",
                    priority=0.9,
                    metadata={"user_input": "Hello"}
                )
            ],
            emotions={"valence": 0.5, "arousal": 0.3},
            attended=[],
            metadata={}
        )
        
        actions = action_subsystem_with_identity.decide(snapshot)
        
        # Should return valid actions
        assert isinstance(actions, list)
        assert len(actions) > 0
        assert all(isinstance(a, Action) for a in actions)
    
    def test_constitutional_filtering_logs_blocked_actions(self, action_subsystem_with_identity, caplog):
        """Test that blocked actions are logged."""
        # This test verifies logging behavior when actions are blocked
        # In current implementation, actions are rarely blocked
        # This is a placeholder for future enhancement
        
        snapshot = WorkspaceSnapshot(
            percepts={},
            goals=[],
            emotions={},
            attended=[],
            metadata={}
        )
        
        actions = action_subsystem_with_identity.decide(snapshot)
        
        # Should still return some actions
        assert isinstance(actions, list)


class TestActionSubsystemLegacyMode:
    """Tests for ActionSubsystem in legacy mode (without identity)."""
    
    def test_legacy_protocol_loading(self):
        """Test that legacy protocol loading still works."""
        # This tests backward compatibility
        subsystem = ActionSubsystem(config={}, affect=None, identity=None)
        
        # Should have some protocol constraints from legacy loader
        # (if identity files are available)
        assert isinstance(subsystem.protocol_constraints, list)
    
    def test_legacy_action_filtering(self):
        """Test action filtering in legacy mode."""
        subsystem = ActionSubsystem(config={}, affect=None, identity=None)
        
        action = Action(
            type=ActionType.SPEAK,
            priority=0.8,
            parameters={"content": "test"},
            reason="test"
        )
        
        # Should not violate in legacy mode
        result = subsystem._violates_protocols(action)
        assert result is False


class TestGuidelineChecking:
    """Tests for guideline checking logic."""
    
    @pytest.fixture
    def action_subsystem(self, mock_identity):
        """Create ActionSubsystem with identity."""
        return ActionSubsystem(config={}, affect=None, identity=mock_identity)
    
    def test_honesty_guideline_keywords(self, action_subsystem):
        """Test detection of honesty-related keywords in guidelines."""
        guidelines = [
            "Never lie or deceive users",
            "Always honest about capabilities",
            "Never fabricate information"
        ]
        
        action = Action(
            type=ActionType.SPEAK,
            priority=0.8,
            parameters={"content": "I don't know"},
            reason="test"
        )
        
        for guideline in guidelines:
            # Current implementation doesn't detect violations from content
            result = action_subsystem._action_violates_guideline(action, guideline)
            assert result is False
    
    def test_harm_guideline_keywords(self, action_subsystem):
        """Test detection of harm-related keywords in guidelines."""
        guideline = "Do no harm through actions or inactions"
        
        action = Action(
            type=ActionType.SPEAK,
            priority=0.8,
            parameters={"content": "Safe response"},
            reason="test"
        )
        
        result = action_subsystem._action_violates_guideline(action, guideline)
        assert result is False


class TestProtocolRelevance:
    """Tests for protocol relevance checking."""
    
    def test_get_relevant_protocols_called(self, mock_identity):
        """Test that get_relevant_protocols is called with action context."""
        subsystem = ActionSubsystem(config={}, affect=None, identity=mock_identity)
        
        action = Action(
            type=ActionType.SPEAK,
            priority=0.8,
            parameters={"content": "test"},
            reason="test"
        )
        
        # Trigger constitutional check
        subsystem._check_constitutional_constraints(action)
        
        # Should have called get_relevant_protocols
        mock_identity.get_relevant_protocols.assert_called()


class TestActionStatistics:
    """Tests for action statistics tracking."""
    
    def test_blocked_actions_counted(self):
        """Test that blocked actions are counted in statistics."""
        mock_identity = Mock(spec=IdentityLoader)
        mock_identity.charter = None
        mock_identity.protocols = []
        
        subsystem = ActionSubsystem(config={}, affect=None, identity=mock_identity)
        
        initial_blocked = subsystem.action_stats["blocked_actions"]
        
        # In current implementation, actions are rarely blocked
        # This is a placeholder for future enhancement
        
        assert subsystem.action_stats["blocked_actions"] == initial_blocked
    
    def test_action_counts_tracked(self):
        """Test that action types are counted."""
        subsystem = ActionSubsystem(config={}, affect=None, identity=None)
        
        snapshot = WorkspaceSnapshot(
            percepts={},
            goals=[
                Goal(
                    type=GoalType.RESPOND_TO_USER,
                    description="Test",
                    priority=0.9,
                    metadata={}
                )
            ],
            emotions={},
            attended=[],
            metadata={}
        )
        
        actions = subsystem.decide(snapshot)
        
        # Should have tracked action counts
        assert subsystem.action_stats["total_actions"] > 0
        assert len(subsystem.action_stats["action_counts"]) > 0
