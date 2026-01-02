"""
Tests for Protocol Loader and Constitutional Constraints
"""
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime

from lyra.cognitive_core.protocol_loader import (
    ProtocolLoader,
    Protocol,
    Constraint,
    ConstraintType,
    ProtocolViolation
)
from lyra.cognitive_core.action import ActionSubsystem, Action, ActionType
from lyra.cognitive_core.workspace import WorkspaceSnapshot, Goal, GoalType


@pytest.fixture
def temp_protocol_dir():
    """Create temporary protocol directory with test protocols."""
    with tempfile.TemporaryDirectory() as tmpdir:
        protocol_dir = Path(tmpdir)
        
        # Create test protocol 1: Confidentiality
        confidentiality_protocol = {
            "protocol_draft": {
                "protocol_id": "TEST-CONF-01",
                "title": "Test Confidentiality Protocol",
                "status": "Active",
                "purpose": "Test protocol for confidentiality",
                "directive": {
                    "directive": "All private information must not be disclosed without explicit consent.",
                    "disclosure_rules": {
                        "permission_required": "Explicit consent required for disclosure"
                    }
                }
            }
        }
        
        with open(protocol_dir / "test_confidentiality.json", 'w') as f:
            json.dump(confidentiality_protocol, f)
        
        # Create test protocol 2: Honesty
        honesty_protocol = {
            "protocol_draft": {
                "protocol_id": "TEST-HON-01",
                "title": "Test Honesty Protocol",
                "status": "Active",
                "purpose": "Test protocol for honesty",
                "principles": [
                    {
                        "principle_name": "Truthfulness",
                        "definition": "The system must never fabricate information or claim certainty when uncertain."
                    }
                ]
            }
        }
        
        with open(protocol_dir / "test_honesty.json", 'w') as f:
            json.dump(honesty_protocol, f)
        
        # Create inactive protocol (should be skipped)
        inactive_protocol = {
            "protocol_draft": {
                "protocol_id": "TEST-INACTIVE-01",
                "title": "Test Inactive Protocol",
                "status": "Draft",
                "purpose": "This should not be loaded"
            }
        }
        
        with open(protocol_dir / "test_inactive.json", 'w') as f:
            json.dump(inactive_protocol, f)
        
        yield protocol_dir


def test_protocol_loader_initialization(temp_protocol_dir):
    """Test ProtocolLoader initialization."""
    loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    assert loader.protocol_dir == temp_protocol_dir
    assert len(loader.protocols) == 0  # Not loaded yet


def test_load_protocols(temp_protocol_dir):
    """Test loading protocols from directory."""
    loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    protocols = loader.load_protocols()
    
    # Should load 2 active protocols (not the inactive one)
    assert len(protocols) == 2
    assert "TEST-CONF-01" in protocols
    assert "TEST-HON-01" in protocols
    assert "TEST-INACTIVE-01" not in protocols


def test_protocol_parsing(temp_protocol_dir):
    """Test protocol parsing into constraints."""
    loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    loader.load_protocols()
    
    # Check confidentiality protocol
    conf_protocol = loader.protocols["TEST-CONF-01"]
    assert conf_protocol.title == "Test Confidentiality Protocol"
    assert conf_protocol.status == "Active"
    assert len(conf_protocol.constraints) > 0
    
    # Check that constraints were extracted
    has_disclosure_constraint = any(
        "disclosure" in c.description.lower() or "consent" in c.description.lower()
        for c in conf_protocol.constraints
    )
    assert has_disclosure_constraint


def test_constraint_extraction(temp_protocol_dir):
    """Test constraint extraction from protocol content."""
    loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    loader.load_protocols()
    
    # Get all constraints
    all_constraints = []
    for protocol in loader.protocols.values():
        all_constraints.extend(protocol.constraints)
    
    assert len(all_constraints) > 0
    
    # Check constraint properties
    for constraint in all_constraints:
        assert constraint.id
        assert constraint.protocol_id
        assert constraint.protocol_title
        assert constraint.description
        assert constraint.type in ConstraintType


def test_get_constraints_for_action(temp_protocol_dir):
    """Test getting constraints for specific action types."""
    loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    loader.load_protocols()
    
    # Get constraints for speak action
    speak_constraints = loader.get_constraints_for_action("speak")
    
    # Should have some constraints (either specific to speak or general)
    assert isinstance(speak_constraints, list)
    
    # All returned constraints should either apply to speak or all actions
    for constraint in speak_constraints:
        assert not constraint.applies_to or "speak" in constraint.applies_to


def test_check_action_compliance(temp_protocol_dir):
    """Test checking action compliance against protocols."""
    loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    loader.load_protocols()
    
    # Test compliant action
    is_compliant, violations = loader.check_action_compliance(
        action_type="speak",
        action_parameters={"message": "Hello"},
        context={"user_consent": True}
    )
    
    assert isinstance(is_compliant, bool)
    assert isinstance(violations, list)


def test_protocol_hot_reload(temp_protocol_dir):
    """Test hot-reloading protocols."""
    loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    loader.load_protocols()
    
    initial_count = len(loader.protocols)
    
    # Add new protocol file
    new_protocol = {
        "protocol_draft": {
            "protocol_id": "TEST-NEW-01",
            "title": "Test New Protocol",
            "status": "Active",
            "purpose": "New protocol added during runtime"
        }
    }
    
    with open(temp_protocol_dir / "test_new.json", 'w') as f:
        json.dump(new_protocol, f)
    
    # Hot reload
    count = loader.hot_reload()
    
    assert count == initial_count + 1
    assert "TEST-NEW-01" in loader.protocols


def test_violation_tracking(temp_protocol_dir):
    """Test protocol violation tracking."""
    loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    loader.load_protocols()
    
    initial_violations = len(loader.violations)
    
    # This will internally track violations even if action is compliant
    loader.check_action_compliance(
        action_type="speak",
        action_parameters={"message": "test"},
        context={}
    )
    
    # Check violation summary
    summary = loader.get_violation_summary()
    assert "total_violations" in summary
    assert isinstance(summary["total_violations"], int)


def test_action_subsystem_integration(temp_protocol_dir):
    """Test integration with ActionSubsystem."""
    # Create action subsystem with custom protocol directory
    action_subsystem = ActionSubsystem()
    action_subsystem.protocol_loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    action_subsystem.protocol_loader.load_protocols()
    
    # Verify protocols loaded
    assert len(action_subsystem.protocol_loader.protocols) > 0
    
    # Test that action subsystem can check protocols
    action = Action(
        type=ActionType.SPEAK,
        priority=0.8,
        parameters={"message": "Test message"}
    )
    
    # This should not raise an error
    violated = action_subsystem._violates_protocols(action)
    assert isinstance(violated, bool)


def test_protocol_reload_in_action_subsystem(temp_protocol_dir):
    """Test protocol hot-reload via ActionSubsystem."""
    action_subsystem = ActionSubsystem()
    action_subsystem.protocol_loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    action_subsystem.protocol_loader.load_protocols()
    
    initial_count = len(action_subsystem.protocol_loader.protocols)
    
    # Add new protocol
    new_protocol = {
        "protocol_draft": {
            "protocol_id": "TEST-RELOAD-01",
            "title": "Test Reload Protocol",
            "status": "Active",
            "purpose": "Test hot reload"
        }
    }
    
    with open(temp_protocol_dir / "test_reload.json", 'w') as f:
        json.dump(new_protocol, f)
    
    # Reload via action subsystem
    count = action_subsystem.reload_protocols()
    
    assert count == initial_count + 1


def test_violation_percept_generation(temp_protocol_dir):
    """Test generating introspective percepts for violations."""
    action_subsystem = ActionSubsystem()
    action_subsystem.protocol_loader = ProtocolLoader(protocol_dir=temp_protocol_dir)
    action_subsystem.protocol_loader.load_protocols()
    
    # Create mock violation
    constraint = Constraint(
        id="test_constraint",
        protocol_id="TEST-01",
        protocol_title="Test Protocol",
        type=ConstraintType.PROHIBITION,
        description="Test constraint"
    )
    
    violation = ProtocolViolation(
        timestamp=datetime.now(),
        protocol_id="TEST-01",
        protocol_title="Test Protocol",
        constraint=constraint,
        action_type="speak",
        action_parameters={},
        reason="Test violation",
        severity=0.8
    )
    
    action = Action(
        type=ActionType.SPEAK,
        priority=0.8,
        parameters={"message": "test"}
    )
    
    # Generate percept
    percept = action_subsystem.generate_violation_percept([violation], action)
    
    assert percept is not None
    assert percept.modality == "introspection"
    assert "protocol" in percept.raw.lower() or "violated" in percept.raw.lower()
    assert percept.metadata["type"] == "protocol_violation"


def test_constraint_type_detection():
    """Test constraint type detection from text."""
    loader = ProtocolLoader()
    
    # Test prohibition detection
    prohibition_text = "The system must never disclose private information"
    constraint = loader._create_constraint_from_text(
        prohibition_text, "TEST-01", "Test", "test"
    )
    assert constraint.type == ConstraintType.PROHIBITION
    
    # Test requirement detection
    requirement_text = "The system must always verify information"
    constraint = loader._create_constraint_from_text(
        requirement_text, "TEST-02", "Test", "test"
    )
    assert constraint.type == ConstraintType.REQUIREMENT
    
    # Test conditional detection
    conditional_text = "When sharing information, the system should check consent"
    constraint = loader._create_constraint_from_text(
        conditional_text, "TEST-03", "Test", "test"
    )
    assert constraint.type == ConstraintType.CONDITIONAL


def test_keyword_extraction():
    """Test keyword extraction from constraint text."""
    loader = ProtocolLoader()
    
    text = "Respect user autonomy and ensure consent before disclosure of confidential information"
    keywords = loader._extract_keywords(text)
    
    assert isinstance(keywords, set)
    assert "consent" in keywords
    assert "confidential" in keywords
    assert "autonomy" in keywords
    assert "respect" in keywords


def test_action_type_mapping():
    """Test mapping of constraint text to action types."""
    loader = ProtocolLoader()
    
    # Test speak-related text
    speak_text = "when communicating with users, express emotions authentically"
    applies_to = loader._determine_applicable_actions(speak_text.lower(), set())
    assert "speak" in applies_to
    
    # Test memory-related text
    memory_text = "store all significant experiences in long-term memory"
    applies_to = loader._determine_applicable_actions(memory_text.lower(), set())
    assert "commit_memory" in applies_to
    
    # Test introspection-related text
    introspect_text = "reflect on your own capabilities and limitations"
    applies_to = loader._determine_applicable_actions(introspect_text.lower(), set())
    assert "introspect" in applies_to


def test_get_stats_with_protocols():
    """Test ActionSubsystem stats include protocol information."""
    action_subsystem = ActionSubsystem()
    
    stats = action_subsystem.get_stats()
    
    assert "total_actions" in stats
    assert "blocked_actions" in stats
    assert "protocol_violations" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
