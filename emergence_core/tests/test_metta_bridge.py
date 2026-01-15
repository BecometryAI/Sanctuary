"""
Unit tests for MeTTa/Atomspace bridge.

Tests cover:
- AtomspaceBridge initialization and feature flag
- Atom creation and conversion
- Workspace state conversion
- Query and execution (stub implementation)
"""

import pytest

from lyra.cognitive_core.metta import (
    AtomspaceBridge,
    COMMUNICATION_DECISION_RULES,
    PREDICTION_RULES
)
from lyra.cognitive_core.metta.atomspace_bridge import Atom


class TestAtom:
    """Test Atom stub class."""
    
    def test_atom_creation(self):
        """Test creating an atom."""
        atom = Atom("string", "hello")
        assert atom.atom_type == "string"
        assert atom.value == "hello"
        assert len(atom.children) == 0
    
    def test_atom_with_children(self):
        """Test atom with children."""
        child1 = Atom("number", 42)
        child2 = Atom("string", "test")
        parent = Atom("list", "List", [child1, child2])
        
        assert len(parent.children) == 2
        assert parent.children[0] == child1
    
    def test_atom_repr(self):
        """Test atom string representation."""
        atom = Atom("string", "hello")
        assert "hello" in str(atom)
        
        child = Atom("number", 42)
        parent = Atom("list", "List", [child])
        assert "List" in str(parent)


class TestAtomspaceBridge:
    """Test AtomspaceBridge class."""
    
    def test_initialization_default(self):
        """Test default initialization (MeTTa disabled)."""
        bridge = AtomspaceBridge()
        assert bridge.use_metta is False
        assert bridge.atomspace is None
    
    def test_initialization_with_flag(self):
        """Test initialization with use_metta flag."""
        bridge = AtomspaceBridge({"use_metta": True})
        # Should fall back to False since MeTTa not available
        assert bridge.use_metta is False
    
    def test_to_atoms_goals(self):
        """Test converting goals to atoms."""
        bridge = AtomspaceBridge()
        workspace_state = {
            "goals": [
                {"description": "Learn Python", "priority": 0.8},
                {"description": "Write code", "priority": 0.6}
            ]
        }
        
        atoms = bridge.to_atoms(workspace_state)
        # Should have at least 2 goal atoms
        goal_atoms = [a for a in atoms if a.atom_type == "goal"]
        assert len(goal_atoms) == 2
    
    def test_to_atoms_percepts(self):
        """Test converting percepts to atoms."""
        bridge = AtomspaceBridge()
        workspace_state = {
            "percepts": [
                {"content": "Hello world"}
            ]
        }
        
        atoms = bridge.to_atoms(workspace_state)
        percept_atoms = [a for a in atoms if a.atom_type == "percept"]
        assert len(percept_atoms) == 1
    
    def test_to_atoms_emotions(self):
        """Test converting emotional state to atoms."""
        bridge = AtomspaceBridge()
        workspace_state = {
            "emotional_state": {
                "valence": 0.5,
                "arousal": 0.7
            }
        }
        
        atoms = bridge.to_atoms(workspace_state)
        emotion_atoms = [a for a in atoms if a.atom_type == "emotion"]
        assert len(emotion_atoms) == 1
    
    def test_from_atoms(self):
        """Test converting atoms back to Python structures."""
        bridge = AtomspaceBridge()
        
        atoms = [
            Atom("goal", "Goal", [
                Atom("string", "Test goal"),
                Atom("float", 0.7)
            ]),
            Atom("percept", "Percept", [
                Atom("string", "Test percept")
            ])
        ]
        
        result = bridge.from_atoms(atoms)
        assert "goals" in result
        assert "percepts" in result
        assert len(result["goals"]) == 1
        assert len(result["percepts"]) == 1
    
    def test_query_stub(self):
        """Test query returns empty list (stub)."""
        bridge = AtomspaceBridge()
        results = bridge.query("(match ?x (goal ?x))")
        assert results == []
    
    def test_execute_stub(self):
        """Test execute returns None (stub)."""
        bridge = AtomspaceBridge()
        result = bridge.execute("(+ 1 2)")
        assert result is None
    
    def test_is_available(self):
        """Test checking if MeTTa is available."""
        bridge = AtomspaceBridge()
        assert bridge.is_available() is False


class TestMeTTaRules:
    """Test MeTTa rule definitions."""
    
    def test_communication_rules_defined(self):
        """Test that communication rules are defined."""
        assert isinstance(COMMUNICATION_DECISION_RULES, str)
        assert len(COMMUNICATION_DECISION_RULES) > 0
        assert "should-speak" in COMMUNICATION_DECISION_RULES
    
    def test_prediction_rules_defined(self):
        """Test that prediction rules are defined."""
        assert isinstance(PREDICTION_RULES, str)
        assert len(PREDICTION_RULES) > 0
        assert "predict-self" in PREDICTION_RULES
