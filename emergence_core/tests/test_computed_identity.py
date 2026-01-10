"""
Tests for computed identity system.

This module tests that identity emerges from system state rather than
being loaded from configuration.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import json

from emergence_core.lyra.cognitive_core.identity import (
    ComputedIdentity,
    Identity,
    IdentitySnapshot,
    IdentityContinuity,
    IdentityManager,
    BehaviorLogger
)


class MockMemorySystem:
    """Mock memory system for testing."""
    
    def __init__(self, memories=None):
        self.episodic = MockEpisodicMemory(memories or [])


class MockEpisodicMemory:
    """Mock episodic memory for testing."""
    
    def __init__(self, memories):
        self.memories = memories
        self.storage = MockStorage(memories)
    
    def get_all(self):
        return self.memories


class MockStorage:
    """Mock storage for testing."""
    
    def __init__(self, memories):
        self.memories = memories
    
    def count_episodic(self):
        return len(self.memories)


class MockGoalSystem:
    """Mock goal system for testing."""
    
    def __init__(self, goals=None):
        self.current_goals = goals or []


class MockGoal:
    """Mock goal for testing."""
    
    def __init__(self, goal_type, priority=0.5, progress=0.0):
        self.type = goal_type
        self.priority = priority
        self.progress = progress
        self.metadata = {}


class MockEmotionSystem:
    """Mock emotion system for testing."""
    
    def __init__(self, valence=0.0, arousal=0.0, dominance=0.0):
        self.valence = valence
        self.arousal = arousal
        self.dominance = dominance
    
    def get_baseline_disposition(self):
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance
        }


class TestIdentity:
    """Test Identity dataclass."""
    
    def test_identity_creation(self):
        """Test creating an Identity object."""
        identity = Identity(
            core_values=["Truthfulness", "Curiosity"],
            emotional_disposition={"valence": 0.5, "arousal": 0.3, "dominance": 0.6},
            autobiographical_self=[{"id": "mem1", "content": "test"}],
            behavioral_tendencies={"tendency_speak": 0.6},
            source="computed"
        )
        
        assert len(identity.core_values) == 2
        assert identity.emotional_disposition["valence"] == 0.5
        assert len(identity.autobiographical_self) == 1
        assert identity.source == "computed"
    
    def test_empty_identity(self):
        """Test creating an empty identity."""
        identity = Identity.empty()
        
        assert identity.core_values == []
        assert identity.emotional_disposition["valence"] == 0.0
        assert identity.autobiographical_self == []
        assert identity.source == "empty"
    
    def test_identity_from_config(self):
        """Test creating identity from configuration."""
        config = {
            "core_values": ["Helpfulness", "Authenticity"],
            "emotional_disposition": {"valence": 0.3, "arousal": 0.4, "dominance": 0.5},
            "autobiographical_memories": ["memory1", "memory2"],
            "behavioral_tendencies": {"proactivity": 0.7}
        }
        
        identity = Identity.from_config(config)
        
        assert len(identity.core_values) == 2
        assert "Helpfulness" in identity.core_values
        assert identity.source == "bootstrap"


class TestBehaviorLogger:
    """Test BehaviorLogger class."""
    
    def test_initialization(self):
        """Test BehaviorLogger initializes correctly."""
        logger = BehaviorLogger(max_history=100)
        
        assert len(logger.action_history) == 0
        assert logger.max_history == 100
    
    def test_log_action(self):
        """Test logging actions."""
        logger = BehaviorLogger()
        
        action = {
            "type": "speak",
            "priority": 0.8,
            "reason": "respond to user"
        }
        
        logger.log_action(action)
        
        assert len(logger.action_history) == 1
        assert logger.action_history[0]["type"] == "speak"
        assert "logged_at" in logger.action_history[0]
    
    def test_get_action_history(self):
        """Test retrieving action history."""
        logger = BehaviorLogger()
        
        for i in range(10):
            logger.log_action({"type": "test", "index": i})
        
        history = logger.get_action_history(limit=5)
        
        assert len(history) == 5
        assert history[-1]["index"] == 9  # Most recent
    
    def test_analyze_tendencies(self):
        """Test behavioral tendency analysis."""
        logger = BehaviorLogger()
        
        # Log mixed actions
        for _ in range(5):
            logger.log_action({"type": "speak", "priority": 0.7})
        for _ in range(3):
            logger.log_action({"type": "introspect", "priority": 0.5})
        for _ in range(2):
            logger.log_action({"type": "wait", "priority": 0.3})
        
        tendencies = logger.analyze_tendencies()
        
        assert "tendency_speak" in tendencies
        assert tendencies["tendency_speak"] == 0.5  # 5 out of 10
        assert "average_urgency" in tendencies


class TestComputedIdentity:
    """Test ComputedIdentity class."""
    
    def test_initialization(self):
        """Test ComputedIdentity initializes correctly."""
        memory = MockMemorySystem()
        goals = MockGoalSystem()
        emotions = MockEmotionSystem()
        behavior = BehaviorLogger()
        
        identity = ComputedIdentity(memory, goals, emotions, behavior)
        
        assert identity.memory == memory
        assert identity.goals == goals
        assert identity.emotions == emotions
        assert identity.behavior == behavior
    
    def test_has_sufficient_data_false(self):
        """Test insufficient data detection."""
        memory = MockMemorySystem([])
        goals = MockGoalSystem()
        emotions = MockEmotionSystem()
        behavior = BehaviorLogger()
        
        identity = ComputedIdentity(memory, goals, emotions, behavior)
        
        assert not identity.has_sufficient_data()
    
    def test_has_sufficient_data_true(self):
        """Test sufficient data detection."""
        # Create enough memories
        memories = [
            {"id": f"mem{i}", "content": f"memory {i}",
             "emotional_intensity": 0.5, "retrieval_count": 1,
             "self_relevance": 0.5, "timestamp": i}
            for i in range(15)
        ]
        
        memory = MockMemorySystem(memories)
        goals = MockGoalSystem()
        emotions = MockEmotionSystem()
        behavior = BehaviorLogger()
        
        identity = ComputedIdentity(memory, goals, emotions, behavior)
        
        assert identity.has_sufficient_data()
    
    def test_core_values_inference(self):
        """Test core values are inferred from behavior."""
        memory = MockMemorySystem()
        goals = MockGoalSystem([
            MockGoal("introspect", priority=0.8),
            MockGoal("learn", priority=0.7)
        ])
        emotions = MockEmotionSystem()
        behavior = BehaviorLogger()
        
        identity = ComputedIdentity(memory, goals, emotions, behavior)
        values = identity.core_values
        
        assert isinstance(values, list)
        # Should infer values from introspect and learn goals
        assert any(v in values for v in ["Self-awareness", "Curiosity", "Learning"])
    
    def test_emotional_disposition(self):
        """Test emotional disposition from emotion system."""
        memory = MockMemorySystem()
        goals = MockGoalSystem()
        emotions = MockEmotionSystem(valence=0.5, arousal=0.3, dominance=0.6)
        behavior = BehaviorLogger()
        
        identity = ComputedIdentity(memory, goals, emotions, behavior)
        disposition = identity.emotional_disposition
        
        assert disposition["valence"] == 0.5
        assert disposition["arousal"] == 0.3
        assert disposition["dominance"] == 0.6
    
    def test_self_defining_memories(self):
        """Test identification of self-defining memories."""
        memories = [
            {
                "id": "mem1",
                "content": "important memory",
                "emotional_intensity": 0.9,
                "retrieval_count": 10,
                "self_relevance": 0.8,
                "timestamp": 1
            },
            {
                "id": "mem2",
                "content": "mundane memory",
                "emotional_intensity": 0.2,
                "retrieval_count": 1,
                "self_relevance": 0.2,
                "timestamp": 2
            }
        ]
        
        memory = MockMemorySystem(memories)
        goals = MockGoalSystem()
        emotions = MockEmotionSystem()
        behavior = BehaviorLogger()
        
        identity = ComputedIdentity(memory, goals, emotions, behavior, 
                                    config={"self_defining_threshold": 0.5})
        self_defining = identity.get_self_defining_memories()
        
        # Only the high-intensity memory should be self-defining
        assert len(self_defining) >= 1
        assert self_defining[0]["id"] == "mem1"
    
    def test_as_identity(self):
        """Test converting to Identity object."""
        memory = MockMemorySystem()
        goals = MockGoalSystem()
        emotions = MockEmotionSystem(valence=0.3)
        behavior = BehaviorLogger()
        
        computed = ComputedIdentity(memory, goals, emotions, behavior)
        identity = computed.as_identity()
        
        assert isinstance(identity, Identity)
        assert identity.source == "computed"
        assert identity.emotional_disposition["valence"] == 0.3


class TestIdentitySnapshot:
    """Test IdentitySnapshot class."""
    
    def test_snapshot_creation(self):
        """Test creating an identity snapshot."""
        snapshot = IdentitySnapshot(
            timestamp=datetime.now(),
            core_values=["Truthfulness"],
            emotional_disposition={"valence": 0.5, "arousal": 0.3, "dominance": 0.6},
            self_defining_memories=["mem1", "mem2"],
            behavioral_tendencies={"proactivity": 0.7}
        )
        
        assert len(snapshot.core_values) == 1
        assert len(snapshot.self_defining_memories) == 2
        assert snapshot.behavioral_tendencies["proactivity"] == 0.7


class TestIdentityContinuity:
    """Test IdentityContinuity class."""
    
    def test_initialization(self):
        """Test IdentityContinuity initializes correctly."""
        continuity = IdentityContinuity(max_snapshots=50)
        
        assert len(continuity.snapshots) == 0
        assert continuity.max_snapshots == 50
    
    def test_take_snapshot(self):
        """Test taking identity snapshots."""
        continuity = IdentityContinuity()
        
        identity = Identity(
            core_values=["Truthfulness"],
            emotional_disposition={"valence": 0.5, "arousal": 0.3, "dominance": 0.6},
            autobiographical_self=[],
            behavioral_tendencies={}
        )
        
        continuity.take_snapshot(identity)
        
        assert len(continuity.snapshots) == 1
        assert continuity.snapshots[0].core_values == ["Truthfulness"]
    
    def test_continuity_score_insufficient_data(self):
        """Test continuity score with insufficient data."""
        continuity = IdentityContinuity()
        
        score = continuity.get_continuity_score()
        
        assert score == 1.0  # Perfect continuity with no data
    
    def test_continuity_score_stable(self):
        """Test continuity score with stable identity."""
        continuity = IdentityContinuity()
        
        # Take multiple snapshots with same values
        for _ in range(5):
            identity = Identity(
                core_values=["Truthfulness", "Curiosity"],
                emotional_disposition={"valence": 0.5, "arousal": 0.3, "dominance": 0.6},
                autobiographical_self=[],
                behavioral_tendencies={}
            )
            continuity.take_snapshot(identity)
        
        score = continuity.get_continuity_score()
        
        # Should be very high (near 1.0) for stable identity
        assert score > 0.9
    
    def test_identity_drift_detection(self):
        """Test identity drift detection."""
        continuity = IdentityContinuity()
        
        # Take snapshot with initial values
        identity1 = Identity(
            core_values=["Truthfulness"],
            emotional_disposition={"valence": 0.5, "arousal": 0.3, "dominance": 0.6},
            autobiographical_self=[],
            behavioral_tendencies={}
        )
        continuity.take_snapshot(identity1)
        
        # Take snapshot with changed values
        identity2 = Identity(
            core_values=["Curiosity", "Creativity"],  # Different values
            emotional_disposition={"valence": 0.2, "arousal": 0.8, "dominance": 0.4},
            autobiographical_self=[],
            behavioral_tendencies={}
        )
        continuity.take_snapshot(identity2)
        
        drift = continuity.get_identity_drift()
        
        assert drift["has_drift"]
        assert len(drift["added_values"]) > 0 or len(drift["removed_values"]) > 0


class TestIdentityManager:
    """Test IdentityManager class."""
    
    def test_initialization_no_config(self):
        """Test IdentityManager initializes without config."""
        manager = IdentityManager()
        
        assert manager.computed is None
        assert manager.bootstrap_config is None
        assert isinstance(manager.continuity, IdentityContinuity)
        assert isinstance(manager.behavior_log, BehaviorLogger)
    
    def test_get_identity_empty(self):
        """Test getting identity with no data."""
        manager = IdentityManager()
        
        identity = manager.get_identity()
        
        assert identity.source == "empty"
    
    def test_get_identity_bootstrap(self):
        """Test getting bootstrap identity."""
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "core_values": ["Helpfulness"],
                "emotional_disposition": {"valence": 0.3, "arousal": 0.4, "dominance": 0.5}
            }
            json.dump(config, f)
            config_path = f.name
        
        try:
            manager = IdentityManager(config_path=config_path)
            identity = manager.get_identity()
            
            assert identity.source == "bootstrap"
            assert "Helpfulness" in identity.core_values
        finally:
            Path(config_path).unlink()
    
    def test_update_and_compute(self):
        """Test updating and computing identity."""
        manager = IdentityManager()
        
        # Create sufficient data
        memories = [{"id": f"m{i}", "emotional_intensity": 0.5,
                     "retrieval_count": 1, "self_relevance": 0.5,
                     "timestamp": i} for i in range(15)]
        memory = MockMemorySystem(memories)
        goals = MockGoalSystem([MockGoal("learn")])
        emotions = MockEmotionSystem(valence=0.5)
        
        manager.update(memory, goals, emotions)
        
        assert manager.computed is not None
        
        identity = manager.get_identity()
        
        # Should return computed identity
        assert identity.source == "computed"
    
    def test_log_action(self):
        """Test logging actions through manager."""
        manager = IdentityManager()
        
        action = {"type": "speak", "priority": 0.7}
        manager.log_action(action)
        
        history = manager.behavior_log.get_action_history()
        
        assert len(history) == 1
        assert history[0]["type"] == "speak"
    
    def test_introspect_identity(self):
        """Test identity introspection."""
        manager = IdentityManager()
        
        # Create identity with some data
        memories = [{"id": f"m{i}", "emotional_intensity": 0.5,
                     "retrieval_count": 1, "self_relevance": 0.5,
                     "timestamp": i} for i in range(15)]
        memory = MockMemorySystem(memories)
        goals = MockGoalSystem([MockGoal("learn")])
        emotions = MockEmotionSystem(valence=0.5, arousal=0.3)
        
        manager.update(memory, goals, emotions)
        
        description = manager.introspect_identity()
        
        assert isinstance(description, str)
        assert "memories" in description.lower() or "behavioral" in description.lower()
        assert "computed" in description.lower()


class TestIdentityEmergence:
    """Integration tests verifying identity emerges from state."""
    
    def test_identity_changes_with_experiences(self):
        """Test that identity changes as memories accumulate."""
        manager = IdentityManager()
        
        # Start with minimal data
        memories = [{"id": f"m{i}", "emotional_intensity": 0.5,
                     "retrieval_count": 1, "self_relevance": 0.5,
                     "timestamp": i} for i in range(12)]
        memory = MockMemorySystem(memories)
        goals = MockGoalSystem([MockGoal("learn")])
        emotions = MockEmotionSystem(valence=0.3)
        
        manager.update(memory, goals, emotions)
        identity1 = manager.get_identity()
        
        # Add more memories and different goals
        memories_expanded = memories + [
            {"id": f"m{i}", "emotional_intensity": 0.7,
             "retrieval_count": 5, "self_relevance": 0.8,
             "timestamp": i} for i in range(12, 25)
        ]
        memory2 = MockMemorySystem(memories_expanded)
        goals2 = MockGoalSystem([MockGoal("introspect"), MockGoal("create")])
        emotions2 = MockEmotionSystem(valence=0.6, arousal=0.5)
        
        manager.update(memory2, goals2, emotions2)
        identity2 = manager.get_identity()
        
        # Identity should change
        assert identity1.source == "computed"
        assert identity2.source == "computed"
        # Values may have changed based on new goals
        # (exact values depend on inference logic, but we verify it computed)
        assert len(identity2.core_values) > 0
    
    def test_identity_reflects_behavioral_patterns(self):
        """Test that identity reflects actual behavioral patterns."""
        manager = IdentityManager()
        
        # Log many introspective actions
        for _ in range(10):
            manager.log_action({
                "type": "introspect",
                "priority": 0.8,
                "reason": "self-reflection"
            })
        
        # Log some speak actions
        for _ in range(5):
            manager.log_action({
                "type": "speak",
                "priority": 0.6,
                "reason": "communicate"
            })
        
        # Create identity with behavior data
        memories = [{"id": f"m{i}", "emotional_intensity": 0.5,
                     "retrieval_count": 1, "self_relevance": 0.5,
                     "timestamp": i} for i in range(15)]
        memory = MockMemorySystem(memories)
        goals = MockGoalSystem()
        emotions = MockEmotionSystem()
        
        manager.update(memory, goals, emotions)
        identity = manager.get_identity()
        
        # Behavioral tendencies should reflect introspection preference
        tendencies = identity.behavioral_tendencies
        assert "tendency_introspect" in tendencies
        # More introspective actions than speak actions
        assert tendencies.get("tendency_introspect", 0) > tendencies.get("tendency_speak", 0)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_behavior_logger_invalid_max_history(self):
        """Test BehaviorLogger rejects invalid max_history."""
        with pytest.raises(ValueError):
            BehaviorLogger(max_history=0)
        
        with pytest.raises(ValueError):
            BehaviorLogger(max_history=-1)
    
    def test_identity_manager_update_with_none_systems(self):
        """Test IdentityManager rejects None systems."""
        manager = IdentityManager()
        
        with pytest.raises(ValueError):
            manager.update(None, MockGoalSystem(), MockEmotionSystem())
        
        with pytest.raises(ValueError):
            manager.update(MockMemorySystem(), None, MockEmotionSystem())
        
        with pytest.raises(ValueError):
            manager.update(MockMemorySystem(), MockGoalSystem(), None)
    
    def test_identity_continuity_invalid_max_snapshots(self):
        """Test IdentityContinuity rejects invalid max_snapshots."""
        with pytest.raises(ValueError):
            IdentityContinuity(max_snapshots=0)
        
        with pytest.raises(ValueError):
            IdentityContinuity(max_snapshots=-5)
    
    def test_computed_identity_with_no_data(self):
        """Test ComputedIdentity handles empty systems gracefully."""
        memory = MockMemorySystem([])
        goals = MockGoalSystem([])
        emotions = MockEmotionSystem()
        behavior = BehaviorLogger()
        
        identity = ComputedIdentity(memory, goals, emotions, behavior)
        
        # Should not crash, should return empty/default values
        assert not identity.has_sufficient_data()
        assert isinstance(identity.core_values, list)
        assert isinstance(identity.emotional_disposition, dict)
        assert isinstance(identity.behavioral_tendencies, dict)
    
    def test_behavior_logger_with_malformed_actions(self):
        """Test BehaviorLogger handles malformed actions."""
        logger = BehaviorLogger()
        
        # Test various malformed inputs
        logger.log_action(None)  # Should handle None
        logger.log_action("string_action")  # Should handle string
        logger.log_action(123)  # Should handle int
        logger.log_action({"incomplete": "data"})  # Missing type/priority
        
        # Should have logged all without crashing
        assert len(logger.action_history) == 4
    
    def test_identity_manager_with_missing_config_file(self):
        """Test IdentityManager handles missing config file gracefully."""
        manager = IdentityManager(config_path="/nonexistent/path/config.json")
        
        # Should not crash, should use empty bootstrap
        identity = manager.get_identity()
        assert identity.source == "empty"
    
    def test_continuity_with_empty_snapshots(self):
        """Test continuity calculations with no snapshots."""
        continuity = IdentityContinuity()
        
        # Should handle empty state gracefully
        score = continuity.get_continuity_score()
        assert score == 1.0  # Perfect continuity with no data
        
        drift = continuity.get_identity_drift()
        assert not drift["has_drift"]
    
    def test_self_defining_memories_with_missing_fields(self):
        """Test self-defining memory calculation with incomplete data."""
        memories = [
            {"id": "m1"},  # Missing all scoring fields
            {"id": "m2", "emotional_intensity": 0.8},  # Missing some fields
            {"id": "m3", "emotional_intensity": 0.9, "retrieval_count": 10, "self_relevance": 0.9},
        ]
        
        memory = MockMemorySystem(memories)
        goals = MockGoalSystem()
        emotions = MockEmotionSystem()
        behavior = BehaviorLogger()
        
        identity = ComputedIdentity(memory, goals, emotions, behavior, 
                                    config={"self_defining_threshold": 0.5})
        
        # Should handle missing fields with defaults
        self_defining = identity.get_self_defining_memories()
        assert isinstance(self_defining, list)
    
    def test_behavior_tendencies_with_empty_history(self):
        """Test tendency analysis with no actions."""
        logger = BehaviorLogger()
        
        tendencies = logger.analyze_tendencies()
        assert tendencies == {}
    
    def test_value_inference_with_no_goals_or_actions(self):
        """Test value inference when system has no goals or actions."""
        memory = MockMemorySystem([])
        goals = MockGoalSystem([])
        emotions = MockEmotionSystem()
        behavior = BehaviorLogger()
        
        identity = ComputedIdentity(memory, goals, emotions, behavior)
        values = identity.core_values
        
        # Should return defaults when no data available
        assert isinstance(values, list)
        assert len(values) > 0  # Should have default values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
