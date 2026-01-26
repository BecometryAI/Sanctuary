"""
Unit tests for GlobalWorkspace and related Pydantic models.

Tests cover:
- Model validation and constraints
- Workspace initialization
- Broadcast immutability
- Update integration
- Goal management (add/remove/duplicate checking)
- Serialization/deserialization
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from mind.cognitive_core.workspace import (
    GlobalWorkspace,
    Goal,
    GoalType,
    Percept,
    Memory,
    WorkspaceSnapshot,
)


class TestGoalModel:
    """Test Goal Pydantic model validation and behavior."""
    
    def test_goal_creation_with_defaults(self):
        """Test creating a Goal with only required fields."""
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer user query")
        
        assert goal.id is not None
        assert goal.type == GoalType.RESPOND_TO_USER
        assert goal.description == "Answer user query"
        assert 0.0 <= goal.priority <= 1.0
        assert goal.priority == 0.5  # default
        assert goal.progress == 0.0  # default
        assert isinstance(goal.created_at, datetime)
        assert isinstance(goal.metadata, dict)
    
    def test_goal_creation_with_custom_values(self):
        """Test creating a Goal with custom values."""
        goal = Goal(
            type=GoalType.COMMIT_MEMORY,
            description="Store important information",
            priority=0.9,
            progress=0.3,
            metadata={"source": "user"}
        )
        
        assert goal.type == GoalType.COMMIT_MEMORY
        assert goal.priority == 0.9
        assert goal.progress == 0.3
        assert goal.metadata["source"] == "user"
    
    def test_goal_priority_validation(self):
        """Test that priority is constrained to 0.0-1.0."""
        # Valid priorities
        goal1 = Goal(type=GoalType.LEARN, description="Test", priority=0.0)
        assert goal1.priority == 0.0
        
        goal2 = Goal(type=GoalType.LEARN, description="Test", priority=1.0)
        assert goal2.priority == 1.0
        
        # Invalid priorities
        with pytest.raises(ValidationError):
            Goal(type=GoalType.LEARN, description="Test", priority=-0.1)
        
        with pytest.raises(ValidationError):
            Goal(type=GoalType.LEARN, description="Test", priority=1.1)
    
    def test_goal_progress_validation(self):
        """Test that progress is constrained to 0.0-1.0."""
        with pytest.raises(ValidationError):
            Goal(type=GoalType.CREATE, description="Test", progress=-0.1)
        
        with pytest.raises(ValidationError):
            Goal(type=GoalType.CREATE, description="Test", progress=1.5)
    
    def test_goal_type_enum(self):
        """Test that GoalType enum works correctly."""
        assert GoalType.RESPOND_TO_USER.value == "respond_to_user"
        assert GoalType.COMMIT_MEMORY.value == "commit_memory"
        assert GoalType.RETRIEVE_MEMORY.value == "retrieve_memory"
        assert GoalType.INTROSPECT.value == "introspect"
        assert GoalType.LEARN.value == "learn"
        assert GoalType.CREATE.value == "create"


class TestPerceptModel:
    """Test Percept Pydantic model validation and behavior."""
    
    def test_percept_creation_minimal(self):
        """Test creating a Percept with minimal required fields."""
        percept = Percept(modality="text", raw="Hello, world!")
        
        assert percept.id is not None
        assert percept.modality == "text"
        assert percept.raw == "Hello, world!"
        assert percept.embedding is None
        assert percept.complexity == 1
        assert isinstance(percept.timestamp, datetime)
    
    def test_percept_with_embedding(self):
        """Test creating a Percept with embedding vector."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        percept = Percept(
            modality="image",
            raw={"path": "/tmp/image.png"},
            embedding=embedding,
            complexity=3
        )
        
        assert percept.modality == "image"
        assert percept.embedding == embedding
        assert percept.complexity == 3
        assert isinstance(percept.raw, dict)
    
    def test_percept_modalities(self):
        """Test different modality types."""
        modalities = ["text", "image", "audio", "introspection"]
        
        for modality in modalities:
            percept = Percept(modality=modality, raw=f"data for {modality}")
            assert percept.modality == modality


class TestMemoryModel:
    """Test Memory Pydantic model validation and behavior."""
    
    def test_memory_creation(self):
        """Test creating a Memory with all required fields."""
        memory = Memory(
            id="mem-123",
            content="Important conversation about AI",
            timestamp=datetime.now(),
            significance=0.8
        )
        
        assert memory.id == "mem-123"
        assert memory.content == "Important conversation about AI"
        assert memory.significance == 0.8
        assert isinstance(memory.timestamp, datetime)
        assert memory.tags == []
    
    def test_memory_with_tags_and_embedding(self):
        """Test Memory with tags and embedding."""
        memory = Memory(
            id="mem-456",
            content="Technical discussion",
            timestamp=datetime.now(),
            significance=0.6,
            tags=["technical", "programming", "AI"],
            embedding=[0.5, 0.6, 0.7]
        )
        
        assert len(memory.tags) == 3
        assert "AI" in memory.tags
        assert memory.embedding == [0.5, 0.6, 0.7]
    
    def test_memory_significance_validation(self):
        """Test that significance is constrained to 0.0-1.0."""
        with pytest.raises(ValidationError):
            Memory(
                id="test",
                content="test",
                timestamp=datetime.now(),
                significance=-0.1
            )
        
        with pytest.raises(ValidationError):
            Memory(
                id="test",
                content="test",
                timestamp=datetime.now(),
                significance=1.5
            )


class TestWorkspaceSnapshot:
    """Test WorkspaceSnapshot immutability."""
    
    def test_snapshot_creation(self):
        """Test creating a WorkspaceSnapshot."""
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10
        )
        
        assert snapshot.cycle_count == 10
        assert snapshot.emotions["valence"] == 0.5
    
    def test_snapshot_immutability(self):
        """Test that WorkspaceSnapshot is immutable."""
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=10
        )
        
        # Attempting to modify should raise an error
        with pytest.raises(ValidationError):
            snapshot.cycle_count = 20


class TestGlobalWorkspace:
    """Test GlobalWorkspace class functionality."""
    
    def test_workspace_initialization(self):
        """Test creating a GlobalWorkspace with default parameters."""
        workspace = GlobalWorkspace()
        
        assert workspace.capacity == 7
        assert len(workspace.current_goals) == 0
        assert len(workspace.active_percepts) == 0
        assert len(workspace.attended_memories) == 0
        assert workspace.cycle_count == 0
        assert isinstance(workspace.timestamp, datetime)
        assert "valence" in workspace.emotional_state
        assert "arousal" in workspace.emotional_state
        assert "dominance" in workspace.emotional_state
    
    def test_workspace_initialization_custom_capacity(self):
        """Test creating a GlobalWorkspace with custom capacity."""
        workspace = GlobalWorkspace(capacity=10, persistence_dir="/tmp/test")
        
        assert workspace.capacity == 10
        assert workspace.persistence_dir == "/tmp/test"
    
    def test_broadcast_returns_snapshot(self):
        """Test that broadcast() returns a WorkspaceSnapshot."""
        workspace = GlobalWorkspace()
        snapshot = workspace.broadcast()
        
        assert isinstance(snapshot, WorkspaceSnapshot)
        assert snapshot.cycle_count == 0
        assert snapshot.goals == []
    
    def test_broadcast_immutability(self):
        """Test that broadcast snapshot cannot be modified."""
        workspace = GlobalWorkspace()
        
        # Add a goal
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Test goal")
        workspace.add_goal(goal)
        
        # Get snapshot
        snapshot = workspace.broadcast()
        
        # Snapshot should be frozen
        with pytest.raises(ValidationError):
            snapshot.cycle_count = 999
    
    def test_broadcast_reflects_current_state(self):
        """Test that broadcast returns current workspace state."""
        workspace = GlobalWorkspace()
        
        # Add goal
        goal = Goal(type=GoalType.LEARN, description="Study", priority=0.8)
        workspace.add_goal(goal)
        
        # Add percept
        percept = Percept(modality="text", raw="Input data")
        workspace.active_percepts[percept.id] = percept
        
        # Update emotions
        workspace.emotional_state["valence"] = 0.7
        
        # Get snapshot
        snapshot = workspace.broadcast()
        
        assert len(snapshot.goals) == 1
        assert snapshot.goals[0].description == "Study"
        assert len(snapshot.percepts) == 1
        assert snapshot.emotions["valence"] == 0.7
    
    def test_add_goal(self):
        """Test adding a goal to the workspace."""
        workspace = GlobalWorkspace()
        
        goal = Goal(type=GoalType.COMMIT_MEMORY, description="Save data")
        workspace.add_goal(goal)
        
        assert len(workspace.current_goals) == 1
        assert workspace.current_goals[0].description == "Save data"
    
    def test_add_goal_duplicate_prevention(self):
        """Test that duplicate goals are not added."""
        workspace = GlobalWorkspace()
        
        goal = Goal(type=GoalType.CREATE, description="Generate art")
        workspace.add_goal(goal)
        workspace.add_goal(goal)  # Try to add same goal again
        
        assert len(workspace.current_goals) == 1
    
    def test_add_goal_priority_ordering(self):
        """Test that goals are ordered by priority."""
        workspace = GlobalWorkspace()
        
        goal1 = Goal(type=GoalType.LEARN, description="Low priority", priority=0.3)
        goal2 = Goal(type=GoalType.RESPOND_TO_USER, description="High priority", priority=0.9)
        goal3 = Goal(type=GoalType.INTROSPECT, description="Medium priority", priority=0.6)
        
        workspace.add_goal(goal1)
        workspace.add_goal(goal2)
        workspace.add_goal(goal3)
        
        # Should be ordered by priority (highest first)
        assert workspace.current_goals[0].priority == 0.9
        assert workspace.current_goals[1].priority == 0.6
        assert workspace.current_goals[2].priority == 0.3
    
    def test_remove_goal(self):
        """Test removing a goal from the workspace."""
        workspace = GlobalWorkspace()
        
        goal = Goal(type=GoalType.RETRIEVE_MEMORY, description="Fetch data")
        workspace.add_goal(goal)
        assert len(workspace.current_goals) == 1
        
        workspace.remove_goal(goal.id)
        assert len(workspace.current_goals) == 0
    
    def test_remove_goal_nonexistent(self):
        """Test removing a goal that doesn't exist."""
        workspace = GlobalWorkspace()
        
        # Should not raise an error
        workspace.remove_goal("nonexistent-id")
        assert len(workspace.current_goals) == 0
    
    def test_update_with_goal(self):
        """Test update() with goal output."""
        workspace = GlobalWorkspace()
        initial_cycle = workspace.cycle_count
        
        goal = Goal(type=GoalType.LEARN, description="New goal")
        outputs = [{"type": "goal", "data": goal}]
        
        workspace.update(outputs)
        
        assert len(workspace.current_goals) == 1
        assert workspace.cycle_count == initial_cycle + 1
    
    def test_update_with_percept(self):
        """Test update() with percept output."""
        workspace = GlobalWorkspace()
        
        percept = Percept(modality="audio", raw="sound data")
        outputs = [{"type": "percept", "data": percept}]
        
        workspace.update(outputs)
        
        assert len(workspace.active_percepts) == 1
        assert percept.id in workspace.active_percepts
    
    def test_update_with_emotion(self):
        """Test update() with emotion output."""
        workspace = GlobalWorkspace()
        
        outputs = [{"type": "emotion", "data": {"valence": 0.8, "arousal": 0.6}}]
        workspace.update(outputs)
        
        assert workspace.emotional_state["valence"] == 0.8
        assert workspace.emotional_state["arousal"] == 0.6
    
    def test_update_with_memory(self):
        """Test update() with memory output."""
        workspace = GlobalWorkspace()
        
        memory = Memory(
            id="mem-789",
            content="Retrieved memory",
            timestamp=datetime.now(),
            significance=0.7
        )
        outputs = [{"type": "memory", "data": memory}]
        
        workspace.update(outputs)
        
        assert len(workspace.attended_memories) == 1
        assert workspace.attended_memories[0].content == "Retrieved memory"
    
    def test_update_increments_cycle_count(self):
        """Test that update() increments cycle_count."""
        workspace = GlobalWorkspace()
        
        initial_count = workspace.cycle_count
        workspace.update([])
        assert workspace.cycle_count == initial_count + 1
        
        workspace.update([])
        assert workspace.cycle_count == initial_count + 2
    
    def test_update_updates_timestamp(self):
        """Test that update() updates timestamp."""
        workspace = GlobalWorkspace()
        
        initial_time = workspace.timestamp
        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.01)
        
        workspace.update([])
        
        assert workspace.timestamp > initial_time
    
    def test_clear(self):
        """Test clear() resets workspace to initial state."""
        workspace = GlobalWorkspace()
        
        # Add some state
        goal = Goal(type=GoalType.CREATE, description="Test")
        workspace.add_goal(goal)
        workspace.emotional_state["valence"] = 0.9
        workspace.cycle_count = 10
        
        # Clear
        workspace.clear()
        
        # Verify reset
        assert len(workspace.current_goals) == 0
        assert len(workspace.active_percepts) == 0
        assert len(workspace.attended_memories) == 0
        assert workspace.emotional_state["valence"] == 0.0
        assert workspace.cycle_count == 0
    
    def test_to_dict(self):
        """Test serialization to dict."""
        workspace = GlobalWorkspace(capacity=10)
        
        # Add some state
        goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer query")
        workspace.add_goal(goal)
        
        percept = Percept(modality="text", raw="Input")
        workspace.active_percepts[percept.id] = percept
        
        # Serialize
        data = workspace.to_dict()
        
        # Verify structure
        assert "current_goals" in data
        assert "active_percepts" in data
        assert "emotional_state" in data
        assert "attended_memories" in data
        assert "timestamp" in data
        assert "cycle_count" in data
        assert "capacity" in data
        
        assert data["capacity"] == 10
        assert len(data["current_goals"]) == 1
        assert len(data["active_percepts"]) == 1
    
    def test_to_dict_json_compatible(self):
        """Test that to_dict() output is JSON-serializable."""
        import json
        
        workspace = GlobalWorkspace()
        goal = Goal(type=GoalType.LEARN, description="Study")
        workspace.add_goal(goal)
        
        data = workspace.to_dict()
        
        # Should not raise an error
        json_str = json.dumps(data)
        assert isinstance(json_str, str)
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        # Create and serialize
        workspace1 = GlobalWorkspace(capacity=15)
        goal = Goal(type=GoalType.INTROSPECT, description="Reflect", priority=0.7)
        workspace1.add_goal(goal)
        workspace1.emotional_state["valence"] = 0.5
        workspace1.cycle_count = 5
        
        data = workspace1.to_dict()
        
        # Deserialize
        workspace2 = GlobalWorkspace.from_dict(data)
        
        # Verify
        assert workspace2.capacity == 15
        assert len(workspace2.current_goals) == 1
        assert workspace2.current_goals[0].description == "Reflect"
        assert workspace2.current_goals[0].priority == 0.7
        assert workspace2.emotional_state["valence"] == 0.5
        assert workspace2.cycle_count == 5
    
    def test_from_dict_with_percepts_and_memories(self):
        """Test from_dict with percepts and memories."""
        # Create workspace with rich state
        workspace1 = GlobalWorkspace()
        
        percept = Percept(modality="image", raw={"url": "test.png"})
        workspace1.active_percepts[percept.id] = percept
        
        memory = Memory(
            id="mem-1",
            content="Test memory",
            timestamp=datetime.now(),
            significance=0.8,
            tags=["test"]
        )
        workspace1.attended_memories.append(memory)
        
        data = workspace1.to_dict()
        
        # Restore
        workspace2 = GlobalWorkspace.from_dict(data)
        
        assert len(workspace2.active_percepts) == 1
        assert len(workspace2.attended_memories) == 1
        assert workspace2.attended_memories[0].content == "Test memory"
        assert workspace2.attended_memories[0].tags == ["test"]
    
    def test_serialization_roundtrip(self):
        """Test complete serialization and deserialization roundtrip."""
        # Create complex workspace state
        workspace1 = GlobalWorkspace(capacity=12)
        
        # Add goals
        for i in range(3):
            goal = Goal(
                type=GoalType.LEARN,
                description=f"Goal {i}",
                priority=0.3 + i * 0.2
            )
            workspace1.add_goal(goal)
        
        # Add percepts
        for i in range(2):
            percept = Percept(modality="text", raw=f"Percept {i}")
            workspace1.active_percepts[percept.id] = percept
        
        # Update emotions
        workspace1.emotional_state["valence"] = 0.6
        workspace1.emotional_state["arousal"] = 0.4
        
        # Serialize and deserialize
        data = workspace1.to_dict()
        workspace2 = GlobalWorkspace.from_dict(data)
        
        # Verify equivalence
        assert workspace2.capacity == workspace1.capacity
        assert len(workspace2.current_goals) == len(workspace1.current_goals)
        assert len(workspace2.active_percepts) == len(workspace1.active_percepts)
        assert workspace2.emotional_state == workspace1.emotional_state
        
        # Goals should maintain priority ordering
        for i in range(len(workspace1.current_goals)):
            assert workspace2.current_goals[i].description == workspace1.current_goals[i].description
            assert workspace2.current_goals[i].priority == workspace1.current_goals[i].priority


class TestIntegration:
    """Integration tests for realistic workspace usage."""
    
    def test_cognitive_cycle_simulation(self):
        """Simulate a few cognitive cycles with various updates."""
        workspace = GlobalWorkspace()
        
        # Cycle 1: Perceive user input and create response goal
        percept1 = Percept(modality="text", raw="What is consciousness?")
        goal1 = Goal(type=GoalType.RESPOND_TO_USER, description="Answer question", priority=0.9)
        
        workspace.update([
            {"type": "percept", "data": percept1},
            {"type": "goal", "data": goal1}
        ])
        
        assert workspace.cycle_count == 1
        assert len(workspace.current_goals) == 1
        assert len(workspace.active_percepts) == 1
        
        # Cycle 2: Retrieve relevant memory
        memory1 = Memory(
            id="mem-consciousness",
            content="Global Workspace Theory explains consciousness",
            timestamp=datetime.now(),
            significance=0.9,
            tags=["consciousness", "theory"]
        )
        
        workspace.update([{"type": "memory", "data": memory1}])
        
        assert workspace.cycle_count == 2
        assert len(workspace.attended_memories) == 1
        
        # Cycle 3: Generate response and complete goal
        workspace.emotional_state["valence"] = 0.7  # Positive emotion from helping
        workspace.remove_goal(goal1.id)
        workspace.update([])
        
        assert workspace.cycle_count == 3
        assert len(workspace.current_goals) == 0
        assert workspace.emotional_state["valence"] == 0.7
    
    def test_multiple_concurrent_goals(self):
        """Test handling multiple goals with different priorities."""
        workspace = GlobalWorkspace()
        
        goals = [
            Goal(type=GoalType.RESPOND_TO_USER, description="Answer Q1", priority=0.9),
            Goal(type=GoalType.COMMIT_MEMORY, description="Save important info", priority=0.7),
            Goal(type=GoalType.INTROSPECT, description="Reflect on behavior", priority=0.3),
        ]
        
        for goal in goals:
            workspace.add_goal(goal)
        
        # Highest priority goal should be first
        assert workspace.current_goals[0].priority == 0.9
        assert workspace.current_goals[0].type == GoalType.RESPOND_TO_USER
        
        # Complete high-priority goal
        workspace.remove_goal(goals[0].id)
        
        # Next highest should now be first
        assert workspace.current_goals[0].priority == 0.7
        assert workspace.current_goals[0].type == GoalType.COMMIT_MEMORY
