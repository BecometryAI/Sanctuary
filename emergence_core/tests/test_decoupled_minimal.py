"""
Minimal tests for decoupled cognitive loop operation (Task #1).

These tests verify core input queue and idle cognition functionality
without requiring full CognitiveCore initialization.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import MagicMock

# Use standard imports instead of sys.path manipulation
from emergence_core.lyra.cognitive_core.input_queue import InputQueue, InputEvent, InputSource
from emergence_core.lyra.cognitive_core.idle_cognition import IdleCognition


class MockWorkspace:
    """Minimal workspace mock for testing."""
    def __init__(self):
        self.current_goals = []
        self.percepts = {}
        self.active_percepts = {}
    
    def add_percept(self, percept):
        self.percepts[percept.id] = percept
    
    def broadcast(self):
        return self


class MockPercept:
    """Minimal percept mock for testing."""
    def __init__(self, modality, raw, complexity=1, metadata=None):
        import uuid
        self.id = str(uuid.uuid4())
        self.modality = modality
        self.raw = raw
        self.complexity = complexity
        self.metadata = metadata or {}


class TestInputQueue:
    """Test non-blocking input queue functionality."""
    
    def test_input_queue_initialization(self):
        """Test that InputQueue initializes correctly."""
        queue = InputQueue(max_size=50)
        
        assert queue.max_size == 50
        assert queue.is_empty()
        assert queue.size() == 0
        assert queue.total_inputs_received == 0
    
    @pytest.mark.asyncio
    async def test_add_input(self):
        """Test adding input to queue."""
        queue = InputQueue()
        
        result = await queue.add_input("Hello world", source="human")
        
        assert result is True
        assert queue.size() == 1
        assert queue.total_inputs_received == 1
        assert not queue.is_empty()
    
    @pytest.mark.asyncio
    async def test_get_pending_inputs_non_blocking(self):
        """Test that getting inputs is non-blocking."""
        queue = InputQueue()
        
        # Get from empty queue should return immediately
        start = time.time()
        inputs = queue.get_pending_inputs()
        elapsed = time.time() - start
        
        assert inputs == []
        assert elapsed < 0.01  # Should be nearly instant
    
    @pytest.mark.asyncio
    async def test_input_queue_preserves_order(self):
        """Test that inputs are retrieved in FIFO order."""
        queue = InputQueue()
        
        await queue.add_input("First", source="human")
        await queue.add_input("Second", source="human")
        await queue.add_input("Third", source="human")
        
        inputs = queue.get_pending_inputs()
        
        assert len(inputs) == 3
        assert inputs[0].text == "First"
        assert inputs[1].text == "Second"
        assert inputs[2].text == "Third"
    
    @pytest.mark.asyncio
    async def test_input_event_has_metadata(self):
        """Test that InputEvent captures metadata."""
        queue = InputQueue()
        
        await queue.add_input(
            "Test input",
            modality="text",
            source="api",
            metadata={"user_id": "123"}
        )
        
        inputs = queue.get_pending_inputs()
        
        assert len(inputs) == 1
        event = inputs[0]
        assert isinstance(event, InputEvent)
        assert event.text == "Test input"
        assert event.modality == "text"
        assert event.source == "api"
        assert event.metadata["user_id"] == "123"
        assert isinstance(event.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_input_queue_handles_full_queue(self):
        """Test behavior when queue is full."""
        queue = InputQueue(max_size=2)
        
        await queue.add_input("First")
        await queue.add_input("Second")
        result = await queue.add_input("Third")  # Should be dropped
        
        assert result is False
        assert queue.size() == 2
        assert queue.stats["dropped"] == 1
    
    @pytest.mark.asyncio
    async def test_input_queue_tracks_by_source(self):
        """Test that stats track inputs by source."""
        queue = InputQueue()
        
        await queue.add_input("From human", source="human")
        await queue.add_input("From API", source="api")
        await queue.add_input("Another human", source="human")
        
        stats = queue.get_stats()
        
        assert stats["total_received"] == 3
        assert stats["by_source"]["human"] == 2
        assert stats["by_source"]["api"] == 1
    
    def test_input_queue_clear(self):
        """Test clearing the queue."""
        queue = InputQueue()
        
        # Use asyncio.run for the async operations
        async def setup():
            await queue.add_input("First")
            await queue.add_input("Second")
            await queue.add_input("Third")
        
        asyncio.run(setup())
        
        assert queue.size() == 3
        
        cleared = queue.clear()
        
        assert cleared == 3
        assert queue.is_empty()
        assert queue.size() == 0
    
    def test_input_queue_validation(self):
        """Test that invalid max_size raises error."""
        with pytest.raises(ValueError, match="max_size must be positive"):
            InputQueue(max_size=0)
        
        with pytest.raises(ValueError, match="max_size must be positive"):
            InputQueue(max_size=-1)
    
    @pytest.mark.asyncio
    async def test_invalid_modality_handling(self):
        """Test that invalid modality is handled gracefully."""
        queue = InputQueue()
        
        # Should accept with warning and default to 'text'
        result = await queue.add_input("Test", modality="invalid_type")
        
        assert result is True
        inputs = queue.get_pending_inputs()
        assert len(inputs) == 1
        assert inputs[0].modality == "text"  # Should default to text


class TestIdleCognition:
    """Test idle cognition activity generation."""
    
    def test_idle_cognition_initialization(self):
        """Test that IdleCognition initializes correctly."""
        idle = IdleCognition()
        
        assert idle.memory_review_probability > 0
        assert idle.goal_evaluation_probability > 0
        assert idle.cycle_count == 0
        assert "total_cycles" in idle.stats
    
    @pytest.mark.asyncio
    async def test_generate_idle_activity(self, monkeypatch):
        """Test that idle cognition generates activities."""
        # Use monkeypatch to replace Percept at module level
        mock_percept_class = MockPercept
        monkeypatch.setattr('emergence_core.lyra.cognitive_core.idle_cognition.Percept', mock_percept_class)
        
        idle = IdleCognition(config={
            "memory_review_probability": 1.0,  # Always trigger
            "goal_evaluation_probability": 1.0,
            "reflection_probability": 1.0,
            "temporal_check_probability": 1.0,
            "emotional_check_probability": 1.0,
        })
        workspace = MockWorkspace()
        
        activities = await idle.generate_idle_activity(workspace)
        
        # Should generate multiple activities
        assert len(activities) > 0
        assert all(isinstance(p, MockPercept) for p in activities)
    
    @pytest.mark.asyncio
    async def test_idle_activity_types(self, monkeypatch):
        """Test different types of idle activities."""
        # Use monkeypatch to replace Percept at module level
        mock_percept_class = MockPercept
        monkeypatch.setattr('emergence_core.lyra.cognitive_core.idle_cognition.Percept', mock_percept_class)
        
        idle = IdleCognition(config={
            "memory_review_probability": 1.0,
            "memory_review_interval": 0.0,  # No interval restriction
        })
        workspace = MockWorkspace()
        
        activities = await idle.generate_idle_activity(workspace)
        
        # Should have memory review
        memory_reviews = [a for a in activities 
                         if a.raw.get("type") == "memory_review_trigger"]
        assert len(memory_reviews) > 0
        
        percept = memory_reviews[0]
        assert percept.modality == "introspection"
        assert percept.metadata["source"] == "idle_cognition"
        assert percept.metadata["activity_type"] == "memory_review"
    
    @pytest.mark.asyncio
    async def test_idle_cognition_respects_intervals(self, monkeypatch):
        """Test that minimum intervals are respected."""
        # Use monkeypatch to replace Percept at module level
        mock_percept_class = MockPercept
        monkeypatch.setattr('emergence_core.lyra.cognitive_core.idle_cognition.Percept', mock_percept_class)
        
        idle = IdleCognition(config={
            "memory_review_probability": 1.0,
            "memory_review_interval": 100.0,  # 100 seconds
        })
        workspace = MockWorkspace()
        
        # First call should trigger
        activities1 = await idle.generate_idle_activity(workspace)
        memory_reviews1 = [a for a in activities1 
                          if a.raw.get("type") == "memory_review_trigger"]
        
        # Second immediate call should NOT trigger (interval not met)
        activities2 = await idle.generate_idle_activity(workspace)
        memory_reviews2 = [a for a in activities2 
                          if a.raw.get("type") == "memory_review_trigger"]
        
        assert len(memory_reviews1) > 0
        assert len(memory_reviews2) == 0
    
    @pytest.mark.asyncio
    async def test_idle_cognition_tracks_stats(self, monkeypatch):
        """Test that statistics are tracked correctly."""
        # Use monkeypatch to replace Percept at module level
        mock_percept_class = MockPercept
        monkeypatch.setattr('emergence_core.lyra.cognitive_core.idle_cognition.Percept', mock_percept_class)
        
        idle = IdleCognition(config={
            "memory_review_probability": 1.0,
            "memory_review_interval": 0.0,
            "goal_evaluation_probability": 1.0,
            "goal_evaluation_interval": 0.0,
        })
        workspace = MockWorkspace()
        
        # Generate activities multiple times
        for _ in range(3):
            await idle.generate_idle_activity(workspace)
        
        stats = idle.get_stats()
        
        assert stats["total_cycles"] == 3
        assert stats["memory_reviews"] == 3
        assert stats["goal_evaluations"] == 3
        assert "memory_review_rate" in stats


class TestAcceptanceCriteria:
    """Tests for acceptance criteria from problem statement."""
    
    @pytest.mark.asyncio
    async def test_input_queue_is_non_blocking(self):
        """Verify input queue doesn't block cognitive loop."""
        queue = InputQueue()
        
        # Get from empty queue should return immediately
        start = time.time()
        inputs = queue.get_pending_inputs()
        elapsed = time.time() - start
        
        assert inputs == []
        assert elapsed < 0.01  # Should be nearly instant
    
    @pytest.mark.asyncio
    async def test_idle_cognition_generates_activity(self, monkeypatch):
        """Verify idle cognition produces internal percepts."""
        # Use monkeypatch to replace Percept at module level
        mock_percept_class = MockPercept
        monkeypatch.setattr('emergence_core.lyra.cognitive_core.idle_cognition.Percept', mock_percept_class)
        
        idle = IdleCognition(config={
            "memory_review_probability": 0.5,
            "goal_evaluation_probability": 0.5,
        })
        workspace = MockWorkspace()
        
        # Generate activities over multiple cycles
        all_activities = []
        for _ in range(10):
            activities = await idle.generate_idle_activity(workspace)
            all_activities.extend(activities)
        
        # Should produce some internal activity
        assert len(all_activities) > 0
        assert all(isinstance(p, MockPercept) for p in all_activities)
        assert all(p.modality == "introspection" for p in all_activities)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
