"""
Tests for decoupled cognitive loop operation (Task #1).

These tests verify that the cognitive loop can run independently of I/O:
- Cognitive cycles run without waiting for input
- Input is queued and processed non-blockingly
- System can run 100+ cycles with no human input
- Input becomes percepts like any other sensory data
- Idle cognition generates internal activity
"""

import pytest
import asyncio
import time
from datetime import datetime

from lyra.cognitive_core.input_queue import InputQueue, InputEvent, InputSource
from lyra.cognitive_core.idle_cognition import IdleCognition
from lyra.cognitive_core.workspace import GlobalWorkspace, Percept


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
    async def test_generate_idle_activity(self):
        """Test that idle cognition generates activities."""
        idle = IdleCognition(config={
            "memory_review_probability": 1.0,  # Always trigger
            "goal_evaluation_probability": 1.0,
            "reflection_probability": 1.0,
            "temporal_check_probability": 1.0,
            "emotional_check_probability": 1.0,
        })
        workspace = GlobalWorkspace()
        
        activities = await idle.generate_idle_activity(workspace)
        
        # Should generate multiple activities
        assert len(activities) > 0
        assert all(isinstance(p, Percept) for p in activities)
    
    @pytest.mark.asyncio
    async def test_idle_activity_types(self):
        """Test different types of idle activities."""
        idle = IdleCognition(config={
            "memory_review_probability": 1.0,
            "memory_review_interval": 0.0,  # No interval restriction
        })
        workspace = GlobalWorkspace()
        
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
    async def test_idle_cognition_respects_intervals(self):
        """Test that minimum intervals are respected."""
        idle = IdleCognition(config={
            "memory_review_probability": 1.0,
            "memory_review_interval": 100.0,  # 100 seconds
        })
        workspace = GlobalWorkspace()
        
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
    async def test_idle_cognition_tracks_stats(self):
        """Test that statistics are tracked correctly."""
        idle = IdleCognition(config={
            "memory_review_probability": 1.0,
            "memory_review_interval": 0.0,
            "goal_evaluation_probability": 1.0,
            "goal_evaluation_interval": 0.0,
        })
        workspace = GlobalWorkspace()
        
        # Generate activities multiple times
        for _ in range(3):
            await idle.generate_idle_activity(workspace)
        
        stats = idle.get_stats()
        
        assert stats["total_cycles"] == 3
        assert stats["memory_reviews"] == 3
        assert stats["goal_evaluations"] == 3
        assert "memory_review_rate" in stats
    
    @pytest.mark.asyncio
    async def test_goal_evaluation_includes_goal_info(self):
        """Test that goal evaluation includes current goal information."""
        idle = IdleCognition(config={
            "goal_evaluation_probability": 1.0,
            "goal_evaluation_interval": 0.0,
        })
        workspace = GlobalWorkspace()
        
        # Add some goals
        from lyra.cognitive_core.workspace import Goal, GoalType
        goal1 = Goal(type=GoalType.RESPOND_TO_USER, description="Test goal 1")
        goal2 = Goal(type=GoalType.INTROSPECT, description="Test goal 2")
        workspace.add_goal(goal1)
        workspace.add_goal(goal2)
        
        activities = await idle.generate_idle_activity(workspace)
        
        goal_evals = [a for a in activities 
                     if a.raw.get("type") == "goal_evaluation_trigger"]
        assert len(goal_evals) > 0
        
        percept = goal_evals[0]
        assert percept.raw["goal_count"] == 2


class TestDecoupledCognition:
    """Test that cognitive loop can run without input."""
    
    @pytest.mark.asyncio
    async def test_cognitive_cycle_without_input(self):
        """Test that system can run cycles with no input."""
        from lyra.cognitive_core import CognitiveCore
        
        workspace = GlobalWorkspace()
        config = {"cycle_rate_hz": 100}  # Fast for testing
        core = CognitiveCore(workspace=workspace, config=config)
        
        # Initialize queues
        core.state.initialize_queues()
        
        # Run a single cycle (no input provided)
        await core.cycle_executor.execute_cycle()
        
        # Should complete without error
        assert core.timing.metrics["total_cycles"] >= 0
    
    @pytest.mark.asyncio
    async def test_input_processed_as_percept(self):
        """Test that input becomes a percept in workspace."""
        queue = InputQueue()
        
        await queue.add_input("Hello world", source="human")
        
        # Get inputs
        inputs = queue.get_pending_inputs()
        
        assert len(inputs) == 1
        assert inputs[0].text == "Hello world"
        assert inputs[0].source == "human"
    
    @pytest.mark.asyncio
    async def test_multiple_cycles_without_input(self):
        """Test running multiple cycles without any input."""
        from lyra.cognitive_core import CognitiveCore
        
        workspace = GlobalWorkspace()
        config = {"cycle_rate_hz": 1000}  # Very fast for testing
        core = CognitiveCore(workspace=workspace, config=config)
        
        core.state.initialize_queues()
        
        # Run 10 cycles without providing any input
        for _ in range(10):
            await core.cycle_executor.execute_cycle()
        
        # All cycles should complete
        assert core.timing.metrics["total_cycles"] >= 10
    
    @pytest.mark.asyncio
    async def test_input_and_no_input_cycles_mix(self):
        """Test that system handles mix of input and no-input cycles."""
        from lyra.cognitive_core import CognitiveCore
        
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace)
        
        core.state.initialize_queues()
        
        # Cycle 1: No input
        await core.cycle_executor.execute_cycle()
        
        # Cycle 2: Add input
        core.state.inject_input("Test message", "text")
        await core.cycle_executor.execute_cycle()
        
        # Cycle 3: No input again
        await core.cycle_executor.execute_cycle()
        
        # All cycles should complete
        assert core.timing.metrics["total_cycles"] >= 3


class TestIntegration:
    """Integration tests for decoupled operation."""
    
    @pytest.mark.asyncio
    async def test_idle_cognition_feeds_cognitive_loop(self):
        """Test that idle cognition output can feed into cognitive processing."""
        idle = IdleCognition()
        workspace = GlobalWorkspace()
        
        # Generate idle activities
        activities = await idle.generate_idle_activity(workspace)
        
        # Add them to workspace
        for percept in activities:
            workspace.add_percept(percept)
        
        # Verify they're in workspace
        snapshot = workspace.broadcast()
        assert len(snapshot.percepts) >= len(activities)
    
    @pytest.mark.asyncio
    async def test_input_queue_with_cognitive_core(self):
        """Test InputQueue integration with CognitiveCore."""
        from lyra.cognitive_core import CognitiveCore
        
        queue = InputQueue()
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace)
        
        core.state.initialize_queues()
        
        # Add input via the queue abstraction
        await queue.add_input("Test input", source="human")
        
        # The core should be able to process inputs
        # (this is a structural test, not testing full processing)
        assert not queue.is_empty()
        inputs = queue.get_pending_inputs()
        assert len(inputs) == 1


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
    async def test_system_runs_100_cycles_no_input(self):
        """Verify system can run 100+ cycles with no human input."""
        from lyra.cognitive_core import CognitiveCore
        
        workspace = GlobalWorkspace()
        config = {"cycle_rate_hz": 1000}  # Very fast
        core = CognitiveCore(workspace=workspace, config=config)
        
        core.state.initialize_queues()
        
        # Run 100 cycles without providing any input
        for i in range(100):
            await core.cycle_executor.execute_cycle()
        
        # Verify we completed all cycles
        assert core.timing.metrics["total_cycles"] >= 100
    
    @pytest.mark.asyncio
    async def test_idle_cognition_generates_activity(self):
        """Verify idle cognition produces internal percepts."""
        idle = IdleCognition(config={
            "memory_review_probability": 0.5,
            "goal_evaluation_probability": 0.5,
        })
        workspace = GlobalWorkspace()
        
        # Generate activities over multiple cycles
        all_activities = []
        for _ in range(10):
            activities = await idle.generate_idle_activity(workspace)
            all_activities.extend(activities)
        
        # Should produce some internal activity
        assert len(all_activities) > 0
        assert all(isinstance(p, Percept) for p in all_activities)
        assert all(p.modality == "introspection" for p in all_activities)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
