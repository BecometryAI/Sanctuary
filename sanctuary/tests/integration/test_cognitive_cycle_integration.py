"""
Integration tests for complete cognitive cycle execution.

Tests that all 12 steps of the cognitive cycle execute correctly
and that state updates persist across cycles.
"""
import pytest
import asyncio
from mind.cognitive_core.core import CognitiveCore
from mind.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType


@pytest.mark.integration
@pytest.mark.asyncio
class TestCompleteCognitiveCycle:
    """Test complete cognitive cycle execution."""
    
    async def test_complete_cycle_executes_without_errors(self):
        """Test that a complete cognitive cycle executes all steps."""
        workspace = GlobalWorkspace()
        config = {
            "cycle_rate_hz": 10,
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
        }
        
        core = CognitiveCore(workspace=workspace, config=config)
        
        # Add a test goal
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Test cycle execution",
            priority=0.8
        )
        workspace.add_goal(goal)
        
        try:
            # Start core
            start_task = asyncio.create_task(core.start())
            
            # Wait for initialization
            await asyncio.sleep(0.5)
            
            # Let it run a few cycles
            await asyncio.sleep(1.0)
            
            # Stop core
            await core.stop()
            
            # Cancel start task
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
            
            # Verify cycles executed
            metrics = core.get_metrics()
            assert metrics['total_cycles'] > 0
            assert metrics['avg_cycle_time_ms'] < 200  # Should be under 200ms
            
        except Exception as e:
            pytest.fail(f"Cognitive cycle failed with error: {e}")
    
    async def test_cycle_updates_workspace_state(self):
        """Test that cognitive cycles update workspace state."""
        workspace = GlobalWorkspace()
        config = {
            "cycle_rate_hz": 10,
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
        }
        
        core = CognitiveCore(workspace=workspace, config=config)
        
        # Record initial state
        initial_snapshot = workspace.broadcast()
        initial_cycle_count = initial_snapshot.cycle_count
        
        try:
            # Start core
            start_task = asyncio.create_task(core.start())
            await asyncio.sleep(0.5)
            
            # Inject input
            core.inject_input("Hello Sanctuary", modality="text")
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            # Stop core
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
            
            # Verify state updated
            final_snapshot = workspace.broadcast()
            
            assert final_snapshot.cycle_count > initial_cycle_count
            # Workspace should have been updated multiple times
            
        except Exception as e:
            pytest.fail(f"Test failed: {e}")
    
    async def test_cycle_timing_enforced(self):
        """Test that cycle timing enforcement is working."""
        workspace = GlobalWorkspace()
        config = {
            "cycle_rate_hz": 10,
            "timing": {
                "warn_threshold_ms": 100,
                "critical_threshold_ms": 200,
            },
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
        }
        
        core = CognitiveCore(workspace=workspace, config=config)
        
        try:
            start_task = asyncio.create_task(core.start())
            await asyncio.sleep(0.5)
            
            # Run for a bit
            await asyncio.sleep(2.0)
            
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
            
            # Check metrics
            metrics = core.get_metrics()
            
            # Verify timing metrics exist
            assert 'slow_cycles' in metrics
            assert 'critical_cycles' in metrics
            assert 'slowest_cycle_ms' in metrics
            
            # Most cycles should be within target (< 100ms)
            if metrics['total_cycles'] > 0:
                slow_percentage = metrics.get('slow_cycle_percentage', 0)
                # Allow some tolerance, but most should be fast
                assert slow_percentage < 50  # Less than 50% slow
                
        except Exception as e:
            pytest.fail(f"Test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestInputOutputFlow:
    """Test input → processing → output flow."""
    
    async def test_language_input_creates_goals(self):
        """Test that process_language_input creates goals in workspace."""
        workspace = GlobalWorkspace()
        config = {
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
        }
        
        core = CognitiveCore(workspace=workspace, config=config)
        
        try:
            start_task = asyncio.create_task(core.start())
            await asyncio.sleep(0.5)
            
            # Process language input
            await core.process_language_input("Hello, how are you?")
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Check that goals were added
            snapshot = workspace.broadcast()
            assert len(snapshot.goals) > 0
            
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            pytest.fail(f"Test failed: {e}")
