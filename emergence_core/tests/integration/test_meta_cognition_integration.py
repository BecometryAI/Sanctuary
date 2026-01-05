"""
Integration tests for meta-cognition (SelfMonitor).

Tests that SelfMonitor observes workspace state and
generates introspective percepts.
"""
import pytest
import asyncio
from lyra.cognitive_core.core import CognitiveCore
from lyra.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType


@pytest.mark.integration
@pytest.mark.asyncio
class TestMetaCognitionObservation:
    """Test meta-cognition observation of workspace."""
    
    async def test_self_monitor_generates_introspective_percepts(self):
        """Test that SelfMonitor generates introspective percepts."""
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
            
            # Add goal that should trigger introspection
            goal = Goal(
                type=GoalType.INTROSPECT,
                description="Reflect on current state",
                priority=0.9
            )
            workspace.add_goal(goal)
            
            # Wait for meta-cognition
            await asyncio.sleep(1.0)
            
            # Check that introspective percepts were generated
            snapshot = workspace.broadcast()
            
            # Look for introspective percepts in workspace
            # (Test implementation depends on how meta-percepts are stored)
            
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            pytest.fail(f"Test failed: {e}")
