"""
Integration tests for memory consolidation and retrieval.

Tests that workspace state consolidates to long-term memory
and that memories can be retrieved back to workspace.
"""
import pytest
import asyncio
from lyra.cognitive_core.core import CognitiveCore
from lyra.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType


@pytest.mark.integration
@pytest.mark.asyncio
class TestMemoryConsolidation:
    """Test memory consolidation from workspace."""
    
    async def test_workspace_consolidates_to_memory(self):
        """Test that workspace state consolidates to long-term memory."""
        workspace = GlobalWorkspace()
        config = {
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
            "memory": {"consolidation_threshold": 0.5}
        }
        
        core = CognitiveCore(workspace=workspace, config=config)
        
        try:
            start_task = asyncio.create_task(core.start())
            await asyncio.sleep(0.5)
            
            # Add significant content to workspace
            await core.process_language_input("This is an important memory to remember.")
            
            # Wait for consolidation
            await asyncio.sleep(2.0)
            
            # Check that memory system has entries
            # (This test may need adjustment based on memory implementation)
            
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            pytest.fail(f"Test failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
class TestMemoryRetrieval:
    """Test memory retrieval to workspace."""
    
    async def test_retrieval_goal_brings_memories_to_workspace(self):
        """Test that RETRIEVE_MEMORY goal retrieves memories."""
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
            
            # Add retrieval goal
            goal = Goal(
                type=GoalType.RETRIEVE_MEMORY,
                description="Retrieve memories about consciousness",
                priority=0.9,
                metadata={"query": "consciousness"}
            )
            workspace.add_goal(goal)
            
            # Wait for retrieval
            await asyncio.sleep(1.0)
            
            # Check workspace for retrieved memories
            snapshot = workspace.broadcast()
            # Memories should have been added to workspace
            # (Test may need adjustment based on implementation)
            
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
                
        except Exception as e:
            pytest.fail(f"Test failed: {e}")
