"""
Tests for temporal awareness integration with the cognitive cycle.

Validates that temporal context is properly wired through the cognitive loop,
workspace broadcasts, and action tracking.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from emergence_core.lyra.cognitive_core.temporal import TemporalGrounding
from emergence_core.lyra.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType
from emergence_core.lyra.cognitive_core.core import CognitiveCore


class TestTemporalContextMethod:
    """Test the get_temporal_context() method."""
    
    def test_get_temporal_context_structure(self):
        """Test that temporal context has expected structure."""
        tg = TemporalGrounding()
        
        # Start a session
        tg.on_interaction()
        
        # Get temporal context
        context = tg.get_temporal_context()
        
        # Verify structure
        assert "cycle_timestamp" in context
        assert "session_start" in context
        assert "session_duration_seconds" in context
        assert "time_since_last_input_seconds" in context
        assert "time_since_last_action_seconds" in context
        assert "time_since_last_output_seconds" in context
        assert "cycles_this_session" in context
        assert "session_id" in context
        
        # Verify types
        assert isinstance(context["cycle_timestamp"], datetime)
        assert isinstance(context["session_id"], str)
        assert isinstance(context["cycles_this_session"], int)
    
    def test_temporal_context_no_session(self):
        """Test temporal context when no session is active."""
        tg = TemporalGrounding()
        
        # Get context without starting session
        context = tg.get_temporal_context()
        
        assert context["session_start"] is None
        assert context["session_duration_seconds"] is None
        assert context["session_id"] is None
    
    def test_temporal_context_with_session(self):
        """Test temporal context with active session."""
        tg = TemporalGrounding()
        
        # Start session
        tg.on_interaction()
        
        # Get context
        context = tg.get_temporal_context()
        
        assert context["session_start"] is not None
        assert context["session_duration_seconds"] is not None
        assert context["session_duration_seconds"] >= 0
        assert context["session_id"] is not None
    
    def test_cycles_this_session_increments(self):
        """Test that cycle count increments."""
        tg = TemporalGrounding()
        tg.on_interaction()
        
        context1 = tg.get_temporal_context()
        context2 = tg.get_temporal_context()
        context3 = tg.get_temporal_context()
        
        assert context2["cycles_this_session"] == context1["cycles_this_session"] + 1
        assert context3["cycles_this_session"] == context2["cycles_this_session"] + 1
    
    def test_session_duration_increases(self):
        """Test that session duration increases over time."""
        tg = TemporalGrounding()
        tg.on_interaction()
        
        context1 = tg.get_temporal_context()
        
        # Wait a bit
        import time
        time.sleep(0.1)
        
        context2 = tg.get_temporal_context()
        
        # Session duration should increase
        assert context2["session_duration_seconds"] > context1["session_duration_seconds"]


class TestTemporalEventRecording:
    """Test recording of temporal events (input, action, output)."""
    
    def test_record_input(self):
        """Test recording input events."""
        tg = TemporalGrounding()
        tg.on_interaction()
        
        # Initially no input time
        context = tg.get_temporal_context()
        assert context["time_since_last_input_seconds"] is None
        
        # Record input
        tg.record_input()
        
        # Now should have input time
        context = tg.get_temporal_context()
        assert context["time_since_last_input_seconds"] is not None
        assert context["time_since_last_input_seconds"] >= 0
    
    def test_record_action(self):
        """Test recording action events."""
        tg = TemporalGrounding()
        tg.on_interaction()
        
        # Initially no action time
        context = tg.get_temporal_context()
        assert context["time_since_last_action_seconds"] is None
        
        # Record action
        tg.record_action()
        
        # Now should have action time
        context = tg.get_temporal_context()
        assert context["time_since_last_action_seconds"] is not None
        assert context["time_since_last_action_seconds"] >= 0
    
    def test_record_output(self):
        """Test recording output events."""
        tg = TemporalGrounding()
        tg.on_interaction()
        
        # Initially no output time
        context = tg.get_temporal_context()
        assert context["time_since_last_output_seconds"] is None
        
        # Record output
        tg.record_output()
        
        # Now should have output time
        context = tg.get_temporal_context()
        assert context["time_since_last_output_seconds"] is not None
        assert context["time_since_last_output_seconds"] >= 0
    
    def test_time_since_input_updates(self):
        """Test that time since input increases over time."""
        tg = TemporalGrounding()
        tg.on_interaction()
        
        # Record input
        tg.record_input()
        
        context1 = tg.get_temporal_context()
        
        # Wait a bit
        import time
        time.sleep(0.1)
        
        context2 = tg.get_temporal_context()
        
        # Time since input should increase
        assert context2["time_since_last_input_seconds"] > context1["time_since_last_input_seconds"]


class TestWorkspaceTemporalContext:
    """Test temporal context in workspace."""
    
    def test_workspace_has_temporal_context_field(self):
        """Test that workspace has temporal context field."""
        workspace = GlobalWorkspace()
        
        assert hasattr(workspace, 'temporal_context')
        assert workspace.temporal_context is None  # Initially None
    
    def test_set_temporal_context(self):
        """Test setting temporal context in workspace."""
        workspace = GlobalWorkspace()
        
        temporal_context = {
            "cycle_timestamp": datetime.now(),
            "session_duration_seconds": 120.5,
            "time_since_last_input_seconds": 5.2,
            "cycles_this_session": 10
        }
        
        workspace.set_temporal_context(temporal_context)
        
        assert workspace.temporal_context == temporal_context
    
    def test_temporal_context_in_broadcast(self):
        """Test that temporal context is included in workspace broadcasts."""
        workspace = GlobalWorkspace()
        
        # Set temporal context
        temporal_context = {
            "cycle_timestamp": datetime.now(),
            "session_duration_seconds": 60.0,
            "cycles_this_session": 5
        }
        workspace.set_temporal_context(temporal_context)
        
        # Get broadcast
        snapshot = workspace.broadcast()
        
        # Verify temporal context is in snapshot
        assert hasattr(snapshot, 'temporal_context')
        assert snapshot.temporal_context is not None
        assert snapshot.temporal_context["session_duration_seconds"] == 60.0
        assert snapshot.temporal_context["cycles_this_session"] == 5
    
    def test_broadcast_without_temporal_context(self):
        """Test broadcast when no temporal context is set."""
        workspace = GlobalWorkspace()
        
        # Broadcast without setting temporal context
        snapshot = workspace.broadcast()
        
        # Should still work, with None temporal_context
        assert hasattr(snapshot, 'temporal_context')
        assert snapshot.temporal_context is None
    
    def test_workspace_clear_resets_temporal_context(self):
        """Test that clearing workspace resets temporal context."""
        workspace = GlobalWorkspace()
        
        # Set temporal context
        workspace.set_temporal_context({"cycles_this_session": 10})
        assert workspace.temporal_context is not None
        
        # Clear workspace
        workspace.clear()
        
        # Temporal context should be reset
        assert workspace.temporal_context is None


@pytest.mark.integration
@pytest.mark.asyncio
class TestTemporalCycleIntegration:
    """Integration tests for temporal context in cognitive cycle."""
    
    async def test_temporal_context_populated_in_cycle(self):
        """Test that temporal context is populated at each cycle."""
        workspace = GlobalWorkspace()
        config = {
            "cycle_rate_hz": 10,
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
        }
        
        core = CognitiveCore(workspace=workspace, config=config)
        
        try:
            # Start core
            start_task = asyncio.create_task(core.start())
            await asyncio.sleep(0.5)
            
            # Let it run a couple cycles
            await asyncio.sleep(0.3)
            
            # Check workspace has temporal context
            snapshot = workspace.broadcast()
            assert snapshot.temporal_context is not None
            assert "cycle_timestamp" in snapshot.temporal_context
            assert "cycles_this_session" in snapshot.temporal_context
            
            # Stop core
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            pytest.fail(f"Temporal cycle integration failed: {e}")
    
    async def test_session_duration_increases_across_cycles(self):
        """Test that session duration increases across cycles."""
        workspace = GlobalWorkspace()
        config = {
            "cycle_rate_hz": 10,
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
        }
        
        core = CognitiveCore(workspace=workspace, config=config)
        
        try:
            # Start core
            start_task = asyncio.create_task(core.start())
            await asyncio.sleep(0.5)
            
            # Get first snapshot
            await asyncio.sleep(0.2)
            snapshot1 = workspace.broadcast()
            duration1 = snapshot1.temporal_context.get("session_duration_seconds") if snapshot1.temporal_context else None
            
            # Wait and get second snapshot
            await asyncio.sleep(0.3)
            snapshot2 = workspace.broadcast()
            duration2 = snapshot2.temporal_context.get("session_duration_seconds") if snapshot2.temporal_context else None
            
            # Session duration should increase
            if duration1 is not None and duration2 is not None:
                assert duration2 > duration1
            
            # Stop core
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            pytest.fail(f"Session duration test failed: {e}")
    
    async def test_time_since_input_updates_after_injection(self):
        """Test that time_since_last_input updates correctly after input injection."""
        workspace = GlobalWorkspace()
        config = {
            "cycle_rate_hz": 10,
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
        }
        
        core = CognitiveCore(workspace=workspace, config=config)
        
        try:
            # Start core
            start_task = asyncio.create_task(core.start())
            await asyncio.sleep(0.5)
            
            # Inject input
            core.inject_input("Test input", modality="text")
            
            # Wait for processing
            await asyncio.sleep(0.3)
            
            # Check temporal context
            snapshot = workspace.broadcast()
            if snapshot.temporal_context:
                time_since_input = snapshot.temporal_context.get("time_since_last_input_seconds")
                # Should have recorded input time
                if time_since_input is not None:
                    assert time_since_input >= 0
                    assert time_since_input < 1.0  # Should be less than 1 second since we just processed
            
            # Stop core
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            pytest.fail(f"Time since input test failed: {e}")
    
    async def test_cycles_this_session_increments(self):
        """Test that cycles_this_session increments with each cycle."""
        workspace = GlobalWorkspace()
        config = {
            "cycle_rate_hz": 10,
            "checkpointing": {"enabled": False},
            "input_llm": {"use_real_model": False},
            "output_llm": {"use_real_model": False},
        }
        
        core = CognitiveCore(workspace=workspace, config=config)
        
        try:
            # Start core
            start_task = asyncio.create_task(core.start())
            await asyncio.sleep(0.5)
            
            # Get first cycle count
            await asyncio.sleep(0.2)
            snapshot1 = workspace.broadcast()
            cycles1 = snapshot1.temporal_context.get("cycles_this_session") if snapshot1.temporal_context else 0
            
            # Wait for more cycles
            await asyncio.sleep(0.3)
            snapshot2 = workspace.broadcast()
            cycles2 = snapshot2.temporal_context.get("cycles_this_session") if snapshot2.temporal_context else 0
            
            # Cycle count should increase
            assert cycles2 > cycles1
            
            # Stop core
            await core.stop()
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                pass
        except Exception as e:
            pytest.fail(f"Cycle count test failed: {e}")
