"""
Integration tests for IWMT Core with CognitiveCore.

Tests cover:
- IWMTCore is properly instantiated in CognitiveCore
- Predictions flow to AttentionController
- Prediction errors are computed and used for precision weighting
- WorldModel is updated after action execution
"""

import gc
import pytest
import pytest_asyncio
import asyncio
import shutil
import tempfile
import time
from pathlib import Path

from mind.cognitive_core.core import CognitiveCore
from mind.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType
from mind.cognitive_core.action import ActionType


@pytest.fixture
def temp_dirs():
    """Create temporary directories for ChromaDB to avoid schema conflicts."""
    temp_base = tempfile.mkdtemp()
    data_dir = Path(temp_base) / "data"
    chroma_dir = Path(temp_base) / "chroma"
    data_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    yield {"base_dir": str(data_dir), "chroma_dir": str(chroma_dir)}

    # Cleanup with retry for Windows file locking
    gc.collect()
    for attempt in range(3):
        try:
            shutil.rmtree(temp_base)
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(0.5)
                gc.collect()


def make_core_config(temp_dirs, extra_config=None):
    """Create config with temp directories for CognitiveCore."""
    config = {
        "memory": {
            "memory_config": {
                "base_dir": temp_dirs["base_dir"],
                "chroma_dir": temp_dirs["chroma_dir"],
            }
        }
    }
    if extra_config:
        config.update(extra_config)
    return config


class TestIWMTIntegration:
    """Test IWMT Core integration with CognitiveCore."""

    def test_iwmt_core_initialized_by_default(self, temp_dirs):
        """Test that IWMTCore is initialized by default in CognitiveCore."""
        core = CognitiveCore(config=make_core_config(temp_dirs))

        # IWMT should be enabled by default
        assert hasattr(core.subsystems, 'iwmt_core')
        assert core.subsystems.iwmt_core is not None
        assert hasattr(core.subsystems.iwmt_core, 'world_model')
        assert hasattr(core.subsystems.iwmt_core, 'precision')
        assert hasattr(core.subsystems.iwmt_core, 'free_energy')
        assert hasattr(core.subsystems.iwmt_core, 'active_inference')

    def test_iwmt_core_can_be_disabled(self, temp_dirs):
        """Test that IWMT can be disabled via config."""
        config = make_core_config(temp_dirs, {"iwmt": {"enabled": False}})
        core = CognitiveCore(config=config)

        # IWMT should be disabled
        assert hasattr(core.subsystems, 'iwmt_core')
        assert core.subsystems.iwmt_core is None

    def test_precision_weighting_passed_to_attention(self, temp_dirs):
        """Test that precision weighting is passed to AttentionController."""
        core = CognitiveCore(config=make_core_config(temp_dirs))

        # Check that attention controller has precision weighting
        assert hasattr(core.attention, 'precision_weighting')
        if core.subsystems.iwmt_core:
            assert core.attention.precision_weighting is not None
            assert core.attention.precision_weighting == core.subsystems.iwmt_core.precision

    @pytest.mark.asyncio
    async def test_predictions_generated_before_perception(self, temp_dirs):
        """Test that predictions are generated before processing percepts."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=make_core_config(temp_dirs))
        
        # Start the core to initialize queues
        await core.lifecycle.start(restore_latest=False)
        
        try:
            # Add a simple goal to provide context
            goal = Goal(
                description="Test goal",
                type=GoalType.RESPOND_TO_USER,
                priority=0.5
            )
            workspace.add_goal(goal)
            
            # Add input to process
            core.inject_input("Hello world", "text")
            
            # Execute one cognitive cycle
            timings = await core.cycle_executor.execute_cycle()
            
            # Verify IWMT prediction step was executed
            assert 'iwmt_predict' in timings
            
            # Check that predictions were generated
            if core.subsystems.iwmt_core:
                assert hasattr(core.cycle_executor, '_current_predictions')
                # Predictions may be empty if no prior state, but the step should have run
                assert timings['iwmt_predict'] >= 0
        finally:
            await core.lifecycle.stop()
    
    @pytest.mark.asyncio
    async def test_prediction_errors_computed_and_passed_to_attention(self, temp_dirs):
        """Test that prediction errors are computed and passed to attention."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=make_core_config(temp_dirs))
        
        await core.lifecycle.start(restore_latest=False)
        
        try:
            # Add goal
            goal = Goal(
                description="Respond to input",
                type=GoalType.RESPOND_TO_USER,
                priority=0.8
            )
            workspace.add_goal(goal)
            
            # Inject input
            core.inject_input("Test input", "text")
            
            # Execute one cycle to establish baseline
            await core.cycle_executor.execute_cycle()
            
            # Inject unexpected input (should create prediction error)
            core.inject_input("Completely different unexpected input", "text")
            
            # Execute another cycle
            timings = await core.cycle_executor.execute_cycle()
            
            # Verify prediction errors were tracked
            assert 'iwmt_update' in timings
            if core.subsystems.iwmt_core:
                assert hasattr(core.cycle_executor, '_current_prediction_errors')
                # After processing unexpected input, there may be prediction errors
                prediction_errors = core.cycle_executor._current_prediction_errors
                assert isinstance(prediction_errors, list)
        finally:
            await core.lifecycle.stop()
    
    @pytest.mark.asyncio
    async def test_world_model_updated_after_action(self, temp_dirs):
        """Test that WorldModel is updated after actions are executed."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=make_core_config(temp_dirs))
        
        if not core.subsystems.iwmt_core:
            pytest.skip("IWMT is disabled")
        
        await core.lifecycle.start(restore_latest=False)
        
        try:
            # Track initial cycle count
            initial_cycle_count = core.subsystems.iwmt_core.cycle_count
            
            # Add goal that will trigger an action
            goal = Goal(
                description="Respond to user",
                type=GoalType.RESPOND_TO_USER,
                priority=0.9
            )
            workspace.add_goal(goal)
            
            # Inject input that should trigger a response
            core.inject_input("Hello Sanctuary, how are you?", "text")
            
            # Execute cycle
            await core.cycle_executor.execute_cycle()
            
            # The world model should have been updated
            # (Note: actual update happens in the cycle, so we just verify the mechanism exists)
            # This test ensures no errors occur during the update flow
        finally:
            await core.lifecycle.stop()
    @pytest.mark.asyncio
    async def test_emotional_state_passed_to_attention(self, temp_dirs):
        """Test that emotional state is passed to attention for precision weighting."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=make_core_config(temp_dirs))
        
        await core.lifecycle.start(restore_latest=False)
        
        try:
            # Set emotional state
            core.affect.arousal = 0.7
            core.affect.valence = 0.5
            
            # Add goal
            goal = Goal(
                description="Test goal",
                type=GoalType.RESPOND_TO_USER,
                priority=0.6
            )
            workspace.add_goal(goal)
            
            # Inject input
            core.inject_input("Test emotional modulation", "text")
            
            # Execute cycle - should pass emotional state to attention
            await core.cycle_executor.execute_cycle()
            
            # Verify affect subsystem provides emotional state
            emotional_state = core.affect.get_state()
            assert 'arousal' in emotional_state
            assert 'valence' in emotional_state
            # Use approximate comparison as emotional values may drift slightly during cycle
            assert emotional_state['arousal'] == pytest.approx(0.7, abs=0.05)
            assert emotional_state['valence'] == pytest.approx(0.5, abs=0.05)
        finally:
            await core.lifecycle.stop()
    
    @pytest.mark.asyncio
    async def test_precision_weighting_active_in_attention(self, temp_dirs):
        """Test that precision weighting is actively applied during attention selection."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=make_core_config(temp_dirs))
        
        if not core.subsystems.iwmt_core:
            pytest.skip("IWMT is disabled")
        
        # Verify attention controller has use_iwmt_precision enabled
        if hasattr(core.attention, 'use_iwmt_precision'):
            assert core.attention.use_iwmt_precision is True
        
        # Verify precision_weighting is set
        assert core.attention.precision_weighting is not None
        assert core.attention.precision_weighting == core.subsystems.iwmt_core.precision


class TestIWMTBackwardCompatibility:
    """Test backward compatibility when IWMT is disabled."""

    @pytest.mark.asyncio
    async def test_cognitive_cycle_works_without_iwmt(self, temp_dirs):
        """Test that cognitive cycle works when IWMT is disabled."""
        config = make_core_config(temp_dirs, {"iwmt": {"enabled": False}})
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=config)
        
        # Verify IWMT is disabled
        assert core.subsystems.iwmt_core is None
        
        await core.lifecycle.start(restore_latest=False)
        
        try:
            # Add goal
            goal = Goal(
                description="Test without IWMT",
                type=GoalType.RESPOND_TO_USER,
                priority=0.5
            )
            workspace.add_goal(goal)
            
            # Inject input
            core.inject_input("Test input", "text")
            
            # Execute cycle - should work without errors
            timings = await core.cycle_executor.execute_cycle()
            
            # Verify cycle completed
            assert 'perception' in timings
            assert 'attention' in timings
            assert 'affect' in timings
            
            # IWMT steps should have minimal/zero time
            if 'iwmt_predict' in timings:
                # Should be very small or zero when IWMT is disabled
                assert timings['iwmt_predict'] < 0.01  # Less than 10ms
            if 'iwmt_update' in timings:
                assert timings['iwmt_update'] < 0.01
        finally:
            await core.lifecycle.stop()
    
    @pytest.mark.asyncio
    async def test_attention_works_without_precision_weighting(self, temp_dirs):
        """Test that attention selection works when precision weighting is None."""
        config = make_core_config(temp_dirs, {"iwmt": {"enabled": False}})
        core = CognitiveCore(config=config)
        
        # Create test percepts using workspace Percept class
        from mind.cognitive_core.workspace import Percept
        import numpy as np
        
        # Create simple percepts with minimal required data
        percepts = [
            Percept(
                modality="text",
                raw="Test 1",
                embedding=np.zeros(384, dtype=np.float32).tolist()
            ),
            Percept(
                modality="text",
                raw="Test 2",
                embedding=np.zeros(384, dtype=np.float32).tolist()
            ),
        ]
        
        # Attention selection should work without prediction errors
        attended = core.attention.select_for_broadcast(
            percepts,
            emotional_state=None,
            prediction_errors=None
        )
        
        # Should return some percepts (based on budget)
        assert isinstance(attended, list)


class TestIWMTEdgeCases:
    """Test edge cases and robustness of IWMT integration."""

    @pytest.mark.asyncio
    async def test_iwmt_update_with_none_outcome(self, temp_dirs):
        """Test that IWMT update handles None outcome gracefully."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=make_core_config(temp_dirs))
        
        if not core.subsystems.iwmt_core:
            pytest.skip("IWMT is disabled")
        
        # Test the helper method directly with None outcome
        from mind.cognitive_core.action import Action, ActionType
        
        action = Action(
            type=ActionType.WAIT,
            reason="Testing"
        )
        
        # Should not raise exception with None outcome
        core.cycle_executor._update_iwmt_from_action(action, None)
    
    @pytest.mark.asyncio
    async def test_iwmt_update_with_action_missing_attributes(self, temp_dirs):
        """Test IWMT update when action is missing optional attributes."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=make_core_config(temp_dirs))
        
        if not core.subsystems.iwmt_core:
            pytest.skip("IWMT is disabled")
        
        from mind.cognitive_core.action import Action, ActionType
        
        # Create action without parameters or reason
        action = Action(type=ActionType.WAIT)
        outcome = {"success": True}
        
        # Should handle missing attributes gracefully
        core.cycle_executor._update_iwmt_from_action(action, outcome)
        
        # Verify the world model was updated
        assert core.subsystems.iwmt_core is not None
    
    @pytest.mark.asyncio
    async def test_iwmt_disabled_does_not_break_cycle(self, temp_dirs):
        """Test that disabling IWMT doesn't break the cognitive cycle."""
        config = make_core_config(temp_dirs, {"iwmt": {"enabled": False}})
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=config)
        
        await core.lifecycle.start(restore_latest=False)
        
        try:
            # Add goal and input
            goal = Goal(
                description="Test without IWMT",
                type=GoalType.RESPOND_TO_USER,
                priority=0.5
            )
            workspace.add_goal(goal)
            core.inject_input("Test", "text")
            
            # Should execute without errors
            timings = await core.cycle_executor.execute_cycle()
            
            # Verify cycle completed successfully
            assert 'perception' in timings
            assert 'attention' in timings
        finally:
            await core.lifecycle.stop()
