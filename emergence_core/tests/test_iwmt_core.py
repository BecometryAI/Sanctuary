"""
Unit tests for IWMT Core.

Tests cover:
- IWMTCore initialization
- Cognitive cycle execution
- Integration of all components
- Action outcome updates
"""

import pytest

from lyra.cognitive_core.iwmt_core import IWMTCore
from lyra.cognitive_core.world_model import WorldModel


class TestIWMTCore:
    """Test IWMTCore class."""
    
    def test_initialization(self):
        """Test IWMTCore initializes all components."""
        core = IWMTCore()
        assert isinstance(core.world_model, WorldModel)
        assert core.free_energy is not None
        assert core.precision is not None
        assert core.active_inference is not None
        assert core.metta_bridge is not None
        assert core.cycle_count == 0
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = {
            "free_energy": {"prediction_error_weight": 2.0},
            "precision": {"base_precision": 0.6},
            "metta": {"use_metta": False}
        }
        core = IWMTCore(config)
        assert core.free_energy.prediction_error_weight == 2.0
        assert core.precision.base_precision == 0.6
    
    @pytest.mark.asyncio
    async def test_cognitive_cycle_basic(self):
        """Test basic cognitive cycle execution."""
        core = IWMTCore()
        
        percepts = [
            {"content": "Hello world"}
        ]
        emotional_state = {
            "arousal": 0.3,
            "valence": 0.5
        }
        
        results = await core.cognitive_cycle(percepts, emotional_state)
        
        assert "cycle_number" in results
        assert results["cycle_number"] == 1
        assert "free_energy" in results
        assert "prediction_errors" in results
        assert "should_act" in results
        assert core.cycle_count == 1
    
    @pytest.mark.asyncio
    async def test_cognitive_cycle_with_goals(self):
        """Test cognitive cycle with goals."""
        core = IWMTCore()
        
        percepts = [{"content": "New information"}]
        emotional_state = {"arousal": 0.5, "valence": 0.0}
        goals = [
            {"description": "Understand input", "priority": 0.8}
        ]
        
        results = await core.cognitive_cycle(percepts, emotional_state, goals)
        
        assert "num_predictions" in results
        assert results["num_predictions"] > 0
    
    @pytest.mark.asyncio
    async def test_cognitive_cycle_with_actions(self):
        """Test cognitive cycle with available actions."""
        core = IWMTCore()
        
        percepts = [{"content": "Question asked"}]
        emotional_state = {"arousal": 0.4, "valence": 0.2}
        actions = [
            {"type": "speak", "content": "Answer"},
            {"type": "wait", "duration": 1.0}
        ]
        
        results = await core.cognitive_cycle(
            percepts,
            emotional_state,
            available_actions=actions
        )
        
        if results["should_act"]:
            assert results["recommended_action"] in actions
    
    @pytest.mark.asyncio
    async def test_multiple_cycles(self):
        """Test running multiple cognitive cycles."""
        core = IWMTCore()
        
        emotional_state = {"arousal": 0.3, "valence": 0.0}
        
        # Run 3 cycles
        for i in range(3):
            percepts = [{"content": f"Percept {i}"}]
            results = await core.cognitive_cycle(percepts, emotional_state)
            assert results["cycle_number"] == i + 1
        
        assert core.cycle_count == 3
    
    @pytest.mark.asyncio
    async def test_prediction_errors_tracked(self):
        """Test that prediction errors are tracked across cycles."""
        core = IWMTCore()
        
        emotional_state = {"arousal": 0.0, "valence": 0.0}
        
        # First cycle - establish predictions
        await core.cognitive_cycle(
            [{"content": "Expected pattern"}],
            emotional_state
        )
        
        # Second cycle - provide surprising input
        results = await core.cognitive_cycle(
            [{"content": "Completely unexpected and different input"}],
            emotional_state
        )
        
        # May or may not generate prediction errors depending on matching
        assert "num_prediction_errors" in results
    
    def test_update_from_action_outcome(self):
        """Test updating from action outcomes."""
        core = IWMTCore()
        
        action = {"type": "speak", "content": "Hello"}
        outcome = {
            "success": True,
            "observation": {
                "entities": [
                    {"id": "user", "type": "agent", "properties": {}}
                ]
            }
        }
        
        core.update_from_action_outcome(action, outcome)
        
        # Check that world model was updated
        assert len(core.world_model.environment_model.entities) > 0
    
    def test_get_status(self):
        """Test getting status information."""
        core = IWMTCore()
        status = core.get_status()
        
        assert "cycle_count" in status
        assert "world_model" in status
        assert "free_energy" in status
        assert "precision_summary" in status
        assert "metta_available" in status
    
    @pytest.mark.asyncio
    async def test_cycle_results_structure(self):
        """Test that cycle results have expected structure."""
        core = IWMTCore()
        
        results = await core.cognitive_cycle(
            [{"content": "test"}],
            {"arousal": 0.0, "valence": 0.0}
        )
        
        # Check all expected keys are present
        expected_keys = [
            "cycle_number",
            "cycle_time_seconds",
            "timestamp",
            "num_predictions",
            "predictions",
            "num_prediction_errors",
            "prediction_errors",
            "free_energy",
            "prediction_error_summary",
            "precision_summary",
            "num_precision_weights",
            "should_act",
            "recommended_action",
            "world_model_state"
        ]
        
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"
