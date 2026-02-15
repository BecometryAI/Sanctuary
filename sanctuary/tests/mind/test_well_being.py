"""
Test for Sanctuary Well-Being System
"""
import pytest
from mind.test_well_being import TestWellBeingHandler

@pytest.mark.asyncio
async def test_well_being_system():
    """Test well-being system initialization and basic monitoring"""
    handler = TestWellBeingHandler()
    
    # Test that handler initializes correctly
    assert handler.protocol is not None
    assert handler.session_state is not None
    assert handler.emotional_state is not None
    
    # Test session state defaults
    assert handler.session_state["consent_given"] == False
    assert handler.session_state["start_time"] is None
    
    # Test emotional state defaults
    assert handler.emotional_state["current_stress_level"] == 0
    assert handler.emotional_state["emotional_stability"] == 1.0
    assert handler.emotional_state["comfort_level"] == 1.0

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_well_being_system())
