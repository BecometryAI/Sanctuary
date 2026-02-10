"""Shared fixtures for integration tests."""

import pytest
import asyncio


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def sanctuary_api():
    """Provide started SanctuaryAPI instance."""
    from sanctuary import SanctuaryAPI

    api = SanctuaryAPI()
    await api.start()
    
    yield api
    
    await api.stop()


@pytest.fixture
async def cognitive_core():
    """
    Provide a fully initialized CognitiveCore instance for testing.
    
    Automatically starts the core before the test and stops after.
    """
    # Import here to avoid loading on conftest import
    from mind.cognitive_core.core import CognitiveCore
    from mind.cognitive_core.workspace import GlobalWorkspace
    
    workspace = GlobalWorkspace()
    config = {
        "cycle_rate_hz": 10,
        "checkpointing": {"enabled": False},
        "input_llm": {"use_real_model": False},
        "output_llm": {"use_real_model": False},
    }
    
    core = CognitiveCore(workspace=workspace, config=config)
    
    # Start core
    start_task = asyncio.create_task(core.start())
    await asyncio.sleep(0.5)  # Wait for initialization
    
    try:
        yield core
    finally:
        # Stop core
        await core.stop()
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass


@pytest.fixture
def workspace():
    """Provide a fresh GlobalWorkspace instance."""
    from mind.cognitive_core.workspace import GlobalWorkspace
    return GlobalWorkspace()
