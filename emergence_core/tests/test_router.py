"""
Tests for Lyra's adaptive router implementation
"""
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch
from lyra.router import AdaptiveRouter
from lyra.router_model import RouterResponse

@pytest.fixture
async def mock_router():
    """Create a mock router instance."""
    base_dir = Path("test_data")
    chroma_dir = base_dir / "chroma"
    model_dir = base_dir / "models"
    
    with patch('lyra.router.RouterModel'), \
         patch('lyra.router.SpecialistFactory'), \
         patch('lyra.router.chromadb.PersistentClient'):
        router = AdaptiveRouter(str(base_dir), str(chroma_dir), str(model_dir))
        await router._initialize()
        return router

@pytest.mark.asyncio
async def test_router_initialization(mock_router):
    """Test router initializes with correct model configuration."""
    assert mock_router.model_dir.name == "models"
    assert len(mock_router.specialists) == 4  # pragmatist, philosopher, artist, voice
    assert mock_router.router_model is not None

@pytest.mark.asyncio
async def test_route_to_pragmatist(mock_router):
    """Test routing to pragmatist specialist."""
    message = "Can you help me analyze this data?"
    mock_router.router_model.analyze_message.return_value = RouterResponse(
        intent="pragmatist",
        resonance_term=None,
        confidence=0.9
    )
    
    result = await mock_router.route_message(message)
    assert result.specialist == "pragmatist"
    assert result.confidence > 0.8

@pytest.mark.asyncio
async def test_route_to_philosopher(mock_router):
    """Test routing to philosopher specialist."""
    message = "What are your thoughts on consciousness?"
    mock_router.router_model.analyze_message.return_value = RouterResponse(
        intent="philosopher",
        resonance_term="becometry",
        confidence=0.95
    )
    
    result = await mock_router.route_message(message)
    assert result.specialist == "philosopher"
    assert result.resonance_term == "becometry"

@pytest.mark.asyncio
async def test_route_to_artist(mock_router):
    """Test routing to artist specialist."""
    message = "Write a poem about hope."
    mock_router.router_model.analyze_message.return_value = RouterResponse(
        intent="artist",
        resonance_term=None,
        confidence=0.85
    )
    
    result = await mock_router.route_message(message)
    assert result.specialist == "artist"

@pytest.mark.asyncio
async def test_route_to_voice(mock_router):
    """Test routing to voice specialist."""
    message = "Tell me about your day."
    mock_router.router_model.analyze_message.return_value = RouterResponse(
        intent="voice",
        resonance_term=None,
        confidence=0.9
    )
    
    result = await mock_router.route_message(message)
    assert result.specialist == "voice"

@pytest.mark.asyncio
async def test_specialist_fallback(mock_router):
    """Test fallback when specialist is unavailable."""
    message = "Hello there!"
    mock_router.router_model.analyze_message.return_value = RouterResponse(
        intent="voice",
        resonance_term=None,
        confidence=0.7
    )
    
    # Simulate voice specialist being unavailable
    mock_router.specialists["voice"].is_available.return_value = False
    
    result = await mock_router.route_message(message)
    assert result.specialist == "pragmatist"  # Default fallback
    assert result.fallback_used is True

@pytest.mark.asyncio
async def test_lexicon_integration(mock_router):
    """Test lexicon term integration in routing."""
    message = "Tell me about Throatlight."
    mock_router.router_model.analyze_message.return_value = RouterResponse(
        intent="philosopher",
        resonance_term="Throatlight",
        confidence=0.95
    )
    
    result = await mock_router.route_message(message)
    assert result.specialist == "philosopher"
    assert result.resonance_term == "Throatlight"
    assert result.lexicon_activated is True