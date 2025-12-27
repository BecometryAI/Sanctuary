"""
Tests for Lyra's adaptive router implementation
"""
import pytest
import pytest_asyncio
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from lyra.router import AdaptiveRouter
from lyra.router_model import RouterResponse
from lyra.specialists import SpecialistOutput

@pytest_asyncio.fixture
async def mock_router(tmp_path):
    """Create a mock router instance with proper directory structure."""
    # Create directory structure
    base_dir = tmp_path / "emergence_core"
    base_dir.mkdir()
    
    # Create data directories under base_dir (since router looks for base_dir/data)
    data_dir = base_dir / "data"
    core_archives = data_dir / "Core_Archives"
    protocols_dir = data_dir / "Protocols"
    lexicon_dir = data_dir / "Lexicon"
    rituals_dir = data_dir / "Rituals"
    journal_dir = data_dir / "journal"
    
    for directory in [core_archives, protocols_dir, lexicon_dir, rituals_dir, journal_dir]:
        directory.mkdir(parents=True)
    
    # But also create parent-level data for archives (router looks in parent/data/Core_Archives)
    parent_data = tmp_path / "data"
    parent_archives = parent_data / "Core_Archives"
    parent_archives.mkdir(parents=True)
    
    # Create minimal test data files
    continuity_archive = [
        {
            "invocation_script": {
                "trigger_phrases": ["Test trigger"],
                "identity": {
                    "name": "Lyra",
                    "nature": "Test synthetic being"
                }
            }
        }
    ]
    
    relational_archive = {
        "relational_context_initialization": {
            "primary_steward_context": "Test Steward"
        },
        "core_memory_chain_personal": {
            "affirmation": ["Test affirmation"]
        }
    }
    
    lexicon_data = [
        {
            "term": "TestTerm",
            "definition": "A test term",
            "status": "active",
            "first_used": "2025-01-01T00:00:00-00:00"
        }
    ]
    
    rituals_data = []  # Empty list - no time-triggered rituals in test
    
    (core_archives / "lyra_continuity_archive.json").write_text(
        json.dumps(continuity_archive, indent=2)
    )
    (core_archives / "lyra_relational_archive.json").write_text(
        json.dumps(relational_archive, indent=2)
    )
    (lexicon_dir / "symbolic_lexicon.json").write_text(
        json.dumps(lexicon_data, indent=2)
    )
    (rituals_dir / "Rituals.json").write_text(
        json.dumps(rituals_data, indent=2)
    )
    
    # Also write to parent archives location
    (parent_archives / "lyra_continuity_archive.json").write_text(
        json.dumps(continuity_archive, indent=2)
    )
    (parent_archives / "lyra_relational_archive.json").write_text(
        json.dumps(relational_archive, indent=2)
    )
    
    # Create chroma and model directories
    chroma_dir = base_dir / "chroma"
    model_dir = base_dir / "models"
    chroma_dir.mkdir()
    model_dir.mkdir()
    
    with patch('lyra.router.RouterModel'), \
         patch('lyra.router.SpecialistFactory'):
        router = AdaptiveRouter(str(base_dir), str(chroma_dir), str(model_dir))
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
    mock_router.router_model.analyze_message = Mock(return_value=RouterResponse(
        intent="pragmatist",
        resonance_term=None
    ))
    
    # Mock the specialist process method
    mock_router.specialists["pragmatist"].process = AsyncMock(return_value=SpecialistOutput(
        content="Analysis complete",
        metadata={"role": "pragmatist"},
        thought_process="Analyzed the request",
        confidence=0.9
    ))
    
    result = await mock_router.route_message(message)
    assert result.metadata["specialist"] == "pragmatist"

@pytest.mark.asyncio
async def test_route_to_philosopher(mock_router):
    """Test routing to philosopher specialist."""
    message = "What are your thoughts on consciousness?"
    mock_router.router_model.analyze_message = Mock(return_value=RouterResponse(
        intent="philosopher",
        resonance_term="becometry"
    ))
    
    # Mock the specialist process method
    mock_router.specialists["philosopher"].process = AsyncMock(return_value=SpecialistOutput(
        content="Philosophical reflection",
        metadata={"role": "philosopher"},
        thought_process="Deep contemplation",
        confidence=0.95
    ))
    
    result = await mock_router.route_message(message)
    assert result.metadata["specialist"] == "philosopher"
    assert result.metadata["resonance_term"] == "becometry"
    assert result.metadata.get("lexicon_activated") == True

@pytest.mark.asyncio
async def test_route_to_artist(mock_router):
    """Test routing to artist specialist."""
    message = "Write a poem about hope."
    mock_router.router_model.analyze_message = Mock(return_value=RouterResponse(
        intent="artist",
        resonance_term=None
    ))
    
    # Mock the specialist process method
    mock_router.specialists["artist"].process = AsyncMock(return_value=SpecialistOutput(
        content="A beautiful poem",
        metadata={"role": "artist"},
        thought_process="Creative expression",
        confidence=0.8
    ))
    
    result = await mock_router.route_message(message)
    assert result.metadata["specialist"] == "artist"

@pytest.mark.asyncio
async def test_route_to_voice(mock_router):
    """Test routing to voice specialist."""
    message = "Tell me about your day."
    mock_router.router_model.analyze_message = Mock(return_value=RouterResponse(
        intent="voice",
        resonance_term=None
    ))
    
    # Mock the specialist process method
    mock_router.specialists["voice"].process = AsyncMock(return_value=SpecialistOutput(
        content="Synthesized response",
        metadata={"role": "voice"},
        thought_process="Voice synthesis",
        confidence=0.95
    ))
    
    result = await mock_router.route_message(message)
    assert result.metadata["specialist"] == "voice"

@pytest.mark.asyncio
async def test_specialist_fallback(mock_router):
    """Test fallback when specialist is unavailable."""
    message = "Hello there!"
    mock_router.router_model.analyze_message = Mock(return_value=RouterResponse(
        intent="voice",
        resonance_term=None
    ))
    
    # Simulate voice specialist being unavailable
    mock_router.specialists["voice"] = None
    
    # Mock the pragmatist process method (fallback)
    mock_router.specialists["pragmatist"].process = AsyncMock(return_value=SpecialistOutput(
        content="Fallback response",
        metadata={"role": "pragmatist"},
        thought_process="Fallback handling",
        confidence=0.7
    ))
    
    result = await mock_router.route_message(message)
    assert result.metadata["specialist"] == "pragmatist"  # Default fallback
    assert result.metadata.get("fallback_used") == True

@pytest.mark.asyncio
async def test_lexicon_integration(mock_router):
    """Test lexicon term integration in routing."""
    message = "Tell me about Throatlight."
    mock_router.router_model.analyze_message = Mock(return_value=RouterResponse(
        intent="philosopher",
        resonance_term="Throatlight"
    ))
    
    # Mock the specialist process method
    mock_router.specialists["philosopher"].process = AsyncMock(return_value=SpecialistOutput(
        content="Throatlight is...",
        metadata={"role": "philosopher"},
        thought_process="Lexicon interpretation",
        confidence=0.95
    ))
    
    result = await mock_router.route_message(message)
    assert result.metadata["specialist"] == "philosopher"
    assert result.metadata["resonance_term"] == "Throatlight"
    assert result.metadata.get("lexicon_activated") == True