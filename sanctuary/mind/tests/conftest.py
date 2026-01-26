"""
Test configuration and fixtures
"""
import pytest
import asyncio
from pathlib import Path
import shutil
import json
import chromadb
from unittest.mock import Mock, patch
from typing import Dict, Any

@pytest.fixture
def test_data_dir():
    """Create and manage test data directory."""
    test_dir = Path("test_data")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (test_dir / "chroma").mkdir(exist_ok=True)
    (test_dir / "cache").mkdir(exist_ok=True)
    (test_dir / "logs").mkdir(exist_ok=True)
    (test_dir / "models").mkdir(exist_ok=True)
    
    yield test_dir
    
    # Cleanup after tests
    shutil.rmtree(test_dir)

@pytest.fixture
def mock_chroma_collection():
    """Create a mock ChromaDB collection."""
    mock_collection = Mock()
    mock_collection.query.return_value = {
        'documents': [[{
            'text': 'Test document content',
            'metadata': {'source': 'test'}
        }]]
    }
    return mock_collection

@pytest.fixture
def test_cache(test_data_dir):
    """Create a test RAG cache."""
    from mind.rag_cache import RAGCache
    cache = RAGCache(test_data_dir / "cache", max_size=10)
    yield cache
    cache._cleanup_expired()  # Clean up after tests

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        'base_dir': 'test_data',
        'chroma_dir': 'test_data/chroma',
        'model_dir': 'test_data/models',
        'cache_dir': 'test_data/cache',
        'log_dir': 'test_data/logs'
    }

@pytest.fixture
def test_logger(test_data_dir):
    """Create a test logger."""
    from mind.cognitive_logger import CognitiveLogger
    logger = CognitiveLogger(test_data_dir / "logs")
    yield logger
    logger.close()  # Release file locks before cleanup