"""
Tests for RAG caching system
"""
import pytest
import asyncio
import time
from lyra.rag_cache import RAGCache

@pytest.mark.asyncio
async def test_cache_set_get(test_cache):
    """Test basic cache set and get operations."""
    test_data = {"key": "value"}
    test_cache.set("test_query", test_data)
    
    result = test_cache.get("test_query")
    assert result == test_data

@pytest.mark.asyncio
async def test_cache_ttl(test_cache):
    """Test cache entry expiration."""
    test_data = {"key": "value"}
    test_cache.set("test_query", test_data, ttl=1)  # 1 second TTL
    
    # Check immediately
    assert test_cache.get("test_query") == test_data
    
    # Wait for expiration
    await asyncio.sleep(1.1)
    assert test_cache.get("test_query") is None

@pytest.mark.asyncio
async def test_cache_max_size(test_cache):
    """Test cache size limitation."""
    # Fill cache to max size
    for i in range(11):  # Cache size is 10
        test_cache.set(f"query_{i}", {"data": i})
    
    # First entry should be evicted
    assert test_cache.get("query_0") is None
    assert test_cache.get("query_10") is not None

@pytest.mark.asyncio
async def test_cache_persistence(test_data_dir):
    """Test cache persistence to disk."""
    # Create cache and add data
    cache1 = RAGCache(test_data_dir / "cache")
    cache1.set("test_query", {"key": "value"})
    
    # Create new cache instance and verify data
    cache2 = RAGCache(test_data_dir / "cache")
    assert cache2.get("test_query") == {"key": "value"}