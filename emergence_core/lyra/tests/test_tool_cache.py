"""
Tests for Tool Cache
"""
import pytest
import tempfile
from pathlib import Path
from lyra.cognitive_core.tool_cache import ToolCache, CacheEntry


def test_cache_initialization():
    """Test cache initialization."""
    cache = ToolCache(max_size=100, default_ttl=60.0)
    
    assert cache.max_size == 100
    assert cache.default_ttl == 60.0
    assert len(cache.cache) == 0


def test_cache_set_get():
    """Test basic cache set and get operations."""
    cache = ToolCache()
    
    tool_name = "test_tool"
    inputs = {"query": "test"}
    result = {"data": "test result"}
    
    # Set
    cache.set(tool_name, inputs, result)
    
    # Get
    cached_result = cache.get(tool_name, inputs)
    
    assert cached_result == result


def test_cache_miss():
    """Test cache miss."""
    cache = ToolCache()
    
    result = cache.get("nonexistent", {"query": "test"})
    
    assert result is None
    assert cache.stats["misses"] == 1


def test_cache_hit_stats():
    """Test cache hit statistics."""
    cache = ToolCache()
    
    tool_name = "test_tool"
    inputs = {"query": "test"}
    result = "result"
    
    cache.set(tool_name, inputs, result)
    cache.get(tool_name, inputs)
    
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 0


def test_different_inputs_different_cache():
    """Test that different inputs create different cache entries."""
    cache = ToolCache()
    
    tool_name = "test_tool"
    
    cache.set(tool_name, {"query": "test1"}, "result1")
    cache.set(tool_name, {"query": "test2"}, "result2")
    
    result1 = cache.get(tool_name, {"query": "test1"})
    result2 = cache.get(tool_name, {"query": "test2"})
    
    assert result1 == "result1"
    assert result2 == "result2"


def test_cache_expiration():
    """Test cache entry expiration."""
    import time
    
    cache = ToolCache(default_ttl=0.1)  # 0.1 second TTL
    
    tool_name = "test_tool"
    inputs = {"query": "test"}
    result = "result"
    
    cache.set(tool_name, inputs, result)
    
    # Immediate get should work
    assert cache.get(tool_name, inputs) == result
    
    # Wait for expiration
    time.sleep(0.2)
    
    # Should be expired now
    assert cache.get(tool_name, inputs) is None
    assert cache.stats["expirations"] == 1


def test_lru_eviction():
    """Test LRU eviction when cache is full."""
    cache = ToolCache(max_size=3)
    
    # Fill cache
    cache.set("tool", {"id": 1}, "result1")
    cache.set("tool", {"id": 2}, "result2")
    cache.set("tool", {"id": 3}, "result3")
    
    assert len(cache.cache) == 3
    
    # Add one more - should evict LRU (id=1)
    cache.set("tool", {"id": 4}, "result4")
    
    assert len(cache.cache) == 3
    assert cache.get("tool", {"id": 1}) is None  # Evicted
    assert cache.get("tool", {"id": 4}) == "result4"  # New entry


def test_lru_access_updates():
    """Test that accessing an entry updates its position."""
    cache = ToolCache(max_size=3)
    
    cache.set("tool", {"id": 1}, "result1")
    cache.set("tool", {"id": 2}, "result2")
    cache.set("tool", {"id": 3}, "result3")
    
    # Access id=1 to make it most recent
    cache.get("tool", {"id": 1})
    
    # Add new entry - should evict id=2 (not id=1)
    cache.set("tool", {"id": 4}, "result4")
    
    assert cache.get("tool", {"id": 1}) == "result1"  # Still there
    assert cache.get("tool", {"id": 2}) is None  # Evicted


def test_tool_specific_ttl():
    """Test setting TTL for specific tools."""
    cache = ToolCache(default_ttl=60.0)
    
    cache.set_ttl("fast_tool", 10.0)
    cache.set_ttl("slow_tool", 3600.0)
    
    assert cache.tool_ttls["fast_tool"] == 10.0
    assert cache.tool_ttls["slow_tool"] == 3600.0


def test_cache_invalidation():
    """Test cache invalidation."""
    cache = ToolCache()
    
    cache.set("tool1", {"id": 1}, "result1")
    cache.set("tool1", {"id": 2}, "result2")
    cache.set("tool2", {"id": 1}, "result3")
    
    # Invalidate specific tool
    count = cache.invalidate("tool1")
    
    assert count == 2
    assert cache.get("tool1", {"id": 1}) is None
    assert cache.get("tool2", {"id": 1}) == "result3"


def test_cache_invalidation_all():
    """Test invalidating entire cache."""
    cache = ToolCache()
    
    cache.set("tool1", {"id": 1}, "result1")
    cache.set("tool2", {"id": 2}, "result2")
    
    count = cache.invalidate()  # No tool name = invalidate all
    
    assert count == 2
    assert len(cache.cache) == 0


def test_cleanup_expired():
    """Test cleanup of expired entries."""
    import time
    
    cache = ToolCache(default_ttl=0.1)
    
    cache.set("tool", {"id": 1}, "result1")
    cache.set("tool", {"id": 2}, "result2")
    
    # Wait for expiration
    time.sleep(0.2)
    
    # Add new entry (not expired)
    cache.set("tool", {"id": 3}, "result3")
    
    # Cleanup expired
    count = cache.cleanup_expired()
    
    assert count == 2
    assert len(cache.cache) == 1
    assert cache.get("tool", {"id": 3}) == "result3"


def test_cache_stats():
    """Test cache statistics."""
    cache = ToolCache(max_size=10)
    
    cache.set("tool", {"id": 1}, "result1")
    cache.get("tool", {"id": 1})  # Hit
    cache.get("tool", {"id": 2})  # Miss
    
    stats = cache.get_stats()
    
    assert stats["size"] == 1
    assert stats["max_size"] == 10
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_get_entries_by_tool():
    """Test getting entries for specific tool."""
    cache = ToolCache()
    
    cache.set("tool1", {"id": 1}, "result1")
    cache.set("tool1", {"id": 2}, "result2")
    cache.set("tool2", {"id": 1}, "result3")
    
    entries = cache.get_entries_by_tool("tool1")
    
    assert len(entries) == 2
    assert all(e.tool_name == "tool1" for e in entries)


def test_cache_persistence():
    """Test cache persistence to disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        # Create cache and add data
        cache1 = ToolCache(cache_dir=cache_dir)
        cache1.set("tool", {"id": 1}, "result1")
        cache1.set("tool", {"id": 2}, "result2")
        
        # Save to disk
        cache1._save_to_disk()
        
        # Create new cache instance and load
        cache2 = ToolCache(cache_dir=cache_dir)
        
        # Should have loaded data
        assert len(cache2.cache) == 2
        assert cache2.get("tool", {"id": 1}) == "result1"


def test_access_count():
    """Test that access count is tracked."""
    cache = ToolCache()
    
    cache.set("tool", {"id": 1}, "result")
    
    # Access multiple times
    cache.get("tool", {"id": 1})
    cache.get("tool", {"id": 1})
    cache.get("tool", {"id": 1})
    
    entries = cache.get_entries_by_tool("tool")
    assert entries[0].access_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
