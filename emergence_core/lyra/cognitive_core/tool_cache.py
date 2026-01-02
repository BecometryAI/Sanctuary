"""
Tool Cache: Cache tool results to avoid redundant calls.

This module implements result caching for external tool invocations.
By caching results based on input hash, the system can avoid expensive
redundant calls to external services like web search or computation engines.

The tool cache is responsible for:
- Caching tool results by input hash
- Configurable TTL (time-to-live) per tool type
- LRU (Least Recently Used) eviction policy
- Cache persistence to disk
- Invalidation on user request
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """
    Single cache entry for a tool result.
    
    Attributes:
        key: Cache key (hash of inputs)
        tool_name: Name of the tool
        inputs: Input parameters used
        result: Cached result
        created_at: When entry was created
        last_accessed: When entry was last accessed
        access_count: Number of times entry was accessed
        ttl: Time-to-live in seconds
    """
    key: str
    tool_name: str
    inputs: Dict[str, Any]
    result: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: float = 3600.0  # 1 hour default
    
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl
    
    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.now()
        self.access_count += 1


class ToolCache:
    """
    LRU cache for tool results.
    
    The ToolCache implements result caching with the following features:
    - Input-based caching using hash of parameters
    - Configurable TTL per tool type
    - LRU eviction when cache size limit is reached
    - Optional disk persistence for cache survival across restarts
    - Cache statistics and hit rate tracking
    
    Example:
        ```python
        cache = ToolCache(max_size=1000, default_ttl=3600)
        
        # Set TTL for specific tool
        cache.set_ttl("web_search", 1800)  # 30 minutes
        
        # Check cache before tool execution
        result = cache.get("web_search", {"query": "Python"})
        if result is None:
            result = await tool_registry.execute_tool("web_search", query="Python")
            cache.set("web_search", {"query": "Python"}, result)
        ```
    
    Attributes:
        max_size: Maximum number of entries in cache
        default_ttl: Default time-to-live in seconds
        tool_ttls: Per-tool TTL overrides
        cache: Ordered dictionary for LRU eviction
        stats: Cache statistics
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the tool cache.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default TTL in seconds
            cache_dir: Optional directory for cache persistence
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache_dir = cache_dir
        self.tool_ttls: Dict[str, float] = {}
        
        # Use OrderedDict for LRU
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
            "invalidations": 0
        }
        
        logger.info(f"💾 ToolCache initialized (max_size: {max_size}, default_ttl: {default_ttl}s)")
        
        # Load from disk if cache_dir provided
        if cache_dir:
            self._load_from_disk()
    
    def _compute_key(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """
        Compute cache key from tool name and inputs.
        
        Args:
            tool_name: Name of the tool
            inputs: Input parameters dictionary
            
        Returns:
            Hash-based cache key
        """
        # Create deterministic string representation
        input_str = json.dumps(inputs, sort_keys=True)
        combined = f"{tool_name}:{input_str}"
        
        # Hash for key
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def set_ttl(self, tool_name: str, ttl: float) -> None:
        """
        Set TTL for specific tool type.
        
        Args:
            tool_name: Tool name
            ttl: Time-to-live in seconds
        """
        self.tool_ttls[tool_name] = ttl
        logger.debug(f"Set TTL for '{tool_name}': {ttl}s")
    
    def get(self, tool_name: str, inputs: Dict[str, Any]) -> Optional[Any]:
        """
        Get cached result if available.
        
        Args:
            tool_name: Name of the tool
            inputs: Input parameters
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._compute_key(tool_name, inputs)
        
        if key not in self.cache:
            self.stats["misses"] += 1
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if entry.is_expired():
            self.stats["expirations"] += 1
            self.stats["misses"] += 1
            del self.cache[key]
            logger.debug(f"Cache entry expired: {tool_name}")
            return None
        
        # Update access info (LRU)
        entry.touch()
        self.cache.move_to_end(key)
        
        self.stats["hits"] += 1
        logger.debug(f"Cache hit: {tool_name}")
        
        return entry.result
    
    def set(self, tool_name: str, inputs: Dict[str, Any], result: Any) -> None:
        """
        Store result in cache.
        
        Args:
            tool_name: Name of the tool
            inputs: Input parameters
            result: Result to cache
        """
        key = self._compute_key(tool_name, inputs)
        
        # Determine TTL for this tool
        ttl = self.tool_ttls.get(tool_name, self.default_ttl)
        
        # Create entry
        entry = CacheEntry(
            key=key,
            tool_name=tool_name,
            inputs=inputs,
            result=result,
            ttl=ttl
        )
        
        # Add to cache
        self.cache[key] = entry
        self.cache.move_to_end(key)
        
        # Evict if over size limit
        if len(self.cache) > self.max_size:
            self._evict_lru()
        
        logger.debug(f"Cached result: {tool_name}")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self.cache:
            key, entry = self.cache.popitem(last=False)
            self.stats["evictions"] += 1
            logger.debug(f"Evicted LRU entry: {entry.tool_name}")
    
    def invalidate(self, tool_name: Optional[str] = None) -> int:
        """
        Invalidate cache entries.
        
        Args:
            tool_name: If provided, only invalidate entries for this tool.
                      If None, clear entire cache.
            
        Returns:
            Number of entries invalidated
        """
        if tool_name is None:
            count = len(self.cache)
            self.cache.clear()
            self.stats["invalidations"] += count
            logger.info(f"🗑️ Invalidated entire cache ({count} entries)")
            return count
        
        # Invalidate specific tool entries
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if entry.tool_name == tool_name
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        count = len(keys_to_remove)
        self.stats["invalidations"] += count
        logger.info(f"🗑️ Invalidated {count} entries for tool: {tool_name}")
        
        return count
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.
        
        Returns:
            Number of entries removed
        """
        keys_to_remove = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in keys_to_remove:
            del self.cache[key]
        
        count = len(keys_to_remove)
        self.stats["expirations"] += count
        
        if count > 0:
            logger.info(f"🧹 Cleaned up {count} expired entries")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": hit_rate,
            "evictions": self.stats["evictions"],
            "expirations": self.stats["expirations"],
            "invalidations": self.stats["invalidations"]
        }
    
    def get_entries_by_tool(self, tool_name: str) -> List[CacheEntry]:
        """
        Get all cache entries for a specific tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            List of cache entries
        """
        return [
            entry for entry in self.cache.values()
            if entry.tool_name == tool_name
        ]
    
    def _save_to_disk(self) -> None:
        """Save cache to disk."""
        if not self.cache_dir:
            return
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / "tool_cache.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    "cache": dict(self.cache),
                    "stats": self.stats,
                    "tool_ttls": self.tool_ttls
                }, f)
            logger.debug(f"💾 Saved cache to {cache_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "tool_cache.pkl"
        
        if not cache_file.exists():
            return
        
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            self.cache = OrderedDict(data["cache"])
            self.stats = data["stats"]
            self.tool_ttls = data["tool_ttls"]
            
            # Clean up expired entries on load
            self.cleanup_expired()
            
            logger.info(f"📂 Loaded cache from {cache_file} ({len(self.cache)} entries)")
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}")
    
    def __del__(self):
        """Save cache on destruction if cache_dir is set."""
        if self.cache_dir:
            self._save_to_disk()
