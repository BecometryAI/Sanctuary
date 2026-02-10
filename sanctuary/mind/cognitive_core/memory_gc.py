"""
Memory Garbage Collection System

This module implements periodic cleanup of low-significance memories to prevent
unbounded growth and maintain system performance over long-term operation.

The garbage collector employs multiple strategies:
- Significance-based removal: Remove memories below threshold
- Age-based decay: Apply time decay to significance scores
- Redundancy detection: Identify and remove near-duplicate memories
- Capacity-based pruning: Enforce maximum memory capacity limits

Safety mechanisms ensure important memories are never accidentally removed.

Author: Sanctuary Emergence Team
Date: January 2026
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Set
from uuid import UUID
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CollectionStats:
    """Statistics from a garbage collection run.
    
    Attributes:
        timestamp: When collection occurred
        memories_analyzed: Total number of memories examined
        memories_removed: Number of memories deleted
        bytes_freed: Estimated bytes freed (if calculable)
        duration_seconds: How long collection took
        removal_reasons: Breakdown of why memories were removed
        avg_significance_before: Average significance before collection
        avg_significance_after: Average significance after collection
    """
    timestamp: datetime
    memories_analyzed: int
    memories_removed: int
    bytes_freed: int
    duration_seconds: float
    removal_reasons: Dict[str, int] = field(default_factory=dict)
    avg_significance_before: float = 0.0
    avg_significance_after: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "memories_analyzed": self.memories_analyzed,
            "memories_removed": self.memories_removed,
            "bytes_freed": self.bytes_freed,
            "duration_seconds": self.duration_seconds,
            "removal_reasons": self.removal_reasons,
            "avg_significance_before": self.avg_significance_before,
            "avg_significance_after": self.avg_significance_after,
        }


@dataclass
class MemoryHealthReport:
    """Analysis of memory system health.
    
    Attributes:
        total_memories: Total count of memories in system
        total_size_mb: Estimated total size in megabytes
        avg_significance: Average significance score
        significance_distribution: Histogram of significance scores
        oldest_memory_age_days: Age of oldest memory in days
        newest_memory_age_days: Age of newest memory in days
        estimated_duplicates: Estimated count of near-duplicate memories
        needs_collection: Whether collection is recommended
        recommended_threshold: Suggested significance threshold for GC
    """
    total_memories: int
    total_size_mb: float
    avg_significance: float
    significance_distribution: Dict[str, int]
    oldest_memory_age_days: float
    newest_memory_age_days: float
    estimated_duplicates: int
    needs_collection: bool
    recommended_threshold: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_memories": self.total_memories,
            "total_size_mb": self.total_size_mb,
            "avg_significance": self.avg_significance,
            "significance_distribution": self.significance_distribution,
            "oldest_memory_age_days": self.oldest_memory_age_days,
            "newest_memory_age_days": self.newest_memory_age_days,
            "estimated_duplicates": self.estimated_duplicates,
            "needs_collection": self.needs_collection,
            "recommended_threshold": self.recommended_threshold,
        }


# ============================================================================
# MEMORY GARBAGE COLLECTOR
# ============================================================================

class MemoryGarbageCollector:
    """Garbage collector for memory system.
    
    Manages periodic cleanup of low-significance memories to prevent
    unbounded growth while preserving important memories and system integrity.
    
    Attributes:
        memory_store: Reference to ChromaDB or vector storage
        config: Configuration dictionary
        collection_history: List of past collection statistics
        scheduled_task: Background task for automatic collection
        is_running: Whether scheduled collection is active
    """
    
    def __init__(
        self,
        memory_store: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the garbage collector.
        
        Args:
            memory_store: ChromaDB client or vector storage interface
            config: Configuration dictionary with GC parameters
        """
        self.memory_store = memory_store
        self.config = config or {}
        
        # Extract configuration with defaults
        self.significance_threshold = self.config.get("significance_threshold", 0.1)
        self.decay_rate_per_day = self.config.get("decay_rate_per_day", 0.01)
        self.duplicate_similarity_threshold = self.config.get("duplicate_similarity_threshold", 0.95)
        self.max_memory_capacity = self.config.get("max_memory_capacity", 10000)
        self.min_memories_per_category = self.config.get("min_memories_per_category", 10)
        self.preserve_tags = set(self.config.get("preserve_tags", ["important", "pinned", "charter_related"]))
        self.aggressive_mode = self.config.get("aggressive_mode", False)
        self.recent_memory_protection_hours = self.config.get("recent_memory_protection_hours", 24)
        self.max_removal_per_run = self.config.get("max_removal_per_run", 100)
        
        # State
        self.collection_history: List[CollectionStats] = []
        self.scheduled_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info(
            f"MemoryGarbageCollector initialized: "
            f"threshold={self.significance_threshold}, "
            f"decay_rate={self.decay_rate_per_day}/day, "
            f"max_capacity={self.max_memory_capacity}"
        )
    
    async def collect(
        self,
        threshold: Optional[float] = None,
        dry_run: bool = False
    ) -> CollectionStats:
        """Execute garbage collection.
        
        Removes memories below significance threshold while preserving
        protected memories. Applies all collection strategies.
        
        Args:
            threshold: Custom significance threshold (overrides config)
            dry_run: If True, analyze but don't actually remove memories
            
        Returns:
            CollectionStats with results of the collection
        """
        start_time = time.time()
        threshold = threshold if threshold is not None else self.significance_threshold
        
        logger.info(
            f"Starting memory garbage collection "
            f"(threshold={threshold}, dry_run={dry_run})"
        )
        
        try:
            # Get all memories from storage
            all_memories = await self._get_all_memories()
            
            if not all_memories:
                logger.info("No memories found, nothing to collect")
                return CollectionStats(
                    timestamp=datetime.now(timezone.utc),
                    memories_analyzed=0,
                    memories_removed=0,
                    bytes_freed=0,
                    duration_seconds=time.time() - start_time,
                )
            
            # Calculate statistics before collection
            avg_sig_before = sum(m.get("significance", 5) for m in all_memories) / len(all_memories)
            
            # Apply collection strategies
            to_remove = await self._identify_removal_candidates(
                all_memories,
                threshold
            )
            
            # Execute removal (or simulate if dry_run)
            bytes_freed = 0
            if to_remove and not dry_run:
                bytes_freed = await self._remove_memories(to_remove)
            
            # Calculate statistics after collection
            remaining_memories = [m for m in all_memories if m["id"] not in {mem["id"] for mem in to_remove}]
            avg_sig_after = (
                sum(m.get("significance", 5) for m in remaining_memories) / len(remaining_memories)
                if remaining_memories else 0.0
            )
            
            # Build statistics
            stats = CollectionStats(
                timestamp=datetime.now(timezone.utc),
                memories_analyzed=len(all_memories),
                memories_removed=len(to_remove),
                bytes_freed=bytes_freed,
                duration_seconds=time.time() - start_time,
                removal_reasons=self._count_removal_reasons(to_remove),
                avg_significance_before=avg_sig_before,
                avg_significance_after=avg_sig_after,
            )
            
            # Store in history
            self.collection_history.append(stats)
            
            # Keep only last 100 stats
            if len(self.collection_history) > 100:
                self.collection_history = self.collection_history[-100:]
            
            logger.info(
                f"Garbage collection completed: "
                f"analyzed={stats.memories_analyzed}, "
                f"removed={stats.memories_removed}, "
                f"duration={stats.duration_seconds:.2f}s"
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Garbage collection failed: {e}", exc_info=True)
            # Return empty stats on failure
            return CollectionStats(
                timestamp=datetime.now(timezone.utc),
                memories_analyzed=0,
                memories_removed=0,
                bytes_freed=0,
                duration_seconds=time.time() - start_time,
            )
    
    async def analyze_memory_health(self) -> MemoryHealthReport:
        """Analyze memory system health metrics.
        
        Returns:
            MemoryHealthReport with current health status
        """
        try:
            all_memories = await self._get_all_memories()
            
            if not all_memories:
                return MemoryHealthReport(
                    total_memories=0,
                    total_size_mb=0.0,
                    avg_significance=0.0,
                    significance_distribution={},
                    oldest_memory_age_days=0.0,
                    newest_memory_age_days=0.0,
                    estimated_duplicates=0,
                    needs_collection=False,
                    recommended_threshold=self.significance_threshold,
                )
            
            # Calculate metrics
            total_memories = len(all_memories)
            
            # Estimate size (rough approximation)
            estimated_size_bytes = sum(
                len(str(m.get("content", ""))) + len(str(m.get("summary", "")))
                for m in all_memories
            )
            total_size_mb = estimated_size_bytes / (1024 * 1024)
            
            # Calculate average significance
            significances = [m.get("significance", 5) for m in all_memories]
            avg_significance = sum(significances) / len(significances)
            
            # Build significance distribution
            distribution = defaultdict(int)
            for sig in significances:
                bucket = f"{int(sig)}.0-{int(sig) + 1}.0"
                distribution[bucket] += 1
            
            # Calculate age metrics
            now = datetime.now(timezone.utc)
            ages = []
            for m in all_memories:
                timestamp = m.get("timestamp")
                if timestamp:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    age = (now - timestamp).total_seconds() / 86400  # days
                    ages.append(age)
            
            oldest_age = max(ages) if ages else 0.0
            newest_age = min(ages) if ages else 0.0
            
            # Estimate duplicates (simplified - just count low-variance in content)
            estimated_duplicates = await self._estimate_duplicates(all_memories)
            
            # Determine if collection is needed
            needs_collection = (
                total_memories > self.max_memory_capacity * 0.8 or
                estimated_duplicates > 10 or
                avg_significance < 3.0
            )
            
            # Recommend threshold based on distribution
            recommended_threshold = self._calculate_recommended_threshold(significances)
            
            return MemoryHealthReport(
                total_memories=total_memories,
                total_size_mb=total_size_mb,
                avg_significance=avg_significance,
                significance_distribution=dict(distribution),
                oldest_memory_age_days=oldest_age,
                newest_memory_age_days=newest_age,
                estimated_duplicates=estimated_duplicates,
                needs_collection=needs_collection,
                recommended_threshold=recommended_threshold,
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze memory health: {e}", exc_info=True)
            return MemoryHealthReport(
                total_memories=0,
                total_size_mb=0.0,
                avg_significance=0.0,
                significance_distribution={},
                oldest_memory_age_days=0.0,
                newest_memory_age_days=0.0,
                estimated_duplicates=0,
                needs_collection=False,
                recommended_threshold=self.significance_threshold,
            )
    
    def schedule_collection(self, interval: float = 3600.0) -> None:
        """Schedule periodic automatic collection.
        
        Args:
            interval: Time between collections in seconds (default: 1 hour)
        """
        if self.is_running:
            logger.warning("Scheduled collection already running")
            return
        
        self.is_running = True
        self.scheduled_task = asyncio.create_task(
            self._scheduled_collection_loop(interval)
        )
        logger.info(f"Scheduled automatic garbage collection every {interval}s")
    
    def stop_scheduled_collection(self) -> None:
        """Stop automatic collection."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.scheduled_task and not self.scheduled_task.done():
            self.scheduled_task.cancel()
        
        logger.info("Stopped automatic garbage collection")
    
    def get_collection_history(self) -> List[CollectionStats]:
        """Get history of past collections.
        
        Returns:
            List of CollectionStats from previous runs
        """
        return self.collection_history.copy()
    
    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================
    
    async def _scheduled_collection_loop(self, interval: float) -> None:
        """Background loop for scheduled collection.
        
        Args:
            interval: Seconds between collection runs
        """
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                
                if not self.is_running:
                    break
                
                logger.info("Running scheduled garbage collection")
                stats = await self.collect()
                
                logger.info(
                    f"Scheduled GC completed: "
                    f"removed {stats.memories_removed}/{stats.memories_analyzed} memories"
                )
                
            except asyncio.CancelledError:
                logger.info("Scheduled collection task cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduled collection failed: {e}", exc_info=True)
                # Continue despite errors
    
    async def _get_all_memories(self) -> List[Dict[str, Any]]:
        """Retrieve all memories from storage.
        
        Returns:
            List of memory dictionaries with metadata
        """
        try:
            # Get all IDs from ChromaDB journal collection
            result = self.memory_store.get()
            
            if not result or not result.get("ids"):
                return []
            
            memories = []
            ids = result["ids"]
            metadatas = result.get("metadatas", [])
            documents = result.get("documents", [])
            
            for i, memory_id in enumerate(ids):
                metadata = metadatas[i] if i < len(metadatas) else {}
                document = documents[i] if i < len(documents) else ""
                
                memory = {
                    "id": memory_id,
                    "summary": document,
                    "metadata": metadata,
                    "timestamp": metadata.get("timestamp", ""),
                    "tags": metadata.get("tags", "").split(",") if metadata.get("tags") else [],
                    "significance": float(metadata.get("significance_score", 5)),
                }
                
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            return []
    
    async def _identify_removal_candidates(
        self,
        memories: List[Dict[str, Any]],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Identify memories to remove based on all strategies.
        
        Args:
            memories: All available memories
            threshold: Significance threshold for removal
            
        Returns:
            List of memories to remove
        """
        to_remove = []
        now = datetime.now(timezone.utc)
        
        for memory in memories:
            # Skip if protected by tag
            if self._is_protected_by_tag(memory):
                continue
            
            # Skip if too recent
            if self._is_too_recent(memory, now):
                continue
            
            # Apply age-based decay
            decayed_significance = self._apply_age_decay(memory, now)
            memory["_decayed_significance"] = decayed_significance
            
            # Mark for removal if below threshold
            if decayed_significance < threshold:
                memory["_removal_reason"] = "low_significance"
                to_remove.append(memory)
                continue
        
        # Apply capacity-based pruning if still over capacity
        if len(memories) - len(to_remove) > self.max_memory_capacity:
            additional = await self._capacity_based_pruning(
                memories,
                to_remove,
                self.max_memory_capacity
            )
            to_remove.extend(additional)
        
        # Identify and mark duplicates
        duplicates = await self._identify_duplicates(memories, to_remove)
        to_remove.extend(duplicates)
        
        # Limit removal rate
        if len(to_remove) > self.max_removal_per_run:
            logger.warning(
                f"Limiting removal to {self.max_removal_per_run} memories "
                f"(would have removed {len(to_remove)})"
            )
            # Keep highest priority (lowest significance) removals
            to_remove.sort(key=lambda m: m.get("_decayed_significance", 10))
            to_remove = to_remove[:self.max_removal_per_run]
        
        return to_remove
    
    def _is_protected_by_tag(self, memory: Dict[str, Any]) -> bool:
        """Check if memory has protected tags.
        
        Args:
            memory: Memory dictionary
            
        Returns:
            True if memory should be protected from removal
        """
        tags = set(memory.get("tags", []))
        return bool(tags & self.preserve_tags)
    
    def _is_too_recent(self, memory: Dict[str, Any], now: datetime) -> bool:
        """Check if memory is too recent to remove.
        
        Args:
            memory: Memory dictionary
            now: Current time
            
        Returns:
            True if memory is protected by recency
        """
        timestamp = memory.get("timestamp")
        if not timestamp:
            return False
        
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except:
                return False
        
        age_hours = (now - timestamp).total_seconds() / 3600
        return age_hours < self.recent_memory_protection_hours
    
    def _apply_age_decay(self, memory: Dict[str, Any], now: datetime) -> float:
        """Apply time-based decay to significance score.
        
        Formula: new_significance = old_significance * exp(-decay_rate * age_days)
        
        Args:
            memory: Memory dictionary
            now: Current time
            
        Returns:
            Decayed significance score
        """
        original_significance = memory.get("significance", 5)
        
        timestamp = memory.get("timestamp")
        if not timestamp:
            return original_significance
        
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except:
                return original_significance
        
        age_days = (now - timestamp).total_seconds() / 86400
        decayed = original_significance * math.exp(-self.decay_rate_per_day * age_days)
        
        return decayed
    
    async def _capacity_based_pruning(
        self,
        memories: List[Dict[str, Any]],
        already_marked: List[Dict[str, Any]],
        max_capacity: int
    ) -> List[Dict[str, Any]]:
        """Remove lowest-significance memories to enforce capacity.
        
        Args:
            memories: All memories
            already_marked: Memories already marked for removal
            max_capacity: Maximum allowed memories
            
        Returns:
            Additional memories to remove
        """
        marked_ids = {m["id"] for m in already_marked}
        remaining = [m for m in memories if m["id"] not in marked_ids]
        
        over_capacity = len(remaining) - max_capacity
        if over_capacity <= 0:
            return []
        
        # Sort by decayed significance (lowest first)
        remaining.sort(key=lambda m: m.get("_decayed_significance", 10))
        
        additional = []
        for memory in remaining[:over_capacity]:
            if not self._is_protected_by_tag(memory):
                memory["_removal_reason"] = "capacity_exceeded"
                additional.append(memory)
        
        return additional
    
    async def _identify_duplicates(
        self,
        memories: List[Dict[str, Any]],
        already_marked: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify near-duplicate memories.
        
        This is a simplified implementation. A full implementation would
        use embeddings and cosine similarity.
        
        Args:
            memories: All memories
            already_marked: Memories already marked for removal
            
        Returns:
            Duplicate memories to remove
        """
        # Simplified: just return empty list
        # Full implementation would query ChromaDB for similar embeddings
        return []
    
    async def _estimate_duplicates(self, memories: List[Dict[str, Any]]) -> int:
        """Estimate number of duplicate memories.
        
        Args:
            memories: All memories
            
        Returns:
            Estimated count of duplicates
        """
        # Simplified: return 0
        # Full implementation would check embedding similarities
        return 0
    
    async def _remove_memories(self, memories: List[Dict[str, Any]]) -> int:
        """Remove memories from storage.
        
        Args:
            memories: Memories to remove
            
        Returns:
            Estimated bytes freed
        """
        if not memories:
            return 0
        
        try:
            memory_ids = [m["id"] for m in memories]
            
            # Delete from ChromaDB
            self.memory_store.delete(ids=memory_ids)
            
            # Estimate bytes freed
            bytes_freed = sum(
                len(str(m.get("summary", ""))) + len(str(m.get("metadata", {})))
                for m in memories
            )
            
            logger.info(f"Removed {len(memories)} memories from storage")
            
            return bytes_freed
            
        except Exception as e:
            logger.error(f"Failed to remove memories: {e}", exc_info=True)
            return 0
    
    def _count_removal_reasons(self, memories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count memories by removal reason.
        
        Args:
            memories: Removed memories
            
        Returns:
            Dictionary mapping reason to count
        """
        reasons = defaultdict(int)
        for memory in memories:
            reason = memory.get("_removal_reason", "unknown")
            reasons[reason] += 1
        return dict(reasons)
    
    def _calculate_recommended_threshold(self, significances: List[float]) -> float:
        """Calculate recommended threshold based on distribution.
        
        Args:
            significances: List of all significance scores
            
        Returns:
            Recommended threshold value
        """
        if not significances:
            return self.significance_threshold
        
        # Use 25th percentile as recommended threshold
        sorted_sigs = sorted(significances)
        index = len(sorted_sigs) // 4
        return max(0.1, sorted_sigs[index])
