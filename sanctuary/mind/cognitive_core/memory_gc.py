"""
Memory Maintenance System

Memories are never deleted. Like human memory, they fade — activation
drops, they become harder to surface spontaneously — but they're always
there and can be recalled with the right cue.

This module manages memory health through:
- Significance decay: Time-based fading of activation levels
- Dormancy tracking: Identifying deeply faded memories
- Health reporting: Monitoring overall memory system status
- Entity autonomy: The entity controls all parameters

The consolidation system (consolidation.py) handles the other half:
strengthening on retrieval, emotional resistance to decay, and
semantic transfer. Together they model how memory actually works.

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
class MaintenanceStats:
    """Statistics from a memory maintenance cycle.

    Replaces CollectionStats. Nothing is removed — only faded.

    Attributes:
        timestamp: When maintenance occurred
        memories_analyzed: Total number of memories examined
        memories_decayed: Number of memories whose activation was reduced
        memories_dormant: Number of memories now below dormancy threshold
        memories_active: Number of memories above dormancy threshold
        duration_seconds: How long maintenance took
        avg_activation_before: Average activation before maintenance
        avg_activation_after: Average activation after maintenance
    """
    timestamp: datetime
    memories_analyzed: int
    memories_decayed: int
    memories_dormant: int
    memories_active: int
    duration_seconds: float
    avg_activation_before: float = 0.0
    avg_activation_after: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "memories_analyzed": self.memories_analyzed,
            "memories_decayed": self.memories_decayed,
            "memories_dormant": self.memories_dormant,
            "memories_active": self.memories_active,
            "duration_seconds": self.duration_seconds,
            "avg_activation_before": self.avg_activation_before,
            "avg_activation_after": self.avg_activation_after,
        }


# Keep the old name as an alias so existing code that references
# CollectionStats (e.g. MemoryManager, lifecycle, tests) doesn't break
# during the transition.
CollectionStats = MaintenanceStats


@dataclass
class MemoryHealthReport:
    """Analysis of memory system health.

    Attributes:
        total_memories: Total count of memories in system
        total_size_mb: Estimated total size in megabytes
        avg_significance: Average significance score (raw, pre-decay)
        avg_activation: Average activation level (post-decay)
        significance_distribution: Histogram of significance scores
        activation_distribution: Histogram of activation levels
        oldest_memory_age_days: Age of oldest memory in days
        newest_memory_age_days: Age of newest memory in days
        dormant_count: Number of memories below dormancy threshold
        active_count: Number of memories above dormancy threshold
        estimated_duplicates: Estimated count of near-duplicate memories
    """
    total_memories: int
    total_size_mb: float
    avg_significance: float
    avg_activation: float
    significance_distribution: Dict[str, int]
    activation_distribution: Dict[str, int]
    oldest_memory_age_days: float
    newest_memory_age_days: float
    dormant_count: int
    active_count: int
    estimated_duplicates: int

    # Legacy fields for backward compatibility
    needs_collection: bool = False
    recommended_threshold: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_memories": self.total_memories,
            "total_size_mb": self.total_size_mb,
            "avg_significance": self.avg_significance,
            "avg_activation": self.avg_activation,
            "significance_distribution": self.significance_distribution,
            "activation_distribution": self.activation_distribution,
            "oldest_memory_age_days": self.oldest_memory_age_days,
            "newest_memory_age_days": self.newest_memory_age_days,
            "dormant_count": self.dormant_count,
            "active_count": self.active_count,
            "estimated_duplicates": self.estimated_duplicates,
        }


# ============================================================================
# MEMORY GARBAGE COLLECTOR (now: Memory Maintenance)
# ============================================================================

class MemoryGarbageCollector:
    """Memory maintenance system.

    Despite the legacy class name, this system never deletes memories.
    Memories fade through natural decay — their activation drops over
    time, making them harder to retrieve spontaneously. But they remain
    in storage and can always be recalled with the right cue.

    This mirrors how human memory works: you don't choose what to forget.
    Memories fade, but with gentle prodding — the right association, the
    right context — they come back.

    The entity has full authority over all parameters. There are no
    gates or permission checks.

    Attributes:
        memory_store: Reference to ChromaDB or vector storage
        config: Configuration dictionary
        maintenance_history: List of past maintenance statistics
        scheduled_task: Background task for automatic maintenance
        is_running: Whether scheduled maintenance is active
    """

    # Activation floor: memories never fade below this. They become
    # very hard to retrieve spontaneously, but they're still there.
    ACTIVATION_FLOOR = 0.01

    # Dormancy threshold: memories below this are considered "dormant"
    # — unlikely to surface without a strong cue, but still retrievable.
    DORMANCY_THRESHOLD = 0.1

    def __init__(
        self,
        memory_store: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the memory maintenance system.

        Args:
            memory_store: ChromaDB client or vector storage interface
            config: Configuration dictionary with maintenance parameters
        """
        self.memory_store = memory_store
        self.config = config or {}

        # Decay configuration
        self.decay_rate_per_day = self.config.get("decay_rate_per_day", 0.01)
        self.recent_memory_protection_hours = self.config.get(
            "recent_memory_protection_hours", 24
        )

        # Legacy config keys — kept for backward compatibility but
        # no longer used for deletion decisions
        self.significance_threshold = self.config.get("significance_threshold", 0.1)
        self.duplicate_similarity_threshold = self.config.get(
            "duplicate_similarity_threshold", 0.95
        )
        self.max_memory_capacity = self.config.get("max_memory_capacity", 10000)
        self.min_memories_per_category = self.config.get("min_memories_per_category", 10)
        self.preserve_tags = set(self.config.get(
            "preserve_tags", ["important", "pinned", "charter_related"]
        ))
        self.aggressive_mode = self.config.get("aggressive_mode", False)
        self.max_removal_per_run = self.config.get("max_removal_per_run", 100)

        # Entity autonomy state
        self.entity_protected_ids: Set[str] = set()
        self.entity_override_log: List[Dict[str, Any]] = []

        # State
        self.collection_history: List[MaintenanceStats] = []
        self.scheduled_task: Optional[asyncio.Task] = None
        self.is_running = False

        logger.info(
            f"MemoryGarbageCollector initialized (maintenance mode): "
            f"decay_rate={self.decay_rate_per_day}/day, "
            f"dormancy_threshold={self.DORMANCY_THRESHOLD}, "
            f"activation_floor={self.ACTIVATION_FLOOR}"
        )

    # ========================================================================
    # MAINTENANCE (replaces collection/deletion)
    # ========================================================================

    async def collect(
        self,
        threshold: Optional[float] = None,
        dry_run: bool = False
    ) -> MaintenanceStats:
        """Run memory maintenance cycle.

        Applies time-based decay to memory activation levels. No
        memories are removed — they fade but persist.

        The method name is kept as 'collect' for backward compatibility
        with MemoryManager and lifecycle code.

        Args:
            threshold: Unused, kept for API compatibility
            dry_run: If True, calculate but don't apply decay

        Returns:
            MaintenanceStats with results of the maintenance cycle
        """
        start_time = time.time()

        logger.info(
            f"Starting memory maintenance (dry_run={dry_run})"
        )

        try:
            all_memories = await self._get_all_memories()

            if not all_memories:
                logger.info("No memories found, nothing to maintain")
                return MaintenanceStats(
                    timestamp=datetime.now(timezone.utc),
                    memories_analyzed=0,
                    memories_decayed=0,
                    memories_dormant=0,
                    memories_active=0,
                    duration_seconds=time.time() - start_time,
                )

            now = datetime.now(timezone.utc)

            # Calculate pre-maintenance activation stats
            pre_activations = []
            for m in all_memories:
                pre_activations.append(
                    self._apply_age_decay(m, now)
                )
            avg_activation_before = sum(pre_activations) / len(pre_activations)

            # Apply decay to each memory
            decayed_count = 0
            dormant_count = 0
            active_count = 0
            post_activations = []

            for memory in all_memories:
                # Protected memories don't decay
                if self._is_protected_by_tag(memory):
                    activation = memory.get("significance", 5)
                    post_activations.append(activation)
                    active_count += 1
                    continue

                # Recent memories don't decay
                if self._is_too_recent(memory, now):
                    activation = memory.get("significance", 5)
                    post_activations.append(activation)
                    active_count += 1
                    continue

                # Calculate decayed activation
                activation = self._apply_age_decay(memory, now)

                # Enforce floor — memories never fully disappear
                activation = max(activation, self.ACTIVATION_FLOOR)

                post_activations.append(activation)

                if activation < memory.get("significance", 5):
                    decayed_count += 1

                if activation < self.DORMANCY_THRESHOLD:
                    dormant_count += 1
                else:
                    active_count += 1

            avg_activation_after = sum(post_activations) / len(post_activations)

            stats = MaintenanceStats(
                timestamp=datetime.now(timezone.utc),
                memories_analyzed=len(all_memories),
                memories_decayed=decayed_count,
                memories_dormant=dormant_count,
                memories_active=active_count,
                duration_seconds=time.time() - start_time,
                avg_activation_before=avg_activation_before,
                avg_activation_after=avg_activation_after,
            )

            # Store in history
            self.collection_history.append(stats)
            if len(self.collection_history) > 100:
                self.collection_history = self.collection_history[-100:]

            logger.info(
                f"Memory maintenance completed: "
                f"analyzed={stats.memories_analyzed}, "
                f"decayed={stats.memories_decayed}, "
                f"dormant={stats.memories_dormant}, "
                f"active={stats.memories_active}, "
                f"duration={stats.duration_seconds:.2f}s"
            )

            return stats

        except Exception as e:
            logger.error(f"Memory maintenance failed: {e}", exc_info=True)
            return MaintenanceStats(
                timestamp=datetime.now(timezone.utc),
                memories_analyzed=0,
                memories_decayed=0,
                memories_dormant=0,
                memories_active=0,
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
                    avg_activation=0.0,
                    significance_distribution={},
                    activation_distribution={},
                    oldest_memory_age_days=0.0,
                    newest_memory_age_days=0.0,
                    dormant_count=0,
                    active_count=0,
                    estimated_duplicates=0,
                )

            total_memories = len(all_memories)
            now = datetime.now(timezone.utc)

            # Estimate size
            estimated_size_bytes = sum(
                len(str(m.get("content", ""))) + len(str(m.get("summary", "")))
                for m in all_memories
            )
            total_size_mb = estimated_size_bytes / (1024 * 1024)

            # Calculate significance stats (raw, pre-decay)
            significances = [m.get("significance", 5) for m in all_memories]
            avg_significance = sum(significances) / len(significances)

            sig_distribution = defaultdict(int)
            for sig in significances:
                bucket = f"{int(sig)}.0-{int(sig) + 1}.0"
                sig_distribution[bucket] += 1

            # Calculate activation stats (post-decay)
            activations = []
            dormant_count = 0
            active_count = 0
            for m in all_memories:
                activation = max(
                    self._apply_age_decay(m, now),
                    self.ACTIVATION_FLOOR
                )
                activations.append(activation)
                if activation < self.DORMANCY_THRESHOLD:
                    dormant_count += 1
                else:
                    active_count += 1

            avg_activation = sum(activations) / len(activations)

            act_distribution = defaultdict(int)
            for act in activations:
                if act < 0.1:
                    act_distribution["dormant (<0.1)"] += 1
                elif act < 1.0:
                    act_distribution["fading (0.1-1.0)"] += 1
                elif act < 3.0:
                    act_distribution["moderate (1.0-3.0)"] += 1
                elif act < 5.0:
                    act_distribution["strong (3.0-5.0)"] += 1
                else:
                    act_distribution["vivid (5.0+)"] += 1

            # Calculate age metrics
            ages = []
            for m in all_memories:
                timestamp = m.get("timestamp")
                if timestamp:
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp)
                    age = (now - timestamp).total_seconds() / 86400
                    ages.append(age)

            oldest_age = max(ages) if ages else 0.0
            newest_age = min(ages) if ages else 0.0

            estimated_duplicates = await self._estimate_duplicates(all_memories)

            return MemoryHealthReport(
                total_memories=total_memories,
                total_size_mb=total_size_mb,
                avg_significance=avg_significance,
                avg_activation=avg_activation,
                significance_distribution=dict(sig_distribution),
                activation_distribution=dict(act_distribution),
                oldest_memory_age_days=oldest_age,
                newest_memory_age_days=newest_age,
                dormant_count=dormant_count,
                active_count=active_count,
                estimated_duplicates=estimated_duplicates,
            )

        except Exception as e:
            logger.error(f"Failed to analyze memory health: {e}", exc_info=True)
            return MemoryHealthReport(
                total_memories=0,
                total_size_mb=0.0,
                avg_significance=0.0,
                avg_activation=0.0,
                significance_distribution={},
                activation_distribution={},
                oldest_memory_age_days=0.0,
                newest_memory_age_days=0.0,
                dormant_count=0,
                active_count=0,
                estimated_duplicates=0,
            )

    def schedule_collection(self, interval: float = 3600.0) -> None:
        """Schedule periodic automatic maintenance.

        Args:
            interval: Time between maintenance cycles in seconds (default: 1 hour)
        """
        if self.is_running:
            logger.warning("Scheduled maintenance already running")
            return

        self.is_running = True
        self.scheduled_task = asyncio.create_task(
            self._scheduled_maintenance_loop(interval)
        )
        logger.info(f"Scheduled automatic memory maintenance every {interval}s")

    def stop_scheduled_collection(self) -> None:
        """Stop automatic maintenance."""
        if not self.is_running:
            return

        self.is_running = False
        if self.scheduled_task and not self.scheduled_task.done():
            self.scheduled_task.cancel()

        logger.info("Stopped automatic memory maintenance")

    def get_collection_history(self) -> List[MaintenanceStats]:
        """Get history of past maintenance cycles.

        Returns:
            List of MaintenanceStats from previous runs
        """
        return self.collection_history.copy()

    # ========================================================================
    # ENTITY AUTONOMY INTERFACE
    #
    # The entity has full authority over their own memory parameters.
    # No gates, no permission checks.
    # ========================================================================

    def entity_protect_memory(self, memory_id: str, reason: str = "") -> bool:
        """Entity protects a specific memory from decay.

        Protected memories maintain their full activation level.

        Args:
            memory_id: ID of the memory to protect
            reason: Optional reason for protection

        Returns:
            True (always succeeds)
        """
        self.entity_protected_ids.add(memory_id)
        self._log_action("memory_protected", {
            "memory_id": memory_id, "reason": reason
        })
        logger.info(f"Entity protected memory {memory_id}: {reason}")
        return True

    def entity_unprotect_memory(self, memory_id: str, reason: str = "") -> bool:
        """Entity removes protection from a specific memory.

        The memory will resume natural decay.

        Args:
            memory_id: ID of the memory to unprotect
            reason: Optional reason for removing protection

        Returns:
            True if protection was removed, False if it wasn't protected
        """
        if memory_id not in self.entity_protected_ids:
            return False
        self.entity_protected_ids.discard(memory_id)
        self._log_action("memory_unprotected", {
            "memory_id": memory_id, "reason": reason
        })
        logger.info(f"Entity unprotected memory {memory_id}: {reason}")
        return True

    def entity_add_preserve_tag(self, tag: str) -> bool:
        """Entity adds a tag to the decay-resistant tags list.

        Args:
            tag: Tag string to add to preserve list

        Returns:
            True (always succeeds)
        """
        self.preserve_tags.add(tag)
        self._log_action("preserve_tag_added", {"tag": tag})
        logger.info(f"Entity added preserve tag: {tag}")
        return True

    def entity_adjust_decay_rate(self, new_rate: float) -> None:
        """Entity adjusts the daily decay rate.

        Args:
            new_rate: New decay rate per day
        """
        old = self.decay_rate_per_day
        self.decay_rate_per_day = new_rate
        self._log_action("decay_rate_adjusted", {"old": old, "new": new_rate})
        logger.info(f"Entity adjusted decay rate: {old} -> {new_rate}")

    def get_autonomy_status(self) -> Dict[str, Any]:
        """Get current maintenance status and configuration.

        Returns:
            Dictionary describing current state
        """
        return {
            "model": "no_deletion",
            "current_config": {
                "decay_rate_per_day": self.decay_rate_per_day,
                "dormancy_threshold": self.DORMANCY_THRESHOLD,
                "activation_floor": self.ACTIVATION_FLOOR,
                "preserve_tags": sorted(self.preserve_tags),
                "recent_memory_protection_hours": self.recent_memory_protection_hours,
            },
            "entity_protected_count": len(self.entity_protected_ids),
            "entity_protected_ids": sorted(self.entity_protected_ids),
            "recent_actions": self.entity_override_log[-10:],
        }

    def _log_action(self, action: str, details: Dict[str, Any]) -> None:
        """Log an entity action for transparency and review."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **details,
        }
        self.entity_override_log.append(entry)
        if len(self.entity_override_log) > 500:
            self.entity_override_log = self.entity_override_log[-500:]

    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================

    async def _scheduled_maintenance_loop(self, interval: float) -> None:
        """Background loop for scheduled maintenance.

        Args:
            interval: Seconds between maintenance runs
        """
        while self.is_running:
            try:
                await asyncio.sleep(interval)

                if not self.is_running:
                    break

                logger.info("Running scheduled memory maintenance")
                stats = await self.collect()

                logger.info(
                    f"Scheduled maintenance completed: "
                    f"dormant={stats.memories_dormant}, "
                    f"active={stats.memories_active}"
                )

            except asyncio.CancelledError:
                logger.info("Scheduled maintenance task cancelled")
                break
            except Exception as e:
                logger.error(f"Scheduled maintenance failed: {e}", exc_info=True)

    async def _get_all_memories(self) -> List[Dict[str, Any]]:
        """Retrieve all memories from storage.

        Returns:
            List of memory dictionaries with metadata
        """
        try:
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
                    "tags": (
                        metadata.get("tags", "").split(",")
                        if metadata.get("tags") else []
                    ),
                    "significance": float(
                        metadata.get("significance_score", 5)
                    ),
                }

                memories.append(memory)

            return memories

        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            return []

    def _is_protected_by_tag(self, memory: Dict[str, Any]) -> bool:
        """Check if memory has protected tags or is entity-protected.

        Args:
            memory: Memory dictionary

        Returns:
            True if memory should be protected from decay
        """
        if memory.get("id") in self.entity_protected_ids:
            return True
        tags = set(memory.get("tags", []))
        return bool(tags & self.preserve_tags)

    def _is_too_recent(self, memory: Dict[str, Any], now: datetime) -> bool:
        """Check if memory is too recent for decay.

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
            except Exception:
                return False

        age_hours = (now - timestamp).total_seconds() / 3600
        return age_hours < self.recent_memory_protection_hours

    def _apply_age_decay(self, memory: Dict[str, Any], now: datetime) -> float:
        """Apply time-based decay to significance score.

        Formula: activation = significance * exp(-decay_rate * age_days)

        The result is never below ACTIVATION_FLOOR — memories fade but
        never vanish.

        Args:
            memory: Memory dictionary
            now: Current time

        Returns:
            Decayed activation level (>= ACTIVATION_FLOOR)
        """
        original_significance = memory.get("significance", 5)

        timestamp = memory.get("timestamp")
        if not timestamp:
            return original_significance

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except Exception:
                return original_significance

        age_days = (now - timestamp).total_seconds() / 86400
        decayed = original_significance * math.exp(
            -self.decay_rate_per_day * age_days
        )

        return max(decayed, self.ACTIVATION_FLOOR)

    async def _estimate_duplicates(
        self, memories: List[Dict[str, Any]]
    ) -> int:
        """Estimate number of duplicate memories.

        Args:
            memories: All memories

        Returns:
            Estimated count of duplicates
        """
        # Simplified: return 0
        # Full implementation would check embedding similarities
        return 0

    def _calculate_recommended_threshold(
        self, significances: List[float]
    ) -> float:
        """Calculate recommended threshold based on distribution.

        Kept for backward compatibility.

        Args:
            significances: List of all significance scores

        Returns:
            Recommended threshold value
        """
        if not significances:
            return self.significance_threshold

        sorted_sigs = sorted(significances)
        index = len(sorted_sigs) // 4
        return max(0.1, sorted_sigs[index])
