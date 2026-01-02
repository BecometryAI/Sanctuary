"""
Test Suite for Memory Garbage Collection

These tests validate the garbage collection system's ability to safely
and effectively manage memory growth while preserving important memories.
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import shutil

from lyra.memory_manager import MemoryManager, JournalEntry
from lyra.cognitive_core.memory_gc import (
    MemoryGarbageCollector,
    CollectionStats,
    MemoryHealthReport
)


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Create temporary directory for test memory storage."""
    memory_dir = tmp_path / "test_memory"
    memory_dir.mkdir()
    yield memory_dir
    if memory_dir.exists():
        shutil.rmtree(memory_dir)


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Create temporary directory for ChromaDB."""
    chroma_dir = tmp_path / "test_chroma"
    chroma_dir.mkdir()
    yield chroma_dir
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)


@pytest.fixture
async def memory_manager(temp_memory_dir, temp_chroma_dir):
    """Create MemoryManager instance for testing."""
    manager = MemoryManager(
        base_dir=temp_memory_dir,
        chroma_dir=temp_chroma_dir,
        blockchain_enabled=False
    )
    return manager


@pytest.fixture
def gc_config():
    """Default GC configuration for testing."""
    return {
        "significance_threshold": 0.1,
        "decay_rate_per_day": 0.01,
        "duplicate_similarity_threshold": 0.95,
        "max_memory_capacity": 100,
        "min_memories_per_category": 5,
        "preserve_tags": ["important", "pinned"],
        "aggressive_mode": False,
        "recent_memory_protection_hours": 24,
        "max_removal_per_run": 50,
    }


@pytest.fixture
async def gc_instance(memory_manager, gc_config):
    """Create MemoryGarbageCollector instance for testing."""
    return MemoryGarbageCollector(
        memory_store=memory_manager.journal_collection,
        config=gc_config
    )


# ============================================================================
# DATA STRUCTURE TESTS
# ============================================================================

class TestCollectionStats:
    """Test CollectionStats dataclass."""
    
    def test_collection_stats_creation(self):
        """Test creating CollectionStats."""
        stats = CollectionStats(
            timestamp=datetime.now(timezone.utc),
            memories_analyzed=100,
            memories_removed=10,
            bytes_freed=1024,
            duration_seconds=1.5,
            removal_reasons={"low_significance": 8, "duplicate": 2},
            avg_significance_before=3.5,
            avg_significance_after=4.2
        )
        
        assert stats.memories_analyzed == 100
        assert stats.memories_removed == 10
        assert stats.bytes_freed == 1024
        assert "low_significance" in stats.removal_reasons
    
    def test_collection_stats_to_dict(self):
        """Test serialization of CollectionStats."""
        stats = CollectionStats(
            timestamp=datetime.now(timezone.utc),
            memories_analyzed=50,
            memories_removed=5,
            bytes_freed=512,
            duration_seconds=0.5
        )
        
        data = stats.to_dict()
        assert isinstance(data, dict)
        assert "timestamp" in data
        assert data["memories_analyzed"] == 50


class TestMemoryHealthReport:
    """Test MemoryHealthReport dataclass."""
    
    def test_health_report_creation(self):
        """Test creating MemoryHealthReport."""
        report = MemoryHealthReport(
            total_memories=100,
            total_size_mb=10.5,
            avg_significance=5.5,
            significance_distribution={"5.0-6.0": 20, "6.0-7.0": 30},
            oldest_memory_age_days=100.0,
            newest_memory_age_days=0.1,
            estimated_duplicates=5,
            needs_collection=False,
            recommended_threshold=0.2
        )
        
        assert report.total_memories == 100
        assert report.needs_collection is False
        assert report.recommended_threshold == 0.2
    
    def test_health_report_to_dict(self):
        """Test serialization of MemoryHealthReport."""
        report = MemoryHealthReport(
            total_memories=50,
            total_size_mb=5.0,
            avg_significance=4.0,
            significance_distribution={},
            oldest_memory_age_days=50.0,
            newest_memory_age_days=1.0,
            estimated_duplicates=0,
            needs_collection=False,
            recommended_threshold=0.1
        )
        
        data = report.to_dict()
        assert isinstance(data, dict)
        assert data["total_memories"] == 50


# ============================================================================
# GARBAGE COLLECTOR CORE TESTS
# ============================================================================

class TestMemoryGarbageCollector:
    """Test MemoryGarbageCollector core functionality."""
    
    @pytest.mark.asyncio
    async def test_gc_initialization(self, gc_instance, gc_config):
        """Test GC initializes with correct configuration."""
        assert gc_instance.significance_threshold == gc_config["significance_threshold"]
        assert gc_instance.max_memory_capacity == gc_config["max_memory_capacity"]
        assert "important" in gc_instance.preserve_tags
        assert gc_instance.is_running is False
    
    @pytest.mark.asyncio
    async def test_collect_empty_memory(self, gc_instance):
        """Test collection with no memories."""
        stats = await gc_instance.collect()
        
        assert stats.memories_analyzed == 0
        assert stats.memories_removed == 0
        assert stats.bytes_freed == 0
    
    @pytest.mark.asyncio
    async def test_collect_with_memories(self, memory_manager, gc_instance):
        """Test collection with various memories."""
        # Create memories with different significance
        for i in range(10):
            entry = JournalEntry(
                content=f"Test memory {i}",
                summary=f"Summary {i}",
                significance_score=i + 1  # 1-10
            )
            await memory_manager.commit_journal(entry)
        
        # Run collection with low threshold
        stats = await gc_instance.collect(threshold=3.0)
        
        assert stats.memories_analyzed == 10
        assert stats.memories_removed > 0  # Should remove some low-significance memories
        assert stats.duration_seconds >= 0
    
    @pytest.mark.asyncio
    async def test_dry_run_mode(self, memory_manager, gc_instance):
        """Test dry-run mode doesn't actually remove memories."""
        # Create some low-significance memories
        for i in range(5):
            entry = JournalEntry(
                content=f"Low significance {i}",
                summary=f"Summary {i}",
                significance_score=1
            )
            await memory_manager.commit_journal(entry)
        
        # Run in dry-run mode
        stats = await gc_instance.collect(threshold=2.0, dry_run=True)
        
        # Should identify removals but not execute
        assert stats.memories_analyzed == 5
        # Memories should still exist
        all_memories = await gc_instance._get_all_memories()
        assert len(all_memories) == 5


# ============================================================================
# COLLECTION STRATEGY TESTS
# ============================================================================

class TestSignificanceBasedRemoval:
    """Test significance-based removal strategy."""
    
    @pytest.mark.asyncio
    async def test_removes_below_threshold(self, memory_manager, gc_instance):
        """Test that memories below threshold are removed."""
        # Create memories with known significance
        low_sig_ids = []
        for i in range(3):
            entry = JournalEntry(
                content=f"Low sig {i}",
                summary=f"Low {i}",
                significance_score=1
            )
            await memory_manager.commit_journal(entry)
            low_sig_ids.append(str(entry.id))
        
        high_sig_ids = []
        for i in range(3):
            entry = JournalEntry(
                content=f"High sig {i}",
                summary=f"High {i}",
                significance_score=8
            )
            await memory_manager.commit_journal(entry)
            high_sig_ids.append(str(entry.id))
        
        # Run collection with threshold = 5
        stats = await gc_instance.collect(threshold=5.0)
        
        # Low significance should be removed
        assert stats.memories_removed >= 3
        
        # High significance should remain
        remaining = await gc_instance._get_all_memories()
        remaining_ids = {m["id"] for m in remaining}
        
        for high_id in high_sig_ids:
            assert high_id in remaining_ids
    
    @pytest.mark.asyncio
    async def test_preserves_protected_tags(self, memory_manager, gc_instance):
        """Test that memories with protected tags are preserved."""
        # Create low-significance memory with protected tag
        protected_entry = JournalEntry(
            content="Important memory",
            summary="Important",
            tags=["important"],
            significance_score=1  # Low significance but protected
        )
        await memory_manager.commit_journal(protected_entry)
        
        # Create low-significance memory without protected tag
        unprotected_entry = JournalEntry(
            content="Unimportant memory",
            summary="Unimportant",
            significance_score=1
        )
        await memory_manager.commit_journal(unprotected_entry)
        
        # Run collection
        stats = await gc_instance.collect(threshold=5.0)
        
        # Check that protected memory remains
        remaining = await gc_instance._get_all_memories()
        remaining_ids = {m["id"] for m in remaining}
        
        assert str(protected_entry.id) in remaining_ids


class TestAgeBasedDecay:
    """Test age-based decay calculation."""
    
    @pytest.mark.asyncio
    async def test_decay_calculation(self, gc_instance):
        """Test that decay formula is applied correctly."""
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=100)
        
        memory = {
            "id": str(uuid4()),
            "significance": 5.0,
            "timestamp": old_time.isoformat()
        }
        
        decayed = gc_instance._apply_age_decay(memory, now)
        
        # After 100 days with 0.01 decay rate:
        # decayed = 5.0 * exp(-0.01 * 100) ≈ 1.84
        assert decayed < 5.0
        assert decayed > 0
        assert decayed < 2.0  # Should be significantly decayed
    
    @pytest.mark.asyncio
    async def test_recent_memory_no_decay(self, gc_instance):
        """Test that very recent memories have minimal decay."""
        now = datetime.now(timezone.utc)
        recent_time = now - timedelta(hours=1)
        
        memory = {
            "id": str(uuid4()),
            "significance": 5.0,
            "timestamp": recent_time.isoformat()
        }
        
        decayed = gc_instance._apply_age_decay(memory, now)
        
        # Should be nearly the same
        assert abs(decayed - 5.0) < 0.01


class TestCapacityBasedPruning:
    """Test capacity-based pruning strategy."""
    
    @pytest.mark.asyncio
    async def test_enforces_max_capacity(self, memory_manager):
        """Test that capacity limit is enforced."""
        # Create GC with low capacity
        gc = MemoryGarbageCollector(
            memory_store=memory_manager.journal_collection,
            config={"max_memory_capacity": 10}
        )
        
        # Create 20 memories
        for i in range(20):
            entry = JournalEntry(
                content=f"Memory {i}",
                summary=f"Summary {i}",
                significance_score=5
            )
            await memory_manager.commit_journal(entry)
        
        # Run collection
        stats = await gc.collect(threshold=0.1)  # Low threshold
        
        # Should remove enough to get under capacity
        remaining = await gc._get_all_memories()
        assert len(remaining) <= 10


class TestRecentMemoryProtection:
    """Test protection of recent memories."""
    
    @pytest.mark.asyncio
    async def test_protects_recent_memories(self, memory_manager, gc_instance):
        """Test that very recent memories are protected."""
        # Create recent low-significance memory
        recent_entry = JournalEntry(
            content="Recent memory",
            summary="Recent",
            significance_score=1
        )
        await memory_manager.commit_journal(recent_entry)
        
        # Create old low-significance memory
        old_entry = JournalEntry(
            content="Old memory",
            summary="Old",
            significance_score=1
        )
        # Manually set old timestamp
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Commit and then manually update timestamp in ChromaDB would be needed
        # For this test, we'll just verify the protection logic
        
        now = datetime.now(timezone.utc)
        recent_mem = {
            "id": str(recent_entry.id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "significance": 1
        }
        
        assert gc_instance._is_too_recent(recent_mem, now) is True
        
        old_mem = {
            "id": str(old_entry.id),
            "timestamp": old_time.isoformat(),
            "significance": 1
        }
        
        assert gc_instance._is_too_recent(old_mem, now) is False


# ============================================================================
# HEALTH ANALYSIS TESTS
# ============================================================================

class TestMemoryHealthAnalysis:
    """Test memory health analysis."""
    
    @pytest.mark.asyncio
    async def test_health_analysis_empty(self, gc_instance):
        """Test health analysis with no memories."""
        health = await gc_instance.analyze_memory_health()
        
        assert health.total_memories == 0
        assert health.needs_collection is False
    
    @pytest.mark.asyncio
    async def test_health_analysis_with_memories(self, memory_manager, gc_instance):
        """Test health analysis with various memories."""
        # Create diverse memories
        for i in range(10):
            entry = JournalEntry(
                content=f"Memory {i}",
                summary=f"Summary {i}",
                significance_score=(i % 10) + 1
            )
            await memory_manager.commit_journal(entry)
        
        health = await gc_instance.analyze_memory_health()
        
        assert health.total_memories == 10
        assert health.avg_significance > 0
        assert health.total_size_mb > 0
        assert len(health.significance_distribution) > 0
    
    @pytest.mark.asyncio
    async def test_health_recommends_collection(self, memory_manager):
        """Test that health analysis recommends collection when needed."""
        # Create GC with low capacity
        gc = MemoryGarbageCollector(
            memory_store=memory_manager.journal_collection,
            config={"max_memory_capacity": 10}
        )
        
        # Create many low-significance memories
        for i in range(15):
            entry = JournalEntry(
                content=f"Memory {i}",
                summary=f"Summary {i}",
                significance_score=2
            )
            await memory_manager.commit_journal(entry)
        
        health = await gc.analyze_memory_health()
        
        # Should recommend collection due to overcapacity
        assert health.needs_collection is True


# ============================================================================
# SCHEDULED COLLECTION TESTS
# ============================================================================

class TestScheduledCollection:
    """Test scheduled automatic collection."""
    
    @pytest.mark.asyncio
    async def test_schedule_starts_collection(self, gc_instance):
        """Test that scheduled collection starts."""
        gc_instance.schedule_collection(interval=3600.0)
        
        assert gc_instance.is_running is True
        assert gc_instance.scheduled_task is not None
        
        # Stop it
        gc_instance.stop_scheduled_collection()
        assert gc_instance.is_running is False
    
    @pytest.mark.asyncio
    async def test_stop_scheduled_collection(self, gc_instance):
        """Test stopping scheduled collection."""
        gc_instance.schedule_collection(interval=3600.0)
        await asyncio.sleep(0.1)  # Let it start
        
        gc_instance.stop_scheduled_collection()
        
        assert gc_instance.is_running is False
        # Task should be cancelled
        if gc_instance.scheduled_task:
            assert gc_instance.scheduled_task.cancelled() or gc_instance.scheduled_task.done()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestMemoryManagerIntegration:
    """Test GC integration with MemoryManager."""
    
    @pytest.mark.asyncio
    async def test_memory_manager_has_gc(self, memory_manager):
        """Test that MemoryManager includes GC."""
        assert hasattr(memory_manager, 'gc')
        assert isinstance(memory_manager.gc, MemoryGarbageCollector)
    
    @pytest.mark.asyncio
    async def test_enable_disable_auto_gc(self, memory_manager):
        """Test enabling and disabling auto GC."""
        memory_manager.enable_auto_gc(interval=3600.0)
        assert memory_manager.gc.is_running is True
        
        memory_manager.disable_auto_gc()
        assert memory_manager.gc.is_running is False
    
    @pytest.mark.asyncio
    async def test_run_gc_method(self, memory_manager):
        """Test running GC through MemoryManager."""
        # Create some memories
        for i in range(5):
            entry = JournalEntry(
                content=f"Test {i}",
                summary=f"Summary {i}",
                significance_score=i + 1
            )
            await memory_manager.commit_journal(entry)
        
        # Run GC
        stats = await memory_manager.run_gc(threshold=3.0)
        
        assert isinstance(stats, CollectionStats)
        assert stats.memories_analyzed == 5
    
    @pytest.mark.asyncio
    async def test_get_memory_health_method(self, memory_manager):
        """Test getting memory health through MemoryManager."""
        health = await memory_manager.get_memory_health()
        
        assert isinstance(health, MemoryHealthReport)
        assert health.total_memories >= 0


# ============================================================================
# COLLECTION HISTORY TESTS
# ============================================================================

class TestCollectionHistory:
    """Test collection history tracking."""
    
    @pytest.mark.asyncio
    async def test_history_is_tracked(self, memory_manager, gc_instance):
        """Test that collection history is recorded."""
        # Create some memories
        for i in range(5):
            entry = JournalEntry(
                content=f"Test {i}",
                summary=f"Summary {i}",
                significance_score=1
            )
            await memory_manager.commit_journal(entry)
        
        # Run collection twice
        await gc_instance.collect(threshold=2.0)
        await gc_instance.collect(threshold=2.0)
        
        history = gc_instance.get_collection_history()
        
        assert len(history) == 2
        assert all(isinstance(s, CollectionStats) for s in history)
    
    @pytest.mark.asyncio
    async def test_history_limit(self, gc_instance):
        """Test that history is limited to 100 entries."""
        # Simulate many collections
        for _ in range(150):
            gc_instance.collection_history.append(
                CollectionStats(
                    timestamp=datetime.now(timezone.utc),
                    memories_analyzed=0,
                    memories_removed=0,
                    bytes_freed=0,
                    duration_seconds=0.0
                )
            )
        
        history = gc_instance.get_collection_history()
        
        # Should be capped at 100
        assert len(history) <= 100


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test GC performance requirements."""
    
    @pytest.mark.asyncio
    async def test_collection_speed(self, memory_manager, gc_instance):
        """Test that collection completes in reasonable time."""
        # Create 100 memories
        for i in range(100):
            entry = JournalEntry(
                content=f"Memory {i} " * 10,  # Some content
                summary=f"Summary {i}",
                significance_score=(i % 10) + 1
            )
            await memory_manager.commit_journal(entry)
        
        # Run collection and time it
        stats = await gc_instance.collect(threshold=5.0)
        
        # Should complete in under 5 seconds for 100 memories
        assert stats.duration_seconds < 5.0
        assert stats.memories_analyzed == 100


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Test error handling in GC."""
    
    @pytest.mark.asyncio
    async def test_graceful_failure(self, gc_instance):
        """Test that GC handles errors gracefully."""
        # Force an error by passing invalid threshold
        stats = await gc_instance.collect(threshold=-1.0)
        
        # Should return valid stats even on error
        assert isinstance(stats, CollectionStats)
        # Should have logged the error but not crashed
