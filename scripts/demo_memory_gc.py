"""
Demo: Memory Garbage Collection

This script demonstrates the Memory GC system's capabilities:
1. Creating memories with various significance levels
2. Running garbage collection
3. Analyzing memory health
4. Scheduled automatic collection

Usage:
    python scripts/demo_memory_gc.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emergence_core.lyra.memory_manager import MemoryManager, JournalEntry, EmotionalState
from emergence_core.lyra.cognitive_core.memory_gc import MemoryGarbageCollector


async def create_test_memories(manager: MemoryManager, count: int = 50):
    """Create test memories with various significance levels."""
    print(f"\n📝 Creating {count} test memories...")
    
    memories = []
    
    for i in range(count):
        # Vary significance (some high, some low)
        if i < 5:
            # A few very important memories
            significance = 9
            tags = ["important", "milestone"]
        elif i < 15:
            # Some moderately important
            significance = 6
            tags = ["reflection"]
        else:
            # Many low-significance memories
            significance = max(1, (i % 5) + 1)
            tags = ["routine"]
        
        entry = JournalEntry(
            content=f"Test memory {i}: This is memory number {i} with significance {significance}. " * 5,
            summary=f"Test memory {i} summary",
            tags=tags,
            emotional_signature=[EmotionalState.SERENITY if significance > 5 else EmotionalState.MELANCHOLY],
            significance_score=significance
        )
        
        success = await manager.commit_journal(entry)
        if success:
            memories.append(entry)
    
    print(f"✅ Created {len(memories)} memories")
    return memories


async def demonstrate_health_analysis(manager: MemoryManager):
    """Demonstrate memory health analysis."""
    print("\n" + "="*60)
    print("📊 MEMORY HEALTH ANALYSIS")
    print("="*60)
    
    health = await manager.get_memory_health()
    
    print(f"\nTotal memories: {health.total_memories}")
    print(f"Total size: {health.total_size_mb:.2f} MB")
    print(f"Average significance: {health.avg_significance:.2f}")
    print(f"Oldest memory: {health.oldest_memory_age_days:.1f} days old")
    print(f"Newest memory: {health.newest_memory_age_days:.1f} days old")
    print(f"Estimated duplicates: {health.estimated_duplicates}")
    print(f"Needs collection: {'Yes ⚠️' if health.needs_collection else 'No ✅'}")
    print(f"Recommended threshold: {health.recommended_threshold:.2f}")
    
    if health.significance_distribution:
        print("\nSignificance Distribution:")
        for bucket, count in sorted(health.significance_distribution.items()):
            bar = "█" * (count // 2)
            print(f"  {bucket}: {count:3d} {bar}")


async def demonstrate_dry_run(manager: MemoryManager):
    """Demonstrate dry-run mode."""
    print("\n" + "="*60)
    print("🔍 DRY RUN - Preview what would be removed")
    print("="*60)
    
    stats = await manager.run_gc(threshold=5.0, dry_run=True)
    
    print(f"\nMemories analyzed: {stats.memories_analyzed}")
    print(f"Would remove: {stats.memories_removed} memories")
    print(f"Would free: {stats.bytes_freed:,} bytes")
    print(f"Analysis took: {stats.duration_seconds:.2f}s")
    
    if stats.removal_reasons:
        print("\nRemoval reasons:")
        for reason, count in stats.removal_reasons.items():
            print(f"  - {reason}: {count}")
    
    print(f"\nAvg significance before: {stats.avg_significance_before:.2f}")
    print(f"Avg significance after: {stats.avg_significance_after:.2f}")
    print(f"Improvement: {stats.avg_significance_after - stats.avg_significance_before:+.2f}")


async def demonstrate_actual_collection(manager: MemoryManager):
    """Demonstrate actual garbage collection."""
    print("\n" + "="*60)
    print("🧹 ACTUAL GARBAGE COLLECTION")
    print("="*60)
    
    print("\nRunning collection with threshold=5.0...")
    stats = await manager.run_gc(threshold=5.0, dry_run=False)
    
    print(f"\n✅ Collection complete!")
    print(f"Memories analyzed: {stats.memories_analyzed}")
    print(f"Memories removed: {stats.memories_removed}")
    print(f"Bytes freed: {stats.bytes_freed:,}")
    print(f"Duration: {stats.duration_seconds:.2f}s")
    
    if stats.removal_reasons:
        print("\nRemoval breakdown:")
        for reason, count in stats.removal_reasons.items():
            print(f"  - {reason}: {count}")
    
    print(f"\nQuality improvement:")
    print(f"  Before: {stats.avg_significance_before:.2f}")
    print(f"  After:  {stats.avg_significance_after:.2f}")
    print(f"  Change: {stats.avg_significance_after - stats.avg_significance_before:+.2f}")


async def demonstrate_scheduled_collection(manager: MemoryManager):
    """Demonstrate scheduled automatic collection."""
    print("\n" + "="*60)
    print("⏰ SCHEDULED AUTOMATIC COLLECTION")
    print("="*60)
    
    print("\nEnabling automatic GC (every 10 seconds for demo)...")
    manager.enable_auto_gc(interval=10.0)
    
    print("✅ Automatic GC enabled")
    print("Waiting 12 seconds for first collection...")
    
    await asyncio.sleep(12)
    
    # Check collection history
    history = manager.gc.get_collection_history()
    if history:
        latest = history[-1]
        print(f"\n✅ Automatic collection ran at {latest.timestamp.strftime('%H:%M:%S')}")
        print(f"   Removed: {latest.memories_removed} memories")
    
    print("\nDisabling automatic GC...")
    manager.disable_auto_gc()
    print("✅ Automatic GC disabled")


async def demonstrate_collection_history(manager: MemoryManager):
    """Demonstrate collection history tracking."""
    print("\n" + "="*60)
    print("📜 COLLECTION HISTORY")
    print("="*60)
    
    history = manager.gc.get_collection_history()
    
    if not history:
        print("\nNo collections yet")
        return
    
    print(f"\nTotal collections: {len(history)}")
    print("\nRecent collections:")
    
    for i, stats in enumerate(history[-5:], 1):  # Show last 5
        print(f"\n{i}. {stats.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Analyzed: {stats.memories_analyzed}, Removed: {stats.memories_removed}")
        print(f"   Duration: {stats.duration_seconds:.2f}s, Freed: {stats.bytes_freed:,} bytes")
        if stats.removal_reasons:
            reasons_str = ", ".join(f"{k}={v}" for k, v in stats.removal_reasons.items())
            print(f"   Reasons: {reasons_str}")


async def demonstrate_protected_memories(manager: MemoryManager):
    """Demonstrate that protected memories are preserved."""
    print("\n" + "="*60)
    print("🛡️  PROTECTED MEMORY DEMONSTRATION")
    print("="*60)
    
    print("\nCreating low-significance memory with 'important' tag...")
    protected = JournalEntry(
        content="This is a very important memory despite low significance score",
        summary="Important protected memory",
        tags=["important"],
        significance_score=1  # Very low!
    )
    await manager.commit_journal(protected)
    
    print("Creating low-significance memory without protection...")
    unprotected = JournalEntry(
        content="This is an unimportant memory with low significance",
        summary="Unprotected memory",
        significance_score=1
    )
    await manager.commit_journal(unprotected)
    
    print("\nRunning GC with low threshold...")
    stats = await manager.run_gc(threshold=5.0)
    
    print(f"\n✅ Collection complete: removed {stats.memories_removed} memories")
    
    # Verify protected memory still exists
    memories = await manager.gc._get_all_memories()
    protected_exists = any(m["id"] == str(protected.id) for m in memories)
    
    if protected_exists:
        print("✅ Protected memory preserved (has 'important' tag)")
    else:
        print("❌ Protected memory was removed (should not happen!)")


async def main():
    """Run all demonstrations."""
    print("="*60)
    print("Memory Garbage Collection Demo")
    print("="*60)
    
    # Create temporary storage
    import tempfile
    import shutil
    
    temp_dir = Path(tempfile.mkdtemp())
    memory_dir = temp_dir / "memories"
    chroma_dir = temp_dir / "chroma"
    
    try:
        print(f"\n📁 Using temporary storage: {temp_dir}")
        
        # Initialize manager with GC config
        gc_config = {
            "significance_threshold": 0.1,
            "max_memory_capacity": 100,
            "preserve_tags": ["important", "pinned"],
            "recent_memory_protection_hours": 24,
            "max_removal_per_run": 50
        }
        
        manager = MemoryManager(
            base_dir=memory_dir,
            chroma_dir=chroma_dir,
            blockchain_enabled=False,
            gc_config=gc_config
        )
        
        # Run demonstrations
        await create_test_memories(manager, count=50)
        await demonstrate_health_analysis(manager)
        await demonstrate_protected_memories(manager)
        await demonstrate_dry_run(manager)
        await demonstrate_actual_collection(manager)
        await demonstrate_health_analysis(manager)  # Show improvement
        await demonstrate_scheduled_collection(manager)
        await demonstrate_collection_history(manager)
        
        print("\n" + "="*60)
        print("✅ Demo complete!")
        print("="*60)
        print("\nKey takeaways:")
        print("1. GC safely removes low-significance memories")
        print("2. Protected memories are never removed")
        print("3. Dry-run lets you preview changes")
        print("4. Automatic collection runs in background")
        print("5. Collection history tracks all operations")
        
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary storage...")
        shutil.rmtree(temp_dir)
        print("✅ Cleanup complete")


if __name__ == "__main__":
    asyncio.run(main())
