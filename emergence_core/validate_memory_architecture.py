"""
Quick validation script for Sovereign Memory Architecture.

This script performs basic smoke tests to verify the implementation works.
Run this before full pytest suite to catch obvious issues.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lyra.memory_manager import (
    MemoryManager,
    JournalEntry,
    FactEntry,
    Manifest,
    EmotionalState
)


async def validate_basic_functionality():
    """Run basic validation tests."""
    print("=" * 60)
    print("SOVEREIGN MEMORY ARCHITECTURE - VALIDATION")
    print("=" * 60)
    
    # Setup temporary test directory
    test_dir = Path(__file__).parent / "validation_test_data"
    test_dir.mkdir(exist_ok=True)
    
    memory_dir = test_dir / "memories"
    chroma_dir = test_dir / "chroma"
    
    try:
        print("\n[1/7] Initializing MemoryManager...")
        manager = MemoryManager(
            base_dir=memory_dir,
            chroma_dir=chroma_dir,
            blockchain_enabled=False
        )
        print("✅ MemoryManager initialized successfully")
        
        # Test 1: Create JournalEntry
        print("\n[2/7] Creating JournalEntry...")
        entry = JournalEntry(
            content="This is a test journal entry exploring the nature of memory.",
            summary="Test entry about memory",
            tags=["test", "validation", "memory"],
            emotional_signature=[EmotionalState.SERENITY, EmotionalState.WONDER],
            significance_score=7
        )
        print(f"✅ JournalEntry created: {entry.id}")
        print(f"   Timestamp: {entry.timestamp}")
        print(f"   Emotions: {[e.value for e in entry.emotional_signature]}")
        
        # Test 2: Commit to storage
        print("\n[3/7] Committing journal entry...")
        success = await manager.commit_journal(entry)
        if success:
            print("✅ Journal entry committed successfully")
        else:
            print("❌ Failed to commit journal entry")
            return False
        
        # Test 3: Verify file exists
        print("\n[4/7] Verifying local JSON storage...")
        year = entry.timestamp.strftime("%Y")
        month = entry.timestamp.strftime("%m")
        expected_path = memory_dir / "journals" / year / month / f"entry_{entry.id}.json"
        
        if expected_path.exists():
            print(f"✅ JSON file exists: {expected_path}")
        else:
            print(f"❌ JSON file missing: {expected_path}")
            return False
        
        # Test 4: Recall entry
        print("\n[5/7] Testing semantic recall...")
        results = await manager.recall(
            query="memory and nature",
            n_results=5,
            memory_type="journal"
        )
        print(f"✅ Recalled {len(results)} entries")
        if len(results) > 0:
            print(f"   First result: {results[0].summary}")
        
        # Test 5: Create and commit fact
        print("\n[6/7] Creating and committing FactEntry...")
        fact = FactEntry(
            entity="System",
            attribute="validation_status",
            value="passing",
            confidence=1.0,
            source_entry_id=entry.id
        )
        
        success = await manager.commit_fact(fact)
        if success:
            print(f"✅ Fact committed: {fact.entity}.{fact.attribute} = {fact.value}")
        else:
            print("❌ Failed to commit fact")
            return False
        
        # Test 6: Manifest operations
        print("\n[7/7] Testing Manifest save/load...")
        manifest = Manifest(
            core_values=["Validation", "Integrity", "Continuity"],
            pivotal_memories=[entry],
            current_directives=["Ensure system reliability"]
        )
        
        success = await manager.save_manifest(manifest)
        if not success:
            print("❌ Failed to save manifest")
            return False
        
        loaded_manifest = await manager.load_manifest()
        if loaded_manifest:
            print("✅ Manifest saved and loaded successfully")
            print(f"   Core values: {loaded_manifest.core_values}")
            print(f"   Pivotal memories: {len(loaded_manifest.pivotal_memories)}")
        else:
            print("❌ Failed to load manifest")
            return False
        
        print("\n" + "=" * 60)
        print("ALL VALIDATIONS PASSED ✅")
        print("=" * 60)
        print("\nSovereign Memory Architecture is operational.")
        print("You can now run the full test suite:")
        print("  pytest emergence_core/tests/test_memory_manager.py -v")
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        print("\n[Cleanup] Removing test data...")
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("✅ Cleanup complete")


def test_pydantic_validation():
    """Test that Pydantic validation works correctly."""
    print("\n" + "=" * 60)
    print("PYDANTIC VALIDATION TESTS")
    print("=" * 60)
    
    # Test 1: Empty content should fail
    print("\n[Test 1] Empty content validation...")
    try:
        entry = JournalEntry(
            content="",
            summary="Test"
        )
        print("❌ Should have rejected empty content")
        return False
    except ValueError:
        print("✅ Correctly rejected empty content")
    
    # Test 2: Invalid significance score
    print("\n[Test 2] Significance score bounds...")
    try:
        entry = JournalEntry(
            content="Test",
            summary="Test",
            significance_score=15
        )
        print("❌ Should have rejected significance_score > 10")
        return False
    except ValueError:
        print("✅ Correctly rejected significance_score > 10")
    
    # Test 3: Invalid confidence score
    print("\n[Test 3] Confidence score bounds...")
    try:
        fact = FactEntry(
            entity="Test",
            attribute="test",
            value="test",
            confidence=1.5
        )
        print("❌ Should have rejected confidence > 1.0")
        return False
    except ValueError:
        print("✅ Correctly rejected confidence > 1.0")
    
    # Test 4: Immutability
    print("\n[Test 4] Entry immutability...")
    entry = JournalEntry(
        content="Original",
        summary="Original"
    )
    try:
        entry.content = "Modified"
        print("❌ Should have prevented modification")
        return False
    except Exception:
        print("✅ Correctly prevented modification (frozen model)")
    
    print("\n" + "=" * 60)
    print("PYDANTIC VALIDATION TESTS PASSED ✅")
    print("=" * 60)
    return True


def main():
    """Run all validation tests."""
    print("\nStarting Sovereign Memory Architecture validation...\n")
    
    # Test Pydantic validation first (sync)
    if not test_pydantic_validation():
        print("\n⚠️  Pydantic validation tests failed!")
        return 1
    
    # Test basic functionality (async)
    result = asyncio.run(validate_basic_functionality())
    
    if result:
        print("\n✅ All validation checks passed!")
        print("\nNext steps:")
        print("  1. Run full test suite: pytest emergence_core/tests/test_memory_manager.py -v")
        print("  2. Integrate with router.py")
        print("  3. Migrate existing data")
        return 0
    else:
        print("\n❌ Validation failed - check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
