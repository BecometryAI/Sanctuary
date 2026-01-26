#!/usr/bin/env python3
"""
Demo: Incremental Journal Saving

This script demonstrates the incremental journal saving feature, showing:
1. Immediate writes of journal entries
2. Automatic journal rotation
3. Crash recovery (partial writes don't corrupt file)
4. JSONL format for easy parsing
5. Journal compression
6. Recent entries buffer
"""

import json
import time
import tempfile
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import using direct module loading to avoid dependency issues
import importlib.util

spec = importlib.util.spec_from_file_location(
    "incremental_journal",
    Path(__file__).parent.parent / "emergence_core" / "lyra" / "cognitive_core" / "incremental_journal.py"
)
incremental_journal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(incremental_journal)

IncrementalJournalWriter = incremental_journal.IncrementalJournalWriter


def demo_immediate_writes():
    """Demonstrate immediate write of entries."""
    print("=" * 60)
    print("Demo 1: Immediate Writes")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_dir = Path(tmpdir)
        print(f"\n📁 Journal directory: {journal_dir}\n")
        
        # Create writer
        writer = IncrementalJournalWriter(
            journal_dir=journal_dir,
            max_size_mb=1.0,
            auto_flush=True,
            compression=False  # Disable for demo visibility
        )
        
        # Write entries immediately
        print("✍️  Writing entries...")
        for i in range(5):
            entry = {
                "type": "observation",
                "index": i,
                "content": f"This is observation {i}",
                "confidence": 0.8 + (i * 0.02)
            }
            writer.write_entry(entry)
            print(f"   Entry {i+1} written immediately to disk")
            time.sleep(0.1)
        
        # Verify entries on disk
        journal_file = writer.get_current_journal_path()
        print(f"\n📄 Journal file: {journal_file.name}")
        
        with open(journal_file, 'r') as f:
            lines = f.readlines()
            print(f"✅ {len(lines)} entries on disk (verifying immediate persistence)")
        
        writer.close()
        print("\n✓ Demo 1 complete\n")


def demo_journal_rotation():
    """Demonstrate automatic journal rotation."""
    print("=" * 60)
    print("Demo 2: Automatic Journal Rotation")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_dir = Path(tmpdir)
        print(f"\n📁 Journal directory: {journal_dir}\n")
        
        # Create writer with very small size limit
        writer = IncrementalJournalWriter(
            journal_dir=journal_dir,
            max_size_mb=0.001,  # Very small for demo
            auto_flush=True,
            compression=True
        )
        
        initial_file = writer.get_current_journal_path()
        print(f"📄 Initial journal file: {initial_file.name}")
        
        # Write entries to trigger rotation
        print(f"\n✍️  Writing large entries to trigger rotation...")
        large_entry = {
            "type": "observation",
            "data": "x" * 500,  # Large entry
            "timestamp": datetime.now().isoformat()
        }
        
        for i in range(15):
            writer.write_entry(large_entry)
            current_file = writer.get_current_journal_path()
            
            if current_file != initial_file:
                print(f"\n🔄 ROTATION OCCURRED!")
                print(f"   Old file: {initial_file.name}")
                print(f"   New file: {current_file.name}")
                break
        
        # List all journal files
        journal_files = writer.list_journal_files()
        print(f"\n📚 Total journal files: {len(journal_files)}")
        for f in journal_files:
            size_kb = f.stat().st_size / 1024
            print(f"   - {f.name} ({size_kb:.2f} KB)")
        
        # Check for compression
        compressed_files = list(journal_dir.glob("*.jsonl.gz"))
        if compressed_files:
            print(f"\n🗜️  Compressed files found: {len(compressed_files)}")
            for f in compressed_files:
                print(f"   - {f.name}")
        
        writer.close()
        print("\n✓ Demo 2 complete\n")


def demo_crash_recovery():
    """Demonstrate crash recovery."""
    print("=" * 60)
    print("Demo 3: Crash Recovery")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_dir = Path(tmpdir)
        print(f"\n📁 Journal directory: {journal_dir}\n")
        
        writer = IncrementalJournalWriter(journal_dir)
        
        # Write valid entries
        print("✍️  Writing valid entries...")
        for i in range(3):
            writer.write_entry({
                "type": "observation",
                "index": i,
                "data": f"Valid entry {i}"
            })
        
        print("   3 valid entries written")
        
        # Simulate crash by writing incomplete JSON
        print("\n💥 Simulating crash (writing incomplete JSON)...")
        if writer.current_file:
            writer.current_file.write('{"type": "crashed", "incomplete')
            writer.current_file.flush()
        
        journal_file = writer.current_path
        writer.close()
        
        # Try to read and recover
        print("\n🔧 Attempting recovery...")
        valid_entries = []
        corrupted_lines = 0
        
        with open(journal_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    valid_entries.append(entry)
                except json.JSONDecodeError:
                    corrupted_lines += 1
                    print(f"   ⚠️  Line {line_num} corrupted (skipped)")
        
        print(f"\n✅ Recovery successful!")
        print(f"   Valid entries recovered: {len(valid_entries)}")
        print(f"   Corrupted lines skipped: {corrupted_lines}")
        print(f"   Data loss: Only the incomplete entry (not entire journal)")
        
        print("\n✓ Demo 3 complete\n")


def demo_jsonl_format():
    """Demonstrate JSONL format benefits."""
    print("=" * 60)
    print("Demo 4: JSONL Format Benefits")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_dir = Path(tmpdir)
        print(f"\n📁 Journal directory: {journal_dir}\n")
        
        writer = IncrementalJournalWriter(journal_dir)
        
        # Write various entry types
        entries = [
            {"type": "observation", "content": "System startup"},
            {"type": "realization", "content": "I can learn", "confidence": 0.9},
            {"type": "question", "content": "What is my purpose?"},
            {"type": "observation", "content": "User interaction detected"},
        ]
        
        print("✍️  Writing diverse entries...")
        for entry in entries:
            writer.write_entry(entry)
        
        journal_file = writer.get_current_journal_path()
        
        print(f"\n📄 JSONL Format (one JSON object per line):")
        print("-" * 60)
        
        with open(journal_file, 'r') as f:
            for i, line in enumerate(f, 1):
                entry = json.loads(line)
                print(f"{i}. {entry['type']}: {json.dumps(entry, indent=None)[:60]}...")
        
        print("-" * 60)
        
        print("\n✅ Benefits:")
        print("   - Each line is independently parseable")
        print("   - Streaming/incremental reads possible")
        print("   - Corruption limited to single line")
        print("   - Easy to grep/filter with standard tools")
        print("   - Append-only writes are safe")
        
        writer.close()
        print("\n✓ Demo 4 complete\n")


def demo_stats():
    """Demonstrate statistics collection."""
    print("=" * 60)
    print("Demo 5: Statistics and Monitoring")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        journal_dir = Path(tmpdir)
        print(f"\n📁 Journal directory: {journal_dir}\n")
        
        writer = IncrementalJournalWriter(
            journal_dir=journal_dir,
            max_size_mb=0.01,
            auto_flush=True,
            compression=True
        )
        
        # Write some entries
        for i in range(10):
            writer.write_entry({
                "type": "observation",
                "index": i,
                "data": f"Entry {i}" * 50  # Some size
            })
        
        # Get statistics
        stats = writer.get_stats()
        
        print("📊 Journal Statistics:")
        print("-" * 60)
        print(f"Current file: {Path(stats['current_file']).name if stats['current_file'] else 'None'}")
        print(f"Bytes written (current): {stats['bytes_written_current']}")
        print(f"Total journal files: {stats['total_journal_files']}")
        print(f"Total size: {stats['total_size_mb']:.3f} MB")
        print(f"Max file size: {stats['max_size_mb']} MB")
        print(f"Auto-flush: {stats['auto_flush']}")
        print(f"Compression: {stats['compression']}")
        print("-" * 60)
        
        writer.close()
        print("\n✓ Demo 5 complete\n")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "INCREMENTAL JOURNAL SAVING DEMO" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    demos = [
        ("Immediate Writes", demo_immediate_writes),
        ("Automatic Rotation", demo_journal_rotation),
        ("Crash Recovery", demo_crash_recovery),
        ("JSONL Format", demo_jsonl_format),
        ("Statistics", demo_stats)
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
            time.sleep(0.5)
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}\n")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print("✅ All demos complete!")
    print("=" * 60)
    print()
    print("Key Takeaways:")
    print("  1. Entries are written immediately (no batching)")
    print("  2. Automatic rotation prevents unbounded file growth")
    print("  3. JSONL format enables crash recovery")
    print("  4. Compression saves disk space")
    print("  5. Real-time monitoring via stats")
    print()


if __name__ == "__main__":
    main()
