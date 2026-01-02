"""
Standalone test for IncrementalJournalWriter that doesn't require full dependencies.

This test directly imports only the incremental_journal module to avoid
dependency issues during testing.
"""

import json
import gzip
import tempfile
import threading
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import only what we need without triggering full imports
import importlib.util

# Load incremental_journal module directly
spec = importlib.util.spec_from_file_location(
    "incremental_journal",
    Path(__file__).parent.parent / "lyra" / "cognitive_core" / "incremental_journal.py"
)
incremental_journal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(incremental_journal)

IncrementalJournalWriter = incremental_journal.IncrementalJournalWriter


def test_initialization():
    """Test that writer initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_journal_dir = Path(tmpdir)
        writer = IncrementalJournalWriter(temp_journal_dir)
        
        assert writer.journal_dir == temp_journal_dir
        assert writer.current_file is not None
        assert writer.current_path is not None
        assert writer.current_path.exists()
        assert writer.bytes_written == 0
        
        writer.close()
    print("✓ test_initialization passed")


def test_write_entry_immediate():
    """Test that entries are written immediately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_journal_dir = Path(tmpdir)
        writer = IncrementalJournalWriter(temp_journal_dir)
        
        entry = {
            "type": "test",
            "content": "Test entry",
            "value": 42
        }
        
        writer.write_entry(entry)
        
        # File should exist and contain the entry
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        assert len(journal_files) >= 1
        
        # Read the file and verify entry
        with open(journal_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 1
            
            written_entry = json.loads(lines[0])
            assert written_entry["type"] == "test"
            assert written_entry["content"] == "Test entry"
            assert written_entry["value"] == 42
            assert "timestamp" in written_entry
        
        writer.close()
    print("✓ test_write_entry_immediate passed")


def test_jsonl_format_validity():
    """Test that JSONL format is valid and parseable line-by-line."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_journal_dir = Path(tmpdir)
        writer = IncrementalJournalWriter(temp_journal_dir)
        
        entries = [
            {"type": "observation", "content": "Test 1"},
            {"type": "realization", "content": "Test 2"},
            {"type": "question", "content": "Test 3"}
        ]
        
        for entry in entries:
            writer.write_entry(entry)
        
        # Verify JSONL format: each line should be valid JSON
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        with open(journal_files[0], 'r') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    # Should not raise exception
                    parsed = json.loads(line)
                    assert isinstance(parsed, dict)
        
        writer.close()
    print("✓ test_jsonl_format_validity passed")


def test_journal_rotation_on_size():
    """Test that journal rotates when size limit is reached."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_journal_dir = Path(tmpdir)
        writer = IncrementalJournalWriter(
            temp_journal_dir,
            max_size_mb=0.001,  # Very small for testing rotation
            compression=True
        )
        
        # Write entries until rotation happens
        large_entry = {"type": "large", "data": "x" * 1000}
        
        initial_path = writer.current_path
        
        # Write enough to trigger rotation
        for _ in range(10):
            writer.write_entry(large_entry)
        
        # Should have rotated and created new file
        assert writer.current_path != initial_path
        
        # Should have at least 2 files (one rotated, one current)
        all_files = list(temp_journal_dir.glob("journal_*"))
        assert len(all_files) >= 2
        
        writer.close()
    print("✓ test_journal_rotation_on_size passed")


def test_compression_of_archived_journals():
    """Test that rotated journals are compressed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_journal_dir = Path(tmpdir)
        writer = IncrementalJournalWriter(
            temp_journal_dir,
            max_size_mb=0.001,
            compression=True
        )
        
        # Write enough to trigger rotation
        large_entry = {"type": "large", "data": "x" * 1000}
        
        for _ in range(10):
            writer.write_entry(large_entry)
        
        # Check for compressed files
        compressed_files = list(temp_journal_dir.glob("journal_*.jsonl.gz"))
        
        # Should have at least one compressed file
        assert len(compressed_files) >= 1
        
        # Verify it can be decompressed and read
        with gzip.open(compressed_files[0], 'rt') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            for line in lines:
                if line.strip():
                    entry = json.loads(line)
                    assert entry["type"] == "large"
        
        writer.close()
    print("✓ test_compression_of_archived_journals passed")


def test_concurrent_writes_thread_safe():
    """Test that concurrent writes are thread-safe."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_journal_dir = Path(tmpdir)
        writer = IncrementalJournalWriter(temp_journal_dir)
        
        def write_entries(thread_id, count):
            for i in range(count):
                writer.write_entry({
                    "thread": thread_id,
                    "index": i,
                    "data": f"Thread {thread_id} entry {i}"
                })
        
        # Create multiple threads writing concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_entries, args=(i, 10))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Count total entries written
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl*"))
        total_entries = 0
        
        for file in journal_files:
            if file.suffix == '.gz':
                with gzip.open(file, 'rt') as f:
                    total_entries += len([line for line in f if line.strip()])
            else:
                with open(file, 'r') as f:
                    total_entries += len([line for line in f if line.strip()])
        
        # Should have all 50 entries (5 threads * 10 entries)
        assert total_entries == 50
        
        writer.close()
    print("✓ test_concurrent_writes_thread_safe passed")


def test_crash_recovery_partial_write():
    """Test that partial writes don't corrupt entire journal."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_journal_dir = Path(tmpdir)
        writer = IncrementalJournalWriter(temp_journal_dir)
        
        # Write some valid entries
        writer.write_entry({"entry": 1, "data": "first"})
        writer.write_entry({"entry": 2, "data": "second"})
        
        # Simulate crash by writing invalid JSON directly
        if writer.current_file:
            writer.current_file.write('{"entry": 3, "incomplete')
            writer.current_file.flush()
        
        writer.close()
        
        # Try to read the file - should be able to read valid entries
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        valid_entries = []
        
        with open(journal_files[0], 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        valid_entries.append(entry)
                    except json.JSONDecodeError:
                        pass  # Skip corrupted line
        
        # Should have 2 valid entries
        assert len(valid_entries) == 2
        assert valid_entries[0]["entry"] == 1
        assert valid_entries[1]["entry"] == 2
    print("✓ test_crash_recovery_partial_write passed")


def test_get_stats():
    """Test getting writer statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_journal_dir = Path(tmpdir)
        writer = IncrementalJournalWriter(
            temp_journal_dir,
            max_size_mb=0.001
        )
        
        writer.write_entry({"type": "test", "data": "value"})
        
        stats = writer.get_stats()
        
        assert "current_file" in stats
        assert "bytes_written_current" in stats
        assert "total_journal_files" in stats
        assert "total_size_bytes" in stats
        assert "total_size_mb" in stats
        assert "max_size_mb" in stats
        assert "auto_flush" in stats
        assert "compression" in stats
        
        assert stats["bytes_written_current"] > 0
        assert stats["total_journal_files"] >= 1
        assert stats["max_size_mb"] == 0.001
        
        writer.close()
    print("✓ test_get_stats passed")


if __name__ == "__main__":
    print("Running IncrementalJournalWriter tests...\n")
    
    test_initialization()
    test_write_entry_immediate()
    test_jsonl_format_validity()
    test_journal_rotation_on_size()
    test_compression_of_archived_journals()
    test_concurrent_writes_thread_safe()
    test_crash_recovery_partial_write()
    test_get_stats()
    
    print("\n✅ All tests passed!")
