"""
Tests for IncrementalJournalWriter and updated IntrospectiveJournal.

This test suite ensures that:
- Entries are written immediately, not batched
- Journal rotation works correctly
- File locking prevents concurrent writes
- Crash recovery is handled properly
- JSONL format is valid
- Compression works for archived journals
- Recent entries buffer is maintained
- Integration with IntrospectiveJournal works
"""

import json
import gzip
import tempfile
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta

import pytest

from emergence_core.lyra.cognitive_core.incremental_journal import IncrementalJournalWriter
from emergence_core.lyra.cognitive_core.meta_cognition import IntrospectiveJournal


class TestIncrementalJournalWriter:
    """Test cases for IncrementalJournalWriter."""
    
    @pytest.fixture
    def temp_journal_dir(self):
        """Create a temporary directory for journal files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def writer(self, temp_journal_dir):
        """Create a journal writer instance."""
        writer = IncrementalJournalWriter(
            journal_dir=temp_journal_dir,
            max_size_mb=0.001,  # Very small for testing rotation
            auto_flush=True,
            compression=True
        )
        yield writer
        writer.close()
    
    def test_initialization(self, temp_journal_dir):
        """Test that writer initializes correctly."""
        writer = IncrementalJournalWriter(temp_journal_dir)
        
        assert writer.journal_dir == temp_journal_dir
        assert writer.current_file is not None
        assert writer.current_path is not None
        assert writer.current_path.exists()
        assert writer.bytes_written == 0
        
        writer.close()
    
    def test_write_entry_immediate(self, writer, temp_journal_dir):
        """Test that entries are written immediately."""
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
    
    def test_write_multiple_entries(self, writer, temp_journal_dir):
        """Test writing multiple entries."""
        entries = [
            {"type": "entry1", "data": "first"},
            {"type": "entry2", "data": "second"},
            {"type": "entry3", "data": "third"}
        ]
        
        for entry in entries:
            writer.write_entry(entry)
        
        # Read and verify all entries
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        with open(journal_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 3
            
            for i, line in enumerate(lines[:3]):
                written_entry = json.loads(line)
                assert written_entry["type"] == f"entry{i+1}"
    
    def test_jsonl_format_validity(self, writer, temp_journal_dir):
        """Test that JSONL format is valid and parseable line-by-line."""
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
    
    def test_journal_rotation_on_size(self, writer, temp_journal_dir):
        """Test that journal rotates when size limit is reached."""
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
    
    def test_compression_of_archived_journals(self, writer, temp_journal_dir):
        """Test that rotated journals are compressed."""
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
    
    def test_write_entries_atomic(self, writer, temp_journal_dir):
        """Test that write_entries() writes atomically."""
        entries = [
            {"type": "batch1", "data": "first"},
            {"type": "batch2", "data": "second"},
            {"type": "batch3", "data": "third"}
        ]
        
        writer.write_entries(entries)
        
        # All entries should be written
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        with open(journal_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 3
    
    def test_concurrent_writes_thread_safe(self, writer, temp_journal_dir):
        """Test that concurrent writes are thread-safe."""
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
    
    def test_crash_recovery_partial_write(self, temp_journal_dir):
        """Test that partial writes don't corrupt entire journal."""
        writer = IncrementalJournalWriter(temp_journal_dir)
        
        # Write some valid entries
        writer.write_entry({"type": "valid1", "data": "first"})
        writer.write_entry({"type": "valid2", "data": "second"})
        
        # Simulate crash by writing invalid JSON directly
        if writer.current_file:
            writer.current_file.write('{"type": "invalid", "incomplete')
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
                        # Invalid entry - skip it (this is the partial write)
                        pass
        
        # Should have 2 valid entries
        assert len(valid_entries) == 2
        assert valid_entries[0]["type"] == "valid1"
        assert valid_entries[1]["type"] == "valid2"
    
    def test_get_current_journal_path(self, writer):
        """Test getting current journal path."""
        path = writer.get_current_journal_path()
        assert path is not None
        assert path.exists()
        assert path.name.startswith("journal_")
        assert path.suffix == ".jsonl"
    
    def test_list_journal_files(self, writer, temp_journal_dir):
        """Test listing all journal files."""
        # Write entries to trigger rotation
        large_entry = {"type": "large", "data": "x" * 1000}
        for _ in range(10):
            writer.write_entry(large_entry)
        
        files = writer.list_journal_files()
        
        # Should have at least 2 files
        assert len(files) >= 2
        
        # Files should be sorted chronologically
        for i in range(len(files) - 1):
            assert files[i] <= files[i + 1]
    
    def test_flush(self, writer, temp_journal_dir):
        """Test manual flush operation."""
        writer.write_entry({"type": "test", "data": "value"}, flush=False)
        
        # Manually flush
        writer.flush()
        
        # Data should be on disk
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        with open(journal_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 1
    
    def test_close(self, writer):
        """Test safe close operation."""
        writer.write_entry({"type": "final", "data": "last"})
        
        path = writer.current_path
        writer.close()
        
        # File should still exist after close
        assert path.exists()
        
        # File handle should be closed
        assert writer.current_file is None
    
    def test_get_stats(self, writer):
        """Test getting writer statistics."""
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


class TestIntrospectiveJournal:
    """Test cases for updated IntrospectiveJournal with incremental writing."""
    
    @pytest.fixture
    def temp_journal_dir(self):
        """Create a temporary directory for journal files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def journal(self, temp_journal_dir):
        """Create a journal instance."""
        config = {
            "max_journal_size_mb": 0.001,
            "auto_flush": True,
            "compress_archived": True
        }
        journal = IntrospectiveJournal(temp_journal_dir, config)
        yield journal
        journal.close()
    
    def test_initialization(self, temp_journal_dir):
        """Test journal initialization."""
        journal = IntrospectiveJournal(temp_journal_dir)
        
        assert journal.journal_dir == temp_journal_dir
        assert journal.writer is not None
        assert len(journal.recent_entries) == 0
        
        journal.close()
    
    def test_record_observation_immediate(self, journal, temp_journal_dir):
        """Test that observations are written immediately."""
        observation = {
            "type": "test_observation",
            "content": "Test content"
        }
        
        journal.record_observation(observation)
        
        # Should be in recent entries
        assert len(journal.recent_entries) == 1
        assert journal.recent_entries[0]["type"] == "observation"
        
        # Should be on disk
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        assert len(journal_files) >= 1
        
        with open(journal_files[0], 'r') as f:
            lines = f.readlines()
            assert len(lines) >= 1
            entry = json.loads(lines[0])
            assert entry["type"] == "observation"
            assert entry["content"]["type"] == "test_observation"
    
    def test_record_realization_immediate(self, journal, temp_journal_dir):
        """Test that realizations are written immediately."""
        journal.record_realization("I have learned something", 0.85)
        
        # Should be in recent entries
        assert len(journal.recent_entries) == 1
        assert journal.recent_entries[0]["type"] == "realization"
        assert journal.recent_entries[0]["confidence"] == 0.85
        
        # Should be on disk
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        with open(journal_files[0], 'r') as f:
            entry = json.loads(f.readline())
            assert entry["type"] == "realization"
            assert entry["realization"] == "I have learned something"
    
    def test_record_question_immediate(self, journal, temp_journal_dir):
        """Test that questions are written immediately."""
        context = {"situation": "testing", "cycle": 100}
        journal.record_question("What is my purpose?", context)
        
        # Should be in recent entries
        assert len(journal.recent_entries) == 1
        assert journal.recent_entries[0]["type"] == "question"
        
        # Should be on disk
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        with open(journal_files[0], 'r') as f:
            entry = json.loads(f.readline())
            assert entry["type"] == "question"
            assert entry["question"] == "What is my purpose?"
            assert entry["context"]["situation"] == "testing"
    
    def test_recent_entries_buffer_size(self, journal):
        """Test that recent entries buffer maintains correct size."""
        # Write more than buffer size
        for i in range(150):
            journal.record_observation({"index": i, "data": f"entry_{i}"})
        
        # Buffer should be limited to 100
        assert len(journal.recent_entries) == 100
        
        # Should have most recent entries
        assert journal.recent_entries[-1]["content"]["index"] == 149
        assert journal.recent_entries[0]["content"]["index"] == 50
    
    def test_get_recent_patterns(self, journal):
        """Test pattern detection from recent entries."""
        # Record various types of entries
        for i in range(5):
            journal.record_realization(f"Realization {i}", 0.8)
        
        for i in range(3):
            journal.record_question(f"Question {i}", {"context": "test"})
        
        for i in range(7):
            journal.record_observation({"type": f"obs_{i}"})
        
        # Get patterns
        patterns = journal.get_recent_patterns(days=7)
        
        # Should have patterns for all three types
        pattern_types = [p["type"] for p in patterns]
        assert "realizations_pattern" in pattern_types
        assert "questions_pattern" in pattern_types
        assert "observations_pattern" in pattern_types
        
        # Check counts
        for pattern in patterns:
            if pattern["type"] == "realizations_pattern":
                assert pattern["count"] == 5
            elif pattern["type"] == "questions_pattern":
                assert pattern["count"] == 3
            elif pattern["type"] == "observations_pattern":
                assert pattern["count"] == 7
    
    def test_get_recent_patterns_time_filter(self, journal):
        """Test that pattern detection filters by time."""
        # Record entry with old timestamp
        old_entry = {
            "type": "observation",
            "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
            "content": {"data": "old"}
        }
        journal.recent_entries.append(old_entry)
        
        # Record recent entry
        journal.record_observation({"data": "recent"})
        
        # Get patterns for last 7 days
        patterns = journal.get_recent_patterns(days=7)
        
        # Should only include recent entry
        for pattern in patterns:
            if pattern["type"] == "observations_pattern":
                assert pattern["count"] == 1
    
    def test_save_session_backward_compatibility(self, journal):
        """Test that save_session() still works (as no-op)."""
        journal.record_observation({"test": "data"})
        
        # Should not raise exception
        journal.save_session()
        
        # Entries should still be on disk
        journal_files = list(journal.journal_dir.glob("journal_*.jsonl"))
        assert len(journal_files) >= 1
    
    def test_flush(self, journal):
        """Test manual flush operation."""
        journal.record_observation({"test": "data"})
        
        # Should not raise exception
        journal.flush()
    
    def test_close(self, journal):
        """Test graceful shutdown."""
        journal.record_observation({"test": "data"})
        
        # Should not raise exception
        journal.close()
        
        # Writer should be closed
        assert journal.writer.current_file is None
    
    def test_integration_with_cognitive_cycle(self, journal):
        """Test typical usage pattern in cognitive cycle."""
        # Simulate multiple cognitive cycles
        for cycle in range(10):
            journal.record_observation({
                "cycle": cycle,
                "type": "cycle_observation"
            })
            
            if cycle % 3 == 0:
                journal.record_realization(f"Cycle {cycle} realization", 0.7)
            
            if cycle % 5 == 0:
                journal.record_question(f"Question at cycle {cycle}", {"cycle": cycle})
        
        # Verify all entries are in buffer
        assert len(journal.recent_entries) == 10 + 4 + 2  # 10 obs + 4 real + 2 quest
        
        # Get patterns
        patterns = journal.get_recent_patterns()
        assert len(patterns) == 3
        
        # Close cleanly
        journal.close()


class TestJournalRecovery:
    """Test journal recovery scenarios."""
    
    @pytest.fixture
    def temp_journal_dir(self):
        """Create a temporary directory for journal files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_recovery_from_corrupted_line(self, temp_journal_dir):
        """Test that journal can recover from corrupted lines."""
        writer = IncrementalJournalWriter(temp_journal_dir)
        
        # Write valid entries
        writer.write_entry({"entry": 1, "data": "first"})
        writer.write_entry({"entry": 2, "data": "second"})
        
        # Write corrupted line
        if writer.current_file:
            writer.current_file.write('{"corrupted": true, "incomplete\n')
            writer.current_file.flush()
        
        # Write more valid entries
        writer.write_entry({"entry": 3, "data": "third"})
        
        writer.close()
        
        # Try to read - should recover valid entries
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        valid_entries = []
        
        with open(journal_files[0], 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    valid_entries.append(entry)
                except json.JSONDecodeError:
                    pass  # Skip corrupted line
        
        # Should have 3 valid entries
        assert len(valid_entries) == 3
        assert valid_entries[0]["entry"] == 1
        assert valid_entries[1]["entry"] == 2
        assert valid_entries[2]["entry"] == 3
    
    def test_recovery_from_missing_timestamp(self, temp_journal_dir):
        """Test that writer adds timestamp if missing."""
        writer = IncrementalJournalWriter(temp_journal_dir)
        
        # Write entry without timestamp
        entry = {"type": "test", "data": "value"}
        writer.write_entry(entry)
        
        writer.close()
        
        # Read and verify timestamp was added
        journal_files = list(temp_journal_dir.glob("journal_*.jsonl"))
        with open(journal_files[0], 'r') as f:
            written_entry = json.loads(f.readline())
            assert "timestamp" in written_entry
            
            # Verify timestamp is valid ISO format
            datetime.fromisoformat(written_entry["timestamp"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
