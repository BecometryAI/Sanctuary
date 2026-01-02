"""
Incremental Journal Writer: Immediate journal entry persistence.

This module provides incremental journal saving to write journal entries
immediately rather than batching at shutdown, preventing data loss from
crashes and providing real-time journaling.

Key Features:
- Append-only writes to active journal file
- Automatic file rotation based on size
- JSONL format (one JSON object per line) for easy parsing and recovery
- File locking to prevent concurrent write conflicts
- Buffered writes with configurable flush policy
- Crash recovery: partial writes don't corrupt entire journal
- Journal compression for archived (rotated) files
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import io

logger = logging.getLogger(__name__)


class IncrementalJournalWriter:
    """
    Writes journal entries incrementally to disk in JSONL format.
    
    This class provides immediate persistence of journal entries, preventing
    data loss from system crashes. It supports automatic file rotation,
    compression of archived files, and thread-safe writes.
    
    Attributes:
        journal_dir: Directory for storing journal files
        max_size_mb: Maximum size in MB before rotation
        auto_flush: Whether to flush after each write
        compression: Whether to compress archived journals
        current_file: Currently active journal file handle
        current_path: Path to currently active journal file
        bytes_written: Bytes written to current file
        write_lock: Thread lock for safe concurrent writes
    """
    
    def __init__(
        self,
        journal_dir: Path,
        max_size_mb: float = 10.0,
        auto_flush: bool = True,
        compression: bool = True
    ):
        """
        Initialize the incremental journal writer.
        
        Args:
            journal_dir: Directory path for storing journal files
            max_size_mb: Maximum file size in MB before rotation (default: 10.0)
            auto_flush: Flush to disk after each write (default: True)
            compression: Compress archived journal files (default: True)
        """
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_mb = max_size_mb
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.auto_flush = auto_flush
        self.compression = compression
        
        self.current_file: Optional[io.TextIOWrapper] = None
        self.current_path: Optional[Path] = None
        self.bytes_written: int = 0
        self.write_lock = threading.Lock()
        
        # Create initial journal file
        self._create_new_journal()
        
        logger.info(f"✅ IncrementalJournalWriter initialized at {self.journal_dir}")
    
    def write_entry(self, entry: Dict, flush: bool = True) -> None:
        """
        Write a single journal entry immediately.
        
        Args:
            entry: Dictionary containing entry data
            flush: Whether to flush to disk (default: True, overrides auto_flush)
        
        Raises:
            IOError: If write fails
        """
        with self.write_lock:
            try:
                # Check if rotation is needed
                if self.bytes_written >= self.max_size_bytes:
                    self.rotate_journal()
                
                # Ensure timestamp is present
                if "timestamp" not in entry:
                    entry["timestamp"] = datetime.now().isoformat()
                
                # Write as JSON line
                json_line = json.dumps(entry, ensure_ascii=False)
                line_bytes = (json_line + "\n").encode('utf-8')
                
                if self.current_file:
                    self.current_file.write(json_line + "\n")
                    self.bytes_written += len(line_bytes)
                    
                    # Flush if requested or auto_flush is enabled
                    if flush or self.auto_flush:
                        self.current_file.flush()
                    
                    logger.debug(f"📝 Wrote journal entry ({len(line_bytes)} bytes)")
                else:
                    logger.error("❌ No active journal file")
                    raise IOError("No active journal file")
                    
            except Exception as e:
                logger.error(f"❌ Failed to write journal entry: {e}")
                raise
    
    def write_entries(self, entries: List[Dict]) -> None:
        """
        Write multiple entries atomically.
        
        All entries are written together, ensuring consistency.
        
        Args:
            entries: List of entry dictionaries
        
        Raises:
            IOError: If write fails
        """
        if not entries:
            return
        
        with self.write_lock:
            try:
                for entry in entries:
                    # Use internal write without lock (we already have it)
                    self._write_entry_unlocked(entry)
                
                # Flush once at the end
                if self.current_file:
                    self.current_file.flush()
                
                logger.debug(f"📝 Wrote {len(entries)} journal entries atomically")
                
            except Exception as e:
                logger.error(f"❌ Failed to write journal entries: {e}")
                raise
    
    def _write_entry_unlocked(self, entry: Dict) -> None:
        """
        Write entry without acquiring lock (internal use only).
        
        Args:
            entry: Dictionary containing entry data
        """
        # Check if rotation is needed
        if self.bytes_written >= self.max_size_bytes:
            self.rotate_journal()
        
        # Ensure timestamp is present
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
        
        # Write as JSON line
        json_line = json.dumps(entry, ensure_ascii=False)
        line_bytes = (json_line + "\n").encode('utf-8')
        
        if self.current_file:
            self.current_file.write(json_line + "\n")
            self.bytes_written += len(line_bytes)
    
    def rotate_journal(self, max_size_mb: Optional[float] = None) -> None:
        """
        Rotate journal file when size limit reached.
        
        Closes current file, optionally compresses it, and creates a new one.
        
        Args:
            max_size_mb: Override max size for this rotation (optional)
        """
        with self.write_lock:
            try:
                # Close current file
                if self.current_file:
                    self.current_file.flush()
                    self.current_file.close()
                    logger.info(f"📦 Closed journal file: {self.current_path}")
                
                # Compress old file if enabled
                if self.compression and self.current_path and self.current_path.exists():
                    self._compress_journal(self.current_path)
                
                # Create new journal file
                self._create_new_journal()
                
                logger.info(f"🔄 Rotated journal (size threshold: {self.max_size_mb}MB)")
                
            except Exception as e:
                logger.error(f"❌ Failed to rotate journal: {e}")
                raise
    
    def _create_new_journal(self) -> None:
        """
        Create a new journal file with timestamp-based name.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_path = self.journal_dir / f"journal_{timestamp}.jsonl"
        
        # Open file in append mode with UTF-8 encoding and line buffering
        self.current_file = open(
            self.current_path,
            'a',
            encoding='utf-8',
            buffering=1  # Line buffering
        )
        self.bytes_written = 0
        
        logger.info(f"📄 Created new journal file: {self.current_path}")
    
    def _compress_journal(self, journal_path: Path) -> None:
        """
        Compress a journal file using gzip.
        
        Args:
            journal_path: Path to journal file to compress
        """
        try:
            compressed_path = journal_path.with_suffix('.jsonl.gz')
            
            with open(journal_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb', compresslevel=6) as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file after successful compression
            journal_path.unlink()
            
            logger.info(f"🗜️ Compressed journal: {journal_path.name} -> {compressed_path.name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to compress journal {journal_path}: {e}")
    
    def get_current_journal_path(self) -> Path:
        """
        Get path to active journal file.
        
        Returns:
            Path to currently active journal file
        """
        if self.current_path:
            return self.current_path
        else:
            raise ValueError("No active journal file")
    
    def list_journal_files(self) -> List[Path]:
        """
        List all journal files in chronological order.
        
        Returns:
            List of journal file paths (both .jsonl and .jsonl.gz)
        """
        journal_files = []
        
        # Find .jsonl files
        journal_files.extend(sorted(self.journal_dir.glob("journal_*.jsonl")))
        
        # Find .jsonl.gz files
        journal_files.extend(sorted(self.journal_dir.glob("journal_*.jsonl.gz")))
        
        # Sort by filename (which includes timestamp)
        journal_files.sort()
        
        return journal_files
    
    def flush(self) -> None:
        """
        Flush any buffered data to disk.
        """
        with self.write_lock:
            if self.current_file:
                self.current_file.flush()
                logger.debug("💾 Flushed journal buffer to disk")
    
    def close(self) -> None:
        """
        Safely close journal file handles.
        
        Should be called during system shutdown to ensure all data is persisted.
        """
        with self.write_lock:
            try:
                if self.current_file:
                    self.current_file.flush()
                    self.current_file.close()
                    logger.info(f"💾 Closed journal file: {self.current_path}")
                    self.current_file = None
                    self.current_path = None
            except Exception as e:
                logger.error(f"❌ Error closing journal file: {e}")
    
    def __del__(self):
        """
        Ensure file is closed on object destruction.
        """
        if self.current_file:
            try:
                self.current_file.close()
            except:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about journal writer.
        
        Returns:
            Dictionary with stats (current file size, total files, etc.)
        """
        journal_files = self.list_journal_files()
        
        total_size = 0
        for file_path in journal_files:
            if file_path.exists():
                total_size += file_path.stat().st_size
        
        return {
            "current_file": str(self.current_path) if self.current_path else None,
            "bytes_written_current": self.bytes_written,
            "total_journal_files": len(journal_files),
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "auto_flush": self.auto_flush,
            "compression": self.compression
        }
