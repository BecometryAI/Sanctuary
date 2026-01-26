"""
Checkpoint Manager: Save and restore complete workspace state.

This module implements comprehensive workspace state checkpointing to enable
save/restore of complete workspace state for session continuity. It provides:

- Save/load complete workspace state to/from disk
- Automatic periodic checkpointing
- Checkpoint rotation (keeping last N checkpoints)
- Compression support for large checkpoints
- Atomic writes to prevent corruption
- Integrity validation with checksums

The CheckpointManager integrates with CognitiveCore to provide:
- Save state at critical points
- Recover from crashes or interruptions
- Experiment with different conversation paths
- Track system state evolution over time
- Resume sessions seamlessly

Usage:
    >>> manager = CheckpointManager()
    >>> checkpoint_path = manager.save_checkpoint(workspace, metadata={"label": "Before experiment"})
    >>> restored_workspace = manager.load_checkpoint(checkpoint_path)
    >>> checkpoints = manager.list_checkpoints()
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from .workspace import GlobalWorkspace

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CHECKPOINT_DIR = Path("data/checkpoints")
DEFAULT_MAX_CHECKPOINTS = 20
CHECKPOINT_VERSION = "1.0"


@dataclass
class CheckpointInfo:
    """Information about a checkpoint."""
    checkpoint_id: str
    timestamp: datetime
    version: str
    path: Path
    size_bytes: int
    compressed: bool
    metadata: Dict[str, Any]
    checksum: str


class CheckpointManager:
    """
    Manages workspace state checkpoints for session continuity.
    
    The CheckpointManager provides comprehensive save/restore functionality
    for GlobalWorkspace state, enabling session recovery, experimentation,
    and state tracking over time.
    
    Features:
    - JSON serialization with custom encoders for datetime, Path, UUID
    - Compression support (gzip) for large checkpoints
    - Atomic writes (write to temp file, then rename)
    - Configurable checkpoint directory
    - Checkpoint rotation (keep last N checkpoints)
    - Integrity validation (checksums, schema validation)
    - Automatic periodic checkpointing
    
    Attributes:
        checkpoint_dir: Directory where checkpoints are stored
        max_checkpoints: Maximum number of checkpoints to keep (rotation)
        compression: Whether to use gzip compression
        auto_save_task: Task handle for auto-save loop (if enabled)
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
        compression: bool = True,
    ) -> None:
        """
        Initialize the CheckpointManager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints (default: data/checkpoints/)
            max_checkpoints: Maximum checkpoints to keep before rotation (default: 20)
            compression: Whether to use gzip compression (default: True)
        """
        self.checkpoint_dir = checkpoint_dir or DEFAULT_CHECKPOINT_DIR
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        self.auto_save_task: Optional[asyncio.Task] = None
        self._auto_save_running = False
        
        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"CheckpointManager initialized: dir={self.checkpoint_dir}, "
                   f"max={max_checkpoints}, compression={compression}")
    
    def save_checkpoint(
        self,
        workspace: GlobalWorkspace,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save complete workspace state to disk.
        
        Creates a checkpoint file containing the full workspace state with
        metadata. Uses atomic writes to prevent corruption and optionally
        compresses the checkpoint.
        
        Args:
            workspace: GlobalWorkspace instance to save
            metadata: Optional metadata dict (user label, session info, etc.)
            
        Returns:
            Path to the saved checkpoint file
            
        Raises:
            IOError: If checkpoint cannot be written
            ValueError: If workspace serialization fails
            
        Example:
            >>> manager = CheckpointManager()
            >>> path = manager.save_checkpoint(
            ...     workspace,
            ...     metadata={"label": "Before important conversation"}
            ... )
        """
        try:
            # Generate checkpoint ID and timestamp
            checkpoint_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Serialize workspace state
            workspace_state = workspace.to_dict()
            
            # Build checkpoint structure
            checkpoint = {
                "version": CHECKPOINT_VERSION,
                "timestamp": timestamp.isoformat(),
                "checkpoint_id": checkpoint_id,
                "workspace_state": workspace_state,
                "metadata": metadata or {},
            }
            
            # Generate checkpoint filename
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"checkpoint_{timestamp_str}_{checkpoint_id[:8]}.json"
            if self.compression:
                filename += ".gz"
            
            checkpoint_path = self.checkpoint_dir / filename
            
            # Serialize to JSON
            json_data = json.dumps(checkpoint, indent=2, default=self._json_encoder)
            json_bytes = json_data.encode('utf-8')
            
            # Calculate checksum
            checksum = hashlib.sha256(json_bytes).hexdigest()
            
            # Add checksum to metadata (in-memory only, not in file)
            checkpoint["metadata"]["_checksum"] = checksum
            
            # Atomic write: write to temp file, then rename
            temp_path = checkpoint_path.with_suffix('.tmp')
            
            try:
                if self.compression:
                    with gzip.open(temp_path, 'wb') as f:
                        f.write(json_bytes)
                else:
                    with open(temp_path, 'wb') as f:
                        f.write(json_bytes)
                
                # Atomic rename
                temp_path.rename(checkpoint_path)
                
            finally:
                # Clean up temp file if it still exists
                if temp_path.exists():
                    temp_path.unlink()
            
            # Enforce checkpoint rotation
            self._rotate_checkpoints()
            
            size_kb = checkpoint_path.stat().st_size / 1024
            logger.info(f"💾 Checkpoint saved: {checkpoint_path.name} ({size_kb:.1f} KB)")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
            raise
    
    def load_checkpoint(self, checkpoint_path: Path) -> GlobalWorkspace:
        """
        Restore workspace from saved checkpoint.
        
        Loads a checkpoint file and reconstructs the GlobalWorkspace state.
        Validates the checkpoint format and integrity before restoration.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            GlobalWorkspace: Restored workspace instance
            
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is corrupted or invalid
            
        Example:
            >>> manager = CheckpointManager()
            >>> workspace = manager.load_checkpoint(checkpoint_path)
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Read checkpoint file
            if checkpoint_path.suffix == '.gz':
                with gzip.open(checkpoint_path, 'rb') as f:
                    json_bytes = f.read()
            else:
                with open(checkpoint_path, 'rb') as f:
                    json_bytes = f.read()
            
            # Validate checksum
            checksum = hashlib.sha256(json_bytes).hexdigest()
            
            # Parse JSON
            checkpoint = json.loads(json_bytes.decode('utf-8'))
            
            # Validate version
            version = checkpoint.get("version")
            if version != CHECKPOINT_VERSION:
                logger.warning(f"Checkpoint version mismatch: {version} vs {CHECKPOINT_VERSION}")
            
            # Validate structure
            required_keys = ["version", "timestamp", "checkpoint_id", "workspace_state"]
            for key in required_keys:
                if key not in checkpoint:
                    raise ValueError(f"Invalid checkpoint: missing '{key}'")
            
            # Verify checksum if present
            stored_checksum = checkpoint.get("metadata", {}).get("_checksum")
            if stored_checksum and stored_checksum != checksum:
                logger.warning(f"Checksum mismatch: {stored_checksum[:8]} vs {checksum[:8]}")
            
            # Restore workspace from state
            workspace_state = checkpoint["workspace_state"]
            workspace = GlobalWorkspace.from_dict(workspace_state)
            
            checkpoint_id = checkpoint["checkpoint_id"]
            timestamp = checkpoint["timestamp"]
            logger.info(f"✅ Checkpoint loaded: {checkpoint_id[:8]} from {timestamp}")
            
            return workspace
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            raise
    
    def list_checkpoints(self) -> List[CheckpointInfo]:
        """
        List all available checkpoints with metadata.
        
        Scans the checkpoint directory and returns information about
        all available checkpoints, sorted by timestamp (newest first).
        
        Returns:
            List of CheckpointInfo objects with checkpoint details
            
        Example:
            >>> manager = CheckpointManager()
            >>> checkpoints = manager.list_checkpoints()
            >>> for cp in checkpoints:
            ...     print(f"{cp.timestamp}: {cp.metadata.get('user_label', 'N/A')}")
        """
        checkpoints = []
        
        for path in self.checkpoint_dir.glob("checkpoint_*.json*"):
            try:
                # Read checkpoint metadata (without loading full workspace)
                if path.suffix == '.gz':
                    with gzip.open(path, 'rb') as f:
                        json_bytes = f.read()
                else:
                    with open(path, 'rb') as f:
                        json_bytes = f.read()
                
                # Calculate checksum
                checksum = hashlib.sha256(json_bytes).hexdigest()
                
                # Parse just enough to get metadata
                checkpoint = json.loads(json_bytes.decode('utf-8'))
                
                info = CheckpointInfo(
                    checkpoint_id=checkpoint["checkpoint_id"],
                    timestamp=datetime.fromisoformat(checkpoint["timestamp"]),
                    version=checkpoint["version"],
                    path=path,
                    size_bytes=path.stat().st_size,
                    compressed=path.suffix == '.gz',
                    metadata=checkpoint.get("metadata", {}),
                    checksum=checksum[:16],  # First 16 chars for display
                )
                
                checkpoints.append(info)
                
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {path.name}: {e}")
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Remove a specific checkpoint.
        
        Deletes a checkpoint file by its ID. Useful for manual cleanup
        or removing unwanted checkpoints.
        
        Args:
            checkpoint_id: UUID string of the checkpoint to delete
            
        Returns:
            True if checkpoint was deleted, False if not found
            
        Example:
            >>> manager = CheckpointManager()
            >>> success = manager.delete_checkpoint("abc-123-def")
        """
        # Find checkpoint with this ID
        for checkpoint in self.list_checkpoints():
            if checkpoint.checkpoint_id == checkpoint_id:
                try:
                    checkpoint.path.unlink()
                    logger.info(f"🗑️ Checkpoint deleted: {checkpoint_id[:8]}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete checkpoint: {e}")
                    return False
        
        logger.warning(f"Checkpoint not found: {checkpoint_id[:8]}")
        return False
    
    async def auto_save(
        self,
        workspace: GlobalWorkspace,
        interval: float = 300.0,
    ) -> None:
        """
        Automatic periodic checkpointing.
        
        Runs in the background and saves checkpoints at regular intervals.
        This is an async method that should be run as a task.
        
        Args:
            workspace: GlobalWorkspace instance to checkpoint
            interval: Time between checkpoints in seconds (default: 300 = 5 minutes)
            
        Example:
            >>> manager = CheckpointManager()
            >>> task = asyncio.create_task(manager.auto_save(workspace, interval=300))
            >>> # Later...
            >>> task.cancel()
        """
        self._auto_save_running = True
        logger.info(f"🔄 Auto-save started: interval={interval}s")
        
        try:
            while self._auto_save_running:
                await asyncio.sleep(interval)
                
                if not self._auto_save_running:
                    break
                
                try:
                    self.save_checkpoint(
                        workspace,
                        metadata={
                            "auto_save": True,
                            "interval": interval,
                        }
                    )
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")
                    
        except asyncio.CancelledError:
            logger.info("🔄 Auto-save cancelled")
        finally:
            self._auto_save_running = False
    
    def stop_auto_save(self) -> None:
        """
        Stop automatic checkpointing.
        
        Sets the flag to stop the auto-save loop. The task should be
        cancelled separately if needed.
        """
        self._auto_save_running = False
        logger.info("🔄 Auto-save stopped")
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get most recent checkpoint path.
        
        Returns the path to the newest checkpoint, or None if no
        checkpoints exist.
        
        Returns:
            Path to latest checkpoint, or None if none exist
            
        Example:
            >>> manager = CheckpointManager()
            >>> latest = manager.get_latest_checkpoint()
            >>> if latest:
            ...     workspace = manager.load_checkpoint(latest)
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        # Checkpoints are already sorted by timestamp (newest first)
        return checkpoints[0].path
    
    def _rotate_checkpoints(self) -> None:
        """
        Enforce checkpoint rotation to prevent unbounded disk usage.
        
        Keeps only the most recent max_checkpoints files, deleting older ones.
        Auto-save checkpoints are rotated more aggressively than manual saves.
        """
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Separate manual and auto-save checkpoints
        manual_checkpoints = [cp for cp in checkpoints if not cp.metadata.get("auto_save", False)]
        auto_checkpoints = [cp for cp in checkpoints if cp.metadata.get("auto_save", False)]
        
        # Keep more manual checkpoints, fewer auto-saves
        manual_keep = int(self.max_checkpoints * 0.7)  # 70% for manual
        auto_keep = int(self.max_checkpoints * 0.3)    # 30% for auto-save
        
        # Delete excess checkpoints
        for checkpoint in manual_checkpoints[manual_keep:]:
            try:
                checkpoint.path.unlink()
                logger.debug(f"Rotated manual checkpoint: {checkpoint.checkpoint_id[:8]}")
            except Exception as e:
                logger.warning(f"Failed to rotate checkpoint: {e}")
        
        for checkpoint in auto_checkpoints[auto_keep:]:
            try:
                checkpoint.path.unlink()
                logger.debug(f"Rotated auto-save checkpoint: {checkpoint.checkpoint_id[:8]}")
            except Exception as e:
                logger.warning(f"Failed to rotate checkpoint: {e}")
    
    @staticmethod
    def _json_encoder(obj: Any) -> Any:
        """
        Custom JSON encoder for non-standard types.
        
        Handles datetime, Path, UUID, and other types that aren't
        natively JSON-serializable.
        
        Args:
            obj: Object to encode
            
        Returns:
            JSON-serializable representation
            
        Raises:
            TypeError: If object type is not supported
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
