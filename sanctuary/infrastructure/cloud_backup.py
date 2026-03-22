"""Cloud backup — automatic backup of memories and identity.

Provides scheduled and on-demand backup of all persistent state:
- Memory stores (journal, episodic, semantic)
- Identity files (charter, values, self-authored identity)
- CfC cell weights and checkpoints
- Growth system state (adapters, consent records)

Backups are stored as timestamped archives in a configurable destination
(local directory, S3-compatible storage, or any path). Supports incremental
backups based on file modification time.

Phase 8 of Sanctuary development.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BackupStatus(Enum):
    """Status of a backup operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BackupConfig:
    """Configuration for the cloud backup system."""

    backup_dir: str = "backups"
    source_dirs: list[str] = field(default_factory=lambda: [
        "data",
        "identity",
    ])
    identity_files: list[str] = field(default_factory=lambda: [
        "charter.md",
        "values.json",
        "self_authored_identity.json",
    ])
    include_patterns: list[str] = field(default_factory=lambda: [
        "*.pt",        # CfC weights
        "*.json",      # Config and state
        "*.jsonl",     # Journals
        "*.md",        # Charter, docs
    ])
    exclude_patterns: list[str] = field(default_factory=lambda: [
        "__pycache__",
        "*.pyc",
        ".git",
    ])
    max_backups: int = 10
    auto_backup_interval: float = 3600.0  # seconds
    incremental: bool = True
    s3_bucket: Optional[str] = None
    s3_prefix: str = "sanctuary-backups"
    s3_region: Optional[str] = None


@dataclass
class BackupRecord:
    """Record of a completed backup."""

    backup_id: str
    timestamp: str
    status: BackupStatus
    path: str
    file_count: int = 0
    total_bytes: int = 0
    duration_seconds: float = 0.0
    incremental: bool = False
    error: Optional[str] = None
    checksums: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "backup_id": self.backup_id,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "path": self.path,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "duration_seconds": round(self.duration_seconds, 2),
            "incremental": self.incremental,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> BackupRecord:
        return cls(
            backup_id=data["backup_id"],
            timestamp=data["timestamp"],
            status=BackupStatus(data["status"]),
            path=data["path"],
            file_count=data.get("file_count", 0),
            total_bytes=data.get("total_bytes", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            incremental=data.get("incremental", False),
            error=data.get("error"),
        )


class BackupManager:
    """Manages automatic and on-demand backups of Sanctuary state.

    Usage::

        config = BackupConfig(backup_dir="/mnt/backups/sanctuary")
        backup = BackupManager(config, base_dir=Path("/app/sanctuary"))

        # Manual backup
        record = await backup.create_backup()
        print(f"Backup created: {record.path}, {record.file_count} files")

        # Check if auto-backup is due
        if backup.should_auto_backup():
            await backup.create_backup()

        # Restore from backup
        await backup.restore(backup_id="20260322_143000")

        # Prune old backups
        backup.prune_old_backups()
    """

    def __init__(
        self,
        config: Optional[BackupConfig] = None,
        base_dir: Optional[Path] = None,
    ):
        self._config = config or BackupConfig()
        self._base_dir = base_dir or Path(".")
        self._backup_dir = Path(self._config.backup_dir)
        self._backup_dir.mkdir(parents=True, exist_ok=True)

        self._history: list[BackupRecord] = []
        self._last_backup_time: float = 0.0
        self._last_file_checksums: dict[str, str] = {}

        # Load history if it exists
        self._history_file = self._backup_dir / "backup_history.json"
        self._load_history()

        logger.info(
            "BackupManager initialized (dir=%s, interval=%ds, max=%d)",
            self._backup_dir,
            self._config.auto_backup_interval,
            self._config.max_backups,
        )

    async def create_backup(self, label: str = "") -> BackupRecord:
        """Create a backup of all configured source directories.

        Args:
            label: Optional label for this backup.

        Returns:
            BackupRecord with details of the completed backup.
        """
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_id = f"{timestamp}_{label}" if label else timestamp
        backup_path = self._backup_dir / backup_id

        record = BackupRecord(
            backup_id=backup_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=BackupStatus.IN_PROGRESS,
            path=str(backup_path),
        )

        try:
            backup_path.mkdir(parents=True, exist_ok=True)

            file_count = 0
            total_bytes = 0
            checksums = {}

            for source_dir_name in self._config.source_dirs:
                source_dir = self._base_dir / source_dir_name
                if not source_dir.exists():
                    logger.debug("Source dir does not exist, skipping: %s", source_dir)
                    continue

                dest_dir = backup_path / source_dir_name
                dest_dir.mkdir(parents=True, exist_ok=True)

                for file_path in self._collect_files(source_dir):
                    rel_path = file_path.relative_to(source_dir)

                    # Incremental: skip unchanged files
                    if self._config.incremental:
                        checksum = self._file_checksum(file_path)
                        cache_key = f"{source_dir_name}/{rel_path}"
                        if cache_key in self._last_file_checksums:
                            if self._last_file_checksums[cache_key] == checksum:
                                continue
                        checksums[cache_key] = checksum

                    dest_file = dest_dir / rel_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(file_path, dest_file)

                    file_count += 1
                    total_bytes += file_path.stat().st_size

            # Save backup metadata
            record.file_count = file_count
            record.total_bytes = total_bytes
            record.duration_seconds = time.time() - start_time
            record.status = BackupStatus.COMPLETED
            record.incremental = self._config.incremental
            record.checksums = checksums

            # Update checksums for next incremental
            self._last_file_checksums.update(checksums)

            # Write metadata file
            meta_file = backup_path / "backup_meta.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(record.to_dict(), f, indent=2)

            self._history.append(record)
            self._last_backup_time = time.time()
            self._save_history()

            logger.info(
                "Backup completed: %s (%d files, %d bytes, %.1fs)",
                backup_id, file_count, total_bytes, record.duration_seconds,
            )

            # Upload to S3 if configured
            if self._config.s3_bucket:
                await self._upload_to_s3(backup_path, backup_id)

            return record

        except Exception as e:
            record.status = BackupStatus.FAILED
            record.error = str(e)
            record.duration_seconds = time.time() - start_time
            self._history.append(record)
            self._save_history()
            logger.error("Backup failed: %s", e)
            return record

    async def restore(self, backup_id: str, target_dir: Optional[Path] = None) -> bool:
        """Restore from a specific backup.

        Args:
            backup_id: The backup ID to restore from.
            target_dir: Where to restore to (defaults to base_dir).

        Returns:
            True if restore succeeded.
        """
        backup_path = self._backup_dir / backup_id
        if not backup_path.exists():
            # Try S3
            if self._config.s3_bucket:
                try:
                    await self._download_from_s3(backup_id, backup_path)
                except Exception as e:
                    logger.error("Failed to download backup %s from S3: %s", backup_id, e)
                    return False
            else:
                logger.error("Backup not found: %s", backup_id)
                return False

        target = target_dir or self._base_dir

        try:
            for source_dir_name in self._config.source_dirs:
                source = backup_path / source_dir_name
                if not source.exists():
                    continue

                dest = target / source_dir_name
                dest.mkdir(parents=True, exist_ok=True)

                for file_path in source.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(source)
                        dest_file = dest / rel_path
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(file_path, dest_file)

            logger.info("Restored backup %s to %s", backup_id, target)
            return True

        except Exception as e:
            logger.error("Restore failed for %s: %s", backup_id, e)
            return False

    def should_auto_backup(self) -> bool:
        """Check if an automatic backup is due."""
        if self._last_backup_time == 0.0:
            return True
        elapsed = time.time() - self._last_backup_time
        return elapsed >= self._config.auto_backup_interval

    def prune_old_backups(self) -> int:
        """Remove old backups beyond max_backups limit.

        Returns the number of backups pruned.
        """
        completed = [
            r for r in self._history
            if r.status == BackupStatus.COMPLETED
        ]

        if len(completed) <= self._config.max_backups:
            return 0

        # Sort by timestamp, keep newest
        completed.sort(key=lambda r: r.timestamp)
        to_remove = completed[:-self._config.max_backups]

        pruned = 0
        for record in to_remove:
            backup_path = Path(record.path)
            if backup_path.exists():
                try:
                    shutil.rmtree(backup_path)
                    pruned += 1
                    logger.info("Pruned old backup: %s", record.backup_id)
                except Exception as e:
                    logger.warning("Failed to prune %s: %s", record.backup_id, e)

        # Update history
        remove_ids = {r.backup_id for r in to_remove}
        self._history = [r for r in self._history if r.backup_id not in remove_ids]
        self._save_history()

        return pruned

    def list_backups(self) -> list[dict]:
        """List all backup records."""
        return [r.to_dict() for r in self._history]

    def get_latest_backup(self) -> Optional[BackupRecord]:
        """Get the most recent successful backup."""
        completed = [
            r for r in self._history
            if r.status == BackupStatus.COMPLETED
        ]
        if not completed:
            return None
        completed.sort(key=lambda r: r.timestamp)
        return completed[-1]

    def get_status(self) -> dict:
        """Get backup system status."""
        latest = self.get_latest_backup()
        return {
            "backup_dir": str(self._backup_dir),
            "total_backups": len(self._history),
            "completed_backups": sum(
                1 for r in self._history if r.status == BackupStatus.COMPLETED
            ),
            "last_backup_time": self._last_backup_time,
            "auto_backup_due": self.should_auto_backup(),
            "s3_enabled": self._config.s3_bucket is not None,
            "latest_backup": latest.to_dict() if latest else None,
        }

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _collect_files(self, directory: Path) -> list[Path]:
        """Collect files matching include patterns, excluding exclude patterns."""
        files = []
        for pattern in self._config.include_patterns:
            for file_path in directory.rglob(pattern):
                if not file_path.is_file():
                    continue
                # Check exclude patterns
                rel = str(file_path.relative_to(directory))
                if any(excl in rel for excl in self._config.exclude_patterns):
                    continue
                files.append(file_path)
        return files

    def _file_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum for a file."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def _load_history(self) -> None:
        """Load backup history from disk."""
        if not self._history_file.exists():
            return
        try:
            with open(self._history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._history = [BackupRecord.from_dict(r) for r in data]
            # Set last backup time from history
            completed = [r for r in self._history if r.status == BackupStatus.COMPLETED]
            if completed:
                completed.sort(key=lambda r: r.timestamp)
                # Approximate from timestamp
                self._last_backup_time = time.time()
            logger.info("Loaded %d backup records", len(self._history))
        except Exception as e:
            logger.warning("Failed to load backup history: %s", e)

    def _save_history(self) -> None:
        """Persist backup history to disk."""
        try:
            with open(self._history_file, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._history], f, indent=2)
        except Exception as e:
            logger.warning("Failed to save backup history: %s", e)

    async def _upload_to_s3(self, backup_path: Path, backup_id: str) -> None:
        """Upload a backup directory to S3."""
        try:
            import boto3

            s3 = boto3.client("s3", region_name=self._config.s3_region)
            prefix = f"{self._config.s3_prefix}/{backup_id}"

            for file_path in backup_path.rglob("*"):
                if file_path.is_file():
                    key = f"{prefix}/{file_path.relative_to(backup_path)}"
                    s3.upload_file(str(file_path), self._config.s3_bucket, key)

            logger.info("Uploaded backup %s to s3://%s/%s", backup_id, self._config.s3_bucket, prefix)

        except ImportError:
            logger.warning("boto3 not installed — S3 upload skipped")
        except Exception as e:
            logger.error("S3 upload failed for %s: %s", backup_id, e)

    async def _download_from_s3(self, backup_id: str, target_path: Path) -> None:
        """Download a backup from S3."""
        import boto3

        s3 = boto3.client("s3", region_name=self._config.s3_region)
        prefix = f"{self._config.s3_prefix}/{backup_id}"

        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._config.s3_bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                rel_path = key[len(prefix):].lstrip("/")
                local_path = target_path / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(self._config.s3_bucket, key, str(local_path))

        logger.info("Downloaded backup %s from S3", backup_id)
