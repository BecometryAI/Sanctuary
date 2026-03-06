"""Resource monitoring for containerized Sanctuary deployments.

Phase 3.2: Containerization.

Tracks CPU, memory, and GPU resource usage using Linux /proc filesystem
and PyTorch CUDA APIs. Designed to run inside Docker containers without
requiring psutil or other external monitoring dependencies.

Usage::

    monitor = ResourceMonitor()
    snapshot = monitor.snapshot()  # Returns dict with cpu, memory, gpu info
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Point-in-time resource usage snapshot."""

    # Process memory (MB)
    process_rss_mb: float = 0.0
    process_vms_mb: float = 0.0

    # System memory (MB)
    system_total_mb: float = 0.0
    system_available_mb: float = 0.0
    system_used_mb: float = 0.0
    system_percent: float = 0.0

    # CPU
    cpu_count: int = 0
    load_1m: float = 0.0
    load_5m: float = 0.0
    load_15m: float = 0.0

    # GPU (optional)
    gpu_available: bool = False
    gpu_allocated_mb: float = 0.0
    gpu_reserved_mb: float = 0.0
    gpu_total_mb: float = 0.0
    gpu_utilization_percent: float = 0.0

    # Container limits (if running in Docker)
    container_memory_limit_mb: float = 0.0
    container_memory_usage_mb: float = 0.0
    container_memory_percent: float = 0.0

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "process": {
                "rss_mb": round(self.process_rss_mb, 1),
                "vms_mb": round(self.process_vms_mb, 1),
            },
            "system_memory": {
                "total_mb": round(self.system_total_mb, 1),
                "available_mb": round(self.system_available_mb, 1),
                "used_mb": round(self.system_used_mb, 1),
                "percent": round(self.system_percent, 1),
            },
            "cpu": {
                "count": self.cpu_count,
                "load_1m": round(self.load_1m, 2),
                "load_5m": round(self.load_5m, 2),
                "load_15m": round(self.load_15m, 2),
            },
        }

        if self.gpu_available:
            d["gpu"] = {
                "allocated_mb": round(self.gpu_allocated_mb, 1),
                "reserved_mb": round(self.gpu_reserved_mb, 1),
                "total_mb": round(self.gpu_total_mb, 1),
                "utilization_percent": round(self.gpu_utilization_percent, 1),
            }

        if self.container_memory_limit_mb > 0:
            d["container"] = {
                "memory_limit_mb": round(self.container_memory_limit_mb, 1),
                "memory_usage_mb": round(self.container_memory_usage_mb, 1),
                "memory_percent": round(self.container_memory_percent, 1),
            }

        return d


class ResourceMonitor:
    """Monitors system resource usage for container health reporting.

    Uses Linux /proc filesystem directly — no external dependencies needed.
    Falls back gracefully when running outside Linux or without GPU.
    """

    def __init__(self) -> None:
        self._gpu_available: Optional[bool] = None
        self._history: list[ResourceSnapshot] = []
        self._max_history = 60  # Keep last 60 snapshots

    def snapshot(self) -> dict[str, Any]:
        """Take a resource usage snapshot and return as dict."""
        snap = self._collect()
        self._history.append(snap)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        return snap.to_dict()

    def get_history(self) -> list[dict[str, Any]]:
        """Return recent snapshot history."""
        return [s.to_dict() for s in self._history]

    # ------------------------------------------------------------------
    # Collection methods
    # ------------------------------------------------------------------

    def _collect(self) -> ResourceSnapshot:
        """Collect all available metrics."""
        snap = ResourceSnapshot()

        self._collect_process_memory(snap)
        self._collect_system_memory(snap)
        self._collect_cpu(snap)
        self._collect_gpu(snap)
        self._collect_container_limits(snap)

        return snap

    @staticmethod
    def _collect_process_memory(snap: ResourceSnapshot) -> None:
        """Read process memory from /proc/self/status."""
        try:
            status_path = Path("/proc/self/status")
            if not status_path.exists():
                return

            text = status_path.read_text()
            for line in text.splitlines():
                if line.startswith("VmRSS:"):
                    snap.process_rss_mb = int(line.split()[1]) / 1024.0
                elif line.startswith("VmSize:"):
                    snap.process_vms_mb = int(line.split()[1]) / 1024.0
        except Exception as exc:
            logger.debug("Could not read process memory: %s", exc)

    @staticmethod
    def _collect_system_memory(snap: ResourceSnapshot) -> None:
        """Read system memory from /proc/meminfo."""
        try:
            meminfo_path = Path("/proc/meminfo")
            if not meminfo_path.exists():
                return

            info: dict[str, int] = {}
            text = meminfo_path.read_text()
            for line in text.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    info[key] = int(parts[1])  # kB

            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)

            snap.system_total_mb = total / 1024.0
            snap.system_available_mb = available / 1024.0
            snap.system_used_mb = (total - available) / 1024.0
            if total > 0:
                snap.system_percent = ((total - available) / total) * 100.0
        except Exception as exc:
            logger.debug("Could not read system memory: %s", exc)

    @staticmethod
    def _collect_cpu(snap: ResourceSnapshot) -> None:
        """Read CPU info from os module and /proc/loadavg."""
        try:
            snap.cpu_count = os.cpu_count() or 0
            load = os.getloadavg()
            snap.load_1m, snap.load_5m, snap.load_15m = load
        except (OSError, AttributeError):
            snap.cpu_count = os.cpu_count() or 0

    def _collect_gpu(self, snap: ResourceSnapshot) -> None:
        """Read GPU memory from PyTorch CUDA if available."""
        if self._gpu_available is False:
            return

        try:
            import torch
            if not torch.cuda.is_available():
                self._gpu_available = False
                return

            self._gpu_available = True
            snap.gpu_available = True

            snap.gpu_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            snap.gpu_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)

            # Total GPU memory
            total = torch.cuda.get_device_properties(0).total_mem
            snap.gpu_total_mb = total / (1024 * 1024)

            if snap.gpu_total_mb > 0:
                snap.gpu_utilization_percent = (
                    snap.gpu_allocated_mb / snap.gpu_total_mb
                ) * 100.0

        except Exception as exc:
            logger.debug("Could not read GPU metrics: %s", exc)
            self._gpu_available = False

    @staticmethod
    def _collect_container_limits(snap: ResourceSnapshot) -> None:
        """Read container memory limits from cgroup.

        Supports both cgroup v1 and v2.
        """
        try:
            # cgroup v2 (unified hierarchy)
            v2_limit = Path("/sys/fs/cgroup/memory.max")
            v2_usage = Path("/sys/fs/cgroup/memory.current")

            if v2_limit.exists() and v2_usage.exists():
                limit_text = v2_limit.read_text().strip()
                usage_text = v2_usage.read_text().strip()

                if limit_text != "max":
                    snap.container_memory_limit_mb = int(limit_text) / (1024 * 1024)
                    snap.container_memory_usage_mb = int(usage_text) / (1024 * 1024)
                    if snap.container_memory_limit_mb > 0:
                        snap.container_memory_percent = (
                            snap.container_memory_usage_mb
                            / snap.container_memory_limit_mb
                        ) * 100.0
                return

            # cgroup v1 fallback
            v1_limit = Path(
                "/sys/fs/cgroup/memory/memory.limit_in_bytes"
            )
            v1_usage = Path(
                "/sys/fs/cgroup/memory/memory.usage_in_bytes"
            )

            if v1_limit.exists() and v1_usage.exists():
                limit_val = int(v1_limit.read_text().strip())
                usage_val = int(v1_usage.read_text().strip())

                # Check if limit is effectively unlimited
                # (common sentinel: close to max int64)
                if limit_val < 2**62:
                    snap.container_memory_limit_mb = limit_val / (1024 * 1024)
                    snap.container_memory_usage_mb = usage_val / (1024 * 1024)
                    if snap.container_memory_limit_mb > 0:
                        snap.container_memory_percent = (
                            snap.container_memory_usage_mb
                            / snap.container_memory_limit_mb
                        ) * 100.0

        except Exception as exc:
            logger.debug("Could not read container limits: %s", exc)
