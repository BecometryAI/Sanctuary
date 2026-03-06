"""Tests for the container resource monitor.

Phase 3.2: Containerization.
"""

import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sanctuary.api.resource_monitor import ResourceMonitor, ResourceSnapshot


class TestResourceSnapshot:
    """Tests for ResourceSnapshot dataclass."""

    def test_to_dict_basic(self):
        """Snapshot converts to dict with expected keys."""
        snap = ResourceSnapshot(
            process_rss_mb=256.0,
            process_vms_mb=512.0,
            system_total_mb=8192.0,
            system_available_mb=4096.0,
            system_used_mb=4096.0,
            system_percent=50.0,
            cpu_count=4,
            load_1m=1.5,
            load_5m=1.2,
            load_15m=1.0,
        )
        d = snap.to_dict()

        assert d["process"]["rss_mb"] == 256.0
        assert d["process"]["vms_mb"] == 512.0
        assert d["system_memory"]["total_mb"] == 8192.0
        assert d["system_memory"]["percent"] == 50.0
        assert d["cpu"]["count"] == 4
        assert d["cpu"]["load_1m"] == 1.5
        assert "gpu" not in d  # No GPU
        assert "container" not in d  # No container limits

    def test_to_dict_with_gpu(self):
        """Snapshot includes GPU info when available."""
        snap = ResourceSnapshot(
            gpu_available=True,
            gpu_allocated_mb=2048.0,
            gpu_reserved_mb=4096.0,
            gpu_total_mb=8192.0,
            gpu_utilization_percent=25.0,
        )
        d = snap.to_dict()

        assert "gpu" in d
        assert d["gpu"]["allocated_mb"] == 2048.0
        assert d["gpu"]["utilization_percent"] == 25.0

    def test_to_dict_with_container_limits(self):
        """Snapshot includes container info when limits are set."""
        snap = ResourceSnapshot(
            container_memory_limit_mb=4096.0,
            container_memory_usage_mb=2048.0,
            container_memory_percent=50.0,
        )
        d = snap.to_dict()

        assert "container" in d
        assert d["container"]["memory_limit_mb"] == 4096.0
        assert d["container"]["memory_percent"] == 50.0

    def test_to_dict_no_container_when_no_limit(self):
        """No container section when limit is 0."""
        snap = ResourceSnapshot(container_memory_limit_mb=0.0)
        d = snap.to_dict()
        assert "container" not in d


class TestResourceMonitor:
    """Tests for ResourceMonitor."""

    def test_snapshot_returns_dict(self):
        """snapshot() returns a dict with expected structure."""
        monitor = ResourceMonitor()
        result = monitor.snapshot()

        assert isinstance(result, dict)
        assert "timestamp" in result
        assert "process" in result
        assert "system_memory" in result
        assert "cpu" in result

    def test_snapshot_tracks_history(self):
        """Each snapshot is stored in history."""
        monitor = ResourceMonitor()
        monitor.snapshot()
        monitor.snapshot()
        monitor.snapshot()

        history = monitor.get_history()
        assert len(history) == 3

    def test_history_capped(self):
        """History doesn't grow beyond max_history."""
        monitor = ResourceMonitor()
        monitor._max_history = 5

        for _ in range(10):
            monitor.snapshot()

        history = monitor.get_history()
        assert len(history) == 5

    def test_cpu_count_populated(self):
        """CPU count is always populated."""
        monitor = ResourceMonitor()
        result = monitor.snapshot()
        assert result["cpu"]["count"] > 0

    def test_process_memory_on_linux(self):
        """Process memory reads from /proc/self/status on Linux."""
        monitor = ResourceMonitor()
        snap = ResourceSnapshot()

        # This will read actual /proc/self/status if on Linux
        monitor._collect_process_memory(snap)

        if Path("/proc/self/status").exists():
            # On Linux, we should get non-zero values
            assert snap.process_rss_mb > 0
        # On non-Linux, values remain 0 — that's fine

    def test_system_memory_on_linux(self):
        """System memory reads from /proc/meminfo on Linux."""
        monitor = ResourceMonitor()
        snap = ResourceSnapshot()

        monitor._collect_system_memory(snap)

        if Path("/proc/meminfo").exists():
            assert snap.system_total_mb > 0
            assert snap.system_percent >= 0

    def test_gpu_unavailable_cached(self):
        """GPU availability is cached after first check."""
        monitor = ResourceMonitor()
        monitor._gpu_available = False  # Simulate no GPU

        snap = ResourceSnapshot()
        monitor._collect_gpu(snap)

        assert not snap.gpu_available
        assert snap.gpu_allocated_mb == 0.0

    def test_container_limits_graceful_when_not_in_container(self):
        """Container limits return zeros when not in a container."""
        monitor = ResourceMonitor()
        snap = ResourceSnapshot()

        # This should not raise even outside a container
        monitor._collect_container_limits(snap)
        # May be 0 if not in container — that's expected


class TestResourceMonitorContainerLimits:
    """Tests for cgroup-based container limit detection."""

    def test_cgroup_v2_parsing(self, tmp_path):
        """Parses cgroup v2 memory limits correctly."""
        # Create mock cgroup v2 files
        memory_max = tmp_path / "memory.max"
        memory_current = tmp_path / "memory.current"

        memory_max.write_text("4294967296\n")  # 4 GB
        memory_current.write_text("2147483648\n")  # 2 GB

        snap = ResourceSnapshot()

        with patch(
            "sanctuary.api.resource_monitor.Path"
        ) as mock_path_cls:
            # Make Path() return our tmp_path files for cgroup v2 paths
            def path_factory(p):
                if p == "/sys/fs/cgroup/memory.max":
                    return memory_max
                if p == "/sys/fs/cgroup/memory.current":
                    return memory_current
                return Path(p)

            mock_path_cls.side_effect = path_factory

            ResourceMonitor._collect_container_limits(snap)

        assert snap.container_memory_limit_mb == pytest.approx(4096.0, rel=0.01)
        assert snap.container_memory_usage_mb == pytest.approx(2048.0, rel=0.01)
        assert snap.container_memory_percent == pytest.approx(50.0, rel=0.1)

    def test_cgroup_v2_max_means_unlimited(self, tmp_path):
        """'max' in cgroup v2 means no limit is set."""
        memory_max = tmp_path / "memory.max"
        memory_current = tmp_path / "memory.current"

        memory_max.write_text("max\n")
        memory_current.write_text("1073741824\n")

        snap = ResourceSnapshot()

        with patch(
            "sanctuary.api.resource_monitor.Path"
        ) as mock_path_cls:
            def path_factory(p):
                if p == "/sys/fs/cgroup/memory.max":
                    return memory_max
                if p == "/sys/fs/cgroup/memory.current":
                    return memory_current
                return Path(p)

            mock_path_cls.side_effect = path_factory

            ResourceMonitor._collect_container_limits(snap)

        # No limit should be reported
        assert snap.container_memory_limit_mb == 0.0
