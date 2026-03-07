"""Tests for the container health check HTTP server.

Phase 3.2: Containerization.
"""

import asyncio
import json

import pytest
import pytest_asyncio

from sanctuary.api.health import HealthServer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockRunner:
    """Minimal mock for SanctuaryRunner."""

    def __init__(self, booted: bool = True, running: bool = True, cycle_count: int = 42):
        self._booted = booted
        self._running = running
        self._cycle_count = cycle_count

    @property
    def running(self) -> bool:
        return self._running

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    def get_status(self) -> dict:
        return {
            "booted": self._booted,
            "running": self._running,
            "cycle_count": self._cycle_count,
            "model": "MockModel",
            "memory_store": "InMemoryStore",
            "active_goals": [],
            "authority_levels": {},
            "motor_stats": {"speech_emitted": 0, "errors": 0},
        }


class MockResourceMonitor:
    """Minimal mock for ResourceMonitor."""

    def snapshot(self) -> dict:
        return {
            "timestamp": 1000.0,
            "process": {"rss_mb": 256.0, "vms_mb": 512.0},
            "cpu": {"count": 4, "load_1m": 1.5},
            "system_memory": {"total_mb": 8192.0, "percent": 50.0},
        }


async def _http_get(host: str, port: int, path: str) -> tuple[int, dict]:
    """Send a simple HTTP GET and return (status_code, json_body)."""
    reader, writer = await asyncio.open_connection(host, port)
    request = f"GET {path} HTTP/1.1\r\nHost: {host}\r\n\r\n"
    writer.write(request.encode())
    await writer.drain()

    # Read response
    data = await asyncio.wait_for(reader.read(4096), timeout=5.0)
    writer.close()
    await writer.wait_closed()

    response = data.decode("utf-8")
    # Parse status code from first line
    status_line = response.split("\r\n")[0]
    status_code = int(status_line.split()[1])

    # Parse JSON body (after empty line)
    body_start = response.index("\r\n\r\n") + 4
    body = json.loads(response[body_start:])
    return status_code, body


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthServer:
    """Tests for HealthServer."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Health server starts and stops cleanly."""
        server = HealthServer(host="127.0.0.1", port=0)
        await server.start()
        assert server.running
        await server.stop()
        assert not server.running

    @pytest.mark.asyncio
    async def test_health_endpoint_healthy(self):
        """GET /health returns 200 when runner is booted and running."""
        runner = MockRunner(booted=True, running=True)
        server = HealthServer(runner=runner, host="127.0.0.1", port=0)
        await server.start()

        # Get the actual port assigned
        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/health")
            assert status == 200
            assert body["status"] == "healthy"
            assert body["cycle_count"] == 42
            assert body["booted"] is True
            assert "uptime_seconds" in body
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_health_endpoint_unhealthy(self):
        """GET /health returns 503 when runner is not running."""
        runner = MockRunner(booted=True, running=False)
        server = HealthServer(runner=runner, host="127.0.0.1", port=0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/health")
            assert status == 503
            assert body["status"] == "unhealthy"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_health_no_runner_is_healthy(self):
        """GET /health returns 200 when no runner (standalone mode)."""
        server = HealthServer(host="127.0.0.1", port=0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/health")
            assert status == 200
            assert body["status"] == "healthy"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_status_endpoint(self):
        """GET /status returns detailed system status."""
        runner = MockRunner()
        monitor = MockResourceMonitor()
        server = HealthServer(
            runner=runner, resource_monitor=monitor, host="127.0.0.1", port=0
        )
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/status")
            assert status == 200
            assert "runner" in body
            assert body["runner"]["booted"] is True
            assert body["runner"]["model"] == "MockModel"
            assert "resources" in body
            assert body["resources"]["cpu"]["count"] == 4
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """GET /metrics returns resource metrics."""
        monitor = MockResourceMonitor()
        server = HealthServer(resource_monitor=monitor, host="127.0.0.1", port=0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/metrics")
            assert status == 200
            assert "process" in body
            assert body["process"]["rss_mb"] == 256.0
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_404_for_unknown_path(self):
        """GET /unknown returns 404."""
        server = HealthServer(host="127.0.0.1", port=0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/unknown")
            assert status == 404
            assert "error" in body
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_boot_grace_period(self):
        """During boot grace period, not-yet-booted runner is healthy."""
        runner = MockRunner(booted=False, running=False)
        server = HealthServer(runner=runner, host="127.0.0.1", port=0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            # Within grace period (120s) — should still be healthy
            status, body = await _http_get("127.0.0.1", port, "/health")
            assert status == 200
            assert body["status"] == "healthy"
        finally:
            await server.stop()


class TestHealthServerEdgeCases:
    """Edge case tests for HealthServer."""

    @pytest.mark.asyncio
    async def test_status_with_runner_error(self):
        """GET /status handles runner errors gracefully."""

        class BrokenRunner:
            _booted = True
            running = True
            cycle_count = 0

            def get_status(self):
                raise RuntimeError("subsystem exploded")

        server = HealthServer(runner=BrokenRunner(), host="127.0.0.1", port=0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            status, body = await _http_get("127.0.0.1", port, "/status")
            assert status == 200
            assert "runner_error" in body
            assert "exploded" in body["runner_error"]
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_double_stop_is_safe(self):
        """Stopping twice is a no-op."""
        server = HealthServer(host="127.0.0.1", port=0)
        await server.start()
        await server.stop()
        await server.stop()  # Should not raise
