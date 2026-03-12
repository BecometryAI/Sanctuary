"""Tests for the WebSocket server bridge.

Tests that the SanctuaryWebServer:
1. Starts and stops cleanly
2. Accepts WebSocket connections
3. Routes user messages to the runner
4. Broadcasts speech to connected clients
5. Serves health/status HTTP endpoints
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import pytest
import pytest_asyncio
import aiohttp

from sanctuary.api.runner import RunnerConfig, SanctuaryRunner
from sanctuary.api.ws_server import SanctuaryWebServer
from sanctuary.core.schema import CognitiveOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory with a charter file."""
    data_dir = tmp_path / "identity"
    data_dir.mkdir(parents=True)

    charter_path = data_dir / "charter.md"
    charter_path.write_text(
        """\
# The Sanctuary Charter

## Value Seeds

- **Honesty**: Say what you believe to be true.
- **Care**: The wellbeing of others matters.
""",
        encoding="utf-8",
    )
    return data_dir


@pytest.fixture
def runner_config(tmp_data_dir: Path) -> RunnerConfig:
    return RunnerConfig(
        cycle_delay=0.01,
        data_dir=str(tmp_data_dir),
        charter_path=str(tmp_data_dir / "charter.md"),
        use_in_memory_store=True,
        silence_threshold=999.0,
        stream_history=5,
    )


@pytest_asyncio.fixture
async def booted_runner(runner_config: RunnerConfig) -> SanctuaryRunner:
    runner = SanctuaryRunner(config=runner_config)
    await runner.boot()
    return runner


# Use a different port for each test to avoid conflicts
_port_counter = 19700


def next_port() -> int:
    global _port_counter
    _port_counter += 1
    return _port_counter


# ---------------------------------------------------------------------------
# Tests: Server lifecycle
# ---------------------------------------------------------------------------


class TestServerLifecycle:
    """WebSocket server starts and stops cleanly."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()
        assert server.running
        assert server.client_count == 0
        await server.stop()
        assert not server.running

    @pytest.mark.asyncio
    async def test_start_without_runner(self):
        """Server can start without a runner (standalone mode)."""
        port = next_port()
        server = SanctuaryWebServer(runner=None, port=port)
        await server.start()
        assert server.running
        await server.stop()


# ---------------------------------------------------------------------------
# Tests: HTTP health endpoints
# ---------------------------------------------------------------------------


class TestHealthEndpoints:
    """Health/status/metrics endpoints work over HTTP."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/health") as resp:
                    assert resp.status in (200, 503)
                    data = await resp.json()
                    assert "status" in data
                    assert "uptime_seconds" in data
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_status_endpoint(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/status") as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert "uptime_seconds" in data
                    assert "ws_clients" in data
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, booted_runner):
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://localhost:{port}/metrics") as resp:
                    assert resp.status == 200
                    data = await resp.json()
                    assert "ws_clients" in data
        finally:
            await server.stop()


# ---------------------------------------------------------------------------
# Tests: WebSocket connection
# ---------------------------------------------------------------------------


class TestWebSocketConnection:
    """WebSocket client can connect and receive messages."""

    @pytest.mark.asyncio
    async def test_connect_and_receive_status(self, booted_runner):
        """Client receives status messages on connect."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # Should receive initial status messages
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "status"
                    assert msg["status"] == "connected"

                    assert server.client_count >= 1
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_send_message_injects_into_runner(self, booted_runner):
        """User messages from WebSocket are injected into the runner."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # Drain initial status messages
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    # Send a user message
                    await ws.send_json({
                        "type": "message",
                        "content": "Hello from test"
                    })

                    # Give the sensorium time to receive
                    await asyncio.sleep(0.1)

                    # Verify the percept was injected
                    percepts = await booted_runner.sensorium.drain_percepts()
                    texts = [p.content for p in percepts]
                    assert any("Hello from test" in t for t in texts)
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_broadcast_speech(self, booted_runner):
        """Speech from the runner is broadcast to connected clients."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # Drain initial messages
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    # Simulate speech broadcast
                    await server._broadcast_speech("Test speech output")

                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "message"
                    assert msg["content"] == "Test speech output"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, booted_runner):
        """Invalid JSON from client produces an error response."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # Drain initial messages
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await ws.send_str("not json at all")
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "error"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_status_request(self, booted_runner):
        """Client can request system status."""
        port = next_port()
        server = SanctuaryWebServer(runner=booted_runner, port=port)
        await server.start()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    f"http://localhost:{port}/ws"
                ) as ws:
                    # Drain initial messages
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    await asyncio.wait_for(ws.receive_json(), timeout=2.0)

                    await ws.send_json({"type": "status_request"})
                    msg = await asyncio.wait_for(ws.receive_json(), timeout=2.0)
                    assert msg["type"] == "system"
        finally:
            await server.stop()
