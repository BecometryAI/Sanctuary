"""WebSocket server — bridges the Electron desktop GUI to SanctuaryRunner.

Provides a WebSocket endpoint that the Electron desktop app (or any WebSocket
client) connects to for bidirectional communication with the cognitive system.

Protocol:
    Client -> Server:
        { "type": "message", "content": "user text" }

    Server -> Client:
        { "type": "message", "content": "entity speech" }
        { "type": "status",  "status": "booted"|"cycling"|"stopped", "message": "..." }
        { "type": "inner",   "content": "inner speech text" }
        { "type": "system",  "content": "system status JSON" }

Also serves the health check HTTP endpoints on the same port:
    GET /health  — liveness check
    GET /status  — detailed system status
    GET /metrics — resource usage

Usage::

    from sanctuary.api.ws_server import SanctuaryWebServer

    server = SanctuaryWebServer(runner=runner, port=8765)
    await server.start()
    # ... server runs until stopped ...
    await server.stop()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import weakref
from typing import Any, Optional

from aiohttp import web, WSMsgType

logger = logging.getLogger(__name__)

DEFAULT_WS_PORT = 8765
DEFAULT_WS_HOST = "0.0.0.0"


class SanctuaryWebServer:
    """Combined HTTP + WebSocket server for the Sanctuary desktop app.

    Manages WebSocket connections and routes messages between the Electron
    GUI and the SanctuaryRunner. Multiple clients can connect simultaneously.
    """

    def __init__(
        self,
        runner: Any = None,
        host: str = DEFAULT_WS_HOST,
        port: int = DEFAULT_WS_PORT,
    ):
        self._runner = runner
        self._host = host
        self._port = port
        self._app: Optional[web.Application] = None
        self._runner_obj: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None
        self._start_time = time.monotonic()
        self._clients: weakref.WeakSet = weakref.WeakSet()

    async def start(self) -> None:
        """Start the HTTP + WebSocket server."""
        self._start_time = time.monotonic()
        self._app = web.Application()
        self._app.router.add_get("/ws", self._handle_websocket)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/status", self._handle_status)
        self._app.router.add_get("/metrics", self._handle_metrics)

        self._runner_obj = web.AppRunner(self._app)
        await self._runner_obj.setup()
        self._site = web.TCPSite(self._runner_obj, self._host, self._port)
        await self._site.start()

        # Wire speech handler to broadcast to all connected clients
        if self._runner:
            self._runner.on_speech(self._broadcast_speech)
            self._runner.on_output(self._broadcast_inner)

        logger.info(
            "WebSocket server listening on ws://%s:%d/ws", self._host, self._port
        )

    async def stop(self) -> None:
        """Stop the server and close all connections."""
        if self._runner_obj:
            await self._runner_obj.cleanup()
            self._runner_obj = None
            self._site = None
            self._app = None
            logger.info("WebSocket server stopped")

    @property
    def running(self) -> bool:
        return self._site is not None

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # ------------------------------------------------------------------
    # WebSocket handler
    # ------------------------------------------------------------------

    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle a WebSocket connection from the desktop app."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._clients.add(ws)
        logger.info(
            "WebSocket client connected (total: %d)", len(self._clients)
        )

        # Send initial status
        await self._send_ws(ws, {
            "type": "status",
            "status": "connected",
            "message": "Connected to Sanctuary",
        })

        # Send boot status if runner is available
        if self._runner:
            booted = getattr(self._runner, "_booted", False)
            status = "booted" if booted else "booting"
            await self._send_ws(ws, {
                "type": "status",
                "status": status,
                "message": f"Sanctuary is {status} (cycles: {self._runner.cycle_count})",
            })

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    await self._handle_client_message(ws, msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(
                        "WebSocket error: %s", ws.exception()
                    )
        finally:
            self._clients.discard(ws)
            logger.info(
                "WebSocket client disconnected (remaining: %d)",
                len(self._clients),
            )

        return ws

    async def _handle_client_message(
        self, ws: web.WebSocketResponse, raw: str
    ) -> None:
        """Process a message from a WebSocket client."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            await self._send_ws(ws, {
                "type": "error",
                "content": "Invalid JSON",
            })
            return

        msg_type = data.get("type", "")

        if msg_type == "message":
            content = data.get("content", "").strip()
            if content and self._runner:
                self._runner.inject_text(content, source="user:desktop")
                logger.debug("User input injected: %s", content[:100])
            elif not self._runner:
                await self._send_ws(ws, {
                    "type": "error",
                    "content": "Sanctuary is not running",
                })

        elif msg_type == "status_request":
            if self._runner:
                status = self._runner.get_status()
                await self._send_ws(ws, {
                    "type": "system",
                    "content": json.dumps(status, default=str),
                })

        else:
            await self._send_ws(ws, {
                "type": "error",
                "content": f"Unknown message type: {msg_type}",
            })

    # ------------------------------------------------------------------
    # Broadcasting to all clients
    # ------------------------------------------------------------------

    async def _broadcast_speech(self, text: str) -> None:
        """Broadcast entity speech to all connected WebSocket clients."""
        message = {"type": "message", "content": text}
        await self._broadcast(message)

    async def _broadcast_inner(self, output: Any) -> None:
        """Broadcast inner speech to all connected clients (for debug/visibility)."""
        if output.inner_speech:
            message = {"type": "inner", "content": output.inner_speech}
            await self._broadcast(message)

    async def _broadcast(self, message: dict) -> None:
        """Send a message to all connected clients."""
        payload = json.dumps(message)
        closed = []
        for ws in list(self._clients):
            try:
                await ws.send_str(payload)
            except (ConnectionError, RuntimeError):
                closed.append(ws)
        for ws in closed:
            self._clients.discard(ws)

    # ------------------------------------------------------------------
    # HTTP health endpoints (same as HealthServer but on the WS port)
    # ------------------------------------------------------------------

    async def _handle_health(self, request: web.Request) -> web.Response:
        """GET /health — liveness check."""
        healthy = self._check_healthy()
        uptime = time.monotonic() - self._start_time
        body = {
            "status": "healthy" if healthy else "unhealthy",
            "uptime_seconds": round(uptime, 1),
        }
        if self._runner:
            body["cycle_count"] = self._runner.cycle_count
            body["booted"] = getattr(self._runner, "_booted", False)
            body["ws_clients"] = len(self._clients)

        return web.json_response(body, status=200 if healthy else 503)

    async def _handle_status(self, request: web.Request) -> web.Response:
        """GET /status — detailed system status."""
        status: dict[str, Any] = {
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            "ws_clients": len(self._clients),
        }
        if self._runner:
            try:
                runner_status = self._runner.get_status()
                sanitized = {}
                for k, v in runner_status.items():
                    try:
                        json.dumps(v)
                        sanitized[k] = v
                    except (TypeError, ValueError):
                        sanitized[k] = str(v)
                status["runner"] = sanitized
            except Exception as exc:
                status["runner_error"] = str(exc)
        return web.json_response(status)

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """GET /metrics — resource usage metrics."""
        metrics: dict[str, Any] = {"ws_clients": len(self._clients)}
        if self._runner:
            metrics["cycle_count"] = self._runner.cycle_count
        return web.json_response(metrics)

    def _check_healthy(self) -> bool:
        """Determine if the system is healthy."""
        if not self._runner:
            return True
        booted = getattr(self._runner, "_booted", False)
        if not booted:
            uptime = time.monotonic() - self._start_time
            return uptime < 120.0
        return getattr(self._runner, "running", False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _send_ws(ws: web.WebSocketResponse, data: dict) -> None:
        """Send a JSON message to a single WebSocket client."""
        try:
            await ws.send_json(data)
        except (ConnectionError, RuntimeError):
            pass
