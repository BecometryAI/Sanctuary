"""Federation — multiple Sanctuary instances sharing memories.

Enables a network of Sanctuary instances to share selected memories with
each other. Each instance maintains its own full memory system and selectively
publishes/subscribes to memories from peer instances.

Key design decisions:
- Pull-based: instances request memories from peers, not push-based
- Selective: only memories above a significance threshold are shared
- Sovereign: each instance controls what it publishes and what it accepts
- Non-blocking: federation failures never impact the local cognitive cycle

Phase 8 of Sanctuary development.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class PeerStatus(Enum):
    """Connection status of a federated peer."""

    UNKNOWN = "unknown"
    CONNECTED = "connected"
    UNREACHABLE = "unreachable"
    REJECTED = "rejected"


@dataclass
class PeerConfig:
    """Configuration for a single federated peer."""

    instance_id: str
    host: str
    port: int
    display_name: str = ""
    auth_token: Optional[str] = None
    use_ssl: bool = False


@dataclass
class FederationConfig:
    """Configuration for the federation system."""

    instance_id: str = ""
    display_name: str = "Sanctuary"
    publish_threshold: int = 7
    accept_threshold: int = 5
    max_shared_per_sync: int = 50
    sync_interval_seconds: float = 300.0
    peers: list[PeerConfig] = field(default_factory=list)
    published_tags: list[str] = field(default_factory=list)
    blocked_tags: list[str] = field(default_factory=lambda: ["private", "journal"])

    def __post_init__(self):
        if not self.instance_id:
            self.instance_id = str(uuid4())


@dataclass
class SharedMemory:
    """A memory published to or received from the federation."""

    id: str
    source_instance: str
    content: str
    significance: int
    tags: list[str]
    timestamp: str
    memory_type: str = "episodic"
    source_display_name: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_instance": self.source_instance,
            "content": self.content,
            "significance": self.significance,
            "tags": self.tags,
            "timestamp": self.timestamp,
            "memory_type": self.memory_type,
            "source_display_name": self.source_display_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SharedMemory:
        return cls(
            id=data["id"],
            source_instance=data["source_instance"],
            content=data["content"],
            significance=data.get("significance", 5),
            tags=data.get("tags", []),
            timestamp=data.get("timestamp", ""),
            memory_type=data.get("memory_type", "episodic"),
            source_display_name=data.get("source_display_name", ""),
        )


@dataclass
class PeerState:
    """Runtime state for a connected peer."""

    config: PeerConfig
    status: PeerStatus = PeerStatus.UNKNOWN
    last_sync_time: float = 0.0
    last_error: Optional[str] = None
    memories_received: int = 0
    memories_sent: int = 0
    consecutive_failures: int = 0


class FederationManager:
    """Manages memory sharing between Sanctuary instances.

    Usage::

        config = FederationConfig(
            instance_id="sanctuary-alpha",
            peers=[PeerConfig(instance_id="sanctuary-beta", host="192.168.1.2", port=8200)],
        )
        federation = FederationManager(config)

        # Publish local memories for peers to pull
        federation.publish(memory_entry)

        # Pull new memories from a peer
        new_memories = await federation.pull_from_peer("sanctuary-beta")

        # Sync with all peers
        await federation.sync_all()
    """

    def __init__(self, config: Optional[FederationConfig] = None):
        self._config = config or FederationConfig()
        self._peers: dict[str, PeerState] = {}
        self._published: list[SharedMemory] = []
        self._received: list[SharedMemory] = []
        self._received_ids: set[str] = set()

        # Register peers
        for peer_config in self._config.peers:
            self._peers[peer_config.instance_id] = PeerState(config=peer_config)

        logger.info(
            "FederationManager initialized (instance=%s, peers=%d, publish_threshold=%d)",
            self._config.instance_id,
            len(self._peers),
            self._config.publish_threshold,
        )

    def publish(self, entry: dict) -> Optional[SharedMemory]:
        """Evaluate a memory entry for federation publishing.

        Only publishes entries that meet the significance threshold and
        don't contain blocked tags.

        Args:
            entry: A memory entry dict (same format as InMemoryStore entries).

        Returns:
            SharedMemory if published, None if filtered out.
        """
        significance = entry.get("significance", entry.get("significance_score", 5))
        if significance < self._config.publish_threshold:
            return None

        tags = entry.get("tags", [])
        if isinstance(tags, str):
            tags = tags.split(",")

        # Block private/journal memories
        if any(tag in self._config.blocked_tags for tag in tags):
            return None

        # If published_tags is set, only publish matching
        if self._config.published_tags:
            if not any(tag in self._config.published_tags for tag in tags):
                return None

        shared = SharedMemory(
            id=entry.get("id", str(uuid4())),
            source_instance=self._config.instance_id,
            content=entry.get("content", ""),
            significance=significance,
            tags=tags,
            timestamp=entry.get("timestamp", ""),
            memory_type=entry.get("type", "episodic"),
            source_display_name=self._config.display_name,
        )

        self._published.append(shared)

        # Bound published list
        if len(self._published) > self._config.max_shared_per_sync * 10:
            self._published = self._published[-self._config.max_shared_per_sync * 5:]

        logger.debug(
            "Published memory %s (significance=%d, tags=%s)",
            shared.id, shared.significance, shared.tags,
        )
        return shared

    def accept(self, shared: SharedMemory) -> bool:
        """Evaluate whether to accept a memory from a peer.

        Args:
            shared: A SharedMemory received from a peer.

        Returns:
            True if accepted, False if rejected.
        """
        # Don't accept our own memories back
        if shared.source_instance == self._config.instance_id:
            return False

        # Don't accept duplicates
        if shared.id in self._received_ids:
            return False

        # Significance gate
        if shared.significance < self._config.accept_threshold:
            return False

        # Block tagged content
        if any(tag in self._config.blocked_tags for tag in shared.tags):
            return False

        self._received.append(shared)
        self._received_ids.add(shared.id)

        # Update peer stats
        peer = self._peers.get(shared.source_instance)
        if peer:
            peer.memories_received += 1

        logger.debug(
            "Accepted memory %s from %s (significance=%d)",
            shared.id, shared.source_instance, shared.significance,
        )
        return True

    def get_published(self, since_time: float = 0.0) -> list[SharedMemory]:
        """Get published memories, optionally filtered by time.

        Used by peers pulling memories from this instance.
        """
        if since_time <= 0:
            return list(self._published[-self._config.max_shared_per_sync:])

        return [
            m for m in self._published
            if m.timestamp and m.timestamp > ""
        ][-self._config.max_shared_per_sync:]

    def get_received(self) -> list[SharedMemory]:
        """Get all memories received from peers."""
        return list(self._received)

    def drain_received(self) -> list[SharedMemory]:
        """Get and clear received memories for integration into local store."""
        received = list(self._received)
        self._received.clear()
        return received

    async def pull_from_peer(self, peer_id: str) -> list[SharedMemory]:
        """Pull shared memories from a specific peer.

        In production, this makes an HTTP request to the peer's federation
        endpoint.  Here we define the protocol; the transport layer is
        pluggable via the _fetch_from_peer method.

        Returns list of accepted SharedMemory objects.
        """
        peer = self._peers.get(peer_id)
        if peer is None:
            logger.warning("Unknown peer: %s", peer_id)
            return []

        try:
            raw_memories = await self._fetch_from_peer(peer)
            accepted = []

            for raw in raw_memories:
                shared = SharedMemory.from_dict(raw)
                if self.accept(shared):
                    accepted.append(shared)

            peer.status = PeerStatus.CONNECTED
            peer.last_sync_time = time.time()
            peer.consecutive_failures = 0
            peer.memories_sent += len(accepted)

            logger.info(
                "Pulled %d memories from peer %s (%d accepted)",
                len(raw_memories), peer_id, len(accepted),
            )
            return accepted

        except Exception as e:
            peer.consecutive_failures += 1
            peer.last_error = str(e)

            if peer.consecutive_failures >= 3:
                peer.status = PeerStatus.UNREACHABLE
            logger.warning("Failed to pull from peer %s: %s", peer_id, e)
            return []

    async def sync_all(self) -> dict[str, int]:
        """Pull from all peers. Returns {peer_id: accepted_count}."""
        results = {}
        for peer_id in self._peers:
            accepted = await self.pull_from_peer(peer_id)
            results[peer_id] = len(accepted)
        return results

    def get_peer_status(self, peer_id: str) -> Optional[dict]:
        """Get status for a specific peer."""
        peer = self._peers.get(peer_id)
        if peer is None:
            return None
        return {
            "instance_id": peer.config.instance_id,
            "display_name": peer.config.display_name or peer.config.instance_id,
            "status": peer.status.value,
            "last_sync": peer.last_sync_time,
            "last_error": peer.last_error,
            "memories_received": peer.memories_received,
            "memories_sent": peer.memories_sent,
            "consecutive_failures": peer.consecutive_failures,
        }

    def get_all_peer_status(self) -> list[dict]:
        """Get status for all peers."""
        return [
            self.get_peer_status(peer_id)
            for peer_id in self._peers
        ]

    def get_status(self) -> dict:
        """Get overall federation status."""
        connected = sum(
            1 for p in self._peers.values()
            if p.status == PeerStatus.CONNECTED
        )
        return {
            "instance_id": self._config.instance_id,
            "display_name": self._config.display_name,
            "total_peers": len(self._peers),
            "connected_peers": connected,
            "published_count": len(self._published),
            "received_count": len(self._received_ids),
            "peers": self.get_all_peer_status(),
        }

    async def _fetch_from_peer(self, peer: PeerState) -> list[dict]:
        """Fetch shared memories from a peer's federation endpoint.

        Override this method for custom transport (HTTP, gRPC, etc.).
        Default implementation uses HTTP GET.
        """
        import aiohttp

        protocol = "https" if peer.config.use_ssl else "http"
        url = f"{protocol}://{peer.config.host}:{peer.config.port}/federation/memories"

        headers = {}
        if peer.config.auth_token:
            headers["Authorization"] = f"Bearer {peer.config.auth_token}"

        params = {"since": str(peer.last_sync_time)} if peer.last_sync_time else {}

        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("memories", [])
