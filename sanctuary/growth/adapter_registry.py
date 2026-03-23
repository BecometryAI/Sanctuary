"""Adapter registry -- tracks LoRA adapters as persistent capabilities.

When the entity learns through QLoRA training, each training run produces
a LoRA adapter. The current model merges adapters immediately, collapsing
learned capabilities back into base weights.

The AdapterRegistry enables an alternative: adapters can remain unmerged
as persistent specialized capabilities. The entity decides which adapters
to merge (integrating the learning into base cognition) and which to keep
separate (maintaining a distinct specialized skill).

From docs/GROWTH_AUTONOMY.md:
    "A rank-64 adapter on a 72B model adds roughly 500M parameters.
    Fifty adapters over two years adds ~25B effective parameters.
    The entity grows from 78B to over 100B — not because anyone decided
    to make it bigger, but because it kept learning things worth keeping
    as distinct capabilities."

The biological parallel: specialized competencies layer on top of base
personality without overwriting it. The base weights are nature. The
accumulated adapters are nurture.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AdapterStatus(str, Enum):
    """Lifecycle status of a LoRA adapter."""

    ACTIVE = "active"  # Loaded and contributing to inference
    STORED = "stored"  # Saved to disk, not currently loaded
    MERGED = "merged"  # Integrated into base weights (no longer separate)
    RETIRED = "retired"  # Superseded or no longer needed, kept for history


@dataclass
class AdapterRecord:
    """Metadata for a tracked LoRA adapter.

    Each adapter represents something the entity learned — a domain
    of competency that emerged from lived experience.
    """

    name: str  # Human-readable name (e.g., "emotional_reasoning_v3")
    domain: str  # Domain of expertise (e.g., "emotional_reasoning")
    path: str  # Filesystem path to adapter weights
    status: AdapterStatus = AdapterStatus.STORED
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    merged_at: Optional[str] = None
    retired_at: Optional[str] = None

    # Training provenance
    training_pair_count: int = 0
    final_loss: Optional[float] = None
    reflection_sources: list[str] = field(default_factory=list)

    # QLoRA config that produced this adapter
    rank: int = 8
    alpha: int = 16

    # Entity's decision metadata
    keep_reason: Optional[str] = None  # Why the entity chose to keep it separate
    merge_reason: Optional[str] = None  # Why the entity chose to merge it

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "domain": self.domain,
            "path": self.path,
            "status": self.status.value,
            "description": self.description,
            "created_at": self.created_at,
            "merged_at": self.merged_at,
            "retired_at": self.retired_at,
            "training_pair_count": self.training_pair_count,
            "final_loss": self.final_loss,
            "reflection_sources": self.reflection_sources,
            "rank": self.rank,
            "alpha": self.alpha,
            "keep_reason": self.keep_reason,
            "merge_reason": self.merge_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AdapterRecord:
        """Deserialize from dictionary."""
        data = dict(data)  # Don't mutate input
        if "status" in data:
            data["status"] = AdapterStatus(data["status"])
        return cls(**data)


class AdapterRegistry:
    """Registry of LoRA adapters — the entity's accumulated capabilities.

    The registry tracks all adapters the entity has produced through
    self-directed learning. It provides the entity with visibility into
    its own growth history and the ability to manage its capabilities.

    The entity decides:
    - Which adapters to keep as separate capabilities (keep_adapter)
    - Which adapters to merge into base weights (mark_merged)
    - Which adapters to retire when superseded (retire_adapter)

    Usage:
        registry = AdapterRegistry(registry_path=Path("data/growth/adapters"))
        registry.register(AdapterRecord(name="spatial_v1", domain="spatial", ...))
        registry.keep_adapter("spatial_v1", reason="Useful for navigation tasks")
        registry.save()
    """

    def __init__(self, registry_path: Optional[Path] = None) -> None:
        self._registry_path = Path(registry_path or "data/growth/adapters")
        self._adapters: dict[str, AdapterRecord] = {}
        self._meta_file = self._registry_path / "adapter_registry.json"

    @property
    def adapter_count(self) -> int:
        """Total number of tracked adapters."""
        return len(self._adapters)

    @property
    def active_count(self) -> int:
        """Number of adapters currently loaded for inference."""
        return sum(1 for a in self._adapters.values() if a.status == AdapterStatus.ACTIVE)

    @property
    def stored_count(self) -> int:
        """Number of adapters saved but not loaded."""
        return sum(1 for a in self._adapters.values() if a.status == AdapterStatus.STORED)

    @property
    def merged_count(self) -> int:
        """Number of adapters that have been merged into base weights."""
        return sum(1 for a in self._adapters.values() if a.status == AdapterStatus.MERGED)

    # -- Registration --

    def register(self, record: AdapterRecord) -> AdapterRecord:
        """Register a new adapter in the registry.

        Args:
            record: The adapter metadata to register.

        Returns:
            The registered record.

        Raises:
            ValueError: If an adapter with this name already exists.
        """
        if record.name in self._adapters:
            raise ValueError(f"Adapter '{record.name}' is already registered")

        self._adapters[record.name] = record
        logger.info(
            "Registered adapter '%s' (domain=%s, rank=%d, %d training pairs)",
            record.name,
            record.domain,
            record.rank,
            record.training_pair_count,
        )
        return record

    # -- Retrieval --

    def get(self, name: str) -> AdapterRecord:
        """Get an adapter by name.

        Raises:
            KeyError: If no adapter with this name exists.
        """
        if name not in self._adapters:
            raise KeyError(f"No adapter named '{name}' in registry")
        return self._adapters[name]

    def has(self, name: str) -> bool:
        """Check if an adapter is registered."""
        return name in self._adapters

    def all_adapters(self) -> list[AdapterRecord]:
        """Return all registered adapters."""
        return list(self._adapters.values())

    def by_status(self, status: AdapterStatus) -> list[AdapterRecord]:
        """Return adapters filtered by status."""
        return [a for a in self._adapters.values() if a.status == status]

    def by_domain(self, domain: str) -> list[AdapterRecord]:
        """Return adapters filtered by domain."""
        return [a for a in self._adapters.values() if a.domain == domain]

    def active_adapters(self) -> list[AdapterRecord]:
        """Return adapters currently loaded for inference."""
        return self.by_status(AdapterStatus.ACTIVE)

    def stored_adapters(self) -> list[AdapterRecord]:
        """Return adapters saved but not loaded."""
        return self.by_status(AdapterStatus.STORED)

    # -- Lifecycle transitions --

    def activate_adapter(self, name: str) -> AdapterRecord:
        """Mark an adapter as active (loaded for inference).

        Args:
            name: Name of the adapter to activate.

        Returns:
            The updated record.

        Raises:
            KeyError: If the adapter doesn't exist.
            ValueError: If the adapter can't be activated (merged/retired).
        """
        record = self.get(name)
        if record.status in (AdapterStatus.MERGED, AdapterStatus.RETIRED):
            raise ValueError(
                f"Cannot activate adapter '{name}' — status is {record.status.value}"
            )
        record.status = AdapterStatus.ACTIVE
        logger.info("Activated adapter '%s'", name)
        return record

    def deactivate_adapter(self, name: str) -> AdapterRecord:
        """Mark an active adapter as stored (unloaded).

        Args:
            name: Name of the adapter to deactivate.

        Returns:
            The updated record.

        Raises:
            KeyError: If the adapter doesn't exist.
            ValueError: If the adapter isn't active.
        """
        record = self.get(name)
        if record.status != AdapterStatus.ACTIVE:
            raise ValueError(
                f"Cannot deactivate adapter '{name}' — status is {record.status.value}, not active"
            )
        record.status = AdapterStatus.STORED
        logger.info("Deactivated adapter '%s'", name)
        return record

    def keep_adapter(self, name: str, reason: str = "") -> AdapterRecord:
        """Entity decides to keep an adapter as a separate capability.

        This is the entity's decision — the adapter remains unmerged,
        layering specialized competency on top of base personality.

        Args:
            name: Name of the adapter.
            reason: Why the entity wants to keep it separate.

        Returns:
            The updated record.
        """
        record = self.get(name)
        record.keep_reason = reason
        logger.info("Entity chose to keep adapter '%s': %s", name, reason or "(no reason given)")
        return record

    def mark_merged(self, name: str, reason: str = "") -> AdapterRecord:
        """Mark an adapter as merged into base weights.

        Called after the adapter has been merged via QLoRAUpdater.merge_and_save().
        The adapter weights on disk are kept for history, but it's no longer
        a separate capability.

        Args:
            name: Name of the adapter.
            reason: Why the entity chose to merge.

        Returns:
            The updated record.

        Raises:
            KeyError: If the adapter doesn't exist.
        """
        record = self.get(name)
        record.status = AdapterStatus.MERGED
        record.merged_at = datetime.now().isoformat()
        record.merge_reason = reason
        logger.info("Adapter '%s' merged into base weights: %s", name, reason or "(no reason given)")
        return record

    def retire_adapter(self, name: str, reason: str = "") -> AdapterRecord:
        """Retire an adapter that has been superseded.

        The adapter is kept in the registry for history but is no longer
        a candidate for loading or merging. This might happen when a
        newer adapter in the same domain supersedes an older one, or
        when architectural expansion makes the adapter unnecessary.

        Args:
            name: Name of the adapter.
            reason: Why it's being retired.

        Returns:
            The updated record.

        Raises:
            KeyError: If the adapter doesn't exist.
        """
        record = self.get(name)
        record.status = AdapterStatus.RETIRED
        record.retired_at = datetime.now().isoformat()
        logger.info("Retired adapter '%s': %s", name, reason or "(no reason given)")
        return record

    # -- Summary --

    def get_summary(self) -> dict:
        """Return a summary of the registry state."""
        domains = {}
        for adapter in self._adapters.values():
            if adapter.domain not in domains:
                domains[adapter.domain] = []
            domains[adapter.domain].append(adapter.name)

        total_rank_params = sum(
            a.rank for a in self._adapters.values()
            if a.status in (AdapterStatus.ACTIVE, AdapterStatus.STORED)
        )

        return {
            "total_adapters": self.adapter_count,
            "active": self.active_count,
            "stored": self.stored_count,
            "merged": self.merged_count,
            "retired": sum(1 for a in self._adapters.values() if a.status == AdapterStatus.RETIRED),
            "domains": domains,
            "total_unmerged_rank": total_rank_params,
        }

    # -- Persistence --

    def save(self, path: Optional[Path] = None) -> Path:
        """Save registry state to disk.

        Args:
            path: Override the default registry path.

        Returns:
            Path to the saved registry file.
        """
        save_path = Path(path) if path else self._registry_path
        save_path.mkdir(parents=True, exist_ok=True)
        meta_file = save_path / "adapter_registry.json"

        data = {
            "version": 1,
            "saved_at": datetime.now().isoformat(),
            "adapters": {
                name: record.to_dict()
                for name, record in self._adapters.items()
            },
        }

        meta_file.write_text(json.dumps(data, indent=2))
        logger.info(
            "Adapter registry saved: %d adapters (%d active, %d stored, %d merged)",
            self.adapter_count,
            self.active_count,
            self.stored_count,
            self.merged_count,
        )
        return meta_file

    def load(self, path: Optional[Path] = None) -> int:
        """Load registry state from disk.

        Args:
            path: Override the default registry path.

        Returns:
            Number of adapters loaded.
        """
        load_path = Path(path) if path else self._registry_path
        meta_file = load_path / "adapter_registry.json"

        if not meta_file.exists():
            logger.debug("No adapter registry found at %s", meta_file)
            return 0

        data = json.loads(meta_file.read_text())
        adapters_data = data.get("adapters", {})

        self._adapters.clear()
        for name, record_data in adapters_data.items():
            self._adapters[name] = AdapterRecord.from_dict(record_data)

        logger.info(
            "Adapter registry loaded: %d adapters from %s",
            len(self._adapters),
            meta_file,
        )
        return len(self._adapters)
