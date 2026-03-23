"""Dynamic CfC cell registry.

The registry manages all active CfC cells — both foundational cells (present
at boot) and knowledge cells (acquired through lived experience). It treats
all cells uniformly: no second-class citizens. The distinction between
foundational and knowledge cells is in their origin, not in how they integrate.

Design constraints from CFC_KNOWLEDGE_CELLS.md:
- No hardcoded cell type lists anywhere in the system
- The registry accepts new types at runtime
- Cell state persists across cycles, restarts, and checkpoints
- No upper limit on cell count (bounded by hardware, not software)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@runtime_checkable
class CellProtocol(Protocol):
    """Protocol that all CfC cells must implement.

    Both foundational cells and knowledge cells implement this interface.
    The ExperientialManager coordinates cells through this protocol only.
    """

    def step(self, **kwargs: float) -> Any:
        """Advance the cell by one step, returning its output."""
        ...

    def reset_hidden(self) -> None:
        """Reset the cell's hidden state."""
        ...

    def get_summary(self) -> dict:
        """Return summary statistics for monitoring."""
        ...

    def save(self, path: Path) -> None:
        """Save cell state to disk."""
        ...

    @classmethod
    def load(cls, path: Path) -> CellProtocol:
        """Load cell state from disk."""
        ...


@dataclass
class CellRegistration:
    """Metadata for a registered cell."""

    name: str
    cell: nn.Module  # The actual CfC cell (implements CellProtocol)
    category: str  # "foundational" or "knowledge"
    domain: str = ""  # Domain for knowledge cells (e.g., "spatial_reasoning")
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)
    connections_from: list[str] = field(default_factory=list)  # Names of cells that feed into this one
    connections_to: list[str] = field(default_factory=list)  # Names of cells this one feeds into
    metadata: dict = field(default_factory=dict)


@dataclass
class InterCellConnection:
    """A directed connection between two cells."""

    source_cell: str
    target_cell: str
    source_output: str  # Which output from the source
    target_input: str  # Which input on the target
    weight: float = 1.0  # Connection strength


class CellRegistry:
    """Dynamic registry of all active CfC cells.

    The registry is the single source of truth for which cells exist,
    how they connect, and how to persist them. The ExperientialManager
    delegates all cell management to the registry.

    Usage:
        registry = CellRegistry()
        registry.register("precision", precision_cell, category="foundational")
        registry.register("spatial", spatial_cell, category="knowledge", domain="spatial")
        for name, reg in registry.all_cells():
            cell = reg.cell
    """

    def __init__(self) -> None:
        self._cells: dict[str, CellRegistration] = {}
        self._connections: list[InterCellConnection] = []

    @property
    def cell_count(self) -> int:
        """Total number of registered cells."""
        return len(self._cells)

    @property
    def foundational_count(self) -> int:
        """Number of foundational cells."""
        return sum(1 for r in self._cells.values() if r.category == "foundational")

    @property
    def knowledge_count(self) -> int:
        """Number of knowledge cells."""
        return sum(1 for r in self._cells.values() if r.category == "knowledge")

    def register(
        self,
        name: str,
        cell: nn.Module,
        category: str = "knowledge",
        domain: str = "",
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        connections_from: Optional[list[str]] = None,
        connections_to: Optional[list[str]] = None,
        metadata: Optional[dict] = None,
    ) -> CellRegistration:
        """Register a cell in the registry.

        Args:
            name: Unique name for this cell.
            cell: The CfC cell (nn.Module implementing CellProtocol).
            category: "foundational" or "knowledge".
            domain: Domain for knowledge cells.
            input_names: Names of the cell's inputs.
            output_names: Names of the cell's outputs.
            connections_from: Names of cells that feed into this one.
            connections_to: Names of cells this one feeds into.
            metadata: Additional metadata.

        Returns:
            The CellRegistration record.

        Raises:
            ValueError: If a cell with this name already exists.
        """
        if name in self._cells:
            raise ValueError(f"Cell '{name}' is already registered")

        if category not in ("foundational", "knowledge"):
            raise ValueError(f"Category must be 'foundational' or 'knowledge', got '{category}'")

        registration = CellRegistration(
            name=name,
            cell=cell,
            category=category,
            domain=domain,
            input_names=input_names or [],
            output_names=output_names or [],
            connections_from=connections_from or [],
            connections_to=connections_to or [],
            metadata=metadata or {},
        )

        self._cells[name] = registration
        logger.info(
            "Registered %s cell '%s' (domain=%s, %d params)",
            category,
            name,
            domain or "n/a",
            sum(p.numel() for p in cell.parameters()),
        )

        return registration

    def unregister(self, name: str) -> CellRegistration:
        """Remove a cell from the registry.

        Args:
            name: Name of the cell to remove.

        Returns:
            The removed CellRegistration.

        Raises:
            KeyError: If no cell with this name exists.
        """
        if name not in self._cells:
            raise KeyError(f"No cell named '{name}' in registry")

        registration = self._cells.pop(name)

        # Remove connections involving this cell
        self._connections = [
            c for c in self._connections
            if c.source_cell != name and c.target_cell != name
        ]

        logger.info("Unregistered cell '%s'", name)
        return registration

    def get(self, name: str) -> CellRegistration:
        """Get a cell registration by name.

        Raises:
            KeyError: If no cell with this name exists.
        """
        if name not in self._cells:
            raise KeyError(f"No cell named '{name}' in registry")
        return self._cells[name]

    def has(self, name: str) -> bool:
        """Check if a cell is registered."""
        return name in self._cells

    def all_cells(self) -> list[tuple[str, CellRegistration]]:
        """Return all registered cells as (name, registration) pairs."""
        return list(self._cells.items())

    def foundational_cells(self) -> list[tuple[str, CellRegistration]]:
        """Return only foundational cells."""
        return [(n, r) for n, r in self._cells.items() if r.category == "foundational"]

    def knowledge_cells(self) -> list[tuple[str, CellRegistration]]:
        """Return only knowledge cells."""
        return [(n, r) for n, r in self._cells.items() if r.category == "knowledge"]

    # -- Inter-cell connections --

    def add_connection(self, connection: InterCellConnection) -> None:
        """Add a directed connection between two cells.

        Raises:
            KeyError: If either cell is not registered.
        """
        if not self.has(connection.source_cell):
            raise KeyError(f"Source cell '{connection.source_cell}' not in registry")
        if not self.has(connection.target_cell):
            raise KeyError(f"Target cell '{connection.target_cell}' not in registry")

        self._connections.append(connection)

        # Update the registration records
        src_reg = self._cells[connection.source_cell]
        tgt_reg = self._cells[connection.target_cell]
        if connection.target_cell not in src_reg.connections_to:
            src_reg.connections_to.append(connection.target_cell)
        if connection.source_cell not in tgt_reg.connections_from:
            tgt_reg.connections_from.append(connection.source_cell)

        logger.debug(
            "Connected %s.%s -> %s.%s (weight=%.2f)",
            connection.source_cell,
            connection.source_output,
            connection.target_cell,
            connection.target_input,
            connection.weight,
        )

    def get_connections(self, cell_name: Optional[str] = None) -> list[InterCellConnection]:
        """Get connections, optionally filtered by cell name."""
        if cell_name is None:
            return list(self._connections)
        return [
            c for c in self._connections
            if c.source_cell == cell_name or c.target_cell == cell_name
        ]

    def get_inputs_for(self, cell_name: str) -> list[InterCellConnection]:
        """Get all connections feeding into a cell."""
        return [c for c in self._connections if c.target_cell == cell_name]

    def get_outputs_from(self, cell_name: str) -> list[InterCellConnection]:
        """Get all connections flowing from a cell."""
        return [c for c in self._connections if c.source_cell == cell_name]

    # -- Persistence --

    def save(self, directory: Path) -> None:
        """Save all cells and registry metadata to directory."""
        directory.mkdir(parents=True, exist_ok=True)

        # Save each cell
        for name, reg in self._cells.items():
            cell_dir = directory / name
            cell_dir.mkdir(parents=True, exist_ok=True)
            reg.cell.save(cell_dir / "cell.pt")

        # Save registry metadata
        registry_meta = {
            "cells": {
                name: {
                    "category": reg.category,
                    "domain": reg.domain,
                    "registered_at": reg.registered_at,
                    "input_names": reg.input_names,
                    "output_names": reg.output_names,
                    "connections_from": reg.connections_from,
                    "connections_to": reg.connections_to,
                    "metadata": reg.metadata,
                    "cell_class": type(reg.cell).__name__,
                    "cell_module": type(reg.cell).__module__,
                }
                for name, reg in self._cells.items()
            },
            "connections": [
                {
                    "source_cell": c.source_cell,
                    "target_cell": c.target_cell,
                    "source_output": c.source_output,
                    "target_input": c.target_input,
                    "weight": c.weight,
                }
                for c in self._connections
            ],
        }
        torch.save(registry_meta, directory / "registry_meta.pt")
        logger.info(
            "Registry saved: %d cells (%d foundational, %d knowledge), %d connections",
            self.cell_count,
            self.foundational_count,
            self.knowledge_count,
            len(self._connections),
        )

    def get_registry_metadata(self) -> dict:
        """Return serializable metadata about the registry state."""
        return {
            "cell_count": self.cell_count,
            "foundational_count": self.foundational_count,
            "knowledge_count": self.knowledge_count,
            "connection_count": len(self._connections),
            "cells": {
                name: {
                    "category": reg.category,
                    "domain": reg.domain,
                    "registered_at": reg.registered_at,
                }
                for name, reg in self._cells.items()
            },
        }

    def reset_all(self) -> None:
        """Reset hidden state of all cells."""
        for name, reg in self._cells.items():
            reg.cell.reset_hidden()
        logger.info("Reset all %d cells", self.cell_count)
