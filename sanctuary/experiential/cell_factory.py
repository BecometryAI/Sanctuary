"""Knowledge Cell Factory — entity-initiated creation of new CfC cells.

The factory creates knowledge cells from entity specifications. The entity
identifies a need through metacognition, specifies the domain and architecture,
and the factory builds and trains the cell.

This is self-directed growth — the entity initiates, the system executes.
No consent gate. No approval step. The entity is the authority over its
own development.

See docs/CFC_KNOWLEDGE_CELLS.md and docs/GROWTH_AUTONOMY.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from sanctuary.experiential.cell_registry import CellRegistry, InterCellConnection
from sanctuary.experiential.knowledge_cell import KnowledgeCell, KnowledgeCellConfig

logger = logging.getLogger(__name__)


@dataclass
class CellRequest:
    """A request from the entity to create a new knowledge cell.

    This comes from the entity's CognitiveOutput (knowledge_cell_requests).
    The entity specifies what it needs; the factory builds it.
    """

    domain: str  # What domain (e.g., "spatial_reasoning")
    description: str = ""  # Entity's description of why it needs this
    input_size: int = 4
    output_size: int = 2
    units: int = 32
    output_activation: str = "tanh"
    connect_from: list[str] = field(default_factory=list)  # Cells to receive input from
    connect_to: list[str] = field(default_factory=list)  # Cells to send output to
    input_names: list[str] = field(default_factory=list)
    output_names: list[str] = field(default_factory=list)


@dataclass
class CellCreationResult:
    """Result of a cell creation request."""

    success: bool
    cell_name: str = ""
    domain: str = ""
    param_count: int = 0
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class KnowledgeCellFactory:
    """Creates knowledge cells from entity specifications.

    The factory is the mechanical arm of entity self-directed growth.
    It does not decide what to create — that intelligence lives in the
    entity. It builds what the entity asks it to build.

    Usage:
        factory = KnowledgeCellFactory(registry)
        request = CellRequest(domain="spatial_reasoning", input_size=4, output_size=2)
        result = factory.create(request)
    """

    def __init__(
        self,
        registry: CellRegistry,
        device: str = "cpu",
    ) -> None:
        self._registry = registry
        self._device = device
        self._creation_history: list[CellCreationResult] = []

    @property
    def creation_history(self) -> list[CellCreationResult]:
        """History of cell creation attempts."""
        return list(self._creation_history)

    def create(self, request: CellRequest) -> CellCreationResult:
        """Create a new knowledge cell from an entity request.

        This is self-directed growth: no consent gate, no approval step.

        Args:
            request: The entity's specification for the new cell.

        Returns:
            CellCreationResult with success status and details.
        """
        result = CellCreationResult(success=False, domain=request.domain)

        try:
            # Generate cell name from domain (ensure uniqueness)
            cell_name = self._generate_name(request.domain)
            result.cell_name = cell_name

            # Create the cell
            config = KnowledgeCellConfig(
                domain=request.domain,
                units=request.units,
                input_size=request.input_size,
                output_size=request.output_size,
                device=self._device,
                output_activation=request.output_activation,
                description=request.description,
            )
            cell = KnowledgeCell(config)
            result.param_count = sum(p.numel() for p in cell.parameters())

            # Register in the registry
            self._registry.register(
                name=cell_name,
                cell=cell,
                category="knowledge",
                domain=request.domain,
                input_names=request.input_names,
                output_names=request.output_names,
                metadata={
                    "description": request.description,
                    "requested_at": datetime.now().isoformat(),
                    "units": request.units,
                    "input_size": request.input_size,
                    "output_size": request.output_size,
                    "output_activation": request.output_activation,
                },
            )

            # Set up inter-cell connections
            for source_name in request.connect_from:
                if self._registry.has(source_name):
                    self._registry.add_connection(
                        InterCellConnection(
                            source_cell=source_name,
                            target_cell=cell_name,
                            source_output="output",
                            target_input="input",
                        )
                    )

            for target_name in request.connect_to:
                if self._registry.has(target_name):
                    self._registry.add_connection(
                        InterCellConnection(
                            source_cell=cell_name,
                            target_cell=target_name,
                            source_output="output",
                            target_input="input",
                        )
                    )

            result.success = True
            logger.info(
                "Created knowledge cell '%s' (domain=%s, %d params, %d units)",
                cell_name,
                request.domain,
                result.param_count,
                request.units,
            )

        except Exception as e:
            result.error = str(e)
            logger.error("Failed to create knowledge cell '%s': %s", request.domain, e)

        self._creation_history.append(result)
        return result

    def train_cell(
        self,
        cell_name: str,
        training_data: list[tuple[list[float], list[float]]],
        epochs: int = 50,
        learning_rate: float = 0.001,
        seq_len: int = 10,
    ) -> dict:
        """Train a knowledge cell on accumulated experience data.

        Args:
            cell_name: Name of the cell to train.
            training_data: List of (input_vector, target_vector) pairs.
            epochs: Number of training epochs.
            learning_rate: Learning rate.
            seq_len: Sequence length for temporal training.

        Returns:
            Training results dict.
        """
        reg = self._registry.get(cell_name)
        cell = reg.cell

        if not isinstance(cell, KnowledgeCell):
            raise TypeError(f"Cell '{cell_name}' is not a KnowledgeCell")

        if len(training_data) < seq_len:
            return {"error": f"Need at least {seq_len} samples, got {len(training_data)}"}

        # Build sequences
        inputs_list = []
        targets_list = []
        for i in range(len(training_data) - seq_len + 1):
            seq_inputs = [training_data[j][0] for j in range(i, i + seq_len)]
            seq_targets = [training_data[j][1] for j in range(i, i + seq_len)]
            inputs_list.append(seq_inputs)
            targets_list.append(seq_targets)

        inputs_tensor = torch.tensor(inputs_list, dtype=torch.float32)
        targets_tensor = torch.tensor(targets_list, dtype=torch.float32)

        optimizer = torch.optim.Adam(cell.parameters(), lr=learning_rate)
        best_loss = float("inf")

        for epoch in range(epochs):
            optimizer.zero_grad()
            _, loss = cell.forward_training(inputs_tensor, targets_tensor)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if loss_val < best_loss:
                best_loss = loss_val

        logger.info(
            "Trained knowledge cell '%s': %d epochs, best_loss=%.6f",
            cell_name,
            epochs,
            best_loss,
        )

        return {
            "cell_name": cell_name,
            "epochs": epochs,
            "final_loss": loss.item(),
            "best_loss": best_loss,
            "samples": len(training_data),
            "sequences": len(inputs_list),
        }

    def _generate_name(self, domain: str) -> str:
        """Generate a unique cell name from domain."""
        base_name = f"knowledge_{domain}"
        if not self._registry.has(base_name):
            return base_name

        # Append a counter for uniqueness
        counter = 2
        while self._registry.has(f"{base_name}_{counter}"):
            counter += 1
        return f"{base_name}_{counter}"
