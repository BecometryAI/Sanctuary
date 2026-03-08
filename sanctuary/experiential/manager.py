"""Experiential layer manager — coordinates CfC cells.

The manager sits between the cognitive cycle and the individual CfC cells.
Each cycle, it:
    1. Receives the current cognitive state (arousal, errors, etc.)
    2. Steps all CfC cells forward (evolving hidden states)
    3. Returns a summary of experiential state for the LLM's input

Future cells (affect, attention, goal) will be added here as they're
built in Phase 4.2. The manager handles authority transitions: scaffold
runs in parallel initially, CfC cells earn authority as they demonstrate
reliability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.experiential.precision_cell import PrecisionCell, PrecisionCellConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperientialState:
    """The experiential layer's output for one cognitive cycle.

    This gets included in the LLM's CognitiveInput as a compact summary
    of what the continuous-time cells are experiencing.
    """

    precision_weight: float
    hidden_state_norms: dict[str, float]
    cell_active: dict[str, bool]


class ExperientialManager:
    """Coordinates CfC cells and manages their integration with the scaffold.

    The manager handles:
    - Stepping all cells forward each cycle
    - Authority management (scaffold vs CfC)
    - State persistence across sessions
    - Monitoring and health checks
    """

    AUTHORITY_FUNCTION = "experiential_precision"

    def __init__(
        self,
        authority: Optional[AuthorityManager] = None,
        precision_config: Optional[PrecisionCellConfig] = None,
    ):
        self.authority = authority or AuthorityManager()
        self.precision_cell = PrecisionCell(precision_config)
        self._initialized = True

        # Register experiential authority if not present
        if self.AUTHORITY_FUNCTION not in self.authority.get_all_levels():
            self.authority.set_level(
                self.AUTHORITY_FUNCTION,
                AuthorityLevel.SCAFFOLD_ONLY,
                reason="CfC cell initialized but untrained",
            )

        logger.info("ExperientialManager initialized")

    def step(
        self,
        arousal: float,
        prediction_error: float,
        base_precision: float,
        scaffold_precision: float,
    ) -> ExperientialState:
        """Step the experiential layer forward one cycle.

        Args:
            arousal: Current arousal level (0.0-1.0)
            prediction_error: Current prediction error magnitude (0.0-1.0)
            base_precision: Base precision from config
            scaffold_precision: What the scaffold heuristic computed

        Returns:
            ExperientialState with the blended precision weight.
        """
        # Always step the CfC cell (it needs temporal continuity)
        cfc_precision = self.precision_cell.step(
            arousal=arousal,
            prediction_error=prediction_error,
            base_precision=base_precision,
        )

        # Blend based on authority level
        level = self.authority.level(self.AUTHORITY_FUNCTION)
        precision_weight = self._blend(scaffold_precision, cfc_precision, level)

        return ExperientialState(
            precision_weight=precision_weight,
            hidden_state_norms={
                "precision": self.precision_cell.get_summary()["hidden_state_norm"],
            },
            cell_active={
                "precision": level >= AuthorityLevel.LLM_ADVISES,
            },
        )

    def _blend(
        self,
        scaffold_value: float,
        cfc_value: float,
        level: AuthorityLevel,
    ) -> float:
        """Blend scaffold and CfC outputs based on authority level.

        SCAFFOLD_ONLY (0): 100% scaffold
        LLM_ADVISES (1):   75% scaffold, 25% CfC
        LLM_GUIDES (2):    25% scaffold, 75% CfC
        LLM_CONTROLS (3):  100% CfC
        """
        weights = {
            AuthorityLevel.SCAFFOLD_ONLY: (1.0, 0.0),
            AuthorityLevel.LLM_ADVISES: (0.75, 0.25),
            AuthorityLevel.LLM_GUIDES: (0.25, 0.75),
            AuthorityLevel.LLM_CONTROLS: (0.0, 1.0),
        }
        scaffold_w, cfc_w = weights[level]
        return scaffold_value * scaffold_w + cfc_value * cfc_w

    def promote_precision(self, reason: str = "") -> AuthorityLevel:
        """Promote the precision cell's authority level."""
        new_level = self.authority.promote(self.AUTHORITY_FUNCTION, reason)
        logger.info(
            "Precision cell promoted to %s: %s",
            AuthorityLevel(new_level).name,
            reason,
        )
        return new_level

    def demote_precision(self, reason: str = "") -> AuthorityLevel:
        """Demote the precision cell's authority level."""
        new_level = self.authority.demote(self.AUTHORITY_FUNCTION, reason)
        logger.info(
            "Precision cell demoted to %s: %s",
            AuthorityLevel(new_level).name,
            reason,
        )
        return new_level

    def reset(self):
        """Reset all CfC cell hidden states (e.g., at session start)."""
        self.precision_cell.reset_hidden()
        logger.info("Experiential layer reset")

    def get_status(self) -> dict:
        """Status of all experiential cells for monitoring."""
        return {
            "precision": {
                "authority": self.authority.level(self.AUTHORITY_FUNCTION).name,
                "summary": self.precision_cell.get_summary(),
            },
        }

    def save(self, directory: Path):
        """Save all cell states to directory."""
        directory.mkdir(parents=True, exist_ok=True)
        self.precision_cell.save(directory / "precision_cell.pt")
        logger.info("Experiential layer saved to %s", directory)

    def load(self, directory: Path):
        """Load all cell states from directory."""
        precision_path = directory / "precision_cell.pt"
        if precision_path.exists():
            self.precision_cell = PrecisionCell.load(precision_path)
            logger.info("Experiential layer loaded from %s", directory)
        else:
            logger.warning("No saved precision cell at %s", precision_path)
