"""Experiential layer manager — coordinates CfC cells.

The manager sits between the cognitive cycle and the individual CfC cells.
Each cycle, it:
    1. Receives the current cognitive state (arousal, errors, etc.)
    2. Steps all CfC cells forward (evolving hidden states)
    3. Returns a summary of experiential state for the LLM's input

Inter-cell connections: affect feeds precision (emotional arousal modulates
precision), attention informs goal prioritization (salient goals get boosted).
These connections create an internal neural ecosystem that evolves between
LLM cycles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.experiential.affect_cell import AffectCell, AffectCellConfig
from sanctuary.experiential.attention_cell import AttentionCell, AttentionCellConfig
from sanctuary.experiential.evolution import (
    ContinuousEvolutionLoop,
    EvolutionConfig,
    EvolutionSnapshot,
    PerceptEvent,
)
from sanctuary.experiential.goal_cell import GoalCell, GoalCellConfig
from sanctuary.experiential.precision_cell import PrecisionCell, PrecisionCellConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperientialState:
    """The experiential layer's output for one cognitive cycle.

    This gets included in the LLM's CognitiveInput as a compact summary
    of what the continuous-time cells are experiencing.
    """

    precision_weight: float
    affect_vad: tuple[float, float, float] = (0.0, 0.2, 0.5)
    attention_salience: float = 0.5
    goal_adjustment: float = 0.0
    hidden_state_norms: dict[str, float] = field(default_factory=dict)
    cell_active: dict[str, bool] = field(default_factory=dict)


# Authority function names for each cell
_AUTH = {
    "precision": "experiential_precision",
    "affect": "experiential_affect",
    "attention": "experiential_attention",
    "goal": "experiential_goal",
}


class ExperientialManager:
    """Coordinates CfC cells and manages their integration with the scaffold.

    The manager handles:
    - Stepping all cells forward each cycle
    - Inter-cell connections (affect->precision, attention->goals)
    - Authority management (scaffold vs CfC per cell)
    - State persistence across sessions
    - Monitoring and health checks
    """

    AUTHORITY_FUNCTION = "experiential_precision"  # backward compat

    def __init__(
        self,
        authority: Optional[AuthorityManager] = None,
        precision_config: Optional[PrecisionCellConfig] = None,
        affect_config: Optional[AffectCellConfig] = None,
        attention_config: Optional[AttentionCellConfig] = None,
        goal_config: Optional[GoalCellConfig] = None,
        evolution_config: Optional[EvolutionConfig] = None,
    ):
        self.authority = authority or AuthorityManager()

        # Create all cells
        self.precision_cell = PrecisionCell(precision_config)
        self.affect_cell = AffectCell(affect_config)
        self.attention_cell = AttentionCell(attention_config)
        self.goal_cell = GoalCell(goal_config)

        # Continuous evolution loop (optional, started explicitly)
        self._evolution_config = evolution_config
        self._evolution_loop: Optional[ContinuousEvolutionLoop] = None

        self._initialized = True

        # Register authority for each cell if not present
        for name, func in _AUTH.items():
            if func not in self.authority.get_all_levels():
                self.authority.set_level(
                    func,
                    AuthorityLevel.SCAFFOLD_ONLY,
                    reason=f"CfC {name} cell initialized but untrained",
                )

        logger.info("ExperientialManager initialized with 4 CfC cells")

    def step(
        self,
        arousal: float,
        prediction_error: float,
        base_precision: float,
        scaffold_precision: float,
        # Affect inputs
        percept_valence_delta: float = 0.0,
        percept_arousal_delta: float = 0.0,
        llm_emotion_shift: float = 0.0,
        scaffold_vad: tuple[float, float, float] = (0.0, 0.2, 0.5),
        # Attention inputs
        goal_relevance: float = 0.0,
        novelty: float = 0.0,
        emotional_salience: float = 0.0,
        recency: float = 0.0,
        scaffold_salience: float = 0.5,
        # Goal inputs
        cycles_stalled_norm: float = 0.0,
        deadline_urgency: float = 0.0,
        emotional_congruence: float = 0.0,
        scaffold_goal_adj: float = 0.0,
    ) -> ExperientialState:
        """Step the experiential layer forward one cycle.

        All cells are always stepped (they need temporal continuity).
        Blending with scaffold depends on each cell's authority level.

        Inter-cell connections:
        - Affect arousal feeds into precision (emotional modulation)
        - Attention salience feeds into goal adjustment (salient goals boosted)
        """
        # 1. Step affect cell first (feeds into precision)
        cfc_v, cfc_a, cfc_d = self.affect_cell.step(
            percept_valence_delta=percept_valence_delta,
            percept_arousal_delta=percept_arousal_delta,
            llm_emotion_shift=llm_emotion_shift,
        )
        affect_level = self.authority.level(_AUTH["affect"])
        blended_vad = (
            self._blend(scaffold_vad[0], cfc_v, affect_level),
            self._blend(scaffold_vad[1], cfc_a, affect_level),
            self._blend(scaffold_vad[2], cfc_d, affect_level),
        )

        # 2. Step precision cell (uses affect arousal as inter-cell connection)
        # Override arousal with blended affect arousal for cross-cell influence
        effective_arousal = blended_vad[1]
        cfc_precision = self.precision_cell.step(
            arousal=effective_arousal,
            prediction_error=prediction_error,
            base_precision=base_precision,
        )
        precision_level = self.authority.level(_AUTH["precision"])
        blended_precision = self._blend(scaffold_precision, cfc_precision, precision_level)

        # 3. Step attention cell
        cfc_salience = self.attention_cell.step(
            goal_relevance=goal_relevance,
            novelty=novelty,
            emotional_salience=emotional_salience,
            recency=recency,
        )
        attention_level = self.authority.level(_AUTH["attention"])
        blended_salience = self._blend(scaffold_salience, cfc_salience, attention_level)

        # 4. Step goal cell (uses attention salience as inter-cell connection)
        # Boost emotional congruence with attention salience for cross-cell influence
        effective_congruence = emotional_congruence + 0.1 * blended_salience
        cfc_goal_adj = self.goal_cell.step(
            cycles_stalled_norm=cycles_stalled_norm,
            deadline_urgency=deadline_urgency,
            emotional_congruence=effective_congruence,
        )
        goal_level = self.authority.level(_AUTH["goal"])
        blended_goal_adj = self._blend(scaffold_goal_adj, cfc_goal_adj, goal_level)

        return ExperientialState(
            precision_weight=blended_precision,
            affect_vad=blended_vad,
            attention_salience=blended_salience,
            goal_adjustment=blended_goal_adj,
            hidden_state_norms={
                "precision": self.precision_cell.get_summary()["hidden_state_norm"],
                "affect": self.affect_cell.get_summary()["hidden_state_norm"],
                "attention": self.attention_cell.get_summary()["hidden_state_norm"],
                "goal": self.goal_cell.get_summary()["hidden_state_norm"],
            },
            cell_active={
                "precision": precision_level >= AuthorityLevel.LLM_ADVISES,
                "affect": affect_level >= AuthorityLevel.LLM_ADVISES,
                "attention": attention_level >= AuthorityLevel.LLM_ADVISES,
                "goal": goal_level >= AuthorityLevel.LLM_ADVISES,
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

    # -- Authority management per cell --

    def promote(self, cell_name: str, reason: str = "") -> AuthorityLevel:
        """Promote a cell's authority level."""
        func = _AUTH[cell_name]
        new_level = self.authority.promote(func, reason)
        logger.info("CfC %s promoted to %s: %s", cell_name, AuthorityLevel(new_level).name, reason)
        return new_level

    def demote(self, cell_name: str, reason: str = "") -> AuthorityLevel:
        """Demote a cell's authority level."""
        func = _AUTH[cell_name]
        new_level = self.authority.demote(func, reason)
        logger.info("CfC %s demoted to %s: %s", cell_name, AuthorityLevel(new_level).name, reason)
        return new_level

    # Backward-compatible aliases
    def promote_precision(self, reason: str = "") -> AuthorityLevel:
        return self.promote("precision", reason)

    def demote_precision(self, reason: str = "") -> AuthorityLevel:
        return self.demote("precision", reason)

    # -- Continuous evolution --

    async def start_evolution(self, config: Optional[EvolutionConfig] = None):
        """Start the continuous evolution background loop.

        CfC cells will evolve between LLM cycles at adaptive tick rates.
        """
        if self._evolution_loop is not None and self._evolution_loop.running:
            return
        cfg = config or self._evolution_config
        self._evolution_loop = ContinuousEvolutionLoop(self, cfg)
        await self._evolution_loop.start()

    async def stop_evolution(self):
        """Stop the continuous evolution loop."""
        if self._evolution_loop is not None:
            await self._evolution_loop.stop()

    def feed_percept(self, event: PerceptEvent):
        """Feed a percept event to the evolution loop for inter-cycle processing."""
        if self._evolution_loop is not None and self._evolution_loop.running:
            self._evolution_loop.feed_percept(event)

    def update_evolution_context(self, **kwargs):
        """Update scaffold context for the evolution loop after each LLM cycle."""
        if self._evolution_loop is not None:
            self._evolution_loop.update_context(**kwargs)

    def evolution_snapshot(self) -> Optional[EvolutionSnapshot]:
        """Read accumulated CfC state from the evolution loop.

        Returns None if the evolution loop is not running.
        """
        if self._evolution_loop is not None and self._evolution_loop.running:
            return self._evolution_loop.snapshot()
        return None

    @property
    def evolution_running(self) -> bool:
        return self._evolution_loop is not None and self._evolution_loop.running

    def reset(self):
        """Reset all CfC cell hidden states."""
        self.precision_cell.reset_hidden()
        self.affect_cell.reset_hidden()
        self.attention_cell.reset_hidden()
        self.goal_cell.reset_hidden()
        logger.info("Experiential layer reset")

    def get_status(self) -> dict:
        """Status of all experiential cells for monitoring."""
        status = {
            name: {
                "authority": self.authority.level(func).name,
                "summary": getattr(self, f"{name}_cell").get_summary(),
            }
            for name, func in _AUTH.items()
        }
        if self._evolution_loop is not None:
            status["evolution"] = {
                "running": self._evolution_loop.running,
                "tick_ms": self._evolution_loop.current_tick_ms,
            }
        return status

    def save(self, directory: Path):
        """Save all cell states to directory."""
        directory.mkdir(parents=True, exist_ok=True)
        self.precision_cell.save(directory / "precision_cell.pt")
        self.affect_cell.save(directory / "affect_cell.pt")
        self.attention_cell.save(directory / "attention_cell.pt")
        self.goal_cell.save(directory / "goal_cell.pt")
        logger.info("Experiential layer saved to %s", directory)

    def load(self, directory: Path):
        """Load all cell states from directory."""
        cells = {
            "precision": (PrecisionCell, "precision_cell"),
            "affect": (AffectCell, "affect_cell"),
            "attention": (AttentionCell, "attention_cell"),
            "goal": (GoalCell, "goal_cell"),
        }
        for name, (cls, attr) in cells.items():
            path = directory / f"{name}_cell.pt"
            if path.exists():
                setattr(self, attr, cls.load(path))
                logger.info("Loaded %s cell from %s", name, path)
            else:
                logger.warning("No saved %s cell at %s", name, path)
