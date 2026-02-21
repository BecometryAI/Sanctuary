"""Scaffold affect computation — the computed track of dual-track emotion.

Maintains a VAD (Valence-Arousal-Dominance) state that decays toward baseline
and responds to percepts and LLM emotional reports. This is the scaffold's
"objective" emotional reading. The LLM's felt_quality is the experiential track.

Divergence between computed VAD and felt quality is informative, not a bug.

When CfC cells are ready (Phase 8), this module delegates to the affect CfC cell
instead of running heuristics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import (
    ComputedVAD,
    EmotionalOutput,
    Percept,
)

logger = logging.getLogger(__name__)

# Keywords used for simple percept-based affect updates
_POSITIVE_KW = frozenset(
    ["happy", "joy", "love", "great", "excellent", "wonderful", "thank", "good"]
)
_NEGATIVE_KW = frozenset(
    ["sad", "angry", "fear", "bad", "terrible", "hate", "error", "fail", "wrong"]
)
_AROUSING_KW = frozenset(
    ["urgent", "exciting", "crisis", "emergency", "surprise", "unexpected"]
)


@dataclass
class AffectConfig:
    """Configuration for the scaffold affect module."""

    baseline_valence: float = 0.1
    baseline_arousal: float = 0.2
    baseline_dominance: float = 0.5
    decay_rate: float = 0.05  # Per cycle, toward baseline
    sensitivity: float = 0.15  # How strongly percepts move VAD
    llm_blend_weight: float = 0.3  # How much LLM shifts blend when LLM_GUIDES


class ScaffoldAffect:
    """Simplified affect computation for the cognitive scaffold.

    Maintains computed VAD, responds to percepts via keyword heuristics,
    and merges LLM emotional output based on authority level.
    """

    def __init__(self, config: Optional[AffectConfig] = None):
        self.config = config or AffectConfig()
        self.valence = self.config.baseline_valence
        self.arousal = self.config.baseline_arousal
        self.dominance = self.config.baseline_dominance

    # -- Public API --

    def get_computed_vad(self) -> ComputedVAD:
        """Return the current computed VAD as a schema-compatible object."""
        return ComputedVAD(
            valence=round(self.valence, 3),
            arousal=round(self.arousal, 3),
            dominance=round(self.dominance, 3),
        )

    def update_from_percepts(self, percepts: list[Percept]) -> None:
        """Shift VAD based on percept content (keyword heuristics)."""
        dv, da, dd = 0.0, 0.0, 0.0

        for p in percepts:
            text = p.content.lower()
            words = set(text.split())

            if words & _POSITIVE_KW:
                dv += 0.2
                da += 0.1
            if words & _NEGATIVE_KW:
                dv -= 0.2
                da += 0.15
                dd -= 0.1
            if words & _AROUSING_KW:
                da += 0.3

        s = self.config.sensitivity
        self.valence = _clamp(self.valence + dv * s, -1.0, 1.0)
        self.arousal = _clamp(self.arousal + da * s, 0.0, 1.0)
        self.dominance = _clamp(self.dominance + dd * s, 0.0, 1.0)

    def merge_llm_emotion(
        self,
        emotion: EmotionalOutput,
        authority: AuthorityManager,
    ) -> None:
        """Blend LLM's emotional self-report into computed VAD.

        The blend weight depends on the authority level for ``emotional_state``:
        - SCAFFOLD_ONLY (0): ignore LLM shifts entirely
        - LLM_ADVISES (1): small blend (~10%)
        - LLM_GUIDES (2): moderate blend (configured llm_blend_weight)
        - LLM_CONTROLS (3): LLM shifts applied fully
        """
        level = authority.level("emotional_state")

        if level == AuthorityLevel.SCAFFOLD_ONLY:
            return

        # Compute effective blend factor
        if level == AuthorityLevel.LLM_ADVISES:
            w = self.config.llm_blend_weight * 0.3
        elif level == AuthorityLevel.LLM_GUIDES:
            w = self.config.llm_blend_weight
        else:  # LLM_CONTROLS
            w = 1.0

        self.valence = _clamp(self.valence + emotion.valence_shift * w, -1.0, 1.0)
        self.arousal = _clamp(self.arousal + emotion.arousal_shift * w, 0.0, 1.0)

    def decay_toward_baseline(self) -> None:
        """Gradually return toward baseline (emotional regulation)."""
        r = self.config.decay_rate
        self.valence = self.valence * (1 - r) + self.config.baseline_valence * r
        self.arousal = self.arousal * (1 - r) + self.config.baseline_arousal * r
        self.dominance = self.dominance * (1 - r) + self.config.baseline_dominance * r

    def get_emotion_label(self) -> str:
        """Simple VAD → emotion label mapping."""
        v, a = self.valence, self.arousal
        if v > 0.3 and a > 0.5:
            return "joy"
        if v > 0.3 and a <= 0.5:
            return "contentment"
        if v < -0.3 and a > 0.5:
            return "anger"
        if v < -0.3 and a <= 0.5:
            return "sadness"
        if abs(v) <= 0.3 and a > 0.7:
            return "surprise"
        return "calm"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
