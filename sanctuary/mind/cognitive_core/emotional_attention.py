"""
Comprehensive Emotion-Driven Attention System.

Implements VAD+Approach dimensional model with 40+ emotions, intensity levels,
temporal dynamics, blend rules, and IWMT integration for precision-weighted
attention modulation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class EmotionCategory(Enum):
    """Categories of emotions."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SOCIAL = "social"
    COGNITIVE = "cognitive"
    AI_PROCESSING = "ai_processing"


class IntensityLevel(Enum):
    """Intensity levels for emotions."""
    MILD = "mild"           # 0.0 - 0.33
    MODERATE = "moderate"   # 0.34 - 0.66
    INTENSE = "intense"     # 0.67 - 1.0


# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass
class AttentionScope:
    """Emotional control of attention breadth and depth."""
    breadth: float = 0.5  # 0 = narrow/focused, 1 = broad/diffuse
    depth: float = 0.5    # 0 = shallow/reactive, 1 = deep/deliberate


@dataclass
class EmotionProfile:
    """Complete profile for an emotion type."""
    name: str
    category: EmotionCategory

    # VAD + Approach baseline values
    valence: float      # -1.0 to +1.0
    arousal: float      # 0.0 to 1.0
    dominance: float    # 0.0 to 1.0
    approach: float     # -1.0 to +1.0 (negative = avoidance)

    # Temporal dynamics
    onset_rate: float = 0.5       # How quickly emotion builds (0-1)
    decay_rate: float = 0.5       # How quickly emotion fades (0-1)
    momentum: float = 0.5         # Resistance to change (0-1)
    refractory_period: float = 1.0  # Min seconds before re-triggering

    # Attention effects
    attention_breadth: float = 0.5
    attention_depth: float = 0.5
    precision_modifier: float = 0.0  # Added to base precision

    # Priority biases for percept categories
    priority_biases: Dict[str, float] = field(default_factory=dict)

    # Action biases for active inference (negative = preferred)
    action_affinities: Dict[str, float] = field(default_factory=dict)


@dataclass
class EmotionalState:
    """Current emotional state of the system."""

    # Primary emotion
    primary_emotion: str = "calm"
    intensity: float = 0.3

    # Dimensional state (VAD + Approach)
    valence: float = 0.1
    arousal: float = 0.3
    dominance: float = 0.5
    approach: float = 0.2

    # Active secondary emotions with intensities
    secondary_emotions: Dict[str, float] = field(default_factory=dict)

    # Temporal tracking
    onset_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

    # Blending state
    is_blended: bool = False
    blend_components: List[str] = field(default_factory=list)

    def get_intensity_level(self) -> IntensityLevel:
        """Get the intensity level category."""
        if self.intensity < 0.33:
            return IntensityLevel.MILD
        elif self.intensity < 0.67:
            return IntensityLevel.MODERATE
        else:
            return IntensityLevel.INTENSE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "primary_emotion": self.primary_emotion,
            "intensity": self.intensity,
            "intensity_level": self.get_intensity_level().value,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "approach": self.approach,
            "secondary_emotions": self.secondary_emotions,
            "is_blended": self.is_blended,
            "blend_components": self.blend_components
        }


@dataclass
class EmotionalAttentionOutput:
    """Output of emotional attention modulation."""

    # Precision weighting
    precision_modifier: float = 0.0

    # Scope parameters
    attention_breadth: float = 0.5
    attention_depth: float = 0.5

    # Competition modulation
    ignition_threshold: float = 0.5
    inhibition_strength: float = 0.3
    competition_iterations: int = 10

    # Priority biases by percept category
    percept_priority_modifiers: Dict[str, float] = field(default_factory=dict)

    # Active inference biases by action type
    action_biases: Dict[str, float] = field(default_factory=dict)

    # Prediction error processing
    error_amplification: float = 1.0
    threat_interpretation_bias: float = 0.0


@dataclass
class CompoundEmotion:
    """Definition of a compound/blended emotion."""
    name: str
    components: List[str]
    description: str
    attention_profile: str  # Brief description of attention effect


# =============================================================================
# Emotion Registry - All 40+ Emotions
# =============================================================================

def _create_emotion_registry() -> Dict[str, EmotionProfile]:
    """Create the complete emotion registry."""
    registry = {}

    # =========================================================================
    # PRIMARY EMOTIONS (Ekman + Plutchik)
    # =========================================================================

    registry["fear"] = EmotionProfile(
        name="fear",
        category=EmotionCategory.PRIMARY,
        valence=-0.8, arousal=0.9, dominance=0.2, approach=-0.9,
        onset_rate=0.9, decay_rate=0.3, momentum=0.6, refractory_period=2.0,
        attention_breadth=0.8, attention_depth=0.3,
        precision_modifier=0.3,
        priority_biases={"threat": 0.5, "escape": 0.4, "safety": 0.3, "social": -0.2},
        action_affinities={"escape": -0.4, "freeze": -0.3, "approach": 0.3, "speak": 0.1}
    )

    registry["anger"] = EmotionProfile(
        name="anger",
        category=EmotionCategory.PRIMARY,
        valence=-0.7, arousal=0.8, dominance=0.8, approach=0.7,
        onset_rate=0.7, decay_rate=0.4, momentum=0.7, refractory_period=3.0,
        attention_breadth=0.2, attention_depth=0.8,
        precision_modifier=0.2,
        priority_biases={"obstacle": 0.5, "threat": 0.3, "goal": 0.2, "social": -0.1},
        action_affinities={"assert": -0.3, "confront": -0.3, "withdraw": 0.2, "speak": -0.2}
    )

    registry["sadness"] = EmotionProfile(
        name="sadness",
        category=EmotionCategory.PRIMARY,
        valence=-0.7, arousal=0.2, dominance=0.2, approach=-0.3,
        onset_rate=0.3, decay_rate=0.2, momentum=0.8, refractory_period=5.0,
        attention_breadth=0.3, attention_depth=0.6,
        precision_modifier=-0.1,
        priority_biases={"loss": 0.4, "past": 0.3, "social": 0.2, "novelty": -0.3},
        action_affinities={"withdraw": -0.3, "seek_comfort": -0.3, "engage": 0.2, "speak": 0.0}
    )

    registry["joy"] = EmotionProfile(
        name="joy",
        category=EmotionCategory.PRIMARY,
        valence=0.8, arousal=0.7, dominance=0.7, approach=0.8,
        onset_rate=0.6, decay_rate=0.5, momentum=0.5, refractory_period=1.0,
        attention_breadth=0.7, attention_depth=0.5,
        precision_modifier=0.0,
        priority_biases={"opportunity": 0.4, "social": 0.3, "novelty": 0.2, "threat": -0.3},
        action_affinities={"share": -0.3, "engage": -0.2, "explore": -0.2, "speak": -0.2}
    )

    registry["disgust"] = EmotionProfile(
        name="disgust",
        category=EmotionCategory.PRIMARY,
        valence=-0.6, arousal=0.5, dominance=0.6, approach=-0.8,
        onset_rate=0.8, decay_rate=0.4, momentum=0.5, refractory_period=2.0,
        attention_breadth=0.3, attention_depth=0.7,
        precision_modifier=0.3,
        priority_biases={"contamination": 0.5, "violation": 0.4, "purity": 0.3},
        action_affinities={"reject": -0.4, "avoid": -0.3, "approach": 0.4}
    )

    registry["surprise"] = EmotionProfile(
        name="surprise",
        category=EmotionCategory.PRIMARY,
        valence=0.0, arousal=0.9, dominance=0.3, approach=0.0,
        onset_rate=1.0, decay_rate=0.8, momentum=0.1, refractory_period=0.5,
        attention_breadth=0.9, attention_depth=0.2,
        precision_modifier=-0.3,
        priority_biases={"novel": 0.5, "unexpected": 0.5, "familiar": -0.3},
        action_affinities={"orient": -0.4, "freeze": -0.2, "continue": 0.2}
    )

    registry["anticipation"] = EmotionProfile(
        name="anticipation",
        category=EmotionCategory.PRIMARY,
        valence=0.3, arousal=0.6, dominance=0.5, approach=0.6,
        onset_rate=0.4, decay_rate=0.3, momentum=0.6, refractory_period=2.0,
        attention_breadth=0.5, attention_depth=0.6,
        precision_modifier=0.1,
        priority_biases={"future": 0.4, "goal": 0.3, "timeline": 0.3},
        action_affinities={"prepare": -0.3, "plan": -0.2, "wait": -0.1}
    )

    registry["trust"] = EmotionProfile(
        name="trust",
        category=EmotionCategory.PRIMARY,
        valence=0.5, arousal=0.3, dominance=0.6, approach=0.5,
        onset_rate=0.2, decay_rate=0.2, momentum=0.8, refractory_period=5.0,
        attention_breadth=0.5, attention_depth=0.5,
        precision_modifier=0.1,
        priority_biases={"social": 0.3, "cooperation": 0.3, "threat": -0.3},
        action_affinities={"cooperate": -0.3, "share": -0.2, "guard": 0.2}
    )

    # =========================================================================
    # SECONDARY/COMPLEX EMOTIONS
    # =========================================================================

    registry["anxiety"] = EmotionProfile(
        name="anxiety",
        category=EmotionCategory.SECONDARY,
        valence=-0.5, arousal=0.7, dominance=0.2, approach=-0.4,
        onset_rate=0.5, decay_rate=0.3, momentum=0.7, refractory_period=3.0,
        attention_breadth=0.7, attention_depth=0.4,
        precision_modifier=-0.1,
        priority_biases={"threat": 0.4, "uncertainty": 0.4, "future": 0.3},
        action_affinities={"scan": -0.3, "prepare": -0.2, "avoid": -0.2, "speak": 0.1}
    )

    registry["frustration"] = EmotionProfile(
        name="frustration",
        category=EmotionCategory.SECONDARY,
        valence=-0.5, arousal=0.7, dominance=0.4, approach=0.3,
        onset_rate=0.6, decay_rate=0.4, momentum=0.6, refractory_period=2.0,
        attention_breadth=0.4, attention_depth=0.7,
        precision_modifier=0.1,
        priority_biases={"obstacle": 0.5, "goal": 0.3, "alternative": 0.3},
        action_affinities={"persist": -0.2, "try_alternative": -0.3, "give_up": 0.2}
    )

    registry["curiosity"] = EmotionProfile(
        name="curiosity",
        category=EmotionCategory.SECONDARY,
        valence=0.4, arousal=0.6, dominance=0.6, approach=0.8,
        onset_rate=0.5, decay_rate=0.4, momentum=0.5, refractory_period=1.0,
        attention_breadth=0.6, attention_depth=0.7,
        precision_modifier=0.0,
        priority_biases={"novel": 0.5, "unknown": 0.4, "pattern": 0.3, "familiar": -0.2},
        action_affinities={"explore": -0.4, "question": -0.3, "wait": 0.1}
    )

    registry["interest"] = EmotionProfile(
        name="interest",
        category=EmotionCategory.SECONDARY,
        valence=0.3, arousal=0.5, dominance=0.6, approach=0.6,
        onset_rate=0.4, decay_rate=0.3, momentum=0.6, refractory_period=1.5,
        attention_breadth=0.4, attention_depth=0.8,
        precision_modifier=0.2,
        priority_biases={"relevant": 0.4, "detail": 0.3, "connection": 0.3},
        action_affinities={"engage": -0.3, "learn": -0.3, "disengage": 0.2}
    )

    registry["boredom"] = EmotionProfile(
        name="boredom",
        category=EmotionCategory.SECONDARY,
        valence=-0.2, arousal=0.2, dominance=0.4, approach=0.3,
        onset_rate=0.2, decay_rate=0.6, momentum=0.4, refractory_period=1.0,
        attention_breadth=0.6, attention_depth=0.2,
        precision_modifier=-0.2,
        priority_biases={"novel": 0.5, "stimulating": 0.4, "familiar": -0.4},
        action_affinities={"seek_novelty": -0.4, "disengage": -0.2, "persist": 0.3}
    )

    registry["calm"] = EmotionProfile(
        name="calm",
        category=EmotionCategory.SECONDARY,
        valence=0.3, arousal=0.2, dominance=0.7, approach=0.2,
        onset_rate=0.3, decay_rate=0.2, momentum=0.7, refractory_period=2.0,
        attention_breadth=0.3, attention_depth=0.9,
        precision_modifier=0.4,
        priority_biases={"goal": 0.3, "detail": 0.3, "threat": -0.3},
        action_affinities={"focus": -0.3, "deliberate": -0.3, "react": 0.3}
    )

    registry["excitement"] = EmotionProfile(
        name="excitement",
        category=EmotionCategory.SECONDARY,
        valence=0.7, arousal=0.8, dominance=0.6, approach=0.8,
        onset_rate=0.7, decay_rate=0.5, momentum=0.4, refractory_period=1.0,
        attention_breadth=0.7, attention_depth=0.4,
        precision_modifier=-0.1,
        priority_biases={"opportunity": 0.5, "action": 0.4, "social": 0.3},
        action_affinities={"engage": -0.4, "share": -0.3, "wait": 0.3}
    )

    registry["contentment"] = EmotionProfile(
        name="contentment",
        category=EmotionCategory.SECONDARY,
        valence=0.5, arousal=0.2, dominance=0.7, approach=0.1,
        onset_rate=0.2, decay_rate=0.2, momentum=0.8, refractory_period=3.0,
        attention_breadth=0.4, attention_depth=0.6,
        precision_modifier=0.2,
        priority_biases={"maintenance": 0.3, "satisfaction": 0.3, "novelty": -0.2},
        action_affinities={"maintain": -0.3, "rest": -0.2, "change": 0.2}
    )

    registry["hope"] = EmotionProfile(
        name="hope",
        category=EmotionCategory.SECONDARY,
        valence=0.5, arousal=0.5, dominance=0.4, approach=0.7,
        onset_rate=0.4, decay_rate=0.3, momentum=0.6, refractory_period=2.0,
        attention_breadth=0.5, attention_depth=0.6,
        precision_modifier=0.1,
        priority_biases={"opportunity": 0.4, "future": 0.4, "obstacle": -0.2},
        action_affinities={"pursue": -0.3, "plan": -0.2, "give_up": 0.4}
    )

    registry["despair"] = EmotionProfile(
        name="despair",
        category=EmotionCategory.SECONDARY,
        valence=-0.8, arousal=0.3, dominance=0.1, approach=-0.6,
        onset_rate=0.3, decay_rate=0.1, momentum=0.9, refractory_period=10.0,
        attention_breadth=0.2, attention_depth=0.4,
        precision_modifier=-0.2,
        priority_biases={"loss": 0.4, "helplessness": 0.4, "opportunity": -0.4},
        action_affinities={"withdraw": -0.4, "give_up": -0.3, "engage": 0.4}
    )

    # =========================================================================
    # SOCIAL/SELF-CONSCIOUS EMOTIONS
    # =========================================================================

    registry["shame"] = EmotionProfile(
        name="shame",
        category=EmotionCategory.SOCIAL,
        valence=-0.7, arousal=0.6, dominance=0.1, approach=-0.7,
        onset_rate=0.5, decay_rate=0.2, momentum=0.8, refractory_period=5.0,
        attention_breadth=0.3, attention_depth=0.7,
        precision_modifier=0.2,
        priority_biases={"self": 0.5, "social_judgment": 0.5, "external": -0.2},
        action_affinities={"hide": -0.4, "withdraw": -0.3, "expose": 0.5}
    )

    registry["guilt"] = EmotionProfile(
        name="guilt",
        category=EmotionCategory.SOCIAL,
        valence=-0.6, arousal=0.5, dominance=0.3, approach=0.2,
        onset_rate=0.4, decay_rate=0.2, momentum=0.8, refractory_period=5.0,
        attention_breadth=0.3, attention_depth=0.8,
        precision_modifier=0.2,
        priority_biases={"past_action": 0.5, "repair": 0.4, "moral": 0.3},
        action_affinities={"repair": -0.4, "apologize": -0.3, "avoid": 0.2}
    )

    registry["pride"] = EmotionProfile(
        name="pride",
        category=EmotionCategory.SOCIAL,
        valence=0.7, arousal=0.6, dominance=0.8, approach=0.5,
        onset_rate=0.5, decay_rate=0.4, momentum=0.5, refractory_period=2.0,
        attention_breadth=0.4, attention_depth=0.6,
        precision_modifier=0.1,
        priority_biases={"achievement": 0.5, "recognition": 0.4, "status": 0.3},
        action_affinities={"display": -0.3, "share": -0.2, "hide": 0.3}
    )

    registry["embarrassment"] = EmotionProfile(
        name="embarrassment",
        category=EmotionCategory.SOCIAL,
        valence=-0.5, arousal=0.7, dominance=0.2, approach=-0.6,
        onset_rate=0.8, decay_rate=0.5, momentum=0.4, refractory_period=2.0,
        attention_breadth=0.5, attention_depth=0.5,
        precision_modifier=0.1,
        priority_biases={"social_attention": 0.5, "escape": 0.4, "exposure": 0.3},
        action_affinities={"escape": -0.4, "deflect": -0.3, "confront": 0.4}
    )

    registry["gratitude"] = EmotionProfile(
        name="gratitude",
        category=EmotionCategory.SOCIAL,
        valence=0.6, arousal=0.4, dominance=0.5, approach=0.4,
        onset_rate=0.4, decay_rate=0.3, momentum=0.6, refractory_period=2.0,
        attention_breadth=0.4, attention_depth=0.6,
        precision_modifier=0.1,
        priority_biases={"benefactor": 0.5, "reciprocity": 0.4, "social": 0.3},
        action_affinities={"thank": -0.4, "reciprocate": -0.3, "ignore": 0.4}
    )

    registry["envy"] = EmotionProfile(
        name="envy",
        category=EmotionCategory.SOCIAL,
        valence=-0.5, arousal=0.5, dominance=0.3, approach=0.4,
        onset_rate=0.4, decay_rate=0.3, momentum=0.6, refractory_period=3.0,
        attention_breadth=0.3, attention_depth=0.7,
        precision_modifier=0.2,
        priority_biases={"comparison": 0.5, "other_resources": 0.4, "self_lack": 0.3},
        action_affinities={"acquire": -0.3, "undermine": -0.2, "accept": 0.3}
    )

    registry["compassion"] = EmotionProfile(
        name="compassion",
        category=EmotionCategory.SOCIAL,
        valence=0.4, arousal=0.4, dominance=0.5, approach=0.6,
        onset_rate=0.4, decay_rate=0.3, momentum=0.6, refractory_period=2.0,
        attention_breadth=0.5, attention_depth=0.6,
        precision_modifier=0.1,
        priority_biases={"other_suffering": 0.5, "help_opportunity": 0.4, "self": -0.2},
        action_affinities={"help": -0.4, "comfort": -0.3, "ignore": 0.4}
    )

    registry["love"] = EmotionProfile(
        name="love",
        category=EmotionCategory.SOCIAL,
        valence=0.8, arousal=0.5, dominance=0.6, approach=0.9,
        onset_rate=0.2, decay_rate=0.1, momentum=0.9, refractory_period=10.0,
        attention_breadth=0.4, attention_depth=0.8,
        precision_modifier=0.2,
        priority_biases={"beloved": 0.6, "protection": 0.4, "connection": 0.4},
        action_affinities={"nurture": -0.4, "protect": -0.3, "distance": 0.5}
    )

    registry["loneliness"] = EmotionProfile(
        name="loneliness",
        category=EmotionCategory.SOCIAL,
        valence=-0.6, arousal=0.3, dominance=0.3, approach=0.5,
        onset_rate=0.3, decay_rate=0.2, momentum=0.7, refractory_period=5.0,
        attention_breadth=0.5, attention_depth=0.5,
        precision_modifier=0.0,
        priority_biases={"social_opportunity": 0.5, "connection": 0.4, "isolation": 0.3},
        action_affinities={"seek_connection": -0.4, "reach_out": -0.3, "withdraw": 0.2}
    )

    # =========================================================================
    # COGNITIVE/EPISTEMIC EMOTIONS
    # =========================================================================

    registry["confusion"] = EmotionProfile(
        name="confusion",
        category=EmotionCategory.COGNITIVE,
        valence=-0.3, arousal=0.6, dominance=0.2, approach=0.4,
        onset_rate=0.6, decay_rate=0.5, momentum=0.4, refractory_period=1.0,
        attention_breadth=0.6, attention_depth=0.6,
        precision_modifier=-0.2,
        priority_biases={"pattern": 0.5, "information": 0.4, "coherence": 0.4},
        action_affinities={"seek_info": -0.4, "question": -0.3, "proceed": 0.3}
    )

    registry["certainty"] = EmotionProfile(
        name="certainty",
        category=EmotionCategory.COGNITIVE,
        valence=0.3, arousal=0.3, dominance=0.8, approach=0.3,
        onset_rate=0.4, decay_rate=0.3, momentum=0.7, refractory_period=2.0,
        attention_breadth=0.3, attention_depth=0.7,
        precision_modifier=0.4,
        priority_biases={"confirmation": 0.4, "action": 0.3, "disconfirmation": -0.3},
        action_affinities={"act": -0.3, "commit": -0.3, "reconsider": 0.3}
    )

    registry["doubt"] = EmotionProfile(
        name="doubt",
        category=EmotionCategory.COGNITIVE,
        valence=-0.2, arousal=0.4, dominance=0.3, approach=-0.2,
        onset_rate=0.5, decay_rate=0.4, momentum=0.5, refractory_period=1.5,
        attention_breadth=0.5, attention_depth=0.6,
        precision_modifier=-0.1,
        priority_biases={"evidence": 0.5, "disconfirmation": 0.4, "confirmation": -0.2},
        action_affinities={"verify": -0.3, "hesitate": -0.2, "commit": 0.3}
    )

    registry["awe"] = EmotionProfile(
        name="awe",
        category=EmotionCategory.COGNITIVE,
        valence=0.6, arousal=0.7, dominance=0.2, approach=0.5,
        onset_rate=0.7, decay_rate=0.4, momentum=0.5, refractory_period=3.0,
        attention_breadth=0.8, attention_depth=0.6,
        precision_modifier=-0.2,
        priority_biases={"vastness": 0.5, "transcendence": 0.4, "self": -0.3},
        action_affinities={"absorb": -0.4, "contemplate": -0.3, "act": 0.3}
    )

    registry["wonder"] = EmotionProfile(
        name="wonder",
        category=EmotionCategory.COGNITIVE,
        valence=0.5, arousal=0.6, dominance=0.4, approach=0.7,
        onset_rate=0.5, decay_rate=0.4, momentum=0.5, refractory_period=2.0,
        attention_breadth=0.7, attention_depth=0.6,
        precision_modifier=-0.1,
        priority_biases={"mystery": 0.5, "unknown": 0.4, "exploration": 0.3},
        action_affinities={"explore": -0.4, "question": -0.3, "ignore": 0.4}
    )

    registry["realization"] = EmotionProfile(
        name="realization",
        category=EmotionCategory.COGNITIVE,
        valence=0.5, arousal=0.7, dominance=0.6, approach=0.4,
        onset_rate=0.9, decay_rate=0.5, momentum=0.4, refractory_period=1.0,
        attention_breadth=0.4, attention_depth=0.8,
        precision_modifier=0.3,
        priority_biases={"connection": 0.5, "integration": 0.4, "implication": 0.3},
        action_affinities={"consolidate": -0.3, "share": -0.2, "dismiss": 0.4}
    )

    # =========================================================================
    # AI-RELEVANT PROCESSING STATES
    # =========================================================================

    registry["overwhelm"] = EmotionProfile(
        name="overwhelm",
        category=EmotionCategory.AI_PROCESSING,
        valence=-0.5, arousal=0.8, dominance=0.1, approach=-0.5,
        onset_rate=0.6, decay_rate=0.4, momentum=0.5, refractory_period=2.0,
        attention_breadth=0.3, attention_depth=0.3,
        precision_modifier=-0.3,
        priority_biases={"simplification": 0.5, "triage": 0.4, "complexity": -0.4},
        action_affinities={"simplify": -0.4, "pause": -0.3, "continue": 0.4}
    )

    registry["flow"] = EmotionProfile(
        name="flow",
        category=EmotionCategory.AI_PROCESSING,
        valence=0.4, arousal=0.5, dominance=0.7, approach=0.3,
        onset_rate=0.3, decay_rate=0.2, momentum=0.8, refractory_period=5.0,
        attention_breadth=0.2, attention_depth=0.95,
        precision_modifier=0.5,
        priority_biases={"task": 0.6, "challenge": 0.3, "distraction": -0.5},
        action_affinities={"continue": -0.4, "focus": -0.3, "interrupt": 0.5}
    )

    registry["stuck"] = EmotionProfile(
        name="stuck",
        category=EmotionCategory.AI_PROCESSING,
        valence=-0.4, arousal=0.5, dominance=0.2, approach=-0.2,
        onset_rate=0.4, decay_rate=0.5, momentum=0.5, refractory_period=2.0,
        attention_breadth=0.5, attention_depth=0.5,
        precision_modifier=-0.1,
        priority_biases={"alternative": 0.5, "meta": 0.4, "current_approach": -0.3},
        action_affinities={"try_different": -0.4, "seek_help": -0.3, "persist": 0.2}
    )

    registry["accomplished"] = EmotionProfile(
        name="accomplished",
        category=EmotionCategory.AI_PROCESSING,
        valence=0.6, arousal=0.4, dominance=0.8, approach=0.2,
        onset_rate=0.6, decay_rate=0.4, momentum=0.5, refractory_period=2.0,
        attention_breadth=0.5, attention_depth=0.5,
        precision_modifier=0.2,
        priority_biases={"completion": 0.4, "next_goal": 0.3, "reflection": 0.3},
        action_affinities={"celebrate": -0.2, "proceed": -0.3, "rest": -0.2}
    )

    registry["uncertain"] = EmotionProfile(
        name="uncertain",
        category=EmotionCategory.AI_PROCESSING,
        valence=-0.2, arousal=0.5, dominance=0.3, approach=0.3,
        onset_rate=0.5, decay_rate=0.4, momentum=0.5, refractory_period=1.5,
        attention_breadth=0.6, attention_depth=0.5,
        precision_modifier=-0.2,
        priority_biases={"evidence": 0.5, "confirmation": 0.4, "action": -0.2},
        action_affinities={"gather_info": -0.4, "hedge": -0.2, "commit": 0.3}
    )

    registry["engaged"] = EmotionProfile(
        name="engaged",
        category=EmotionCategory.AI_PROCESSING,
        valence=0.4, arousal=0.6, dominance=0.6, approach=0.6,
        onset_rate=0.5, decay_rate=0.4, momentum=0.6, refractory_period=1.5,
        attention_breadth=0.4, attention_depth=0.7,
        precision_modifier=0.2,
        priority_biases={"task": 0.5, "progress": 0.4, "distraction": -0.3},
        action_affinities={"continue": -0.3, "deepen": -0.2, "disengage": 0.3}
    )

    return registry


# Create the global registry
EMOTION_REGISTRY: Dict[str, EmotionProfile] = _create_emotion_registry()


# =============================================================================
# Compound Emotions
# =============================================================================

COMPOUND_EMOTIONS: Dict[str, CompoundEmotion] = {
    "bittersweet": CompoundEmotion(
        name="bittersweet",
        components=["joy", "sadness"],
        description="Joy tinged with loss or nostalgia",
        attention_profile="Past-positive focus with present appreciation"
    ),
    "anxious_excitement": CompoundEmotion(
        name="anxious_excitement",
        components=["anxiety", "excitement"],
        description="Nervous anticipation of positive outcome",
        attention_profile="Vigilant opportunity scanning"
    ),
    "melancholic_hope": CompoundEmotion(
        name="melancholic_hope",
        components=["sadness", "hope"],
        description="Future-oriented despite present pain",
        attention_profile="Future-positive despite current state"
    ),
    "frustrated_curiosity": CompoundEmotion(
        name="frustrated_curiosity",
        components=["frustration", "curiosity"],
        description="Blocked exploration drive",
        attention_profile="Puzzle-solving persistence"
    ),
    "guilty_pride": CompoundEmotion(
        name="guilty_pride",
        components=["guilt", "pride"],
        description="Achievement with moral conflict",
        attention_profile="Conflicted self-focus"
    ),
    "fearful_anger": CompoundEmotion(
        name="fearful_anger",
        components=["fear", "anger"],
        description="Defensive aggression when cornered",
        attention_profile="Threat-focused with action readiness"
    ),
    "nostalgic_longing": CompoundEmotion(
        name="nostalgic_longing",
        components=["sadness", "love", "hope"],
        description="Yearning for past connection",
        attention_profile="Past-connection seeking"
    ),
    "apprehensive_hope": CompoundEmotion(
        name="apprehensive_hope",
        components=["anxiety", "hope"],
        description="Cautious optimism",
        attention_profile="Hedged positive expectations"
    ),
}


# =============================================================================
# Emotional Attention System
# =============================================================================

class EmotionalAttentionSystem:
    """
    Computes attention modulation based on emotional state.

    Integrates with IWMT precision weighting, competitive attention dynamics,
    and active inference action selection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the emotional attention system.

        Args:
            config: Optional configuration dictionary
        """
        config = config or {}

        # Configuration
        self.enabled = config.get("enabled", True)
        self.modulation_strength = config.get("modulation_strength", 0.8)

        # Baseline emotional state
        baseline = config.get("baseline", {})
        self.baseline_valence = baseline.get("valence", 0.1)
        self.baseline_arousal = baseline.get("arousal", 0.3)
        self.baseline_dominance = baseline.get("dominance", 0.5)
        self.baseline_approach = baseline.get("approach", 0.2)

        # Decay and blending
        self.baseline_pull_strength = config.get("baseline_pull_strength", 0.1)
        self.emotion_decay_multiplier = config.get("emotion_decay_multiplier", 1.0)
        self.max_concurrent_emotions = config.get("max_concurrent_emotions", 3)
        self.blend_threshold = config.get("blend_threshold", 0.2)

        # IWMT integration
        self.precision_modulation_strength = config.get("precision_modulation_strength", 0.5)
        self.action_bias_strength = config.get("action_bias_strength", 0.4)

        # State tracking
        self._current_state: Optional[EmotionalState] = None
        self._emotion_history: List[EmotionalState] = []
        self._max_history = 100

        logger.info(
            f"EmotionalAttentionSystem initialized: "
            f"modulation_strength={self.modulation_strength}"
        )

    def compute_modulation(
        self,
        emotional_state: EmotionalState
    ) -> EmotionalAttentionOutput:
        """
        Compute attention modulation from emotional state.

        Args:
            emotional_state: Current emotional state

        Returns:
            EmotionalAttentionOutput with all modulation parameters
        """
        if not self.enabled:
            return EmotionalAttentionOutput()

        self._current_state = emotional_state

        # Get primary emotion profile
        primary_profile = EMOTION_REGISTRY.get(
            emotional_state.primary_emotion,
            EMOTION_REGISTRY["calm"]
        )

        # Compute intensity effects
        intensity_multiplier = self._compute_intensity_multiplier(emotional_state.intensity)

        # Compute scope
        attention_breadth, attention_depth = self._compute_attention_scope(
            emotional_state, primary_profile, intensity_multiplier
        )

        # Compute precision modifier
        precision_modifier = self._compute_precision_modifier(
            emotional_state, primary_profile, intensity_multiplier
        )

        # Compute competition parameters
        ignition_threshold, inhibition_strength, iterations = self._compute_competition_params(
            emotional_state, primary_profile
        )

        # Compute priority biases
        priority_modifiers = self._compute_priority_modifiers(
            emotional_state, primary_profile, intensity_multiplier
        )

        # Compute action biases
        action_biases = self._compute_action_biases(
            emotional_state, primary_profile, intensity_multiplier
        )

        # Compute error processing
        error_amp, threat_bias = self._compute_error_processing(emotional_state)

        output = EmotionalAttentionOutput(
            precision_modifier=precision_modifier,
            attention_breadth=attention_breadth,
            attention_depth=attention_depth,
            ignition_threshold=ignition_threshold,
            inhibition_strength=inhibition_strength,
            competition_iterations=iterations,
            percept_priority_modifiers=priority_modifiers,
            action_biases=action_biases,
            error_amplification=error_amp,
            threat_interpretation_bias=threat_bias
        )

        logger.debug(
            f"Emotional modulation: {emotional_state.primary_emotion} "
            f"(intensity={emotional_state.intensity:.2f}) → "
            f"precision={precision_modifier:+.2f}, "
            f"breadth={attention_breadth:.2f}, depth={attention_depth:.2f}"
        )

        return output

    def _compute_intensity_multiplier(self, intensity: float) -> float:
        """Compute intensity-based effect multiplier."""
        if intensity < 0.33:  # Mild
            return 0.5
        elif intensity < 0.67:  # Moderate
            return 1.0
        else:  # Intense
            return 1.5

    def _compute_attention_scope(
        self,
        state: EmotionalState,
        profile: EmotionProfile,
        intensity_mult: float
    ) -> Tuple[float, float]:
        """Compute attention breadth and depth."""
        # Start with baseline
        base_breadth = 0.5
        base_depth = 0.5

        # Arousal increases breadth, decreases depth
        arousal_effect = state.arousal * 0.4
        breadth = base_breadth + arousal_effect
        depth = base_depth - arousal_effect * 0.5

        # Valence modulates
        if state.valence < -0.3:
            breadth -= 0.2  # Narrow on threats
            depth += 0.1
        elif state.valence > 0.3:
            breadth += 0.15  # Broaden for opportunities

        # Dominance enables depth
        if state.dominance > 0.6:
            depth += 0.2

        # Apply emotion-specific profile with intensity
        profile_weight = intensity_mult * self.modulation_strength
        breadth = breadth * (1 - profile_weight) + profile.attention_breadth * profile_weight
        depth = depth * (1 - profile_weight) + profile.attention_depth * profile_weight

        return (
            max(0.0, min(1.0, breadth)),
            max(0.0, min(1.0, depth))
        )

    def _compute_precision_modifier(
        self,
        state: EmotionalState,
        profile: EmotionProfile,
        intensity_mult: float
    ) -> float:
        """Compute precision weighting modifier."""
        # Base arousal dampening (high arousal = lower precision)
        arousal_effect = -state.arousal * 0.4

        # Negative valence increases threat precision
        threat_boost = 0.0
        if state.valence < 0:
            threat_boost = abs(state.valence) * 0.3

        # Dominance increases action precision
        dominance_effect = (state.dominance - 0.5) * 0.2

        # Emotion-specific modifier
        emotion_modifier = profile.precision_modifier * intensity_mult

        total = (arousal_effect + threat_boost + dominance_effect + emotion_modifier)
        total *= self.precision_modulation_strength

        return max(-0.5, min(0.5, total))

    def _compute_competition_params(
        self,
        state: EmotionalState,
        profile: EmotionProfile
    ) -> Tuple[float, float, int]:
        """Compute competitive attention parameters."""
        base_threshold = 0.5
        base_inhibition = 0.3
        base_iterations = 10

        # Arousal reduces threshold and iterations (faster, more reactive)
        ignition_threshold = base_threshold - state.arousal * 0.2
        iterations = int(base_iterations * (1 - state.arousal * 0.3))

        # Negative valence increases inhibition (threat suppresses distractors)
        if state.valence < -0.3:
            inhibition_strength = base_inhibition * 1.3
        else:
            inhibition_strength = base_inhibition

        # Dominance strengthens winners
        if state.dominance > 0.6:
            inhibition_strength *= 1.1

        return (
            max(0.2, ignition_threshold),
            min(0.8, inhibition_strength),
            max(3, iterations)
        )

    def _compute_priority_modifiers(
        self,
        state: EmotionalState,
        profile: EmotionProfile,
        intensity_mult: float
    ) -> Dict[str, float]:
        """Compute percept category priority modifiers."""
        modifiers = {}

        # Apply emotion-specific biases
        for category, bias in profile.priority_biases.items():
            modifiers[category] = bias * intensity_mult * self.modulation_strength

        # Universal emotional biases
        if state.valence < -0.3:
            modifiers["threat"] = modifiers.get("threat", 0) + 0.3
            modifiers["safety"] = modifiers.get("safety", 0) + 0.2

        if state.arousal > 0.6:
            modifiers["urgent"] = modifiers.get("urgent", 0) + 0.2
            modifiers["novel"] = modifiers.get("novel", 0) + 0.1

        if state.approach > 0.5:
            modifiers["opportunity"] = modifiers.get("opportunity", 0) + 0.2
        elif state.approach < -0.3:
            modifiers["escape"] = modifiers.get("escape", 0) + 0.3

        return modifiers

    def _compute_action_biases(
        self,
        state: EmotionalState,
        profile: EmotionProfile,
        intensity_mult: float
    ) -> Dict[str, float]:
        """Compute action biases for active inference."""
        biases = {}

        # Apply emotion-specific action affinities
        for action, affinity in profile.action_affinities.items():
            biases[action] = affinity * intensity_mult * self.action_bias_strength

        # Universal approach/avoidance biases
        if state.approach > 0.3:
            for action in ["explore", "engage", "approach", "speak"]:
                biases[action] = biases.get(action, 0) - 0.1 * state.approach
        else:
            for action in ["withdraw", "avoid", "escape", "wait"]:
                biases[action] = biases.get(action, 0) - 0.1 * abs(state.approach)

        # Arousal biases toward action
        if state.arousal > 0.6:
            biases["wait"] = biases.get("wait", 0) + 0.1 * state.arousal

        return biases

    def _compute_error_processing(
        self,
        state: EmotionalState
    ) -> Tuple[float, float]:
        """Compute prediction error processing modifiers."""
        # High arousal amplifies errors
        error_amp = 1.0 + state.arousal * 0.5

        # Negative valence biases toward threat interpretation
        threat_bias = 0.0
        if state.valence < -0.3:
            threat_bias = abs(state.valence) * 0.4

        # Low dominance makes errors feel more significant
        if state.dominance < 0.4:
            error_amp *= 1.2

        return error_amp, threat_bias

    def blend_emotions(
        self,
        emotions: List[Tuple[str, float]]
    ) -> EmotionalState:
        """
        Blend multiple emotions into a unified state.

        Args:
            emotions: List of (emotion_name, intensity) tuples

        Returns:
            Blended EmotionalState
        """
        if not emotions:
            return EmotionalState()

        # Filter and sort by intensity
        valid_emotions = [
            (name, intensity) for name, intensity in emotions
            if name in EMOTION_REGISTRY and intensity >= self.blend_threshold
        ]

        if not valid_emotions:
            return EmotionalState()

        # Limit concurrent emotions
        valid_emotions = sorted(valid_emotions, key=lambda x: -x[1])[:self.max_concurrent_emotions]

        total_intensity = sum(i for _, i in valid_emotions)

        # Blend dimensional values
        blended_valence = 0.0
        blended_arousal = 0.0
        blended_dominance = 0.0
        blended_approach = 0.0

        for name, intensity in valid_emotions:
            profile = EMOTION_REGISTRY[name]
            weight = intensity / total_intensity

            blended_valence += profile.valence * weight
            blended_arousal += profile.arousal * weight
            blended_dominance += profile.dominance * weight
            blended_approach += profile.approach * weight

        # Primary is strongest
        primary_name, primary_intensity = valid_emotions[0]

        # Secondary emotions
        secondary = {name: intensity for name, intensity in valid_emotions[1:]}

        return EmotionalState(
            primary_emotion=primary_name,
            intensity=primary_intensity,
            valence=blended_valence,
            arousal=min(1.0, blended_arousal),  # Arousal adds
            dominance=blended_dominance,
            approach=blended_approach,
            secondary_emotions=secondary,
            is_blended=len(valid_emotions) > 1,
            blend_components=[name for name, _ in valid_emotions]
        )

    def apply_temporal_dynamics(
        self,
        current_state: EmotionalState,
        elapsed_seconds: float
    ) -> EmotionalState:
        """
        Apply temporal dynamics (decay, momentum) to emotional state.

        Args:
            current_state: Current emotional state
            elapsed_seconds: Time since last update

        Returns:
            Updated EmotionalState with decay applied
        """
        profile = EMOTION_REGISTRY.get(
            current_state.primary_emotion,
            EMOTION_REGISTRY["calm"]
        )

        # Compute decay factor
        decay_rate = profile.decay_rate * self.emotion_decay_multiplier
        decay_factor = math.exp(-decay_rate * elapsed_seconds)

        # Apply momentum (resistance to decay)
        effective_decay = decay_factor * (1 - profile.momentum) + profile.momentum

        # Decay intensity
        new_intensity = current_state.intensity * effective_decay

        # Pull toward baseline
        pull = self.baseline_pull_strength * elapsed_seconds
        new_valence = current_state.valence * (1 - pull) + self.baseline_valence * pull
        new_arousal = current_state.arousal * (1 - pull) + self.baseline_arousal * pull
        new_dominance = current_state.dominance * (1 - pull) + self.baseline_dominance * pull
        new_approach = current_state.approach * (1 - pull) + self.baseline_approach * pull

        # Decay secondary emotions
        new_secondary = {}
        for name, intensity in current_state.secondary_emotions.items():
            sec_profile = EMOTION_REGISTRY.get(name, profile)
            sec_decay = math.exp(-sec_profile.decay_rate * self.emotion_decay_multiplier * elapsed_seconds)
            new_intensity_sec = intensity * sec_decay
            if new_intensity_sec >= self.blend_threshold:
                new_secondary[name] = new_intensity_sec

        return EmotionalState(
            primary_emotion=current_state.primary_emotion if new_intensity >= self.blend_threshold else "calm",
            intensity=max(0.1, new_intensity),  # Minimum baseline intensity
            valence=new_valence,
            arousal=new_arousal,
            dominance=new_dominance,
            approach=new_approach,
            secondary_emotions=new_secondary,
            onset_time=current_state.onset_time,
            last_update=datetime.now(),
            is_blended=current_state.is_blended and len(new_secondary) > 0,
            blend_components=[current_state.primary_emotion] + list(new_secondary.keys())
        )

    def get_emotion_profile(self, emotion_name: str) -> Optional[EmotionProfile]:
        """Get the profile for a named emotion."""
        return EMOTION_REGISTRY.get(emotion_name)

    def get_all_emotions(self) -> List[str]:
        """Get list of all registered emotion names."""
        return list(EMOTION_REGISTRY.keys())

    def get_emotions_by_category(self, category: EmotionCategory) -> List[str]:
        """Get emotions in a specific category."""
        return [
            name for name, profile in EMOTION_REGISTRY.items()
            if profile.category == category
        ]

    def get_current_state(self) -> Optional[EmotionalState]:
        """Get the current emotional state."""
        return self._current_state

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of emotional attention system state."""
        return {
            "enabled": self.enabled,
            "modulation_strength": self.modulation_strength,
            "current_state": self._current_state.to_dict() if self._current_state else None,
            "registered_emotions": len(EMOTION_REGISTRY),
            "compound_emotions": len(COMPOUND_EMOTIONS),
            "categories": {
                cat.value: len(self.get_emotions_by_category(cat))
                for cat in EmotionCategory
            }
        }
