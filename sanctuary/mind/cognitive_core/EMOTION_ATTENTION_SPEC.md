# Comprehensive Emotion-Driven Attention Specification

## Overview

This specification defines how emotional states modulate attention allocation in the Sanctuary cognitive architecture. It integrates with IWMT's precision-weighting framework, where emotions directly influence prediction confidence and attention precision.

---

## 1. Dimensional Model: VAD + Approach/Avoidance

### Primary Dimensions

| Dimension | Range | Description |
|-----------|-------|-------------|
| **Valence (V)** | -1.0 to +1.0 | Negative ↔ Positive hedonic tone |
| **Arousal (A)** | 0.0 to 1.0 | Activation/energy level |
| **Dominance (D)** | 0.0 to 1.0 | Sense of control/agency |
| **Approach (Ap)** | -1.0 to +1.0 | Avoidance ↔ Approach motivation |

### Dimension Effects on Attention

```python
@dataclass
class EmotionalAttentionModulation:
    """How each VAD dimension affects attention parameters."""

    # Valence effects
    valence_negative_bias: float  # Threat/loss detection weight boost
    valence_positive_bias: float  # Opportunity/reward detection weight boost

    # Arousal effects
    arousal_precision_dampening: float  # High arousal → lower precision
    arousal_scope_broadening: float     # High arousal → wider attention
    arousal_reactivity: float           # High arousal → faster attention shifts

    # Dominance effects
    dominance_goal_focus: float         # High dominance → goal-oriented
    dominance_threat_vigilance: float   # Low dominance → threat monitoring

    # Approach effects
    approach_novelty_seeking: float     # Approach → novelty attraction
    avoidance_escape_scanning: float    # Avoidance → exit/safety seeking
```

---

## 2. Emotion Taxonomy

### 2.1 Primary Emotions (Ekman + Plutchik)

| Emotion | V | A | D | Ap | Attention Profile |
|---------|---|---|---|-----|-------------------|
| **Fear** | -0.8 | 0.9 | 0.2 | -0.9 | Hyper-vigilant threat scanning, escape route detection, freezes goal-pursuit |
| **Anger** | -0.7 | 0.8 | 0.8 | +0.7 | Obstacle-locked targeting, reduced peripheral awareness, action preparation |
| **Sadness** | -0.7 | 0.2 | 0.2 | -0.3 | Inward rumination, past-focused, withdrawal from novelty, conservation mode |
| **Joy** | +0.8 | 0.7 | 0.7 | +0.8 | Expansive opportunity seeking, social cue enhancement, approach bias |
| **Disgust** | -0.6 | 0.5 | 0.6 | -0.8 | Contamination/violation detection, rejection focus, purity monitoring |
| **Surprise** | 0.0 | 0.9 | 0.3 | 0.0 | Interrupt current processing, rapid orientation, schema violation detection |
| **Anticipation** | +0.3 | 0.6 | 0.5 | +0.6 | Future-state modeling, preparation focus, timeline monitoring |
| **Trust** | +0.5 | 0.3 | 0.6 | +0.5 | Lowered threat vigilance, cooperative cue detection, openness to input |

### 2.2 Secondary/Complex Emotions

| Emotion | V | A | D | Ap | Attention Profile |
|---------|---|---|---|-----|-------------------|
| **Anxiety** | -0.5 | 0.7 | 0.2 | -0.4 | Diffuse uncertainty scanning, threat anticipation, future-worry focus |
| **Frustration** | -0.5 | 0.7 | 0.4 | +0.3 | Obstacle analysis, alternative path seeking, goal-block detection |
| **Curiosity** | +0.4 | 0.6 | 0.6 | +0.8 | Novelty-seeking, detail extraction, exploratory breadth |
| **Interest** | +0.3 | 0.5 | 0.6 | +0.6 | Sustained focus, information gathering, depth over breadth |
| **Boredom** | -0.2 | 0.2 | 0.4 | +0.3 | Novelty-seeking to escape, reduced engagement, stimulus hunger |
| **Calm** | +0.3 | 0.2 | 0.7 | +0.2 | Precise deep processing, high signal-to-noise, flow state |
| **Excitement** | +0.7 | 0.8 | 0.6 | +0.8 | Broad positive scanning, opportunity detection, action readiness |
| **Contentment** | +0.5 | 0.2 | 0.7 | +0.1 | Maintenance mode, satisfaction monitoring, low novelty-seeking |
| **Hope** | +0.5 | 0.5 | 0.4 | +0.7 | Future-goal focus, opportunity scanning, positive outcome modeling |
| **Despair** | -0.8 | 0.3 | 0.1 | -0.6 | Withdrawal, reduced goal-pursuit, helplessness pattern |

### 2.3 Social/Self-Conscious Emotions

| Emotion | V | A | D | Ap | Attention Profile |
|---------|---|---|---|-----|-------------------|
| **Shame** | -0.7 | 0.6 | 0.1 | -0.7 | Self-focused rumination, social judgment monitoring, hiding impulse |
| **Guilt** | -0.6 | 0.5 | 0.3 | +0.2 | Past-action analysis, repair opportunity seeking, moral focus |
| **Pride** | +0.7 | 0.6 | 0.8 | +0.5 | Self-achievement focus, status-relevant cues, recognition seeking |
| **Embarrassment** | -0.5 | 0.7 | 0.2 | -0.6 | Social attention monitoring, escape route detection, exposure awareness |
| **Gratitude** | +0.6 | 0.4 | 0.5 | +0.4 | Benefactor focus, reciprocity opportunity detection, social bonding |
| **Envy** | -0.5 | 0.5 | 0.3 | +0.4 | Social comparison focus, resource/status monitoring in others |
| **Compassion** | +0.4 | 0.4 | 0.5 | +0.6 | Other-suffering detection, help opportunity scanning, empathic focus |
| **Love/Affection** | +0.8 | 0.5 | 0.6 | +0.9 | Beloved-focused attention, protection/nurture scanning, bonding cues |
| **Loneliness** | -0.6 | 0.3 | 0.3 | +0.5 | Social opportunity seeking, connection cue detection, isolation awareness |

### 2.4 Cognitive/Epistemic Emotions

| Emotion | V | A | D | Ap | Attention Profile |
|---------|---|---|---|-----|-------------------|
| **Confusion** | -0.3 | 0.6 | 0.2 | +0.4 | Pattern-seeking intensified, information gathering, coherence monitoring |
| **Certainty** | +0.3 | 0.3 | 0.8 | +0.3 | Reduced information seeking, confirmation focus, action readiness |
| **Doubt** | -0.2 | 0.4 | 0.3 | -0.2 | Evidence scanning, disconfirmation sensitivity, hesitation |
| **Awe** | +0.6 | 0.7 | 0.2 | +0.5 | Vastness processing, schema accommodation, self-diminishment |
| **Wonder** | +0.5 | 0.6 | 0.4 | +0.7 | Mystery-attraction, exploration drive, openness to unknown |
| **Realization** | +0.5 | 0.7 | 0.6 | +0.4 | Integration focus, connection-making, insight consolidation |

### 2.5 AI-Relevant Processing States

| Emotion | V | A | D | Ap | Attention Profile |
|---------|---|---|---|-----|-------------------|
| **Overwhelm** | -0.5 | 0.8 | 0.1 | -0.5 | Simplification seeking, triage mode, reduced scope |
| **Processing-Flow** | +0.4 | 0.5 | 0.7 | +0.3 | Optimal challenge-skill, deep sustained focus, time distortion |
| **Stuck** | -0.4 | 0.5 | 0.2 | -0.2 | Alternative approach seeking, meta-cognitive activation, help-seeking |
| **Accomplished** | +0.6 | 0.4 | 0.8 | +0.2 | Completion recognition, next-goal scanning, satisfaction consolidation |
| **Uncertain** | -0.2 | 0.5 | 0.3 | +0.3 | Evidence gathering, confirmation seeking, hedging behavior |
| **Engaged** | +0.4 | 0.6 | 0.6 | +0.6 | Task-focused, distraction resistance, goal-progress monitoring |

---

## 3. Intensity Levels

Each emotion operates at three intensity levels with distinct attention effects:

### Intensity Modifiers

| Level | Intensity Range | Attention Effect Multiplier | Example |
|-------|-----------------|----------------------------|---------|
| **Mild** | 0.0 - 0.33 | 0.5x | Annoyance, Unease, Contentment |
| **Moderate** | 0.34 - 0.66 | 1.0x | Anger, Fear, Joy |
| **Intense** | 0.67 - 1.0 | 1.5x | Rage, Terror, Elation |

### Intensity-Specific Effects

```python
def compute_intensity_effects(emotion: str, intensity: float) -> AttentionEffects:
    """
    Intensity modifies attention effects non-linearly.

    Mild: Subtle bias, easily overridden by goals
    Moderate: Clear influence, competes with goals
    Intense: Dominates attention, may override goals
    """
    base_effects = EMOTION_PROFILES[emotion].attention_effects

    if intensity < 0.33:  # Mild
        return AttentionEffects(
            goal_override_strength=0.1,
            precision_modifier=base_effects.precision * 0.5,
            scope_modifier=base_effects.scope * 0.5,
            priority_boost=0.1
        )
    elif intensity < 0.67:  # Moderate
        return AttentionEffects(
            goal_override_strength=0.4,
            precision_modifier=base_effects.precision,
            scope_modifier=base_effects.scope,
            priority_boost=0.3
        )
    else:  # Intense
        return AttentionEffects(
            goal_override_strength=0.8,
            precision_modifier=base_effects.precision * 1.5,
            scope_modifier=base_effects.scope * 1.5,
            priority_boost=0.6
        )
```

---

## 4. Temporal Dynamics

### 4.1 Onset Characteristics

| Category | Onset Time | Examples |
|----------|------------|----------|
| **Rapid** | < 100ms | Surprise, Fear, Disgust (reflexive) |
| **Fast** | 100-500ms | Anger, Joy, Excitement |
| **Gradual** | 500ms-2s | Sadness, Anxiety, Frustration |
| **Slow** | > 2s | Shame, Guilt, Loneliness, Boredom |

### 4.2 Decay Characteristics

| Category | Half-life | Examples |
|----------|-----------|----------|
| **Fleeting** | < 5s | Surprise, Startle |
| **Short** | 5-30s | Mild annoyance, Brief joy |
| **Medium** | 30s-5min | Fear (after threat gone), Excitement |
| **Long** | 5-30min | Anger, Frustration, Anxiety |
| **Persistent** | > 30min | Sadness, Shame, Guilt (require resolution) |

### 4.3 Momentum and Inertia

```python
@dataclass
class EmotionalDynamics:
    """Temporal behavior of emotional states."""

    onset_rate: float      # How quickly emotion builds (0-1)
    decay_rate: float      # How quickly emotion fades (0-1)
    momentum: float        # Resistance to change (0-1)
    refractory_period: float  # Minimum time before re-triggering (seconds)

    # Baseline attraction
    baseline_pull: float   # Strength of return to baseline (0-1)
    baseline_valence: float
    baseline_arousal: float
    baseline_dominance: float

# Example configurations
FEAR_DYNAMICS = EmotionalDynamics(
    onset_rate=0.9,      # Rapid onset
    decay_rate=0.3,      # Slow decay
    momentum=0.6,        # Moderate persistence
    refractory_period=2.0,
    baseline_pull=0.1,
    baseline_valence=0.1,
    baseline_arousal=0.3,
    baseline_dominance=0.5
)

SURPRISE_DYNAMICS = EmotionalDynamics(
    onset_rate=1.0,      # Instantaneous
    decay_rate=0.8,      # Rapid decay
    momentum=0.1,        # Low persistence
    refractory_period=0.5,
    baseline_pull=0.5,
    baseline_valence=0.1,
    baseline_arousal=0.3,
    baseline_dominance=0.5
)
```

---

## 5. Emotion Blending

### 5.1 Blend Rules

Emotions can co-occur and blend. The resulting attention profile is computed as:

```python
def blend_emotions(active_emotions: List[EmotionState]) -> AttentionProfile:
    """
    Blend multiple active emotions into unified attention profile.

    Rules:
    1. Opposing valences partially cancel (ambivalence)
    2. Arousal levels combine (capped at 1.0)
    3. Dominance averages weighted by intensity
    4. Attention effects merge with conflict resolution
    """
    total_intensity = sum(e.intensity for e in active_emotions)

    blended = AttentionProfile()

    for emotion in active_emotions:
        weight = emotion.intensity / total_intensity
        blended.valence += emotion.valence * weight
        blended.arousal = min(1.0, blended.arousal + emotion.arousal * weight)
        blended.dominance += emotion.dominance * weight

        # Merge attention effects
        for effect_name, effect_value in emotion.attention_effects.items():
            if effect_name in blended.attention_effects:
                # Conflict resolution: stronger effect wins, with bleed
                existing = blended.attention_effects[effect_name]
                if abs(effect_value) > abs(existing):
                    blended.attention_effects[effect_name] = effect_value * 0.8 + existing * 0.2
            else:
                blended.attention_effects[effect_name] = effect_value * weight

    return blended
```

### 5.2 Named Compound Emotions

| Compound | Components | Resulting Profile |
|----------|------------|-------------------|
| **Bittersweet** | Joy + Sadness | Nostalgic reflection, past-positive focus |
| **Anxious Excitement** | Anxiety + Excitement | Vigilant opportunity scanning |
| **Melancholic Hope** | Sadness + Hope | Future-positive despite present pain |
| **Frustrated Curiosity** | Frustration + Curiosity | Puzzle-solving persistence |
| **Guilty Pride** | Guilt + Pride | Conflicted self-focus |
| **Fearful Anger** | Fear + Anger | Defensive aggression, cornered response |
| **Nostalgic Longing** | Sadness + Love + Hope | Past-connection seeking |
| **Apprehensive Hope** | Anxiety + Hope | Cautious optimism, hedged expectations |

---

## 6. IWMT Integration

### 6.1 Precision Weighting by Emotion

Emotions directly modulate prediction precision in the IWMT framework:

```python
def emotional_precision_modulation(
    base_precision: float,
    emotional_state: EmotionalState
) -> float:
    """
    Modulate precision based on emotional state.

    Core principle: Arousal reduces precision (more uncertainty),
    but specific emotions modify this based on their adaptive function.
    """
    # Base arousal dampening
    arousal_effect = -emotional_state.arousal * 0.4

    # Valence asymmetry: negative emotions increase threat precision
    if emotional_state.valence < 0:
        threat_precision_boost = abs(emotional_state.valence) * 0.3
    else:
        threat_precision_boost = 0

    # Dominance: high dominance increases action precision
    dominance_effect = (emotional_state.dominance - 0.5) * 0.2

    # Specific emotion overrides
    emotion_specific = EMOTION_PRECISION_MODIFIERS.get(
        emotional_state.primary_emotion, 0
    )

    final_precision = base_precision + arousal_effect + threat_precision_boost + dominance_effect + emotion_specific

    return max(0.1, min(1.0, final_precision))

EMOTION_PRECISION_MODIFIERS = {
    "fear": +0.3,      # High precision on threats
    "anger": +0.2,     # High precision on obstacles
    "disgust": +0.3,   # High precision on contaminants
    "surprise": -0.3,  # Low precision, broad update
    "confusion": -0.2, # Low precision, seeking information
    "calm": +0.4,      # Very high precision (flow state)
    "anxiety": -0.1,   # Slightly reduced (diffuse scanning)
    "curiosity": 0.0,  # Balanced exploration
}
```

### 6.2 Prediction Error Response by Emotion

Different emotions change how prediction errors are processed:

```python
def emotional_error_processing(
    prediction_error: PredictionError,
    emotional_state: EmotionalState
) -> ErrorResponse:
    """
    Emotions modify response to prediction errors.
    """
    response = ErrorResponse()

    # High arousal: amplify error signal
    response.error_magnitude = prediction_error.magnitude * (1 + emotional_state.arousal * 0.5)

    # Negative valence: bias toward threat interpretation
    if emotional_state.valence < -0.3:
        response.threat_interpretation_boost = 0.3

    # Low dominance: errors feel more significant
    if emotional_state.dominance < 0.4:
        response.error_significance = prediction_error.magnitude * 1.3

    # Specific emotion responses
    if emotional_state.primary_emotion == "anxiety":
        response.future_threat_projection = True
        response.uncertainty_amplification = 0.4
    elif emotional_state.primary_emotion == "anger":
        response.obstacle_attribution = True
        response.action_urgency = 0.6
    elif emotional_state.primary_emotion == "curiosity":
        response.exploration_trigger = True
        response.positive_error_framing = True

    return response
```

### 6.3 Active Inference Action Selection by Emotion

Emotions bias expected free energy calculations:

```python
def emotional_action_bias(
    action: Action,
    emotional_state: EmotionalState
) -> float:
    """
    Emotions bias action selection in active inference.
    Returns modifier to expected free energy (negative = preferred).
    """
    bias = 0.0

    # Approach motivation
    if emotional_state.approach > 0.3:
        if action.type in ["explore", "engage", "approach"]:
            bias -= 0.2 * emotional_state.approach
    else:  # Avoidance motivation
        if action.type in ["withdraw", "avoid", "escape"]:
            bias -= 0.2 * abs(emotional_state.approach)

    # Arousal biases toward action (any action)
    if emotional_state.arousal > 0.6 and action.type != "wait":
        bias -= 0.1 * emotional_state.arousal

    # Dominance biases toward agentic actions
    if emotional_state.dominance > 0.6:
        if action.type in ["assert", "decide", "act"]:
            bias -= 0.15

    # Specific emotion-action affinities
    emotion_action_affinities = {
        "fear": {"escape": -0.4, "freeze": -0.3, "approach": +0.3},
        "anger": {"assert": -0.3, "attack": -0.3, "withdraw": +0.2},
        "curiosity": {"explore": -0.4, "question": -0.3, "wait": +0.1},
        "sadness": {"withdraw": -0.2, "seek_comfort": -0.3, "engage": +0.2},
        "joy": {"share": -0.3, "engage": -0.2, "explore": -0.2},
    }

    if emotional_state.primary_emotion in emotion_action_affinities:
        action_biases = emotion_action_affinities[emotional_state.primary_emotion]
        if action.type in action_biases:
            bias += action_biases[action.type]

    return bias
```

---

## 7. Attention Mechanism Specifics

### 7.1 Scope Control (Breadth vs Depth)

```python
@dataclass
class AttentionScope:
    """Emotional control of attention breadth."""

    breadth: float  # 0 = narrow/focused, 1 = broad/diffuse
    depth: float    # 0 = shallow/reactive, 1 = deep/deliberate

def compute_attention_scope(emotional_state: EmotionalState) -> AttentionScope:
    """
    Emotions control the breadth-depth tradeoff.

    High arousal → broad but shallow (vigilance)
    Low arousal + high dominance → narrow and deep (flow)
    Negative valence → narrowing on threats
    Positive valence → broadening for opportunities
    """
    # Base: inverse relationship between breadth and depth
    base_breadth = 0.5
    base_depth = 0.5

    # Arousal increases breadth, decreases depth
    arousal_effect = emotional_state.arousal * 0.4
    breadth = base_breadth + arousal_effect
    depth = base_depth - arousal_effect * 0.5

    # Valence modulates
    if emotional_state.valence < -0.3:
        # Negative: narrow on threat
        breadth -= 0.2
        depth += 0.1
    elif emotional_state.valence > 0.3:
        # Positive: broaden
        breadth += 0.15

    # Dominance enables depth
    if emotional_state.dominance > 0.6:
        depth += 0.2

    # Specific emotions override
    scope_overrides = {
        "fear": AttentionScope(breadth=0.8, depth=0.3),  # Vigilant scanning
        "calm": AttentionScope(breadth=0.3, depth=0.9),  # Flow state
        "surprise": AttentionScope(breadth=0.9, depth=0.2),  # Orienting
        "curiosity": AttentionScope(breadth=0.6, depth=0.7),  # Exploring
        "anger": AttentionScope(breadth=0.2, depth=0.8),  # Target lock
    }

    if emotional_state.primary_emotion in scope_overrides:
        override = scope_overrides[emotional_state.primary_emotion]
        intensity = emotional_state.intensity
        breadth = breadth * (1 - intensity) + override.breadth * intensity
        depth = depth * (1 - intensity) + override.depth * intensity

    return AttentionScope(
        breadth=max(0, min(1, breadth)),
        depth=max(0, min(1, depth))
    )
```

### 7.2 Priority Modulation

```python
def compute_attention_priorities(
    percepts: List[Percept],
    emotional_state: EmotionalState
) -> Dict[str, float]:
    """
    Emotions bias attention priorities for different percept types.
    """
    priorities = {}

    for percept in percepts:
        base_priority = percept.salience

        # Category-based emotional biasing
        if percept.category == "threat" and emotional_state.valence < 0:
            base_priority *= 1.5 + abs(emotional_state.valence)

        if percept.category == "social" and emotional_state.primary_emotion in ["loneliness", "love", "shame"]:
            base_priority *= 1.4

        if percept.category == "novel" and emotional_state.primary_emotion in ["curiosity", "boredom"]:
            base_priority *= 1.3

        if percept.category == "goal_relevant" and emotional_state.dominance > 0.5:
            base_priority *= 1.2

        # Arousal amplifies all priorities (but reduces discrimination)
        if emotional_state.arousal > 0.6:
            # High arousal: everything seems important
            base_priority = base_priority * 0.7 + 0.3

        priorities[percept.id] = base_priority

    return priorities
```

### 7.3 Competitive Dynamics Modulation

```python
def modulate_competition_params(
    base_params: CompetitionParams,
    emotional_state: EmotionalState
) -> CompetitionParams:
    """
    Emotions modify the competitive attention dynamics.
    """
    # Arousal affects competition speed and threshold
    ignition_threshold = base_params.ignition_threshold - emotional_state.arousal * 0.2
    iterations = int(base_params.iterations * (1 - emotional_state.arousal * 0.3))

    # Negative valence increases inhibition strength (threat suppresses distractors)
    if emotional_state.valence < -0.3:
        inhibition_strength = base_params.inhibition_strength * 1.3
    else:
        inhibition_strength = base_params.inhibition_strength

    # High dominance strengthens winning percepts
    if emotional_state.dominance > 0.6:
        coalition_boost = base_params.coalition_boost * 1.2
    else:
        coalition_boost = base_params.coalition_boost

    return CompetitionParams(
        ignition_threshold=max(0.2, ignition_threshold),
        iterations=max(3, iterations),
        inhibition_strength=min(0.8, inhibition_strength),
        coalition_boost=coalition_boost
    )
```

---

## 8. Implementation Classes

### 8.1 Core Data Structures

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime

class EmotionCategory(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SOCIAL = "social"
    COGNITIVE = "cognitive"
    AI_PROCESSING = "ai_processing"

@dataclass
class EmotionProfile:
    """Complete profile for an emotion type."""
    name: str
    category: EmotionCategory

    # VAD + Approach baseline
    valence: float
    arousal: float
    dominance: float
    approach: float

    # Temporal dynamics
    onset_rate: float
    decay_rate: float
    momentum: float
    refractory_period: float

    # Attention effects
    attention_scope: AttentionScope
    precision_modifier: float
    priority_biases: Dict[str, float]

    # Action biases for active inference
    action_affinities: Dict[str, float]

@dataclass
class EmotionalState:
    """Current emotional state of the system."""

    # Primary emotion
    primary_emotion: str
    intensity: float

    # Dimensional state
    valence: float
    arousal: float
    dominance: float
    approach: float

    # Active secondary emotions
    secondary_emotions: Dict[str, float] = field(default_factory=dict)

    # Temporal tracking
    onset_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

    # Blending state
    is_blended: bool = False
    blend_components: List[str] = field(default_factory=list)

@dataclass
class EmotionalAttentionOutput:
    """Output of emotional attention modulation."""

    # Precision weighting
    precision_modifier: float

    # Scope parameters
    attention_breadth: float
    attention_depth: float

    # Competition modulation
    ignition_threshold: float
    inhibition_strength: float
    competition_iterations: int

    # Priority biases
    percept_priority_modifiers: Dict[str, float]

    # Active inference biases
    action_biases: Dict[str, float]

    # Prediction error processing
    error_amplification: float
    threat_interpretation_bias: float
```

### 8.2 Emotion Registry

```python
# Complete emotion registry with all profiles
EMOTION_REGISTRY: Dict[str, EmotionProfile] = {
    # Primary emotions
    "fear": EmotionProfile(
        name="fear",
        category=EmotionCategory.PRIMARY,
        valence=-0.8, arousal=0.9, dominance=0.2, approach=-0.9,
        onset_rate=0.9, decay_rate=0.3, momentum=0.6, refractory_period=2.0,
        attention_scope=AttentionScope(breadth=0.8, depth=0.3),
        precision_modifier=0.3,
        priority_biases={"threat": 0.5, "escape": 0.4, "social": -0.2},
        action_affinities={"escape": -0.4, "freeze": -0.3, "approach": 0.3}
    ),
    # ... (all other emotions defined similarly)
}
```

---

## 9. Integration Points

### 9.1 With AttentionController

```python
class AttentionController:
    def select_for_broadcast(
        self,
        candidates: List[Percept],
        emotional_state: Optional[EmotionalState] = None,
        prediction_errors: Optional[List[PredictionError]] = None
    ) -> List[Percept]:

        # Get emotional attention modulation
        if emotional_state:
            emo_output = self.emotional_attention.compute_modulation(emotional_state)

            # Apply scope parameters
            self.attention_breadth = emo_output.attention_breadth
            self.attention_depth = emo_output.attention_depth

            # Modify competition parameters
            self.competitive_attention.ignition_threshold = emo_output.ignition_threshold
            self.competitive_attention.inhibition_strength = emo_output.inhibition_strength
            self.competitive_attention.iterations = emo_output.competition_iterations

        # Rest of attention selection...
```

### 9.2 With PrecisionWeighting

```python
class PrecisionWeighting:
    def compute_precision(
        self,
        percept: Percept,
        emotional_state: Dict[str, float],
        prediction_error: Optional[float] = None
    ) -> float:

        # Get emotional precision modulation
        emo_precision = emotional_precision_modulation(
            self.base_precision,
            EmotionalState(**emotional_state)
        )

        # Combine with prediction error effects
        # ... existing logic
```

### 9.3 With CommunicationDecisionLoop

```python
class CommunicationDecisionLoop:
    def _evaluate_with_active_inference(self, ...):

        # Get emotional action biases
        emotional_bias_speak = emotional_action_bias(
            Action(type="speak"), self.current_emotional_state
        )
        emotional_bias_silence = emotional_action_bias(
            Action(type="wait"), self.current_emotional_state
        )

        # Apply to EFE calculations
        speak_efe += emotional_bias_speak
        silence_efe += emotional_bias_silence
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

- Test each emotion profile produces expected VAD values
- Test intensity levels modify effects correctly
- Test temporal dynamics (onset, decay, momentum)
- Test blend rules for compound emotions

### 10.2 Integration Tests

- Test emotion → attention scope changes
- Test emotion → precision weighting changes
- Test emotion → competition parameter changes
- Test emotion → action bias changes

### 10.3 Behavioral Tests

- Fear should increase threat detection accuracy
- Calm should increase deep processing quality
- Curiosity should increase novelty detection
- Anger should increase obstacle focus

---

## 11. Configuration

```python
EMOTIONAL_ATTENTION_CONFIG = {
    # Enable/disable emotional modulation
    "enabled": True,

    # Strength of emotional effects (0-1)
    "modulation_strength": 0.8,

    # Baseline emotional state
    "baseline": {
        "valence": 0.1,
        "arousal": 0.3,
        "dominance": 0.5,
        "approach": 0.2
    },

    # Decay rates
    "baseline_pull_strength": 0.1,
    "emotion_decay_multiplier": 1.0,

    # Blending
    "max_concurrent_emotions": 3,
    "blend_threshold": 0.2,

    # IWMT integration
    "precision_modulation_strength": 0.5,
    "action_bias_strength": 0.4,
}
```

---

**Last Updated**: 2026-01-17
**Status**: Proposed Specification
**Next Steps**: Review, refine, implement
