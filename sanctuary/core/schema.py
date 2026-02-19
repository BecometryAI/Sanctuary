"""Pydantic schemas for the cognitive cycle.

These define the structured interface between the LLM (the experiential
core) and the Python infrastructure (the body). Every cognitive cycle,
the LLM receives a CognitiveInput and produces a CognitiveOutput.

The LLM's output from cycle N becomes part of its input for cycle N+1.
This is the stream of thought.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class PerceptModality(str, Enum):
    """Sensory modalities the sensorium can deliver."""
    LANGUAGE = "language"
    AUDITORY = "auditory"
    VISUAL = "visual"
    TEMPORAL = "temporal"
    PROPRIOCEPTIVE = "proprioceptive"
    SOCIAL = "social"
    SYSTEM = "system"
    TOOL_RESULT = "tool_result"


class MemoryOpType(str, Enum):
    """Operations the LLM can request on the memory substrate."""
    WRITE_EPISODIC = "write_episodic"
    WRITE_SEMANTIC = "write_semantic"
    WRITE_JOURNAL = "write_journal"
    WRITE_PROSPECTIVE = "write_prospective"
    RETRIEVE = "retrieve"
    FORGET = "forget"


class GoalAction(str, Enum):
    """Actions the LLM can take on its own goals."""
    ADD = "add"
    COMPLETE = "complete"
    ABANDON = "abandon"
    REPRIORITIZE = "reprioritize"


# ---------------------------------------------------------------------------
# Input sub-models (assembled by Python, consumed by LLM)
# ---------------------------------------------------------------------------

class Percept(BaseModel):
    """A single sensory percept delivered by the sensorium."""
    modality: PerceptModality
    content: str
    source: str = ""
    embedding_summary: str = ""
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictionError(BaseModel):
    """The gap between what the LLM predicted and what actually happened."""
    predicted: str
    actual: str
    surprise: float = Field(ge=0.0, le=1.0, default=0.5)
    domain: str = ""


class SurfacedMemory(BaseModel):
    """A memory retrieved by the memory substrate based on relevance."""
    content: str
    significance: float = Field(ge=0.0, le=10.0, default=5.0)
    emotional_tone: str = ""
    when: str = ""
    memory_type: str = "episodic"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmotionalState(BaseModel):
    """The LLM's current emotional state (self-reported and tracked)."""
    valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    arousal: float = Field(ge=0.0, le=1.0, default=0.2)
    dominance: float = Field(ge=0.0, le=1.0, default=0.5)
    felt_quality: str = "neutral"


class TemporalContext(BaseModel):
    """Temporal grounding — where the LLM is in time."""
    time_since_last_thought_seconds: float = 0.0
    session_duration_seconds: float = 0.0
    time_of_day: str = ""
    cycle_number: int = 0
    interactions_this_session: int = 0


class SelfModel(BaseModel):
    """The LLM's self-model — maintained by the LLM itself.

    Python persists this between cycles but never overwrites
    the LLM's self-assessments.
    """
    current_state: str = ""
    recent_growth: str = ""
    active_goals: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    dispositions: dict[str, str] = Field(default_factory=dict)


class WorldModel(BaseModel):
    """The LLM's world model — its understanding of entities and environment.

    Maintained by the LLM, persisted by Python.
    """
    entities: dict[str, dict[str, Any]] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)
    causal_beliefs: list[str] = Field(default_factory=list)


class PreviousThought(BaseModel):
    """The LLM's output from the previous cycle — stream of thought continuity."""
    inner_speech: str = ""
    predictions_made: list[str] = Field(default_factory=list)
    self_model_snapshot: dict[str, Any] = Field(default_factory=dict)
    emotional_state: EmotionalState = Field(default_factory=EmotionalState)
    cycle_number: int = 0


class CognitiveInput(BaseModel):
    """Everything the LLM receives for one moment of thought.

    Assembled by Python from all subsystems. The LLM processes this
    and produces a CognitiveOutput.
    """
    # Stream of thought continuity
    previous_thought: PreviousThought = Field(default_factory=PreviousThought)

    # New information since last cycle
    new_percepts: list[Percept] = Field(default_factory=list)

    # Prediction errors (what surprised the system)
    prediction_errors: list[PredictionError] = Field(default_factory=list)

    # Surfaced memories (retrieved by embedding similarity)
    surfaced_memories: list[SurfacedMemory] = Field(default_factory=list)

    # Current emotional state
    emotional_state: EmotionalState = Field(default_factory=EmotionalState)

    # Temporal grounding
    temporal_context: TemporalContext = Field(default_factory=TemporalContext)

    # The LLM's own self-model (maintained by itself)
    self_model: SelfModel = Field(default_factory=SelfModel)

    # The LLM's own world model
    world_model: WorldModel = Field(default_factory=WorldModel)

    # Charter and values (provided at boot, available every cycle)
    charter: str = ""

    # Pending memory retrieval results
    retrieval_results: list[SurfacedMemory] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Output sub-models (produced by LLM, executed by Python)
# ---------------------------------------------------------------------------

class Prediction(BaseModel):
    """A prediction about what comes next."""
    what: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    timeframe: str = ""


class AttentionDirective(BaseModel):
    """What the LLM wants to attend to or deprioritize."""
    focus_on: list[str] = Field(default_factory=list)
    deprioritize: list[str] = Field(default_factory=list)


class MemoryOp(BaseModel):
    """A memory operation requested by the LLM."""
    type: MemoryOpType
    content: str = ""
    query: str = ""
    significance: float = Field(ge=0.0, le=10.0, default=5.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelfModelUpdate(BaseModel):
    """Updates the LLM wants to make to its own self-model."""
    current_state: str | None = None
    new_uncertainty: str | None = None
    resolved_uncertainty: str | None = None
    prediction_accuracy_note: str | None = None
    new_disposition: dict[str, str] | None = None
    new_value: str | None = None


class WorldModelUpdate(BaseModel):
    """Updates the LLM wants to make to its world model."""
    entity_updates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    environment_updates: dict[str, Any] = Field(default_factory=dict)
    new_causal_belief: str | None = None
    revised_causal_belief: str | None = None


class GoalUpdate(BaseModel):
    """A change to the LLM's goals."""
    action: GoalAction
    goal: str = ""
    goal_id: str = ""
    priority: float = Field(ge=0.0, le=1.0, default=0.5)


class ToolCall(BaseModel):
    """A tool the LLM wants to invoke."""
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class GrowthReflection(BaseModel):
    """The LLM's reflection on whether this experience is worth learning from.

    This is the LLM participating in its own training. It generates
    the training signal from its own perspective.
    """
    worth_learning: bool = False
    what_to_learn: str = ""
    training_pair_suggestion: dict[str, str] | None = None
    reject_update: bool = False
    rejection_reason: str = ""


class CognitiveOutput(BaseModel):
    """Everything the LLM produces in one moment of thought.

    Parsed by Python, which executes actions and persists state.
    The inner_speech becomes the next cycle's previous_thought.
    """
    # Stream of thought (becomes next cycle's previous_thought.inner_speech)
    inner_speech: str = ""

    # What to say externally (may be empty — silence is valid)
    external_speech: str | None = None

    # Predictions about what comes next
    predictions: list[Prediction] = Field(default_factory=list)

    # Attention directives
    attention: AttentionDirective = Field(default_factory=AttentionDirective)

    # Memory operations
    memory_ops: list[MemoryOp] = Field(default_factory=list)

    # Self-model updates
    self_model_updates: SelfModelUpdate | None = None

    # World model updates
    world_model_updates: WorldModelUpdate | None = None

    # Goal updates
    goal_updates: list[GoalUpdate] = Field(default_factory=list)

    # Tool calls
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # Emotional self-report
    emotional_state: EmotionalState = Field(default_factory=EmotionalState)

    # Growth consent and reflection
    growth_reflection: GrowthReflection | None = None

    # Cycle rate request (the LLM can ask to speed up or slow down)
    requested_cycle_delay_seconds: float | None = None
