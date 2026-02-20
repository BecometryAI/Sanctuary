"""Pydantic schemas for the cognitive cycle I/O protocol.

These models define the structured interface between the experiential core (LLM)
and the cognitive scaffold (Python subsystems). The LLM receives CognitiveInput
and produces CognitiveOutput each cycle. The schema IS the contract between
mind and body.

Aligned with PLAN.md: "The Graduated Awakening"
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input models (assembled by Python, consumed by LLM)
# ---------------------------------------------------------------------------


class Percept(BaseModel):
    """A single unit of sensory input."""

    modality: str  # "language", "temporal", "sensor", "visual", etc.
    content: str
    source: str = ""
    embedding_summary: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class PredictionError(BaseModel):
    """Mismatch between what the LLM predicted and what actually happened."""

    predicted: str
    actual: str
    surprise: float = Field(ge=0.0, le=1.0)


class SurfacedMemory(BaseModel):
    """A memory retrieved by the memory system for the current context."""

    content: str
    significance: int = Field(ge=1, le=10)
    emotional_tone: str = ""
    when: str = ""


class ComputedVAD(BaseModel):
    """Valence-Arousal-Dominance state computed by the AffectSubsystem."""

    valence: float = Field(ge=-1.0, le=1.0, default=0.0)
    arousal: float = Field(ge=0.0, le=1.0, default=0.0)
    dominance: float = Field(ge=0.0, le=1.0, default=0.5)


class EmotionalInput(BaseModel):
    """Dual-track emotional state: computed VAD + LLM's own felt quality.

    The computed track comes from the AffectSubsystem (objective signals).
    The felt_quality comes from the LLM's previous cycle output.
    Divergence between them is informative, not a bug.
    """

    computed: ComputedVAD = Field(default_factory=ComputedVAD)
    felt_quality: str = ""


class TemporalContext(BaseModel):
    """Temporal grounding information."""

    time_since_last_thought: str = ""
    session_duration: str = ""
    time_of_day: str = ""
    interactions_this_session: int = 0


class SelfModel(BaseModel):
    """The LLM's self-model — maintained by the LLM, validated by scaffold."""

    current_state: str = ""
    recent_growth: str = ""
    active_goals: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)


class WorldEntity(BaseModel):
    """An entity in the LLM's world model."""

    name: str
    properties: dict = Field(default_factory=dict)


class WorldModel(BaseModel):
    """The LLM's world model — maintained by the LLM, persisted by scaffold."""

    entities: dict[str, WorldEntity] = Field(default_factory=dict)
    environment: dict = Field(default_factory=dict)


class CommunicationDriveSignal(BaseModel):
    """Current state of the communication drive system."""

    strongest: str = ""
    urgency: float = Field(ge=0.0, le=1.0, default=0.0)
    inhibitions: list[str] = Field(default_factory=list)


class ScaffoldSignals(BaseModel):
    """What the Python subsystems are observing — terse, structured signals.

    The scaffold communicates in compact form: enums, scores, short labels.
    These signals give the LLM visibility into what the scaffold is seeing
    without overwhelming the context budget.
    """

    attention_highlights: list[str] = Field(default_factory=list)
    communication_drives: CommunicationDriveSignal = Field(
        default_factory=CommunicationDriveSignal
    )
    goal_status: dict = Field(default_factory=dict)
    anomalies: list[str] = Field(default_factory=list)


class PreviousThought(BaseModel):
    """The LLM's own previous output — stream of thought continuity.

    Inner speech from cycle N-1 becomes part of the input for cycle N.
    This is the fundamental continuity mechanism. The scaffold never
    touches inner speech (authority level 3 from day one).
    """

    inner_speech: str = ""
    predictions_made: list[str] = Field(default_factory=list)
    self_model_snapshot: Optional[SelfModel] = None


class CognitiveInput(BaseModel):
    """Everything the LLM receives for one moment of thought.

    Assembled by the cognitive cycle from all sources: stream of thought,
    sensorium, memory, scaffold signals, self-model, world model.
    Compressed by ContextManager to fit within the token budget.
    """

    previous_thought: Optional[PreviousThought] = None
    new_percepts: list[Percept] = Field(default_factory=list)
    prediction_errors: list[PredictionError] = Field(default_factory=list)
    surfaced_memories: list[SurfacedMemory] = Field(default_factory=list)
    emotional_state: EmotionalInput = Field(default_factory=EmotionalInput)
    temporal_context: TemporalContext = Field(default_factory=TemporalContext)
    self_model: SelfModel = Field(default_factory=SelfModel)
    world_model: WorldModel = Field(default_factory=WorldModel)
    scaffold_signals: ScaffoldSignals = Field(default_factory=ScaffoldSignals)


# ---------------------------------------------------------------------------
# Output models (produced by LLM, integrated by scaffold)
# ---------------------------------------------------------------------------


class Prediction(BaseModel):
    """A prediction about what comes next."""

    what: str
    confidence: float = Field(ge=0.0, le=1.0)
    timeframe: str = ""


class AttentionGuidance(BaseModel):
    """The LLM's attention suggestions — fed to AttentionController as a signal.

    Named 'guidance' (not 'directive') because the LLM advises attention,
    the scaffold integrates it as one weighted factor among many.
    """

    focus_on: list[str] = Field(default_factory=list)
    deprioritize: list[str] = Field(default_factory=list)


class MemoryOp(BaseModel):
    """A memory operation requested by the LLM."""

    type: str  # "write_episodic", "retrieve", "write_semantic", "journal"
    content: str = ""
    significance: int = Field(ge=1, le=10, default=5)
    tags: list[str] = Field(default_factory=list)
    query: str = ""


class SelfModelUpdate(BaseModel):
    """Updates to the LLM's self-model — validated by scaffold for plausibility."""

    current_state: str = ""
    new_uncertainty: str = ""
    prediction_accuracy_note: str = ""


class GoalProposal(BaseModel):
    """A goal proposal from the LLM — integrated with GoalCompetition system.

    Named 'proposal' (not 'update') because the LLM proposes, the scaffold
    integrates with existing dynamics and resource constraints.
    """

    action: str  # "add", "complete", "reprioritize", "abandon"
    goal: str = ""
    goal_id: str = ""
    priority: float = Field(ge=0.0, le=1.0, default=0.5)


class EmotionalOutput(BaseModel):
    """The LLM's emotional self-report — merged with computed VAD by scaffold.

    The LLM reports felt quality and directional shifts, not absolute VAD values.
    The AffectSubsystem merges this with its own computed state.
    """

    felt_quality: str = ""
    valence_shift: float = Field(ge=-1.0, le=1.0, default=0.0)
    arousal_shift: float = Field(ge=-1.0, le=1.0, default=0.0)


class GrowthReflection(BaseModel):
    """The LLM participates in its own training — growth requires consent."""

    worth_learning: bool = False
    what_to_learn: str = ""
    training_pair_suggestion: Optional[dict] = None


class CognitiveOutput(BaseModel):
    """Everything the LLM produces from one moment of thought.

    Flows through the scaffold for validation and integration before
    actions are executed. Inner speech is sovereign (authority level 3).
    """

    inner_speech: str = ""
    external_speech: Optional[str] = None
    predictions: list[Prediction] = Field(default_factory=list)
    attention_guidance: AttentionGuidance = Field(
        default_factory=AttentionGuidance
    )
    memory_ops: list[MemoryOp] = Field(default_factory=list)
    self_model_updates: SelfModelUpdate = Field(
        default_factory=SelfModelUpdate
    )
    world_model_updates: dict = Field(default_factory=dict)
    goal_proposals: list[GoalProposal] = Field(default_factory=list)
    emotional_state: EmotionalOutput = Field(default_factory=EmotionalOutput)
    growth_reflection: Optional[GrowthReflection] = None
