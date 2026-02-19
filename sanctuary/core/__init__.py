"""Sanctuary Core — The Experiential Core.

This package implements the cognitive cycle where the LLM is the
experiential center. Everything else (sensorium, motor, memory, growth)
serves the LLM — not the other way around.

The LLM runs continuously in a cognitive loop. Its output from cycle N
becomes part of its input for cycle N+1. This is the stream of thought.
"""

from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    PreviousThought,
    Percept,
    PredictionError,
    SurfacedMemory,
    EmotionalState,
    TemporalContext,
    SelfModel,
    WorldModel,
    Prediction,
    AttentionDirective,
    MemoryOp,
    SelfModelUpdate,
    WorldModelUpdate,
    GoalUpdate,
    GrowthReflection,
)

__all__ = [
    "CognitiveInput",
    "CognitiveOutput",
    "PreviousThought",
    "Percept",
    "PredictionError",
    "SurfacedMemory",
    "EmotionalState",
    "TemporalContext",
    "SelfModel",
    "WorldModel",
    "Prediction",
    "AttentionDirective",
    "MemoryOp",
    "SelfModelUpdate",
    "WorldModelUpdate",
    "GoalUpdate",
    "GrowthReflection",
]
