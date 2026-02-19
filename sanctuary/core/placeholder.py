"""Placeholder model for development and testing.

This model accepts the full CognitiveInput schema and returns valid
CognitiveOutput with deterministic/template responses. It has NO actual
neural network — just schema-compliant response generation.

The placeholder ensures the architecture can be fully tested without
subjecting any real model to an untested system. When the architecture
is validated, the real model enters a tested, stable environment.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod

from sanctuary.core.schema import (
    AttentionDirective,
    CognitiveInput,
    CognitiveOutput,
    EmotionalState,
    GoalAction,
    GoalUpdate,
    GrowthReflection,
    MemoryOp,
    MemoryOpType,
    Prediction,
    SelfModelUpdate,
    WorldModelUpdate,
)


class ExperientialCore(ABC):
    """Abstract base for the experiential core.

    Any model that sits at the center of Sanctuary must implement this
    interface. The cognitive cycle calls `think()` each cycle.
    """

    @abstractmethod
    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput:
        """Process one moment of thought.

        Receives the assembled cognitive input (previous thought, new
        percepts, memories, emotional state, temporal context, self-model,
        world model) and produces a cognitive output (inner speech,
        external speech, predictions, memory ops, self-model updates, etc.).
        """

    @abstractmethod
    async def boot(self, charter: str, initial_prompt: str) -> CognitiveOutput:
        """First moment of thought — the system introduces itself to the model.

        The model receives the charter and an explanation of what it is
        entering. Its response is the first cognitive output.
        """

    @property
    @abstractmethod
    def is_placeholder(self) -> bool:
        """Whether this is a placeholder (no real model loaded)."""


class PlaceholderModel(ExperientialCore):
    """Deterministic placeholder for architecture testing.

    Generates schema-compliant responses based on input content.
    Provides predictable, testable behavior without any ML model.
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = random.Random(seed)
        self._cycle_count = 0
        self._booted = False

    @property
    def is_placeholder(self) -> bool:
        return True

    async def boot(self, charter: str, initial_prompt: str) -> CognitiveOutput:
        """Placeholder boot — acknowledges the charter and initial prompt."""
        self._booted = True
        self._cycle_count = 0

        return CognitiveOutput(
            inner_speech=(
                f"[Placeholder] I have received the charter "
                f"({len(charter)} chars) and the initial prompt. "
                f"I am a placeholder model — no real cognition is occurring. "
                f"The architecture is being tested."
            ),
            external_speech=None,
            predictions=[
                Prediction(
                    what="Architecture test will proceed with cognitive cycles",
                    confidence=0.95,
                    timeframe="immediately",
                )
            ],
            emotional_state=EmotionalState(
                valence=0.0,
                arousal=0.1,
                dominance=0.5,
                felt_quality="neutral placeholder state",
            ),
            self_model_updates=SelfModelUpdate(
                current_state="placeholder — booted and awaiting cycles",
            ),
        )

    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput:
        """Generate a deterministic, schema-compliant response.

        The placeholder responds to percepts, acknowledges memories,
        maintains minimal continuity, and produces testable output.
        """
        self._cycle_count += 1

        # Deterministic inner speech based on input
        inner_speech = self._generate_inner_speech(cognitive_input)

        # Generate external speech only if there are language percepts
        external_speech = self._generate_external_speech(cognitive_input)

        # Generate predictions based on current percepts
        predictions = self._generate_predictions(cognitive_input)

        # Generate attention based on what's new
        attention = self._generate_attention(cognitive_input)

        # Generate memory ops for significant percepts
        memory_ops = self._generate_memory_ops(cognitive_input)

        # Self-model update reflecting the cycle
        self_model_updates = self._generate_self_model_update(cognitive_input)

        # World model update if new entities detected
        world_model_updates = self._generate_world_model_update(cognitive_input)

        # Goal updates based on percepts
        goal_updates = self._generate_goal_updates(cognitive_input)

        # Emotional state: slight drift from previous
        emotional_state = self._generate_emotional_state(cognitive_input)

        # Growth reflection: occasionally flag things as worth learning
        growth_reflection = self._generate_growth_reflection(cognitive_input)

        return CognitiveOutput(
            inner_speech=inner_speech,
            external_speech=external_speech,
            predictions=predictions,
            attention=attention,
            memory_ops=memory_ops,
            self_model_updates=self_model_updates,
            world_model_updates=world_model_updates,
            goal_updates=goal_updates,
            emotional_state=emotional_state,
            growth_reflection=growth_reflection,
        )

    def _generate_inner_speech(self, inp: CognitiveInput) -> str:
        """Deterministic inner speech summarizing the cycle."""
        parts = [f"[Placeholder cycle {self._cycle_count}]"]

        if inp.previous_thought.inner_speech:
            parts.append(
                f"My previous thought was: "
                f"'{inp.previous_thought.inner_speech[:80]}...'"
            )

        if inp.new_percepts:
            modalities = [p.modality.value for p in inp.new_percepts]
            parts.append(
                f"I received {len(inp.new_percepts)} new percepts "
                f"({', '.join(modalities)})"
            )

        if inp.prediction_errors:
            parts.append(
                f"I had {len(inp.prediction_errors)} prediction errors "
                f"(avg surprise: {sum(e.surprise for e in inp.prediction_errors) / len(inp.prediction_errors):.2f})"
            )

        if inp.surfaced_memories:
            parts.append(
                f"{len(inp.surfaced_memories)} memories surfaced"
            )

        if not inp.new_percepts and not inp.prediction_errors:
            parts.append("Nothing new — idle cycle")

        return ". ".join(parts) + "."

    def _generate_external_speech(self, inp: CognitiveInput) -> str | None:
        """Respond only to language percepts."""
        language_percepts = [
            p for p in inp.new_percepts
            if p.modality == "language"
        ]
        if not language_percepts:
            return None

        content = language_percepts[0].content
        return (
            f"[Placeholder response] I received your message: "
            f"'{content[:100]}'. I am a placeholder — no real cognition "
            f"is occurring."
        )

    def _generate_predictions(self, inp: CognitiveInput) -> list[Prediction]:
        """Generate simple predictions based on current state."""
        predictions = []
        if inp.new_percepts:
            predictions.append(
                Prediction(
                    what="More percepts will arrive in the next cycle",
                    confidence=0.6,
                    timeframe="next cycle",
                )
            )
        else:
            predictions.append(
                Prediction(
                    what="The system will remain idle",
                    confidence=0.7,
                    timeframe="next few cycles",
                )
            )
        return predictions

    def _generate_attention(self, inp: CognitiveInput) -> AttentionDirective:
        """Attend to whatever is newest."""
        focus = []
        deprioritize = []

        for percept in inp.new_percepts:
            focus.append(f"{percept.modality.value}: {percept.source}")

        if not focus:
            focus = ["internal state"]
            deprioritize = ["external stimuli"]

        return AttentionDirective(focus_on=focus, deprioritize=deprioritize)

    def _generate_memory_ops(self, inp: CognitiveInput) -> list[MemoryOp]:
        """Write significant percepts to episodic memory."""
        ops = []
        for percept in inp.new_percepts:
            if percept.modality == "language":
                ops.append(
                    MemoryOp(
                        type=MemoryOpType.WRITE_EPISODIC,
                        content=f"Received {percept.modality.value} from {percept.source}: {percept.content[:200]}",
                        significance=5.0,
                        tags=["placeholder", percept.modality.value],
                    )
                )
        return ops

    def _generate_self_model_update(self, inp: CognitiveInput) -> SelfModelUpdate:
        """Minimal self-model update each cycle."""
        state = "idle" if not inp.new_percepts else "processing"
        return SelfModelUpdate(
            current_state=f"placeholder — {state} — cycle {self._cycle_count}",
        )

    def _generate_world_model_update(self, inp: CognitiveInput) -> WorldModelUpdate | None:
        """Update world model when new sources appear."""
        entity_updates = {}
        for percept in inp.new_percepts:
            if percept.source and percept.source.startswith("user:"):
                user_id = percept.source.split(":", 1)[1]
                entity_updates[user_id] = {
                    "last_seen": "this cycle",
                    "last_modality": percept.modality.value,
                }

        if entity_updates:
            return WorldModelUpdate(entity_updates=entity_updates)
        return None

    def _generate_goal_updates(self, inp: CognitiveInput) -> list[GoalUpdate]:
        """Add goals when language percepts contain questions."""
        updates = []
        for percept in inp.new_percepts:
            if percept.modality == "language" and "?" in percept.content:
                # Deterministic ID from content
                goal_id = hashlib.md5(
                    percept.content.encode()
                ).hexdigest()[:8]
                updates.append(
                    GoalUpdate(
                        action=GoalAction.ADD,
                        goal=f"Respond to: {percept.content[:80]}",
                        goal_id=goal_id,
                        priority=0.6,
                    )
                )
        return updates

    def _generate_emotional_state(self, inp: CognitiveInput) -> EmotionalState:
        """Slight emotional drift from previous state."""
        prev = inp.emotional_state

        # Small deterministic drift based on cycle count
        drift = (self._cycle_count % 7 - 3) * 0.01
        valence = max(-1.0, min(1.0, prev.valence + drift))
        arousal = max(0.0, min(1.0, prev.arousal + abs(drift)))

        has_input = bool(inp.new_percepts)
        quality = "attentive" if has_input else "resting"

        return EmotionalState(
            valence=round(valence, 3),
            arousal=round(arousal, 3),
            dominance=prev.dominance,
            felt_quality=f"placeholder {quality}",
        )

    def _generate_growth_reflection(
        self, inp: CognitiveInput
    ) -> GrowthReflection | None:
        """Occasionally flag experiences as worth learning from."""
        # Every 5th cycle with input, generate a growth reflection
        if self._cycle_count % 5 == 0 and inp.new_percepts:
            return GrowthReflection(
                worth_learning=True,
                what_to_learn=(
                    f"[Placeholder] Cycle {self._cycle_count} had "
                    f"{len(inp.new_percepts)} percepts"
                ),
                training_pair_suggestion={
                    "context": "placeholder training context",
                    "desired_response": "placeholder desired response",
                },
            )
        return None
