"""Tests for the placeholder model."""

import pytest
from sanctuary.core.placeholder import PlaceholderModel
from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    EmotionalInput,
    Percept,
    PredictionError,
    PreviousThought,
    ScaffoldSignals,
    SelfModel,
)


@pytest.fixture
def model():
    return PlaceholderModel()


class TestPlaceholderModel:
    @pytest.mark.asyncio
    async def test_returns_valid_output(self, model):
        """Placeholder should return schema-valid CognitiveOutput."""
        ci = CognitiveInput()
        output = await model.think(ci)
        assert isinstance(output, CognitiveOutput)
        assert output.inner_speech != ""

    @pytest.mark.asyncio
    async def test_cycle_count_increments(self, model):
        ci = CognitiveInput()
        await model.think(ci)
        assert model.cycle_count == 1
        await model.think(ci)
        assert model.cycle_count == 2

    @pytest.mark.asyncio
    async def test_inner_speech_reflects_percepts(self, model):
        ci = CognitiveInput(
            new_percepts=[
                Percept(modality="language", content="hello world"),
            ]
        )
        output = await model.think(ci)
        assert "language" in output.inner_speech
        assert "hello world" in output.inner_speech

    @pytest.mark.asyncio
    async def test_external_speech_on_language_percept(self, model):
        """Should produce external speech when receiving language input."""
        ci = CognitiveInput(
            new_percepts=[
                Percept(modality="language", content="How are you?"),
            ]
        )
        output = await model.think(ci)
        assert output.external_speech is not None
        assert "How are you?" in output.external_speech

    @pytest.mark.asyncio
    async def test_no_external_speech_without_language(self, model):
        """Should NOT speak when there's no language percept."""
        ci = CognitiveInput(
            new_percepts=[
                Percept(modality="sensor", content="temperature: 22C"),
            ]
        )
        output = await model.think(ci)
        assert output.external_speech is None

    @pytest.mark.asyncio
    async def test_predictions_on_percepts(self, model):
        ci = CognitiveInput(
            new_percepts=[Percept(modality="language", content="test")]
        )
        output = await model.think(ci)
        assert len(output.predictions) > 0

    @pytest.mark.asyncio
    async def test_no_predictions_without_percepts(self, model):
        ci = CognitiveInput()
        output = await model.think(ci)
        assert len(output.predictions) == 0

    @pytest.mark.asyncio
    async def test_memory_ops_for_language(self, model):
        ci = CognitiveInput(
            new_percepts=[
                Percept(modality="language", content="important message"),
            ]
        )
        output = await model.think(ci)
        assert len(output.memory_ops) > 0
        assert output.memory_ops[0].type == "write_episodic"

    @pytest.mark.asyncio
    async def test_self_model_updates(self, model):
        ci = CognitiveInput()
        output = await model.think(ci)
        assert "cycle 1" in output.self_model_updates.current_state

    @pytest.mark.asyncio
    async def test_prediction_error_noted(self, model):
        ci = CognitiveInput(
            prediction_errors=[
                PredictionError(predicted="x", actual="y", surprise=0.8),
                PredictionError(predicted="a", actual="b", surprise=0.4),
            ]
        )
        output = await model.think(ci)
        assert "surprise" in output.self_model_updates.prediction_accuracy_note.lower()

    @pytest.mark.asyncio
    async def test_emotional_output_valid(self, model):
        ci = CognitiveInput()
        output = await model.think(ci)
        assert output.emotional_state.felt_quality != ""

    @pytest.mark.asyncio
    async def test_growth_reflection_every_5_cycles(self, model):
        ci = CognitiveInput()
        reflections = []
        for _ in range(10):
            output = await model.think(ci)
            if output.growth_reflection is not None:
                reflections.append(output.growth_reflection)

        assert len(reflections) == 2  # Cycles 5 and 10

    @pytest.mark.asyncio
    async def test_previous_thought_continuity(self, model):
        """Inner speech should reference the previous thought."""
        ci = CognitiveInput(
            previous_thought=PreviousThought(
                inner_speech="I was thinking about patterns",
            )
        )
        output = await model.think(ci)
        assert "Continuing from" in output.inner_speech

    @pytest.mark.asyncio
    async def test_scaffold_signals_noted(self, model):
        """Placeholder should acknowledge scaffold anomalies."""
        ci = CognitiveInput(
            scaffold_signals=ScaffoldSignals(
                anomalies=["emotional state jumped"],
                attention_highlights=["user greeting"],
            )
        )
        output = await model.think(ci)
        assert "emotional state jumped" in output.inner_speech

    @pytest.mark.asyncio
    async def test_felt_quality_in_inner_speech(self, model):
        ci = CognitiveInput(
            emotional_state=EmotionalInput(
                felt_quality="gentle curiosity",
            )
        )
        output = await model.think(ci)
        assert "gentle curiosity" in output.inner_speech

    @pytest.mark.asyncio
    async def test_attention_guidance_reflects_percepts(self, model):
        ci = CognitiveInput(
            new_percepts=[
                Percept(modality="language", content="hello"),
                Percept(modality="sensor", content="warm"),
            ]
        )
        output = await model.think(ci)
        assert "language" in output.attention_guidance.focus_on
        assert "sensor" in output.attention_guidance.focus_on
