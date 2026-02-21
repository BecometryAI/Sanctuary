"""Tests for the CognitiveScaffold facade."""

import pytest

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import (
    CognitiveOutput,
    EmotionalOutput,
    GoalProposal,
    MemoryOp,
    Percept,
)
from sanctuary.scaffold.cognitive_scaffold import CognitiveScaffold


@pytest.fixture
def scaffold():
    return CognitiveScaffold()


@pytest.fixture
def authority():
    return AuthorityManager()


class TestCognitiveScaffoldIntegrate:
    """Test the integrate() method of CognitiveScaffold."""

    @pytest.mark.asyncio
    async def test_clean_output_passes_through(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="Thinking clearly.",
            external_speech="Hello there.",
        )
        # Simulate user percept so speech isn't gated
        scaffold.notify_percepts([
            Percept(modality="language", content="Hi", source="user:alice")
        ])
        result = await scaffold.integrate(output, authority)
        assert result.inner_speech == "Thinking clearly."
        assert result.external_speech == "Hello there."

    @pytest.mark.asyncio
    async def test_anomalies_detected(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="",
            emotional_state=EmotionalOutput(valence_shift=0.95),
        )
        await scaffold.integrate(output, authority)
        signals = scaffold.get_signals()
        assert len(signals.anomalies) > 0

    @pytest.mark.asyncio
    async def test_invalid_memory_ops_filtered(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="test",
            memory_ops=[
                MemoryOp(type="write_episodic", content="valid"),
                MemoryOp(type="invalid_type", content="bad"),
            ],
        )
        result = await scaffold.integrate(output, authority)
        assert len(result.memory_ops) == 1

    @pytest.mark.asyncio
    async def test_goal_proposals_integrated(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="setting goals",
            goal_proposals=[
                GoalProposal(action="add", goal="Learn patience", priority=0.7),
            ],
        )
        await scaffold.integrate(output, authority)
        status = scaffold.goals.get_status()
        assert status["active_count"] == 1

    @pytest.mark.asyncio
    async def test_speech_gated_without_user_input(self, scaffold, authority):
        """External speech should be gated when no user percept is present."""
        output = CognitiveOutput(
            inner_speech="thinking",
            external_speech="Unprompted speech",
        )
        # No notify_percepts called — no user input
        result = await scaffold.integrate(output, authority)
        # With default settings, idle drive (0.2) < threshold (0.4) → gated
        assert result.external_speech is None

    @pytest.mark.asyncio
    async def test_speech_passes_with_user_input(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="responding",
            external_speech="Here is my response.",
        )
        scaffold.notify_percepts([
            Percept(modality="language", content="Tell me something", source="user:bob")
        ])
        result = await scaffold.integrate(output, authority)
        assert result.external_speech == "Here is my response."

    @pytest.mark.asyncio
    async def test_affect_merged(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="feeling happy",
            emotional_state=EmotionalOutput(
                felt_quality="warm",
                valence_shift=0.3,
            ),
        )
        initial_v = scaffold.affect.valence
        await scaffold.integrate(output, authority)
        # At LLM_GUIDES (default for emotional_state), valence should increase
        assert scaffold.affect.valence > initial_v

    @pytest.mark.asyncio
    async def test_decay_applied_each_cycle(self, scaffold, authority):
        """Affect should decay toward baseline after integration."""
        scaffold.affect.valence = 0.8  # Push high
        output = CognitiveOutput(inner_speech="steady")
        await scaffold.integrate(output, authority)
        # Should have decayed slightly
        assert scaffold.affect.valence < 0.8


class TestCognitiveScaffoldSignals:
    """Test the get_signals() method."""

    @pytest.mark.asyncio
    async def test_signals_include_goal_status(self, scaffold, authority):
        output = CognitiveOutput(
            inner_speech="working",
            goal_proposals=[
                GoalProposal(action="add", goal="Test goal", priority=0.5),
            ],
        )
        await scaffold.integrate(output, authority)
        signals = scaffold.get_signals()
        assert signals.goal_status["active_count"] == 1

    @pytest.mark.asyncio
    async def test_signals_include_communication_state(self, scaffold, authority):
        scaffold.notify_percepts([
            Percept(modality="language", content="Hello", source="user:x")
        ])
        output = CognitiveOutput(
            inner_speech="responding",
            external_speech="Hi!",
        )
        await scaffold.integrate(output, authority)
        signals = scaffold.get_signals()
        assert signals.communication_drives.strongest != ""

    @pytest.mark.asyncio
    async def test_signals_include_anomalies(self, scaffold, authority):
        output = CognitiveOutput(inner_speech="")
        await scaffold.integrate(output, authority)
        signals = scaffold.get_signals()
        assert len(signals.anomalies) > 0

    def test_signals_default_empty(self, scaffold):
        """Before any cycle, signals should be clean."""
        signals = scaffold.get_signals()
        assert signals.goal_status["active_count"] == 0
        assert len(signals.anomalies) == 0


class TestCognitiveScaffoldBroadcast:
    """Test the broadcast mechanism."""

    @pytest.mark.asyncio
    async def test_broadcast_calls_handlers(self, scaffold):
        received = []

        async def handler(output):
            received.append(output)

        scaffold.on_broadcast(handler)
        output = CognitiveOutput(inner_speech="broadcasting")
        await scaffold.broadcast(output)
        assert len(received) == 1
        assert received[0].inner_speech == "broadcasting"

    @pytest.mark.asyncio
    async def test_broadcast_handles_errors(self, scaffold):
        """A failing handler shouldn't crash the broadcast."""

        async def bad_handler(output):
            raise RuntimeError("oops")

        scaffold.on_broadcast(bad_handler)
        output = CognitiveOutput(inner_speech="safe")
        # Should not raise
        await scaffold.broadcast(output)


class TestCognitiveScaffoldPercepts:
    """Test percept notification."""

    def test_notify_percepts_updates_affect(self, scaffold):
        initial_v = scaffold.affect.valence
        scaffold.notify_percepts([
            Percept(modality="language", content="wonderful great news")
        ])
        assert scaffold.affect.valence > initial_v

    def test_notify_percepts_detects_user_input(self, scaffold):
        scaffold.notify_percepts([
            Percept(modality="language", content="hello", source="user:alice")
        ])
        assert scaffold._has_user_percept is True

    def test_notify_percepts_no_user_input(self, scaffold):
        scaffold.notify_percepts([
            Percept(modality="temporal", content="5 seconds elapsed")
        ])
        assert scaffold._has_user_percept is False

    def test_computed_vad_reflects_state(self, scaffold):
        scaffold.affect.valence = 0.5
        scaffold.affect.arousal = 0.3
        vad = scaffold.get_computed_vad()
        assert vad.valence == 0.5
        assert vad.arousal == 0.3
