"""
End-to-end pipeline integration test: text in → cognitive processing → text out.

Verifies the complete path through the cognitive architecture using mock LLMs,
suitable for CI.  The test boots a CognitiveCore, injects text input, waits
for cognitive cycles to produce an output, and checks that the response
propagates through perception → attention → affect → action → language output.
"""

import asyncio
import pytest

from mind.cognitive_core.core import CognitiveCore
from mind.cognitive_core.workspace import GlobalWorkspace, GoalType
from mind.cognitive_core.conversation import ConversationManager, ConversationTurn
from mind.client import SanctuaryAPI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_config(**overrides):
    """Return a CognitiveCore config with mock LLMs and no checkpointing."""
    cfg = {
        "cycle_rate_hz": 10,
        "checkpointing": {"enabled": False},
        "input_llm": {"use_real_model": False},
        "output_llm": {"use_real_model": False},
        "perception": {"mock_mode": True},
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.asyncio
class TestFullPipelineE2E:
    """Full text-in → cognitive processing → text-out pipeline."""

    async def test_inject_text_produces_output(self):
        """Inject raw text and verify output appears on the output queue."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=_mock_config())

        try:
            await core.start()

            # Let at least one idle cycle execute
            await asyncio.sleep(0.3)

            # Inject text input
            await core.process_language_input("Hello, Sanctuary!")

            # Wait for output on the output queue (may be SPEAK or SPEAK_AUTONOMOUS)
            response = await core.get_response(timeout=8.0)

            # Verify we got a speech response
            assert response is not None, "Expected a SPEAK output from the pipeline"
            assert response.get("type") in ("SPEAK", "SPEAK_AUTONOMOUS"), (
                f"Expected SPEAK or SPEAK_AUTONOMOUS, got {response.get('type')}"
            )
            assert len(response.get("text", "")) > 0

        finally:
            await core.stop()

    async def test_cognitive_cycles_advance_during_processing(self):
        """Verify cognitive cycles advance while processing text input."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=_mock_config())

        try:
            await core.start()
            await asyncio.sleep(0.3)

            initial_cycles = core.get_metrics()["total_cycles"]

            # Inject input
            await core.process_language_input("Tell me something interesting.")

            # Wait for cycles to advance
            await asyncio.sleep(1.5)

            later_cycles = core.get_metrics()["total_cycles"]
            assert later_cycles > initial_cycles, "Cycles should advance during processing"

        finally:
            await core.stop()

    async def test_workspace_receives_input_percept(self):
        """Verify the input creates a percept and goals in the workspace."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=_mock_config())

        try:
            await core.start()
            await asyncio.sleep(0.3)

            await core.process_language_input("What is consciousness?")

            # Allow one cycle for the percept to land
            await asyncio.sleep(0.5)

            snapshot = workspace.broadcast()

            # Should have at least a RESPOND_TO_USER goal
            goal_types = [g.type for g in snapshot.goals]
            assert GoalType.RESPOND_TO_USER in goal_types, (
                f"Expected RESPOND_TO_USER goal, got {goal_types}"
            )

        finally:
            await core.stop()

    async def test_emotional_state_present_after_processing(self):
        """Verify emotional state is populated after processing input."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=_mock_config())

        try:
            await core.start()
            await asyncio.sleep(0.3)

            await core.process_language_input("I feel curious today.")
            await asyncio.sleep(1.0)

            snapshot = workspace.broadcast()
            emotions = snapshot.emotions
            assert isinstance(emotions, dict)
            # Affect subsystem should have set at least valence/arousal
            assert "valence" in emotions or "arousal" in emotions or len(emotions) > 0

        finally:
            await core.stop()


@pytest.mark.integration
@pytest.mark.asyncio
class TestConversationPipelineE2E:
    """Test the SanctuaryAPI → ConversationManager → CognitiveCore pipeline."""

    async def test_api_chat_returns_conversation_turn(self):
        """Test the high-level chat() API returns a well-formed turn."""
        config = {
            "cognitive_core": _mock_config(),
            "conversation": {"response_timeout": 10.0},
        }
        api = SanctuaryAPI(config)

        try:
            await api.start()
            turn = await api.chat("Hello, how are you?")

            assert isinstance(turn, ConversationTurn)
            assert turn.user_input == "Hello, how are you?"
            assert len(turn.system_response) > 0
            assert turn.response_time > 0
            assert isinstance(turn.emotional_state, dict)

        finally:
            await api.stop()

    async def test_multi_turn_updates_history(self):
        """Multiple chat turns should accumulate in history."""
        config = {
            "cognitive_core": _mock_config(),
            "conversation": {"response_timeout": 10.0},
        }
        api = SanctuaryAPI(config)

        try:
            await api.start()

            await api.chat("First message.")
            await api.chat("Second message.")

            history = api.get_conversation_history(10)
            assert len(history) >= 2
            assert history[0].user_input == "First message."
            assert history[1].user_input == "Second message."

        finally:
            await api.stop()

    async def test_metrics_track_turns(self):
        """Metrics should reflect completed turns."""
        config = {
            "cognitive_core": _mock_config(),
            "conversation": {"response_timeout": 10.0},
        }
        api = SanctuaryAPI(config)

        try:
            await api.start()
            await api.chat("Metrics test.")

            metrics = api.get_metrics()
            assert metrics["conversation"]["total_turns"] >= 1
            assert metrics["conversation"]["avg_response_time"] > 0
            assert metrics["cognitive_core"]["total_cycles"] > 0

        finally:
            await api.stop()


@pytest.mark.integration
@pytest.mark.asyncio
class TestHealthReportE2E:
    """Verify subsystem health reporting works end-to-end."""

    async def test_health_report_after_cycles(self):
        """Health report should show subsystem statuses after running cycles."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace, config=_mock_config())

        try:
            await core.start()
            await asyncio.sleep(1.0)

            report = core.get_health_report()
            assert "subsystems" in report
            # At least some subsystems should have executed
            assert len(report["subsystems"]) > 0
            # Should report overall health status
            assert "overall_status" in report or "status_summary" in report

        finally:
            await core.stop()
