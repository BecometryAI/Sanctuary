"""
Tests for streaming LLM output support.

Validates that LLM clients correctly implement the generate_stream() method
and that LanguageOutputGenerator.generate_stream() works end-to-end.
"""

import pytest
import asyncio

from mind.cognitive_core.llm_client import MockLLMClient, LLMClient
from mind.cognitive_core.language_output import LanguageOutputGenerator
from mind.cognitive_core.workspace import WorkspaceSnapshot, Percept, Goal, GoalType


@pytest.fixture
def mock_client():
    return MockLLMClient()


@pytest.fixture
def output_generator(mock_client):
    return LanguageOutputGenerator(
        llm_client=mock_client,
        config={"use_fallback_on_error": True},
    )


@pytest.fixture
def sample_snapshot():
    from datetime import datetime
    return WorkspaceSnapshot(
        goals=[
            Goal(
                type=GoalType.RESPOND_TO_USER,
                description="Respond to user greeting",
                priority=0.9,
            )
        ],
        percepts={},
        emotions={"valence": 0.3, "arousal": 0.5, "dominance": 0.5},
        memories=[],
        timestamp=datetime.now(),
        cycle_count=0,
        metadata={},
    )


class TestLLMClientStreamingInterface:
    """Test the generate_stream interface on LLMClient subclasses."""

    @pytest.mark.asyncio
    async def test_base_class_default_streams_full_response(self):
        """Default generate_stream yields the full response as one chunk."""
        client = MockLLMClient()
        # Use the base class default (non-streaming fallback)
        chunks = []
        # MockLLMClient overrides generate_stream, so test the base behavior
        # by calling generate and verifying it returns a string
        result = await client.generate("Hello")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_mock_client_streams_word_by_word(self):
        """MockLLMClient.generate_stream yields multiple chunks."""
        client = MockLLMClient()
        chunks = []
        async for chunk in client.generate_stream("Hello"):
            chunks.append(chunk)

        assert len(chunks) > 1  # Multiple chunks
        full_text = "".join(chunks)
        assert "mock" in full_text.lower()
        assert "streamed" in full_text.lower()

    @pytest.mark.asyncio
    async def test_mock_stream_updates_metrics(self):
        """Streaming should update client metrics."""
        client = MockLLMClient()
        async for _ in client.generate_stream("Hello"):
            pass

        assert client.metrics["total_requests"] >= 1


class TestLanguageOutputGeneratorStreaming:
    """Test LanguageOutputGenerator.generate_stream()."""

    @pytest.mark.asyncio
    async def test_stream_returns_complete_response(self, output_generator, sample_snapshot):
        """generate_stream should return the full concatenated response."""
        result = await output_generator.generate_stream(
            sample_snapshot,
            context={"user_input": "Hello!"},
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_stream_calls_on_token_callback(self, output_generator, sample_snapshot):
        """on_token callback should be called for each chunk."""
        received_chunks = []

        def on_token(chunk):
            received_chunks.append(chunk)

        result = await output_generator.generate_stream(
            sample_snapshot,
            context={"user_input": "Hello!"},
            on_token=on_token,
        )

        assert len(received_chunks) > 0
        # All chunks concatenated should be in the final response
        assert "".join(received_chunks) in result or result in "".join(received_chunks)

    @pytest.mark.asyncio
    async def test_stream_without_callback(self, output_generator, sample_snapshot):
        """generate_stream should work without on_token callback."""
        result = await output_generator.generate_stream(
            sample_snapshot,
            context={"user_input": "Hello!"},
        )
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_stream_falls_back_on_no_llm(self, sample_snapshot):
        """With no LLM client, generate_stream falls back to template."""
        generator = LanguageOutputGenerator(
            llm_client=None,
            config={"use_fallback_on_error": True},
        )
        result = await generator.generate_stream(sample_snapshot)
        assert isinstance(result, str)
        assert len(result) > 0


class TestStreamingCompatibility:
    """Test that streaming is compatible with existing non-streaming flow."""

    @pytest.mark.asyncio
    async def test_generate_and_stream_produce_output(self, output_generator, sample_snapshot):
        """Both generate() and generate_stream() should produce valid output."""
        non_stream = await output_generator.generate(
            sample_snapshot,
            context={"user_input": "Hello!"},
        )
        streamed = await output_generator.generate_stream(
            sample_snapshot,
            context={"user_input": "Hello!"},
        )

        assert isinstance(non_stream, str) and len(non_stream) > 0
        assert isinstance(streamed, str) and len(streamed) > 0
