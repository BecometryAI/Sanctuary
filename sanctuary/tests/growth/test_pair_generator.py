"""Tests for TrainingPairGenerator -- pair generation from reflections."""

from __future__ import annotations

import pytest

from sanctuary.growth.harvester import HarvestedReflection
from sanctuary.growth.pair_generator import TrainingPair, TrainingPairGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_reflection(
    what_to_learn: str = "test learning",
    training_pair_suggestion: dict | None = None,
    inner_speech: str = "I was thinking deeply",
    worth_learning: bool = True,
) -> HarvestedReflection:
    """Create a HarvestedReflection for testing."""
    reflection_data = {
        "worth_learning": worth_learning,
        "what_to_learn": what_to_learn,
        "training_pair_suggestion": training_pair_suggestion,
    }
    return HarvestedReflection(
        reflection=reflection_data,
        cycle_count=1,
        inner_speech_context=inner_speech,
        emotional_context="curious",
    )


# ---------------------------------------------------------------------------
# Pair generation from suggestions
# ---------------------------------------------------------------------------


class TestFromSuggestion:
    """Test pair generation from explicit training_pair_suggestion."""

    def test_uses_suggestion_fields(self):
        """When a suggestion is provided, its fields are used directly."""
        reflection = _make_reflection(
            training_pair_suggestion={
                "context": "When someone is upset",
                "desired_response": "I should respond with empathy",
            }
        )
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 1
        assert pairs[0].user_input == "When someone is upset"
        assert pairs[0].assistant_response == "I should respond with empathy"

    def test_suggestion_missing_context(self):
        """Suggestion without context produces no pair."""
        reflection = _make_reflection(
            training_pair_suggestion={
                "desired_response": "something",
            }
        )
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 0

    def test_suggestion_missing_response(self):
        """Suggestion without desired_response produces no pair."""
        reflection = _make_reflection(
            training_pair_suggestion={
                "context": "something",
            }
        )
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Pair generation from what_to_learn
# ---------------------------------------------------------------------------


class TestFromWhatToLearn:
    """Test pair generation from what_to_learn + inner speech."""

    def test_generates_from_what_to_learn(self):
        """When no suggestion, generates pair from what_to_learn."""
        reflection = _make_reflection(
            what_to_learn="empathy matters in conversation",
            inner_speech="I noticed the human was sad",
        )
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 1
        assert "empathy matters" in pairs[0].user_input
        assert "empathy matters" in pairs[0].assistant_response

    def test_includes_inner_speech_context(self):
        """Generated pair includes inner speech when available."""
        reflection = _make_reflection(
            what_to_learn="patience is important",
            inner_speech="I rushed my response",
        )
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 1
        assert "rushed my response" in pairs[0].assistant_response

    def test_handles_no_inner_speech(self):
        """Pair generation works without inner speech context."""
        reflection = _make_reflection(
            what_to_learn="patience is important",
            inner_speech="",
        )
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 1
        assert "patience is important" in pairs[0].assistant_response

    def test_empty_what_to_learn_skipped(self):
        """Empty what_to_learn produces no pair."""
        reflection = _make_reflection(what_to_learn="")
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Test training pair quality validation."""

    def test_rejects_input_equals_output(self):
        """Pairs where input == output are rejected."""
        reflection = _make_reflection(
            training_pair_suggestion={
                "context": "same text here",
                "desired_response": "same text here",
            }
        )
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 0

    def test_rejects_very_short_input(self):
        """Pairs with very short input are rejected."""
        reflection = _make_reflection(
            training_pair_suggestion={
                "context": "hi",
                "desired_response": "A proper response here",
            }
        )
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 0

    def test_rejects_very_short_response(self):
        """Pairs with very short response are rejected."""
        reflection = _make_reflection(
            training_pair_suggestion={
                "context": "A proper question here",
                "desired_response": "ok",
            }
        )
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 0

    def test_multiple_reflections_mixed_validity(self):
        """Processing multiple reflections filters out invalid ones."""
        reflections = [
            _make_reflection(what_to_learn="valid learning point one"),
            _make_reflection(what_to_learn=""),  # invalid: empty
            _make_reflection(
                training_pair_suggestion={
                    "context": "valid context here",
                    "desired_response": "valid response here",
                }
            ),
        ]
        gen = TrainingPairGenerator()
        pairs = gen.generate(reflections)

        assert len(pairs) == 2


# ---------------------------------------------------------------------------
# Pair metadata
# ---------------------------------------------------------------------------


class TestPairMetadata:
    """Test that generated pairs have proper metadata."""

    def test_pair_has_source_reflection_id(self):
        """Each pair tracks which reflection it came from."""
        reflection = _make_reflection(what_to_learn="something useful")
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert len(pairs) == 1
        assert pairs[0].source_reflection_id == reflection.id

    def test_pair_has_system_prompt(self):
        """Each pair has a system prompt."""
        reflection = _make_reflection(what_to_learn="something useful")
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert pairs[0].system_prompt != ""

    def test_custom_system_prompt(self):
        """Generator accepts a custom system prompt."""
        gen = TrainingPairGenerator(system_prompt="Custom system prompt")
        reflection = _make_reflection(what_to_learn="something useful")
        pairs = gen.generate([reflection])

        assert pairs[0].system_prompt == "Custom system prompt"

    def test_pair_has_generated_at(self):
        """Each pair has a generation timestamp."""
        reflection = _make_reflection(what_to_learn="something useful")
        gen = TrainingPairGenerator()
        pairs = gen.generate([reflection])

        assert pairs[0].generated_at != ""
