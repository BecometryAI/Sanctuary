"""Tests for context window budget allocation and compression."""

import pytest
from sanctuary.core.context_manager import BudgetConfig, ContextManager
from sanctuary.core.schema import (
    CognitiveInput,
    EmotionalInput,
    Percept,
    PreviousThought,
    ScaffoldSignals,
    SelfModel,
    SurfacedMemory,
    WorldEntity,
    WorldModel,
)


class TestBudgetConfig:
    def test_default_target(self):
        config = BudgetConfig()
        assert config.total_target == 4000

    def test_budget_bytes(self):
        config = BudgetConfig(previous_thought=500, chars_per_token=4)
        assert config.budget_bytes("previous_thought") == 2000

    def test_custom_config(self):
        config = BudgetConfig(previous_thought=1000, new_percepts=1500)
        assert config.previous_thought == 1000
        assert config.new_percepts == 1500


class TestContextManager:
    def test_passthrough_when_within_budget(self):
        """Input that fits within budget should pass through unchanged."""
        mgr = ContextManager()
        ci = CognitiveInput(
            previous_thought=PreviousThought(inner_speech="short thought"),
            new_percepts=[Percept(modality="language", content="hi")],
        )
        compressed = mgr.compress(ci)
        assert compressed.previous_thought.inner_speech == "short thought"
        assert compressed.new_percepts[0].content == "hi"

    def test_inner_speech_truncation(self):
        """Long inner speech should be truncated, preserving the end."""
        config = BudgetConfig(previous_thought=50, chars_per_token=1)
        mgr = ContextManager(config)

        long_speech = "A" * 200
        ci = CognitiveInput(
            previous_thought=PreviousThought(inner_speech=long_speech),
        )
        compressed = mgr.compress(ci)
        assert len(compressed.previous_thought.inner_speech) <= 50
        assert compressed.previous_thought.inner_speech.startswith("...")

    def test_percept_batching(self):
        """Many percepts of same modality should be batched."""
        config = BudgetConfig(new_percepts=100, chars_per_token=1)
        mgr = ContextManager(config)

        percepts = [
            Percept(modality="sensor", content=f"reading_{i}" * 10)
            for i in range(20)
        ]
        ci = CognitiveInput(new_percepts=percepts)
        compressed = mgr.compress(ci)

        # Should have fewer percepts than original
        assert len(compressed.new_percepts) < 20

    def test_memory_prioritization(self):
        """Memories should be kept by significance within budget."""
        config = BudgetConfig(surfaced_memories=50, chars_per_token=1)
        mgr = ContextManager(config)

        memories = [
            SurfacedMemory(content="low significance memory " * 5, significance=2),
            SurfacedMemory(content="high significance memory", significance=9),
            SurfacedMemory(content="medium memory " * 3, significance=5),
        ]
        ci = CognitiveInput(surfaced_memories=memories)
        compressed = mgr.compress(ci)

        # Highest significance should be kept first
        if compressed.surfaced_memories:
            assert compressed.surfaced_memories[0].significance == 9

    def test_compression_stats(self):
        """Compression should track what was compressed."""
        config = BudgetConfig(previous_thought=20, chars_per_token=1)
        mgr = ContextManager(config)

        ci = CognitiveInput(
            previous_thought=PreviousThought(inner_speech="A" * 200),
        )
        mgr.compress(ci)
        stats = mgr.get_last_stats()

        assert stats is not None
        assert "previous_thought" in stats.sections_compressed
        assert stats.original_chars > 0
        assert stats.savings_ratio > 0

    def test_self_model_compression(self):
        """Verbose self-model should be truncated."""
        config = BudgetConfig(self_model=50, chars_per_token=1)
        mgr = ContextManager(config)

        ci = CognitiveInput(
            self_model=SelfModel(
                current_state="x" * 500,
                active_goals=["goal"] * 20,
                uncertainties=["doubt"] * 20,
            )
        )
        compressed = mgr.compress(ci)
        assert len(compressed.self_model.current_state) <= 200
        assert len(compressed.self_model.active_goals) <= 5

    def test_world_model_entity_limit(self):
        """World model with too many entities should be trimmed."""
        config = BudgetConfig(world_model=100, chars_per_token=1)
        mgr = ContextManager(config)

        entities = {
            f"entity_{i}": WorldEntity(
                name=f"entity_{i}",
                properties={"data": "x" * 50},
            )
            for i in range(50)
        }
        ci = CognitiveInput(
            world_model=WorldModel(entities=entities)
        )
        compressed = mgr.compress(ci)
        assert len(compressed.world_model.entities) <= 10

    def test_no_stats_before_compress(self):
        mgr = ContextManager()
        assert mgr.get_last_stats() is None

    def test_empty_input_passthrough(self):
        """Empty input should pass through with no compression."""
        mgr = ContextManager()
        ci = CognitiveInput()
        compressed = mgr.compress(ci)
        assert compressed.previous_thought is None
        assert compressed.new_percepts == []

        stats = mgr.get_last_stats()
        assert stats.sections_compressed == []
