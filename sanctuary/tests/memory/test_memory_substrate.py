"""Tests for MemorySubstrate — the main memory system interface."""

import pytest
from sanctuary.core.schema import MemoryOp, SurfacedMemory
from sanctuary.memory.manager import InMemoryStore, MemorySubstrate, MemorySubstrateConfig
from sanctuary.memory.journal import JournalConfig


@pytest.fixture
def substrate():
    return MemorySubstrate()


class TestSurfacing:
    @pytest.mark.asyncio
    async def test_surface_empty(self, substrate):
        result = await substrate.surface("")
        assert result == []

    @pytest.mark.asyncio
    async def test_surface_with_stored_memory(self, substrate):
        """After storing a memory, it should be surfaceable."""
        await substrate.execute_ops(
            [MemoryOp(type="write_episodic", content="Alice visited today", significance=6, tags=["alice"])]
        )
        result = await substrate.surface("alice")
        assert len(result) >= 1
        assert any("Alice" in m.content for m in result)

    @pytest.mark.asyncio
    async def test_surface_includes_prospective(self, substrate):
        """Triggered prospective intentions should appear in surfaced memories."""
        substrate.add_intention(
            "Remember to ask about project",
            trigger_type="cycle",
            trigger_value="1",
        )
        result = await substrate.surface("anything")
        assert any("[Prospective]" in m.content for m in result)

    @pytest.mark.asyncio
    async def test_surface_includes_queued_retrievals(self, substrate):
        """Queued retrievals from previous cycle should be included."""
        # Store something first
        await substrate.execute_ops(
            [MemoryOp(type="write_episodic", content="Bob's birthday is March 5", significance=5, tags=["bob"])]
        )
        # Queue a retrieval (uses substring match against content)
        await substrate.queue_retrieval("bob")
        # Surface with unrelated context — queued retrieval should still find Bob
        result = await substrate.surface("unrelated context")
        assert any("Bob" in m.content for m in result)
        # Verify the queue was drained
        assert len(substrate._retrieval_queue) == 0


class TestExecuteOps:
    @pytest.mark.asyncio
    async def test_write_episodic(self, substrate):
        results = await substrate.execute_ops(
            [MemoryOp(type="write_episodic", content="An event happened", significance=5)]
        )
        assert len(results) == 1
        assert "episodic" in results[0].lower()
        assert substrate.store.entry_count == 1

    @pytest.mark.asyncio
    async def test_write_semantic(self, substrate):
        results = await substrate.execute_ops(
            [MemoryOp(type="write_semantic", content="Python is a language", significance=4, tags=["python"])]
        )
        assert len(results) == 1
        assert "semantic" in results[0].lower()

    @pytest.mark.asyncio
    async def test_write_journal(self, substrate):
        results = await substrate.execute_ops(
            [MemoryOp(type="journal", content="I noticed a pattern today", significance=6, tags=["reflection"])]
        )
        assert len(results) == 1
        assert "journal" in results[0].lower()
        assert substrate.journal.entry_count == 1

    @pytest.mark.asyncio
    async def test_retrieve_op_queues(self, substrate):
        results = await substrate.execute_ops(
            [MemoryOp(type="retrieve", query="alice birthday")]
        )
        assert len(results) == 1
        assert "retrieval" in results[0].lower()

    @pytest.mark.asyncio
    async def test_multiple_ops(self, substrate):
        results = await substrate.execute_ops([
            MemoryOp(type="write_episodic", content="Event 1", significance=5),
            MemoryOp(type="journal", content="Reflection 1", significance=6),
            MemoryOp(type="retrieve", query="past events"),
        ])
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_emotional_tone_passed_to_journal(self, substrate):
        await substrate.execute_ops(
            [MemoryOp(type="journal", content="Warm moment", significance=7)],
            emotional_tone="warmth",
        )
        entry = substrate.journal.entries[0]
        assert entry.emotional_tone == "warmth"

    @pytest.mark.asyncio
    async def test_unknown_op_type(self, substrate):
        results = await substrate.execute_ops(
            [MemoryOp(type="unknown_type", content="test")]
        )
        assert "unknown" in results[0].lower()


class TestInMemoryStore:
    @pytest.mark.asyncio
    async def test_store_and_recall(self):
        store = InMemoryStore()
        store.store({"content": "Hello world", "significance_score": 5, "tags": ["test"]})
        results = await store.recall("hello")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_recall_empty(self):
        store = InMemoryStore()
        results = await store.recall("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_recall_min_significance(self):
        store = InMemoryStore()
        store.store({"content": "low sig", "significance_score": 2, "tags": []})
        store.store({"content": "high sig", "significance_score": 8, "tags": []})
        results = await store.recall("sig", min_significance=5)
        assert len(results) == 1
        assert results[0]["content"] == "high sig"

    @pytest.mark.asyncio
    async def test_recall_n_results(self):
        store = InMemoryStore()
        for i in range(10):
            store.store({"content": f"test entry {i}", "tags": ["test"]})
        results = await store.recall("test", n_results=3)
        assert len(results) == 3


class TestTick:
    def test_tick_advances_journal(self, substrate):
        substrate.tick()
        substrate.tick()
        substrate.journal.write("After ticks")
        entry = substrate.journal.entries[0]
        assert entry.cycle_number == 2


class TestAccessors:
    def test_journal_accessor(self, substrate):
        assert substrate.journal is not None

    def test_prospective_accessor(self, substrate):
        assert substrate.prospective is not None

    def test_surfacer_accessor(self, substrate):
        assert substrate.surfacer is not None

    def test_store_accessor(self, substrate):
        assert substrate.store is not None
        assert isinstance(substrate.store, InMemoryStore)


class TestConfig:
    def test_custom_config(self):
        config = MemorySubstrateConfig(
            journal=JournalConfig(max_entries_in_memory=10),
        )
        substrate = MemorySubstrate(config=config)
        assert substrate.journal._config.max_entries_in_memory == 10
