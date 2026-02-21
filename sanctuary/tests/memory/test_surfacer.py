"""Tests for the memory surfacer — context-based memory retrieval."""

import pytest
from sanctuary.memory.surfacer import MemorySurfacer, SurfacerConfig


class FakeStore:
    """Minimal store for testing surfacer behavior."""

    def __init__(self, entries=None):
        self._entries = entries or []

    async def recall(self, query, n_results=5, min_significance=None):
        results = []
        for e in self._entries:
            if min_significance and e.get("significance_score", 5) < min_significance:
                continue
            if query.lower() in str(e.get("content", "")).lower():
                results.append(e)
            if len(results) >= n_results:
                break
        return results


@pytest.fixture
def store():
    return FakeStore(
        entries=[
            {
                "content": "Alice greeted me warmly yesterday",
                "significance_score": 6,
                "emotional_signature": "warmth",
                "timestamp": "2026-01-01T12:00:00",
            },
            {
                "content": "Alice mentioned her project deadline",
                "significance_score": 7,
                "emotional_signature": "concern",
                "timestamp": "2026-01-02T10:00:00",
            },
            {
                "content": "Reflected on the nature of memory",
                "significance_score": 4,
                "emotional_signature": "curiosity",
                "timestamp": "2026-01-03T08:00:00",
            },
        ]
    )


@pytest.fixture
def surfacer(store):
    return MemorySurfacer(store=store)


class TestSurfacerBasic:
    @pytest.mark.asyncio
    async def test_surface_returns_list(self, surfacer):
        result = await surfacer.surface("alice")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_surface_finds_matching(self, surfacer):
        result = await surfacer.surface("alice")
        assert len(result) >= 1
        assert any("Alice" in m.content for m in result)

    @pytest.mark.asyncio
    async def test_surface_empty_context(self, surfacer):
        result = await surfacer.surface("")
        assert result == []

    @pytest.mark.asyncio
    async def test_surface_no_store(self):
        surfacer = MemorySurfacer(store=None)
        result = await surfacer.surface("anything")
        assert result == []

    @pytest.mark.asyncio
    async def test_surface_no_matches(self, surfacer):
        result = await surfacer.surface("xyznonexistent")
        assert result == []


class TestSurfacerDeduplication:
    @pytest.mark.asyncio
    async def test_no_repeat_within_cooldown(self, surfacer):
        """Same memory should not surface twice within cooldown period."""
        r1 = await surfacer.surface("alice")
        r2 = await surfacer.surface("alice")
        # First call: results. Second call within cooldown: deduped
        assert len(r1) >= 1
        # r2 should be empty or at least not repeat r1's contents
        r1_contents = {m.content for m in r1}
        r2_contents = {m.content for m in r2}
        assert len(r1_contents & r2_contents) == 0

    @pytest.mark.asyncio
    async def test_resurfaces_after_cooldown(self, store):
        """Memory should resurface after cooldown period expires."""
        config = SurfacerConfig(cooldown_cycles=2)
        surfacer = MemorySurfacer(store=store, config=config)

        r1 = await surfacer.surface("alice")
        assert len(r1) >= 1

        # Advance past cooldown
        await surfacer.surface("nothing")  # cycle 2
        await surfacer.surface("nothing")  # cycle 3

        r4 = await surfacer.surface("alice")  # cycle 4 — past cooldown
        assert len(r4) >= 1


class TestSurfacerConfig:
    @pytest.mark.asyncio
    async def test_max_results(self, store):
        config = SurfacerConfig(max_results=1)
        surfacer = MemorySurfacer(store=store, config=config)
        result = await surfacer.surface("alice")
        assert len(result) <= 1

    @pytest.mark.asyncio
    async def test_min_significance(self, store):
        config = SurfacerConfig(min_significance=7)
        surfacer = MemorySurfacer(store=store, config=config)
        result = await surfacer.surface("alice")
        for m in result:
            assert m.significance >= 7


class TestSurfacerEntryConversion:
    @pytest.mark.asyncio
    async def test_converts_dict_entries(self, surfacer):
        result = await surfacer.surface("alice")
        for m in result:
            assert hasattr(m, "content")
            assert hasattr(m, "significance")
            assert hasattr(m, "emotional_tone")

    @pytest.mark.asyncio
    async def test_handles_object_entries(self):
        """Test conversion of object-style entries (like JournalEntry)."""

        class FakeEntry:
            content = "Test memory content"
            significance_score = 8
            emotional_signature = ["joy"]
            timestamp = "2026-01-01"

        class ObjectStore:
            async def recall(self, query, n_results=5, min_significance=None):
                return [FakeEntry()]

        surfacer = MemorySurfacer(store=ObjectStore())
        result = await surfacer.surface("test")
        assert len(result) == 1
        assert "Test memory content" in result[0].content
        assert result[0].significance == 8

    @pytest.mark.asyncio
    async def test_handles_store_exception(self):
        """Surfacer should handle store errors gracefully."""

        class FailingStore:
            async def recall(self, query, n_results=5, min_significance=None):
                raise RuntimeError("store failure")

        surfacer = MemorySurfacer(store=FailingStore())
        result = await surfacer.surface("test")
        assert result == []
