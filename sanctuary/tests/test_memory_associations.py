"""
Tests for cross-memory association detection.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mind.cognitive_core.memory_associations import (
    MemoryAssociationDetector,
    MemoryAssociation,
    ThemeCluster,
)


def _make_memory(mem_id="mem1", tags=None, emotional_signature=None, sig_score=5):
    mem = MagicMock()
    mem.id = mem_id
    mem.tags = tags or []
    mem.emotional_signature = emotional_signature or []
    mem.significance_score = sig_score
    return mem


def _make_emotion_sig(label):
    sig = MagicMock()
    sig.value = label
    return sig


class TestMemoryAssociationDetector:
    def test_init(self):
        detector = MemoryAssociationDetector()
        assert detector.min_cluster_size == 2
        assert detector._known_themes == {}

    @pytest.mark.asyncio
    async def test_no_memory_id_returns_empty(self):
        detector = MemoryAssociationDetector()
        mm = MagicMock()
        result = await detector.detect_associations(mm, recent_memory_id=None)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_find_associated_returns_empty(self):
        detector = MemoryAssociationDetector()
        mm = MagicMock(spec=[])  # no find_associated
        result = await detector.detect_associations(mm, recent_memory_id="abc")
        assert result == []

    @pytest.mark.asyncio
    async def test_too_few_associations_returns_empty(self):
        detector = MemoryAssociationDetector(config={"min_cluster_size": 3})
        mm = MagicMock()
        mm.find_associated = AsyncMock(return_value=[_make_memory()])
        result = await detector.detect_associations(mm, recent_memory_id="abc")
        assert result == []

    @pytest.mark.asyncio
    async def test_tag_based_association(self):
        detector = MemoryAssociationDetector()
        mm = MagicMock()
        mm.find_associated = AsyncMock(return_value=[
            _make_memory("m1", tags=["learning", "insight"]),
            _make_memory("m2", tags=["learning", "reflection"]),
            _make_memory("m3", tags=["learning"]),
        ])
        result = await detector.detect_associations(mm, recent_memory_id="source")
        # Should detect "learning" as a shared theme
        learning_assocs = [a for a in result if a.theme == "learning"]
        assert len(learning_assocs) == 1
        assert "learning" in learning_assocs[0].evidence

    @pytest.mark.asyncio
    async def test_emotional_signature_association(self):
        detector = MemoryAssociationDetector()
        mm = MagicMock()
        mm.find_associated = AsyncMock(return_value=[
            _make_memory("m1", emotional_signature=[_make_emotion_sig("JOY")]),
            _make_memory("m2", emotional_signature=[_make_emotion_sig("JOY")]),
            _make_memory("m3", emotional_signature=[_make_emotion_sig("FEAR")]),
        ])
        result = await detector.detect_associations(mm, recent_memory_id="src")
        emotion_assocs = [a for a in result if a.theme.startswith("emotion:")]
        assert len(emotion_assocs) >= 1
        joy_assoc = [a for a in emotion_assocs if "JOY" in a.theme]
        assert len(joy_assoc) == 1

    @pytest.mark.asyncio
    async def test_excludes_generic_tags(self):
        detector = MemoryAssociationDetector()
        mm = MagicMock()
        mm.find_associated = AsyncMock(return_value=[
            _make_memory("m1", tags=["episodic", "workspace_consolidation"]),
            _make_memory("m2", tags=["episodic", "workspace_consolidation"]),
        ])
        result = await detector.detect_associations(mm, recent_memory_id="src")
        # "episodic" and "workspace_consolidation" should be excluded
        tag_assocs = [a for a in result if not a.theme.startswith("emotion:")]
        assert len(tag_assocs) == 0

    @pytest.mark.asyncio
    async def test_theme_cluster_tracking(self):
        detector = MemoryAssociationDetector()
        mm = MagicMock()
        mm.find_associated = AsyncMock(return_value=[
            _make_memory("m1", tags=["curiosity"]),
            _make_memory("m2", tags=["curiosity"]),
        ])

        await detector.detect_associations(mm, recent_memory_id="s1")
        await detector.detect_associations(mm, recent_memory_id="s2")

        assert "curiosity" in detector._known_themes
        assert detector._known_themes["curiosity"].recurrence_count >= 2

    def test_get_theme_summary(self):
        detector = MemoryAssociationDetector()
        detector._known_themes["test"] = ThemeCluster(
            theme="test", memory_ids=["a", "b"], recurrence_count=3
        )
        summary = detector.get_theme_summary()
        assert summary["total_themes"] == 1
        assert summary["themes"]["test"]["recurrence"] == 3

    def test_get_recurring_themes(self):
        detector = MemoryAssociationDetector()
        detector._known_themes["rare"] = ThemeCluster(
            theme="rare", memory_ids=["a"], recurrence_count=1
        )
        detector._known_themes["common"] = ThemeCluster(
            theme="common", memory_ids=["a", "b", "c"], recurrence_count=5
        )
        recurring = detector.get_recurring_themes(min_recurrence=2)
        assert len(recurring) == 1
        assert recurring[0].theme == "common"

    @pytest.mark.asyncio
    async def test_exception_handling(self):
        detector = MemoryAssociationDetector()
        mm = MagicMock()
        mm.find_associated = AsyncMock(side_effect=RuntimeError("db error"))
        result = await detector.detect_associations(mm, recent_memory_id="x")
        assert result == []
