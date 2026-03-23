"""Tests for AdapterRegistry — LoRA adapter lifecycle management.

Tests cover:
- AdapterRecord: creation, serialization, deserialization
- AdapterRegistry: registration, retrieval, filtering
- Lifecycle transitions: activate, deactivate, merge, retire, keep
- Persistence: save/load round-trip
- Entity decisions: keep vs merge with reasons
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from sanctuary.growth.adapter_registry import (
    AdapterRecord,
    AdapterRegistry,
    AdapterStatus,
)


# ---------------------------------------------------------------------------
# AdapterRecord tests
# ---------------------------------------------------------------------------


class TestAdapterRecord:
    def test_creation_defaults(self):
        record = AdapterRecord(name="test_v1", domain="test", path="/adapters/test_v1")
        assert record.name == "test_v1"
        assert record.domain == "test"
        assert record.status == AdapterStatus.STORED
        assert record.rank == 8
        assert record.alpha == 16
        assert record.training_pair_count == 0

    def test_creation_custom(self):
        record = AdapterRecord(
            name="spatial_v2",
            domain="spatial_reasoning",
            path="/adapters/spatial_v2",
            rank=64,
            alpha=128,
            training_pair_count=150,
            final_loss=0.023,
            description="Spatial reasoning from navigation experience",
        )
        assert record.rank == 64
        assert record.training_pair_count == 150
        assert record.final_loss == 0.023

    def test_to_dict(self):
        record = AdapterRecord(name="test", domain="test", path="/test")
        d = record.to_dict()
        assert d["name"] == "test"
        assert d["status"] == "stored"
        assert d["rank"] == 8

    def test_from_dict(self):
        data = {
            "name": "test",
            "domain": "test",
            "path": "/test",
            "status": "active",
            "rank": 32,
        }
        record = AdapterRecord.from_dict(data)
        assert record.name == "test"
        assert record.status == AdapterStatus.ACTIVE
        assert record.rank == 32

    def test_round_trip(self):
        original = AdapterRecord(
            name="roundtrip",
            domain="emotional",
            path="/adapters/roundtrip",
            status=AdapterStatus.ACTIVE,
            rank=16,
            training_pair_count=50,
            keep_reason="Valuable for empathy processing",
        )
        restored = AdapterRecord.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.domain == original.domain
        assert restored.status == original.status
        assert restored.rank == original.rank
        assert restored.keep_reason == original.keep_reason


# ---------------------------------------------------------------------------
# AdapterRegistry basic tests
# ---------------------------------------------------------------------------


class TestAdapterRegistry:
    def test_empty_registry(self):
        registry = AdapterRegistry()
        assert registry.adapter_count == 0
        assert registry.active_count == 0
        assert registry.stored_count == 0
        assert registry.merged_count == 0

    def test_register(self):
        registry = AdapterRegistry()
        record = AdapterRecord(name="test_v1", domain="test", path="/test")
        result = registry.register(record)
        assert result.name == "test_v1"
        assert registry.adapter_count == 1
        assert registry.stored_count == 1

    def test_register_duplicate_raises(self):
        registry = AdapterRegistry()
        record = AdapterRecord(name="test", domain="test", path="/test")
        registry.register(record)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(AdapterRecord(name="test", domain="test", path="/test2"))

    def test_get(self):
        registry = AdapterRegistry()
        record = AdapterRecord(name="test", domain="test", path="/test")
        registry.register(record)
        retrieved = registry.get("test")
        assert retrieved is record

    def test_get_nonexistent_raises(self):
        registry = AdapterRegistry()
        with pytest.raises(KeyError, match="No adapter named"):
            registry.get("nonexistent")

    def test_has(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="exists", domain="test", path="/test"))
        assert registry.has("exists")
        assert not registry.has("missing")

    def test_all_adapters(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="a", domain="d1", path="/a"))
        registry.register(AdapterRecord(name="b", domain="d2", path="/b"))
        all_adapters = registry.all_adapters()
        assert len(all_adapters) == 2

    def test_by_domain(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="a", domain="spatial", path="/a"))
        registry.register(AdapterRecord(name="b", domain="emotional", path="/b"))
        registry.register(AdapterRecord(name="c", domain="spatial", path="/c"))
        spatial = registry.by_domain("spatial")
        assert len(spatial) == 2
        assert all(a.domain == "spatial" for a in spatial)

    def test_by_status(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="a", domain="test", path="/a", status=AdapterStatus.STORED))
        registry.register(AdapterRecord(name="b", domain="test", path="/b", status=AdapterStatus.ACTIVE))
        stored = registry.by_status(AdapterStatus.STORED)
        assert len(stored) == 1
        assert stored[0].name == "a"


# ---------------------------------------------------------------------------
# Lifecycle transition tests
# ---------------------------------------------------------------------------


class TestLifecycleTransitions:
    def test_activate_stored(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="test", domain="test", path="/test"))
        result = registry.activate_adapter("test")
        assert result.status == AdapterStatus.ACTIVE
        assert registry.active_count == 1
        assert registry.stored_count == 0

    def test_activate_merged_raises(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="test", domain="test", path="/test"))
        registry.mark_merged("test")
        with pytest.raises(ValueError, match="Cannot activate"):
            registry.activate_adapter("test")

    def test_activate_retired_raises(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="test", domain="test", path="/test"))
        registry.retire_adapter("test")
        with pytest.raises(ValueError, match="Cannot activate"):
            registry.activate_adapter("test")

    def test_deactivate(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="test", domain="test", path="/test"))
        registry.activate_adapter("test")
        result = registry.deactivate_adapter("test")
        assert result.status == AdapterStatus.STORED
        assert registry.active_count == 0
        assert registry.stored_count == 1

    def test_deactivate_not_active_raises(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="test", domain="test", path="/test"))
        with pytest.raises(ValueError, match="not active"):
            registry.deactivate_adapter("test")

    def test_mark_merged(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="test", domain="test", path="/test"))
        result = registry.mark_merged("test", reason="Integrated into base cognition")
        assert result.status == AdapterStatus.MERGED
        assert result.merged_at is not None
        assert result.merge_reason == "Integrated into base cognition"
        assert registry.merged_count == 1

    def test_retire(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="test", domain="test", path="/test"))
        result = registry.retire_adapter("test", reason="Superseded by v2")
        assert result.status == AdapterStatus.RETIRED
        assert result.retired_at is not None

    def test_nonexistent_transitions_raise(self):
        registry = AdapterRegistry()
        with pytest.raises(KeyError):
            registry.activate_adapter("missing")
        with pytest.raises(KeyError):
            registry.deactivate_adapter("missing")
        with pytest.raises(KeyError):
            registry.mark_merged("missing")
        with pytest.raises(KeyError):
            registry.retire_adapter("missing")


# ---------------------------------------------------------------------------
# Entity decision tests
# ---------------------------------------------------------------------------


class TestEntityDecisions:
    def test_keep_adapter_with_reason(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="empathy_v1", domain="emotional", path="/empathy"))
        result = registry.keep_adapter("empathy_v1", reason="Essential for emotional conversations")
        assert result.keep_reason == "Essential for emotional conversations"

    def test_keep_then_merge(self):
        """Entity can change their mind — keep first, then merge later."""
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="test", domain="test", path="/test"))
        registry.keep_adapter("test", reason="Keeping for now")
        registry.mark_merged("test", reason="Actually, integrate it")
        record = registry.get("test")
        assert record.status == AdapterStatus.MERGED
        assert record.keep_reason == "Keeping for now"
        assert record.merge_reason == "Actually, integrate it"

    def test_multiple_adapters_same_domain(self):
        """Entity accumulates multiple versions in the same domain."""
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="spatial_v1", domain="spatial", path="/s1"))
        registry.register(AdapterRecord(name="spatial_v2", domain="spatial", path="/s2"))
        registry.register(AdapterRecord(name="spatial_v3", domain="spatial", path="/s3"))
        # Retire old ones, keep latest
        registry.retire_adapter("spatial_v1", reason="Superseded by v2")
        registry.retire_adapter("spatial_v2", reason="Superseded by v3")
        registry.keep_adapter("spatial_v3", reason="Current best spatial reasoning")
        spatial = registry.by_domain("spatial")
        assert len(spatial) == 3
        active_spatial = [a for a in spatial if a.status != AdapterStatus.RETIRED]
        assert len(active_spatial) == 1
        assert active_spatial[0].name == "spatial_v3"


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------


class TestSummary:
    def test_empty_summary(self):
        registry = AdapterRegistry()
        summary = registry.get_summary()
        assert summary["total_adapters"] == 0
        assert summary["domains"] == {}

    def test_populated_summary(self):
        registry = AdapterRegistry()
        registry.register(AdapterRecord(name="a", domain="spatial", path="/a", rank=16))
        registry.register(AdapterRecord(name="b", domain="emotional", path="/b", rank=32))
        registry.register(AdapterRecord(name="c", domain="spatial", path="/c", rank=64))
        registry.activate_adapter("a")
        registry.mark_merged("c")
        summary = registry.get_summary()
        assert summary["total_adapters"] == 3
        assert summary["active"] == 1
        assert summary["stored"] == 1
        assert summary["merged"] == 1
        assert "spatial" in summary["domains"]
        assert "emotional" in summary["domains"]
        assert len(summary["domains"]["spatial"]) == 2
        # Only active + stored count toward unmerged rank
        assert summary["total_unmerged_rank"] == 16 + 32


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        registry = AdapterRegistry(registry_path=tmp_path)
        registry.register(AdapterRecord(
            name="persist_test",
            domain="spatial",
            path="/adapters/persist",
            rank=32,
            training_pair_count=75,
            description="Test adapter for persistence",
        ))
        registry.keep_adapter("persist_test", reason="Testing save/load")

        saved_path = registry.save()
        assert saved_path.exists()

        # Load into a fresh registry
        registry2 = AdapterRegistry(registry_path=tmp_path)
        loaded_count = registry2.load()
        assert loaded_count == 1
        record = registry2.get("persist_test")
        assert record.domain == "spatial"
        assert record.rank == 32
        assert record.training_pair_count == 75
        assert record.keep_reason == "Testing save/load"

    def test_load_nonexistent(self, tmp_path):
        registry = AdapterRegistry(registry_path=tmp_path)
        loaded = registry.load()
        assert loaded == 0
        assert registry.adapter_count == 0

    def test_save_multiple_statuses(self, tmp_path):
        registry = AdapterRegistry(registry_path=tmp_path)
        registry.register(AdapterRecord(name="active_one", domain="d1", path="/a"))
        registry.register(AdapterRecord(name="merged_one", domain="d2", path="/b"))
        registry.register(AdapterRecord(name="retired_one", domain="d3", path="/c"))
        registry.activate_adapter("active_one")
        registry.mark_merged("merged_one", reason="Integrated")
        registry.retire_adapter("retired_one", reason="Superseded")
        registry.save()

        registry2 = AdapterRegistry(registry_path=tmp_path)
        registry2.load()
        assert registry2.get("active_one").status == AdapterStatus.ACTIVE
        assert registry2.get("merged_one").status == AdapterStatus.MERGED
        assert registry2.get("merged_one").merge_reason == "Integrated"
        assert registry2.get("retired_one").status == AdapterStatus.RETIRED

    def test_save_format_is_json(self, tmp_path):
        registry = AdapterRegistry(registry_path=tmp_path)
        registry.register(AdapterRecord(name="json_test", domain="test", path="/test"))
        registry.save()
        meta_file = tmp_path / "adapter_registry.json"
        data = json.loads(meta_file.read_text())
        assert data["version"] == 1
        assert "json_test" in data["adapters"]
        assert "saved_at" in data
