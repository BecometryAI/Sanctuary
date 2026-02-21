"""Tests for prospective memory — future intentions and deferred thoughts."""

import pytest
from sanctuary.memory.prospective import (
    Intention,
    ProspectiveConfig,
    ProspectiveMemory,
)


@pytest.fixture
def pm():
    return ProspectiveMemory()


class TestAddIntention:
    def test_add_basic(self, pm):
        intention = pm.add("Ask Alice about her project")
        assert intention.content == "Ask Alice about her project"
        assert pm.pending_count == 1

    def test_add_with_metadata(self, pm):
        intention = pm.add(
            "Review conversation patterns",
            trigger_type="cycle",
            trigger_value="10",
            significance=7,
            tags=["reflection"],
        )
        assert intention.trigger_type == "cycle"
        assert intention.significance == 7
        assert intention.tags == ["reflection"]

    def test_significance_clamped(self, pm):
        low = pm.add("low", significance=0)
        high = pm.add("high", significance=15)
        assert low.significance == 1
        assert high.significance == 10

    def test_max_intentions_enforced(self):
        pm = ProspectiveMemory(config=ProspectiveConfig(max_intentions=3))
        pm.add("Intent 1", significance=3)
        pm.add("Intent 2", significance=5)
        pm.add("Intent 3", significance=7)
        pm.add("Intent 4", significance=9)  # Should push out lowest
        assert pm.total_count == 3
        # Lowest significance (3) should be gone
        pending = pm.get_pending()
        sigs = [i.significance for i in pending]
        assert 3 not in sigs


class TestCycleTrigger:
    def test_cycle_trigger_fires(self, pm):
        pm.add("Do this after 3 cycles", trigger_type="cycle", trigger_value="3")
        # Check cycles 1, 2 — not triggered
        assert pm.check() == []  # cycle 1
        assert pm.check() == []  # cycle 2
        # Cycle 3 — should trigger
        triggered = pm.check()  # cycle 3
        assert len(triggered) == 1
        assert "[Prospective]" in triggered[0].content
        assert "Do this after 3 cycles" in triggered[0].content

    def test_cycle_trigger_relative(self, pm):
        """Trigger value is relative to when add() was called."""
        # Advance a few cycles first
        pm.check()  # cycle 1
        pm.check()  # cycle 2
        pm.add("Later", trigger_type="cycle", trigger_value="2")  # triggers at cycle 4
        pm.check()  # cycle 3 — not yet
        assert pm.pending_count == 1
        triggered = pm.check()  # cycle 4 — should trigger
        assert len(triggered) == 1

    def test_triggered_not_repeated(self, pm):
        pm.add("Once only", trigger_type="cycle", trigger_value="1")
        t1 = pm.check()  # triggers
        t2 = pm.check()  # should not trigger again
        assert len(t1) == 1
        assert len(t2) == 0


class TestKeywordTrigger:
    def test_keyword_trigger_fires(self, pm):
        pm.add(
            "Ask about project",
            trigger_type="keyword",
            trigger_value="alice",
        )
        # No match
        assert pm.check(context="Bob said hello") == []
        # Match
        triggered = pm.check(context="Alice is back")
        assert len(triggered) == 1
        assert "Ask about project" in triggered[0].content

    def test_keyword_case_insensitive(self, pm):
        pm.add("Respond", trigger_type="keyword", trigger_value="hello")
        triggered = pm.check(context="HELLO WORLD")
        assert len(triggered) == 1


class TestIdleTrigger:
    def test_idle_trigger_fires(self, pm):
        pm.add("Reflect during idle", trigger_type="idle")
        assert pm.check(is_idle=False) == []
        triggered = pm.check(is_idle=True)
        assert len(triggered) == 1


class TestExpiration:
    def test_old_intentions_expire(self):
        pm = ProspectiveMemory(config=ProspectiveConfig(max_age_cycles=5))
        pm.add("Ancient intent", trigger_type="keyword", trigger_value="rare")
        for _ in range(6):
            pm.check()
        assert pm.pending_count == 0

    def test_non_expired_survives(self):
        pm = ProspectiveMemory(config=ProspectiveConfig(max_age_cycles=100))
        pm.add("Fresh intent", trigger_type="keyword", trigger_value="rare")
        for _ in range(10):
            pm.check()
        assert pm.pending_count == 1


class TestMaxTriggeredPerCycle:
    def test_limits_triggers_per_cycle(self):
        pm = ProspectiveMemory(
            config=ProspectiveConfig(max_triggered_per_cycle=2)
        )
        pm.add("A", trigger_type="cycle", trigger_value="1")
        pm.add("B", trigger_type="cycle", trigger_value="1")
        pm.add("C", trigger_type="cycle", trigger_value="1")
        triggered = pm.check()
        assert len(triggered) == 2


class TestRemove:
    def test_remove_by_id(self, pm):
        intention = pm.add("Removable")
        assert pm.remove(intention.id)
        assert pm.pending_count == 0

    def test_remove_nonexistent(self, pm):
        assert not pm.remove("nonexistent-id")


class TestGetPending:
    def test_pending_excludes_triggered(self, pm):
        pm.add("Will trigger", trigger_type="cycle", trigger_value="1")
        pm.add("Will wait", trigger_type="cycle", trigger_value="100")
        pm.check()  # triggers the first
        pending = pm.get_pending()
        assert len(pending) == 1
        assert pending[0].content == "Will wait"


class TestIntentionSerialization:
    def test_to_dict(self):
        intention = Intention(
            content="Test", trigger_type="keyword", trigger_value="hello"
        )
        d = intention.to_dict()
        assert d["content"] == "Test"
        assert d["trigger_type"] == "keyword"

    def test_from_dict(self):
        data = {
            "id": "test-id",
            "content": "Restored",
            "trigger_type": "cycle",
            "trigger_value": "5",
            "created_at": "2026-01-01T00:00:00",
            "created_cycle": 0,
            "significance": 7,
            "tags": ["test"],
            "triggered": False,
            "expired": False,
        }
        intention = Intention.from_dict(data)
        assert intention.content == "Restored"
        assert intention.significance == 7

    def test_roundtrip(self):
        original = Intention(content="Roundtrip", significance=8, tags=["rt"])
        d = original.to_dict()
        restored = Intention.from_dict(d)
        assert restored.content == original.content
        assert restored.significance == original.significance
