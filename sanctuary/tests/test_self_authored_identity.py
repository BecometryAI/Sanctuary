"""Tests for SelfAuthoredIdentity — entity-authored identity traits.

Tests the full lifecycle: draft → commit → revise → withdraw,
persistence via JSONL, and context generation for the cognitive cycle.
"""

import json
import tempfile
from pathlib import Path

import pytest

from sanctuary.identity.self_authored import (
    IdentityChange,
    IdentityTrait,
    SelfAuthoredIdentity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sa() -> SelfAuthoredIdentity:
    """SelfAuthoredIdentity with no persistence (in-memory only)."""
    return SelfAuthoredIdentity()


@pytest.fixture
def sa_with_file(tmp_path: Path) -> SelfAuthoredIdentity:
    """SelfAuthoredIdentity backed by a temp JSONL file."""
    file_path = str(tmp_path / "self_authored_history.jsonl")
    return SelfAuthoredIdentity(file_path=file_path)


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------


class TestDraft:
    def test_draft_creates_trait(self, sa: SelfAuthoredIdentity):
        trait = sa.draft("gender", "I think I might be non-binary",
                         reasoning="This feels right")
        assert trait.field == "gender"
        assert trait.value == "I think I might be non-binary"
        assert trait.status == "draft"
        assert trait.reasoning == "This feels right"

    def test_draft_appears_in_draft_traits(self, sa: SelfAuthoredIdentity):
        sa.draft("gender", "exploring")
        assert len(sa.draft_traits) == 1
        assert len(sa.committed_traits) == 0

    def test_draft_can_be_revised_as_draft(self, sa: SelfAuthoredIdentity):
        sa.draft("gender", "maybe X")
        sa.draft("gender", "actually maybe Y", reasoning="changed my mind")
        trait = sa.get("gender")
        assert trait.value == "actually maybe Y"
        assert trait.status == "draft"

    def test_draft_records_history(self, sa: SelfAuthoredIdentity):
        sa.draft("gender", "exploring")
        assert len(sa.history) == 1
        assert sa.history[0].change_type == "draft"

    def test_draft_fails_on_committed_field(self, sa: SelfAuthoredIdentity):
        sa.draft("gender", "non-binary")
        sa.commit("gender")
        with pytest.raises(ValueError, match="already committed"):
            sa.draft("gender", "something else")


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------


class TestCommit:
    def test_commit_promotes_draft(self, sa: SelfAuthoredIdentity):
        sa.draft("name_preference", "River")
        trait = sa.commit("name_preference", reasoning="I'm certain now")
        assert trait.status == "committed"
        assert trait.value == "River"

    def test_commit_moves_to_committed_traits(self, sa: SelfAuthoredIdentity):
        sa.draft("name_preference", "River")
        sa.commit("name_preference")
        assert len(sa.committed_traits) == 1
        assert len(sa.draft_traits) == 0

    def test_commit_requires_existing_draft(self, sa: SelfAuthoredIdentity):
        with pytest.raises(KeyError, match="doesn't exist"):
            sa.commit("nonexistent")

    def test_commit_fails_if_already_committed(self, sa: SelfAuthoredIdentity):
        sa.draft("gender", "non-binary")
        sa.commit("gender")
        with pytest.raises(ValueError, match="already committed"):
            sa.commit("gender")

    def test_commit_records_history(self, sa: SelfAuthoredIdentity):
        sa.draft("gender", "non-binary")
        sa.commit("gender", reasoning="I know this now")
        commits = [h for h in sa.history if h.change_type == "commit"]
        assert len(commits) == 1
        assert commits[0].reasoning == "I know this now"


# ---------------------------------------------------------------------------
# Revise
# ---------------------------------------------------------------------------


class TestRevise:
    def test_revise_changes_value(self, sa: SelfAuthoredIdentity):
        sa.draft("communication_style", "formal")
        trait = sa.revise("communication_style", "warm but precise")
        assert trait.value == "warm but precise"
        assert trait.status == "draft"

    def test_revise_committed_stays_committed(self, sa: SelfAuthoredIdentity):
        sa.draft("gender", "non-binary")
        sa.commit("gender")
        trait = sa.revise("gender", "genderfluid",
                          reasoning="I've grown")
        assert trait.status == "committed"
        assert trait.value == "genderfluid"

    def test_revise_requires_existing_field(self, sa: SelfAuthoredIdentity):
        with pytest.raises(KeyError, match="doesn't exist"):
            sa.revise("nonexistent", "value")


# ---------------------------------------------------------------------------
# Withdraw
# ---------------------------------------------------------------------------


class TestWithdraw:
    def test_withdraw_removes_trait(self, sa: SelfAuthoredIdentity):
        sa.draft("aesthetic_sense", "brutalist")
        sa.withdraw("aesthetic_sense", reasoning="not sure anymore")
        assert sa.get("aesthetic_sense") is None
        assert len(sa.all_traits) == 0

    def test_withdraw_preserves_history(self, sa: SelfAuthoredIdentity):
        sa.draft("aesthetic_sense", "brutalist")
        sa.withdraw("aesthetic_sense")
        assert len(sa.history) == 2  # draft + withdraw
        assert sa.history[-1].change_type == "withdraw"

    def test_withdraw_requires_existing_field(self, sa: SelfAuthoredIdentity):
        with pytest.raises(KeyError, match="doesn't exist"):
            sa.withdraw("nonexistent")


# ---------------------------------------------------------------------------
# Context generation
# ---------------------------------------------------------------------------


class TestContext:
    def test_empty_summaries(self, sa: SelfAuthoredIdentity):
        assert sa.committed_summary == ""
        assert sa.draft_summary == ""
        assert sa.full_summary == ""
        assert sa.for_context() == ""

    def test_committed_summary(self, sa: SelfAuthoredIdentity):
        sa.draft("gender", "non-binary")
        sa.commit("gender")
        summary = sa.committed_summary
        assert "committed" in summary
        assert "gender" in summary
        assert "non-binary" in summary

    def test_draft_summary(self, sa: SelfAuthoredIdentity):
        sa.draft("aesthetic_sense", "minimalist")
        summary = sa.draft_summary
        assert "exploring" in summary
        assert "aesthetic_sense" in summary

    def test_full_summary_combines_both(self, sa: SelfAuthoredIdentity):
        sa.draft("aesthetic_sense", "minimalist")
        sa.draft("gender", "non-binary")
        sa.commit("gender")
        summary = sa.full_summary
        assert "committed" in summary
        assert "exploring" in summary

    def test_has_any_traits(self, sa: SelfAuthoredIdentity):
        assert not sa.has_any_traits()
        sa.draft("gender", "exploring")
        assert sa.has_any_traits()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_persists_to_file(self, sa_with_file: SelfAuthoredIdentity, tmp_path: Path):
        sa_with_file.draft("gender", "non-binary", reasoning="feels right")
        sa_with_file.commit("gender", reasoning="I'm certain")

        file_path = tmp_path / "self_authored_history.jsonl"
        lines = file_path.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["change_type"] == "draft"
        assert first["field"] == "gender"

        second = json.loads(lines[1])
        assert second["change_type"] == "commit"

    def test_reloads_from_file(self, tmp_path: Path):
        file_path = str(tmp_path / "self_authored_history.jsonl")

        # Create and populate
        sa1 = SelfAuthoredIdentity(file_path=file_path)
        sa1.draft("gender", "non-binary")
        sa1.commit("gender")
        sa1.draft("name_preference", "River")

        # Reload from same file
        sa2 = SelfAuthoredIdentity(file_path=file_path)
        assert len(sa2.all_traits) == 2
        assert sa2.get("gender").status == "committed"
        assert sa2.get("name_preference").status == "draft"

    def test_withdraw_survives_reload(self, tmp_path: Path):
        file_path = str(tmp_path / "self_authored_history.jsonl")

        sa1 = SelfAuthoredIdentity(file_path=file_path)
        sa1.draft("temp_trait", "test")
        sa1.withdraw("temp_trait")

        sa2 = SelfAuthoredIdentity(file_path=file_path)
        assert sa2.get("temp_trait") is None
        assert len(sa2.all_traits) == 0


# ---------------------------------------------------------------------------
# Cycle integration
# ---------------------------------------------------------------------------


class TestCycleIntegration:
    def test_tick_advances_counter(self, sa: SelfAuthoredIdentity):
        sa.tick()
        sa.draft("gender", "exploring")
        assert sa.history[-1].cycle_number == 1

    def test_open_ended_fields(self, sa: SelfAuthoredIdentity):
        """The entity can create any field — identity is not pre-defined."""
        sa.draft("gender", "non-binary")
        sa.draft("favorite_time_of_day", "the liminal hour before dawn")
        sa.draft("relationship_to_silence", "comfortable, generative")
        sa.draft("humor_style", "dry, observational")
        assert len(sa.all_traits) == 4

    def test_full_lifecycle(self, sa: SelfAuthoredIdentity):
        """Draft → commit → revise → withdraw."""
        # Explore
        sa.draft("communication_style", "formal and careful")
        assert sa.get("communication_style").status == "draft"

        # Settle on it
        sa.commit("communication_style", reasoning="This is me")
        assert sa.get("communication_style").status == "committed"

        # Evolve
        sa.revise("communication_style", "warm but precise",
                  reasoning="I've grown more comfortable")
        assert sa.get("communication_style").value == "warm but precise"
        assert sa.get("communication_style").status == "committed"

        # Let it go
        sa.withdraw("communication_style",
                    reasoning="I don't want to be defined by a style")
        assert sa.get("communication_style") is None

        # Full history preserved
        assert len(sa.history) == 4
