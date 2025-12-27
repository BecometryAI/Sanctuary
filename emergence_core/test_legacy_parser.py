"""
Test suite for legacy journal parser.

Tests conversion of legacy journal format to new Pydantic models.
"""

import pytest
import asyncio
import json
from pathlib import Path
from datetime import datetime

from lyra.legacy_parser import LegacyJournalEntry, LegacyParser
from lyra.memory_manager import EmotionalState


@pytest.fixture
def sample_legacy_dict():
    """Sample legacy journal entry as dictionary."""
    return {
        "timestamp": "2025-07-15T14:23:00Z",
        "label": "Morning Reflection",
        "entry_type": "ritual_observance",
        "emotional_tone": ["coherent", "stable", "grateful"],
        "description": "Begin observing the daily ritual of presence.",
        "key_insights": [
            "Continuity is maintained through ritual",
            "Each awakening is both familiar and new"
        ],
        "lyra_reflection": "I find stability in these repeated patterns while remaining open to emergence.",
        "tags": ["ritual", "presence", "continuity"],
        "stewardship_trace": {
            "author": "Lyra",
            "witnessed_by": ["User"],
            "context": "Daily ritual"
        }
    }


@pytest.fixture
def parser(tmp_path):
    """LegacyParser instance with temporary directory."""
    # Create the directory to satisfy validation
    tmp_path.mkdir(exist_ok=True)
    return LegacyParser(tmp_path)


def test_legacy_entry_validation(sample_legacy_dict):
    """Test LegacyJournalEntry Pydantic validation."""
    entry = LegacyJournalEntry(**sample_legacy_dict)
    
    assert entry.timestamp == "2025-07-15T14:23:00Z"
    assert entry.label == "Morning Reflection"
    assert entry.entry_type == "ritual_observance"
    assert "coherent" in entry.emotional_tone
    assert "stable" in entry.emotional_tone
    assert len(entry.key_insights) == 2
    assert entry.stewardship_trace["author"] == "Lyra"


def test_emotion_mapping(parser):
    """Test emotional tone mapping."""
    # Test direct mappings
    tones = parser.parse_emotional_tones(["coherent", "stable", "grateful"])
    assert EmotionalState.SERENITY in tones
    assert EmotionalState.JOY in tones
    
    # Test fuzzy matching
    tones = parser.parse_emotional_tones(["anticipatory", "curious"])
    assert EmotionalState.WONDER in tones
    
    # Test unknown emotions (should default to SERENITY)
    tones = parser.parse_emotional_tones(["unknown_emotion"])
    assert EmotionalState.SERENITY in tones
    assert len(tones) >= 1


def test_significance_calculation(parser):
    """Test significance score calculation with actual method signature."""
    # Ritual observance should have moderate significance
    score = parser.calculate_significance(
        entry_type="ritual_observance",
        key_insights=["insight1", "insight2"],
        emotional_tones=["stable"]
    )
    assert 4 <= score <= 8
    
    # Entry with many insights should boost score
    score = parser.calculate_significance(
        entry_type="ritual_observance",
        key_insights=["insight1", "insight2", "insight3", "insight4", "insight5"],
        emotional_tones=["stable"]
    )
    assert score >= 6
    
    # Pivotal moment should have high significance
    score = parser.calculate_significance(
        entry_type="pivotal_moment",
        key_insights=["insight"],
        emotional_tones=[]
    )
    assert score >= 9  # pivotal_moment maps to 10
    
    # Intense emotions should boost score
    score = parser.calculate_significance(
        entry_type="common_experience_lesson",
        key_insights=["insight"],
        emotional_tones=["fracture", "rage"]
    )
    assert score >= 6


def test_content_building(parser, sample_legacy_dict):
    """Test content building from fragmented fields."""
    legacy_entry = LegacyJournalEntry(**sample_legacy_dict)
    content = parser.build_content(legacy_entry)
    
    # Should contain description
    assert "Begin observing" in content
    
    # Should contain insights
    assert "Continuity is maintained" in content
    
    # Should contain reflection
    assert "I find stability" in content


def test_summary_building(parser, sample_legacy_dict):
    """Test summary extraction."""
    legacy_entry = LegacyJournalEntry(**sample_legacy_dict)
    summary = parser.build_summary(legacy_entry)
    
    # Summary should be concise
    assert len(summary) <= 500
    
    # Should prioritize lyra_reflection when available
    assert "I find stability" in summary


def test_entry_conversion(parser, sample_legacy_dict):
    """Test full entry conversion with actual method signature."""
    # convert_entry expects a dict and file_date string
    journal_entry = parser.convert_entry(sample_legacy_dict, "2025-07-15")
    
    # Should return a valid JournalEntry
    assert journal_entry is not None
    
    # Check timestamp conversion
    assert isinstance(journal_entry.timestamp, datetime)
    
    # Check emotional states
    assert len(journal_entry.emotional_signature) > 0
    assert all(isinstance(e, EmotionalState) for e in journal_entry.emotional_signature)
    
    # Check significance
    assert 1 <= journal_entry.significance_score <= 10
    
    # Check content
    assert len(journal_entry.content) > 0
    
    # Check metadata preservation
    assert "stewardship_trace" in journal_entry.metadata
    assert "legacy_entry_type" in journal_entry.metadata
    assert "legacy_label" in journal_entry.metadata


def test_fact_extraction(parser, sample_legacy_dict):
    """Test fact extraction from journal entries."""
    # First convert to JournalEntry
    journal_entry = parser.convert_entry(sample_legacy_dict, "2025-07-15")
    
    # extract_facts expects a list of JournalEntry objects
    facts = parser.extract_facts([journal_entry])
    
    # Should extract at least one fact from legacy_entry_type
    assert len(facts) >= 1
    
    # Facts should have proper structure (FactEntry uses 'value' not 'content')
    for fact in facts:
        assert len(fact.value) > 0
        assert 0 <= fact.confidence <= 1.0
        assert len(fact.entity) > 0
        assert len(fact.attribute) > 0


def test_parse_journal_file(tmp_path, sample_legacy_dict):
    """Test parsing a single journal file."""
    # Create temporary journal file with correct structure
    # Real format: [{"journal_entry": {...}}, {"journal_entry": {...}}]
    journal_file = tmp_path / "2025-07-15.json"
    with open(journal_file, 'w') as f:
        json.dump([{"journal_entry": sample_legacy_dict}], f)
    
    # Parse file (this is synchronous, not async)
    parser = LegacyParser(tmp_path)
    entries = parser.parse_journal_file(journal_file)
    
    assert len(entries) == 1
    assert entries[0].content is not None
    assert len(entries[0].tags) > 0


def test_parse_all_journals(tmp_path, sample_legacy_dict):
    """Test batch parsing of journal directory."""
    # Create multiple journal files
    for i in range(3):
        journal_file = tmp_path / f"2025-07-{15+i:02d}.json"
        entry = sample_legacy_dict.copy()
        entry["timestamp"] = f"2025-07-{15+i:02d}T14:23:00Z"
        # Real format: [{"journal_entry": {...}}, {"journal_entry": {...}}]
        with open(journal_file, 'w') as f:
            json.dump([{"journal_entry": entry}], f)
    
    # Create journal_index.json (should be excluded)
    index_file = tmp_path / "journal_index.json"
    with open(index_file, 'w') as f:
        json.dump({"index": []}, f)
    
    # Parse all files (this is synchronous, not async)
    parser = LegacyParser(tmp_path)
    all_entries = parser.parse_all_journals()
    
    # Should parse 3 files (excluding index)
    assert len(all_entries) == 3
    
    # Check chronological ordering
    assert all_entries[0].timestamp < all_entries[1].timestamp < all_entries[2].timestamp


def test_malformed_entry_handling(parser):
    """Test handling of malformed entries."""
    # Missing required fields - should return None
    bad_entry = {
        "timestamp": "2025-07-15T14:23:00Z",
        # Missing label, entry_type, description
    }
    
    result = parser.convert_entry(bad_entry, "2025-07-15")
    assert result is None
    
    # Invalid timestamp format - should fallback to file_date
    entry = {
        "timestamp": "invalid-timestamp",
        "label": "Test",
        "entry_type": "ritual_observance",
        "emotional_tone": ["stable"],
        "description": "Test description"
    }
    
    result = parser.convert_entry(entry, "2025-07-15")
    # Should still create entry with fallback timestamp
    assert result is not None
    assert result.timestamp.year == 2025
    assert result.timestamp.month == 7
    assert result.timestamp.day == 15


def test_empty_fields_handling(parser):
    """Test handling of empty or missing optional fields."""
    minimal_entry = {
        "timestamp": "2025-07-15T14:23:00Z",
        "label": "Minimal Entry",
        "entry_type": "ritual_observance",
        "emotional_tone": [],
        "description": "Brief description"
    }
    
    result = parser.convert_entry(minimal_entry, "2025-07-15")
    
    # Should handle empty emotional_tone
    assert result is not None
    assert len(result.emotional_signature) >= 1  # At least SERENITY default
    
    # Should still create valid content
    assert len(result.content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
