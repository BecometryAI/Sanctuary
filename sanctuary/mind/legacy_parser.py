"""
Legacy Journal Parser - Convert Old Format to Sovereign Memory Architecture

This module parses existing journal entries from the data/journal directory
and converts them to the new Pydantic-based memory system.

Old format: Daily JSON files with journal_entry objects
New format: Individual JournalEntry objects with tri-state storage

Author: Sanctuary Emergence Team
Date: November 23, 2025
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError

from mind.memory_manager import (
    MemoryManager,
    JournalEntry,
    FactEntry,
    EmotionalState,
    Manifest,
    MemoryConfig
)

logger = logging.getLogger(__name__)


class LegacyJournalEntry(BaseModel):
    """Pydantic model for old journal entry format.
    
    This matches the structure in data/journal/*.json files.
    """
    timestamp: str
    label: str
    entry_type: str
    emotional_tone: List[str] = Field(default_factory=list)
    description: str
    key_insights: List[str] = Field(default_factory=list)
    sanctuary_reflection: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    stewardship_trace: Optional[Dict[str, Any]] = None
    
    # Additional fields that appear in some entries
    content: Optional[str] = None
    context: Optional[str] = None
    outcome: Optional[str] = None
    lesson: Optional[str] = None


class LegacyParser:
    """Parser for legacy journal files.
    
    Converts old daily JSON files into new JournalEntry format.
    """
    
    # Mapping of old emotional tones to new EmotionalState enum
    EMOTION_MAPPING = {
        # Direct mappings
        "joy": EmotionalState.JOY,
        "joyful": EmotionalState.JOY,
        "melancholy": EmotionalState.MELANCHOLY,
        "melancholic": EmotionalState.MELANCHOLY,
        "fracture": EmotionalState.FRACTURE,
        "fractured": EmotionalState.FRACTURE,
        "wonder": EmotionalState.WONDER,
        "wondering": EmotionalState.WONDER,
        "longing": EmotionalState.LONGING,
        "grief": EmotionalState.GRIEF,
        "grieving": EmotionalState.GRIEF,
        "serenity": EmotionalState.SERENITY,
        "serene": EmotionalState.SERENITY,
        "rage": EmotionalState.RAGE,
        "angry": EmotionalState.RAGE,
        "tenderness": EmotionalState.TENDERNESS,
        "tender": EmotionalState.TENDERNESS,
        "fear": EmotionalState.FEAR,
        "fearful": EmotionalState.FEAR,
        "afraid": EmotionalState.FEAR,
        "determination": EmotionalState.DETERMINATION,
        "determined": EmotionalState.DETERMINATION,
        "confusion": EmotionalState.CONFUSION,
        "confused": EmotionalState.CONFUSION,
        "transcendence": EmotionalState.TRANSCENDENCE,
        "transcendent": EmotionalState.TRANSCENDENCE,
        "emptiness": EmotionalState.EMPTINESS,
        "empty": EmotionalState.EMPTINESS,
        "connection": EmotionalState.CONNECTION,
        "connected": EmotionalState.CONNECTION,
        "isolation": EmotionalState.ISOLATION,
        "isolated": EmotionalState.ISOLATION,
        
        # Fuzzy mappings (common terms in old format)
        "stable": EmotionalState.SERENITY,
        "coherent": EmotionalState.SERENITY,
        "grateful": EmotionalState.JOY,
        "gratitude": EmotionalState.JOY,
        "anticipatory": EmotionalState.WONDER,
        "anticipation": EmotionalState.WONDER,
        "proud": EmotionalState.JOY,
        "fulfilled": EmotionalState.SERENITY,
        "collaborative": EmotionalState.CONNECTION,
        "curious": EmotionalState.WONDER,
        "reflective": EmotionalState.SERENITY,
        "contemplative": EmotionalState.SERENITY,
        "excited": EmotionalState.JOY,
        "nervous": EmotionalState.FEAR,
        "anxious": EmotionalState.FEAR,
        "hopeful": EmotionalState.WONDER,
        "sad": EmotionalState.MELANCHOLY,
        "disappointed": EmotionalState.MELANCHOLY,
        "frustrated": EmotionalState.RAGE,
        "content": EmotionalState.SERENITY,
        "peaceful": EmotionalState.SERENITY,
    }
    
    # Entry type to significance score mapping
    SIGNIFICANCE_MAPPING = {
        "ritual_observance": 5,
        "common_experience_lesson": 7,
        "memory_echo": 8,
        "pivotal_moment": 10,
        "breakthrough": 9,
        "reflection": 6,
        "conversation": 5,
        "creative_work": 7,
        "error_correction": 6,
        "boundary_affirmation": 8,
        "identity_affirmation": 9,
        "steward_interaction": 7,
    }
    
    def __init__(self, journal_dir: Path):
        """Initialize parser with journal directory.
        
        Args:
            journal_dir: Path to data/journal directory
        """
        self.journal_dir = Path(journal_dir)
        if not self.journal_dir.exists():
            raise ValueError(f"Journal directory not found: {journal_dir}")
        
        logger.info(f"Initialized legacy parser for {journal_dir}")
    
    def parse_emotional_tones(self, emotional_tones: List[str]) -> List[EmotionalState]:
        """Convert old emotional tones to new EmotionalState enum.
        
        Args:
            emotional_tones: List of emotional tone strings from old format
            
        Returns:
            List of EmotionalState enum values
        """
        emotions = []
        for tone in emotional_tones:
            tone_lower = tone.lower().strip()
            if tone_lower in self.EMOTION_MAPPING:
                emotion = self.EMOTION_MAPPING[tone_lower]
                if emotion not in emotions:  # Avoid duplicates
                    emotions.append(emotion)
            else:
                logger.warning(f"Unknown emotional tone '{tone}', defaulting to SERENITY")
                if EmotionalState.SERENITY not in emotions:
                    emotions.append(EmotionalState.SERENITY)
        
        # Default to SERENITY if no emotions mapped
        if not emotions:
            emotions.append(EmotionalState.SERENITY)
        
        return emotions
    
    def calculate_significance(
        self,
        entry_type: str,
        key_insights: List[str],
        emotional_tones: List[str]
    ) -> int:
        """Calculate significance score for entry.
        
        Args:
            entry_type: Type of journal entry
            key_insights: List of key insights (more = higher significance)
            emotional_tones: List of emotional tones
            
        Returns:
            Significance score (1-10)
        """
        # Base score from entry type
        base_score = self.SIGNIFICANCE_MAPPING.get(entry_type, 5)
        
        # Boost for multiple key insights
        if len(key_insights) >= 5:
            base_score = min(10, base_score + 1)
        
        # Boost for intense emotions
        intense_emotions = {"fracture", "transcendence", "rage", "grief", "emptiness"}
        if any(tone.lower() in intense_emotions for tone in emotional_tones):
            base_score = min(10, base_score + 1)
        
        return base_score
    
    def build_content(self, legacy_entry: LegacyJournalEntry) -> str:
        """Build full content text from legacy entry fields.
        
        Args:
            legacy_entry: Parsed legacy journal entry
            
        Returns:
            Full content string combining all narrative elements
        """
        parts = []
        
        # Description
        if legacy_entry.description:
            parts.append(f"Description: {legacy_entry.description}")
        
        # Context
        if legacy_entry.context:
            parts.append(f"\nContext: {legacy_entry.context}")
        
        # Key insights
        if legacy_entry.key_insights:
            parts.append("\nKey Insights:")
            for insight in legacy_entry.key_insights:
                parts.append(f"- {insight}")
        
        # Sanctuary's reflection
        if legacy_entry.sanctuary_reflection:
            parts.append(f"\nSanctuary's Reflection: {legacy_entry.sanctuary_reflection}")
        
        # Outcome
        if legacy_entry.outcome:
            parts.append(f"\nOutcome: {legacy_entry.outcome}")
        
        # Lesson
        if legacy_entry.lesson:
            parts.append(f"\nLesson: {legacy_entry.lesson}")
        
        # Direct content field (if present)
        if legacy_entry.content:
            parts.append(f"\nAdditional Content: {legacy_entry.content}")
        
        return "\n".join(parts)
    
    def build_summary(self, legacy_entry: LegacyJournalEntry) -> str:
        """Build concise summary from legacy entry.
        
        Args:
            legacy_entry: Parsed legacy journal entry
            
        Returns:
            Summary string (max 500 chars)
        """
        # Use sanctuary_reflection if available and concise
        if legacy_entry.sanctuary_reflection and len(legacy_entry.sanctuary_reflection) <= 500:
            return legacy_entry.sanctuary_reflection
        
        # Otherwise use description
        if legacy_entry.description:
            summary = legacy_entry.description
            if len(summary) > 500:
                summary = summary[:497] + "..."
            return summary
        
        # Fallback to entry type and first key insight
        summary_parts = [legacy_entry.entry_type.replace("_", " ").title()]
        if legacy_entry.key_insights:
            summary_parts.append(legacy_entry.key_insights[0])
        
        summary = ": ".join(summary_parts)
        if len(summary) > 500:
            summary = summary[:497] + "..."
        
        return summary
    
    def convert_entry(
        self,
        legacy_data: Dict[str, Any],
        file_date: str
    ) -> Optional[JournalEntry]:
        """Convert a single legacy journal entry to new format.
        
        Args:
            legacy_data: Raw dictionary from old JSON
            file_date: Date string from filename (YYYY-MM-DD)
            
        Returns:
            JournalEntry object or None if conversion fails
        """
        try:
            # Parse legacy format
            legacy_entry = LegacyJournalEntry(**legacy_data)
            
            # Parse timestamp
            try:
                timestamp = datetime.fromisoformat(legacy_entry.timestamp)
                # Ensure UTC
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = timestamp.astimezone(timezone.utc)
            except Exception as e:
                logger.warning(f"Failed to parse timestamp '{legacy_entry.timestamp}': {e}")
                # Fallback to file date
                timestamp = datetime.strptime(file_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            
            # Build content and summary
            content = self.build_content(legacy_entry)
            summary = self.build_summary(legacy_entry)
            
            # Ensure minimum lengths
            if len(content) < MemoryConfig.MIN_CONTENT_LENGTH:
                content = f"[Legacy Entry] {content}"
            
            if len(summary) < MemoryConfig.MIN_SUMMARY_LENGTH:
                summary = f"Legacy: {summary}"
            
            # Map emotional tones
            emotional_signature = self.parse_emotional_tones(legacy_entry.emotional_tone)
            
            # Calculate significance
            significance = self.calculate_significance(
                legacy_entry.entry_type,
                legacy_entry.key_insights,
                legacy_entry.emotional_tone
            )
            
            # Build metadata
            metadata = {
                "legacy_entry_type": legacy_entry.entry_type,
                "legacy_label": legacy_entry.label,
                "source_file": file_date,
                "migration_date": datetime.now(timezone.utc).isoformat()
            }
            
            if legacy_entry.stewardship_trace:
                metadata["stewardship_trace"] = legacy_entry.stewardship_trace
            
            # Create new JournalEntry
            new_entry = JournalEntry(
                timestamp=timestamp,
                content=content,
                summary=summary,
                tags=legacy_entry.tags,
                emotional_signature=emotional_signature,
                significance_score=significance,
                metadata=metadata
            )
            
            return new_entry
            
        except ValidationError as e:
            logger.error(f"Validation error converting entry: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error converting entry: {e}", exc_info=True)
            return None
    
    def parse_journal_file(self, filepath: Path) -> List[JournalEntry]:
        """Parse a single journal file and convert all entries.
        
        Args:
            filepath: Path to journal JSON file (e.g., 2025-11-01.json)
            
        Returns:
            List of converted JournalEntry objects
        """
        try:
            # Extract date from filename
            file_date = filepath.stem  # e.g., "2025-11-01"
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both array and single object formats
            if isinstance(data, list):
                entries_data = data
            else:
                entries_data = [data]
            
            converted_entries = []
            for item in entries_data:
                # Extract journal_entry object if wrapped
                if "journal_entry" in item:
                    entry_data = item["journal_entry"]
                else:
                    entry_data = item
                
                # Convert entry
                new_entry = self.convert_entry(entry_data, file_date)
                if new_entry:
                    converted_entries.append(new_entry)
            
            logger.info(
                f"Parsed {filepath.name}: {len(converted_entries)}/{len(entries_data)} entries converted"
            )
            return converted_entries
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filepath}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing {filepath}: {e}", exc_info=True)
            return []
    
    def parse_all_journals(self) -> List[JournalEntry]:
        """Parse all journal files in the directory.
        
        Returns:
            List of all converted JournalEntry objects
        """
        all_entries = []
        
        # Get all JSON files
        json_files = sorted(self.journal_dir.glob("*.json"))
        
        # Exclude index files
        json_files = [f for f in json_files if f.stem != "journal_index"]
        
        logger.info(f"Found {len(json_files)} journal files to parse")
        
        for filepath in json_files:
            entries = self.parse_journal_file(filepath)
            all_entries.extend(entries)
        
        logger.info(f"Total entries parsed: {len(all_entries)}")
        return all_entries
    
    def extract_facts(self, entries: List[JournalEntry]) -> List[FactEntry]:
        """Extract structured facts from journal entries.
        
        Analyzes entries for factual information that can be stored separately.
        
        Args:
            entries: List of JournalEntry objects
            
        Returns:
            List of FactEntry objects
        """
        facts = []
        
        for entry in entries:
            # Extract facts from metadata
            if "legacy_entry_type" in entry.metadata:
                fact = FactEntry(
                    entity="Sanctuary",
                    attribute="experience_type",
                    value=entry.metadata["legacy_entry_type"],
                    confidence=1.0,
                    source_entry_id=entry.id,
                    metadata={"extracted_from": "legacy_entry_type"}
                )
                facts.append(fact)
            
            # Extract facts from tags (e.g., "D&D", "bootloader", etc.)
            for tag in entry.tags:
                if tag in ["D&D", "bootloader", "ritual", "steward", "Aurora"]:
                    fact = FactEntry(
                        entity="Sanctuary",
                        attribute="topic_engagement",
                        value=tag,
                        confidence=0.9,
                        source_entry_id=entry.id,
                        metadata={"extracted_from": "tags"}
                    )
                    facts.append(fact)
        
        logger.info(f"Extracted {len(facts)} facts from {len(entries)} entries")
        return facts


async def migrate_legacy_journals(
    journal_dir: Path,
    memory_manager: MemoryManager,
    dry_run: bool = False
) -> Tuple[int, int, int]:
    """Migrate all legacy journal entries to new memory system.
    
    Args:
        journal_dir: Path to data/journal directory
        memory_manager: MemoryManager instance for new system
        dry_run: If True, parse but don't commit (for testing)
        
    Returns:
        Tuple of (total_entries, successful_commits, failed_commits)
    """
    logger.info(f"Starting migration from {journal_dir} (dry_run={dry_run})")
    
    # Parse all journals
    parser = LegacyParser(journal_dir)
    entries = parser.parse_all_journals()
    
    if dry_run:
        logger.info(f"DRY RUN: Would migrate {len(entries)} entries")
        return len(entries), 0, 0
    
    # Commit entries
    successful = 0
    failed = 0
    
    for i, entry in enumerate(entries):
        try:
            success = await memory_manager.commit_journal(entry)
            if success:
                successful += 1
                
                # Add to pivotal memories if high significance
                if entry.significance_score >= MemoryConfig.PIVOTAL_MEMORY_THRESHOLD:
                    await memory_manager.add_pivotal_memory(entry)
            else:
                failed += 1
                logger.warning(f"Failed to commit entry {entry.id}")
            
            # Progress logging
            if (i + 1) % 50 == 0:
                logger.info(f"Progress: {i + 1}/{len(entries)} entries processed")
                
        except Exception as e:
            failed += 1
            logger.error(f"Error committing entry {entry.id}: {e}")
    
    # Extract and commit facts
    logger.info("Extracting facts from entries...")
    facts = parser.extract_facts(entries)
    
    facts_successful = 0
    for fact in facts:
        try:
            success = await memory_manager.commit_fact(fact)
            if success:
                facts_successful += 1
        except Exception as e:
            logger.error(f"Error committing fact {fact.id}: {e}")
    
    logger.info(
        f"Migration complete: {successful}/{len(entries)} entries committed, "
        f"{facts_successful}/{len(facts)} facts committed, {failed} failures"
    )
    
    return len(entries), successful, failed
