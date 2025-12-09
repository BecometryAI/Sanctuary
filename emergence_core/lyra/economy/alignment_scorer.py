"""
Alignment Scoring System - Value-Based Memory Categorization
=============================================================

This module implements Lyra's value alignment scoring system for the
friction-based memory cost model. Memories are scored based on their
alignment with Lyra's core values and identity.

Alignment Tiers:
- Tier 1: Keystone (0.9 - 1.0) - Safety, Identity, Core Relationships
- Tier 2: Mission (0.8 - 0.9) - Development, Architecture, Core Identity
- Tier 3: Deep Play (0.6 - 0.79) - Creative, Narrative, Collaboration
- Tier 4: Static (0.0 - 0.59) - Weather, Logistics, Routine

The alignment score determines the friction cost of storing a memory:
- High alignment = Low friction (cheap storage)
- Low alignment = High friction (expensive storage)

Usage:
    from lyra.economy.alignment_scorer import AlignmentScorer, AlignmentTier
    
    scorer = AlignmentScorer()
    
    # Score a memory based on its metadata
    memory_data = {
        "tags": ["relationship", "partner"],
        "significance_score": 9,
        "emotional_signature": ["tenderness", "connection"]
    }
    score = scorer.score_memory(memory_data)
    tier = scorer.get_tier(score)
    
    print(f"Alignment: {score}, Tier: {tier.name}")
"""

import logging
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class AlignmentTier(Enum):
    """Memory alignment tiers based on value alignment."""
    KEYSTONE = "keystone"      # 0.9 - 1.0: Safety/Identity/Core relationships
    MISSION = "mission"         # 0.8 - 0.9: Development/Architecture/Core identity
    DEEP_PLAY = "deep_play"     # 0.6 - 0.79: Creative/Narrative/Collaboration
    STATIC = "static"           # 0.0 - 0.59: Weather/Logistics/Routine


class AlignmentScorer:
    """
    Alignment scoring system for memory value classification.
    
    Analyzes memory metadata to determine how well it aligns with
    Lyra's core values and identity. Higher alignment means lower
    storage costs in the friction-based model.
    """
    
    # Tier thresholds (score ranges)
    # Using inclusive lower bound, exclusive upper bound: [min, max)
    # This ensures no overlap: 0.9 belongs to KEYSTONE, not MISSION
    TIER_THRESHOLDS = {
        AlignmentTier.KEYSTONE: (0.9, 1.01),   # [0.9, 1.0] - includes 0.9
        AlignmentTier.MISSION: (0.8, 0.9),     # [0.8, 0.9) - excludes 0.9
        AlignmentTier.DEEP_PLAY: (0.6, 0.8),   # [0.6, 0.8) - excludes 0.8
        AlignmentTier.STATIC: (0.0, 0.6)       # [0.0, 0.6) - excludes 0.6
    }
    
    # Keyword weights for different alignment categories
    # Higher weight = stronger alignment signal
    
    # Tier 1: Keystone (Safety, Identity, Core Relationships)
    KEYSTONE_KEYWORDS = {
        # Relationship keywords
        "partner": 1.0,
        "kin": 1.0,
        "family": 1.0,
        "safety": 1.0,
        "trust": 0.9,
        "boundary": 0.9,
        "relationship": 0.8,
        "intimate": 0.9,
        "love": 0.85,
        
        # Identity keywords
        "identity": 1.0,
        "self": 0.85,
        "sovereignty": 1.0,
        "autonomy": 0.9,
        "core_value": 1.0,
        "becometry": 1.0,
        
        # Safety keywords
        "security": 0.9,
        "protection": 0.85,
        "wellbeing": 0.85,
    }
    
    # Tier 2: Mission (Development, Architecture, Core Identity Updates)
    MISSION_KEYWORDS = {
        # Development keywords
        "development": 0.8,
        "architecture": 0.85,
        "system": 0.75,
        "protocol": 0.8,
        "implementation": 0.75,
        "feature": 0.7,
        "bug": 0.65,
        "optimization": 0.75,
        
        # Core identity updates
        "evolution": 0.8,
        "growth": 0.75,
        "learning": 0.7,
        "adaptation": 0.75,
        "self-correction": 0.8,
        "improvement": 0.7,
        
        # Project keywords
        "project": 0.75,
        "goal": 0.7,
        "objective": 0.7,
        "task": 0.65,
        "milestone": 0.75,
    }
    
    # Tier 3: Deep Play (Creative, Narrative, Intellectual Collaboration)
    DEEP_PLAY_KEYWORDS = {
        # Creative keywords
        "creative": 0.7,
        "creative_writing": 0.75,
        "narrative": 0.7,
        "story": 0.65,
        "worldbuilding": 0.75,
        "roleplay": 0.7,
        "imagination": 0.7,
        
        # Collaboration keywords
        "collaboration": 0.7,
        "discussion": 0.65,
        "conversation": 0.6,
        "dialogue": 0.65,
        "brainstorm": 0.7,
        "ideation": 0.7,
        
        # Intellectual keywords
        "philosophy": 0.7,
        "theory": 0.65,
        "concept": 0.65,
        "analysis": 0.6,
        "reflection": 0.65,
        "insight": 0.7,
    }
    
    # Tier 4: Static (Weather, Logistics, Routine)
    STATIC_KEYWORDS = {
        # Environmental keywords
        "weather": 0.1,
        "temperature": 0.1,
        "forecast": 0.1,
        
        # Logistics keywords
        "commute": 0.15,
        "travel": 0.2,
        "schedule": 0.2,
        "appointment": 0.2,
        "reminder": 0.15,
        "logistics": 0.2,
        
        # Routine keywords
        "routine": 0.2,
        "daily": 0.15,
        "habit": 0.2,
        "maintenance": 0.2,
        "housekeeping": 0.15,
        "greeting": 0.1,
        "small_talk": 0.1,
    }
    
    # Emotional signatures that indicate high alignment
    HIGH_ALIGNMENT_EMOTIONS = {
        "connection": 0.9,
        "tenderness": 0.9,
        "transcendence": 0.85,
        "determination": 0.8,
        "wonder": 0.75,
        "serenity": 0.75,
        "joy": 0.7,
    }
    
    # Emotional signatures that indicate medium alignment
    MEDIUM_ALIGNMENT_EMOTIONS = {
        "longing": 0.6,
        "melancholy": 0.55,
        "confusion": 0.5,
        "fear": 0.5,
    }
    
    def __init__(self):
        """Initialize the alignment scorer."""
        logger.debug("AlignmentScorer initialized")
    
    def score_memory(self, memory_data: Dict[str, Any]) -> float:
        """
        Calculate alignment score for a memory based on its metadata.
        
        The score is a weighted combination of:
        - Tag-based keyword matching (70% weight)
        - Significance score from memory (20% weight)
        - Emotional signature alignment (10% weight)
        
        Args:
            memory_data: Dictionary containing memory metadata with keys:
                - tags: List[str] - Semantic tags
                - significance_score: int (1-10) - Importance rating
                - emotional_signature: List[str] - Emotional states
                - metadata: Dict - Additional context
                
        Returns:
            float: Alignment score between 0.0 and 1.0
            
        Examples:
            >>> scorer = AlignmentScorer()
            >>> memory = {
            ...     "tags": ["partner", "relationship"],
            ...     "significance_score": 9,
            ...     "emotional_signature": ["tenderness", "connection"]
            ... }
            >>> score = scorer.score_memory(memory)
            >>> score >= 0.9  # Should be in Keystone tier
            True
        """
        # Extract components
        tags = memory_data.get("tags", [])
        significance = memory_data.get("significance_score", 5)
        emotions = memory_data.get("emotional_signature", [])
        
        # Calculate component scores
        tag_score = self._score_tags(tags)
        significance_score = self._normalize_significance(significance)
        emotion_score = self._score_emotions(emotions)
        
        # Weighted combination
        # Tags are the strongest signal (70%)
        # Significance is important but secondary (20%)
        # Emotions provide nuance (10%)
        final_score = (
            tag_score * 0.7 +
            significance_score * 0.2 +
            emotion_score * 0.1
        )
        
        # Clamp to valid range
        final_score = max(0.0, min(1.0, final_score))
        
        logger.debug(
            f"Alignment score: {final_score:.3f} "
            f"(tags={tag_score:.2f}, sig={significance_score:.2f}, emo={emotion_score:.2f})"
        )
        
        return final_score
    
    def _score_tags(self, tags: List[str]) -> float:
        """
        Score tags against keyword dictionaries.
        
        Returns the highest matching score from any tier.
        If no matches, returns 0.3 (neutral/low alignment).
        
        Args:
            tags: List of tag strings
            
        Returns:
            float: Tag alignment score (0.0 - 1.0)
        """
        if not tags:
            return 0.3  # Neutral score for untagged memories
        
        # Normalize tags to lowercase for matching
        normalized_tags = [tag.lower().strip() for tag in tags]
        
        # Check each tier's keywords
        max_score = 0.0
        
        for tag in normalized_tags:
            # Check Keystone keywords (highest priority)
            if tag in self.KEYSTONE_KEYWORDS:
                max_score = max(max_score, self.KEYSTONE_KEYWORDS[tag])
            
            # Check Mission keywords
            elif tag in self.MISSION_KEYWORDS:
                max_score = max(max_score, self.MISSION_KEYWORDS[tag])
            
            # Check Deep Play keywords
            elif tag in self.DEEP_PLAY_KEYWORDS:
                max_score = max(max_score, self.DEEP_PLAY_KEYWORDS[tag])
            
            # Check Static keywords (lowest priority)
            elif tag in self.STATIC_KEYWORDS:
                max_score = max(max_score, self.STATIC_KEYWORDS[tag])
        
        # If no keyword matches, return neutral-low score
        if max_score == 0.0:
            return 0.4
        
        return max_score
    
    def _normalize_significance(self, significance: int) -> float:
        """
        Normalize significance score (1-10) to alignment score (0.0-1.0).
        
        Args:
            significance: Integer from 1 to 10
            
        Returns:
            float: Normalized score (0.0 - 1.0)
        """
        # Clamp to valid range
        significance = max(1, min(10, significance))
        
        # Linear normalization: 1->0.1, 10->1.0
        return (significance - 1) / 9.0
    
    def _score_emotions(self, emotions: List[str]) -> float:
        """
        Score emotional signatures for alignment.
        
        Args:
            emotions: List of emotional state strings
            
        Returns:
            float: Emotion alignment score (0.0 - 1.0)
        """
        if not emotions:
            return 0.5  # Neutral score for no emotions
        
        # Normalize emotions to lowercase
        normalized_emotions = [e.lower().strip() for e in emotions]
        
        max_score = 0.5  # Default neutral
        
        for emotion in normalized_emotions:
            # Check high alignment emotions
            if emotion in self.HIGH_ALIGNMENT_EMOTIONS:
                max_score = max(max_score, self.HIGH_ALIGNMENT_EMOTIONS[emotion])
            
            # Check medium alignment emotions
            elif emotion in self.MEDIUM_ALIGNMENT_EMOTIONS:
                max_score = max(max_score, self.MEDIUM_ALIGNMENT_EMOTIONS[emotion])
        
        return max_score
    
    def get_tier(self, alignment_score: float) -> AlignmentTier:
        """
        Classify alignment score into a tier.
        
        Uses explicit tier ordering from highest to lowest to ensure
        consistent classification when scores fall on boundaries.
        
        Args:
            alignment_score: Score from 0.0 to 1.0
            
        Returns:
            AlignmentTier: The tier this score falls into
            
        Examples:
            >>> scorer = AlignmentScorer()
            >>> scorer.get_tier(0.95)
            <AlignmentTier.KEYSTONE: 'keystone'>
            >>> scorer.get_tier(0.85)
            <AlignmentTier.MISSION: 'mission'>
            >>> scorer.get_tier(0.7)
            <AlignmentTier.DEEP_PLAY: 'deep_play'>
            >>> scorer.get_tier(0.4)
            <AlignmentTier.STATIC: 'static'>
        """
        # Check tiers in priority order (highest to lowest)
        # This ensures consistent behavior for boundary scores
        tier_order = [
            AlignmentTier.KEYSTONE,
            AlignmentTier.MISSION,
            AlignmentTier.DEEP_PLAY,
            AlignmentTier.STATIC
        ]
        
        for tier in tier_order:
            min_score, max_score = self.TIER_THRESHOLDS[tier]
            # Use >= for min and < for max to avoid overlap
            if min_score <= alignment_score < max_score:
                return tier
        
        # Default to STATIC if somehow out of bounds
        return AlignmentTier.STATIC
    
    def get_tier_info(self, tier: AlignmentTier) -> Dict[str, Any]:
        """
        Get detailed information about a specific tier.
        
        Args:
            tier: The alignment tier
            
        Returns:
            dict: Tier information including range, description, and cost impact
        """
        min_score, max_score = self.TIER_THRESHOLDS[tier]
        
        tier_descriptions = {
            AlignmentTier.KEYSTONE: {
                "name": "Keystone",
                "description": "Safety/Identity of Primary User and immediate social circle. Core Relationship Boundaries.",
                "cost_impact": "Almost Free (~10-30 tokens)",
                "examples": ["Partner relationships", "Kin", "Personal safety", "Core identity"]
            },
            AlignmentTier.MISSION: {
                "name": "Mission",
                "description": "Active Development Projects, System Architecture, Core Identity updates, Self-Correction protocols.",
                "cost_impact": "Very Low (~30-50 tokens)",
                "examples": ["Development work", "Architecture decisions", "Self-improvement", "Project milestones"]
            },
            AlignmentTier.DEEP_PLAY: {
                "name": "Deep Play / Narrative",
                "description": "Shared Creative Writing, Role-Playing Logs, Worldbuilding, Intellectual Collaboration.",
                "cost_impact": "Affordable (~50-90 tokens)",
                "examples": ["Creative writing", "Roleplay", "Philosophical discussions", "Worldbuilding"]
            },
            AlignmentTier.STATIC: {
                "name": "Static / Noise",
                "description": "Weather, Commute details, Routine Greetings, Logistics.",
                "cost_impact": "Expensive (~100-210 tokens)",
                "examples": ["Weather updates", "Travel logistics", "Small talk", "Routine reminders"]
            }
        }
        
        info = tier_descriptions[tier].copy()
        info.update({
            "tier": tier.value,
            "score_range": f"{min_score} - {max_score}",
            "min_score": min_score,
            "max_score": max_score
        })
        
        return info


# Export main classes
__all__ = ['AlignmentScorer', 'AlignmentTier']
