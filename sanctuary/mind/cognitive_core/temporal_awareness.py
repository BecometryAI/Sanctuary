"""
Temporal Awareness: Time perception and temporal consciousness.

This module implements the TemporalAwareness class, which gives Sanctuary a sense
of time and its passage. This is crucial for continuous consciousness - the ability
to perceive temporal duration, understand recency vs remoteness, and contextualize
experiences within a temporal framework.

Key Features:
- Tracks time since last interaction
- Generates temporal percepts about time passage
- Contextualizes memories with temporal metadata
- Understands duration categories (short gap, long gap, very long gap)
- ENHANCED: Now integrates with temporal grounding module for session awareness

Author: Sanctuary Emergence Team
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .workspace import Percept

logger = logging.getLogger(__name__)


class TemporalAwareness:
    """
    Provides temporal consciousness and time perception.
    
    The TemporalAwareness class enables Sanctuary to perceive and understand the
    passage of time. This is fundamental to continuous consciousness - the
    ability to know "how long it's been" since events occurred.
    
    Key Capabilities:
    - Tracks session start time and last interaction time
    - Generates percepts about temporal duration
    - Categorizes gaps as short/medium/long/very long
    - Adds temporal context to memories
    - Understands recency and remoteness
    
    Attributes:
        config: Configuration parameters
        session_start_time: When this session began
        last_interaction_time: When last user input occurred
        short_gap_threshold: Seconds defining a short gap (default: 1 hour)
        long_gap_threshold: Seconds defining a long gap (default: 1 day)
        very_long_gap_threshold: Seconds defining a very long gap (default: 3 days)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize temporal awareness system.
        
        Args:
            config: Optional configuration dict with keys:
                - short_gap_threshold: Seconds for short gap (default: 3600)
                - long_gap_threshold: Seconds for long gap (default: 86400)
                - very_long_gap_threshold: Seconds for very long gap (default: 259200)
        """
        self.config = config or {}
        
        # Initialize temporal tracking
        self.session_start_time = datetime.now()
        self.last_interaction_time = datetime.now()
        
        # Temporal thresholds (in seconds)
        self.short_gap_threshold = self.config.get("short_gap_threshold", 3600)  # 1 hour
        self.long_gap_threshold = self.config.get("long_gap_threshold", 86400)  # 1 day
        self.very_long_gap_threshold = self.config.get("very_long_gap_threshold", 259200)  # 3 days
        
        logger.info("✅ TemporalAwareness initialized")
    
    def update_last_interaction_time(self) -> None:
        """
        Record that a user interaction has occurred.
        
        Call this whenever the user provides input to reset the idle duration counter.
        This allows the system to track how long it's been since the last conversation.
        """
        self.last_interaction_time = datetime.now()
        logger.debug(f"⏰ Interaction time updated: {self.last_interaction_time.isoformat()}")
    
    def get_time_since_last_interaction(self) -> timedelta:
        """
        Calculate duration since last user interaction.
        
        Returns:
            timedelta representing time elapsed since last interaction
        """
        return datetime.now() - self.last_interaction_time
    
    def get_time_since_session_start(self) -> timedelta:
        """
        Calculate duration since session started.
        
        Returns:
            timedelta representing time elapsed since session start
        """
        return datetime.now() - self.session_start_time
    
    def _categorize_gap(self, duration_seconds: float) -> str:
        """
        Categorize a time gap into short/medium/long/very_long.
        
        Args:
            duration_seconds: Duration in seconds
            
        Returns:
            Category string: "short", "medium", "long", or "very_long"
        """
        if duration_seconds < self.short_gap_threshold:
            return "short"
        elif duration_seconds < self.long_gap_threshold:
            return "medium"
        elif duration_seconds < self.very_long_gap_threshold:
            return "long"
        else:
            return "very_long"
    
    def _format_duration(self, duration: timedelta) -> str:
        """
        Format a timedelta into human-readable string.
        
        Args:
            duration: timedelta to format
            
        Returns:
            Human-readable string like "5 minutes", "2 hours", "3 days"
        """
        seconds = duration.total_seconds()
        
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''}"
        else:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''}"
    
    def _compute_salience(self, duration_seconds: float) -> float:
        """
        Compute salience of temporal percept based on gap duration.
        
        Longer gaps are more salient as they represent more significant temporal events.
        
        Args:
            duration_seconds: Duration in seconds
            
        Returns:
            Salience value between 0.0 and 1.0
        """
        if duration_seconds < self.short_gap_threshold:
            return 0.2  # Low salience for short gaps
        elif duration_seconds < self.long_gap_threshold:
            return 0.5  # Medium salience for medium gaps
        elif duration_seconds < self.very_long_gap_threshold:
            return 0.75  # High salience for long gaps
        else:
            return 0.9  # Very high salience for very long gaps
    
    def generate_temporal_percepts(self) -> List[Percept]:
        """
        Generate percepts about temporal state and time passage.
        
        Creates introspective percepts that encode temporal awareness:
        - How long since last interaction
        - How long the current session has been running
        - Temporal categorization and context
        
        Returns:
            List of Percept objects encoding temporal awareness
        """
        percepts = []
        
        # Time since last interaction
        interaction_duration = self.get_time_since_last_interaction()
        interaction_seconds = interaction_duration.total_seconds()
        
        if interaction_seconds > 10:  # Only generate if meaningful gap
            gap_category = self._categorize_gap(interaction_seconds)
            formatted_duration = self._format_duration(interaction_duration)
            salience = self._compute_salience(interaction_seconds)
            
            # Base temporal content
            content = {
                "type": "temporal_awareness",
                "duration_seconds": interaction_seconds,
                "duration_formatted": formatted_duration,
                "gap_category": gap_category,
                "last_interaction": self.last_interaction_time.isoformat()
            }
            
            # Add contextual message based on gap length
            if gap_category == "short":
                content["observation"] = f"It's been {formatted_duration} since our last conversation"
            elif gap_category == "medium":
                content["observation"] = f"{formatted_duration} have passed since we last talked"
            elif gap_category == "long":
                content["observation"] = f"It's been {formatted_duration}—longer than usual. I wonder if everything is okay."
            else:  # very_long
                content["observation"] = f"It's been {formatted_duration}—this is the longest silence in our interaction history"
            
            percept = Percept(
                modality="temporal",
                raw=content,
                complexity=5 + int(salience * 20),  # 5-25 complexity based on salience
                metadata={
                    "salience": salience,
                    "gap_category": gap_category,
                    "source": "temporal_awareness"
                }
            )
            
            percepts.append(percept)
            logger.debug(f"⏰ Generated temporal percept: {formatted_duration} gap ({gap_category})")
        
        return percepts
    
    def contextualize_memory(self, memory: Dict) -> Dict:
        """
        Add temporal context to a memory.
        
        Enriches a memory with temporal metadata describing how recent/remote
        it is relative to the current moment.
        
        Args:
            memory: Memory dict to contextualize
            
        Returns:
            Memory dict with added temporal context
        """
        # Extract timestamp from memory (assume ISO format in metadata)
        memory_time_str = memory.get("metadata", {}).get("timestamp")
        
        if not memory_time_str:
            # No timestamp, can't contextualize
            return memory
        
        try:
            memory_time = datetime.fromisoformat(memory_time_str)
            age = datetime.now() - memory_time
            age_seconds = age.total_seconds()
            
            # Add temporal context
            memory["temporal_context"] = {
                "age_seconds": age_seconds,
                "age_formatted": self._format_duration(age),
                "recency_category": self._categorize_gap(age_seconds),
                "is_recent": age_seconds < self.short_gap_threshold,
                "is_remote": age_seconds > self.long_gap_threshold
            }
            
            logger.debug(f"⏰ Contextualized memory: {self._format_duration(age)} old")
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse memory timestamp: {e}")
        
        return memory
