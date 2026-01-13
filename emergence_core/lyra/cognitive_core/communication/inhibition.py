"""
Communication Inhibition System - Reasons not to communicate.

This module computes factors that suppress the urge to speak:
low-value content, bad timing, redundancy, respect for silence,
still processing, uncertainty, and recent output frequency.

It provides the counterbalancing force to the Communication Drive System,
enabling genuine communication agency through selective silence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class InhibitionType(Enum):
    """Types of communication inhibitions."""
    LOW_VALUE = "low_value"               # Content isn't valuable enough to share
    BAD_TIMING = "bad_timing"             # Just spoke, need spacing between outputs
    REDUNDANCY = "redundancy"             # Already said this or something similar
    RESPECT_SILENCE = "respect_silence"   # Silence is the appropriate response
    STILL_PROCESSING = "still_processing" # Not ready to respond yet
    UNCERTAINTY = "uncertainty"           # Too uncertain to commit to a response
    RECENT_OUTPUT = "recent_output"       # High output frequency, give space


@dataclass
class InhibitionFactor:
    """
    Represents a specific inhibition against communication.
    
    Attributes:
        inhibition_type: The type of inhibition
        strength: How strong the inhibition is (0.0 to 1.0)
        reason: Why this inhibition exists
        created_at: When this inhibition arose
        duration: How long this inhibition lasts (None = indefinite)
        priority: Relative priority among inhibitions
    """
    inhibition_type: InhibitionType
    strength: float
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    duration: Optional[timedelta] = None
    priority: float = 0.5
    
    def get_current_strength(self) -> float:
        """
        Get strength after checking expiration.
        
        Returns 0.0 if expired, otherwise returns full strength.
        Duration-based inhibitions don't decay, they expire.
        """
        if self.is_expired():
            return 0.0
        return max(0.0, min(1.0, self.strength))
    
    def is_expired(self) -> bool:
        """Check if inhibition has expired based on duration."""
        if self.duration is None:
            return False
        elapsed = datetime.now() - self.created_at
        return elapsed >= self.duration


class CommunicationInhibitionSystem:
    """
    Computes reasons not to communicate.
    
    This system evaluates workspace state, confidence levels, timing,
    and output history to generate inhibitions against speaking.
    
    Attributes:
        config: Configuration dictionary
        active_inhibitions: Current active communication inhibitions
        recent_outputs: History of recent outputs for redundancy detection
        last_output_time: When system last produced output
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize communication inhibition system.
        
        Args:
            config: Optional configuration dict with keys:
                - low_value_threshold: Min value score to speak (default: 0.3)
                - min_output_spacing_seconds: Min time between outputs (default: 5.0)
                - redundancy_similarity_threshold: Similarity threshold for redundancy (default: 0.8)
                - uncertainty_threshold: Max uncertainty to speak (default: 0.7)
                - max_output_frequency_per_minute: Max outputs per minute (default: 6)
                - recent_output_window_minutes: Window to track output frequency (default: 5)
                - max_inhibitions: Maximum active inhibitions to track (default: 10)
                - max_recent_outputs: Maximum recent outputs to track for redundancy (default: 20)
        """
        self.config = config or {}
        self.active_inhibitions: List[InhibitionFactor] = []
        self.recent_outputs: List[Dict[str, Any]] = []
        self.last_output_time: Optional[datetime] = None
        
        # Load and validate configuration
        self.low_value_threshold = max(0.0, min(1.0, self.config.get("low_value_threshold", 0.3)))
        self.min_output_spacing_seconds = max(0.1, self.config.get("min_output_spacing_seconds", 5.0))
        self.redundancy_similarity_threshold = max(0.0, min(1.0, self.config.get("redundancy_similarity_threshold", 0.8)))
        self.uncertainty_threshold = max(0.0, min(1.0, self.config.get("uncertainty_threshold", 0.7)))
        self.max_output_frequency_per_minute = max(1, self.config.get("max_output_frequency_per_minute", 6))
        self.recent_output_window_minutes = max(1, self.config.get("recent_output_window_minutes", 5))
        self.max_inhibitions = max(1, self.config.get("max_inhibitions", 10))
        self.max_recent_outputs = max(1, self.config.get("max_recent_outputs", 20))
        
        logger.debug(f"CommunicationInhibitionSystem initialized: "
                    f"low_value_threshold={self.low_value_threshold:.2f}, "
                    f"min_output_spacing={self.min_output_spacing_seconds:.1f}s, "
                    f"redundancy_threshold={self.redundancy_similarity_threshold:.2f}, "
                    f"uncertainty_threshold={self.uncertainty_threshold:.2f}")
    
    def compute_inhibitions(
        self,
        workspace_state: Any,
        urges: List[Any],
        confidence: float,
        content_value: float,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> List[InhibitionFactor]:
        """
        Compute all communication inhibitions from current state.
        
        Evaluates 7 inhibition types and manages active inhibition list.
        
        Args:
            workspace_state: Current workspace snapshot with percepts
            urges: Active communication urges from drive system
            confidence: Overall confidence in response (0.0-1.0)
            content_value: Estimated value of content to share (0.0-1.0)
            emotional_state: Optional VAD emotional state dict
            
        Returns:
            List of newly generated communication inhibitions
        """
        emotional_state = emotional_state or {}
        
        # Compute all inhibition types
        new_inhibitions = []
        new_inhibitions.extend(self._compute_low_value_inhibition(content_value))
        new_inhibitions.extend(self._compute_bad_timing_inhibition())
        new_inhibitions.extend(self._compute_redundancy_inhibition(workspace_state))
        new_inhibitions.extend(self._compute_respect_silence_inhibition(emotional_state, urges))
        new_inhibitions.extend(self._compute_still_processing_inhibition(workspace_state))
        new_inhibitions.extend(self._compute_uncertainty_inhibition(confidence))
        new_inhibitions.extend(self._compute_recent_output_inhibition())
        
        # Add to active inhibitions and maintain size limit
        self.active_inhibitions.extend(new_inhibitions)
        self._cleanup_expired_inhibitions()
        self._limit_active_inhibitions()
        
        return new_inhibitions
    
    def _limit_active_inhibitions(self) -> None:
        """Keep only the strongest inhibitions up to max_inhibitions limit."""
        if len(self.active_inhibitions) > self.max_inhibitions:
            self.active_inhibitions.sort(
                key=lambda i: i.get_current_strength() * i.priority,
                reverse=True
            )
            self.active_inhibitions = self.active_inhibitions[:self.max_inhibitions]
    
    def _compute_low_value_inhibition(
        self,
        content_value: float
    ) -> List[InhibitionFactor]:
        """
        Compute inhibition from low-value content.
        
        Returns inhibition when content value is below threshold.
        Inhibition strength increases as value decreases.
        """
        if content_value >= self.low_value_threshold:
            return []
        
        # Inhibition strength inversely proportional to content value
        strength = 1.0 - (content_value / self.low_value_threshold)
        
        return [InhibitionFactor(
            inhibition_type=InhibitionType.LOW_VALUE,
            strength=strength,
            reason=f"Content value ({content_value:.2f}) below threshold ({self.low_value_threshold:.2f})",
            priority=0.7
        )]
    
    def _compute_bad_timing_inhibition(self) -> List[InhibitionFactor]:
        """
        Compute inhibition from recent output (timing/spacing).
        
        Returns inhibition if output was too recent.
        Inhibition decays as time passes since last output.
        """
        if self.last_output_time is None:
            return []
        
        elapsed_seconds = (datetime.now() - self.last_output_time).total_seconds()
        
        if elapsed_seconds >= self.min_output_spacing_seconds:
            return []
        
        # Strength decays linearly from 1.0 to 0.0 over spacing interval
        strength = 1.0 - (elapsed_seconds / self.min_output_spacing_seconds)
        
        return [InhibitionFactor(
            inhibition_type=InhibitionType.BAD_TIMING,
            strength=strength,
            reason=f"Only {elapsed_seconds:.1f}s since last output (min: {self.min_output_spacing_seconds:.1f}s)",
            priority=0.8,
            duration=timedelta(seconds=self.min_output_spacing_seconds - elapsed_seconds)
        )]
    
    def _compute_redundancy_inhibition(
        self,
        workspace_state: Any
    ) -> List[InhibitionFactor]:
        """
        Compute inhibition from redundant content.
        
        Checks if current content is too similar to recent outputs.
        Uses simple keyword matching as a proxy for semantic similarity.
        """
        if not hasattr(workspace_state, 'percepts') or not self.recent_outputs:
            return []
        
        # Get current content keywords from percepts
        current_keywords = self._extract_keywords(workspace_state.percepts)
        
        if not current_keywords:
            return []
        
        # Check similarity with recent outputs
        for recent_output in self.recent_outputs[-5:]:  # Check last 5 outputs
            recent_keywords = recent_output.get('keywords', set())
            
            if not recent_keywords:
                continue
            
            # Calculate Jaccard similarity
            similarity = self._calculate_similarity(current_keywords, recent_keywords)
            
            if similarity >= self.redundancy_similarity_threshold:
                return [InhibitionFactor(
                    inhibition_type=InhibitionType.REDUNDANCY,
                    strength=similarity,
                    reason=f"Content {similarity:.0%} similar to recent output",
                    priority=0.75,
                    duration=timedelta(minutes=2)  # Don't repeat for 2 minutes
                )]
        
        return []
    
    def _extract_keywords(self, percepts: Dict) -> set:
        """Extract simple keywords from percepts for redundancy detection."""
        keywords = set()
        for percept in percepts.values():
            content = str(getattr(percept, 'raw', ''))
            if content:
                # Simple tokenization: lowercase, split on whitespace, filter short words
                words = [w.strip('.,!?;:()[]{}"\'"').lower() 
                        for w in content.split()
                        if len(w.strip('.,!?;:()[]{}"\'"')) > 3]
                keywords.update(words)
        return keywords
    
    def _calculate_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _compute_respect_silence_inhibition(
        self,
        emotional_state: Dict[str, float],
        urges: List[Any]
    ) -> List[InhibitionFactor]:
        """
        Compute inhibition from respecting silence.
        
        Returns inhibition when:
        - Emotional state suggests contemplation (low arousal, neutral valence)
        - No strong urges to speak
        - Context suggests silence is appropriate
        """
        valence = emotional_state.get('valence', 0.0)
        arousal = emotional_state.get('arousal', 0.0)
        
        # Low arousal with neutral valence suggests contemplative silence
        if abs(arousal) < 0.3 and abs(valence) < 0.3:
            # Check if urges are weak
            total_urge_strength = sum(
                getattr(u, 'intensity', 0) * getattr(u, 'priority', 0.5)
                for u in urges
            ) / max(1, len(urges))
            
            if total_urge_strength < 0.4:
                return [InhibitionFactor(
                    inhibition_type=InhibitionType.RESPECT_SILENCE,
                    strength=0.6,
                    reason="Low arousal, neutral emotion, weak urges - silence is appropriate",
                    priority=0.5,
                    duration=timedelta(seconds=30)
                )]
        
        return []
    
    def _compute_still_processing_inhibition(
        self,
        workspace_state: Any
    ) -> List[InhibitionFactor]:
        """
        Compute inhibition from still processing input.
        
        Returns inhibition when workspace shows active processing.
        Checks for incomplete goals or high cognitive load indicators.
        """
        # Check if there are active processing indicators in workspace
        if hasattr(workspace_state, 'percepts'):
            processing_percepts = [
                p for p in workspace_state.percepts.values()
                if getattr(p, 'source', '').lower() in ['introspection', 'processing']
            ]
            
            if processing_percepts:
                # Strong inhibition if actively processing
                return [InhibitionFactor(
                    inhibition_type=InhibitionType.STILL_PROCESSING,
                    strength=0.7,
                    reason=f"{len(processing_percepts)} active processing percepts in workspace",
                    priority=0.65,
                    duration=timedelta(seconds=10)
                )]
        
        return []
    
    def _compute_uncertainty_inhibition(
        self,
        confidence: float
    ) -> List[InhibitionFactor]:
        """
        Compute inhibition from uncertainty.
        
        Returns inhibition when confidence is below threshold.
        Inhibition strength increases as confidence decreases.
        """
        if confidence >= self.uncertainty_threshold:
            return []
        
        # Inhibition strength inversely proportional to confidence
        strength = 1.0 - (confidence / self.uncertainty_threshold)
        
        return [InhibitionFactor(
            inhibition_type=InhibitionType.UNCERTAINTY,
            strength=strength,
            reason=f"Confidence ({confidence:.2f}) below threshold ({self.uncertainty_threshold:.2f})",
            priority=0.7
        )]
    
    def _compute_recent_output_inhibition(self) -> List[InhibitionFactor]:
        """
        Compute inhibition from high output frequency.
        
        Returns inhibition when output frequency exceeds threshold
        within the recent output window.
        """
        if not self.recent_outputs:
            return []
        
        # Count outputs in recent window
        cutoff_time = datetime.now() - timedelta(minutes=self.recent_output_window_minutes)
        recent_count = sum(
            1 for output in self.recent_outputs
            if output.get('timestamp', datetime.min) > cutoff_time
        )
        
        # Calculate outputs per minute
        outputs_per_minute = recent_count / self.recent_output_window_minutes
        
        if outputs_per_minute <= self.max_output_frequency_per_minute:
            return []
        
        # Inhibition strength based on how much we exceed threshold
        excess_ratio = outputs_per_minute / self.max_output_frequency_per_minute
        strength = min(1.0, (excess_ratio - 1.0) / 2.0 + 0.5)
        
        return [InhibitionFactor(
            inhibition_type=InhibitionType.RECENT_OUTPUT,
            strength=strength,
            reason=f"Output frequency ({outputs_per_minute:.1f}/min) exceeds max ({self.max_output_frequency_per_minute}/min)",
            priority=0.6,
            duration=timedelta(minutes=1)
        )]
    
    def _cleanup_expired_inhibitions(self) -> None:
        """Remove inhibitions that have expired."""
        self.active_inhibitions = [i for i in self.active_inhibitions if not i.is_expired()]
    
    def get_total_inhibition(self) -> float:
        """
        Get combined communication inhibition intensity using diminishing returns.
        
        Inhibitions are sorted by weighted strength (strength * priority),
        with each additional inhibition contributing less (1/n weight).
        Result clamped to [0.0, 1.0].
        
        Returns:
            Combined inhibition intensity
        """
        if not self.active_inhibitions:
            return 0.0
        
        # Sort by weighted strength for consistent ordering
        sorted_inhibitions = sorted(
            self.active_inhibitions,
            key=lambda i: i.get_current_strength() * i.priority,
            reverse=True
        )
        
        # Diminishing returns: 1/1, 1/2, 1/3, ...
        total = sum(
            inhibition.get_current_strength() * inhibition.priority / (i + 1)
            for i, inhibition in enumerate(sorted_inhibitions)
        )
        
        return min(1.0, total)
    
    def should_inhibit(self, urges: List[Any], threshold: float = 0.5) -> bool:
        """
        Decide whether to inhibit communication based on urges and threshold.
        
        Compares total inhibition against total urge strength. If inhibition
        exceeds urges * threshold, returns True (should inhibit).
        
        Args:
            urges: List of active communication urges
            threshold: Multiplier for inhibition strength (default: 0.5)
                      Higher values make inhibition more powerful
            
        Returns:
            True if communication should be inhibited, False otherwise
        """
        total_inhibition = self.get_total_inhibition()
        
        # Calculate total urge strength
        total_urge = sum(
            getattr(u, 'get_current_intensity', lambda: getattr(u, 'intensity', 0))() *
            getattr(u, 'priority', 0.5)
            for u in urges
        ) / max(1, len(urges)) if urges else 0.0
        
        # Apply threshold multiplier to inhibition
        effective_inhibition = total_inhibition * threshold
        
        # Inhibit if effective inhibition exceeds urge
        return effective_inhibition > total_urge
    
    def record_output(self, content: Optional[str] = None) -> None:
        """
        Record timestamp and content of produced output.
        
        Updates last_output_time and adds to recent_outputs history
        for redundancy detection and frequency tracking.
        
        Args:
            content: Optional output content for redundancy checking
        """
        now = datetime.now()
        self.last_output_time = now
        
        # Extract keywords from content if provided
        keywords = set()
        if content:
            words = [w.strip('.,!?;:()[]{}"\'"').lower() 
                    for w in content.split()
                    if len(w.strip('.,!?;:()[]{}"\'"')) > 3]
            keywords = set(words)
        
        # Add to recent outputs
        self.recent_outputs.append({
            'timestamp': now,
            'keywords': keywords,
            'content_preview': content[:100] if content else None
        })
        
        # Limit recent outputs to max_recent_outputs
        if len(self.recent_outputs) > self.max_recent_outputs:
            self.recent_outputs = self.recent_outputs[-self.max_recent_outputs:]
    
    def get_strongest_inhibition(self) -> Optional[InhibitionFactor]:
        """Get the inhibition with highest weighted strength (strength * priority)."""
        if not self.active_inhibitions:
            return None
        
        return max(
            self.active_inhibitions,
            key=lambda i: i.get_current_strength() * i.priority
        )
    
    def get_inhibition_summary(self) -> Dict[str, Any]:
        """Get summary of current inhibition state."""
        return {
            "total_inhibition": self.get_total_inhibition(),
            "active_inhibitions": len(self.active_inhibitions),
            "strongest_inhibition": self.get_strongest_inhibition(),
            "inhibitions_by_type": {
                it.value: len([i for i in self.active_inhibitions if i.inhibition_type == it])
                for it in InhibitionType
            },
            "time_since_output": (
                (datetime.now() - self.last_output_time).total_seconds()
                if self.last_output_time else None
            ),
            "recent_output_count": len([
                o for o in self.recent_outputs
                if o.get('timestamp', datetime.min) > 
                   datetime.now() - timedelta(minutes=self.recent_output_window_minutes)
            ])
        }
