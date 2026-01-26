"""
Identity Continuity: Track identity stability and changes over time.

This module implements tracking of identity snapshots over time to measure
consistency and detect identity changes or drift.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class IdentitySnapshot:
    """
    Immutable snapshot of identity state at a point in time.
    
    Attributes:
        timestamp: When this snapshot was taken
        core_values: List of core values at this time
        emotional_disposition: Baseline emotional state (VAD)
        self_defining_memories: IDs of self-defining memories
        behavioral_tendencies: Behavioral tendency scores
        metadata: Additional snapshot metadata
    """
    timestamp: datetime
    core_values: List[str]
    emotional_disposition: Dict[str, float]
    self_defining_memories: List[str]  # Memory IDs
    behavioral_tendencies: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class IdentityContinuity:
    """
    Track identity stability and changes over time.
    
    This class maintains a history of identity snapshots and provides
    methods to analyze how consistent identity has been, detect changes,
    and measure continuity.
    
    Attributes:
        snapshots: List of identity snapshots over time
        max_snapshots: Maximum number of snapshots to retain
        config: Configuration dictionary
    """
    
    def __init__(self, max_snapshots: int = 100, config: Dict = None):
        """
        Initialize identity continuity tracker.
        
        Args:
            max_snapshots: Maximum snapshots to keep in history
            config: Optional configuration dictionary
        """
        if max_snapshots < 1:
            raise ValueError("max_snapshots must be at least 1")
        
        self.snapshots: List[IdentitySnapshot] = []
        self.max_snapshots = max_snapshots
        self.config = config or {}
        
        logger.debug(f"IdentityContinuity initialized (max_snapshots={max_snapshots})")
    
    def take_snapshot(self, identity: Any) -> None:
        """
        Record current identity state as a snapshot.
        
        Args:
            identity: ComputedIdentity or Identity object
        """
        # Extract snapshot data from identity
        if hasattr(identity, 'as_identity'):
            # ComputedIdentity - convert to Identity first
            identity_obj = identity.as_identity()
        else:
            # Already an Identity object
            identity_obj = identity
        
        # Create snapshot
        snapshot = IdentitySnapshot(
            timestamp=datetime.now(),
            core_values=identity_obj.core_values.copy(),
            emotional_disposition=identity_obj.emotional_disposition.copy(),
            self_defining_memories=[
                m.get('id', str(m)) if isinstance(m, dict) else str(m)
                for m in identity_obj.autobiographical_self
            ],
            behavioral_tendencies=identity_obj.behavioral_tendencies.copy(),
            metadata={
                "source": identity_obj.source,
                "snapshot_count": len(self.snapshots) + 1
            }
        )
        
        # Add to history
        self.snapshots.append(snapshot)
        
        # Trim if needed
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        
        logger.debug(f"Identity snapshot taken (total: {len(self.snapshots)})")
    
    def get_continuity_score(self) -> float:
        """
        Calculate how consistent identity has been over time.
        
        Returns:
            Continuity score from 0.0 (completely unstable) to 1.0 (perfectly stable)
        """
        if len(self.snapshots) < 2:
            return 1.0  # Perfect continuity with insufficient data
        
        # Use recent snapshots for continuity check
        recent = self.snapshots[-10:] if len(self.snapshots) >= 10 else self.snapshots
        
        # Calculate value consistency
        value_consistency = self._value_overlap(recent)
        
        # Calculate disposition stability
        disposition_consistency = self._disposition_stability(recent)
        
        # Calculate memory consistency
        memory_consistency = self._memory_consistency(recent)
        
        # Weighted average
        score = (
            value_consistency * 0.4 +
            disposition_consistency * 0.3 +
            memory_consistency * 0.3
        )
        
        logger.debug(f"Continuity score: {score:.3f} "
                    f"(values={value_consistency:.3f}, "
                    f"disposition={disposition_consistency:.3f}, "
                    f"memory={memory_consistency:.3f})")
        
        return score
    
    def _value_overlap(self, snapshots: List[IdentitySnapshot]) -> float:
        """
        Calculate consistency of core values across snapshots.
        
        Args:
            snapshots: List of snapshots to compare
            
        Returns:
            Overlap score from 0.0 to 1.0
        """
        if len(snapshots) < 2:
            return 1.0
        
        # Compare adjacent snapshots
        overlaps = []
        for i in range(len(snapshots) - 1):
            curr_values = set(snapshots[i].core_values)
            next_values = set(snapshots[i + 1].core_values)
            
            if not curr_values and not next_values:
                overlaps.append(1.0)
            elif not curr_values or not next_values:
                overlaps.append(0.0)
            else:
                # Jaccard similarity
                intersection = len(curr_values & next_values)
                union = len(curr_values | next_values)
                overlaps.append(intersection / union if union > 0 else 0.0)
        
        return sum(overlaps) / len(overlaps) if overlaps else 1.0
    
    def _disposition_stability(self, snapshots: List[IdentitySnapshot]) -> float:
        """
        Calculate stability of emotional disposition across snapshots.
        
        Args:
            snapshots: List of snapshots to compare
            
        Returns:
            Stability score from 0.0 to 1.0
        """
        if len(snapshots) < 2:
            return 1.0
        
        # Calculate variance in VAD dimensions
        valences = [s.emotional_disposition.get('valence', 0.0) for s in snapshots]
        arousals = [s.emotional_disposition.get('arousal', 0.0) for s in snapshots]
        dominances = [s.emotional_disposition.get('dominance', 0.0) for s in snapshots]
        
        # Calculate standard deviations
        import statistics
        try:
            valence_std = statistics.stdev(valences) if len(valences) > 1 else 0.0
            arousal_std = statistics.stdev(arousals) if len(arousals) > 1 else 0.0
            dominance_std = statistics.stdev(dominances) if len(dominances) > 1 else 0.0
            
            # Lower std = higher stability (normalize to 0-1)
            # Max possible std for range [-1, 1] is ~1.15, so we use that
            avg_std = (valence_std + arousal_std + dominance_std) / 3.0
            stability = max(0.0, 1.0 - (avg_std / 1.15))
            
            return stability
        except statistics.StatisticsError:
            return 1.0
    
    def _memory_consistency(self, snapshots: List[IdentitySnapshot]) -> float:
        """
        Calculate consistency of self-defining memories across snapshots.
        
        Args:
            snapshots: List of snapshots to compare
            
        Returns:
            Consistency score from 0.0 to 1.0
        """
        if len(snapshots) < 2:
            return 1.0
        
        # Compare memory overlap between adjacent snapshots
        overlaps = []
        for i in range(len(snapshots) - 1):
            curr_memories = set(snapshots[i].self_defining_memories)
            next_memories = set(snapshots[i + 1].self_defining_memories)
            
            if not curr_memories and not next_memories:
                overlaps.append(1.0)
            elif not curr_memories or not next_memories:
                overlaps.append(0.5)  # One empty, one not
            else:
                # Calculate overlap percentage
                intersection = len(curr_memories & next_memories)
                max_size = max(len(curr_memories), len(next_memories))
                overlaps.append(intersection / max_size if max_size > 0 else 0.0)
        
        return sum(overlaps) / len(overlaps) if overlaps else 1.0
    
    def get_identity_drift(self, lookback_snapshots: int = 10) -> Dict[str, Any]:
        """
        Analyze how identity has changed recently.
        
        Args:
            lookback_snapshots: Number of recent snapshots to analyze
            
        Returns:
            Dictionary describing identity drift/changes
        """
        if len(self.snapshots) < 2:
            return {
                "has_drift": False,
                "message": "Insufficient data to measure drift"
            }
        
        recent = self.snapshots[-lookback_snapshots:] if len(self.snapshots) >= lookback_snapshots else self.snapshots
        
        if len(recent) < 2:
            return {
                "has_drift": False,
                "message": "Insufficient recent data"
            }
        
        # Compare first and last in recent window
        first = recent[0]
        last = recent[-1]
        
        # Detect value changes
        first_values = set(first.core_values)
        last_values = set(last.core_values)
        added_values = last_values - first_values
        removed_values = first_values - last_values
        
        # Detect disposition changes
        disposition_change = self._compute_disposition_change(
            first.emotional_disposition,
            last.emotional_disposition
        )
        
        # Determine if significant drift occurred
        has_drift = (
            len(added_values) > 0 or
            len(removed_values) > 0 or
            disposition_change > 0.3  # Threshold for significant change
        )
        
        return {
            "has_drift": has_drift,
            "added_values": list(added_values),
            "removed_values": list(removed_values),
            "disposition_change": disposition_change,
            "continuity_score": self.get_continuity_score(),
            "snapshots_analyzed": len(recent),
            "time_span": (last.timestamp - first.timestamp).total_seconds() / 3600  # hours
        }
    
    def _compute_disposition_change(
        self,
        first: Dict[str, float],
        last: Dict[str, float]
    ) -> float:
        """
        Compute magnitude of disposition change.
        
        Args:
            first: First disposition state
            last: Last disposition state
            
        Returns:
            Change magnitude (Euclidean distance in VAD space)
        """
        import math
        
        v_diff = last.get('valence', 0.0) - first.get('valence', 0.0)
        a_diff = last.get('arousal', 0.0) - first.get('arousal', 0.0)
        d_diff = last.get('dominance', 0.0) - first.get('dominance', 0.0)
        
        # Euclidean distance
        return math.sqrt(v_diff**2 + a_diff**2 + d_diff**2)
    
    def get_recent_snapshots(self, count: int = 5) -> List[IdentitySnapshot]:
        """
        Get the most recent identity snapshots.
        
        Args:
            count: Number of snapshots to retrieve
            
        Returns:
            List of recent snapshots
        """
        return self.snapshots[-count:] if self.snapshots else []
    
    def clear_history(self) -> None:
        """Clear all snapshot history."""
        self.snapshots.clear()
        logger.info("Identity snapshot history cleared")
