"""
Identity Continuity: Track identity stability and changes over time.

This module implements tracking of identity snapshots over time to measure
consistency and detect identity changes or drift. Snapshots are persisted
to disk as JSONL for long-term identity evolution tracking.
"""

from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class IdentityEvolutionEvent:
    """Records a detected identity change and what caused it."""
    timestamp: datetime
    event_type: str  # "value_added", "value_removed", "disposition_shift", "tendency_change"
    description: str
    old_value: Any = None
    new_value: Any = None
    trigger: str = ""  # what caused the change (e.g. "behavior_pattern", "emotional_shift")


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

        # Evolution event log
        self.evolution_events: List[IdentityEvolutionEvent] = []

        # Disk persistence (only when explicitly configured)
        persistence_dir = self.config.get("persistence_dir")
        self._persistence_enabled = persistence_dir is not None
        self._persistence_dir = Path(persistence_dir) if persistence_dir else Path("data/identity/evolution")
        self._snapshot_file = self._persistence_dir / "snapshots.jsonl"
        self._events_file = self._persistence_dir / "events.jsonl"
        if self._persistence_enabled:
            self._ensure_persistence_dir()
            self._load_persisted_snapshots()

        logger.debug(f"IdentityContinuity initialized (max_snapshots={max_snapshots})")

    def _ensure_persistence_dir(self) -> None:
        """Create persistence directory if it doesn't exist."""
        try:
            self._persistence_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.debug(f"Could not create persistence dir: {e}")

    def _load_persisted_snapshots(self) -> None:
        """Load previously persisted snapshots from disk."""
        if not self._snapshot_file.exists():
            return
        try:
            with open(self._snapshot_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    snapshot = IdentitySnapshot(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        core_values=data.get("core_values", []),
                        emotional_disposition=data.get("emotional_disposition", {}),
                        self_defining_memories=data.get("self_defining_memories", []),
                        behavioral_tendencies=data.get("behavioral_tendencies", {}),
                        metadata=data.get("metadata", {}),
                    )
                    self.snapshots.append(snapshot)
            # Trim to max
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
            logger.info(f"Loaded {len(self.snapshots)} identity snapshots from disk")
        except Exception as e:
            logger.debug(f"Could not load persisted snapshots: {e}")

    def _persist_snapshot(self, snapshot: IdentitySnapshot) -> None:
        """Append a snapshot to the JSONL file."""
        if not self._persistence_enabled:
            return
        try:
            data = {
                "timestamp": snapshot.timestamp.isoformat(),
                "core_values": snapshot.core_values,
                "emotional_disposition": snapshot.emotional_disposition,
                "self_defining_memories": snapshot.self_defining_memories,
                "behavioral_tendencies": snapshot.behavioral_tendencies,
                "metadata": snapshot.metadata,
            }
            with open(self._snapshot_file, 'a') as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.debug(f"Could not persist snapshot: {e}")

    def _persist_event(self, event: IdentityEvolutionEvent) -> None:
        """Append an evolution event to the JSONL file."""
        if not self._persistence_enabled:
            return
        try:
            data = {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "description": event.description,
                "old_value": event.old_value,
                "new_value": event.new_value,
                "trigger": event.trigger,
            }
            with open(self._events_file, 'a') as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.debug(f"Could not persist event: {e}")
    
    def take_snapshot(self, identity: Any, trigger: str = "") -> None:
        """
        Record current identity state as a snapshot.

        Detects changes from the previous snapshot and logs evolution events.

        Args:
            identity: ComputedIdentity or Identity object
            trigger: What caused this snapshot (e.g. "periodic", "behavior_pattern")
        """
        # Extract snapshot data from identity
        if hasattr(identity, 'as_identity'):
            identity_obj = identity.as_identity()
        else:
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
                "snapshot_count": len(self.snapshots) + 1,
                "trigger": trigger,
            }
        )

        # Detect evolution events by comparing to previous snapshot
        if self.snapshots:
            self._detect_evolution_events(self.snapshots[-1], snapshot, trigger)

        # Add to history
        self.snapshots.append(snapshot)

        # Trim if needed
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]

        # Persist to disk
        self._persist_snapshot(snapshot)

        logger.debug(f"Identity snapshot taken (total: {len(self.snapshots)})")

    def _detect_evolution_events(
        self, prev: IdentitySnapshot, curr: IdentitySnapshot, trigger: str
    ) -> None:
        """Compare two snapshots and log any identity changes as events."""
        now = datetime.now()

        # Value changes
        prev_values = set(prev.core_values)
        curr_values = set(curr.core_values)
        for added in curr_values - prev_values:
            event = IdentityEvolutionEvent(
                timestamp=now,
                event_type="value_added",
                description=f"New core value emerged: {added}",
                new_value=added,
                trigger=trigger,
            )
            self.evolution_events.append(event)
            self._persist_event(event)
            logger.info(f"🧬 Identity evolution: value added — {added}")

        for removed in prev_values - curr_values:
            event = IdentityEvolutionEvent(
                timestamp=now,
                event_type="value_removed",
                description=f"Core value faded: {removed}",
                old_value=removed,
                trigger=trigger,
            )
            self.evolution_events.append(event)
            self._persist_event(event)
            logger.info(f"🧬 Identity evolution: value removed — {removed}")

        # Disposition shift (only log if significant)
        disp_change = self._compute_disposition_change(
            prev.emotional_disposition, curr.emotional_disposition
        )
        if disp_change > 0.2:
            event = IdentityEvolutionEvent(
                timestamp=now,
                event_type="disposition_shift",
                description=f"Emotional disposition shifted by {disp_change:.3f}",
                old_value=prev.emotional_disposition,
                new_value=curr.emotional_disposition,
                trigger=trigger,
            )
            self.evolution_events.append(event)
            self._persist_event(event)

        # Behavioral tendency changes
        for key in set(list(prev.behavioral_tendencies.keys()) + list(curr.behavioral_tendencies.keys())):
            old_val = prev.behavioral_tendencies.get(key, 0.0)
            new_val = curr.behavioral_tendencies.get(key, 0.0)
            if abs(new_val - old_val) > 0.15:
                event = IdentityEvolutionEvent(
                    timestamp=now,
                    event_type="tendency_change",
                    description=f"Behavioral tendency '{key}' changed: {old_val:.2f} → {new_val:.2f}",
                    old_value=old_val,
                    new_value=new_val,
                    trigger=trigger,
                )
                self.evolution_events.append(event)
                self._persist_event(event)

    def get_evolution_summary(self, last_n: int = 20) -> Dict[str, Any]:
        """Return a summary of recent identity evolution events."""
        recent = self.evolution_events[-last_n:] if self.evolution_events else []
        by_type: Dict[str, int] = {}
        for e in recent:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
        return {
            "total_events": len(self.evolution_events),
            "recent_events": [
                {"type": e.event_type, "description": e.description, "timestamp": e.timestamp.isoformat()}
                for e in recent
            ],
            "event_counts": by_type,
        }
    
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
