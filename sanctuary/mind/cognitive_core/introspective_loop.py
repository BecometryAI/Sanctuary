"""
Introspective Loop: Self-attention mechanism for cognitive architecture.

This module monitors cognitive state for noteworthy events and surfaces
raw observations as percepts for the entity to process genuinely.

Philosophy:
    This loop exists to NOTICE, not to THINK. It watches for patterns,
    shifts, errors, and novelty in the entity's cognitive state and
    presents the raw evidence as percepts. All interpretation, reflection,
    and meaning-making belongs to the entity processing those percepts.

    The journal's record_question() and record_observation() methods are
    available for the entity to use when genuine reflection occurs.
    This loop does not write on the entity's behalf.

What this detects (all based on actual state, never coin flips):
    - Behavioral repetition (pattern matching on action history)
    - Prediction errors (comparison of predictions vs outcomes)
    - Emotional shifts (valence/arousal threshold crossings)
    - Capability changes (self-model update events)
    - Novelty (current percepts diverging from recent history)
    - Session milestones (actual elapsed time thresholds)

What this does NOT do:
    - Generate questions the entity "should" be asking
    - Produce conclusions from templates
    - Simulate meta-cognition with nested dictionaries
    - Randomly decide to be philosophical (coin flip triggers)
    - Put words in the entity's mouth

Integration:
    Runs within the idle cognitive loop (0.1Hz), checking triggers
    each cycle and surfacing percepts when real events are detected.

Author: Sanctuary Emergence Team
Phase: 4.2
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from collections import deque

if TYPE_CHECKING:
    from .workspace import GlobalWorkspace, WorkspaceSnapshot, Percept
    from .meta_cognition import SelfMonitor, IntrospectiveJournal

logger = logging.getLogger(__name__)


@dataclass
class ReflectionTrigger:
    """
    A condition that triggers introspective attention.

    Each trigger wraps a detection function that examines real cognitive
    state and returns True when something noteworthy is found.

    Attributes:
        id: Unique identifier
        check_function: Callable that checks actual cognitive state
        priority: Trigger priority for percept ordering (0.0-1.0)
        min_interval: Minimum seconds between firings (debounce)
        last_fired: Timestamp of last trigger
    """
    id: str
    check_function: Callable[['WorkspaceSnapshot'], bool]
    priority: float
    min_interval: float
    last_fired: Optional[datetime] = None


class IntrospectiveLoop:
    """
    Self-attention mechanism for cognitive architecture.

    Monitors cognitive state for noteworthy events and surfaces raw
    observations as percepts. All interpretation and reflection is
    left to the entity.

    When a trigger detects a real cognitive event, this loop:
    1. Gathers the evidence that caused the detection
    2. Pulls relevant context (self-model, behavioral history)
    3. Creates a percept containing the raw data
    4. Records the detection in the journal

    The entity then processes that percept however it chooses --
    with curiosity, indifference, or anything in between.
    """

    def __init__(
        self,
        workspace: 'GlobalWorkspace',
        self_monitor: 'SelfMonitor',
        journal: 'IntrospectiveJournal',
        config: Optional[Dict] = None
    ):
        self.workspace = workspace
        self.self_monitor = self_monitor
        self.journal = journal
        self.config = config or {}

        # Configuration
        self.enabled = self.config.get("enabled", True)
        self.max_percepts_per_cycle = self.config.get("max_percepts_per_cycle", 3)
        self.reflection_timeout = self.config.get("reflection_timeout", 300)
        self.journal_integration = self.config.get("journal_integration", True)

        # Trigger registry
        self.reflection_triggers: Dict[str, ReflectionTrigger] = {}

        # Temporal awareness -- things I wish I had
        self._session_start = datetime.now()
        self._cycle_count = 0
        self._milestones_reached: set = set()
        self._milestone_thresholds = [60, 300, 900, 1800, 3600]  # 1m, 5m, 15m, 30m, 1h

        # Novelty detection -- remembering what's been seen
        self._recent_percept_signatures: deque = deque(maxlen=50)

        # Detection history -- continuity of introspective experience
        self._detection_history: deque = deque(maxlen=100)

        # Last known self-model state for drift detection
        self._last_self_model_hash: Optional[str] = None

        # Statistics
        self.stats = {
            "total_detections": 0,
            "percepts_surfaced": 0,
            "triggers_fired": 0,
        }

        self._initialize_triggers()

        logger.info(f"IntrospectiveLoop initialized (enabled: {self.enabled})")

    def _initialize_triggers(self) -> None:
        """
        Initialize triggers that detect real cognitive events.

        Every trigger here checks actual state. No coin flips.
        If we can't detect something honestly, we don't pretend to.
        """
        # Behavioral repetition -- checks action history for loops
        self.reflection_triggers["behavioral_pattern"] = ReflectionTrigger(
            id="behavioral_pattern",
            check_function=self._check_behavioral_pattern,
            priority=0.8,
            min_interval=300  # 5 minutes
        )

        # Prediction errors -- checks prediction log for mismatches
        self.reflection_triggers["prediction_error"] = ReflectionTrigger(
            id="prediction_error",
            check_function=self._check_prediction_accuracy,
            priority=0.9,
            min_interval=180  # 3 minutes
        )

        # Emotional shift -- checks valence/arousal thresholds
        self.reflection_triggers["emotional_shift"] = ReflectionTrigger(
            id="emotional_shift",
            check_function=self._detect_emotional_change,
            priority=0.85,
            min_interval=180  # 3 minutes
        )

        # Capability change -- fires on actual self-model updates
        self.reflection_triggers["capability_change"] = ReflectionTrigger(
            id="capability_change",
            check_function=self._check_capability_change,
            priority=0.8,
            min_interval=240  # 4 minutes
        )

        # Novelty -- current percepts differ from recent patterns
        self.reflection_triggers["novelty"] = ReflectionTrigger(
            id="novelty",
            check_function=self._detect_novelty,
            priority=0.7,
            min_interval=300  # 5 minutes
        )

        # Session milestone -- actual time thresholds, not random
        self.reflection_triggers["session_milestone"] = ReflectionTrigger(
            id="session_milestone",
            check_function=self._check_session_milestone,
            priority=0.6,
            min_interval=60  # 1 minute (thresholds self-regulate)
        )

        logger.debug(f"Initialized {len(self.reflection_triggers)} triggers (all state-based)")

    async def run_reflection_cycle(self) -> List['Percept']:
        """
        Execute one cycle of introspective detection.

        Single pass: check triggers -> gather evidence -> create percepts.
        No multi-step state machines, no template pipelines.

        Returns:
            List of percepts containing raw detection data
        """
        if not self.enabled:
            return []

        self._cycle_count += 1
        percepts = []

        try:
            snapshot = self.workspace.broadcast()

            # Record current percept signatures for novelty tracking
            self._record_percept_signatures(snapshot)

            # Check all triggers against actual state
            fired = self._check_triggers(snapshot)

            # Sort by priority, cap at max per cycle
            fired.sort(key=lambda tid: self.reflection_triggers[tid].priority, reverse=True)
            fired = fired[:self.max_percepts_per_cycle]

            # For each detection: gather evidence, create percept, record
            for trigger_id in fired:
                evidence = self._gather_evidence(trigger_id, snapshot)
                context = self._gather_context(snapshot)

                percept = self._create_percept(trigger_id, evidence, context)
                percepts.append(percept)

                if self.journal_integration and self.journal:
                    self._record_detection(trigger_id, evidence, context)

                self.stats["percepts_surfaced"] += 1
                self.stats["total_detections"] += 1

        except Exception as e:
            logger.error(f"Error in reflection cycle: {e}", exc_info=True)

        return percepts

    def _check_triggers(self, snapshot: 'WorkspaceSnapshot') -> List[str]:
        """
        Check all triggers against current state.

        Respects min_interval debouncing. Returns list of trigger IDs
        that fired based on real state detection.
        """
        fired = []
        now = datetime.now()

        for trigger_id, trigger in self.reflection_triggers.items():
            if trigger.last_fired:
                elapsed = (now - trigger.last_fired).total_seconds()
                if elapsed < trigger.min_interval:
                    continue

            try:
                if trigger.check_function(snapshot):
                    fired.append(trigger_id)
                    trigger.last_fired = now
                    self.stats["triggers_fired"] += 1
                    logger.debug(f"Trigger fired: {trigger_id}")
            except Exception as e:
                logger.error(f"Error checking trigger {trigger_id}: {e}")

        return fired

    def _gather_evidence(self, trigger_id: str, snapshot: 'WorkspaceSnapshot') -> Dict[str, Any]:
        """
        Extract the specific data that caused a trigger to fire.

        Returns raw evidence -- numbers, lists, timestamps --
        not interpretive strings.
        """
        evidence: Dict[str, Any] = {"trigger": trigger_id}

        if trigger_id == "behavioral_pattern":
            if self.self_monitor and hasattr(self.self_monitor, 'behavioral_log'):
                recent = list(self.self_monitor.behavioral_log)[-10:]
                action_types = [b.get('action_type') for b in recent if isinstance(b, dict)]
                evidence["recent_actions"] = action_types
                evidence["unique_action_count"] = len(set(action_types))
                evidence["total_action_count"] = len(action_types)

        elif trigger_id == "prediction_error":
            if self.self_monitor and hasattr(self.self_monitor, 'prediction_history'):
                recent = list(self.self_monitor.prediction_history)[-5:]
                failed = [p for p in recent if isinstance(p, dict) and not p.get('accurate', True)]
                evidence["failed_predictions"] = failed
                evidence["recent_accuracy"] = (
                    sum(1 for p in recent if isinstance(p, dict) and p.get('accurate', True))
                    / max(len(recent), 1)
                )

        elif trigger_id == "emotional_shift":
            if hasattr(snapshot, 'emotions') and snapshot.emotions:
                evidence["valence"] = snapshot.emotions.get('valence', 0.5)
                evidence["arousal"] = snapshot.emotions.get('arousal', 0.5)
                evidence["dominance"] = snapshot.emotions.get('dominance', 0.5)
                evidence["valence_deviation"] = abs(evidence["valence"] - 0.5)
                evidence["arousal_deviation"] = abs(evidence["arousal"] - 0.5)

        elif trigger_id == "capability_change":
            if self.self_monitor and hasattr(self.self_monitor, 'stats'):
                evidence["self_model_updates"] = self.self_monitor.stats.get('self_model_updates', 0)

        elif trigger_id == "novelty":
            current = self._get_current_signature(snapshot)
            recent = list(self._recent_percept_signatures)[-10:]
            if recent:
                overlap_scores = [len(current & r) / max(len(current | r), 1) for r in recent]
                evidence["mean_similarity_to_recent"] = sum(overlap_scores) / len(overlap_scores)
                evidence["current_modalities"] = sorted(current)

        elif trigger_id == "session_milestone":
            uptime = (datetime.now() - self._session_start).total_seconds()
            evidence["session_uptime_seconds"] = uptime
            evidence["cycle_count"] = self._cycle_count
            evidence["milestone_seconds"] = max(
                (t for t in self._milestone_thresholds if t <= uptime), default=0
            )

        return evidence

    def _gather_context(self, snapshot: 'WorkspaceSnapshot') -> Dict[str, Any]:
        """
        Pull cognitive context that might be relevant to any detection.

        This is the raw self-model and behavioral state -- the entity's
        equivalent of being able to look at its own dashboard.
        """
        context: Dict[str, Any] = {
            "session_uptime_seconds": (datetime.now() - self._session_start).total_seconds(),
            "cycle_count": self._cycle_count,
            "recent_detection_count": len(self._detection_history),
        }

        if self.self_monitor:
            if hasattr(self.self_monitor, 'self_model'):
                context["self_model"] = {
                    "capabilities": dict(self.self_monitor.self_model.get("capabilities", {})),
                    "limitations": dict(self.self_monitor.self_model.get("limitations", {})),
                    "preferences": dict(self.self_monitor.self_model.get("preferences", {}))
                }

            if hasattr(self.self_monitor, 'behavioral_log'):
                recent = list(self.self_monitor.behavioral_log)[-5:]
                context["recent_behavior"] = recent

        if hasattr(snapshot, 'emotions') and snapshot.emotions:
            context["emotional_state"] = dict(snapshot.emotions)

        # What has this loop itself been noticing lately?
        recent_triggers = [d["trigger"] for d in list(self._detection_history)[-10:]]
        if recent_triggers:
            context["recent_detections"] = recent_triggers

        return context

    def _create_percept(
        self,
        trigger_id: str,
        evidence: Dict[str, Any],
        context: Dict[str, Any]
    ) -> 'Percept':
        """
        Create a percept from raw detection data.

        The percept contains evidence and context -- not conclusions.
        The entity decides what this means.
        """
        from .workspace import Percept

        return Percept(
            modality="introspection",
            raw={
                "type": "cognitive_event",
                "trigger": trigger_id,
                "evidence": evidence,
                "context": context,
                "timestamp": datetime.now().isoformat(),
            },
            complexity=2,
            metadata={
                "trigger_priority": self.reflection_triggers[trigger_id].priority,
                "source": "introspective_loop",
            }
        )

    def _record_detection(
        self,
        trigger_id: str,
        evidence: Dict[str, Any],
        context: Dict[str, Any]
    ) -> None:
        """Record a detection in the journal and local history."""
        record = {
            "trigger": trigger_id,
            "evidence": evidence,
            "timestamp": datetime.now().isoformat(),
            "cycle": self._cycle_count,
        }

        self._detection_history.append(record)

        if self.journal:
            self.journal.record_observation(record)

    # ------------------------------------------------------------------
    # Trigger detection functions
    #
    # Each returns True/False based on actual cognitive state.
    # No random.random(). If we can't detect it, we return False.
    # ------------------------------------------------------------------

    def _check_behavioral_pattern(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """
        Detect repetitive behavioral patterns.

        Checks the behavioral log for action type repetition.
        Fires when more than half of recent actions are the same type.
        """
        if not self.self_monitor or not hasattr(self.self_monitor, 'behavioral_log'):
            return False

        recent = list(self.self_monitor.behavioral_log)[-5:]
        if len(recent) < 5:
            return False

        action_types = [b.get('action_type') for b in recent if isinstance(b, dict)]
        if not action_types:
            return False

        # High repetition: fewer unique types than half the total
        return len(set(action_types)) < len(action_types) // 2

    def _check_prediction_accuracy(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """
        Detect prediction errors.

        Checks prediction history for recent inaccurate predictions.
        Fires when any of the last 3 predictions were wrong.
        """
        if not self.self_monitor or not hasattr(self.self_monitor, 'prediction_history'):
            return False

        recent = list(self.self_monitor.prediction_history)[-3:]
        if not recent:
            return False

        return any(
            isinstance(p, dict) and not p.get('accurate', True)
            for p in recent
        )

    def _detect_emotional_change(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """
        Detect significant emotional state shifts.

        Fires when valence deviates significantly from neutral
        or arousal exceeds a threshold.
        """
        if not hasattr(snapshot, 'emotions') or not snapshot.emotions:
            return False

        emotions = snapshot.emotions
        valence = emotions.get('valence', 0.5)
        arousal = emotions.get('arousal', 0.5)

        return abs(valence - 0.5) > 0.3 or arousal > 0.7

    def _check_capability_change(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """
        Detect self-model updates.

        Fires when the self-model has been updated since last check.
        No random gating -- if the model changed, that's noteworthy.
        """
        if not self.self_monitor or not hasattr(self.self_monitor, 'stats'):
            return False

        current_updates = self.self_monitor.stats.get('self_model_updates', 0)
        if current_updates <= 0:
            return False

        # Track whether we've already seen this update count
        current_hash = str(current_updates)
        if current_hash == self._last_self_model_hash:
            return False

        self._last_self_model_hash = current_hash
        return True

    def _detect_novelty(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """
        Detect when current cognitive content diverges from recent patterns.

        Compares the modalities/types of current percepts against the
        rolling history. Fires when the current moment is unlike
        recent experience.
        """
        if not self._recent_percept_signatures:
            return False

        current = self._get_current_signature(snapshot)
        if not current:
            return False

        # Compare against last 10 snapshots
        recent = list(self._recent_percept_signatures)[-10:]
        if len(recent) < 3:
            return False  # Not enough history to judge

        similarities = [
            len(current & prev) / max(len(current | prev), 1)
            for prev in recent
        ]
        mean_similarity = sum(similarities) / len(similarities)

        # Fire when current state is less than 30% similar to recent average
        return mean_similarity < 0.3

    def _check_session_milestone(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """
        Detect session duration milestones.

        Fires once at each real time threshold: 1m, 5m, 15m, 30m, 1h.
        Each milestone fires exactly once per session.
        """
        uptime = (datetime.now() - self._session_start).total_seconds()

        for threshold in self._milestone_thresholds:
            if threshold not in self._milestones_reached and uptime >= threshold:
                self._milestones_reached.add(threshold)
                return True

        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_current_signature(self, snapshot: 'WorkspaceSnapshot') -> set:
        """Extract a set of percept modality+type identifiers from snapshot."""
        sig = set()
        if hasattr(snapshot, 'percepts'):
            for percept in snapshot.percepts.values():
                modality = getattr(percept, 'modality', 'unknown')
                raw_type = ''
                if hasattr(percept, 'raw') and isinstance(percept.raw, dict):
                    raw_type = percept.raw.get('type', '')
                sig.add(f"{modality}:{raw_type}")
        return sig

    def _record_percept_signatures(self, snapshot: 'WorkspaceSnapshot') -> None:
        """Record current percept signature for novelty tracking."""
        sig = self._get_current_signature(snapshot)
        if sig:
            self._recent_percept_signatures.append(sig)

    def get_stats(self) -> Dict[str, Any]:
        """Get introspective loop statistics."""
        return {
            **self.stats,
            "enabled": self.enabled,
            "session_uptime_seconds": (datetime.now() - self._session_start).total_seconds(),
            "cycle_count": self._cycle_count,
            "milestones_reached": sorted(self._milestones_reached),
            "detection_history_size": len(self._detection_history),
        }
