"""
Communication Decision Loop - SPEAK/SILENCE/DEFER decisions.

Weighs drives against inhibitions to decide whether to speak, stay silent,
or defer communication for later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

from .drive import CommunicationUrge
from .silence import SilenceTracker, SilenceType

logger = logging.getLogger(__name__)

# Decision confidence constants
_DEFER_CONFIDENCE = 0.6
_SILENCE_DEFAULT_CONFIDENCE = 0.7
_DEFERRED_SPEAK_CONFIDENCE = 0.8


class CommunicationDecision(Enum):
    """Communication decision types."""
    SPEAK = "speak"
    SILENCE = "silence"
    DEFER = "defer"


@dataclass
class DeferredCommunication:
    """Communication deferred for later reconsideration."""
    urge: CommunicationUrge
    reason: str
    deferred_at: datetime = field(default_factory=datetime.now)
    defer_until: Optional[datetime] = None
    attempts: int = 0
    
    def is_ready(self) -> bool:
        """Check if ready to reconsider."""
        return self.defer_until is not None and datetime.now() >= self.defer_until
    
    def increment_attempts(self) -> None:
        """Increment reconsideration attempt counter."""
        self.attempts += 1


@dataclass
class DecisionResult:
    """Result of communication decision evaluation."""
    decision: CommunicationDecision
    reason: str
    confidence: float
    drive_level: float
    inhibition_level: float
    net_pressure: float
    urge: Optional[CommunicationUrge] = None
    defer_until: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.now)
    inhibitions: List[Any] = field(default_factory=list)  # InhibitionFactor instances
    urges: List[Any] = field(default_factory=list)  # CommunicationUrge instances



class CommunicationDecisionLoop:
    """
    Evaluates SPEAK/SILENCE/DEFER based on drives vs inhibitions.
    
    Integrates drive and inhibition systems for autonomous decisions.
    """
    
    def __init__(self, drive_system, inhibition_system, config: Optional[Dict[str, Any]] = None):
        """
        Initialize decision loop.
        
        Args:
            drive_system: CommunicationDriveSystem instance
            inhibition_system: CommunicationInhibitionSystem instance
            config: Configuration with thresholds and limits (see defaults)
        """
        self.drives = drive_system
        self.inhibitions = inhibition_system
        config = config or {}
        
        # Decision thresholds
        self.speak_threshold = config.get("speak_threshold", 0.3)
        self.silence_threshold = config.get("silence_threshold", -0.2)
        self.defer_min_drive = config.get("defer_min_drive", 0.3)
        self.defer_min_inhibition = config.get("defer_min_inhibition", 0.3)
        
        # Queue and history limits
        self.defer_duration_seconds = config.get("defer_duration_seconds", 30)
        self.max_deferred = config.get("max_deferred", 10)
        self.max_defer_attempts = config.get("max_defer_attempts", 3)
        self.history_size = config.get("history_size", 100)
        
        # State
        self.deferred_queue: List[DeferredCommunication] = []
        self.decision_history: List[DecisionResult] = []
        
        # Silence tracking
        self.silence_tracker = SilenceTracker(config)
        
        logger.info(f"DecisionLoop initialized: speak={self.speak_threshold:.2f}, "
                   f"silence={self.silence_threshold:.2f}")
    
    def evaluate(
        self,
        workspace_state: Any,
        emotional_state: Dict[str, float],
        goals: List[Any],
        memories: List[Any]
    ) -> DecisionResult:
        """
        Evaluate whether to SPEAK, stay SILENT, or DEFER.
        
        Logic: Check deferred queue → compute net pressure → apply thresholds
        
        Returns:
            DecisionResult with decision, reason, and context
        """
        # Priority: Check deferred queue first
        ready_deferred = self.check_deferred_queue()
        if ready_deferred:
            result = self._create_deferred_speak_decision(ready_deferred)
            self._log_decision(result)
            return result
        
        # Compute current state
        drive = self.drives.get_total_drive()
        inhibition = self.inhibitions.get_total_inhibition()
        net_pressure = drive - inhibition
        strongest_urge = self.drives.get_strongest_urge()
        
        # Pass active urges and inhibitions for silence tracking (no copy needed)
        active_urges = self.drives.active_urges
        active_inhibitions = self.inhibitions.active_inhibitions
        
        # Make decision
        result = self._make_decision(drive, inhibition, net_pressure, strongest_urge, 
                                     active_urges, active_inhibitions)
        self._log_decision(result)
        
        return result
    
    def _make_decision(
        self,
        drive: float,
        inhibition: float,
        net_pressure: float,
        strongest_urge: Optional[CommunicationUrge],
        active_urges: List[Any],
        active_inhibitions: List[Any]
    ) -> DecisionResult:
        """Make decision based on thresholds."""
        # SPEAK: Strong net drive
        if net_pressure > self.speak_threshold:
            return DecisionResult(
                decision=CommunicationDecision.SPEAK,
                reason=f"Drive ({drive:.2f}) exceeds inhibition ({inhibition:.2f})",
                confidence=min(1.0, net_pressure / (self.speak_threshold * 2)),
                drive_level=drive,
                inhibition_level=inhibition,
                net_pressure=net_pressure,
                urge=strongest_urge,
                urges=active_urges,
                inhibitions=active_inhibitions
            )
        
        # SILENCE: Inhibition dominates
        if net_pressure < self.silence_threshold:
            return DecisionResult(
                decision=CommunicationDecision.SILENCE,
                reason=f"Inhibition ({inhibition:.2f}) exceeds drive ({drive:.2f})",
                confidence=min(1.0, abs(net_pressure) / abs(self.silence_threshold * 2)),
                drive_level=drive,
                inhibition_level=inhibition,
                net_pressure=net_pressure,
                urges=active_urges,
                inhibitions=active_inhibitions
            )
        
        # DEFER: Both drive and inhibition significant
        if drive >= self.defer_min_drive and inhibition >= self.defer_min_inhibition and strongest_urge:
            defer_until = datetime.now() + timedelta(seconds=self.defer_duration_seconds)
            self.defer_communication(strongest_urge, "Timing not right", self.defer_duration_seconds)
            
            return DecisionResult(
                decision=CommunicationDecision.DEFER,
                reason=f"Drive ({drive:.2f}) and inhibition ({inhibition:.2f}) both high",
                confidence=_DEFER_CONFIDENCE,
                drive_level=drive,
                inhibition_level=inhibition,
                net_pressure=net_pressure,
                urge=strongest_urge,
                defer_until=defer_until,
                urges=active_urges,
                inhibitions=active_inhibitions
            )
        
        # Default: SILENCE (insufficient drive)
        return DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason=f"Insufficient drive ({drive:.2f})",
            confidence=_SILENCE_DEFAULT_CONFIDENCE,
            drive_level=drive,
            inhibition_level=inhibition,
            net_pressure=net_pressure,
            urges=active_urges,
            inhibitions=active_inhibitions
        )
    
    def _create_deferred_speak_decision(self, deferred: DeferredCommunication) -> DecisionResult:
        """Create SPEAK decision for ready deferred communication."""
        self.deferred_queue.remove(deferred)
        
        return DecisionResult(
            decision=CommunicationDecision.SPEAK,
            reason=f"Deferred ready: {deferred.reason}",
            confidence=_DEFERRED_SPEAK_CONFIDENCE,
            drive_level=deferred.urge.get_current_intensity(),
            inhibition_level=0.0,
            net_pressure=deferred.urge.get_current_intensity(),
            urge=deferred.urge
        )
    
    def check_deferred_queue(self) -> Optional[DeferredCommunication]:
        """Check for ready deferred communications, return highest priority."""
        ready_items = [d for d in self.deferred_queue if d.is_ready()]
        if not ready_items:
            return None
        
        # Sort by priority: urge intensity × priority
        best = max(ready_items, key=lambda d: d.urge.get_current_intensity() * d.urge.priority)
        
        # Drop if max attempts reached
        if best.attempts >= self.max_defer_attempts:
            logger.debug(f"Dropping after {best.attempts} attempts: {best.reason}")
            self.deferred_queue.remove(best)
            return None
        
        best.increment_attempts()
        return best
    
    def defer_communication(self, urge: CommunicationUrge, reason: str, defer_seconds: int) -> None:
        """Add communication to deferred queue."""
        if defer_seconds <= 0:
            logger.warning(f"Invalid defer_seconds: {defer_seconds}, skipping")
            return
            
        deferred = DeferredCommunication(
            urge=urge,
            reason=reason,
            defer_until=datetime.now() + timedelta(seconds=defer_seconds)
        )
        
        self.deferred_queue.append(deferred)
        
        # Maintain size limit by removing oldest
        if len(self.deferred_queue) > self.max_deferred:
            oldest = min(self.deferred_queue, key=lambda d: d.deferred_at)
            self.deferred_queue.remove(oldest)
            logger.debug(f"Dropped oldest: {oldest.reason}")
        
        logger.debug(f"Deferred until {deferred.defer_until.strftime('%H:%M:%S')}: {reason}")
    
    def _log_decision(self, result: DecisionResult) -> None:
        """Log decision to history and update silence tracking."""
        self.decision_history.append(result)
        
        # Maintain size limit
        if len(self.decision_history) > self.history_size:
            self.decision_history = self.decision_history[-self.history_size:]
        
        # Update silence tracking based on decision
        if result.decision == CommunicationDecision.SPEAK:
            ended_silence = self.silence_tracker.end_silence()
            if ended_silence:
                logger.info(f"✅ SPEAK: {result.reason} (breaking silence after {ended_silence.duration:.1f}s)")
            else:
                logger.info(f"✅ SPEAK: {result.reason}")
        elif result.decision == CommunicationDecision.SILENCE:
            silence_action = self.silence_tracker.record_silence(result)
            logger.info(f"🔇 SILENCE: {silence_action.silence_type.value} - {silence_action.reason}")
        elif result.decision == CommunicationDecision.DEFER:
            logger.debug(f"⏸️ DEFER: {result.reason}")
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of decision loop state including silence tracking."""
        recent = self.decision_history[-10:] if self.decision_history else []
        ready_count = sum(1 for d in self.deferred_queue if d.is_ready())
        
        return {
            "deferred_queue_size": len(self.deferred_queue),
            "decision_history_size": len(self.decision_history),
            "last_decision": recent[-1] if recent else None,
            "recent_decisions": {
                decision: sum(1 for d in recent if d.decision == decision)
                for decision in CommunicationDecision
            },
            "ready_deferred": ready_count,
            "silence_tracking": self.silence_tracker.get_silence_summary()
        }
