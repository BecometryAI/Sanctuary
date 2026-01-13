"""
Communication Decision Loop - SPEAK/SILENCE/DEFER decisions.

Weighs drives against inhibitions to decide communication actions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional

from .drive import CommunicationUrge
from .silence import SilenceTracker
from .deferred import DeferredQueue, DeferralReason, DeferredCommunication

logger = logging.getLogger(__name__)

# Decision confidence constants
DEFER_CONFIDENCE = 0.6
SILENCE_DEFAULT_CONFIDENCE = 0.7
DEFERRED_SPEAK_CONFIDENCE = 0.8

# Inhibition to deferral reason mapping (cached for efficiency)
_INHIBITION_TO_DEFERRAL_MAP = None


class CommunicationDecision(Enum):
    """Communication decision types."""
    SPEAK = "speak"
    SILENCE = "silence"
    DEFER = "defer"


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
    inhibitions: List[Any] = field(default_factory=list)
    urges: List[Any] = field(default_factory=list)



class CommunicationDecisionLoop:
    """Evaluates SPEAK/SILENCE/DEFER based on drives vs inhibitions."""
    
    def __init__(self, drive_system, inhibition_system, config: Optional[Dict[str, Any]] = None):
        """Initialize decision loop with drive/inhibition systems and optional config."""
        self.drives = drive_system
        self.inhibitions = inhibition_system
        config = config or {}
        
        # Decision thresholds
        self.speak_threshold = config.get("speak_threshold", 0.3)
        self.silence_threshold = config.get("silence_threshold", -0.2)
        self.defer_min_drive = config.get("defer_min_drive", 0.3)
        self.defer_min_inhibition = config.get("defer_min_inhibition", 0.3)
        self.defer_duration_seconds = config.get("defer_duration_seconds", 30)
        self.history_size = config.get("history_size", 100)
        
        # State
        self.deferred_queue = DeferredQueue(config)
        self.decision_history: List[DecisionResult] = []
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
        """Evaluate SPEAK/SILENCE/DEFER decision based on current state."""
        # Cleanup and check deferred queue first
        self.deferred_queue.cleanup_expired()
        ready_deferred = self.deferred_queue.check_ready()
        if ready_deferred:
            result = self._create_deferred_speak_decision(ready_deferred)
            self._log_decision(result)
            return result
        
        # Compute decision factors
        drive = self.drives.get_total_drive()
        inhibition = self.inhibitions.get_total_inhibition()
        net_pressure = drive - inhibition
        strongest_urge = self.drives.get_strongest_urge()
        
        # Make and log decision
        result = self._make_decision(
            drive, inhibition, net_pressure, strongest_urge,
            self.drives.active_urges, self.inhibitions.active_inhibitions
        )
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
        """Make decision based on net pressure thresholds."""
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
            deferral_reason = self._determine_deferral_reason(active_inhibitions)
            deferred = self.deferred_queue.defer(
                urge=strongest_urge,
                reason=deferral_reason,
                release_seconds=self.defer_duration_seconds,
                condition=f"Wait {self.defer_duration_seconds}s"
            )
            
            return DecisionResult(
                decision=CommunicationDecision.DEFER,
                reason=f"Both high (drive={drive:.2f}, inhibition={inhibition:.2f}) - {deferral_reason.value}",
                confidence=DEFER_CONFIDENCE,
                drive_level=drive,
                inhibition_level=inhibition,
                net_pressure=net_pressure,
                urge=strongest_urge,
                defer_until=deferred.release_after,
                urges=active_urges,
                inhibitions=active_inhibitions
            )
        
        # Default: SILENCE (insufficient drive)
        return DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason=f"Insufficient drive ({drive:.2f})",
            confidence=SILENCE_DEFAULT_CONFIDENCE,
            drive_level=drive,
            inhibition_level=inhibition,
            net_pressure=net_pressure,
            urges=active_urges,
            inhibitions=active_inhibitions
        )
    
    def _determine_deferral_reason(self, inhibitions: List[Any]) -> DeferralReason:
        """Determine deferral reason from strongest inhibition."""
        global _INHIBITION_TO_DEFERRAL_MAP
        
        if not inhibitions:
            return DeferralReason.BAD_TIMING
        
        # Get strongest inhibition
        strongest = max(
            inhibitions,
            key=lambda i: getattr(i, 'get_current_strength', lambda: 0)() * getattr(i, 'priority', 0.5)
        )
        
        inhibition_type = getattr(strongest, 'inhibition_type', None)
        if inhibition_type is None:
            return DeferralReason.BAD_TIMING
        
        # Build mapping once and cache
        if _INHIBITION_TO_DEFERRAL_MAP is None:
            from .inhibition import InhibitionType
            _INHIBITION_TO_DEFERRAL_MAP = {
                InhibitionType.BAD_TIMING: DeferralReason.BAD_TIMING,
                InhibitionType.RECENT_OUTPUT: DeferralReason.BAD_TIMING,
                InhibitionType.STILL_PROCESSING: DeferralReason.PROCESSING,
                InhibitionType.RESPECT_SILENCE: DeferralReason.COURTESY,
                InhibitionType.UNCERTAINTY: DeferralReason.PROCESSING,
                InhibitionType.REDUNDANCY: DeferralReason.BAD_TIMING,
                InhibitionType.LOW_VALUE: DeferralReason.PROCESSING,
            }
        
        return _INHIBITION_TO_DEFERRAL_MAP.get(inhibition_type, DeferralReason.BAD_TIMING)
    
    def _create_deferred_speak_decision(self, deferred: DeferredCommunication) -> DecisionResult:
        """Create SPEAK decision for ready deferred communication."""
        return DecisionResult(
            decision=CommunicationDecision.SPEAK,
            reason=f"Deferred ready ({deferred.reason.value}): {deferred.release_condition}",
            confidence=DEFERRED_SPEAK_CONFIDENCE,
            drive_level=deferred.urge.get_current_intensity(),
            inhibition_level=0.0,
            net_pressure=deferred.urge.get_current_intensity(),
            urge=deferred.urge
        )
    
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
        
        return {
            "deferred_queue": self.deferred_queue.get_queue_summary(),
            "decision_history_size": len(self.decision_history),
            "last_decision": recent[-1] if recent else None,
            "recent_decisions": {
                decision: sum(1 for d in recent if d.decision == decision)
                for decision in CommunicationDecision
            },
            "silence_tracking": self.silence_tracker.get_silence_summary()
        }
