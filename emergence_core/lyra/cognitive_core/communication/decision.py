"""
Communication Decision Loop - SPEAK/SILENCE/DEFER decisions.

This module implements the decision-making component that weighs drives
against inhibitions and decides whether to speak, stay silent, or defer
communication for later.

Key Features:
- Evaluates drives vs inhibitions each cognitive cycle
- Makes explicit decisions: SPEAK, SILENCE, or DEFER
- Logs reasoning for each decision
- Handles deferred communications (queue for later)
- Integrates with output generation when SPEAK is decided
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any, Optional

from .drive import CommunicationUrge
from .inhibition import InhibitionFactor

logger = logging.getLogger(__name__)


class CommunicationDecision(Enum):
    """Types of communication decisions."""
    SPEAK = "speak"        # Generate and emit output
    SILENCE = "silence"    # Explicitly choose not to respond
    DEFER = "defer"        # Queue for later


@dataclass
class DeferredCommunication:
    """
    Represents a communication deferred for later.
    
    Attributes:
        urge: The communication urge to act on later
        reason: Why this was deferred
        deferred_at: When this was deferred
        defer_until: When to reconsider this communication
        attempts: Number of times this has been reconsidered
    """
    urge: CommunicationUrge
    reason: str
    deferred_at: datetime = field(default_factory=datetime.now)
    defer_until: Optional[datetime] = None
    attempts: int = 0
    
    def is_ready(self) -> bool:
        """Check if deferred communication is ready to reconsider."""
        if self.defer_until is None:
            return False
        return datetime.now() >= self.defer_until
    
    def increment_attempts(self) -> None:
        """Increment attempt counter."""
        self.attempts += 1


@dataclass
class DecisionResult:
    """
    Result of a communication decision evaluation.
    
    Attributes:
        decision: The decision made (SPEAK, SILENCE, or DEFER)
        reason: Human-readable explanation for the decision
        confidence: Confidence in the decision (0.0 to 1.0)
        drive_level: Total drive level at decision time
        inhibition_level: Total inhibition level at decision time
        net_pressure: Drive - inhibition
        urge: What to say if SPEAK (strongest urge)
        defer_until: When to reconsider if DEFER
        timestamp: When decision was made
    """
    decision: CommunicationDecision
    reason: str
    confidence: float
    drive_level: float
    inhibition_level: float
    net_pressure: float
    urge: Optional[CommunicationUrge] = None
    defer_until: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.now)


class CommunicationDecisionLoop:
    """
    Decision loop that evaluates SPEAK/SILENCE/DEFER based on drives vs inhibitions.
    
    This is the core decision-making component that integrates the drive system
    and inhibition system to make autonomous communication decisions.
    
    Attributes:
        drives: CommunicationDriveSystem instance
        inhibitions: CommunicationInhibitionSystem instance
        config: Configuration dictionary
        deferred_queue: Queue of deferred communications
        decision_history: History of decisions with reasoning
    """
    
    def __init__(self, drive_system, inhibition_system, config: Optional[Dict[str, Any]] = None):
        """
        Initialize communication decision loop.
        
        Args:
            drive_system: CommunicationDriveSystem instance
            inhibition_system: CommunicationInhibitionSystem instance
            config: Optional configuration dict with keys:
                - speak_threshold: Net pressure required to speak (default: 0.3)
                - silence_threshold: Net pressure below which to stay silent (default: -0.2)
                - defer_min_drive: Minimum drive for deferral (default: 0.3)
                - defer_min_inhibition: Minimum inhibition for deferral (default: 0.3)
                - defer_duration_seconds: Default deferral duration (default: 30)
                - max_deferred: Maximum deferred communications (default: 10)
                - max_defer_attempts: Maximum reconsideration attempts (default: 3)
                - history_size: Maximum decision history size (default: 100)
        """
        self.drives = drive_system
        self.inhibitions = inhibition_system
        self.config = config or {}
        
        # Load configuration
        self.speak_threshold = self.config.get("speak_threshold", 0.3)
        self.silence_threshold = self.config.get("silence_threshold", -0.2)
        self.defer_min_drive = self.config.get("defer_min_drive", 0.3)
        self.defer_min_inhibition = self.config.get("defer_min_inhibition", 0.3)
        self.defer_duration_seconds = self.config.get("defer_duration_seconds", 30)
        self.max_deferred = self.config.get("max_deferred", 10)
        self.max_defer_attempts = self.config.get("max_defer_attempts", 3)
        self.history_size = self.config.get("history_size", 100)
        
        # State tracking
        self.deferred_queue: List[DeferredCommunication] = []
        self.decision_history: List[DecisionResult] = []
        
        logger.info(f"CommunicationDecisionLoop initialized: "
                   f"speak_threshold={self.speak_threshold:.2f}, "
                   f"silence_threshold={self.silence_threshold:.2f}, "
                   f"defer_enabled={self.defer_min_drive > 0}")
    
    def evaluate(
        self,
        workspace_state: Any,
        emotional_state: Dict[str, float],
        goals: List[Any],
        memories: List[Any]
    ) -> DecisionResult:
        """
        Evaluate whether to SPEAK, stay SILENT, or DEFER.
        
        Decision Logic:
        1. Check deferred queue for ready items (priority)
        2. Compute total drive and inhibition
        3. Calculate net communication pressure (drive - inhibition)
        4. Apply decision thresholds:
           - SPEAK: net_pressure > speak_threshold
           - SILENCE: net_pressure < silence_threshold
           - DEFER: ambiguous (drive exists but inhibited)
        5. Log decision with reasoning
        
        Args:
            workspace_state: Current workspace snapshot
            emotional_state: VAD emotional state dict
            goals: Active goal objects
            memories: Recently retrieved memory objects
            
        Returns:
            DecisionResult with decision, reason, and context
        """
        # Step 1: Check deferred queue first (highest priority)
        ready_deferred = self.check_deferred_queue()
        if ready_deferred:
            result = self._create_deferred_speak_decision(ready_deferred)
            self._log_decision(result)
            return result
        
        # Step 2: Compute drives and inhibitions
        drive = self.drives.get_total_drive()
        inhibition = self.inhibitions.get_total_inhibition()
        net_pressure = drive - inhibition
        
        # Step 3: Get strongest urge for potential communication
        strongest_urge = self.drives.get_strongest_urge()
        
        # Step 4: Make decision based on thresholds
        result = self._make_decision(
            drive=drive,
            inhibition=inhibition,
            net_pressure=net_pressure,
            strongest_urge=strongest_urge
        )
        
        # Step 5: Log decision
        self._log_decision(result)
        
        return result
    
    def _make_decision(
        self,
        drive: float,
        inhibition: float,
        net_pressure: float,
        strongest_urge: Optional[CommunicationUrge]
    ) -> DecisionResult:
        """
        Make decision based on drive, inhibition, and thresholds.
        
        Args:
            drive: Total drive level
            inhibition: Total inhibition level
            net_pressure: drive - inhibition
            strongest_urge: Strongest active urge
            
        Returns:
            DecisionResult with decision and reasoning
        """
        # Decision: SPEAK (strong net drive)
        if net_pressure > self.speak_threshold:
            confidence = min(1.0, net_pressure / (self.speak_threshold * 2))
            return DecisionResult(
                decision=CommunicationDecision.SPEAK,
                reason=f"Drive ({drive:.2f}) significantly exceeds inhibition ({inhibition:.2f})",
                confidence=confidence,
                drive_level=drive,
                inhibition_level=inhibition,
                net_pressure=net_pressure,
                urge=strongest_urge
            )
        
        # Decision: SILENCE (inhibition dominates)
        if net_pressure < self.silence_threshold:
            confidence = min(1.0, abs(net_pressure) / abs(self.silence_threshold * 2))
            return DecisionResult(
                decision=CommunicationDecision.SILENCE,
                reason=f"Inhibition ({inhibition:.2f}) exceeds drive ({drive:.2f})",
                confidence=confidence,
                drive_level=drive,
                inhibition_level=inhibition,
                net_pressure=net_pressure
            )
        
        # Ambiguous zone - check for deferral conditions
        # DEFER if both drive and inhibition are significant
        if drive >= self.defer_min_drive and inhibition >= self.defer_min_inhibition:
            if strongest_urge:
                defer_until = datetime.now() + timedelta(seconds=self.defer_duration_seconds)
                self.defer_communication(
                    urge=strongest_urge,
                    reason="Drive exists but inhibited - timing not right",
                    defer_seconds=self.defer_duration_seconds
                )
                
                confidence = 0.6  # Medium confidence in deferral
                return DecisionResult(
                    decision=CommunicationDecision.DEFER,
                    reason=f"Drive ({drive:.2f}) and inhibition ({inhibition:.2f}) both significant - defer",
                    confidence=confidence,
                    drive_level=drive,
                    inhibition_level=inhibition,
                    net_pressure=net_pressure,
                    urge=strongest_urge,
                    defer_until=defer_until
                )
        
        # Default: SILENCE (insufficient drive)
        confidence = 0.7
        return DecisionResult(
            decision=CommunicationDecision.SILENCE,
            reason=f"Insufficient drive ({drive:.2f}) to overcome inhibition ({inhibition:.2f})",
            confidence=confidence,
            drive_level=drive,
            inhibition_level=inhibition,
            net_pressure=net_pressure
        )
    
    def _create_deferred_speak_decision(
        self,
        deferred: DeferredCommunication
    ) -> DecisionResult:
        """
        Create a SPEAK decision for a deferred communication.
        
        Args:
            deferred: Ready deferred communication
            
        Returns:
            DecisionResult with SPEAK decision
        """
        # Remove from queue
        self.deferred_queue.remove(deferred)
        
        return DecisionResult(
            decision=CommunicationDecision.SPEAK,
            reason=f"Deferred communication ready (was: {deferred.reason})",
            confidence=0.8,
            drive_level=deferred.urge.get_current_intensity(),
            inhibition_level=0.0,  # Assumed ready means inhibition cleared
            net_pressure=deferred.urge.get_current_intensity(),
            urge=deferred.urge
        )
    
    def check_deferred_queue(self) -> Optional[DeferredCommunication]:
        """
        Check if any deferred communications are ready.
        
        Returns:
            Ready deferred communication with highest priority, or None
        """
        ready_items = [d for d in self.deferred_queue if d.is_ready()]
        
        if not ready_items:
            return None
        
        # Return highest priority (by urge intensity * priority)
        ready_items.sort(
            key=lambda d: d.urge.get_current_intensity() * d.urge.priority,
            reverse=True
        )
        
        best = ready_items[0]
        
        # Check if this has been deferred too many times
        if best.attempts >= self.max_defer_attempts:
            logger.debug(f"Dropping deferred communication after {best.attempts} attempts")
            self.deferred_queue.remove(best)
            return None
        
        best.increment_attempts()
        return best
    
    def defer_communication(
        self,
        urge: CommunicationUrge,
        reason: str,
        defer_seconds: int
    ) -> None:
        """
        Add communication to deferred queue.
        
        Args:
            urge: Communication urge to defer
            reason: Why this is being deferred
            defer_seconds: How long to defer (seconds)
        """
        defer_until = datetime.now() + timedelta(seconds=defer_seconds)
        
        deferred = DeferredCommunication(
            urge=urge,
            reason=reason,
            defer_until=defer_until
        )
        
        self.deferred_queue.append(deferred)
        
        # Maintain queue size limit
        if len(self.deferred_queue) > self.max_deferred:
            # Remove oldest deferred item
            self.deferred_queue.sort(key=lambda d: d.deferred_at)
            removed = self.deferred_queue.pop(0)
            logger.debug(f"Dropped oldest deferred communication: {removed.reason}")
        
        logger.debug(f"Deferred communication until {defer_until.strftime('%H:%M:%S')}: {reason}")
    
    def _log_decision(self, result: DecisionResult) -> None:
        """
        Log decision to history.
        
        Args:
            result: Decision result to log
        """
        self.decision_history.append(result)
        
        # Maintain history size
        if len(self.decision_history) > self.history_size:
            self.decision_history = self.decision_history[-self.history_size:]
        
        # Log at appropriate level
        if result.decision == CommunicationDecision.SPEAK:
            logger.info(f"✅ SPEAK: {result.reason}")
        elif result.decision == CommunicationDecision.DEFER:
            logger.debug(f"⏸️  DEFER: {result.reason}")
        else:
            logger.debug(f"🔇 SILENCE: {result.reason}")
    
    def get_decision_summary(self) -> Dict[str, Any]:
        """Get summary of decision loop state."""
        recent_decisions = self.decision_history[-10:] if self.decision_history else []
        
        decision_counts = {
            CommunicationDecision.SPEAK: sum(1 for d in recent_decisions if d.decision == CommunicationDecision.SPEAK),
            CommunicationDecision.SILENCE: sum(1 for d in recent_decisions if d.decision == CommunicationDecision.SILENCE),
            CommunicationDecision.DEFER: sum(1 for d in recent_decisions if d.decision == CommunicationDecision.DEFER)
        }
        
        return {
            "deferred_queue_size": len(self.deferred_queue),
            "decision_history_size": len(self.decision_history),
            "last_decision": recent_decisions[-1] if recent_decisions else None,
            "recent_decisions": decision_counts,
            "ready_deferred": len([d for d in self.deferred_queue if d.is_ready()])
        }
