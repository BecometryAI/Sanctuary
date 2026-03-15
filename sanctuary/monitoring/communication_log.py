"""Communication decision logger — tracking speak/silence decisions and reasons.

Records every communication decision the system makes: when it chose to speak,
when it chose to stay silent, and why. Enables understanding of the system's
communication agency — one of the most important aspects of autonomous behavior.

This is distinct from simply logging what was said. It captures the decision
process: what drives were active, what inhibitions were present, what the
final decision was, and how confident the system was.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class CommunicationDecision(str, Enum):
    """Types of communication decisions."""

    SPEAK = "speak"
    SILENCE = "silence"
    DEFER = "defer"  # Decided to speak later


@dataclass
class CommunicationLogEntry:
    """A single communication decision with full context."""

    cycle: int = 0
    decision: CommunicationDecision = CommunicationDecision.SILENCE
    confidence: float = 0.5

    # What drove the decision
    active_drives: list[str] = field(default_factory=list)
    drive_urgency: float = 0.0
    inhibitions: list[str] = field(default_factory=list)

    # Context
    had_external_input: bool = False
    was_addressed: bool = False
    emotional_state: str = ""

    # Outcome (if spoke)
    speech_content: Optional[str] = None
    speech_type: str = ""  # "response", "proactive", "deferred"

    # Reasoning
    reason: str = ""

    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CommunicationLogConfig:
    """Configuration for communication logging."""

    max_entries: int = 1000
    track_content: bool = True  # Whether to store speech content
    max_content_length: int = 200


class CommunicationDecisionLogger:
    """Logs and analyzes communication decisions.

    Usage::

        log = CommunicationDecisionLogger()

        # Record a decision
        log.record(
            cycle=42,
            decision=CommunicationDecision.SPEAK,
            confidence=0.8,
            active_drives=["respond_to_input"],
            reason="User asked a direct question",
            speech_content="Here's my answer...",
            speech_type="response",
        )

        # Analyze patterns
        rate = log.get_speech_rate()
        patterns = log.get_decision_patterns()
    """

    def __init__(self, config: Optional[CommunicationLogConfig] = None):
        self.config = config or CommunicationLogConfig()
        self._entries: deque[CommunicationLogEntry] = deque(
            maxlen=self.config.max_entries
        )

    def record(
        self,
        cycle: int = 0,
        decision: CommunicationDecision = CommunicationDecision.SILENCE,
        confidence: float = 0.5,
        active_drives: list[str] | None = None,
        drive_urgency: float = 0.0,
        inhibitions: list[str] | None = None,
        had_external_input: bool = False,
        was_addressed: bool = False,
        emotional_state: str = "",
        speech_content: Optional[str] = None,
        speech_type: str = "",
        reason: str = "",
    ) -> CommunicationLogEntry:
        """Record a communication decision."""
        content = speech_content
        if content and self.config.track_content:
            if len(content) > self.config.max_content_length:
                content = content[:self.config.max_content_length] + "..."
        elif not self.config.track_content:
            content = None

        entry = CommunicationLogEntry(
            cycle=cycle,
            decision=decision,
            confidence=max(0.0, min(1.0, confidence)),
            active_drives=active_drives or [],
            drive_urgency=max(0.0, min(1.0, drive_urgency)),
            inhibitions=inhibitions or [],
            had_external_input=had_external_input,
            was_addressed=was_addressed,
            emotional_state=emotional_state,
            speech_content=content,
            speech_type=speech_type,
            reason=reason,
        )
        self._entries.append(entry)
        return entry

    def get_recent(self, n: int = 20) -> list[CommunicationLogEntry]:
        """Get recent log entries."""
        return list(self._entries)[-n:]

    def get_speech_entries(self, n: int = 20) -> list[CommunicationLogEntry]:
        """Get recent entries where the system spoke."""
        speech = [
            e for e in self._entries
            if e.decision == CommunicationDecision.SPEAK
        ]
        return speech[-n:]

    def get_silence_entries(self, n: int = 20) -> list[CommunicationLogEntry]:
        """Get recent entries where the system chose silence."""
        silence = [
            e for e in self._entries
            if e.decision == CommunicationDecision.SILENCE
        ]
        return silence[-n:]

    def get_speech_rate(self, window: int = 100) -> float:
        """Get the fraction of recent cycles where the system spoke."""
        recent = list(self._entries)[-window:]
        if not recent:
            return 0.0
        speech = sum(
            1 for e in recent if e.decision == CommunicationDecision.SPEAK
        )
        return speech / len(recent)

    def get_decision_patterns(self) -> dict:
        """Analyze communication decision patterns."""
        entries = list(self._entries)
        if not entries:
            return {
                "total_decisions": 0,
                "speak_count": 0,
                "silence_count": 0,
                "defer_count": 0,
                "speech_rate": 0.0,
                "avg_confidence": 0.0,
                "top_drives": [],
                "top_inhibitions": [],
            }

        speak = sum(1 for e in entries if e.decision == CommunicationDecision.SPEAK)
        silence = sum(1 for e in entries if e.decision == CommunicationDecision.SILENCE)
        defer = sum(1 for e in entries if e.decision == CommunicationDecision.DEFER)

        # Count drives
        drive_counts: dict[str, int] = {}
        for e in entries:
            for drive in e.active_drives:
                drive_counts[drive] = drive_counts.get(drive, 0) + 1

        # Count inhibitions
        inhibition_counts: dict[str, int] = {}
        for e in entries:
            for inh in e.inhibitions:
                inhibition_counts[inh] = inhibition_counts.get(inh, 0) + 1

        top_drives = sorted(
            drive_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        top_inhibitions = sorted(
            inhibition_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "total_decisions": len(entries),
            "speak_count": speak,
            "silence_count": silence,
            "defer_count": defer,
            "speech_rate": speak / len(entries),
            "avg_confidence": sum(e.confidence for e in entries) / len(entries),
            "top_drives": top_drives,
            "top_inhibitions": top_inhibitions,
        }

    def get_proactive_vs_reactive(self) -> dict:
        """Analyze proactive vs reactive speech patterns."""
        speech_entries = [
            e for e in self._entries
            if e.decision == CommunicationDecision.SPEAK
        ]
        if not speech_entries:
            return {"proactive": 0, "reactive": 0, "deferred": 0}

        proactive = sum(1 for e in speech_entries if e.speech_type == "proactive")
        reactive = sum(1 for e in speech_entries if e.speech_type == "response")
        deferred = sum(1 for e in speech_entries if e.speech_type == "deferred")

        return {
            "proactive": proactive,
            "reactive": reactive,
            "deferred": deferred,
        }

    def get_stats(self) -> dict:
        """Get logger statistics."""
        patterns = self.get_decision_patterns()
        return {
            "total_entries": len(self._entries),
            "speech_rate": patterns["speech_rate"],
            "avg_confidence": patterns["avg_confidence"],
        }
