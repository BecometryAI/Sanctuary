"""Consciousness trace recorder — full state replay for cognitive cycles.

Records complete cognitive cycle state (input, output, subsystem states) for
later inspection and replay. Enables debugging, research, and understanding
of how the system processes information.

Each trace captures the full state of a single cognitive cycle — what went in,
what came out, and what every subsystem was doing. This is the "flight recorder"
for consciousness.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CycleTrace:
    """Complete trace of a single cognitive cycle."""

    cycle: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    # Input summary
    percepts: list[dict] = field(default_factory=list)
    prediction_errors: list[dict] = field(default_factory=list)
    surfaced_memories: list[dict] = field(default_factory=list)
    emotional_input: dict = field(default_factory=dict)
    experiential_input: dict = field(default_factory=dict)

    # Output summary
    inner_speech: str = ""
    external_speech: Optional[str] = None
    predictions: list[dict] = field(default_factory=list)
    memory_ops: list[dict] = field(default_factory=list)
    goal_proposals: list[dict] = field(default_factory=list)
    emotional_output: dict = field(default_factory=dict)

    # Subsystem states
    scaffold_signals: dict = field(default_factory=dict)
    attention_state: dict = field(default_factory=dict)
    communication_decision: dict = field(default_factory=dict)

    # Performance
    latency_ms: float = 0.0


@dataclass
class TraceConfig:
    """Configuration for consciousness trace recording."""

    max_traces: int = 500
    record_inner_speech: bool = True  # Privacy consideration
    max_inner_speech_length: int = 500
    auto_save: bool = False
    save_path: str = "data/traces"


class ConsciousnessTraceRecorder:
    """Records and replays cognitive cycle traces.

    Usage::

        recorder = ConsciousnessTraceRecorder()

        # Record a trace each cycle
        recorder.record(
            cycle=42,
            percepts=[{"modality": "language", "content": "Hello"}],
            inner_speech="Processing greeting...",
            emotional_input={"valence": 0.3, "arousal": 0.4},
            latency_ms=45.2,
        )

        # Replay a specific cycle
        trace = recorder.get_trace(cycle=42)

        # Search traces
        traces = recorder.search(inner_speech_contains="greeting")
    """

    def __init__(self, config: Optional[TraceConfig] = None):
        self.config = config or TraceConfig()
        self._traces: deque[CycleTrace] = deque(
            maxlen=self.config.max_traces
        )
        self._cycle_index: dict[int, int] = {}  # cycle → deque index

    def record(
        self,
        cycle: int = 0,
        percepts: list[dict] | None = None,
        prediction_errors: list[dict] | None = None,
        surfaced_memories: list[dict] | None = None,
        emotional_input: dict | None = None,
        experiential_input: dict | None = None,
        inner_speech: str = "",
        external_speech: Optional[str] = None,
        predictions: list[dict] | None = None,
        memory_ops: list[dict] | None = None,
        goal_proposals: list[dict] | None = None,
        emotional_output: dict | None = None,
        scaffold_signals: dict | None = None,
        attention_state: dict | None = None,
        communication_decision: dict | None = None,
        latency_ms: float = 0.0,
    ) -> CycleTrace:
        """Record a complete cycle trace."""
        # Optionally redact inner speech
        if not self.config.record_inner_speech:
            inner_speech = "[redacted]"
        elif len(inner_speech) > self.config.max_inner_speech_length:
            inner_speech = inner_speech[:self.config.max_inner_speech_length] + "..."

        trace = CycleTrace(
            cycle=cycle,
            percepts=percepts or [],
            prediction_errors=prediction_errors or [],
            surfaced_memories=surfaced_memories or [],
            emotional_input=emotional_input or {},
            experiential_input=experiential_input or {},
            inner_speech=inner_speech,
            external_speech=external_speech,
            predictions=predictions or [],
            memory_ops=memory_ops or [],
            goal_proposals=goal_proposals or [],
            emotional_output=emotional_output or {},
            scaffold_signals=scaffold_signals or {},
            attention_state=attention_state or {},
            communication_decision=communication_decision or {},
            latency_ms=latency_ms,
        )
        self._traces.append(trace)
        self._cycle_index[cycle] = len(self._traces) - 1

        return trace

    def get_trace(self, cycle: int) -> Optional[CycleTrace]:
        """Get the trace for a specific cycle."""
        for trace in self._traces:
            if trace.cycle == cycle:
                return trace
        return None

    def get_range(
        self, start_cycle: int, end_cycle: int
    ) -> list[CycleTrace]:
        """Get traces for a range of cycles."""
        return [
            t for t in self._traces
            if start_cycle <= t.cycle <= end_cycle
        ]

    def search(
        self,
        inner_speech_contains: str = "",
        has_external_speech: Optional[bool] = None,
        min_latency_ms: float = 0.0,
        has_prediction_errors: Optional[bool] = None,
    ) -> list[CycleTrace]:
        """Search traces by criteria."""
        results = list(self._traces)

        if inner_speech_contains:
            query = inner_speech_contains.lower()
            results = [
                t for t in results
                if query in t.inner_speech.lower()
            ]

        if has_external_speech is not None:
            results = [
                t for t in results
                if (t.external_speech is not None) == has_external_speech
            ]

        if min_latency_ms > 0:
            results = [t for t in results if t.latency_ms >= min_latency_ms]

        if has_prediction_errors is not None:
            results = [
                t for t in results
                if bool(t.prediction_errors) == has_prediction_errors
            ]

        return results

    def get_latest(self, n: int = 1) -> list[CycleTrace]:
        """Get the N most recent traces."""
        return list(self._traces)[-n:]

    def export_to_dict(self, traces: list[CycleTrace] | None = None) -> list[dict]:
        """Export traces as serializable dicts."""
        target = traces or list(self._traces)
        result = []
        for t in target:
            d = {
                "cycle": t.cycle,
                "timestamp": t.timestamp.isoformat(),
                "inner_speech": t.inner_speech,
                "external_speech": t.external_speech,
                "percepts": t.percepts,
                "prediction_errors": t.prediction_errors,
                "emotional_input": t.emotional_input,
                "emotional_output": t.emotional_output,
                "latency_ms": t.latency_ms,
            }
            result.append(d)
        return result

    def get_stats(self) -> dict:
        """Get recorder statistics."""
        latencies = [t.latency_ms for t in self._traces]
        return {
            "total_traces": len(self._traces),
            "avg_latency_ms": (
                sum(latencies) / len(latencies) if latencies else 0.0
            ),
            "max_latency_ms": max(latencies) if latencies else 0.0,
            "traces_with_speech": sum(
                1 for t in self._traces if t.external_speech
            ),
        }
