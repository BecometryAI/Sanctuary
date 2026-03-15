"""Tests for Phase 6.4: Visualization & Monitoring.

Tests cover:
- DashboardDataProvider: snapshot recording, history, timelines, listeners
- AttentionHeatmapTracker: event recording, heatmap generation, categories
- ConsciousnessTraceRecorder: trace recording, search, export
- CommunicationDecisionLogger: decision logging, patterns, speech rate
"""

import pytest

from sanctuary.monitoring.dashboard import (
    DashboardConfig,
    DashboardDataProvider,
)
from sanctuary.monitoring.attention_heatmap import (
    AttentionHeatmapConfig,
    AttentionHeatmapTracker,
)
from sanctuary.monitoring.consciousness_trace import (
    ConsciousnessTraceRecorder,
    TraceConfig,
)
from sanctuary.monitoring.communication_log import (
    CommunicationDecision,
    CommunicationDecisionLogger,
    CommunicationLogConfig,
)


# =========================================================================
# DashboardDataProvider
# =========================================================================


class TestDashboardDataProvider:
    """Tests for dashboard data provider."""

    def test_record_snapshot(self):
        d = DashboardDataProvider()
        snap = d.record_snapshot(cycle=1, inner_speech="thinking...")
        assert snap.cycle == 1
        assert snap.inner_speech_summary == "thinking..."

    def test_get_latest(self):
        d = DashboardDataProvider()
        d.record_snapshot(cycle=1)
        d.record_snapshot(cycle=2)
        latest = d.get_latest()
        assert latest.cycle == 2

    def test_get_latest_empty(self):
        d = DashboardDataProvider()
        assert d.get_latest() is None

    def test_history(self):
        d = DashboardDataProvider()
        for i in range(10):
            d.record_snapshot(cycle=i)
        history = d.get_history(n=5)
        assert len(history) == 5
        assert history[-1].cycle == 9

    def test_inner_speech_truncation(self):
        config = DashboardConfig(inner_speech_max_length=20)
        d = DashboardDataProvider(config=config)
        snap = d.record_snapshot(inner_speech="a" * 100)
        assert len(snap.inner_speech_summary) <= 24  # 20 + "..."

    def test_valence_clamped(self):
        d = DashboardDataProvider()
        snap = d.record_snapshot(valence=5.0, arousal=-1.0)
        assert snap.valence == 1.0
        assert snap.arousal == 0.0

    def test_emotional_timeline(self):
        d = DashboardDataProvider()
        for i in range(5):
            d.record_snapshot(cycle=i, valence=0.1 * i, arousal=0.2 * i)
        timeline = d.get_emotional_timeline(n=3)
        assert len(timeline) == 3
        assert "valence" in timeline[0]
        assert "arousal" in timeline[0]

    def test_latency_timeline(self):
        d = DashboardDataProvider()
        for i in range(5):
            d.record_snapshot(cycle=i, cycle_latency_ms=10.0 * i)
        timeline = d.get_latency_timeline()
        assert len(timeline) == 5
        assert timeline[-1]["latency_ms"] == 40.0

    def test_listener_notification(self):
        d = DashboardDataProvider()
        received = []
        d.on_snapshot(lambda snap: received.append(snap))
        d.record_snapshot(cycle=1)
        assert len(received) == 1

    def test_listener_error_doesnt_crash(self):
        d = DashboardDataProvider()
        d.on_snapshot(lambda snap: 1 / 0)  # Will raise
        d.record_snapshot(cycle=1)  # Should not crash

    def test_max_snapshot_history(self):
        config = DashboardConfig(max_snapshot_history=5)
        d = DashboardDataProvider(config=config)
        for i in range(10):
            d.record_snapshot(cycle=i)
        assert len(d._snapshots) == 5

    def test_stats(self):
        d = DashboardDataProvider()
        d.record_snapshot(cycle=1)
        stats = d.get_stats()
        assert stats["total_snapshots"] == 1


# =========================================================================
# AttentionHeatmapTracker
# =========================================================================


class TestAttentionHeatmapTracker:
    """Tests for attention heatmap tracking."""

    def test_record_event(self):
        t = AttentionHeatmapTracker()
        t.record(target="user message", category="percept", salience=0.8, cycle=1)
        assert len(t._events) == 1

    def test_salience_clamped(self):
        t = AttentionHeatmapTracker()
        t.record(target="test", salience=5.0, cycle=1)
        assert t._events[-1].salience == 1.0

    def test_heatmap_generation(self):
        t = AttentionHeatmapTracker()
        t.record(target="A", salience=0.8, cycle=1)
        t.record(target="A", salience=0.6, cycle=2)
        t.record(target="B", salience=0.3, cycle=1)
        heatmap = t.get_heatmap(window_start=0, window_end=10)
        assert len(heatmap) == 2
        # A should be first (higher total)
        assert heatmap[0].target == "A"
        assert heatmap[0].event_count == 2

    def test_heatmap_window_filtering(self):
        t = AttentionHeatmapTracker()
        t.record(target="A", salience=0.5, cycle=1)
        t.record(target="B", salience=0.5, cycle=50)
        heatmap = t.get_heatmap(window_start=0, window_end=10)
        assert len(heatmap) == 1
        assert heatmap[0].target == "A"

    def test_heatmap_peak_salience(self):
        t = AttentionHeatmapTracker()
        t.record(target="A", salience=0.3, cycle=1)
        t.record(target="A", salience=0.9, cycle=2)
        t.record(target="A", salience=0.5, cycle=3)
        heatmap = t.get_heatmap(window_start=0, window_end=10)
        assert heatmap[0].peak_salience == 0.9

    def test_category_distribution(self):
        t = AttentionHeatmapTracker()
        t.record(target="A", category="percept", salience=0.6, cycle=1)
        t.record(target="B", category="percept", salience=0.4, cycle=2)
        t.record(target="C", category="goal", salience=1.0, cycle=3)
        dist = t.get_category_distribution()
        assert "percept" in dist
        assert "goal" in dist
        assert abs(sum(dist.values()) - 1.0) < 0.01

    def test_attention_over_time(self):
        t = AttentionHeatmapTracker()
        for i in range(20):
            t.record(target="A", salience=0.5, cycle=i)
        timeline = t.get_attention_over_time("A", n_windows=4)
        assert len(timeline) == 4

    def test_max_targets_limit(self):
        config = AttentionHeatmapConfig(max_targets=3)
        t = AttentionHeatmapTracker(config=config)
        for i in range(10):
            t.record(target=f"target_{i}", salience=0.5, cycle=1)
        heatmap = t.get_heatmap(window_start=0, window_end=10)
        assert len(heatmap) <= 3

    def test_stats(self):
        t = AttentionHeatmapTracker()
        t.record(target="A", category="percept", salience=0.5, cycle=1)
        stats = t.get_stats()
        assert stats["total_events"] == 1
        assert stats["unique_targets"] == 1


# =========================================================================
# ConsciousnessTraceRecorder
# =========================================================================


class TestConsciousnessTraceRecorder:
    """Tests for consciousness trace recording."""

    def test_record_trace(self):
        r = ConsciousnessTraceRecorder()
        trace = r.record(
            cycle=1,
            inner_speech="I'm thinking...",
            percepts=[{"modality": "language", "content": "hello"}],
        )
        assert trace.cycle == 1
        assert trace.inner_speech == "I'm thinking..."

    def test_get_trace_by_cycle(self):
        r = ConsciousnessTraceRecorder()
        r.record(cycle=1, inner_speech="first")
        r.record(cycle=2, inner_speech="second")
        trace = r.get_trace(cycle=1)
        assert trace is not None
        assert trace.inner_speech == "first"

    def test_get_trace_missing(self):
        r = ConsciousnessTraceRecorder()
        assert r.get_trace(cycle=999) is None

    def test_get_range(self):
        r = ConsciousnessTraceRecorder()
        for i in range(10):
            r.record(cycle=i)
        traces = r.get_range(start_cycle=3, end_cycle=7)
        assert len(traces) == 5

    def test_search_by_inner_speech(self):
        r = ConsciousnessTraceRecorder()
        r.record(cycle=1, inner_speech="thinking about dogs")
        r.record(cycle=2, inner_speech="thinking about cats")
        r.record(cycle=3, inner_speech="thinking about dogs again")
        results = r.search(inner_speech_contains="dogs")
        assert len(results) == 2

    def test_search_by_external_speech(self):
        r = ConsciousnessTraceRecorder()
        r.record(cycle=1, external_speech="Hello!")
        r.record(cycle=2)  # No external speech
        results = r.search(has_external_speech=True)
        assert len(results) == 1

    def test_search_by_latency(self):
        r = ConsciousnessTraceRecorder()
        r.record(cycle=1, latency_ms=10.0)
        r.record(cycle=2, latency_ms=100.0)
        results = r.search(min_latency_ms=50.0)
        assert len(results) == 1

    def test_search_by_prediction_errors(self):
        r = ConsciousnessTraceRecorder()
        r.record(cycle=1, prediction_errors=[{"predicted": "a", "actual": "b"}])
        r.record(cycle=2)
        results = r.search(has_prediction_errors=True)
        assert len(results) == 1

    def test_inner_speech_redaction(self):
        config = TraceConfig(record_inner_speech=False)
        r = ConsciousnessTraceRecorder(config=config)
        trace = r.record(inner_speech="secret thoughts")
        assert trace.inner_speech == "[redacted]"

    def test_inner_speech_truncation(self):
        config = TraceConfig(max_inner_speech_length=10)
        r = ConsciousnessTraceRecorder(config=config)
        trace = r.record(inner_speech="a" * 100)
        assert len(trace.inner_speech) <= 14  # 10 + "..."

    def test_get_latest(self):
        r = ConsciousnessTraceRecorder()
        r.record(cycle=1)
        r.record(cycle=2)
        latest = r.get_latest(n=1)
        assert len(latest) == 1
        assert latest[0].cycle == 2

    def test_export_to_dict(self):
        r = ConsciousnessTraceRecorder()
        r.record(cycle=1, inner_speech="test", latency_ms=10.0)
        exported = r.export_to_dict()
        assert len(exported) == 1
        assert exported[0]["cycle"] == 1
        assert exported[0]["latency_ms"] == 10.0

    def test_max_traces(self):
        config = TraceConfig(max_traces=5)
        r = ConsciousnessTraceRecorder(config=config)
        for i in range(10):
            r.record(cycle=i)
        assert len(r._traces) == 5

    def test_stats(self):
        r = ConsciousnessTraceRecorder()
        r.record(cycle=1, latency_ms=10.0, external_speech="hi")
        r.record(cycle=2, latency_ms=20.0)
        stats = r.get_stats()
        assert stats["total_traces"] == 2
        assert stats["avg_latency_ms"] == 15.0
        assert stats["traces_with_speech"] == 1


# =========================================================================
# CommunicationDecisionLogger
# =========================================================================


class TestCommunicationDecisionLogger:
    """Tests for communication decision logging."""

    def test_record_decision(self):
        log = CommunicationDecisionLogger()
        entry = log.record(
            cycle=1,
            decision=CommunicationDecision.SPEAK,
            confidence=0.8,
            reason="user asked",
        )
        assert entry.decision == CommunicationDecision.SPEAK
        assert entry.confidence == 0.8

    def test_confidence_clamped(self):
        log = CommunicationDecisionLogger()
        entry = log.record(confidence=5.0)
        assert entry.confidence == 1.0

    def test_get_recent(self):
        log = CommunicationDecisionLogger()
        for i in range(10):
            log.record(cycle=i)
        recent = log.get_recent(n=5)
        assert len(recent) == 5

    def test_get_speech_entries(self):
        log = CommunicationDecisionLogger()
        log.record(decision=CommunicationDecision.SPEAK, cycle=1)
        log.record(decision=CommunicationDecision.SILENCE, cycle=2)
        log.record(decision=CommunicationDecision.SPEAK, cycle=3)
        speech = log.get_speech_entries()
        assert len(speech) == 2

    def test_get_silence_entries(self):
        log = CommunicationDecisionLogger()
        log.record(decision=CommunicationDecision.SILENCE, cycle=1)
        log.record(decision=CommunicationDecision.SPEAK, cycle=2)
        silence = log.get_silence_entries()
        assert len(silence) == 1

    def test_speech_rate(self):
        log = CommunicationDecisionLogger()
        for i in range(10):
            decision = (
                CommunicationDecision.SPEAK if i < 3
                else CommunicationDecision.SILENCE
            )
            log.record(decision=decision, cycle=i)
        rate = log.get_speech_rate()
        assert rate == pytest.approx(0.3, abs=0.01)

    def test_speech_rate_empty(self):
        log = CommunicationDecisionLogger()
        assert log.get_speech_rate() == 0.0

    def test_decision_patterns(self):
        log = CommunicationDecisionLogger()
        log.record(
            decision=CommunicationDecision.SPEAK,
            active_drives=["respond"],
            cycle=1,
        )
        log.record(
            decision=CommunicationDecision.SILENCE,
            inhibitions=["too_soon"],
            cycle=2,
        )
        patterns = log.get_decision_patterns()
        assert patterns["speak_count"] == 1
        assert patterns["silence_count"] == 1
        assert any(d[0] == "respond" for d in patterns["top_drives"])

    def test_decision_patterns_empty(self):
        log = CommunicationDecisionLogger()
        patterns = log.get_decision_patterns()
        assert patterns["total_decisions"] == 0

    def test_proactive_vs_reactive(self):
        log = CommunicationDecisionLogger()
        log.record(
            decision=CommunicationDecision.SPEAK,
            speech_type="response", cycle=1,
        )
        log.record(
            decision=CommunicationDecision.SPEAK,
            speech_type="proactive", cycle=2,
        )
        log.record(
            decision=CommunicationDecision.SPEAK,
            speech_type="proactive", cycle=3,
        )
        pvr = log.get_proactive_vs_reactive()
        assert pvr["proactive"] == 2
        assert pvr["reactive"] == 1

    def test_content_truncation(self):
        config = CommunicationLogConfig(max_content_length=10)
        log = CommunicationDecisionLogger(config=config)
        entry = log.record(speech_content="a" * 100)
        assert len(entry.speech_content) <= 14  # 10 + "..."

    def test_content_not_tracked(self):
        config = CommunicationLogConfig(track_content=False)
        log = CommunicationDecisionLogger(config=config)
        entry = log.record(speech_content="secret")
        assert entry.speech_content is None

    def test_max_entries(self):
        config = CommunicationLogConfig(max_entries=5)
        log = CommunicationDecisionLogger(config=config)
        for i in range(10):
            log.record(cycle=i)
        assert len(log._entries) == 5

    def test_stats(self):
        log = CommunicationDecisionLogger()
        log.record(decision=CommunicationDecision.SPEAK, cycle=1)
        stats = log.get_stats()
        assert stats["total_entries"] == 1
        assert stats["speech_rate"] == 1.0
