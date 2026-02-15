"""Tests for BottleneckDetector module."""

import pytest
from mind.cognitive_core.meta_cognition.bottleneck_detector import (
    BottleneckDetector,
    BottleneckSignal,
    BottleneckState,
    BottleneckType,
)


class TestBottleneckDetector:
    """Tests for BottleneckDetector class."""

    def test_initialization_defaults(self):
        detector = BottleneckDetector()
        assert detector.workspace_threshold == 20
        assert detector.slowdown_factor == 2.0
        assert detector.resource_threshold == 0.9
        assert not detector.is_overloaded()
        assert detector.get_load() == 0.0

    def test_initialization_with_config(self):
        config = {"workspace_overload_threshold": 10, "resource_exhaustion_threshold": 0.8}
        detector = BottleneckDetector(config=config)
        assert detector.workspace_threshold == 10
        assert detector.resource_threshold == 0.8

    def test_resource_threshold_capped_at_99(self):
        """Threshold of 1.0 would cause division by zero."""
        detector = BottleneckDetector({"resource_exhaustion_threshold": 1.0})
        assert detector.resource_threshold == 0.99

    def test_no_bottleneck_normal_operation(self):
        detector = BottleneckDetector()
        state = detector.update(
            subsystem_timings={"perception": 10, "attention": 5},
            workspace_percept_count=5,
            goal_resource_utilization=0.3,
        )
        assert not state.is_bottlenecked
        assert len(state.active_bottlenecks) == 0
        assert state.recommendation == "normal_operation"

    def test_workspace_overload_detection(self):
        detector = BottleneckDetector({"workspace_overload_threshold": 10})
        state = detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=25,
            goal_resource_utilization=0.3,
        )
        workspace_bottlenecks = [
            b for b in state.active_bottlenecks
            if b.bottleneck_type == BottleneckType.WORKSPACE_OVERLOAD
        ]
        assert len(workspace_bottlenecks) == 1
        assert workspace_bottlenecks[0].severity > 0

    def test_resource_exhaustion_detection(self):
        detector = BottleneckDetector({"resource_exhaustion_threshold": 0.9})
        state = detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=5,
            goal_resource_utilization=0.95,
            waiting_goals=3,
        )
        resource_bottlenecks = [
            b for b in state.active_bottlenecks
            if b.bottleneck_type == BottleneckType.GOAL_RESOURCE_EXHAUSTION
        ]
        assert len(resource_bottlenecks) == 1
        assert "waiting" in resource_bottlenecks[0].description.lower()

    def test_cycle_overrun_detection(self):
        detector = BottleneckDetector({"cycle_duration_target_ms": 100})
        state = detector.update(
            subsystem_timings={"perception": 50, "attention": 40, "action": 60, "memory_consolidation": 100},
            workspace_percept_count=5,
            goal_resource_utilization=0.3,
        )
        cycle_bottlenecks = [
            b for b in state.active_bottlenecks
            if b.bottleneck_type == BottleneckType.CYCLE_OVERRUN
        ]
        assert len(cycle_bottlenecks) == 1

    def test_memory_lag_detection(self):
        detector = BottleneckDetector({"memory_lag_threshold_ms": 100})
        state = detector.update(
            subsystem_timings={"perception": 10, "memory_consolidation": 500},
            workspace_percept_count=5,
            goal_resource_utilization=0.3,
        )
        memory_bottlenecks = [
            b for b in state.active_bottlenecks
            if b.bottleneck_type == BottleneckType.MEMORY_LAG
        ]
        assert len(memory_bottlenecks) == 1

    def test_subsystem_slowdown_detection(self):
        detector = BottleneckDetector({"subsystem_slowdown_factor": 2.0})
        # Build baseline
        for _ in range(15):
            detector.update(
                subsystem_timings={"perception": 10, "attention": 5},
                workspace_percept_count=5,
                goal_resource_utilization=0.3,
            )
        # Trigger slowdown
        state = detector.update(
            subsystem_timings={"perception": 50, "attention": 5},
            workspace_percept_count=5,
            goal_resource_utilization=0.3,
        )
        slowdown_bottlenecks = [
            b for b in state.active_bottlenecks
            if b.bottleneck_type == BottleneckType.SUBSYSTEM_SLOWDOWN
        ]
        assert len(slowdown_bottlenecks) == 1
        assert slowdown_bottlenecks[0].source == "perception"

    def test_bottleneck_persistence_required(self):
        detector = BottleneckDetector({
            "workspace_overload_threshold": 10,
            "consecutive_cycles_for_bottleneck": 3,
        })
        # First two cycles - not yet bottlenecked
        for _ in range(2):
            state = detector.update(
                subsystem_timings={"perception": 10},
                workspace_percept_count=25,
                goal_resource_utilization=0.3,
            )
            assert not state.is_bottlenecked
        # Third cycle - now bottlenecked
        state = detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=25,
            goal_resource_utilization=0.3,
        )
        assert state.is_bottlenecked

    def test_bottleneck_clears_when_resolved(self):
        detector = BottleneckDetector({
            "workspace_overload_threshold": 10,
            "consecutive_cycles_for_bottleneck": 1,
        })
        # Trigger bottleneck
        state1 = detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=25,
            goal_resource_utilization=0.3,
        )
        assert state1.is_bottlenecked
        # Resolve
        state2 = detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=5,
            goal_resource_utilization=0.3,
        )
        assert not state2.is_bottlenecked

    def test_overall_load_computation(self):
        detector = BottleneckDetector({"workspace_overload_threshold": 20, "cycle_duration_target_ms": 100})
        # Low load
        state1 = detector.update(
            subsystem_timings={"perception": 10, "attention": 5},
            workspace_percept_count=5,
            goal_resource_utilization=0.2,
        )
        assert state1.overall_load < 0.3
        # High load
        state2 = detector.update(
            subsystem_timings={"perception": 100, "attention": 100},
            workspace_percept_count=30,
            goal_resource_utilization=0.9,
        )
        assert state2.overall_load > 0.7

    def test_communication_inhibition_check(self):
        detector = BottleneckDetector({
            "workspace_overload_threshold": 10,
            "consecutive_cycles_for_bottleneck": 1,
        })
        # Normal - no inhibition
        detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=5,
            goal_resource_utilization=0.3,
        )
        assert not detector.should_inhibit_communication()
        # Bottlenecked with high load
        detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=50,
            goal_resource_utilization=0.95,
        )
        assert detector.should_inhibit_communication()

    def test_introspection_text_generation(self):
        detector = BottleneckDetector({
            "workspace_overload_threshold": 10,
            "consecutive_cycles_for_bottleneck": 1,
        })
        # No bottleneck
        detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=5,
            goal_resource_utilization=0.3,
        )
        assert detector.get_introspection_text() is None
        # With bottleneck
        detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=30,
            goal_resource_utilization=0.3,
        )
        text = detector.get_introspection_text()
        assert text is not None
        assert "constraints" in text.lower()

    def test_recommendation_generation(self):
        detector = BottleneckDetector({
            "resource_exhaustion_threshold": 0.5,
            "consecutive_cycles_for_bottleneck": 1,
        })
        state = detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=5,
            goal_resource_utilization=0.95,
        )
        assert "goal" in state.recommendation.lower()

    def test_average_load_tracking(self):
        detector = BottleneckDetector()
        for i in range(10):
            detector.update(
                subsystem_timings={"perception": 10 + i * 5},
                workspace_percept_count=5 + i,
                goal_resource_utilization=0.3 + i * 0.05,
            )
        avg = detector.get_average_load(cycles=5)
        assert 0.0 < avg < 1.0

    def test_summary_generation(self):
        detector = BottleneckDetector({
            "workspace_overload_threshold": 10,
            "consecutive_cycles_for_bottleneck": 1,
        })
        detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=25,
            goal_resource_utilization=0.3,
        )
        summary = detector.get_summary()
        assert "is_bottlenecked" in summary
        assert "overall_load" in summary
        assert "recommendation" in summary
        assert "should_inhibit" in summary
        assert isinstance(summary["bottleneck_types"], list)

    # Edge case tests
    def test_negative_percept_count_clamped(self):
        """Negative inputs should be clamped to 0."""
        detector = BottleneckDetector()
        state = detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=-5,
            goal_resource_utilization=0.3,
        )
        assert state.overall_load >= 0

    def test_utilization_above_1_clamped(self):
        """Utilization > 1.0 should be clamped."""
        detector = BottleneckDetector()
        state = detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=5,
            goal_resource_utilization=1.5,
        )
        assert state.overall_load <= 1.0

    def test_negative_timings_clamped(self):
        """Negative timings should be clamped to 0."""
        detector = BottleneckDetector()
        state = detector.update(
            subsystem_timings={"perception": -100},
            workspace_percept_count=5,
            goal_resource_utilization=0.3,
        )
        assert state.overall_load >= 0

    def test_empty_timings_dict(self):
        """Empty timings should not cause errors."""
        detector = BottleneckDetector()
        state = detector.update(
            subsystem_timings={},
            workspace_percept_count=5,
            goal_resource_utilization=0.3,
        )
        assert not state.is_bottlenecked

    def test_exactly_at_threshold_no_detection(self):
        """Values exactly at threshold should not trigger detection."""
        detector = BottleneckDetector({"workspace_overload_threshold": 20})
        state = detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=20,  # Exactly at threshold
            goal_resource_utilization=0.3,
        )
        workspace_bottlenecks = [
            b for b in state.active_bottlenecks
            if b.bottleneck_type == BottleneckType.WORKSPACE_OVERLOAD
        ]
        assert len(workspace_bottlenecks) == 0

    def test_zero_workspace_threshold_handled(self):
        """Zero threshold should not cause division by zero."""
        detector = BottleneckDetector({"workspace_overload_threshold": 0})
        detector.workspace_threshold = 0  # Force to 0
        state = detector.update(
            subsystem_timings={"perception": 10},
            workspace_percept_count=5,
            goal_resource_utilization=0.3,
        )
        # Should detect overload (5 > 0) without crashing
        assert len(state.active_bottlenecks) > 0


class TestBottleneckSignal:
    def test_signal_creation(self):
        signal = BottleneckSignal(
            bottleneck_type=BottleneckType.WORKSPACE_OVERLOAD,
            severity=0.7, source="workspace", description="Test",
        )
        assert signal.bottleneck_type == BottleneckType.WORKSPACE_OVERLOAD
        assert signal.severity == 0.7
        assert signal.consecutive_cycles == 1

    def test_severity_clamping_high(self):
        signal = BottleneckSignal(
            bottleneck_type=BottleneckType.WORKSPACE_OVERLOAD,
            severity=1.5, source="workspace", description="Test",
        )
        assert signal.severity == 1.0

    def test_severity_clamping_low(self):
        signal = BottleneckSignal(
            bottleneck_type=BottleneckType.WORKSPACE_OVERLOAD,
            severity=-0.5, source="workspace", description="Test",
        )
        assert signal.severity == 0.0


class TestBottleneckState:
    def test_state_defaults(self):
        state = BottleneckState()
        assert not state.is_bottlenecked
        assert state.overall_load == 0.0
        assert state.active_bottlenecks == []
        assert state.recommendation == "normal_operation"

    def test_get_severity_with_bottlenecks(self):
        state = BottleneckState(active_bottlenecks=[
            BottleneckSignal(BottleneckType.WORKSPACE_OVERLOAD, 0.5, "workspace", "Test 1"),
            BottleneckSignal(BottleneckType.MEMORY_LAG, 0.8, "memory", "Test 2"),
        ])
        assert state.get_severity() == 0.8

    def test_get_severity_empty(self):
        state = BottleneckState()
        assert state.get_severity() == 0.0
