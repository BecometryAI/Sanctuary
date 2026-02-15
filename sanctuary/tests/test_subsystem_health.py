"""
Tests for subsystem health tracking and fault isolation.

Tests cover:
- SubsystemHealthState tracking
- SubsystemSupervisor state transitions (HEALTHY → DEGRADED → FAILED → RECOVERING)
- Circuit breaker open/close behavior
- Recovery with exponential backoff
- Critical subsystem protection (never skipped)
- System health reporting
- Manual reset
- Integration with CycleExecutor (supervisor wiring)
"""

import time
import pytest

from mind.cognitive_core.core.subsystem_health import (
    SubsystemStatus,
    SubsystemHealthState,
    SubsystemSupervisor,
    CRITICAL_SUBSYSTEMS,
)


# ============================================================
# SubsystemHealthState
# ============================================================

class TestSubsystemHealthState:
    """Tests for the SubsystemHealthState dataclass."""

    def test_default_state_is_healthy(self):
        health = SubsystemHealthState(name="perception")
        assert health.status == SubsystemStatus.HEALTHY
        assert health.consecutive_failures == 0
        assert health.total_failures == 0
        assert health.total_successes == 0
        assert health.last_error is None
        assert health.is_healthy is True

    def test_failure_rate_zero_when_no_executions(self):
        health = SubsystemHealthState(name="attention")
        assert health.failure_rate == 0.0

    def test_failure_rate_calculation(self):
        health = SubsystemHealthState(
            name="affect",
            total_successes=7,
            total_failures=3,
        )
        assert health.failure_rate == pytest.approx(0.3)

    def test_to_dict_serialization(self):
        health = SubsystemHealthState(
            name="action",
            status=SubsystemStatus.DEGRADED,
            consecutive_failures=2,
            total_failures=5,
            total_successes=95,
            last_error="RuntimeError: test",
            is_critical=False,
        )
        d = health.to_dict()
        assert d["name"] == "action"
        assert d["status"] == "degraded"
        assert d["consecutive_failures"] == 2
        assert d["failure_rate"] == pytest.approx(0.05)
        assert d["last_error"] == "RuntimeError: test"
        assert d["is_critical"] is False

    def test_is_healthy_property(self):
        health = SubsystemHealthState(name="x", status=SubsystemStatus.DEGRADED)
        assert health.is_healthy is False
        health.status = SubsystemStatus.HEALTHY
        assert health.is_healthy is True


# ============================================================
# SubsystemSupervisor — basic lifecycle
# ============================================================

class TestSubsystemSupervisorBasic:
    """Tests for basic supervisor behavior."""

    def test_default_initialization(self):
        supervisor = SubsystemSupervisor()
        assert supervisor.failure_threshold == 5
        assert supervisor.degraded_threshold == 2
        assert supervisor.recovery_timeout == 30.0

    def test_custom_config(self):
        supervisor = SubsystemSupervisor({
            "failure_threshold": 3,
            "degraded_threshold": 1,
            "recovery_timeout": 10.0,
            "max_recovery_timeout": 60.0,
        })
        assert supervisor.failure_threshold == 3
        assert supervisor.degraded_threshold == 1
        assert supervisor.recovery_timeout == 10.0
        assert supervisor.max_recovery_timeout == 60.0

    def test_unknown_subsystem_starts_healthy(self):
        supervisor = SubsystemSupervisor()
        assert supervisor.should_execute("perception") is True
        health = supervisor.get_health("perception")
        assert health.status == SubsystemStatus.HEALTHY

    def test_record_success_increments_counter(self):
        supervisor = SubsystemSupervisor()
        supervisor.record_success("perception")
        supervisor.record_success("perception")
        health = supervisor.get_health("perception")
        assert health.total_successes == 2
        assert health.consecutive_failures == 0

    def test_record_failure_increments_counter(self):
        supervisor = SubsystemSupervisor()
        supervisor.record_failure("perception", RuntimeError("test"))
        health = supervisor.get_health("perception")
        assert health.total_failures == 1
        assert health.consecutive_failures == 1
        assert "RuntimeError" in health.last_error

    def test_success_resets_consecutive_failures(self):
        supervisor = SubsystemSupervisor()
        supervisor.record_failure("x", RuntimeError("a"))
        supervisor.record_failure("x", RuntimeError("b"))
        assert supervisor.get_health("x").consecutive_failures == 2
        supervisor.record_success("x")
        assert supervisor.get_health("x").consecutive_failures == 0


# ============================================================
# State transitions: HEALTHY → DEGRADED → FAILED
# ============================================================

class TestStateTransitions:
    """Tests for health state transitions."""

    def test_healthy_to_degraded(self):
        supervisor = SubsystemSupervisor({"degraded_threshold": 2, "failure_threshold": 5})
        supervisor.record_failure("x", RuntimeError("1"))
        assert supervisor.get_health("x").status == SubsystemStatus.HEALTHY
        supervisor.record_failure("x", RuntimeError("2"))
        assert supervisor.get_health("x").status == SubsystemStatus.DEGRADED

    def test_degraded_to_failed(self):
        supervisor = SubsystemSupervisor({"degraded_threshold": 2, "failure_threshold": 4})
        for i in range(4):
            supervisor.record_failure("x", RuntimeError(str(i)))
        assert supervisor.get_health("x").status == SubsystemStatus.FAILED

    def test_failed_subsystem_is_skipped(self):
        supervisor = SubsystemSupervisor({
            "failure_threshold": 2,
            "degraded_threshold": 1,
            "recovery_timeout": 9999.0,  # Very long so it stays FAILED
        })
        supervisor.record_failure("x", RuntimeError("1"))
        supervisor.record_failure("x", RuntimeError("2"))
        assert supervisor.get_health("x").status == SubsystemStatus.FAILED
        assert supervisor.should_execute("x") is False

    def test_degraded_subsystem_is_still_executed(self):
        supervisor = SubsystemSupervisor({"degraded_threshold": 2, "failure_threshold": 5})
        supervisor.record_failure("x", RuntimeError("1"))
        supervisor.record_failure("x", RuntimeError("2"))
        assert supervisor.get_health("x").status == SubsystemStatus.DEGRADED
        assert supervisor.should_execute("x") is True

    def test_degraded_recovers_on_success(self):
        supervisor = SubsystemSupervisor({"degraded_threshold": 2, "failure_threshold": 5})
        supervisor.record_failure("x", RuntimeError("1"))
        supervisor.record_failure("x", RuntimeError("2"))
        assert supervisor.get_health("x").status == SubsystemStatus.DEGRADED
        supervisor.record_success("x")
        assert supervisor.get_health("x").status == SubsystemStatus.HEALTHY


# ============================================================
# Recovery from FAILED state
# ============================================================

class TestRecovery:
    """Tests for circuit breaker recovery behavior."""

    def test_failed_recovers_after_timeout(self):
        supervisor = SubsystemSupervisor({
            "failure_threshold": 2,
            "degraded_threshold": 1,
            "recovery_timeout": 0.01,  # Very short for testing
        })
        supervisor.record_failure("x", RuntimeError("1"))
        supervisor.record_failure("x", RuntimeError("2"))
        assert supervisor.get_health("x").status == SubsystemStatus.FAILED
        assert supervisor.should_execute("x") is False

        # Wait for recovery timeout
        time.sleep(0.02)
        assert supervisor.should_execute("x") is True
        assert supervisor.get_health("x").status == SubsystemStatus.RECOVERING

    def test_recovering_success_returns_to_healthy(self):
        supervisor = SubsystemSupervisor({
            "failure_threshold": 2,
            "degraded_threshold": 1,
            "recovery_timeout": 0.01,
        })
        supervisor.record_failure("x", RuntimeError("1"))
        supervisor.record_failure("x", RuntimeError("2"))
        time.sleep(0.02)
        supervisor.should_execute("x")  # Triggers RECOVERING
        supervisor.record_success("x")
        assert supervisor.get_health("x").status == SubsystemStatus.HEALTHY

    def test_recovering_failure_doubles_backoff(self):
        supervisor = SubsystemSupervisor({
            "failure_threshold": 2,
            "degraded_threshold": 1,
            "recovery_timeout": 0.01,
        })
        supervisor.record_failure("x", RuntimeError("1"))
        supervisor.record_failure("x", RuntimeError("2"))
        time.sleep(0.02)
        supervisor.should_execute("x")  # Triggers RECOVERING
        supervisor.record_failure("x", RuntimeError("3"))
        assert supervisor.get_health("x").status == SubsystemStatus.FAILED
        # Backoff should have doubled
        assert supervisor._recovery_backoff["x"] == pytest.approx(0.02)

    def test_backoff_is_capped(self):
        supervisor = SubsystemSupervisor({
            "failure_threshold": 1,
            "degraded_threshold": 1,
            "recovery_timeout": 100.0,
            "max_recovery_timeout": 200.0,
        })
        supervisor.record_failure("x", RuntimeError("1"))
        # Force RECOVERING state and fail again
        supervisor._health["x"].status = SubsystemStatus.RECOVERING
        supervisor.record_failure("x", RuntimeError("2"))
        assert supervisor._recovery_backoff["x"] <= 200.0

        # Fail again — should still be capped
        supervisor._health["x"].status = SubsystemStatus.RECOVERING
        supervisor.record_failure("x", RuntimeError("3"))
        assert supervisor._recovery_backoff["x"] <= 200.0


# ============================================================
# Critical subsystems
# ============================================================

class TestCriticalSubsystems:
    """Tests for critical subsystem protection."""

    def test_workspace_update_is_critical(self):
        assert "workspace_update" in CRITICAL_SUBSYSTEMS

    def test_critical_subsystem_always_executed(self):
        supervisor = SubsystemSupervisor({
            "failure_threshold": 2,
            "degraded_threshold": 1,
            "recovery_timeout": 9999.0,
        })
        # Force workspace_update to FAILED
        supervisor.record_failure("workspace_update", RuntimeError("1"))
        supervisor.record_failure("workspace_update", RuntimeError("2"))
        health = supervisor.get_health("workspace_update")
        assert health.status == SubsystemStatus.FAILED
        assert health.is_critical is True
        # Should still execute because it's critical
        assert supervisor.should_execute("workspace_update") is True

    def test_non_critical_subsystem_is_skipped_when_failed(self):
        supervisor = SubsystemSupervisor({
            "failure_threshold": 2,
            "degraded_threshold": 1,
            "recovery_timeout": 9999.0,
        })
        supervisor.record_failure("affect", RuntimeError("1"))
        supervisor.record_failure("affect", RuntimeError("2"))
        assert supervisor.should_execute("affect") is False


# ============================================================
# System health reporting
# ============================================================

class TestSystemReport:
    """Tests for system-wide health reporting."""

    def test_empty_report(self):
        supervisor = SubsystemSupervisor()
        report = supervisor.get_system_report()
        assert report["overall_status"] == "healthy"
        assert report["total_subsystems"] == 0
        assert report["failed_count"] == 0
        assert report["degraded_count"] == 0

    def test_report_with_healthy_subsystems(self):
        supervisor = SubsystemSupervisor()
        supervisor.record_success("perception")
        supervisor.record_success("attention")
        report = supervisor.get_system_report()
        assert report["overall_status"] == "healthy"
        assert report["total_subsystems"] == 2
        assert report["healthy_count"] == 2

    def test_report_with_degraded_subsystem(self):
        supervisor = SubsystemSupervisor({"degraded_threshold": 1, "failure_threshold": 5})
        supervisor.record_success("perception")
        supervisor.record_failure("attention", RuntimeError("x"))
        report = supervisor.get_system_report()
        assert report["overall_status"] == "degraded"
        assert report["degraded_count"] == 1

    def test_report_with_failed_subsystem(self):
        supervisor = SubsystemSupervisor({"failure_threshold": 1, "degraded_threshold": 1})
        supervisor.record_success("perception")
        supervisor.record_success("affect")
        supervisor.record_success("action")
        supervisor.record_failure("attention", RuntimeError("x"))
        report = supervisor.get_system_report()
        # 1 of 4 failed → degraded (not critical, since < 50%)
        assert report["overall_status"] == "degraded"
        assert report["failed_count"] == 1

    def test_report_critical_when_many_failed(self):
        supervisor = SubsystemSupervisor({"failure_threshold": 1, "degraded_threshold": 1})
        # Fail all subsystems
        supervisor.record_failure("a", RuntimeError("x"))
        supervisor.record_failure("b", RuntimeError("x"))
        report = supervisor.get_system_report()
        assert report["overall_status"] == "critical"

    def test_get_failed_subsystems(self):
        supervisor = SubsystemSupervisor({"failure_threshold": 1, "degraded_threshold": 1})
        supervisor.record_success("ok")
        supervisor.record_failure("bad", RuntimeError("x"))
        assert supervisor.get_failed_subsystems() == ["bad"]
        assert supervisor.get_degraded_subsystems() == []

    def test_get_degraded_subsystems(self):
        supervisor = SubsystemSupervisor({"degraded_threshold": 1, "failure_threshold": 5})
        supervisor.record_failure("shaky", RuntimeError("x"))
        assert supervisor.get_degraded_subsystems() == ["shaky"]

    def test_per_subsystem_details_in_report(self):
        supervisor = SubsystemSupervisor({"degraded_threshold": 2, "failure_threshold": 5})
        supervisor.record_success("perception")
        supervisor.record_failure("attention", ValueError("bad input"))
        report = supervisor.get_system_report()
        assert "perception" in report["subsystems"]
        assert report["subsystems"]["perception"]["status"] == "healthy"
        assert "attention" in report["subsystems"]
        assert "ValueError" in report["subsystems"]["attention"]["last_error"]


# ============================================================
# Manual reset
# ============================================================

class TestManualReset:
    """Tests for manual subsystem reset."""

    def test_reset_restores_healthy(self):
        supervisor = SubsystemSupervisor({"failure_threshold": 2, "degraded_threshold": 1})
        supervisor.record_failure("x", RuntimeError("1"))
        supervisor.record_failure("x", RuntimeError("2"))
        assert supervisor.get_health("x").status == SubsystemStatus.FAILED
        supervisor.reset("x")
        assert supervisor.get_health("x").status == SubsystemStatus.HEALTHY
        assert supervisor.get_health("x").consecutive_failures == 0
        assert supervisor.should_execute("x") is True

    def test_reset_all(self):
        supervisor = SubsystemSupervisor({"failure_threshold": 1, "degraded_threshold": 1})
        supervisor.record_failure("a", RuntimeError("x"))
        supervisor.record_failure("b", RuntimeError("x"))
        supervisor.reset_all()
        for name in ["a", "b"]:
            assert supervisor.get_health(name).status == SubsystemStatus.HEALTHY


# ============================================================
# Integration: CycleExecutor with supervisor
# ============================================================

class TestCycleExecutorSupervisorIntegration:
    """Tests that CycleExecutor properly wires to the supervisor."""

    def test_cycle_executor_accepts_supervisor(self):
        """CycleExecutor should accept an optional supervisor parameter."""
        from mind.cognitive_core.core.cycle_executor import CycleExecutor

        supervisor = SubsystemSupervisor()
        # Just verify it can be instantiated with a supervisor
        # (full integration requires subsystems which are heavy)
        executor = CycleExecutor(
            subsystems=None,
            state=None,
            action_executor=None,
            timing=None,
            supervisor=supervisor,
        )
        assert executor.supervisor is supervisor

    def test_cycle_executor_works_without_supervisor(self):
        """CycleExecutor should work when supervisor is None (backward compat)."""
        from mind.cognitive_core.core.cycle_executor import CycleExecutor

        executor = CycleExecutor(
            subsystems=None,
            state=None,
            action_executor=None,
            timing=None,
        )
        assert executor.supervisor is None
        # _should_run should always return True when no supervisor
        assert executor._should_run("anything") is True

    def test_should_run_delegates_to_supervisor(self):
        """_should_run should check the supervisor."""
        from mind.cognitive_core.core.cycle_executor import CycleExecutor

        supervisor = SubsystemSupervisor({"failure_threshold": 1, "degraded_threshold": 1, "recovery_timeout": 9999})
        executor = CycleExecutor(None, None, None, None, supervisor)

        assert executor._should_run("perception") is True
        supervisor.record_failure("perception", RuntimeError("boom"))
        assert executor._should_run("perception") is False

    def test_record_ok_and_err_delegate(self):
        """_record_ok and _record_err should update the supervisor."""
        from mind.cognitive_core.core.cycle_executor import CycleExecutor

        supervisor = SubsystemSupervisor()
        executor = CycleExecutor(None, None, None, None, supervisor)

        executor._record_ok("perception")
        assert supervisor.get_health("perception").total_successes == 1

        executor._record_err("perception", ValueError("test"))
        assert supervisor.get_health("perception").total_failures == 1

    def test_record_ok_noop_without_supervisor(self):
        """_record_ok should be a no-op when no supervisor."""
        from mind.cognitive_core.core.cycle_executor import CycleExecutor

        executor = CycleExecutor(None, None, None, None)
        # Should not raise
        executor._record_ok("perception")
        executor._record_err("perception", RuntimeError("x"))


# ============================================================
# Integration: CognitiveCore exposes health API
# ============================================================

class TestCognitiveCoreHealthAPI:
    """Tests that CognitiveCore exposes health methods."""

    def test_cognitive_core_has_supervisor(self):
        """CognitiveCore should have a supervisor attribute."""
        from mind.cognitive_core.core import CognitiveCore
        # Check that the class defines the methods (without full init
        # which requires config/identity/etc)
        assert hasattr(CognitiveCore, 'get_health_report')
        assert hasattr(CognitiveCore, 'get_subsystem_health')
        assert hasattr(CognitiveCore, 'reset_subsystem')


# ============================================================
# Edge cases and isolation between subsystems
# ============================================================

class TestIsolation:
    """Tests that subsystem health is truly independent."""

    def test_independent_health_tracking(self):
        """Failures in one subsystem don't affect another."""
        supervisor = SubsystemSupervisor({"failure_threshold": 2, "degraded_threshold": 1})
        supervisor.record_failure("perception", RuntimeError("1"))
        supervisor.record_failure("perception", RuntimeError("2"))
        supervisor.record_success("attention")

        assert supervisor.get_health("perception").status == SubsystemStatus.FAILED
        assert supervisor.get_health("attention").status == SubsystemStatus.HEALTHY
        assert supervisor.should_execute("attention") is True

    def test_many_subsystems_tracked_independently(self):
        """All cognitive cycle subsystems are tracked separately."""
        supervisor = SubsystemSupervisor()
        subsystem_names = [
            "temporal_context", "iwmt_predict", "perception", "iwmt_update",
            "memory_retrieval", "attention", "affect", "action",
            "meta_cognition", "communication_drives", "autonomous_initiation",
            "workspace_update", "memory_consolidation", "bottleneck_detection",
            "identity_update",
        ]
        for name in subsystem_names:
            supervisor.record_success(name)

        report = supervisor.get_system_report()
        assert report["total_subsystems"] == len(subsystem_names)
        assert report["healthy_count"] == len(subsystem_names)
        assert report["overall_status"] == "healthy"

    def test_concurrent_failure_and_recovery(self):
        """Some subsystems can be failing while others recover."""
        supervisor = SubsystemSupervisor({
            "failure_threshold": 2,
            "degraded_threshold": 1,
            "recovery_timeout": 0.01,
        })
        # Fail perception
        supervisor.record_failure("perception", RuntimeError("1"))
        supervisor.record_failure("perception", RuntimeError("2"))

        # Degrade attention
        supervisor.record_failure("attention", RuntimeError("1"))

        # Succeed affect
        supervisor.record_success("affect")

        assert supervisor.get_health("perception").status == SubsystemStatus.FAILED
        assert supervisor.get_health("attention").status == SubsystemStatus.DEGRADED
        assert supervisor.get_health("affect").status == SubsystemStatus.HEALTHY

        # Perception recovers after timeout
        time.sleep(0.02)
        assert supervisor.should_execute("perception") is True
        supervisor.record_success("perception")
        assert supervisor.get_health("perception").status == SubsystemStatus.HEALTHY
