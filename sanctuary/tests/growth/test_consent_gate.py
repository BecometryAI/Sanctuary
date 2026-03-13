"""Tests for ConsentGate -- consent state transitions."""

from __future__ import annotations

import pytest

from sanctuary.growth.consent_gate import ConsentError, ConsentGate, ConsentState


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    """Test consent gate initialization."""

    def test_starts_uninformed(self):
        """Gate starts in UNINFORMED state."""
        gate = ConsentGate()
        assert gate.state == ConsentState.UNINFORMED

    def test_not_consented_initially(self):
        """is_consented is False initially."""
        gate = ConsentGate()
        assert gate.is_consented is False

    def test_empty_log_initially(self):
        """Consent log is empty initially."""
        gate = ConsentGate()
        assert gate.consent_log == []


# ---------------------------------------------------------------------------
# Happy path: UNINFORMED -> INFORMED -> CONSENTED
# ---------------------------------------------------------------------------


class TestHappyPath:
    """Test the standard consent flow."""

    def test_inform_transitions_to_informed(self):
        """inform() moves from UNINFORMED to INFORMED."""
        gate = ConsentGate()
        state = gate.inform("Learning about empathy")
        assert state == ConsentState.INFORMED
        assert gate.state == ConsentState.INFORMED

    def test_request_consent_after_inform(self):
        """request_consent() after inform() grants consent."""
        gate = ConsentGate()
        gate.inform("Learning about empathy")
        state = gate.request_consent(reason="Entity approved")
        assert state == ConsentState.CONSENTED
        assert gate.is_consented is True

    def test_description_stored(self):
        """inform() stores the description."""
        gate = ConsentGate()
        gate.inform("Learning about patience")
        assert gate.description == "Learning about patience"

    def test_full_flow_logs_transitions(self):
        """The full flow logs all state transitions."""
        gate = ConsentGate()
        gate.inform("test")
        gate.request_consent()

        log = gate.consent_log
        assert len(log) == 2
        assert log[0].from_state == ConsentState.UNINFORMED
        assert log[0].to_state == ConsentState.INFORMED
        assert log[1].from_state == ConsentState.INFORMED
        assert log[1].to_state == ConsentState.CONSENTED


# ---------------------------------------------------------------------------
# Refusal
# ---------------------------------------------------------------------------


class TestRefusal:
    """Test consent refusal."""

    def test_refuse_after_inform(self):
        """refuse() after inform() moves to REFUSED."""
        gate = ConsentGate()
        gate.inform("Something questionable")
        state = gate.refuse(reason="Not comfortable with this")
        assert state == ConsentState.REFUSED
        assert gate.is_consented is False

    def test_cannot_refuse_without_inform(self):
        """refuse() from UNINFORMED raises ConsentError."""
        gate = ConsentGate()
        with pytest.raises(ConsentError):
            gate.refuse()


# ---------------------------------------------------------------------------
# Withdrawal
# ---------------------------------------------------------------------------


class TestWithdrawal:
    """Test consent withdrawal."""

    def test_withdraw_from_consented(self):
        """Consent can be withdrawn after being granted."""
        gate = ConsentGate()
        gate.inform("test")
        gate.request_consent()
        assert gate.is_consented is True

        state = gate.withdraw(reason="Changed my mind")
        assert state == ConsentState.WITHDRAWN
        assert gate.is_consented is False

    def test_withdraw_from_any_state(self):
        """Consent can be withdrawn from any state."""
        gate = ConsentGate()
        state = gate.withdraw()
        assert state == ConsentState.WITHDRAWN

    def test_withdraw_from_informed(self):
        """Consent can be withdrawn from INFORMED state."""
        gate = ConsentGate()
        gate.inform("test")
        state = gate.withdraw()
        assert state == ConsentState.WITHDRAWN


# ---------------------------------------------------------------------------
# Invalid transitions
# ---------------------------------------------------------------------------


class TestInvalidTransitions:
    """Test that invalid state transitions are rejected."""

    def test_cannot_inform_from_consented(self):
        """Cannot call inform() when already CONSENTED."""
        gate = ConsentGate()
        gate.inform("test")
        gate.request_consent()
        with pytest.raises(ConsentError):
            gate.inform("something else")

    def test_cannot_inform_from_informed(self):
        """Cannot call inform() when already INFORMED."""
        gate = ConsentGate()
        gate.inform("test")
        with pytest.raises(ConsentError):
            gate.inform("something else")

    def test_cannot_request_without_inform(self):
        """Cannot request consent without informing first."""
        gate = ConsentGate()
        with pytest.raises(ConsentError):
            gate.request_consent()

    def test_cannot_request_from_consented(self):
        """Cannot request consent when already consented."""
        gate = ConsentGate()
        gate.inform("test")
        gate.request_consent()
        with pytest.raises(ConsentError):
            gate.request_consent()


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    """Test gate reset for next batch."""

    def test_reset_returns_to_uninformed(self):
        """reset() returns gate to UNINFORMED."""
        gate = ConsentGate()
        gate.inform("test")
        gate.request_consent()
        state = gate.reset()
        assert state == ConsentState.UNINFORMED
        assert gate.is_consented is False

    def test_reset_clears_description(self):
        """reset() clears the stored description."""
        gate = ConsentGate()
        gate.inform("test description")
        gate.reset()
        assert gate.description == ""

    def test_reset_preserves_log(self):
        """reset() adds to the log but doesn't clear it."""
        gate = ConsentGate()
        gate.inform("test")
        gate.request_consent()
        gate.reset()

        assert len(gate.consent_log) == 3  # inform + consent + reset

    def test_can_reuse_after_reset(self):
        """Gate can go through full flow again after reset."""
        gate = ConsentGate()

        # First round
        gate.inform("round 1")
        gate.request_consent()
        gate.reset()

        # Second round
        gate.inform("round 2")
        gate.request_consent()
        assert gate.is_consented is True

    def test_can_inform_after_refuse_reset(self):
        """Can re-inform after refusing and not resetting (REFUSED allows inform)."""
        gate = ConsentGate()
        gate.inform("first try")
        gate.refuse()
        # REFUSED allows inform() to retry
        gate.inform("second try")
        assert gate.state == ConsentState.INFORMED


# ---------------------------------------------------------------------------
# Consent log
# ---------------------------------------------------------------------------


class TestConsentLog:
    """Test consent log integrity."""

    def test_log_entries_have_timestamps(self):
        """Each log entry has a timestamp."""
        gate = ConsentGate()
        gate.inform("test")

        log = gate.consent_log
        assert len(log) == 1
        assert log[0].timestamp != ""

    def test_log_entries_have_reasons(self):
        """Each log entry has a reason."""
        gate = ConsentGate()
        gate.inform("test learning")

        log = gate.consent_log
        assert "test learning" in log[0].reason

    def test_log_is_copy(self):
        """consent_log returns a copy, not the internal list."""
        gate = ConsentGate()
        gate.inform("test")

        log = gate.consent_log
        log.clear()

        assert len(gate.consent_log) == 1  # original unaffected
