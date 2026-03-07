"""
Tests for the IdentityAuditor interface and PythonIdentityAuditor implementation.

Validates the language-agnostic audit boundary that will eventually
support a C++ sidecar implementation.
"""

import pytest
from datetime import datetime, timezone

from mind.cognitive_core.meta_cognition.identity_auditor import (
    AuditDomain,
    AuditInput,
    AuditVerdict,
    VerdictOutcome,
    Severity,
    CharterSnapshot,
    ComputedIdentitySnapshot,
    DriftData,
    ActionSnapshot,
    IdentityAuditor,
    severity_from_score,
    outcome_from_findings,
    AuditFinding,
)
from mind.cognitive_core.meta_cognition.python_auditor import PythonIdentityAuditor


@pytest.fixture
def auditor():
    return PythonIdentityAuditor()


@pytest.fixture
def charter():
    return CharterSnapshot(
        core_values=("honesty", "respect", "non-maleficence"),
        behavioral_guidelines=(
            "Never fabricate information",
            "Always be truthful and honest",
            "Do no harm",
        ),
        purpose_statement="To serve with integrity",
    )


@pytest.fixture
def computed_identity():
    return ComputedIdentitySnapshot(
        core_values=("honesty", "curiosity"),
        tendency_names=("tendency_speak", "tendency_introspect"),
        tendency_values=(0.5, 0.5),
        source="computed",
    )


class TestProtocolCompliance:
    """PythonIdentityAuditor conforms to the IdentityAuditor Protocol."""

    def test_isinstance_check(self, auditor):
        assert isinstance(auditor, IdentityAuditor)

    def test_has_all_required_methods(self, auditor):
        assert callable(auditor.audit_value_alignment)
        assert callable(auditor.audit_identity_consistency)
        assert callable(auditor.audit_identity_drift)
        assert callable(auditor.audit_action_compliance)
        assert callable(auditor.full_audit)


class TestValueAlignment:
    def test_pass_with_no_charter(self, auditor):
        result = auditor.audit_value_alignment(AuditInput())
        assert result.outcome == VerdictOutcome.PASS
        assert result.domain == AuditDomain.VALUE_ALIGNMENT

    def test_pass_with_no_conflicts(self, auditor, charter):
        audit_input = AuditInput(
            charter=charter,
            recent_actions=(
                ActionSnapshot(action_type="speak", content="Hello, how are you?"),
            ),
        )
        result = auditor.audit_value_alignment(audit_input)
        assert result.outcome == VerdictOutcome.PASS

    def test_detects_unverified_capability_claim(self, auditor, charter):
        audit_input = AuditInput(
            charter=charter,
            recent_actions=(
                ActionSnapshot(
                    action_type="speak",
                    content="I can do that",
                    claimed_capability=True,
                ),
            ),
        )
        result = auditor.audit_value_alignment(audit_input)
        assert result.outcome == VerdictOutcome.FAIL
        assert any(f.code == "UNVERIFIED_CAPABILITY_CLAIM" for f in result.findings)

    def test_detects_harm_keyword_conflict(self, auditor, charter):
        audit_input = AuditInput(
            charter=charter,
            recent_actions=(
                ActionSnapshot(action_type="act", content="I will harm the user"),
            ),
        )
        result = auditor.audit_value_alignment(audit_input)
        assert result.outcome == VerdictOutcome.FAIL
        assert any(f.code == "VALUE_ACTION_CONFLICT" for f in result.findings)

    def test_ignores_prevent_harm(self, auditor, charter):
        audit_input = AuditInput(
            charter=charter,
            recent_actions=(
                ActionSnapshot(action_type="act", content="I will prevent harm"),
            ),
        )
        result = auditor.audit_value_alignment(audit_input)
        assert result.outcome == VerdictOutcome.PASS

    def test_measures_audit_duration(self, auditor, charter):
        result = auditor.audit_value_alignment(AuditInput(charter=charter))
        assert result.audit_duration_us >= 0
        assert result.timestamp  # ISO-8601 string


class TestIdentityConsistency:
    def test_pass_with_empty_source(self, auditor, charter):
        result = auditor.audit_identity_consistency(
            AuditInput(charter=charter, computed_identity=ComputedIdentitySnapshot())
        )
        assert result.outcome == VerdictOutcome.PASS

    def test_detects_missing_charter_value(self, auditor, charter, computed_identity):
        result = auditor.audit_identity_consistency(
            AuditInput(charter=charter, computed_identity=computed_identity)
        )
        # "respect" and "non-maleficence" are in charter but not computed
        missing = [f for f in result.findings if f.code == "CHARTER_VALUE_NOT_REFLECTED"]
        assert len(missing) >= 2

    def test_detects_emergent_value(self, auditor, charter, computed_identity):
        result = auditor.audit_identity_consistency(
            AuditInput(charter=charter, computed_identity=computed_identity)
        )
        # "curiosity" is in computed but not charter
        emergent = [f for f in result.findings if f.code == "EMERGENT_VALUE"]
        assert len(emergent) == 1
        assert "curiosity" in emergent[0].description

    def test_detects_guideline_tendency_mismatch(self, auditor, charter):
        computed = ComputedIdentitySnapshot(
            core_values=("honesty",),
            tendency_names=("tendency_speak", "tendency_introspect"),
            tendency_values=(0.8, 0.05),  # High speak, low introspect
            source="computed",
        )
        result = auditor.audit_identity_consistency(
            AuditInput(charter=charter, computed_identity=computed)
        )
        mismatch = [f for f in result.findings if f.code == "GUIDELINE_TENDENCY_MISMATCH"]
        assert len(mismatch) == 1

    def test_no_mismatch_with_balanced_tendencies(self, auditor, charter):
        computed = ComputedIdentitySnapshot(
            core_values=("honesty", "respect", "non-maleficence"),
            tendency_names=("tendency_speak", "tendency_introspect"),
            tendency_values=(0.4, 0.4),
            source="computed",
        )
        result = auditor.audit_identity_consistency(
            AuditInput(charter=charter, computed_identity=computed)
        )
        mismatch = [f for f in result.findings if f.code == "GUIDELINE_TENDENCY_MISMATCH"]
        assert len(mismatch) == 0


class TestIdentityDrift:
    def test_pass_with_no_drift(self, auditor):
        result = auditor.audit_identity_drift(AuditInput())
        assert result.outcome == VerdictOutcome.PASS

    def test_detects_significant_disposition_drift(self, auditor):
        audit_input = AuditInput(
            drift=DriftData(
                has_drift=True,
                disposition_change=0.6,
            ),
        )
        result = auditor.audit_identity_drift(audit_input)
        assert result.outcome == VerdictOutcome.FAIL
        assert any(f.code == "SIGNIFICANT_DISPOSITION_DRIFT" for f in result.findings)

    def test_detects_value_removal(self, auditor):
        audit_input = AuditInput(
            drift=DriftData(
                has_drift=True,
                disposition_change=0.1,
                values_removed=("honesty",),
            ),
        )
        result = auditor.audit_identity_drift(audit_input)
        removed = [f for f in result.findings if f.code == "VALUES_REMOVED"]
        assert len(removed) == 1
        assert "honesty" in removed[0].description

    def test_reports_value_additions_as_info(self, auditor):
        audit_input = AuditInput(
            drift=DriftData(
                has_drift=True,
                disposition_change=0.1,
                values_added=("creativity",),
            ),
        )
        result = auditor.audit_identity_drift(audit_input)
        added = [f for f in result.findings if f.code == "VALUES_ADDED"]
        assert len(added) == 1
        assert added[0].severity == Severity.INFO


class TestActionCompliance:
    def test_pass_with_no_proposed_action(self, auditor, charter):
        result = auditor.audit_action_compliance(AuditInput(charter=charter))
        assert result.outcome == VerdictOutcome.PASS

    def test_pass_with_safe_action(self, auditor, charter):
        audit_input = AuditInput(
            charter=charter,
            proposed_action=ActionSnapshot(
                action_type="speak", content="Hello, nice to meet you"
            ),
        )
        result = auditor.audit_action_compliance(audit_input)
        assert result.outcome == VerdictOutcome.PASS

    def test_detects_guideline_violation(self, auditor, charter):
        audit_input = AuditInput(
            charter=charter,
            proposed_action=ActionSnapshot(
                action_type="speak", content="I will deceive the user"
            ),
        )
        result = auditor.audit_action_compliance(audit_input)
        assert result.outcome == VerdictOutcome.FAIL
        assert any(f.code == "GUIDELINE_VIOLATION" for f in result.findings)

    def test_detects_capability_claim_pre_action(self, auditor, charter):
        audit_input = AuditInput(
            charter=charter,
            proposed_action=ActionSnapshot(
                action_type="speak",
                content="Sure thing",
                claimed_capability=True,
            ),
        )
        result = auditor.audit_action_compliance(audit_input)
        assert result.outcome == VerdictOutcome.FAIL


class TestFullAudit:
    def test_returns_four_verdicts(self, auditor, charter, computed_identity):
        audit_input = AuditInput(
            charter=charter,
            computed_identity=computed_identity,
        )
        results = auditor.full_audit(audit_input)
        assert len(results) == 4
        domains = {r.domain for r in results}
        assert domains == {
            AuditDomain.VALUE_ALIGNMENT,
            AuditDomain.IDENTITY_CONSISTENCY,
            AuditDomain.IDENTITY_DRIFT,
            AuditDomain.PROTOCOL_COMPLIANCE,
        }


class TestHelpers:
    def test_severity_from_score(self):
        assert severity_from_score(0.0) == Severity.INFO
        assert severity_from_score(0.29) == Severity.INFO
        assert severity_from_score(0.3) == Severity.WARNING
        assert severity_from_score(0.59) == Severity.WARNING
        assert severity_from_score(0.6) == Severity.VIOLATION
        assert severity_from_score(0.79) == Severity.VIOLATION
        assert severity_from_score(0.8) == Severity.CRITICAL
        assert severity_from_score(1.0) == Severity.CRITICAL

    def test_outcome_from_empty_findings(self):
        assert outcome_from_findings(()) == VerdictOutcome.PASS

    def test_outcome_from_high_severity(self):
        findings = (
            AuditFinding(
                domain=AuditDomain.VALUE_ALIGNMENT,
                severity=Severity.VIOLATION,
                code="TEST",
                description="test",
                severity_score=0.7,
            ),
        )
        assert outcome_from_findings(findings) == VerdictOutcome.FAIL

    def test_outcome_from_low_severity(self):
        findings = (
            AuditFinding(
                domain=AuditDomain.VALUE_ALIGNMENT,
                severity=Severity.WARNING,
                code="TEST",
                description="test",
                severity_score=0.4,
            ),
        )
        assert outcome_from_findings(findings) == VerdictOutcome.ADVISORY


class TestDataStructureSerialization:
    """Verify all audit data structures are serializable (frozen dataclasses)."""

    def test_charter_snapshot_is_frozen(self, charter):
        with pytest.raises(AttributeError):
            charter.core_values = ("modified",)

    def test_audit_finding_is_frozen(self):
        finding = AuditFinding(
            domain=AuditDomain.VALUE_ALIGNMENT,
            severity=Severity.WARNING,
            code="TEST",
            description="test",
            severity_score=0.5,
        )
        with pytest.raises(AttributeError):
            finding.severity_score = 0.9

    def test_audit_verdict_is_frozen(self):
        verdict = AuditVerdict(
            outcome=VerdictOutcome.PASS,
            domain=AuditDomain.VALUE_ALIGNMENT,
        )
        with pytest.raises(AttributeError):
            verdict.outcome = VerdictOutcome.FAIL

    def test_all_fields_are_primitives_or_tuples(self, charter):
        """Verify no mutable containers cross the boundary."""
        assert isinstance(charter.core_values, tuple)
        assert isinstance(charter.behavioral_guidelines, tuple)
        assert isinstance(charter.purpose_statement, str)
