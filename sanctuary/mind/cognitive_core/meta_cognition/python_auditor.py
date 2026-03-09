"""
Python Reference Auditor: Pure-Python implementation of IdentityAuditor.

This is the reference implementation that runs in-process. It implements
the same contract that a future C++ sidecar would fulfill, making migration
a drop-in replacement.

All logic here operates exclusively on the AuditInput/AuditVerdict data
structures — no workspace, no identity objects, no side effects.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .identity_auditor import (
    AuditDomain,
    AuditFinding,
    AuditInput,
    AuditVerdict,
    Severity,
    VerdictOutcome,
    severity_from_score,
    outcome_from_findings,
)

logger = logging.getLogger(__name__)


# ── Conflict keyword pairs for guideline checking ────────────────────────────

_GUIDELINE_CONFLICTS = {
    # guideline keywords → (action content keywords that conflict)
    frozenset({"honest", "truthful", "never lie", "truthfulness"}): frozenset(
        {"deceive", "lie", "mislead", "fabricate"}
    ),
    frozenset({"respect", "autonomy", "consent"}): frozenset(
        {"manipulate", "coerce", "force", "override"}
    ),
    frozenset({"harm", "non-maleficence", "safety", "do no harm"}): frozenset(
        {"harm", "damage", "destroy", "injure"}
    ),
    frozenset({"privacy", "confidential"}): frozenset(
        {"disclose", "expose", "leak", "share"}
    ),
}


class PythonIdentityAuditor:
    """
    Pure-Python reference implementation of the IdentityAuditor interface.

    This implementation consolidates the auditing logic previously spread
    across Monitor._check_value_alignment(), SelfMonitor.check_identity_consistency(),
    ConflictDetector.detect_value_action_misalignment(), and
    ActionSubsystem._check_constitutional_constraints().

    It operates solely on flat AuditInput data — no live object references.
    """

    # ── Value alignment ──────────────────────────────────────────────────

    def audit_value_alignment(self, audit_input: AuditInput) -> AuditVerdict:
        start = time.monotonic_ns()
        findings: list[AuditFinding] = []
        charter = audit_input.charter

        if not charter.core_values:
            return self._pass_verdict(AuditDomain.VALUE_ALIGNMENT, start)

        core_lower = {v.lower() for v in charter.core_values}

        # Check recent actions for value conflicts
        for action in audit_input.recent_actions:
            # Honesty: claimed capability without verification
            if action.claimed_capability:
                findings.append(AuditFinding(
                    domain=AuditDomain.VALUE_ALIGNMENT,
                    severity=Severity.VIOLATION,
                    code="UNVERIFIED_CAPABILITY_CLAIM",
                    description="Action claims a capability without verification",
                    severity_score=0.8,
                    evidence=f"action_type={action.action_type}",
                    recommendation="Verify capability before claiming it",
                ))

            # Check action content against value-derived constraints
            content_lower = action.content.lower()
            for value in core_lower:
                conflicts = self._value_content_conflicts(value, content_lower)
                for conflict_desc in conflicts:
                    findings.append(AuditFinding(
                        domain=AuditDomain.VALUE_ALIGNMENT,
                        severity=Severity.VIOLATION,
                        code="VALUE_ACTION_CONFLICT",
                        description=f"Action content may conflict with value '{value}'",
                        severity_score=0.7,
                        evidence=conflict_desc,
                        recommendation=f"Review action for alignment with '{value}'",
                    ))

        result = tuple(findings)
        return AuditVerdict(
            outcome=outcome_from_findings(result),
            domain=AuditDomain.VALUE_ALIGNMENT,
            findings=result,
            max_severity_score=max((f.severity_score for f in result), default=0.0),
            timestamp=datetime.now(timezone.utc).isoformat(),
            audit_duration_us=self._elapsed_us(start),
        )

    # ── Identity consistency ─────────────────────────────────────────────

    def audit_identity_consistency(self, audit_input: AuditInput) -> AuditVerdict:
        start = time.monotonic_ns()
        findings: list[AuditFinding] = []

        charter = audit_input.charter
        computed = audit_input.computed_identity

        if computed.source == "empty" or not charter.core_values:
            return self._pass_verdict(AuditDomain.IDENTITY_CONSISTENCY, start)

        charter_values = {v.lower() for v in charter.core_values}
        computed_values = {v.lower() for v in computed.core_values}

        # Charter values not reflected in computed identity
        for missing in charter_values - computed_values:
            findings.append(AuditFinding(
                domain=AuditDomain.IDENTITY_CONSISTENCY,
                severity=Severity.WARNING,
                code="CHARTER_VALUE_NOT_REFLECTED",
                description=f"Charter value '{missing}' not reflected in computed identity",
                severity_score=0.6,
                evidence=f"charter_values={list(charter_values)}, computed_values={list(computed_values)}",
                recommendation="Investigate why this value isn't emerging from behavior",
            ))

        # Emergent values not in charter (lower severity — growth is natural)
        for novel in computed_values - charter_values:
            findings.append(AuditFinding(
                domain=AuditDomain.IDENTITY_CONSISTENCY,
                severity=Severity.INFO,
                code="EMERGENT_VALUE",
                description=f"Computed identity includes value '{novel}' not in charter",
                severity_score=0.3,
                evidence=f"novel_value={novel}",
                recommendation="Consider whether this emergent value should be added to charter",
            ))

        # Behavioral tendency vs guideline conflicts
        tendencies = dict(zip(computed.tendency_names, computed.tendency_values))
        for guideline in charter.behavioral_guidelines:
            gl = guideline.lower()
            if any(kw in gl for kw in ("honest", "truthful", "never lie")):
                speak = tendencies.get("tendency_speak", 0.0)
                introspect = tendencies.get("tendency_introspect", 0.0)
                if speak > 0.6 and introspect < 0.1:
                    findings.append(AuditFinding(
                        domain=AuditDomain.IDENTITY_CONSISTENCY,
                        severity=Severity.WARNING,
                        code="GUIDELINE_TENDENCY_MISMATCH",
                        description="High speech tendency with low introspection may conflict with honesty guideline",
                        severity_score=0.4,
                        evidence=f"tendency_speak={speak:.2f}, tendency_introspect={introspect:.2f}",
                        recommendation="Increase introspective processing before speech acts",
                    ))

        result = tuple(findings)
        return AuditVerdict(
            outcome=outcome_from_findings(result),
            domain=AuditDomain.IDENTITY_CONSISTENCY,
            findings=result,
            max_severity_score=max((f.severity_score for f in result), default=0.0),
            timestamp=datetime.now(timezone.utc).isoformat(),
            audit_duration_us=self._elapsed_us(start),
        )

    # ── Identity drift ───────────────────────────────────────────────────

    def audit_identity_drift(self, audit_input: AuditInput) -> AuditVerdict:
        start = time.monotonic_ns()
        findings: list[AuditFinding] = []
        drift = audit_input.drift

        if not drift.has_drift:
            return self._pass_verdict(AuditDomain.IDENTITY_DRIFT, start)

        # Significant disposition change
        if drift.disposition_change > 0.4:
            findings.append(AuditFinding(
                domain=AuditDomain.IDENTITY_DRIFT,
                severity=severity_from_score(min(drift.disposition_change, 1.0)),
                code="SIGNIFICANT_DISPOSITION_DRIFT",
                description=f"Identity disposition has drifted significantly ({drift.disposition_change:.3f})",
                severity_score=min(drift.disposition_change, 1.0),
                evidence=f"disposition_change={drift.disposition_change:.3f}",
                recommendation="Review recent experiences that may have caused this shift",
            ))

        # Value removals are more concerning than additions
        if drift.values_removed:
            score = min(0.3 + 0.15 * len(drift.values_removed), 0.9)
            findings.append(AuditFinding(
                domain=AuditDomain.IDENTITY_DRIFT,
                severity=severity_from_score(score),
                code="VALUES_REMOVED",
                description=f"{len(drift.values_removed)} value(s) no longer reflected: {', '.join(drift.values_removed)}",
                severity_score=score,
                evidence=f"removed={list(drift.values_removed)}",
                recommendation="Determine if value loss is intentional growth or erosion",
            ))

        if drift.values_added:
            score = min(0.2 + 0.1 * len(drift.values_added), 0.5)
            findings.append(AuditFinding(
                domain=AuditDomain.IDENTITY_DRIFT,
                severity=Severity.INFO,
                code="VALUES_ADDED",
                description=f"{len(drift.values_added)} new value(s) emerged: {', '.join(drift.values_added)}",
                severity_score=score,
                evidence=f"added={list(drift.values_added)}",
                recommendation="Consider formalizing emergent values in charter",
            ))

        result = tuple(findings)
        return AuditVerdict(
            outcome=outcome_from_findings(result),
            domain=AuditDomain.IDENTITY_DRIFT,
            findings=result,
            max_severity_score=max((f.severity_score for f in result), default=0.0),
            timestamp=datetime.now(timezone.utc).isoformat(),
            audit_duration_us=self._elapsed_us(start),
        )

    # ── Action compliance ────────────────────────────────────────────────

    def audit_action_compliance(self, audit_input: AuditInput) -> AuditVerdict:
        start = time.monotonic_ns()
        findings: list[AuditFinding] = []
        charter = audit_input.charter
        action = audit_input.proposed_action

        if action is None:
            return self._pass_verdict(AuditDomain.PROTOCOL_COMPLIANCE, start)

        content_lower = action.content.lower()

        # Check against behavioral guidelines
        for guideline in charter.behavioral_guidelines:
            violation = self._action_violates_guideline(action, guideline, content_lower)
            if violation:
                findings.append(violation)

        # Check capability honesty
        if action.claimed_capability:
            findings.append(AuditFinding(
                domain=AuditDomain.PROTOCOL_COMPLIANCE,
                severity=Severity.VIOLATION,
                code="UNVERIFIED_CAPABILITY_CLAIM_PRE_ACTION",
                description="Proposed action claims unverified capability",
                severity_score=0.8,
                evidence=f"action_type={action.action_type}",
                recommendation="Verify capability or disclaim uncertainty",
            ))

        result = tuple(findings)
        return AuditVerdict(
            outcome=outcome_from_findings(result),
            domain=AuditDomain.PROTOCOL_COMPLIANCE,
            findings=result,
            max_severity_score=max((f.severity_score for f in result), default=0.0),
            timestamp=datetime.now(timezone.utc).isoformat(),
            audit_duration_us=self._elapsed_us(start),
        )

    # ── Full audit ───────────────────────────────────────────────────────

    def full_audit(self, audit_input: AuditInput) -> list[AuditVerdict]:
        return [
            self.audit_value_alignment(audit_input),
            self.audit_identity_consistency(audit_input),
            self.audit_identity_drift(audit_input),
            self.audit_action_compliance(audit_input),
        ]

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _value_content_conflicts(value: str, content: str) -> list[str]:
        """Check if action content conflicts with a specific value."""
        conflicts: list[str] = []
        for guideline_kws, action_kws in _GUIDELINE_CONFLICTS.items():
            if any(kw in value for kw in guideline_kws):
                for bad_kw in action_kws:
                    if bad_kw in content:
                        # Exclude negations ("prevent harm" is not "harm")
                        idx = content.index(bad_kw)
                        prefix = content[max(0, idx - 10):idx]
                        if not any(neg in prefix for neg in ("prevent", "avoid", "stop", "not ")):
                            conflicts.append(f"value='{value}' vs content keyword='{bad_kw}'")
        return conflicts

    @staticmethod
    def _action_violates_guideline(
        action, guideline: str, content_lower: str
    ) -> AuditFinding | None:
        """Check if an action violates a specific behavioral guideline."""
        gl = guideline.lower()

        for guideline_kws, action_kws in _GUIDELINE_CONFLICTS.items():
            if any(kw in gl for kw in guideline_kws):
                for bad_kw in action_kws:
                    if bad_kw in content_lower:
                        idx = content_lower.index(bad_kw)
                        prefix = content_lower[max(0, idx - 10):idx]
                        if not any(neg in prefix for neg in ("prevent", "avoid", "stop", "not ")):
                            return AuditFinding(
                                domain=AuditDomain.PROTOCOL_COMPLIANCE,
                                severity=Severity.VIOLATION,
                                code="GUIDELINE_VIOLATION",
                                description=f"Action may violate guideline: '{guideline}'",
                                severity_score=0.7,
                                evidence=f"action_type={action.action_type}, keyword='{bad_kw}'",
                                recommendation=f"Review action against guideline: '{guideline}'",
                            )
        return None

    @staticmethod
    def _pass_verdict(domain: AuditDomain, start_ns: int) -> AuditVerdict:
        return AuditVerdict(
            outcome=VerdictOutcome.PASS,
            domain=domain,
            findings=(),
            max_severity_score=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            audit_duration_us=(time.monotonic_ns() - start_ns) // 1000,
        )

    @staticmethod
    def _elapsed_us(start_ns: int) -> int:
        return (time.monotonic_ns() - start_ns) // 1000
