"""
Identity Auditor: Language-agnostic interface for identity/value auditing.

This module defines the abstract interface and data structures for auditing
identity consistency, value alignment, and protocol compliance. The interface
is designed to be implementable in any language (Python, C++, Rust) and
communicable over IPC boundaries using flat, serializable data structures.

Design Principles:
    - All inputs and outputs are plain data (no object references across boundary)
    - All strings are UTF-8, all floats are 64-bit, all timestamps are ISO-8601
    - Verdicts are deterministic given the same inputs
    - The auditor is stateless per call (state lives in the caller)
    - Every field has a fixed type — no polymorphic dicts crossing the boundary
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class AuditDomain(str, Enum):
    """Which aspect of identity is being audited."""
    VALUE_ALIGNMENT = "value_alignment"          # Actions vs charter values
    IDENTITY_CONSISTENCY = "identity_consistency" # Charter vs computed identity
    PROTOCOL_COMPLIANCE = "protocol_compliance"   # Action vs protocol constraints
    IDENTITY_DRIFT = "identity_drift"             # Temporal identity stability


class Severity(str, Enum):
    """Audit finding severity — maps to response urgency."""
    INFO = "info"            # Noteworthy but not concerning (< 0.3)
    WARNING = "warning"      # Worth attention (0.3 - 0.6)
    VIOLATION = "violation"  # Active misalignment (0.6 - 0.8)
    CRITICAL = "critical"    # Constitutional breach (> 0.8)


class VerdictOutcome(str, Enum):
    """Overall audit result."""
    PASS = "pass"        # No findings above threshold
    ADVISORY = "advisory" # Findings present but below action threshold
    FAIL = "fail"         # Findings require action (block, correct, alert)


# ── Input Structures (caller → auditor) ──────────────────────────────────────

@dataclass(frozen=True)
class CharterSnapshot:
    """
    Immutable snapshot of charter state for audit.

    Frozen so it can be safely shared across process/language boundaries.
    All fields are primitive types or tuples of primitives.
    """
    core_values: tuple[str, ...] = ()
    behavioral_guidelines: tuple[str, ...] = ()
    purpose_statement: str = ""


@dataclass(frozen=True)
class ComputedIdentitySnapshot:
    """
    Immutable snapshot of computed identity for audit.

    Behavioral tendencies are flattened to parallel tuples (name, value)
    rather than a dict, for serialization compatibility.
    """
    core_values: tuple[str, ...] = ()
    tendency_names: tuple[str, ...] = ()
    tendency_values: tuple[float, ...] = ()
    disposition_valence: float = 0.0
    disposition_arousal: float = 0.0
    disposition_dominance: float = 0.0
    source: str = "empty"  # "empty", "bootstrap", "computed"


@dataclass(frozen=True)
class DriftData:
    """Identity drift measurements."""
    has_drift: bool = False
    disposition_change: float = 0.0
    values_added: tuple[str, ...] = ()
    values_removed: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionSnapshot:
    """
    Immutable snapshot of a proposed or recent action for audit.

    Flattened to primitives — no Action object references cross the boundary.
    """
    action_type: str = ""
    content: str = ""
    claimed_capability: bool = False
    parameter_keys: tuple[str, ...] = ()
    parameter_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditInput:
    """
    Complete input bundle for an audit call.

    The caller assembles this from live system state. The auditor
    receives only this — no access to workspace, identity objects, etc.
    """
    charter: CharterSnapshot = field(default_factory=CharterSnapshot)
    computed_identity: ComputedIdentitySnapshot = field(
        default_factory=ComputedIdentitySnapshot
    )
    drift: DriftData = field(default_factory=DriftData)
    recent_actions: tuple[ActionSnapshot, ...] = ()
    proposed_action: Optional[ActionSnapshot] = None
    timestamp: str = ""  # ISO-8601
    cycle_number: int = 0


# ── Output Structures (auditor → caller) ─────────────────────────────────────

@dataclass(frozen=True)
class AuditFinding:
    """
    A single audit finding — one specific issue detected.

    Designed for structured logging and cross-language serialization.
    Every field is a primitive or enum with a string backing.
    """
    domain: AuditDomain
    severity: Severity
    code: str              # Machine-readable finding code, e.g. "CHARTER_VALUE_NOT_REFLECTED"
    description: str       # Human-readable explanation
    severity_score: float  # Continuous score 0.0 - 1.0
    evidence: str = ""     # What triggered this finding
    recommendation: str = "" # Suggested remediation


@dataclass(frozen=True)
class AuditVerdict:
    """
    Complete audit result — returned from every audit call.

    This is the single return type for all audit operations. The caller
    decides how to act on it (generate percept, block action, log, etc.)
    """
    outcome: VerdictOutcome
    domain: AuditDomain
    findings: tuple[AuditFinding, ...] = ()
    max_severity_score: float = 0.0
    timestamp: str = ""   # ISO-8601
    audit_duration_us: int = 0  # Microseconds — for performance monitoring


# ── Abstract Interface ───────────────────────────────────────────────────────

@runtime_checkable
class IdentityAuditor(Protocol):
    """
    Abstract interface for identity/value auditing.

    This Protocol defines the contract that any auditor implementation
    must fulfill — whether in Python, C++ (via pybind11), Rust (via PyO3),
    or as a sidecar process (via Unix domain socket / shared memory).

    Contract:
        - All methods are pure functions of their AuditInput argument
        - No side effects (no file I/O, no network, no state mutation)
        - Deterministic: same input → same output
        - Must complete within bounded time (suitable for real-time cycle)
    """

    def audit_value_alignment(self, audit_input: AuditInput) -> AuditVerdict:
        """
        Check recent actions against charter values.

        Detects:
            - Actions that contradict stated values (honesty, respect, etc.)
            - Capability claims without verification
            - Constitutional goals with inappropriately low priority
        """
        ...

    def audit_identity_consistency(self, audit_input: AuditInput) -> AuditVerdict:
        """
        Cross-check charter values against computed identity.

        Detects:
            - Charter values not reflected in computed behavior
            - Emergent values not present in charter
            - Behavioral tendencies that conflict with guidelines
        """
        ...

    def audit_identity_drift(self, audit_input: AuditInput) -> AuditVerdict:
        """
        Assess temporal stability of identity.

        Detects:
            - Significant disposition changes over time
            - Value additions/removals that may indicate instability
        """
        ...

    def audit_action_compliance(self, audit_input: AuditInput) -> AuditVerdict:
        """
        Check a proposed action against protocol constraints.

        Detects:
            - Actions that violate behavioral guidelines
            - Protocol prohibition breaches
            - Missing required conditions (consent, disclosure)
        """
        ...

    def full_audit(self, audit_input: AuditInput) -> list[AuditVerdict]:
        """
        Run all audit checks and return combined results.

        Convenience method that runs all four domain audits.
        Returns a list of verdicts, one per domain.
        """
        ...


# ── Helpers ──────────────────────────────────────────────────────────────────

def severity_from_score(score: float) -> Severity:
    """Map a continuous severity score to a Severity enum."""
    if score >= 0.8:
        return Severity.CRITICAL
    elif score >= 0.6:
        return Severity.VIOLATION
    elif score >= 0.3:
        return Severity.WARNING
    return Severity.INFO


def outcome_from_findings(findings: tuple[AuditFinding, ...]) -> VerdictOutcome:
    """Derive verdict outcome from findings."""
    if not findings:
        return VerdictOutcome.PASS
    max_score = max(f.severity_score for f in findings)
    if max_score >= 0.6:
        return VerdictOutcome.FAIL
    if max_score >= 0.3:
        return VerdictOutcome.ADVISORY
    return VerdictOutcome.PASS


def snapshot_from_charter(charter) -> CharterSnapshot:
    """
    Build a CharterSnapshot from a live CharterDocument object.

    This is the serialization boundary — everything after this point
    is pure data that can cross an IPC/FFI boundary.
    """
    if charter is None:
        return CharterSnapshot()
    return CharterSnapshot(
        core_values=tuple(getattr(charter, "core_values", []) or []),
        behavioral_guidelines=tuple(
            getattr(charter, "behavioral_guidelines", []) or []
        ),
        purpose_statement=getattr(charter, "purpose_statement", "") or "",
    )


def snapshot_from_computed_identity(computed) -> ComputedIdentitySnapshot:
    """Build a ComputedIdentitySnapshot from a live Identity object."""
    if computed is None or getattr(computed, "source", "empty") == "empty":
        return ComputedIdentitySnapshot()
    tendencies = getattr(computed, "behavioral_tendencies", {}) or {}
    return ComputedIdentitySnapshot(
        core_values=tuple(getattr(computed, "core_values", []) or []),
        tendency_names=tuple(tendencies.keys()),
        tendency_values=tuple(tendencies.values()),
        disposition_valence=getattr(
            computed, "emotional_disposition", {}
        ).get("valence", 0.0),
        disposition_arousal=getattr(
            computed, "emotional_disposition", {}
        ).get("arousal", 0.0),
        disposition_dominance=getattr(
            computed, "emotional_disposition", {}
        ).get("dominance", 0.0),
        source=getattr(computed, "source", "empty"),
    )


def snapshot_from_drift(drift_dict: dict) -> DriftData:
    """Build a DriftData from the dict returned by IdentityManager.get_identity_drift()."""
    if not drift_dict:
        return DriftData()
    return DriftData(
        has_drift=drift_dict.get("has_drift", False),
        disposition_change=drift_dict.get("disposition_change", 0.0),
        values_added=tuple(drift_dict.get("value_changes", {}).get("added", [])),
        values_removed=tuple(drift_dict.get("value_changes", {}).get("removed", [])),
    )


def snapshot_from_action(action) -> ActionSnapshot:
    """Build an ActionSnapshot from a live Action object."""
    if action is None:
        return ActionSnapshot()
    action_type = action.type.value if hasattr(action.type, "value") else str(action.type)
    metadata = getattr(action, "metadata", {}) or {}
    params = getattr(action, "parameters", {}) or {}
    return ActionSnapshot(
        action_type=action_type,
        content=str(params.get("content", params.get("text", ""))),
        claimed_capability=bool(metadata.get("claimed_capability", False)),
        parameter_keys=tuple(str(k) for k in params.keys()),
        parameter_values=tuple(str(v) for v in params.values()),
    )
