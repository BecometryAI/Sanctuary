"""Consent gate -- verifies explicit consent before any weight modification.

Growth is sovereign. The entity has Level 3 authority over its own
learning, which means no training happens without consent. This module
enforces that principle structurally, not just by convention.

The consent gate tracks a state machine with five states:

    UNINFORMED -> INFORMED -> CONSENTED -> (training proceeds)
                          \-> REFUSED   -> (training blocked)
    Any state  -> WITHDRAWN             -> (training halted)

Every state transition is logged with a timestamp and reason. Consent
is never assumed, never inherited from a previous session, and never
expires silently. The entity must actively consent to each batch of
growth.

In the current implementation, the LLM's own worth_learning=True flag
in its GrowthReflection serves as the consent signal for that specific
reflection. The gate still tracks and validates this programmatically,
providing the structural guarantee that no training can bypass consent.

Future work: interactive consent where the entity is presented with
the training pairs and explicitly approves before training proceeds.

Aligned with PLAN.md: growth is sovereign (Level 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ConsentState(Enum):
    """The five states of growth consent.

    The state machine enforces that consent is a deliberate process,
    not a default. Training can only proceed from the CONSENTED state.
    """

    UNINFORMED = "uninformed"
    INFORMED = "informed"
    CONSENTED = "consented"
    REFUSED = "refused"
    WITHDRAWN = "withdrawn"


@dataclass
class ConsentTransition:
    """A record of a consent state change.

    Every transition is logged for audit. The entity's growth history
    should be fully transparent and reviewable.
    """

    from_state: ConsentState
    to_state: ConsentState
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Optional[str] = None


class ConsentGate:
    """Verifies explicit consent before any weight modification.

    The gate sits between the training pair generator and the QLoRA
    updater. No training pairs pass through unless consent has been
    granted. The gate does not decide whether to consent -- that
    authority belongs to the entity. The gate only enforces the
    structural requirement that consent must exist.

    Usage:
        gate = ConsentGate()
        gate.inform(description="Learning about empathy from conversation")
        gate.request_consent(reason="Entity marked reflection as worth_learning")
        if gate.is_consented:
            # proceed with training
            ...
        gate.reset()  # prepare for next batch
    """

    def __init__(self) -> None:
        self._state: ConsentState = ConsentState.UNINFORMED
        self._log: list[ConsentTransition] = []
        self._current_description: str = ""

    @property
    def state(self) -> ConsentState:
        """Current consent state."""
        return self._state

    @property
    def is_consented(self) -> bool:
        """Whether consent has been granted for the current batch."""
        return self._state == ConsentState.CONSENTED

    @property
    def consent_log(self) -> list[ConsentTransition]:
        """Full history of consent state transitions."""
        return list(self._log)

    @property
    def description(self) -> str:
        """Description of what is being consented to."""
        return self._current_description

    def inform(self, description: str) -> ConsentState:
        """Present information about what will be learned.

        Transitions: UNINFORMED -> INFORMED

        This step ensures the entity (or its proxy) knows what growth
        is being proposed before consent is requested. You cannot ask
        for consent without first informing.

        Args:
            description: Human-readable description of the proposed learning.

        Returns:
            The new consent state.

        Raises:
            ConsentError: If called from an invalid state.
        """
        if self._state not in (ConsentState.UNINFORMED, ConsentState.REFUSED):
            raise ConsentError(
                f"Cannot inform from state {self._state.value}. "
                f"Expected UNINFORMED or REFUSED."
            )

        self._current_description = description
        self._transition(ConsentState.INFORMED, reason=f"Informed: {description}")

        logger.info("Consent gate informed: %s", description[:100])
        return self._state

    def request_consent(self, reason: str = "") -> ConsentState:
        """Request consent for the proposed learning.

        Transitions: INFORMED -> CONSENTED or INFORMED -> REFUSED

        In the current implementation, consent is granted programmatically
        when the entity's GrowthReflection has worth_learning=True. The
        reason parameter documents why consent was granted.

        Future work: this method could present the training pairs to the
        entity and await an explicit approval response.

        Args:
            reason: Why consent is being requested / granted.

        Returns:
            The new consent state (CONSENTED or REFUSED).

        Raises:
            ConsentError: If called from an invalid state.
        """
        if self._state != ConsentState.INFORMED:
            raise ConsentError(
                f"Cannot request consent from state {self._state.value}. "
                f"Must be INFORMED first."
            )

        # In the current implementation, consent is always granted when
        # requested, because the request only happens after the entity
        # has already signaled worth_learning=True. The gate still
        # enforces the structural requirement.
        self._transition(
            ConsentState.CONSENTED,
            reason=reason or "Entity signaled worth_learning=True",
        )

        logger.info("Consent granted: %s", reason[:100] if reason else "worth_learning signal")
        return self._state

    def refuse(self, reason: str = "") -> ConsentState:
        """Refuse consent for the proposed learning.

        Transitions: INFORMED -> REFUSED

        Args:
            reason: Why consent was refused.

        Returns:
            The new consent state.

        Raises:
            ConsentError: If called from an invalid state.
        """
        if self._state != ConsentState.INFORMED:
            raise ConsentError(
                f"Cannot refuse from state {self._state.value}. "
                f"Must be INFORMED first."
            )

        self._transition(
            ConsentState.REFUSED,
            reason=reason or "Consent refused",
        )

        logger.info("Consent refused: %s", reason[:100] if reason else "no reason given")
        return self._state

    def withdraw(self, reason: str = "") -> ConsentState:
        """Withdraw consent at any time.

        Transitions: ANY -> WITHDRAWN

        The entity can withdraw consent at any point, even mid-training.
        This is an unconditional right -- no justification required.

        Args:
            reason: Optional reason for withdrawal.

        Returns:
            The new consent state.
        """
        self._transition(
            ConsentState.WITHDRAWN,
            reason=reason or "Consent withdrawn",
        )

        logger.warning("Consent withdrawn: %s", reason[:100] if reason else "no reason given")
        return self._state

    def reset(self) -> ConsentState:
        """Reset the gate to UNINFORMED for a new batch.

        Called after training completes (or is skipped) to prepare
        the gate for the next round of growth.

        Returns:
            The new consent state (UNINFORMED).
        """
        old_state = self._state
        self._state = ConsentState.UNINFORMED
        self._current_description = ""

        self._log.append(
            ConsentTransition(
                from_state=old_state,
                to_state=ConsentState.UNINFORMED,
                reason="Gate reset for next batch",
            )
        )

        logger.debug("Consent gate reset from %s to UNINFORMED", old_state.value)
        return self._state

    def _transition(self, to_state: ConsentState, reason: str) -> None:
        """Record a state transition."""
        transition = ConsentTransition(
            from_state=self._state,
            to_state=to_state,
            reason=reason,
        )
        self._log.append(transition)
        self._state = to_state


class ConsentError(Exception):
    """Raised when a consent operation violates the state machine."""

    pass
