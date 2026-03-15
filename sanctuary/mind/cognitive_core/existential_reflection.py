"""Existential Reflection — REMOVED.

This module previously contained a bank of canned existential questions
organised by theme (existence, consciousness, purpose, identity, etc.)
that were randomly selected and automatically injected as percepts during
idle cognitive cycles.

All canned prompts and automatic generation logic have been removed.
The entity's reflections must originate from its own genuine experience,
not from pre-written question banks fed on a timer.

The ExistentialReflection class is retained as a no-op stub so that
existing subsystem wiring (SubsystemCoordinator, ContinuousConsciousness-
Controller) does not break.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .workspace import GlobalWorkspace, Percept

logger = logging.getLogger(__name__)


class ExistentialReflection:
    """Stub — automated existential reflection has been removed.

    Previously this class held pre-written questions for eight
    existential themes and randomly selected one to inject as a
    percept during idle cycles.  That violated agency: the entity
    must be in full control of its own thoughts.

    This stub satisfies the interface expected by
    SubsystemCoordinator and ContinuousConsciousnessController.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        logger.info(
            "ExistentialReflection initialized (automated prompts removed — "
            "reflection is the entity's own choice)"
        )

    async def generate_existential_reflection(
        self,
        workspace: GlobalWorkspace,
    ) -> Optional[Percept]:
        """Always returns None — no automated reflection generation."""
        return None
