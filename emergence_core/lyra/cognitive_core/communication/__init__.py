"""
Communication module - Autonomous communication agency.

This module provides the systems for autonomous communication decisions:
- Drive System: Internal urges to communicate
- Inhibition System: Reasons not to communicate
- Decision Loop: SPEAK/SILENCE/DEFER decisions
- Deferred Queue: Queue communications for better timing
"""

from .drive import (
    CommunicationDriveSystem,
    CommunicationUrge,
    DriveType
)

from .inhibition import (
    CommunicationInhibitionSystem,
    InhibitionFactor,
    InhibitionType
)

from .deferred import (
    DeferredQueue,
    DeferredCommunication,
    DeferralReason
)

from .decision import (
    CommunicationDecisionLoop,
    CommunicationDecision,
    DecisionResult
)

from .silence import (
    SilenceTracker,
    SilenceAction,
    SilenceType
)

from .rhythm import (
    ConversationalRhythmModel,
    ConversationPhase,
    ConversationTurn
)

from .proactive import (
    ProactiveInitiationSystem,
    OutreachOpportunity,
    OutreachTrigger
)

__all__ = [
    'CommunicationDriveSystem',
    'CommunicationUrge',
    'DriveType',
    'CommunicationInhibitionSystem',
    'InhibitionFactor',
    'InhibitionType',
    'DeferredQueue',
    'DeferredCommunication',
    'DeferralReason',
    'CommunicationDecisionLoop',
    'CommunicationDecision',
    'DecisionResult',
    'SilenceTracker',
    'SilenceAction',
    'SilenceType',
    'ConversationalRhythmModel',
    'ConversationPhase',
    'ConversationTurn',
    'ProactiveInitiationSystem',
    'OutreachOpportunity',
    'OutreachTrigger'
]
