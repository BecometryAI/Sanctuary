"""
Communication module - Autonomous communication agency.

This module provides the systems for autonomous communication decisions:
- Drive System: Internal urges to communicate
- Inhibition System: Reasons not to communicate (future)
- Decision Loop: SPEAK/SILENCE/DEFER decisions (future)
"""

from .drive import (
    CommunicationDriveSystem,
    CommunicationUrge,
    DriveType
)

__all__ = [
    'CommunicationDriveSystem',
    'CommunicationUrge',
    'DriveType'
]
