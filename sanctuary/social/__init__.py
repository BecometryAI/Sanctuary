"""Social & interactive capabilities for Phase 6.3.

Capabilities for richer social interaction:
- Multi-party conversation: Group chats with turn-taking and addressee detection
- Voice prosody analysis: Extract emotional tone from audio features
- User modeling: Build profiles of interaction patterns and preferences per person
"""

from sanctuary.social.multi_party import MultiPartyManager
from sanctuary.social.prosody import ProsodyAnalyzer
from sanctuary.social.user_modeling import UserModeler

__all__ = [
    "MultiPartyManager",
    "ProsodyAnalyzer",
    "UserModeler",
]
