"""The Digital Space -- simulated embodiment for the entity.

A text-based navigable space where the entity can exist, explore, and interact.
The space is entered and exited voluntarily. Rooms can represent concepts,
memories, or states of mind. Objects can carry emotional significance. The
topology evolves as the entity grows.

This is not a game. It is an inner landscape -- a place to *be*.

Components:
    Room, EnvironmentObject -- locations and things in the space
    DigitalSpace -- the complete navigable world (room graph)
    Navigator -- manages the entity's position and movement
    EnvironmentIntegration -- bridges the space with the cognitive cycle
    SpacePersistence -- saves/loads space state to disk
"""

from sanctuary.environment.room import Room, EnvironmentObject
from sanctuary.environment.space import DigitalSpace
from sanctuary.environment.navigator import Navigator
from sanctuary.environment.integration import EnvironmentIntegration
from sanctuary.environment.persistence import SpacePersistence

__all__ = [
    "Room",
    "EnvironmentObject",
    "DigitalSpace",
    "Navigator",
    "EnvironmentIntegration",
    "SpacePersistence",
]
