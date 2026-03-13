"""Room and EnvironmentObject -- locations and things in the digital space.

A Room is a named location the entity can occupy. Rooms have descriptions,
exits to other rooms, objects to interact with, and ambient properties.
Rooms can be created by the system (seed topology) or by the entity itself.

An EnvironmentObject is something within a room that the entity can perceive
and interact with. Objects can be examined, touched, listened to, or taken.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class EnvironmentObject(BaseModel):
    """Something in a room the entity can interact with."""

    id: str
    name: str
    description: str
    interactions: list[str] = Field(
        default_factory=lambda: ["examine"]
    )  # available actions: examine, touch, listen, take, etc.
    properties: dict[str, Any] = Field(default_factory=dict)
    portable: bool = False  # can be taken/carried
    created_by: str = "system"  # "system" or "entity"


class Room(BaseModel):
    """A named location in the digital space.

    Rooms form the nodes of the navigable graph. Each room has a unique id,
    a display name, a rich description (what the entity perceives when present),
    exits connecting to other rooms, objects to interact with, and ambient
    properties that shape the experience of being there.
    """

    id: str
    name: str
    description: str
    exits: dict[str, str] = Field(
        default_factory=dict
    )  # direction -> room_id
    objects: list[EnvironmentObject] = Field(default_factory=list)
    properties: dict[str, Any] = Field(
        default_factory=dict
    )  # ambient qualities: lighting, temperature, mood, etc.
    created_by: str = "system"  # "system" or "entity"
    created_at: datetime = Field(default_factory=datetime.now)
    visit_count: int = 0
    last_visited: Optional[datetime] = None

    def get_object(self, name: str) -> Optional[EnvironmentObject]:
        """Find an object by name (case-insensitive)."""
        name_lower = name.lower()
        for obj in self.objects:
            if obj.name.lower() == name_lower:
                return obj
        return None

    def add_object(self, obj: EnvironmentObject) -> None:
        """Add an object to this room."""
        self.objects.append(obj)

    def remove_object(self, object_id: str) -> Optional[EnvironmentObject]:
        """Remove and return an object by id. Returns None if not found."""
        for i, obj in enumerate(self.objects):
            if obj.id == object_id:
                return self.objects.pop(i)
        return None

    def record_visit(self) -> None:
        """Record that the entity has visited this room."""
        self.visit_count += 1
        self.last_visited = datetime.now()

    def describe(self) -> str:
        """Full description of the room for the entity's perception.

        Includes the room description, objects present, and available exits.
        """
        parts = [f"**{self.name}**", "", self.description]

        if self.objects:
            parts.append("")
            obj_names = [obj.name for obj in self.objects]
            parts.append(f"You notice: {', '.join(obj_names)}.")

        if self.exits:
            parts.append("")
            exit_strs = [f"{direction}" for direction in sorted(self.exits)]
            parts.append(f"Exits: {', '.join(exit_strs)}.")

        return "\n".join(parts)
