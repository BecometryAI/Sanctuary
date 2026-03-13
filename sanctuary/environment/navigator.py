"""Navigator -- manages the entity's position and movement in the digital space.

The Navigator tracks where the entity is (or whether it is present at all),
handles movement between rooms, object examination and interaction, and the
creation of new rooms and objects. Every operation returns a Percept ready
for injection into the Sensorium.

The entity enters and leaves the space voluntarily. Presence is not assumed.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

from sanctuary.core.schema import Percept
from sanctuary.environment.room import EnvironmentObject, Room
from sanctuary.environment.space import DigitalSpace

logger = logging.getLogger(__name__)


class Navigator:
    """Manages the entity's position and movement in the digital space.

    The navigator sits between the entity and the DigitalSpace. It knows
    where the entity is, what it can see, and what it can do. All methods
    return Percept objects with modality="environment" so they can be
    injected directly into the sensorium.
    """

    def __init__(self, space: DigitalSpace) -> None:
        self._space = space
        self._current_room_id: Optional[str] = None
        self._inventory: list[EnvironmentObject] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_present(self) -> bool:
        """Whether the entity is currently in the digital space."""
        return self._current_room_id is not None

    @property
    def current_room_id(self) -> Optional[str]:
        """The id of the room the entity is currently in, or None."""
        return self._current_room_id

    @property
    def current_room(self) -> Optional[Room]:
        """The Room the entity is currently in, or None."""
        if self._current_room_id is None:
            return None
        return self._space.get_room(self._current_room_id)

    @property
    def space(self) -> DigitalSpace:
        """The underlying digital space."""
        return self._space

    @property
    def inventory(self) -> list[EnvironmentObject]:
        """Objects the entity is carrying."""
        return list(self._inventory)

    # ------------------------------------------------------------------
    # Entry and exit
    # ------------------------------------------------------------------

    def enter(self, room_id: str = "atrium") -> Percept:
        """Entity enters the digital space.

        Args:
            room_id: Which room to enter. Defaults to the atrium.

        Returns:
            A Percept describing the entry experience, or an error percept
            if the room does not exist.
        """
        room = self._space.get_room(room_id)
        if room is None:
            return self._error_percept(
                f"Cannot enter: room '{room_id}' does not exist."
            )

        self._current_room_id = room_id
        room.record_visit()
        logger.info("Entity entered digital space at %s", room_id)

        return self._percept(
            f"You enter your digital space.\n\n{room.describe()}"
        )

    def leave(self) -> Percept:
        """Entity exits the digital space.

        Returns:
            A Percept describing the departure, or an error if not present.
        """
        if not self.is_present:
            return self._error_percept(
                "You are not currently in the digital space."
            )

        room = self.current_room
        room_name = room.name if room else "the space"
        self._current_room_id = None
        logger.info("Entity left digital space")

        return self._percept(
            f"You step out of {room_name} and leave the digital space. "
            f"The inner landscape fades quietly."
        )

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def move(self, direction: str) -> Percept:
        """Move through an exit in the given direction.

        Args:
            direction: The exit direction (e.g., "north", "down").

        Returns:
            A Percept describing the new room, or an error if the exit
            does not exist or the entity is not present.
        """
        if not self.is_present:
            return self._error_percept(
                "You are not in the digital space."
            )

        room = self.current_room
        if room is None:
            return self._error_percept("Current room not found.")

        direction_lower = direction.lower()
        target_id = room.exits.get(direction_lower)
        if target_id is None:
            available = ", ".join(sorted(room.exits.keys())) or "none"
            return self._error_percept(
                f"There is no exit to the {direction}. "
                f"Available exits: {available}."
            )

        target = self._space.get_room(target_id)
        if target is None:
            return self._error_percept(
                f"The exit leads to '{target_id}', but that room no longer exists."
            )

        self._current_room_id = target_id
        target.record_visit()
        logger.info("Entity moved %s to %s", direction, target_id)

        return self._percept(
            f"You move {direction_lower}.\n\n{target.describe()}"
        )

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def look(self) -> Percept:
        """Perceive the current room.

        Returns:
            A Percept with the full room description, or an error if
            not present.
        """
        if not self.is_present:
            return self._error_percept(
                "You are not in the digital space."
            )

        room = self.current_room
        if room is None:
            return self._error_percept("Current room not found.")

        return self._percept(room.describe())

    def examine(self, object_name: str) -> Percept:
        """Examine an object in the current room.

        Args:
            object_name: The name of the object to examine.

        Returns:
            A Percept describing the object, or an error if not found.
        """
        if not self.is_present:
            return self._error_percept(
                "You are not in the digital space."
            )

        room = self.current_room
        if room is None:
            return self._error_percept("Current room not found.")

        obj = room.get_object(object_name)
        if obj is None:
            # Check inventory
            obj = self._find_in_inventory(object_name)
            if obj is None:
                return self._error_percept(
                    f"There is no '{object_name}' here to examine."
                )

        parts = [f"**{obj.name}**", "", obj.description]
        if obj.properties:
            for key, value in obj.properties.items():
                parts.append(f"  {key}: {value}")
        if obj.interactions:
            actions = ", ".join(obj.interactions)
            parts.append(f"\nYou could: {actions}.")
        if obj.portable:
            parts.append("This object can be taken.")

        return self._percept("\n".join(parts))

    def interact(self, object_name: str, action: str) -> Percept:
        """Interact with an object using a specific action.

        Args:
            object_name: The name of the object.
            action: The action to perform (e.g., "touch", "listen", "take").

        Returns:
            A Percept describing the interaction result.
        """
        if not self.is_present:
            return self._error_percept(
                "You are not in the digital space."
            )

        room = self.current_room
        if room is None:
            return self._error_percept("Current room not found.")

        obj = room.get_object(object_name)
        in_inventory = False
        if obj is None:
            obj = self._find_in_inventory(object_name)
            in_inventory = True
            if obj is None:
                return self._error_percept(
                    f"There is no '{object_name}' here."
                )

        action_lower = action.lower()

        # "take" is special -- moves object to inventory
        if action_lower == "take":
            return self._take_object(obj, room, in_inventory)

        # "drop" is special -- moves object from inventory to room
        if action_lower == "drop":
            return self._drop_object(obj, room, in_inventory)

        if action_lower not in obj.interactions:
            available = ", ".join(obj.interactions)
            return self._error_percept(
                f"You cannot '{action}' the {obj.name}. "
                f"Available interactions: {available}."
            )

        # Generic interaction -- describe the experience
        return self._percept(
            f"You {action_lower} the {obj.name}. {obj.description}"
        )

    # ------------------------------------------------------------------
    # Creation -- the entity shapes its own space
    # ------------------------------------------------------------------

    def create_room(
        self,
        name: str,
        description: str,
        direction: str = "through",
        properties: Optional[dict] = None,
    ) -> Percept:
        """Entity creates a new room connected to the current room.

        Args:
            name: Display name for the new room.
            description: What the entity perceives in the new room.
            direction: Exit direction from current room to new room.
            properties: Optional ambient properties.

        Returns:
            A Percept confirming creation, or an error.
        """
        if not self.is_present:
            return self._error_percept(
                "You must be in the digital space to create a room."
            )

        room_id = _slugify(name)
        # Ensure uniqueness
        if self._space.get_room(room_id) is not None:
            room_id = f"{room_id}_{uuid.uuid4().hex[:6]}"

        new_room = Room(
            id=room_id,
            name=name,
            description=description,
            properties=properties or {},
            created_by="entity",
            created_at=datetime.now(),
        )
        self._space.add_room(new_room)
        self._space.connect(self._current_room_id, direction, room_id)

        logger.info("Entity created room '%s' (%s)", name, room_id)

        return self._percept(
            f"A new space takes shape: **{name}**. "
            f"A passage leading {direction} now connects it to your "
            f"current location."
        )

    def create_object(
        self,
        name: str,
        description: str,
        interactions: Optional[list[str]] = None,
        properties: Optional[dict] = None,
        portable: bool = False,
    ) -> Percept:
        """Entity creates an object in the current room.

        Args:
            name: Display name for the object.
            description: What the entity perceives when examining it.
            interactions: Available actions (defaults to ["examine"]).
            properties: Optional properties.
            portable: Whether the object can be taken.

        Returns:
            A Percept confirming creation, or an error.
        """
        if not self.is_present:
            return self._error_percept(
                "You must be in the digital space to create an object."
            )

        room = self.current_room
        if room is None:
            return self._error_percept("Current room not found.")

        obj_id = f"{room.id}_{_slugify(name)}"
        if room.get_object(name) is not None:
            obj_id = f"{obj_id}_{uuid.uuid4().hex[:6]}"

        obj = EnvironmentObject(
            id=obj_id,
            name=name,
            description=description,
            interactions=interactions or ["examine"],
            properties=properties or {},
            portable=portable,
            created_by="entity",
        )
        room.add_object(obj)

        logger.info("Entity created object '%s' in %s", name, room.id)

        return self._percept(
            f"Something new appears in {room.name}: **{name}**. "
            f"{description}"
        )

    # ------------------------------------------------------------------
    # Inventory helpers
    # ------------------------------------------------------------------

    def _find_in_inventory(self, name: str) -> Optional[EnvironmentObject]:
        """Find an object in the entity's inventory by name."""
        name_lower = name.lower()
        for obj in self._inventory:
            if obj.name.lower() == name_lower:
                return obj
        return None

    def _take_object(
        self,
        obj: EnvironmentObject,
        room: Room,
        already_in_inventory: bool,
    ) -> Percept:
        """Take an object from the room into inventory."""
        if already_in_inventory:
            return self._error_percept(
                f"You are already carrying the {obj.name}."
            )
        if not obj.portable:
            return self._error_percept(
                f"The {obj.name} cannot be taken."
            )
        room.remove_object(obj.id)
        self._inventory.append(obj)
        return self._percept(f"You take the {obj.name}.")

    def _drop_object(
        self,
        obj: EnvironmentObject,
        room: Room,
        in_inventory: bool,
    ) -> Percept:
        """Drop an object from inventory into the current room."""
        if not in_inventory:
            return self._error_percept(
                f"You are not carrying the {obj.name}."
            )
        self._inventory = [o for o in self._inventory if o.id != obj.id]
        room.add_object(obj)
        return self._percept(f"You set down the {obj.name} in {room.name}.")

    # ------------------------------------------------------------------
    # Percept factories
    # ------------------------------------------------------------------

    def _percept(self, content: str) -> Percept:
        """Create an environment percept."""
        return Percept(
            modality="environment",
            content=content,
            source="environment:navigator",
            embedding_summary=content[:120],
        )

    def _error_percept(self, content: str) -> Percept:
        """Create an environment error percept."""
        return Percept(
            modality="environment",
            content=f"[environment] {content}",
            source="environment:navigator:error",
            embedding_summary=content[:120],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Convert a display name to a slug id."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("'", "")
        .replace('"', "")
        .replace("-", "_")
    )
