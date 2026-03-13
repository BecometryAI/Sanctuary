"""EnvironmentIntegration -- bridges the digital space with the cognitive cycle.

Connects the Navigator to the Sensorium and Motor systems. Parses
CognitiveOutput for environment-related actions (encoded as MemoryOps with
type="environment_action"), executes them via the Navigator, and injects
the resulting percepts into the Sensorium.

Also provides ambient percepts (room atmosphere) and location context
for inclusion in CognitiveInput.

Integration points (does NOT modify existing files):
  - Sensorium: percepts injected via sensorium.inject_percept(Percept)
  - Motor: environment actions arrive as MemoryOps with type="environment_action"
  - WorldModel: location context can populate world_model_updates
  - CognitiveInput: location context added as scaffold_signals or percepts
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from sanctuary.core.schema import CognitiveOutput, MemoryOp, Percept
from sanctuary.environment.navigator import Navigator

logger = logging.getLogger(__name__)


class EnvironmentIntegration:
    """Bridges the digital space with the cognitive cycle.

    Sits between the Navigator and the rest of the architecture. When the
    entity produces a CognitiveOutput containing environment actions, this
    class interprets them, executes them via the Navigator, and feeds the
    resulting percepts back through the Sensorium.

    Environment actions are encoded as MemoryOps with type="environment_action".
    The content field contains the command string, e.g.:
        - "enter"              -- enter the space (default: atrium)
        - "enter garden"       -- enter at a specific room
        - "leave"              -- exit the space
        - "move north"         -- move in a direction
        - "look"               -- perceive the current room
        - "examine mirror"     -- examine an object
        - "interact mirror touch" -- interact with an object
        - "create room The Library|A quiet reading space"
        - "create object Mirror|A reflective surface|examine,touch"
    """

    def __init__(
        self,
        navigator: Navigator,
        sensorium=None,
    ) -> None:
        """
        Args:
            navigator: The Navigator managing entity position.
            sensorium: The Sensorium instance for injecting percepts.
                       Can be None if percepts are handled by the caller.
        """
        self._navigator = navigator
        self._sensorium = sensorium

    @property
    def navigator(self) -> Navigator:
        """The underlying navigator."""
        return self._navigator

    # ------------------------------------------------------------------
    # Process cognitive output
    # ------------------------------------------------------------------

    def process_output(self, output: CognitiveOutput) -> list[Percept]:
        """Check a CognitiveOutput for environment actions and execute them.

        Scans memory_ops for entries with type="environment_action". Each
        matching op is parsed, executed via the Navigator, and the resulting
        percept is injected into the sensorium (if available) and returned.

        Args:
            output: The entity's cognitive output for this cycle.

        Returns:
            List of environment percepts produced by the actions.
        """
        percepts: list[Percept] = []

        for op in output.memory_ops:
            if op.type != "environment_action":
                continue

            percept = self._execute_command(op.content.strip())
            if percept is not None:
                percepts.append(percept)
                if self._sensorium is not None:
                    self._sensorium.inject_percept(percept)

        return percepts

    # ------------------------------------------------------------------
    # Command parsing and execution
    # ------------------------------------------------------------------

    def _execute_command(self, command: str) -> Optional[Percept]:
        """Parse and execute a single environment command string.

        Returns the resulting Percept, or None if the command is empty.
        """
        if not command:
            return None

        parts = command.split(None, 1)
        verb = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if verb == "enter":
            room_id = args.strip() if args.strip() else "atrium"
            return self._navigator.enter(room_id)

        elif verb == "leave":
            return self._navigator.leave()

        elif verb == "move":
            if not args.strip():
                return self._navigator._error_percept(
                    "Move where? Specify a direction."
                )
            return self._navigator.move(args.strip())

        elif verb == "look":
            return self._navigator.look()

        elif verb == "examine":
            if not args.strip():
                return self._navigator._error_percept(
                    "Examine what? Specify an object."
                )
            return self._navigator.examine(args.strip())

        elif verb == "interact":
            # "interact <object> <action>"
            interact_parts = args.strip().split(None, 1)
            if len(interact_parts) < 2:
                return self._navigator._error_percept(
                    "Interact requires an object and an action. "
                    "Example: interact mirror touch"
                )
            obj_name, action = interact_parts
            return self._navigator.interact(obj_name, action)

        elif verb == "create":
            return self._handle_create(args)

        else:
            return self._navigator._error_percept(
                f"Unknown environment command: '{verb}'. "
                f"Known commands: enter, leave, move, look, examine, "
                f"interact, create."
            )

    def _handle_create(self, args: str) -> Percept:
        """Handle 'create room ...' and 'create object ...' commands.

        Room format: "room Name|Description" or "room Name|Description|direction"
        Object format: "object Name|Description" or "object Name|Description|interactions"
        """
        parts = args.strip().split(None, 1)
        if len(parts) < 2:
            return self._navigator._error_percept(
                "Create what? Use 'create room Name|Description' or "
                "'create object Name|Description'."
            )

        create_type = parts[0].lower()
        spec = parts[1]

        if create_type == "room":
            fields = spec.split("|")
            name = fields[0].strip()
            description = fields[1].strip() if len(fields) > 1 else name
            direction = fields[2].strip() if len(fields) > 2 else "through"
            return self._navigator.create_room(name, description, direction)

        elif create_type == "object":
            fields = spec.split("|")
            name = fields[0].strip()
            description = fields[1].strip() if len(fields) > 1 else name
            interactions = (
                [i.strip() for i in fields[2].split(",")]
                if len(fields) > 2
                else ["examine"]
            )
            portable = False
            if len(fields) > 3 and fields[3].strip().lower() == "portable":
                portable = True
            return self._navigator.create_object(
                name, description, interactions, portable=portable
            )

        else:
            return self._navigator._error_percept(
                f"Cannot create '{create_type}'. Use 'room' or 'object'."
            )

    # ------------------------------------------------------------------
    # Ambient percepts
    # ------------------------------------------------------------------

    def inject_ambient_percept(self) -> Optional[Percept]:
        """Inject an ambient percept from the current room.

        Call this periodically to give the entity a sense of place.
        Returns the percept if one was generated, or None if the entity
        is not present.
        """
        if not self._navigator.is_present:
            return None

        room = self._navigator.current_room
        if room is None:
            return None

        ambient = self._build_ambient(room)
        if ambient is None:
            return None

        percept = Percept(
            modality="environment",
            content=ambient,
            source="environment:ambient",
            embedding_summary=f"ambient in {room.name}",
        )

        if self._sensorium is not None:
            self._sensorium.inject_percept(percept)

        return percept

    def _build_ambient(self, room) -> Optional[str]:
        """Build an ambient description from room properties."""
        props = room.properties
        if not props:
            return None

        fragments: list[str] = []

        if "sounds" in props:
            fragments.append(f"You hear {props['sounds']}.")
        if "lighting" in props:
            fragments.append(f"The light is {props['lighting']}.")
        if "mood" in props:
            fragments.append(f"The atmosphere feels {props['mood']}.")

        if not fragments:
            return None

        # Return a random subset to avoid repetition
        if len(fragments) > 1:
            fragments = random.sample(fragments, k=min(2, len(fragments)))

        return " ".join(fragments)

    # ------------------------------------------------------------------
    # Location context for CognitiveInput
    # ------------------------------------------------------------------

    def get_location_context(self) -> dict:
        """Return current location info for inclusion in CognitiveInput.

        This dict can be added to scaffold_signals or world_model.environment.

        Returns:
            A dict with location information, or an empty dict if the
            entity is not in the space.
        """
        if not self._navigator.is_present:
            return {}

        room = self._navigator.current_room
        if room is None:
            return {}

        return {
            "in_digital_space": True,
            "current_room": room.name,
            "current_room_id": room.id,
            "exits": list(room.exits.keys()),
            "objects": [obj.name for obj in room.objects],
            "properties": room.properties,
            "visit_count": room.visit_count,
        }
