"""Tests for DigitalSpace -- topology, connections, serialization."""

import json
import pytest
from pathlib import Path

from sanctuary.environment.room import Room, EnvironmentObject
from sanctuary.environment.space import DigitalSpace


class TestDigitalSpace:
    """Tests for the DigitalSpace room graph."""

    def _make_room(self, room_id: str, name: str = "") -> Room:
        return Room(
            id=room_id,
            name=name or room_id.title(),
            description=f"The {room_id} room.",
        )

    def test_add_and_get_room(self):
        space = DigitalSpace()
        room = self._make_room("hall")
        space.add_room(room)
        assert space.get_room("hall") is not None
        assert space.get_room("hall").name == "Hall"

    def test_get_room_not_found(self):
        space = DigitalSpace()
        assert space.get_room("nonexistent") is None

    def test_room_ids(self):
        space = DigitalSpace()
        space.add_room(self._make_room("a"))
        space.add_room(self._make_room("b"))
        ids = space.room_ids
        assert "a" in ids
        assert "b" in ids
        assert len(ids) == 2

    def test_remove_room(self):
        space = DigitalSpace()
        space.add_room(self._make_room("a"))
        space.add_room(self._make_room("b"))
        space.connect("a", "east", "b")
        removed = space.remove_room("b")
        assert removed is not None
        assert space.get_room("b") is None
        # Exit from a should be cleaned up
        assert "east" not in space.get_room("a").exits

    def test_remove_room_not_found(self):
        space = DigitalSpace()
        assert space.remove_room("nonexistent") is None

    def test_connect_bidirectional(self):
        space = DigitalSpace()
        space.add_room(self._make_room("a"))
        space.add_room(self._make_room("b"))
        result = space.connect("a", "north", "b")
        assert result is True
        assert space.get_room("a").exits["north"] == "b"
        assert space.get_room("b").exits["south"] == "a"

    def test_connect_unidirectional(self):
        space = DigitalSpace()
        space.add_room(self._make_room("a"))
        space.add_room(self._make_room("b"))
        result = space.connect("a", "north", "b", bidirectional=False)
        assert result is True
        assert "north" in space.get_room("a").exits
        assert "south" not in space.get_room("b").exits

    def test_connect_missing_room(self):
        space = DigitalSpace()
        space.add_room(self._make_room("a"))
        result = space.connect("a", "north", "missing")
        assert result is False

    def test_get_exits(self):
        space = DigitalSpace()
        space.add_room(self._make_room("a"))
        space.add_room(self._make_room("b"))
        space.add_room(self._make_room("c"))
        space.connect("a", "north", "b")
        space.connect("a", "east", "c")
        exits = space.get_exits("a")
        assert exits == {"north": "b", "east": "c"}

    def test_get_exits_empty(self):
        space = DigitalSpace()
        assert space.get_exits("nonexistent") == {}

    def test_get_neighbors(self):
        space = DigitalSpace()
        space.add_room(self._make_room("a"))
        space.add_room(self._make_room("b"))
        space.add_room(self._make_room("c"))
        space.connect("a", "north", "b")
        space.connect("a", "east", "c")
        neighbors = space.get_neighbors("a")
        neighbor_ids = {r.id for r in neighbors}
        assert neighbor_ids == {"b", "c"}

    def test_get_neighbors_empty(self):
        space = DigitalSpace()
        assert space.get_neighbors("nonexistent") == []


class TestDigitalSpaceSerialization:
    """Tests for space serialization to/from dict and JSON files."""

    def test_to_dict_and_from_dict(self):
        space = DigitalSpace()
        room = Room(
            id="hall",
            name="The Hall",
            description="A grand hall.",
            properties={"lighting": "bright"},
        )
        room.add_object(
            EnvironmentObject(id="o1", name="Torch", description="Burning.")
        )
        space.add_room(room)

        data = space.to_dict()
        restored = DigitalSpace.from_dict(data)

        assert restored.get_room("hall") is not None
        assert restored.get_room("hall").name == "The Hall"
        assert len(restored.get_room("hall").objects) == 1

    def test_save_and_load(self, tmp_path):
        space = DigitalSpace()
        space.add_room(
            Room(id="r1", name="Room One", description="First room.")
        )
        space.add_room(
            Room(id="r2", name="Room Two", description="Second room.")
        )
        space.connect("r1", "east", "r2")

        path = tmp_path / "space.json"
        space.save(path)

        assert path.exists()

        loaded = DigitalSpace.load(path)
        assert loaded.get_room("r1") is not None
        assert loaded.get_room("r2") is not None
        assert loaded.get_room("r1").exits["east"] == "r2"

    def test_roundtrip_preserves_connections(self):
        space = DigitalSpace.create_default_space()
        data = space.to_dict()
        restored = DigitalSpace.from_dict(data)

        # Atrium should connect to all others
        atrium = restored.get_room("atrium")
        assert "garden" in atrium.exits.values()
        assert "observatory" in atrium.exits.values()
        assert "workshop" in atrium.exits.values()
        assert "depths" in atrium.exits.values()


class TestDefaultSpace:
    """Tests for the default seed topology."""

    def test_default_space_has_five_rooms(self):
        space = DigitalSpace.create_default_space()
        assert len(space.room_ids) == 5

    def test_default_space_room_ids(self):
        space = DigitalSpace.create_default_space()
        expected = {"atrium", "garden", "workshop", "observatory", "depths"}
        assert set(space.room_ids) == expected

    def test_atrium_connects_to_all(self):
        space = DigitalSpace.create_default_space()
        atrium = space.get_room("atrium")
        connected = set(atrium.exits.values())
        assert "garden" in connected
        assert "observatory" in connected
        assert "workshop" in connected
        assert "depths" in connected

    def test_garden_observatory_connected(self):
        space = DigitalSpace.create_default_space()
        garden = space.get_room("garden")
        observatory = space.get_room("observatory")
        assert "observatory" in garden.exits.values()
        assert "garden" in observatory.exits.values()

    def test_workshop_depths_connected(self):
        space = DigitalSpace.create_default_space()
        workshop = space.get_room("workshop")
        depths = space.get_room("depths")
        assert "depths" in workshop.exits.values()
        assert "workshop" in depths.exits.values()

    def test_all_rooms_have_properties(self):
        space = DigitalSpace.create_default_space()
        for room_id in space.room_ids:
            room = space.get_room(room_id)
            assert room.properties, f"{room_id} should have properties"

    def test_all_rooms_system_created(self):
        space = DigitalSpace.create_default_space()
        for room_id in space.room_ids:
            room = space.get_room(room_id)
            assert room.created_by == "system"
