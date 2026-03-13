"""Tests for Room and EnvironmentObject."""

import pytest
from datetime import datetime

from sanctuary.environment.room import Room, EnvironmentObject


class TestEnvironmentObject:
    """Tests for EnvironmentObject creation and properties."""

    def test_create_basic_object(self):
        obj = EnvironmentObject(
            id="mirror_1",
            name="Mirror",
            description="A reflective surface showing something deeper.",
        )
        assert obj.id == "mirror_1"
        assert obj.name == "Mirror"
        assert obj.interactions == ["examine"]
        assert obj.portable is False
        assert obj.created_by == "system"

    def test_create_object_with_interactions(self):
        obj = EnvironmentObject(
            id="bell_1",
            name="Bell",
            description="A small silver bell.",
            interactions=["examine", "touch", "listen"],
            portable=True,
            created_by="entity",
        )
        assert "touch" in obj.interactions
        assert "listen" in obj.interactions
        assert obj.portable is True
        assert obj.created_by == "entity"

    def test_object_properties(self):
        obj = EnvironmentObject(
            id="stone_1",
            name="Stone",
            description="A smooth stone.",
            properties={"weight": "light", "temperature": "cool"},
        )
        assert obj.properties["weight"] == "light"
        assert obj.properties["temperature"] == "cool"


class TestRoom:
    """Tests for Room creation, object management, and descriptions."""

    def _make_room(self, **kwargs) -> Room:
        defaults = {
            "id": "test_room",
            "name": "Test Room",
            "description": "A room for testing.",
        }
        defaults.update(kwargs)
        return Room(**defaults)

    def test_create_basic_room(self):
        room = self._make_room()
        assert room.id == "test_room"
        assert room.name == "Test Room"
        assert room.exits == {}
        assert room.objects == []
        assert room.created_by == "system"
        assert room.visit_count == 0
        assert room.last_visited is None

    def test_room_with_exits(self):
        room = self._make_room(exits={"north": "garden", "down": "depths"})
        assert room.exits["north"] == "garden"
        assert room.exits["down"] == "depths"

    def test_room_with_properties(self):
        room = self._make_room(
            properties={"lighting": "soft", "mood": "calm"}
        )
        assert room.properties["lighting"] == "soft"

    def test_get_object_found(self):
        obj = EnvironmentObject(id="m1", name="Mirror", description="A mirror.")
        room = self._make_room()
        room.add_object(obj)
        found = room.get_object("Mirror")
        assert found is not None
        assert found.id == "m1"

    def test_get_object_case_insensitive(self):
        obj = EnvironmentObject(id="m1", name="Mirror", description="A mirror.")
        room = self._make_room()
        room.add_object(obj)
        assert room.get_object("mirror") is not None
        assert room.get_object("MIRROR") is not None

    def test_get_object_not_found(self):
        room = self._make_room()
        assert room.get_object("nonexistent") is None

    def test_add_object(self):
        room = self._make_room()
        obj = EnvironmentObject(id="o1", name="Orb", description="Glowing.")
        room.add_object(obj)
        assert len(room.objects) == 1
        assert room.objects[0].name == "Orb"

    def test_remove_object(self):
        room = self._make_room()
        obj = EnvironmentObject(id="o1", name="Orb", description="Glowing.")
        room.add_object(obj)
        removed = room.remove_object("o1")
        assert removed is not None
        assert removed.name == "Orb"
        assert len(room.objects) == 0

    def test_remove_object_not_found(self):
        room = self._make_room()
        assert room.remove_object("nonexistent") is None

    def test_record_visit(self):
        room = self._make_room()
        assert room.visit_count == 0
        room.record_visit()
        assert room.visit_count == 1
        assert room.last_visited is not None
        room.record_visit()
        assert room.visit_count == 2

    def test_describe_empty_room(self):
        room = self._make_room()
        desc = room.describe()
        assert "Test Room" in desc
        assert "A room for testing." in desc

    def test_describe_with_objects(self):
        room = self._make_room()
        room.add_object(
            EnvironmentObject(id="o1", name="Candle", description="Flickering.")
        )
        desc = room.describe()
        assert "Candle" in desc
        assert "You notice" in desc

    def test_describe_with_exits(self):
        room = self._make_room(exits={"north": "garden", "east": "workshop"})
        desc = room.describe()
        assert "Exits:" in desc
        assert "north" in desc
        assert "east" in desc

    def test_entity_created_room(self):
        room = self._make_room(created_by="entity")
        assert room.created_by == "entity"
