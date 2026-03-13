"""Tests for Navigator -- navigation, entry/exit, object interaction, creation."""

import pytest

from sanctuary.environment.navigator import Navigator
from sanctuary.environment.room import EnvironmentObject, Room
from sanctuary.environment.space import DigitalSpace


@pytest.fixture
def space() -> DigitalSpace:
    """A default space for testing."""
    return DigitalSpace.create_default_space()


@pytest.fixture
def nav(space) -> Navigator:
    """A navigator backed by the default space."""
    return Navigator(space)


class TestEntryAndExit:
    """Tests for entering and leaving the digital space."""

    def test_not_present_initially(self, nav):
        assert nav.is_present is False
        assert nav.current_room_id is None
        assert nav.current_room is None

    def test_enter_default(self, nav):
        percept = nav.enter()
        assert nav.is_present is True
        assert nav.current_room_id == "atrium"
        assert percept.modality == "environment"
        assert "Atrium" in percept.content

    def test_enter_specific_room(self, nav):
        percept = nav.enter("garden")
        assert nav.current_room_id == "garden"
        assert "Memory Garden" in percept.content

    def test_enter_nonexistent_room(self, nav):
        percept = nav.enter("nowhere")
        assert nav.is_present is False
        assert "does not exist" in percept.content

    def test_leave(self, nav):
        nav.enter()
        percept = nav.leave()
        assert nav.is_present is False
        assert percept.modality == "environment"
        assert "leave" in percept.content.lower()

    def test_leave_when_not_present(self, nav):
        percept = nav.leave()
        assert "not currently" in percept.content.lower()

    def test_enter_records_visit(self, nav, space):
        nav.enter("atrium")
        room = space.get_room("atrium")
        assert room.visit_count == 1
        assert room.last_visited is not None


class TestMovement:
    """Tests for moving between rooms."""

    def test_move_valid_direction(self, nav):
        nav.enter("atrium")
        percept = nav.move("north")
        assert nav.current_room_id == "garden"
        assert percept.modality == "environment"
        assert "Memory Garden" in percept.content

    def test_move_invalid_direction(self, nav):
        nav.enter("atrium")
        percept = nav.move("northwest")
        assert nav.current_room_id == "atrium"  # didn't move
        assert "no exit" in percept.content.lower()

    def test_move_when_not_present(self, nav):
        percept = nav.move("north")
        assert "not in" in percept.content.lower()

    def test_move_records_visit(self, nav, space):
        nav.enter("atrium")
        nav.move("north")
        garden = space.get_room("garden")
        assert garden.visit_count == 1

    def test_move_case_insensitive(self, nav):
        nav.enter("atrium")
        percept = nav.move("North")
        assert nav.current_room_id == "garden"


class TestLook:
    """Tests for perceiving the current room."""

    def test_look(self, nav):
        nav.enter("atrium")
        percept = nav.look()
        assert percept.modality == "environment"
        assert "Atrium" in percept.content

    def test_look_when_not_present(self, nav):
        percept = nav.look()
        assert "not in" in percept.content.lower()


class TestObjectInteraction:
    """Tests for examining and interacting with objects."""

    @pytest.fixture
    def nav_with_object(self, space):
        """Navigator in a room with an object."""
        room = space.get_room("atrium")
        room.add_object(
            EnvironmentObject(
                id="mirror_1",
                name="Mirror",
                description="A reflective surface.",
                interactions=["examine", "touch"],
                properties={"material": "glass"},
            )
        )
        nav = Navigator(space)
        nav.enter("atrium")
        return nav

    def test_examine_object(self, nav_with_object):
        percept = nav_with_object.examine("Mirror")
        assert percept.modality == "environment"
        assert "Mirror" in percept.content
        assert "reflective" in percept.content.lower()

    def test_examine_nonexistent_object(self, nav_with_object):
        percept = nav_with_object.examine("Sword")
        assert "no" in percept.content.lower()

    def test_examine_when_not_present(self, nav):
        percept = nav.examine("anything")
        assert "not in" in percept.content.lower()

    def test_interact_valid_action(self, nav_with_object):
        percept = nav_with_object.interact("Mirror", "touch")
        assert percept.modality == "environment"
        assert "touch" in percept.content.lower()

    def test_interact_invalid_action(self, nav_with_object):
        percept = nav_with_object.interact("Mirror", "eat")
        assert "cannot" in percept.content.lower()

    def test_interact_nonexistent_object(self, nav_with_object):
        percept = nav_with_object.interact("Sword", "touch")
        assert "no" in percept.content.lower()


class TestTakeAndDrop:
    """Tests for taking and dropping objects."""

    @pytest.fixture
    def nav_with_portable(self, space):
        room = space.get_room("workshop")
        room.add_object(
            EnvironmentObject(
                id="gem_1",
                name="Gem",
                description="A glowing gem.",
                interactions=["examine", "take"],
                portable=True,
            )
        )
        room.add_object(
            EnvironmentObject(
                id="anvil_1",
                name="Anvil",
                description="Heavy.",
                interactions=["examine"],
                portable=False,
            )
        )
        nav = Navigator(space)
        nav.enter("workshop")
        return nav

    def test_take_portable_object(self, nav_with_portable):
        percept = nav_with_portable.interact("Gem", "take")
        assert "take" in percept.content.lower()
        assert len(nav_with_portable.inventory) == 1
        assert nav_with_portable.inventory[0].name == "Gem"
        # Object removed from room
        room = nav_with_portable.current_room
        assert room.get_object("Gem") is None

    def test_take_non_portable(self, nav_with_portable):
        percept = nav_with_portable.interact("Anvil", "take")
        assert "cannot" in percept.content.lower()
        assert len(nav_with_portable.inventory) == 0

    def test_drop_object(self, nav_with_portable):
        nav_with_portable.interact("Gem", "take")
        percept = nav_with_portable.interact("Gem", "drop")
        assert "set down" in percept.content.lower()
        assert len(nav_with_portable.inventory) == 0
        room = nav_with_portable.current_room
        assert room.get_object("Gem") is not None

    def test_examine_inventory_object(self, nav_with_portable):
        nav_with_portable.interact("Gem", "take")
        percept = nav_with_portable.examine("Gem")
        assert "Gem" in percept.content


class TestRoomCreation:
    """Tests for entity creating new rooms."""

    def test_create_room(self, nav, space):
        nav.enter("atrium")
        percept = nav.create_room("The Library", "A quiet reading space.")
        assert percept.modality == "environment"
        assert "Library" in percept.content

        # Room exists in space
        room_ids = space.room_ids
        library_ids = [rid for rid in room_ids if "library" in rid.lower()]
        assert len(library_ids) == 1

        # Room is connected to atrium
        library = space.get_room(library_ids[0])
        assert library is not None
        assert library.created_by == "entity"
        assert library.description == "A quiet reading space."

    def test_create_room_with_direction(self, nav, space):
        nav.enter("atrium")
        nav.create_room("The Cellar", "Dark and cool.", direction="down_further")
        atrium = space.get_room("atrium")
        # Should have a new exit
        assert "down_further" in atrium.exits

    def test_create_room_when_not_present(self, nav):
        percept = nav.create_room("Test", "Test room")
        assert "must be in" in percept.content.lower()


class TestObjectCreation:
    """Tests for entity creating new objects."""

    def test_create_object(self, nav):
        nav.enter("atrium")
        percept = nav.create_object(
            "Crystal",
            "A shimmering crystal.",
            interactions=["examine", "touch"],
        )
        assert percept.modality == "environment"
        assert "Crystal" in percept.content

        room = nav.current_room
        obj = room.get_object("Crystal")
        assert obj is not None
        assert obj.created_by == "entity"
        assert "touch" in obj.interactions

    def test_create_object_when_not_present(self, nav):
        percept = nav.create_object("Test", "A test object")
        assert "must be in" in percept.content.lower()
