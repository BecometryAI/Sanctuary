"""Tests for EnvironmentIntegration with mock sensorium."""

import pytest

from sanctuary.core.schema import CognitiveOutput, MemoryOp, Percept
from sanctuary.environment.integration import EnvironmentIntegration
from sanctuary.environment.navigator import Navigator
from sanctuary.environment.space import DigitalSpace


class MockSensorium:
    """Minimal mock sensorium that records injected percepts."""

    def __init__(self):
        self.percepts: list[Percept] = []

    def inject_percept(self, percept: Percept) -> None:
        self.percepts.append(percept)


@pytest.fixture
def space() -> DigitalSpace:
    return DigitalSpace.create_default_space()


@pytest.fixture
def nav(space) -> Navigator:
    return Navigator(space)


@pytest.fixture
def sensorium() -> MockSensorium:
    return MockSensorium()


@pytest.fixture
def integration(nav, sensorium) -> EnvironmentIntegration:
    return EnvironmentIntegration(navigator=nav, sensorium=sensorium)


def _make_output(*commands: str) -> CognitiveOutput:
    """Create a CognitiveOutput with environment_action memory ops."""
    ops = [
        MemoryOp(type="environment_action", content=cmd)
        for cmd in commands
    ]
    return CognitiveOutput(memory_ops=ops)


class TestProcessOutput:
    """Tests for processing CognitiveOutput with environment actions."""

    def test_enter_via_output(self, integration, nav, sensorium):
        output = _make_output("enter")
        percepts = integration.process_output(output)
        assert len(percepts) == 1
        assert nav.is_present
        assert nav.current_room_id == "atrium"
        # Percept injected into sensorium
        assert len(sensorium.percepts) == 1

    def test_enter_specific_room(self, integration, nav):
        output = _make_output("enter garden")
        integration.process_output(output)
        assert nav.current_room_id == "garden"

    def test_move_via_output(self, integration, nav):
        integration.process_output(_make_output("enter"))
        percepts = integration.process_output(_make_output("move north"))
        assert nav.current_room_id == "garden"
        assert len(percepts) == 1

    def test_look_via_output(self, integration, nav, sensorium):
        integration.process_output(_make_output("enter"))
        sensorium.percepts.clear()
        percepts = integration.process_output(_make_output("look"))
        assert len(percepts) == 1
        assert "Atrium" in percepts[0].content

    def test_leave_via_output(self, integration, nav):
        integration.process_output(_make_output("enter"))
        integration.process_output(_make_output("leave"))
        assert not nav.is_present

    def test_multiple_commands_in_one_output(self, integration, nav, sensorium):
        output = _make_output("enter", "move north")
        percepts = integration.process_output(output)
        assert len(percepts) == 2
        assert nav.current_room_id == "garden"
        assert len(sensorium.percepts) == 2

    def test_non_environment_ops_ignored(self, integration, nav):
        output = CognitiveOutput(
            memory_ops=[
                MemoryOp(type="write_episodic", content="A memory"),
                MemoryOp(type="environment_action", content="enter"),
            ]
        )
        percepts = integration.process_output(output)
        assert len(percepts) == 1
        assert nav.is_present

    def test_unknown_command(self, integration, sensorium):
        output = _make_output("fly")
        percepts = integration.process_output(output)
        assert len(percepts) == 1
        assert "unknown" in percepts[0].content.lower()

    def test_empty_command_ignored(self, integration, sensorium):
        output = _make_output("")
        percepts = integration.process_output(output)
        assert len(percepts) == 0


class TestCreateCommands:
    """Tests for create room/object via integration commands."""

    def test_create_room_via_command(self, integration, nav, space):
        integration.process_output(_make_output("enter"))
        output = _make_output("create room The Library|A quiet reading space")
        percepts = integration.process_output(output)
        assert len(percepts) == 1
        assert "Library" in percepts[0].content
        library_ids = [r for r in space.room_ids if "library" in r]
        assert len(library_ids) == 1

    def test_create_room_with_direction(self, integration, nav, space):
        integration.process_output(_make_output("enter"))
        output = _make_output(
            "create room The Cellar|A dark cellar|below"
        )
        integration.process_output(output)
        atrium = space.get_room("atrium")
        assert "below" in atrium.exits

    def test_create_object_via_command(self, integration, nav):
        integration.process_output(_make_output("enter"))
        output = _make_output(
            "create object Crystal|A shimmering crystal|examine,touch"
        )
        percepts = integration.process_output(output)
        assert len(percepts) == 1
        room = nav.current_room
        obj = room.get_object("Crystal")
        assert obj is not None
        assert "touch" in obj.interactions


class TestExamineAndInteract:
    """Tests for examine/interact commands via integration."""

    @pytest.fixture(autouse=True)
    def setup_room(self, integration, nav, space):
        integration.process_output(_make_output("enter"))
        room = space.get_room("atrium")
        from sanctuary.environment.room import EnvironmentObject

        room.add_object(
            EnvironmentObject(
                id="bell_1",
                name="Bell",
                description="A small bell.",
                interactions=["examine", "listen"],
            )
        )

    def test_examine_via_command(self, integration):
        percepts = integration.process_output(_make_output("examine Bell"))
        assert len(percepts) == 1
        assert "Bell" in percepts[0].content

    def test_interact_via_command(self, integration):
        percepts = integration.process_output(
            _make_output("interact Bell listen")
        )
        assert len(percepts) == 1
        assert "listen" in percepts[0].content.lower()

    def test_examine_missing_arg(self, integration):
        percepts = integration.process_output(_make_output("examine"))
        assert "what" in percepts[0].content.lower()

    def test_interact_missing_args(self, integration):
        percepts = integration.process_output(_make_output("interact Bell"))
        assert "requires" in percepts[0].content.lower()


class TestAmbientPercepts:
    """Tests for ambient percept injection."""

    def test_ambient_when_present(self, integration, nav):
        nav.enter("garden")
        percept = integration.inject_ambient_percept()
        assert percept is not None
        assert percept.modality == "environment"
        assert percept.source == "environment:ambient"

    def test_ambient_when_not_present(self, integration):
        percept = integration.inject_ambient_percept()
        assert percept is None

    def test_ambient_injected_into_sensorium(self, integration, nav, sensorium):
        nav.enter("garden")
        sensorium.percepts.clear()
        integration.inject_ambient_percept()
        assert len(sensorium.percepts) == 1


class TestLocationContext:
    """Tests for location context generation."""

    def test_context_when_present(self, integration, nav):
        nav.enter("atrium")
        ctx = integration.get_location_context()
        assert ctx["in_digital_space"] is True
        assert ctx["current_room"] == "The Atrium"
        assert ctx["current_room_id"] == "atrium"
        assert "exits" in ctx

    def test_context_when_not_present(self, integration):
        ctx = integration.get_location_context()
        assert ctx == {}

    def test_context_includes_objects(self, integration, nav, space):
        room = space.get_room("atrium")
        from sanctuary.environment.room import EnvironmentObject

        room.add_object(
            EnvironmentObject(id="o1", name="Lamp", description="A lamp.")
        )
        nav.enter("atrium")
        ctx = integration.get_location_context()
        assert "Lamp" in ctx["objects"]


class TestNoSensorium:
    """Tests that integration works without a sensorium (caller handles percepts)."""

    def test_process_without_sensorium(self, nav):
        integration = EnvironmentIntegration(navigator=nav, sensorium=None)
        output = _make_output("enter")
        percepts = integration.process_output(output)
        assert len(percepts) == 1
        assert nav.is_present

    def test_ambient_without_sensorium(self, nav):
        integration = EnvironmentIntegration(navigator=nav, sensorium=None)
        nav.enter()
        percept = integration.inject_ambient_percept()
        # Still returns the percept even without sensorium
        assert percept is not None
