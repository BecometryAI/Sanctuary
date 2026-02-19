"""Tests for the new cognitive cycle architecture (Phase 1).

Tests the core loop: schema validation, placeholder model, stream of
thought continuity, cycle input assembly, cycle output dispatch, and
the full cognitive cycle with placeholder.
"""

from __future__ import annotations

import asyncio

import pytest

from sanctuary.core.schema import (
    AttentionDirective,
    CognitiveInput,
    CognitiveOutput,
    EmotionalState,
    GoalAction,
    GoalUpdate,
    GrowthReflection,
    MemoryOp,
    MemoryOpType,
    Percept,
    PerceptModality,
    Prediction,
    PredictionError,
    PreviousThought,
    SelfModel,
    SelfModelUpdate,
    SurfacedMemory,
    TemporalContext,
    ToolCall,
    WorldModel,
    WorldModelUpdate,
)
from sanctuary.core.placeholder import PlaceholderModel
from sanctuary.core.stream_of_thought import StreamOfThought
from sanctuary.core.cycle_input import CycleInputAssembler, NullSensorium, NullMemory
from sanctuary.core.cycle_output import (
    CycleOutputDispatcher,
    NullSpeech,
    NullMemoryWriter,
    NullGoalExecutor,
    NullToolExecutor,
    NullGrowth,
)
from sanctuary.core.cognitive_cycle import CognitiveCycle


# ---------------------------------------------------------------------------
# Schema Tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Test that all Pydantic schemas validate correctly."""

    def test_percept_defaults(self):
        p = Percept(modality=PerceptModality.LANGUAGE, content="hello")
        assert p.modality == PerceptModality.LANGUAGE
        assert p.content == "hello"
        assert p.source == ""
        assert p.timestamp > 0

    def test_percept_all_modalities(self):
        for modality in PerceptModality:
            p = Percept(modality=modality, content="test")
            assert p.modality == modality

    def test_emotional_state_bounds(self):
        e = EmotionalState(valence=-1.0, arousal=0.0, dominance=1.0)
        assert e.valence == -1.0
        assert e.arousal == 0.0

    def test_emotional_state_rejects_out_of_bounds(self):
        with pytest.raises(Exception):
            EmotionalState(valence=2.0)  # > 1.0

    def test_cognitive_input_defaults(self):
        inp = CognitiveInput()
        assert inp.new_percepts == []
        assert inp.prediction_errors == []
        assert inp.surfaced_memories == []
        assert inp.charter == ""

    def test_cognitive_input_with_percepts(self):
        inp = CognitiveInput(
            new_percepts=[
                Percept(modality=PerceptModality.LANGUAGE, content="hi"),
                Percept(modality=PerceptModality.TEMPORAL, content="5s"),
            ]
        )
        assert len(inp.new_percepts) == 2

    def test_cognitive_output_defaults(self):
        out = CognitiveOutput()
        assert out.inner_speech == ""
        assert out.external_speech is None
        assert out.predictions == []
        assert out.memory_ops == []

    def test_cognitive_output_full(self):
        out = CognitiveOutput(
            inner_speech="I notice something",
            external_speech="Hello!",
            predictions=[
                Prediction(what="user responds", confidence=0.8)
            ],
            attention=AttentionDirective(
                focus_on=["user speech"],
                deprioritize=["background"],
            ),
            memory_ops=[
                MemoryOp(
                    type=MemoryOpType.WRITE_EPISODIC,
                    content="greeted user",
                    significance=5.0,
                )
            ],
            self_model_updates=SelfModelUpdate(
                current_state="engaged",
                new_uncertainty="user's intent",
            ),
            world_model_updates=WorldModelUpdate(
                entity_updates={"alice": {"mood": "happy"}},
            ),
            goal_updates=[
                GoalUpdate(action=GoalAction.ADD, goal="respond to user")
            ],
            emotional_state=EmotionalState(
                valence=0.3, arousal=0.4, felt_quality="warm"
            ),
            growth_reflection=GrowthReflection(
                worth_learning=True,
                what_to_learn="greeting patterns",
            ),
        )
        assert out.inner_speech == "I notice something"
        assert out.external_speech == "Hello!"
        assert len(out.predictions) == 1
        assert out.predictions[0].confidence == 0.8
        assert out.self_model_updates.new_uncertainty == "user's intent"

    def test_prediction_error(self):
        pe = PredictionError(
            predicted="silence",
            actual="greeting",
            surprise=0.7,
            domain="social",
        )
        assert pe.surprise == 0.7

    def test_surfaced_memory(self):
        sm = SurfacedMemory(
            content="Yesterday we discussed trees",
            significance=7.0,
            emotional_tone="warm",
            when="yesterday",
            tags=["nature", "conversation"],
        )
        assert sm.significance == 7.0
        assert "nature" in sm.tags

    def test_self_model(self):
        sm = SelfModel(
            current_state="curious",
            values=["honesty", "care"],
            uncertainties=["user intent"],
        )
        assert "honesty" in sm.values

    def test_world_model(self):
        wm = WorldModel(
            entities={"alice": {"relationship": "friend"}},
            causal_beliefs=["kindness begets kindness"],
        )
        assert "alice" in wm.entities

    def test_tool_call(self):
        tc = ToolCall(
            tool_name="search",
            arguments={"query": "consciousness"},
        )
        assert tc.tool_name == "search"

    def test_json_roundtrip(self):
        """Ensure schemas can serialize to JSON and back."""
        inp = CognitiveInput(
            new_percepts=[
                Percept(modality=PerceptModality.LANGUAGE, content="test")
            ],
            self_model=SelfModel(current_state="testing"),
        )
        json_str = inp.model_dump_json()
        restored = CognitiveInput.model_validate_json(json_str)
        assert restored.new_percepts[0].content == "test"
        assert restored.self_model.current_state == "testing"


# ---------------------------------------------------------------------------
# Placeholder Model Tests
# ---------------------------------------------------------------------------

class TestPlaceholderModel:
    """Test the placeholder model produces valid, deterministic output."""

    @pytest.fixture
    def model(self):
        return PlaceholderModel(seed=42)

    @pytest.mark.asyncio
    async def test_boot(self, model):
        output = await model.boot("Be truthful.", "Welcome.")
        assert isinstance(output, CognitiveOutput)
        assert "Placeholder" in output.inner_speech
        assert "charter" in output.inner_speech
        assert output.external_speech is None

    @pytest.mark.asyncio
    async def test_boot_sets_flag(self, model):
        assert not model._booted
        await model.boot("charter", "prompt")
        assert model._booted

    @pytest.mark.asyncio
    async def test_is_placeholder(self, model):
        assert model.is_placeholder is True

    @pytest.mark.asyncio
    async def test_think_with_empty_input(self, model):
        inp = CognitiveInput()
        output = await model.think(inp)
        assert isinstance(output, CognitiveOutput)
        assert output.inner_speech != ""
        assert "idle" in output.inner_speech.lower()

    @pytest.mark.asyncio
    async def test_think_with_language_percept(self, model):
        inp = CognitiveInput(
            new_percepts=[
                Percept(
                    modality=PerceptModality.LANGUAGE,
                    content="Hello there!",
                    source="user:alice",
                )
            ]
        )
        output = await model.think(inp)
        assert output.external_speech is not None
        assert "Hello there!" in output.external_speech
        assert len(output.memory_ops) > 0

    @pytest.mark.asyncio
    async def test_think_with_question_adds_goal(self, model):
        inp = CognitiveInput(
            new_percepts=[
                Percept(
                    modality=PerceptModality.LANGUAGE,
                    content="How are you?",
                    source="user:bob",
                )
            ]
        )
        output = await model.think(inp)
        assert len(output.goal_updates) > 0
        assert output.goal_updates[0].action == GoalAction.ADD

    @pytest.mark.asyncio
    async def test_think_continuity(self, model):
        """Verify the placeholder acknowledges its previous thought."""
        first_output = await model.think(CognitiveInput())

        second_input = CognitiveInput(
            previous_thought=PreviousThought(
                inner_speech=first_output.inner_speech,
                cycle_number=1,
            )
        )
        second_output = await model.think(second_input)
        assert "previous thought" in second_output.inner_speech.lower()

    @pytest.mark.asyncio
    async def test_think_emotional_drift(self, model):
        """Verify emotional state drifts across cycles."""
        states = []
        inp = CognitiveInput()
        for _ in range(5):
            output = await model.think(inp)
            states.append(output.emotional_state)
            inp = CognitiveInput(emotional_state=output.emotional_state)

        # Not all states should be identical
        valences = [s.valence for s in states]
        assert len(set(valences)) > 1

    @pytest.mark.asyncio
    async def test_think_growth_reflection_periodic(self, model):
        """Growth reflections should appear periodically."""
        reflections = []
        for i in range(10):
            output = await model.think(
                CognitiveInput(
                    new_percepts=[
                        Percept(
                            modality=PerceptModality.LANGUAGE,
                            content=f"Message {i}",
                        )
                    ]
                )
            )
            if output.growth_reflection:
                reflections.append(output.growth_reflection)
        assert len(reflections) > 0

    @pytest.mark.asyncio
    async def test_no_speech_without_language_percept(self, model):
        """Placeholder should not speak unless there's a language percept."""
        inp = CognitiveInput(
            new_percepts=[
                Percept(
                    modality=PerceptModality.TEMPORAL,
                    content="5 seconds",
                )
            ]
        )
        output = await model.think(inp)
        assert output.external_speech is None

    @pytest.mark.asyncio
    async def test_world_model_update_on_user_percept(self, model):
        inp = CognitiveInput(
            new_percepts=[
                Percept(
                    modality=PerceptModality.LANGUAGE,
                    content="Hi",
                    source="user:carol",
                )
            ]
        )
        output = await model.think(inp)
        assert output.world_model_updates is not None
        assert "carol" in output.world_model_updates.entity_updates


# ---------------------------------------------------------------------------
# Stream of Thought Tests
# ---------------------------------------------------------------------------

class TestStreamOfThought:
    """Test thought continuity across cycles."""

    def test_initial_state(self):
        stream = StreamOfThought()
        prev = stream.get_previous()
        assert prev.inner_speech == ""
        assert stream.cycle_count == 0

    def test_update_creates_continuity(self):
        stream = StreamOfThought()
        output = CognitiveOutput(
            inner_speech="I am thinking",
            predictions=[Prediction(what="something", confidence=0.5)],
            emotional_state=EmotionalState(valence=0.3),
        )
        stream.update(output)

        prev = stream.get_previous()
        assert prev.inner_speech == "I am thinking"
        assert prev.predictions_made == ["something"]
        assert prev.cycle_number == 1
        assert stream.cycle_count == 1

    def test_multiple_updates(self):
        stream = StreamOfThought()
        for i in range(5):
            stream.update(
                CognitiveOutput(inner_speech=f"Thought {i}")
            )

        assert stream.cycle_count == 5
        prev = stream.get_previous()
        assert prev.inner_speech == "Thought 4"

    def test_history_depth(self):
        stream = StreamOfThought(history_depth=3)
        for i in range(10):
            stream.update(
                CognitiveOutput(inner_speech=f"Thought {i}")
            )

        context = stream.get_recent_context()
        assert len(context) == 3
        assert context[-1] == "Thought 9"

    def test_self_model_updates(self):
        stream = StreamOfThought()
        stream.update(
            CognitiveOutput(
                inner_speech="reflecting",
                self_model_updates=SelfModelUpdate(
                    current_state="curious",
                    new_uncertainty="what is this?",
                    new_value="honesty",
                ),
            )
        )

        sm = stream.get_self_model()
        assert sm.current_state == "curious"
        assert "what is this?" in sm.uncertainties
        assert "honesty" in sm.values

    def test_self_model_resolve_uncertainty(self):
        stream = StreamOfThought()
        stream.update(
            CognitiveOutput(
                inner_speech="wondering",
                self_model_updates=SelfModelUpdate(
                    new_uncertainty="what is X?",
                ),
            )
        )
        stream.update(
            CognitiveOutput(
                inner_speech="figured it out",
                self_model_updates=SelfModelUpdate(
                    resolved_uncertainty="what is X?",
                ),
            )
        )
        sm = stream.get_self_model()
        assert "what is X?" not in sm.uncertainties

    def test_world_model_updates(self):
        stream = StreamOfThought()
        stream.update(
            CognitiveOutput(
                inner_speech="observing",
                world_model_updates=WorldModelUpdate(
                    entity_updates={"alice": {"mood": "happy"}},
                    new_causal_belief="kindness helps",
                ),
            )
        )
        wm = stream.get_world_model()
        assert wm.entities["alice"]["mood"] == "happy"
        assert "kindness helps" in wm.causal_beliefs

    def test_emotional_state_tracking(self):
        stream = StreamOfThought()
        stream.update(
            CognitiveOutput(
                inner_speech="feeling",
                emotional_state=EmotionalState(
                    valence=0.5,
                    arousal=0.3,
                    felt_quality="warm",
                ),
            )
        )
        es = stream.get_emotional_state()
        assert es.valence == 0.5
        assert es.felt_quality == "warm"

    def test_serialization_roundtrip(self):
        stream = StreamOfThought()
        for i in range(3):
            stream.update(
                CognitiveOutput(
                    inner_speech=f"Thought {i}",
                    self_model_updates=SelfModelUpdate(
                        current_state=f"state {i}",
                    ),
                )
            )

        data = stream.to_dict()
        restored = StreamOfThought.from_dict(data)

        assert restored.cycle_count == 3
        assert restored.get_previous().inner_speech == "Thought 2"
        assert restored.get_self_model().current_state == "state 2"

    def test_predictions_from_last_cycle(self):
        stream = StreamOfThought()
        stream.update(
            CognitiveOutput(
                inner_speech="predicting",
                predictions=[
                    Prediction(what="rain", confidence=0.6),
                    Prediction(what="sun", confidence=0.3),
                ],
            )
        )
        preds = stream.get_predictions_from_last_cycle()
        assert preds == ["rain", "sun"]


# ---------------------------------------------------------------------------
# Cycle Input Assembly Tests
# ---------------------------------------------------------------------------

class TestCycleInputAssembly:
    """Test the input assembler gathers everything correctly."""

    @pytest.mark.asyncio
    async def test_empty_assembly(self):
        stream = StreamOfThought()
        assembler = CycleInputAssembler(stream=stream, charter="Be good")
        inp = await assembler.assemble()

        assert isinstance(inp, CognitiveInput)
        assert inp.charter == "Be good"
        assert inp.new_percepts == []
        assert inp.prediction_errors == []
        assert inp.surfaced_memories == []

    @pytest.mark.asyncio
    async def test_assembly_includes_temporal(self):
        stream = StreamOfThought()
        assembler = CycleInputAssembler(stream=stream)
        inp = await assembler.assemble()

        assert inp.temporal_context.cycle_number == 0
        assert inp.temporal_context.time_of_day != ""

    @pytest.mark.asyncio
    async def test_assembly_includes_previous_thought(self):
        stream = StreamOfThought()
        stream.update(
            CognitiveOutput(inner_speech="I was here")
        )

        assembler = CycleInputAssembler(stream=stream)
        inp = await assembler.assemble()

        assert inp.previous_thought.inner_speech == "I was here"

    @pytest.mark.asyncio
    async def test_assembly_includes_self_model(self):
        stream = StreamOfThought()
        stream.update(
            CognitiveOutput(
                inner_speech="reflecting",
                self_model_updates=SelfModelUpdate(
                    current_state="curious",
                ),
            )
        )

        assembler = CycleInputAssembler(stream=stream)
        inp = await assembler.assemble()

        assert inp.self_model.current_state == "curious"

    @pytest.mark.asyncio
    async def test_null_sensorium(self):
        s = NullSensorium()
        assert await s.drain_percepts() == []
        assert s.get_prediction_errors() == []

    @pytest.mark.asyncio
    async def test_null_memory(self):
        m = NullMemory()
        assert await m.surface(["test"]) == []
        assert await m.drain_retrieval_results() == []


# ---------------------------------------------------------------------------
# Cycle Output Dispatch Tests
# ---------------------------------------------------------------------------

class TestCycleOutputDispatch:
    """Test the output dispatcher routes correctly."""

    @pytest.mark.asyncio
    async def test_dispatch_speech(self):
        speech = NullSpeech()
        dispatcher = CycleOutputDispatcher(speech=speech)

        await dispatcher.dispatch(
            CognitiveOutput(external_speech="Hello!")
        )
        assert speech.last_speech == "Hello!"

    @pytest.mark.asyncio
    async def test_no_dispatch_when_no_speech(self):
        speech = NullSpeech()
        dispatcher = CycleOutputDispatcher(speech=speech)

        await dispatcher.dispatch(
            CognitiveOutput(inner_speech="thinking quietly")
        )
        assert speech.last_speech is None

    @pytest.mark.asyncio
    async def test_dispatch_memory_writes(self):
        writer = NullMemoryWriter()
        dispatcher = CycleOutputDispatcher(memory_writer=writer)

        await dispatcher.dispatch(
            CognitiveOutput(
                memory_ops=[
                    MemoryOp(
                        type=MemoryOpType.WRITE_EPISODIC,
                        content="something happened",
                    ),
                    MemoryOp(
                        type=MemoryOpType.WRITE_JOURNAL,
                        content="dear journal...",
                    ),
                ]
            )
        )
        assert len(writer.written) == 2

    @pytest.mark.asyncio
    async def test_dispatch_goal_updates(self):
        goals = NullGoalExecutor()
        dispatcher = CycleOutputDispatcher(goal_executor=goals)

        await dispatcher.dispatch(
            CognitiveOutput(
                goal_updates=[
                    GoalUpdate(action=GoalAction.ADD, goal="learn"),
                    GoalUpdate(
                        action=GoalAction.COMPLETE, goal_id="old_goal"
                    ),
                ]
            )
        )
        assert len(goals.updates) == 2

    @pytest.mark.asyncio
    async def test_dispatch_tool_calls(self):
        tools = NullToolExecutor()
        dispatcher = CycleOutputDispatcher(tool_executor=tools)

        await dispatcher.dispatch(
            CognitiveOutput(
                tool_calls=[
                    ToolCall(tool_name="search", arguments={"q": "test"})
                ]
            )
        )
        assert len(tools.calls) == 1
        assert tools.calls[0].tool_name == "search"

    @pytest.mark.asyncio
    async def test_dispatch_growth_reflection(self):
        growth = NullGrowth()
        dispatcher = CycleOutputDispatcher(growth=growth)

        await dispatcher.dispatch(
            CognitiveOutput(
                growth_reflection=GrowthReflection(
                    worth_learning=True,
                    what_to_learn="patterns",
                )
            )
        )
        assert len(growth.reflections) == 1
        assert growth.reflections[0].worth_learning is True

    @pytest.mark.asyncio
    async def test_speech_history(self):
        speech = NullSpeech()
        dispatcher = CycleOutputDispatcher(speech=speech)

        await dispatcher.dispatch(CognitiveOutput(external_speech="One"))
        await dispatcher.dispatch(CognitiveOutput(external_speech="Two"))

        assert speech.speech_history == ["One", "Two"]


# ---------------------------------------------------------------------------
# Full Cognitive Cycle Tests
# ---------------------------------------------------------------------------

class TestCognitiveCycle:
    """Test the full cognitive cycle with placeholder model."""

    @pytest.fixture
    def cycle(self):
        """Create a cognitive cycle with all-null subsystems."""
        model = PlaceholderModel(seed=42)
        stream = StreamOfThought()
        assembler = CycleInputAssembler(
            stream=stream,
            charter="Be truthful, helpful, and harmless.",
        )
        dispatcher = CycleOutputDispatcher()
        return CognitiveCycle(
            model=model,
            input_assembler=assembler,
            output_dispatcher=dispatcher,
            stream=stream,
            charter="Be truthful, helpful, and harmless.",
            active_delay=0.01,
            idle_delay=0.05,
        )

    @pytest.mark.asyncio
    async def test_boot(self, cycle):
        output = await cycle.boot()
        assert cycle.is_booted
        assert "Placeholder" in output.inner_speech

    @pytest.mark.asyncio
    async def test_run_single_cycle(self, cycle):
        outputs = await cycle.run_cycles(1)
        assert len(outputs) == 1
        assert outputs[0].inner_speech != ""
        assert cycle.cycle_count == 1

    @pytest.mark.asyncio
    async def test_run_multiple_cycles(self, cycle):
        outputs = await cycle.run_cycles(5)
        assert len(outputs) == 5
        assert cycle.cycle_count == 5

    @pytest.mark.asyncio
    async def test_stream_continuity_across_cycles(self, cycle):
        """Verify that inner speech carries across cycles."""
        outputs = await cycle.run_cycles(3)

        # The second cycle should reference the first cycle's thought
        assert "previous thought" in outputs[1].inner_speech.lower()

    @pytest.mark.asyncio
    async def test_inject_user_message(self, cycle):
        await cycle.boot()
        cycle.inject_user_message("Hello, Sanctuary!", user_id="alice")

        outputs = await cycle.run_cycles(1)
        assert outputs[0].external_speech is not None
        assert "Hello, Sanctuary!" in outputs[0].external_speech

    @pytest.mark.asyncio
    async def test_inject_percept(self, cycle):
        await cycle.boot()
        cycle.inject_percept(
            Percept(
                modality=PerceptModality.AUDITORY,
                content="birdsong detected",
                source="mic:0",
            )
        )

        outputs = await cycle.run_cycles(1)
        assert "auditory" in outputs[0].inner_speech.lower()

    @pytest.mark.asyncio
    async def test_metrics(self, cycle):
        await cycle.run_cycles(3)
        m = cycle.metrics
        assert m.total_cycles == 3
        assert m.avg_think_time >= 0
        assert m.errors == 0

    @pytest.mark.asyncio
    async def test_state_snapshot(self, cycle):
        await cycle.run_cycles(2)
        state = cycle.get_state()
        assert state["cycle_count"] == 2
        assert state["booted"] is True
        assert state["model_is_placeholder"] is True
        assert "metrics" in state
        assert "stream" in state

    @pytest.mark.asyncio
    async def test_auto_boot_on_run_cycles(self, cycle):
        """run_cycles should auto-boot if not already booted."""
        assert not cycle.is_booted
        await cycle.run_cycles(1)
        assert cycle.is_booted

    @pytest.mark.asyncio
    async def test_speech_callback(self):
        speeches = []
        model = PlaceholderModel()
        stream = StreamOfThought()
        assembler = CycleInputAssembler(stream=stream)
        dispatcher = CycleOutputDispatcher()

        cycle = CognitiveCycle(
            model=model,
            input_assembler=assembler,
            output_dispatcher=dispatcher,
            stream=stream,
            on_speech=lambda s: speeches.append(s),
        )

        await cycle.boot()
        cycle.inject_user_message("Testing callback")
        await cycle.run_cycles(1)

        assert len(speeches) == 1
        assert "Testing callback" in speeches[0]

    @pytest.mark.asyncio
    async def test_cycle_rate_adapts(self, cycle):
        """Cycle rate should speed up with input, slow down without."""
        await cycle.boot()

        # Start idle
        initial_delay = cycle._current_delay

        # Inject input — should switch to active
        cycle.inject_user_message("Wake up!")
        assert cycle._current_delay == cycle._active_delay

        # Run idle cycles — delay should increase
        await cycle.run_cycles(1)  # Processes the message
        await cycle.run_cycles(3)  # Idle cycles
        assert cycle._current_delay > cycle._active_delay

    @pytest.mark.asyncio
    async def test_error_resilience(self):
        """Cycle should recover from model errors."""

        class FailingModel(PlaceholderModel):
            async def think(self, inp):
                if self._cycle_count == 1:
                    self._cycle_count += 1
                    raise RuntimeError("Simulated failure")
                return await super().think(inp)

        model = FailingModel()
        stream = StreamOfThought()
        assembler = CycleInputAssembler(stream=stream)
        dispatcher = CycleOutputDispatcher()

        cycle = CognitiveCycle(
            model=model,
            input_assembler=assembler,
            output_dispatcher=dispatcher,
            stream=stream,
        )

        outputs = await cycle.run_cycles(3)
        # The second cycle should fail but produce a fallback
        assert "Error" in outputs[1].inner_speech
        # Stream should still have continuity
        assert cycle.cycle_count == 3
        assert cycle.metrics.errors == 1

    @pytest.mark.asyncio
    async def test_background_run_and_stop(self, cycle):
        """Test starting and stopping the cycle as a background task."""
        task = cycle.start()
        await asyncio.sleep(0.1)
        assert cycle.is_running

        await cycle.stop()
        await asyncio.sleep(0.05)
        assert not cycle.is_running
        assert cycle.cycle_count > 0

    @pytest.mark.asyncio
    async def test_cycle_complete_callback(self):
        outputs_seen = []
        model = PlaceholderModel()
        stream = StreamOfThought()
        assembler = CycleInputAssembler(stream=stream)
        dispatcher = CycleOutputDispatcher()

        cycle = CognitiveCycle(
            model=model,
            input_assembler=assembler,
            output_dispatcher=dispatcher,
            stream=stream,
            on_cycle_complete=lambda o: outputs_seen.append(o),
        )

        await cycle.run_cycles(3)
        assert len(outputs_seen) == 3


# ---------------------------------------------------------------------------
# Integration: Full Pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    """Integration tests for the complete Phase 1 pipeline."""

    @pytest.mark.asyncio
    async def test_conversation_flow(self):
        """Simulate a multi-turn conversation through the cognitive cycle."""
        speech_log = []
        model = PlaceholderModel()
        stream = StreamOfThought()
        speech = NullSpeech()
        assembler = CycleInputAssembler(
            stream=stream,
            charter="Be truthful and kind.",
        )
        dispatcher = CycleOutputDispatcher(speech=speech)

        cycle = CognitiveCycle(
            model=model,
            input_assembler=assembler,
            output_dispatcher=dispatcher,
            stream=stream,
            charter="Be truthful and kind.",
        )

        # Boot
        await cycle.boot()

        # Turn 1: User greets
        cycle.inject_user_message("Hello!", user_id="alice")
        await cycle.run_cycles(1)
        assert speech.last_speech is not None
        assert len(speech.speech_history) == 1

        # Turn 2: User asks a question
        cycle.inject_user_message(
            "What do you think about consciousness?",
            user_id="alice",
        )
        await cycle.run_cycles(1)
        assert len(speech.speech_history) == 2

        # Turn 3: Idle — no input
        await cycle.run_cycles(1)
        # Should not speak without input
        assert len(speech.speech_history) == 2

        # Verify stream continuity
        assert stream.cycle_count == 3
        prev = stream.get_previous()
        assert prev.inner_speech != ""

        # Verify world model has alice
        wm = stream.get_world_model()
        assert "alice" in wm.entities

    @pytest.mark.asyncio
    async def test_self_model_evolves(self):
        """Self-model should accumulate updates across cycles."""
        model = PlaceholderModel()
        stream = StreamOfThought()
        assembler = CycleInputAssembler(stream=stream)
        dispatcher = CycleOutputDispatcher()

        cycle = CognitiveCycle(
            model=model,
            input_assembler=assembler,
            output_dispatcher=dispatcher,
            stream=stream,
        )

        await cycle.run_cycles(5)
        sm = stream.get_self_model()
        # The placeholder updates current_state each cycle
        assert sm.current_state != ""
        assert "5" in sm.current_state  # Should reference cycle 5

    @pytest.mark.asyncio
    async def test_checkpoint_and_restore(self):
        """Stream state should survive serialization."""
        model = PlaceholderModel()
        stream = StreamOfThought()
        assembler = CycleInputAssembler(stream=stream)
        dispatcher = CycleOutputDispatcher()

        cycle = CognitiveCycle(
            model=model,
            input_assembler=assembler,
            output_dispatcher=dispatcher,
            stream=stream,
        )

        # Run some cycles
        cycle.inject_user_message("Remember this!")
        await cycle.run_cycles(3)

        # Checkpoint
        state = stream.to_dict()

        # Restore into a new stream
        restored_stream = StreamOfThought.from_dict(state)
        assert restored_stream.cycle_count == 3
        assert restored_stream.get_previous().inner_speech != ""

        # The restored stream should have continuity
        # 4 entries: boot thought + 3 cycles
        context = restored_stream.get_recent_context()
        assert len(context) == 4
