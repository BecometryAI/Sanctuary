"""Integration tests for CfC experiential layer wiring (Phase 4.1).

Tests the full pipeline:
    1. Scaffold PrecisionWeighting logs data via DataCollector
    2. CfCTrainer trains a PrecisionCell from collected data
    3. ExperientialManager runs inside CognitiveCycle
    4. CognitiveInput includes experiential_state
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.cognitive_cycle import CognitiveCycle, NullSensorium
from sanctuary.core.schema import (
    CognitiveInput,
    CognitiveOutput,
    ExperientialSignals,
    Percept,
)
from sanctuary.experiential.manager import ExperientialManager
from sanctuary.experiential.precision_cell import PrecisionCell
from sanctuary.experiential.trainer import CfCTrainer, DataCollector, TrainingRecord
from sanctuary.mind.cognitive_core.precision_weighting import PrecisionWeighting


# ---------------------------------------------------------------------------
# Test 1: DataCollector wired into scaffold PrecisionWeighting
# ---------------------------------------------------------------------------


class TestDataCollectorWiring:
    def test_attach_collector_to_precision_weighting(self):
        pw = PrecisionWeighting()
        collector = DataCollector()
        pw.attach_collector(collector)

        # Compute precision — collector should capture it
        pw.compute_precision(
            percept="test",
            emotional_state={"arousal": 0.7, "valence": -0.2},
            prediction_error=0.4,
        )

        assert collector.count == 1
        record = collector.records[0]
        assert record.arousal == 0.7
        assert record.prediction_error == 0.4
        assert record.base_precision == 0.5
        # Heuristic: 0.5 - 0.5*0.7 + 0.3*0.4 = 0.5 - 0.35 + 0.12 = 0.27
        assert abs(record.precision_output - 0.27) < 0.01

    def test_collector_captures_many_computations(self):
        pw = PrecisionWeighting()
        collector = DataCollector()
        pw.attach_collector(collector)

        for i in range(50):
            pw.compute_precision(
                percept=f"percept_{i}",
                emotional_state={"arousal": i / 50.0},
                prediction_error=1.0 - i / 50.0,
            )

        assert collector.count == 50

    def test_no_collector_does_not_error(self):
        """PrecisionWeighting works fine without a collector attached."""
        pw = PrecisionWeighting()
        result = pw.compute_precision(
            percept="test",
            emotional_state={"arousal": 0.5},
            prediction_error=0.3,
        )
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Test 2: Full pipeline — collect → train → validate
# ---------------------------------------------------------------------------


class TestCollectTrainPipeline:
    def test_scaffold_data_trains_cfc_cell(self):
        """Collect data from scaffold, train CfC, verify it learned."""
        pw = PrecisionWeighting()
        collector = DataCollector()
        pw.attach_collector(collector)

        # Simulate 300 scaffold cycles with varied inputs
        import random
        random.seed(42)
        for _ in range(300):
            pw.compute_precision(
                percept="sim",
                emotional_state={"arousal": random.random()},
                prediction_error=random.random(),
            )

        assert collector.count == 300

        # Train CfC cell on collected data
        cell = PrecisionCell()
        trainer = CfCTrainer(cell, seq_len=10, batch_size=16)
        result = trainer.train(collector.records, epochs=50)

        assert result.final_train_loss < 0.1
        assert result.best_val_loss < 0.2


# ---------------------------------------------------------------------------
# Test 3: ExperientialSignals in CognitiveInput schema
# ---------------------------------------------------------------------------


class TestExperientialSignalsSchema:
    def test_default_experiential_signals(self):
        ci = CognitiveInput()
        assert ci.experiential_state.precision_weight == 0.5
        assert ci.experiential_state.cells_active == {}

    def test_custom_experiential_signals(self):
        ci = CognitiveInput(
            experiential_state=ExperientialSignals(
                precision_weight=0.73,
                cells_active={"precision": True},
            )
        )
        assert ci.experiential_state.precision_weight == 0.73
        assert ci.experiential_state.cells_active["precision"] is True

    def test_serialization_roundtrip(self):
        ci = CognitiveInput(
            experiential_state=ExperientialSignals(
                precision_weight=0.8,
                cells_active={"precision": True, "affect": False},
            )
        )
        data = ci.model_dump()
        ci2 = CognitiveInput.model_validate(data)
        assert ci2.experiential_state.precision_weight == 0.8
        assert ci2.experiential_state.cells_active["precision"] is True


# ---------------------------------------------------------------------------
# Test 4: ExperientialManager inside CognitiveCycle
# ---------------------------------------------------------------------------


class StubModel:
    """Minimal model that returns valid CognitiveOutput."""
    async def think(self, cognitive_input: CognitiveInput) -> CognitiveOutput:
        return CognitiveOutput(inner_speech="thinking...")


class TestCognitiveCycleWithExperiential:
    @pytest.mark.asyncio
    async def test_cycle_without_experiential(self):
        """Cycle works normally without experiential manager."""
        model = StubModel()
        cycle = CognitiveCycle(model=model)
        await cycle.run(max_cycles=3)
        assert cycle.cycle_count == 3

    @pytest.mark.asyncio
    async def test_cycle_with_experiential(self):
        """Cycle includes experiential state when manager is provided."""
        model = StubModel()
        authority = AuthorityManager()
        experiential = ExperientialManager(authority=authority)

        cycle = CognitiveCycle(
            model=model,
            authority=authority,
            experiential=experiential,
        )

        captured_inputs = []
        original_think = model.think

        async def capturing_think(ci: CognitiveInput) -> CognitiveOutput:
            captured_inputs.append(ci)
            return await original_think(ci)

        model.think = capturing_think

        await cycle.run(max_cycles=3)

        assert cycle.cycle_count == 3
        assert len(captured_inputs) == 3

        # Every input should have experiential signals
        for ci in captured_inputs:
            assert hasattr(ci, "experiential_state")
            assert 0.0 <= ci.experiential_state.precision_weight <= 1.0

    @pytest.mark.asyncio
    async def test_experiential_state_evolves(self):
        """The CfC hidden state should evolve across cycles."""
        model = StubModel()
        experiential = ExperientialManager()

        cycle = CognitiveCycle(
            model=model,
            experiential=experiential,
        )

        await cycle.run(max_cycles=5)

        # Precision cell should have been stepped 5 times
        summary = experiential.precision_cell.get_summary()
        assert summary["total_steps"] == 5

    @pytest.mark.asyncio
    async def test_experiential_with_percepts(self):
        """Experiential layer processes prediction errors from percepts."""
        model = StubModel()
        experiential = ExperientialManager()

        cycle = CognitiveCycle(
            model=model,
            experiential=experiential,
        )

        # Inject a percept before running
        cycle.inject_percept(
            Percept(modality="language", content="hello", source="user")
        )

        await cycle.run(max_cycles=1)
        assert cycle.cycle_count == 1
        assert experiential.precision_cell.get_summary()["total_steps"] == 1
