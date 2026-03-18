"""Tests for CfC Knowledge Cells and Dynamic Registry (Phase 7.5).

Tests cover:
- CellRegistry: registration, unregistration, connections, persistence
- KnowledgeCell: creation, stepping, hidden state, save/load, maturity
- KnowledgeCellFactory: entity-initiated creation, training, naming
- ExperientialManager: dynamic registry integration, knowledge cell stepping
- ExperientialSignals: knowledge_signals field
- Growth Autonomy: self-directed vs external consent paths
"""

from __future__ import annotations

import pytest
import torch
from pathlib import Path

from sanctuary.experiential.cell_registry import (
    CellRegistry,
    CellRegistration,
    InterCellConnection,
)
from sanctuary.experiential.knowledge_cell import (
    KnowledgeCell,
    KnowledgeCellConfig,
    KnowledgeCellReading,
)
from sanctuary.experiential.cell_factory import (
    CellRequest,
    CellCreationResult,
    KnowledgeCellFactory,
)
from sanctuary.experiential.manager import (
    ExperientialManager,
    ExperientialState,
)
from sanctuary.experiential.precision_cell import PrecisionCell
from sanctuary.core.authority import AuthorityLevel, AuthorityManager
from sanctuary.core.schema import (
    ExperientialSignals,
    KnowledgeCellRequest,
    CognitiveOutput,
)
from sanctuary.growth.consent_gate import ConsentGate


# ---------------------------------------------------------------------------
# CellRegistry tests
# ---------------------------------------------------------------------------


class TestCellRegistry:
    def test_empty_registry(self):
        registry = CellRegistry()
        assert registry.cell_count == 0
        assert registry.foundational_count == 0
        assert registry.knowledge_count == 0

    def test_register_foundational_cell(self):
        registry = CellRegistry()
        cell = PrecisionCell()
        reg = registry.register("precision", cell, category="foundational")
        assert reg.name == "precision"
        assert reg.category == "foundational"
        assert registry.cell_count == 1
        assert registry.foundational_count == 1
        assert registry.knowledge_count == 0

    def test_register_knowledge_cell(self):
        registry = CellRegistry()
        config = KnowledgeCellConfig(domain="spatial")
        cell = KnowledgeCell(config)
        reg = registry.register("spatial", cell, category="knowledge", domain="spatial")
        assert reg.category == "knowledge"
        assert reg.domain == "spatial"
        assert registry.knowledge_count == 1

    def test_duplicate_name_raises(self):
        registry = CellRegistry()
        cell = PrecisionCell()
        registry.register("test", cell, category="foundational")
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", cell, category="foundational")

    def test_invalid_category_raises(self):
        registry = CellRegistry()
        cell = PrecisionCell()
        with pytest.raises(ValueError, match="must be"):
            registry.register("test", cell, category="invalid")

    def test_unregister(self):
        registry = CellRegistry()
        cell = PrecisionCell()
        registry.register("test", cell, category="foundational")
        assert registry.cell_count == 1
        reg = registry.unregister("test")
        assert reg.name == "test"
        assert registry.cell_count == 0

    def test_unregister_nonexistent_raises(self):
        registry = CellRegistry()
        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_get(self):
        registry = CellRegistry()
        cell = PrecisionCell()
        registry.register("test", cell, category="foundational")
        reg = registry.get("test")
        assert reg.cell is cell

    def test_get_nonexistent_raises(self):
        registry = CellRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_has(self):
        registry = CellRegistry()
        cell = PrecisionCell()
        registry.register("test", cell, category="foundational")
        assert registry.has("test")
        assert not registry.has("other")

    def test_all_cells(self):
        registry = CellRegistry()
        cell1 = PrecisionCell()
        config = KnowledgeCellConfig(domain="test")
        cell2 = KnowledgeCell(config)
        registry.register("c1", cell1, category="foundational")
        registry.register("c2", cell2, category="knowledge")
        cells = registry.all_cells()
        assert len(cells) == 2

    def test_foundational_cells_filter(self):
        registry = CellRegistry()
        cell1 = PrecisionCell()
        config = KnowledgeCellConfig(domain="test")
        cell2 = KnowledgeCell(config)
        registry.register("c1", cell1, category="foundational")
        registry.register("c2", cell2, category="knowledge")
        foundational = registry.foundational_cells()
        assert len(foundational) == 1
        assert foundational[0][0] == "c1"

    def test_knowledge_cells_filter(self):
        registry = CellRegistry()
        cell1 = PrecisionCell()
        config = KnowledgeCellConfig(domain="test")
        cell2 = KnowledgeCell(config)
        registry.register("c1", cell1, category="foundational")
        registry.register("c2", cell2, category="knowledge")
        knowledge = registry.knowledge_cells()
        assert len(knowledge) == 1
        assert knowledge[0][0] == "c2"


class TestInterCellConnections:
    def test_add_connection(self):
        registry = CellRegistry()
        cell1 = PrecisionCell()
        cell2 = PrecisionCell()
        registry.register("a", cell1, category="foundational")
        registry.register("b", cell2, category="foundational")
        conn = InterCellConnection(
            source_cell="a", target_cell="b",
            source_output="output", target_input="input",
        )
        registry.add_connection(conn)
        conns = registry.get_connections()
        assert len(conns) == 1
        assert conns[0].source_cell == "a"
        assert conns[0].target_cell == "b"

    def test_connection_updates_registrations(self):
        registry = CellRegistry()
        cell1 = PrecisionCell()
        cell2 = PrecisionCell()
        registry.register("a", cell1, category="foundational")
        registry.register("b", cell2, category="foundational")
        conn = InterCellConnection(
            source_cell="a", target_cell="b",
            source_output="output", target_input="input",
        )
        registry.add_connection(conn)
        assert "b" in registry.get("a").connections_to
        assert "a" in registry.get("b").connections_from

    def test_connection_missing_cell_raises(self):
        registry = CellRegistry()
        cell = PrecisionCell()
        registry.register("a", cell, category="foundational")
        conn = InterCellConnection(
            source_cell="a", target_cell="missing",
            source_output="output", target_input="input",
        )
        with pytest.raises(KeyError):
            registry.add_connection(conn)

    def test_get_inputs_for(self):
        registry = CellRegistry()
        c1 = PrecisionCell()
        c2 = PrecisionCell()
        c3 = PrecisionCell()
        registry.register("a", c1, category="foundational")
        registry.register("b", c2, category="foundational")
        registry.register("c", c3, category="foundational")
        registry.add_connection(InterCellConnection("a", "c", "o", "i"))
        registry.add_connection(InterCellConnection("b", "c", "o", "i"))
        inputs = registry.get_inputs_for("c")
        assert len(inputs) == 2

    def test_unregister_removes_connections(self):
        registry = CellRegistry()
        c1 = PrecisionCell()
        c2 = PrecisionCell()
        registry.register("a", c1, category="foundational")
        registry.register("b", c2, category="foundational")
        registry.add_connection(InterCellConnection("a", "b", "o", "i"))
        assert len(registry.get_connections()) == 1
        registry.unregister("a")
        assert len(registry.get_connections()) == 0

    def test_reset_all(self):
        registry = CellRegistry()
        cell = PrecisionCell()
        cell.step(arousal=0.5, prediction_error=0.3, base_precision=0.5)
        assert cell.get_hidden_state() is not None
        registry.register("test", cell, category="foundational")
        registry.reset_all()
        assert cell.get_hidden_state() is None


class TestRegistryPersistence:
    def test_save_and_metadata(self, tmp_path):
        registry = CellRegistry()
        cell = PrecisionCell()
        registry.register("precision", cell, category="foundational")
        config = KnowledgeCellConfig(domain="test_domain")
        kcell = KnowledgeCell(config)
        registry.register("knowledge_test", kcell, category="knowledge", domain="test_domain")
        registry.save(tmp_path)

        meta_path = tmp_path / "registry_meta.pt"
        assert meta_path.exists()
        assert (tmp_path / "precision" / "cell.pt").exists()
        assert (tmp_path / "knowledge_test" / "cell.pt").exists()

    def test_get_registry_metadata(self):
        registry = CellRegistry()
        cell = PrecisionCell()
        registry.register("test", cell, category="foundational")
        meta = registry.get_registry_metadata()
        assert meta["cell_count"] == 1
        assert meta["foundational_count"] == 1
        assert "test" in meta["cells"]


# ---------------------------------------------------------------------------
# KnowledgeCell tests
# ---------------------------------------------------------------------------


class TestKnowledgeCell:
    def test_creation(self):
        config = KnowledgeCellConfig(domain="test")
        cell = KnowledgeCell(config)
        assert cell.domain == "test"
        param_count = sum(p.numel() for p in cell.parameters())
        assert param_count > 0

    def test_step_returns_correct_output_size(self):
        config = KnowledgeCellConfig(domain="test", input_size=3, output_size=2)
        cell = KnowledgeCell(config)
        outputs = cell.step(input_0=0.5, input_1=0.3, input_2=0.8)
        assert len(outputs) == 2

    def test_step_with_named_inputs(self):
        config = KnowledgeCellConfig(domain="test", input_size=2, output_size=1)
        cell = KnowledgeCell(config)
        outputs = cell.step(valence=0.5, arousal=0.3)
        assert len(outputs) == 1

    def test_hidden_state_persists(self):
        config = KnowledgeCellConfig(domain="test")
        cell = KnowledgeCell(config)
        cell.step(input_0=0.5, input_1=0.3, input_2=0.8, input_3=0.1)
        h1 = cell.get_hidden_state()
        assert h1 is not None
        cell.step(input_0=0.9, input_1=0.1, input_2=0.2, input_3=0.5)
        h2 = cell.get_hidden_state()
        assert not torch.allclose(h1, h2)

    def test_reset_hidden(self):
        config = KnowledgeCellConfig(domain="test")
        cell = KnowledgeCell(config)
        cell.step(input_0=0.5)
        assert cell.get_hidden_state() is not None
        cell.reset_hidden()
        assert cell.get_hidden_state() is None

    def test_maturity_increases(self):
        config = KnowledgeCellConfig(domain="test")
        cell = KnowledgeCell(config)
        assert cell.maturity == 0.0
        for _ in range(100):
            cell.step(input_0=0.5)
        assert cell.maturity > 0.0
        assert cell.maturity <= 1.0

    def test_maturity_clamped(self):
        config = KnowledgeCellConfig(domain="test")
        cell = KnowledgeCell(config)
        cell.maturity = 1.5
        assert cell.maturity == 1.0
        cell.maturity = -0.5
        assert cell.maturity == 0.0

    def test_output_activation_sigmoid(self):
        config = KnowledgeCellConfig(domain="test", output_activation="sigmoid")
        cell = KnowledgeCell(config)
        for _ in range(10):
            outputs = cell.step(input_0=1.0, input_1=1.0, input_2=1.0, input_3=1.0)
            for v in outputs:
                assert 0.0 <= v <= 1.0

    def test_output_activation_tanh(self):
        config = KnowledgeCellConfig(domain="test", output_activation="tanh")
        cell = KnowledgeCell(config)
        for _ in range(10):
            outputs = cell.step(input_0=1.0, input_1=1.0, input_2=1.0, input_3=1.0)
            for v in outputs:
                assert -1.0 <= v <= 1.0

    def test_single_output(self):
        config = KnowledgeCellConfig(domain="test", output_size=1)
        cell = KnowledgeCell(config)
        outputs = cell.step(input_0=0.5)
        assert len(outputs) == 1

    def test_get_summary(self):
        config = KnowledgeCellConfig(domain="spatial")
        cell = KnowledgeCell(config)
        cell.step(input_0=0.5)
        summary = cell.get_summary()
        assert summary["domain"] == "spatial"
        assert summary["total_steps"] == 1
        assert "hidden_state_norm" in summary
        assert "param_count" in summary

    def test_get_history(self):
        config = KnowledgeCellConfig(domain="test")
        cell = KnowledgeCell(config)
        cell.step(input_0=0.5)
        cell.step(input_0=0.6)
        history = cell.get_history()
        assert len(history) == 2
        assert isinstance(history[0], KnowledgeCellReading)

    def test_invalid_units_raises(self):
        with pytest.raises(ValueError, match="Units must be between"):
            KnowledgeCell(KnowledgeCellConfig(domain="test", units=4))
        with pytest.raises(ValueError, match="Units must be between"):
            KnowledgeCell(KnowledgeCellConfig(domain="test", units=512))

    def test_save_and_load(self, tmp_path):
        config = KnowledgeCellConfig(domain="test_save", units=16, input_size=3, output_size=2)
        cell = KnowledgeCell(config)
        # Step a few times to build hidden state
        for i in range(5):
            cell.step(input_0=float(i) / 5, input_1=0.3, input_2=0.7)

        path = tmp_path / "knowledge_cell.pt"
        cell.save(path)
        assert path.exists()

        loaded = KnowledgeCell.load(path)
        assert loaded.config.domain == "test_save"
        assert loaded.config.units == 16
        assert loaded._step_count == 5

    def test_forward_training(self):
        config = KnowledgeCellConfig(domain="test", input_size=3, output_size=2)
        cell = KnowledgeCell(config)
        inputs = torch.randn(4, 10, 3)  # batch=4, seq=10, features=3
        targets = torch.randn(4, 10, 2)
        preds, loss = cell.forward_training(inputs, targets)
        assert preds.shape == (4, 10, 2)
        assert loss.item() >= 0


# ---------------------------------------------------------------------------
# KnowledgeCellFactory tests
# ---------------------------------------------------------------------------


class TestKnowledgeCellFactory:
    def test_create_basic(self):
        registry = CellRegistry()
        factory = KnowledgeCellFactory(registry)
        request = CellRequest(domain="spatial")
        result = factory.create(request)
        assert result.success
        assert result.param_count > 0
        assert registry.knowledge_count == 1

    def test_create_with_connections(self):
        registry = CellRegistry()
        cell = PrecisionCell()
        registry.register("precision", cell, category="foundational")
        factory = KnowledgeCellFactory(registry)
        request = CellRequest(domain="spatial", connect_from=["precision"])
        result = factory.create(request)
        assert result.success
        conns = registry.get_inputs_for(result.cell_name)
        assert len(conns) == 1
        assert conns[0].source_cell == "precision"

    def test_create_unique_names(self):
        registry = CellRegistry()
        factory = KnowledgeCellFactory(registry)
        r1 = factory.create(CellRequest(domain="spatial"))
        r2 = factory.create(CellRequest(domain="spatial"))
        assert r1.cell_name != r2.cell_name
        assert registry.knowledge_count == 2

    def test_create_custom_config(self):
        registry = CellRegistry()
        factory = KnowledgeCellFactory(registry)
        request = CellRequest(
            domain="creative",
            input_size=6,
            output_size=3,
            units=64,
            output_activation="sigmoid",
        )
        result = factory.create(request)
        assert result.success
        reg = registry.get(result.cell_name)
        cell = reg.cell
        assert isinstance(cell, KnowledgeCell)
        assert cell.config.units == 64
        assert cell.config.input_size == 6
        assert cell.config.output_size == 3

    def test_creation_history(self):
        registry = CellRegistry()
        factory = KnowledgeCellFactory(registry)
        factory.create(CellRequest(domain="a"))
        factory.create(CellRequest(domain="b"))
        assert len(factory.creation_history) == 2

    def test_train_cell(self):
        registry = CellRegistry()
        factory = KnowledgeCellFactory(registry)
        factory.create(CellRequest(domain="trainable", input_size=2, output_size=1, units=8))
        cell_name = factory.creation_history[0].cell_name

        # Generate simple training data
        data = [(
            [float(i) / 20, float(i % 5) / 5],
            [float(i) / 20],
        ) for i in range(20)]

        result = factory.train_cell(cell_name, data, epochs=10, seq_len=5)
        assert "final_loss" in result
        assert result["epochs"] == 10

    def test_train_insufficient_data(self):
        registry = CellRegistry()
        factory = KnowledgeCellFactory(registry)
        factory.create(CellRequest(domain="small"))
        cell_name = factory.creation_history[0].cell_name
        data = [([0.0, 0.0, 0.0, 0.0], [0.0, 0.0])]
        result = factory.train_cell(cell_name, data, seq_len=10)
        assert "error" in result


# ---------------------------------------------------------------------------
# ExperientialManager + Knowledge Cell integration tests
# ---------------------------------------------------------------------------


class TestManagerWithKnowledgeCells:
    def test_manager_has_registry(self):
        manager = ExperientialManager()
        assert manager.registry is not None
        assert manager.registry.foundational_count == 4

    def test_foundational_cells_registered(self):
        manager = ExperientialManager()
        assert manager.registry.has("precision")
        assert manager.registry.has("affect")
        assert manager.registry.has("attention")
        assert manager.registry.has("goal")

    def test_foundational_connections(self):
        manager = ExperientialManager()
        conns = manager.registry.get_connections()
        # affect -> precision and attention -> goal
        assert len(conns) == 2

    def test_step_with_no_knowledge_cells(self):
        manager = ExperientialManager()
        state = manager.step(
            arousal=0.5, prediction_error=0.3,
            base_precision=0.5, scaffold_precision=0.5,
        )
        assert isinstance(state, ExperientialState)
        assert state.knowledge_signals == {}

    def test_step_with_knowledge_cell(self):
        manager = ExperientialManager()
        config = KnowledgeCellConfig(domain="test", input_size=4, output_size=2)
        cell = KnowledgeCell(config)
        manager.registry.register("knowledge_test", cell, category="knowledge", domain="test")
        state = manager.step(
            arousal=0.5, prediction_error=0.3,
            base_precision=0.5, scaffold_precision=0.5,
        )
        assert "knowledge_test" in state.knowledge_signals
        assert len(state.knowledge_signals["knowledge_test"]) == 2
        assert "knowledge_test" in state.cell_active
        assert state.cell_active["knowledge_test"] is True

    def test_step_knowledge_cell_with_connection(self):
        manager = ExperientialManager()
        config = KnowledgeCellConfig(domain="connected", input_size=4, output_size=1)
        cell = KnowledgeCell(config)
        manager.registry.register("knowledge_connected", cell, category="knowledge", domain="connected")
        manager.registry.add_connection(InterCellConnection(
            source_cell="affect", target_cell="knowledge_connected",
            source_output="arousal", target_input="input_0",
        ))
        state = manager.step(
            arousal=0.5, prediction_error=0.3,
            base_precision=0.5, scaffold_precision=0.5,
        )
        assert "knowledge_connected" in state.knowledge_signals

    def test_status_includes_knowledge_cells(self):
        manager = ExperientialManager()
        config = KnowledgeCellConfig(domain="status_test")
        cell = KnowledgeCell(config)
        manager.registry.register("knowledge_status", cell, category="knowledge", domain="status_test")
        status = manager.get_status()
        assert "knowledge_status" in status
        assert status["knowledge_status"]["category"] == "knowledge"
        assert "registry" in status
        assert status["registry"]["knowledge_count"] == 1

    def test_reset_includes_knowledge_cells(self):
        manager = ExperientialManager()
        config = KnowledgeCellConfig(domain="reset_test")
        cell = KnowledgeCell(config)
        cell.step(input_0=0.5)
        assert cell.get_hidden_state() is not None
        manager.registry.register("knowledge_reset", cell, category="knowledge")
        manager.reset()
        assert cell.get_hidden_state() is None

    def test_save_and_load_with_knowledge_cells(self, tmp_path):
        # Create manager with a knowledge cell
        manager = ExperientialManager()
        config = KnowledgeCellConfig(domain="persist_test", units=16, input_size=2, output_size=1)
        cell = KnowledgeCell(config)
        cell.step(input_0=0.5, input_1=0.3)
        manager.registry.register("knowledge_persist", cell, category="knowledge", domain="persist_test")
        manager.save(tmp_path)

        # Create a new manager and load
        manager2 = ExperientialManager()
        assert manager2.registry.knowledge_count == 0
        manager2.load(tmp_path)
        assert manager2.registry.has("knowledge_persist")
        loaded_cell = manager2.registry.get("knowledge_persist").cell
        assert isinstance(loaded_cell, KnowledgeCell)
        assert loaded_cell.config.domain == "persist_test"

    def test_save_load_backward_compatible(self, tmp_path):
        """Verify loading from legacy flat file layout still works."""
        manager = ExperientialManager()
        # Save in legacy format
        manager.precision_cell.save(tmp_path / "precision_cell.pt")
        manager.affect_cell.save(tmp_path / "affect_cell.pt")
        manager.attention_cell.save(tmp_path / "attention_cell.pt")
        manager.goal_cell.save(tmp_path / "goal_cell.pt")

        manager2 = ExperientialManager()
        manager2.load(tmp_path)
        # Should not crash and should load cells


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemaUpdates:
    def test_experiential_signals_knowledge_signals(self):
        signals = ExperientialSignals(
            knowledge_signals={"spatial": [0.5, 0.3], "temporal": [0.1]},
        )
        assert "spatial" in signals.knowledge_signals
        assert signals.knowledge_signals["spatial"] == [0.5, 0.3]

    def test_experiential_signals_default_empty(self):
        signals = ExperientialSignals()
        assert signals.knowledge_signals == {}

    def test_knowledge_cell_request_schema(self):
        req = KnowledgeCellRequest(
            domain="spatial_reasoning",
            description="I need better spatial processing",
            input_size=4,
            output_size=2,
            units=32,
        )
        assert req.domain == "spatial_reasoning"
        assert req.units == 32

    def test_cognitive_output_has_knowledge_cell_requests(self):
        output = CognitiveOutput(
            inner_speech="I need a spatial cell",
            knowledge_cell_requests=[
                KnowledgeCellRequest(domain="spatial", description="test"),
            ],
        )
        assert len(output.knowledge_cell_requests) == 1
        assert output.knowledge_cell_requests[0].domain == "spatial"

    def test_cognitive_output_default_empty_requests(self):
        output = CognitiveOutput()
        assert output.knowledge_cell_requests == []


# ---------------------------------------------------------------------------
# Growth Autonomy tests
# ---------------------------------------------------------------------------


class TestGrowthAutonomy:
    def test_is_self_directed_with_worth_learning(self):
        assert ConsentGate.is_self_directed(worth_learning=True)
        assert not ConsentGate.is_self_directed(worth_learning=False)

    def test_is_self_directed_with_reflection_dict(self):
        assert ConsentGate.is_self_directed(
            reflection={"worth_learning": True, "what_to_learn": "test"}
        )
        assert not ConsentGate.is_self_directed(
            reflection={"worth_learning": False}
        )

    def test_is_self_directed_with_none(self):
        assert not ConsentGate.is_self_directed()
        assert not ConsentGate.is_self_directed(reflection=None)

    def test_external_modification_requires_consent(self):
        gate = ConsentGate()
        # External modification — must go through full flow
        gate.inform("Researcher wants to optimize weights")
        gate.request_consent("Approved by entity")
        assert gate.is_consented

    def test_external_modification_can_be_refused(self):
        gate = ConsentGate()
        gate.inform("Researcher wants to optimize weights")
        gate.refuse("I don't want this")
        assert not gate.is_consented


# ---------------------------------------------------------------------------
# Integration test: full knowledge cell lifecycle
# ---------------------------------------------------------------------------


class TestKnowledgeCellLifecycle:
    def test_full_lifecycle(self, tmp_path):
        """Entity requests cell -> factory creates -> manager steps -> persist -> reload."""
        # 1. Create manager and factory
        manager = ExperientialManager()
        factory = KnowledgeCellFactory(manager.registry)

        # 2. Entity requests a knowledge cell
        request = CellRequest(
            domain="conversational_dynamics",
            description="I need to track conversation patterns",
            input_size=3,
            output_size=2,
            units=16,
            connect_from=["affect"],
        )
        result = factory.create(request)
        assert result.success

        # 3. Step the manager — knowledge cell participates
        state = manager.step(
            arousal=0.5, prediction_error=0.3,
            base_precision=0.5, scaffold_precision=0.5,
        )
        assert result.cell_name in state.knowledge_signals
        assert len(state.knowledge_signals[result.cell_name]) == 2

        # 4. Step multiple times — maturity increases
        cell = manager.registry.get(result.cell_name).cell
        initial_maturity = cell.config.maturity
        for _ in range(100):
            manager.step(
                arousal=0.5, prediction_error=0.3,
                base_precision=0.5, scaffold_precision=0.5,
            )
        assert cell.config.maturity > initial_maturity

        # 5. Save and reload
        manager.save(tmp_path)
        manager2 = ExperientialManager()
        manager2.load(tmp_path)
        assert manager2.registry.has(result.cell_name)

        # 6. Reloaded cell can still step
        state2 = manager2.step(
            arousal=0.5, prediction_error=0.3,
            base_precision=0.5, scaffold_precision=0.5,
        )
        assert result.cell_name in state2.knowledge_signals
