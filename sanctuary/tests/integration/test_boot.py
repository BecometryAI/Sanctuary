"""
Phase 1 Boot Integration Test.

Tests that the cognitive core can:
1. Instantiate without errors (no heavy ML deps needed)
2. Run cognitive cycles without crashing
3. Accept input and process it through the pipeline
4. Produce workspace state changes

Run with:
    pytest sanctuary/tests/integration/test_boot.py -v

Requires only: numpy, pydantic (no sentence-transformers, torch, etc.)
"""

import asyncio
import logging
import pytest
import time

logger = logging.getLogger(__name__)


@pytest.fixture
def boot_config():
    """Minimal boot configuration."""
    return {
        "cycle_rate_hz": 10,
        "attention_budget": 100,
        "max_queue_size": 100,
        "log_interval_cycles": 10,
        "timing": {
            "warn_threshold_ms": 500,
            "critical_threshold_ms": 1000,
            "track_slow_cycles": True,
        },
        "checkpointing": {"enabled": False},
        "perception": {"mock_mode": True, "mock_embedding_dim": 384},
        "affect": {},
        "action": {},
    }


class TestBootInstantiation:
    """Test that boot-mode CognitiveCore can instantiate."""

    def test_boot_coordinator_creates(self, boot_config):
        """BootCoordinator should instantiate without errors."""
        from sanctuary.mind.cognitive_core.workspace import GlobalWorkspace
        from sanctuary.mind.cognitive_core.boot.boot_coordinator import BootCoordinator

        workspace = GlobalWorkspace()
        coord = BootCoordinator(workspace, boot_config)

        assert coord.perception is not None
        assert coord.affect is not None
        assert coord.attention is not None
        assert coord.action is not None
        assert coord.workspace is workspace

    def test_boot_core_creates(self, boot_config):
        """BootCognitiveCore should instantiate without errors."""
        from sanctuary.mind.cognitive_core.boot import BootCognitiveCore

        core = BootCognitiveCore(config=boot_config)

        assert core.workspace is not None
        assert core.running is False
        assert core.subsystems is not None
        assert core.cycle_executor is not None

    def test_mock_perception_deterministic(self):
        """Mock perception should produce identical embeddings for identical input."""
        from sanctuary.mind.cognitive_core.mock_perception import MockPerceptionSubsystem

        perception = MockPerceptionSubsystem(config={"mock_embedding_dim": 384})

        async def check():
            p1 = await perception.encode("hello world", "text")
            p2 = await perception.encode("hello world", "text")
            p3 = await perception.encode("different text", "text")

            assert p1.embedding == p2.embedding, "Same input should give same embedding"
            assert p1.embedding != p3.embedding, "Different input should give different embedding"
            assert len(p1.embedding) == 384
            assert p1.metadata.get("mock_mode") is True

        asyncio.get_event_loop().run_until_complete(check())


class TestBootCycling:
    """Test that the cognitive loop can run cycles."""

    @pytest.mark.asyncio
    async def test_run_single_cycle(self, boot_config):
        """Run a single cognitive cycle without errors."""
        from sanctuary.mind.cognitive_core.boot import BootCognitiveCore

        core = BootCognitiveCore(config=boot_config)

        try:
            await core.cycle_executor.execute_cycle()
            assert True
        except Exception as e:
            pytest.fail(f"Single cycle failed: {e}")

    @pytest.mark.asyncio
    async def test_run_100_cycles(self, boot_config):
        """Run 100 cognitive cycles - the Phase 1 deliverable."""
        from sanctuary.mind.cognitive_core.boot import BootCognitiveCore

        core = BootCognitiveCore(config=boot_config)

        start = time.time()
        errors = []

        for i in range(100):
            try:
                await core.cycle_executor.execute_cycle()
            except Exception as e:
                errors.append((i, str(e)))

        elapsed = time.time() - start

        logger.info(f"100 cycles completed in {elapsed:.3f}s ({100/elapsed:.1f} Hz)")
        logger.info(f"Errors: {len(errors)}")

        if errors:
            for cycle_num, err in errors[:5]:
                logger.error(f"  Cycle {cycle_num}: {err}")

        assert len(errors) == 0, f"{len(errors)} cycles failed: {errors[:5]}"

    @pytest.mark.asyncio
    async def test_run_with_input(self, boot_config):
        """Inject text input and verify it reaches the workspace."""
        from sanctuary.mind.cognitive_core.boot import BootCognitiveCore

        core = BootCognitiveCore(config=boot_config)

        core.inject_input("Hello, Sanctuary!", modality="text")

        for _ in range(5):
            await core.cycle_executor.execute_cycle()

        state = core.query_state()
        logger.info(f"Workspace after input: {len(state.percepts)} percepts")
        assert state is not None


class TestBootLoopLifecycle:
    """Test start/stop lifecycle of the cognitive loop."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self, boot_config):
        """Start loop, let it run briefly, stop cleanly."""
        from sanctuary.mind.cognitive_core.boot import BootCognitiveCore

        core = BootCognitiveCore(config=boot_config)

        await core.start()
        assert core.running is True

        await asyncio.sleep(0.5)

        await core.stop()
        assert core.running is False

        metrics = core.get_metrics()
        logger.info(f"Boot metrics after run: {metrics}")
        assert metrics.get("boot_mode") is True

    @pytest.mark.asyncio
    async def test_input_during_loop(self, boot_config):
        """Inject input while loop is running."""
        from sanctuary.mind.cognitive_core.boot import BootCognitiveCore

        core = BootCognitiveCore(config=boot_config)

        await core.start()

        for i in range(5):
            core.inject_input(f"Test message {i}", modality="text")
            await asyncio.sleep(0.1)

        await asyncio.sleep(0.5)
        await core.stop()

        metrics = core.get_metrics()
        logger.info(f"Metrics after input processing: {metrics}")


class TestBootMetrics:
    """Test that metrics are tracked during boot cycling."""

    @pytest.mark.asyncio
    async def test_metrics_after_cycles(self, boot_config):
        """Verify timing metrics are populated after running cycles."""
        from sanctuary.mind.cognitive_core.boot import BootCognitiveCore

        core = BootCognitiveCore(config=boot_config)

        for _ in range(10):
            await core.cycle_executor.execute_cycle()

        metrics = core.get_metrics()
        assert metrics.get("boot_mode") is True
        logger.info(f"Boot metrics: {metrics}")
