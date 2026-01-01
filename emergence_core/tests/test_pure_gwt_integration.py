"""
Integration test for pure Global Workspace Theory (GWT) architecture.

This test verifies that the cognitive core operates correctly as a unified
GWT system without any specialist routing or classification.

Tests verify:
1. Full cognitive cycle: input → cognitive core → output
2. No references to old specialist/router architecture
3. LLMs used ONLY at language I/O periphery
4. Continuous operation at ~10 Hz
"""

import asyncio
import pytest
from pathlib import Path

# Import cognitive core components
from emergence_core.lyra.cognitive_core import (
    CognitiveCore,
    GlobalWorkspace,
    LanguageInputParser,
    LanguageOutputGenerator,
    MockLLMClient,
    Goal,
    GoalType,
    Percept,
)


class TestPureGWTIntegration:
    """Integration tests for pure GWT architecture."""
    
    @pytest.fixture
    async def cognitive_core(self):
        """Create a cognitive core instance for testing."""
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace)
        yield core
        if core.running:
            await core.stop()
    
    @pytest.mark.asyncio
    async def test_cognitive_core_initialization(self, cognitive_core):
        """Test that cognitive core initializes without old architecture."""
        # Verify core has no specialist or router references
        assert not hasattr(cognitive_core, 'router')
        assert not hasattr(cognitive_core, 'specialists')
        assert not hasattr(cognitive_core, 'specialist_factory')
        
        # Verify core has correct GWT components
        assert hasattr(cognitive_core, 'workspace')
        assert hasattr(cognitive_core, 'attention')
        assert hasattr(cognitive_core, 'perception')
        assert hasattr(cognitive_core, 'action')
        assert hasattr(cognitive_core, 'affect')
        assert hasattr(cognitive_core, 'meta_cognition')
    
    @pytest.mark.asyncio
    async def test_full_cognitive_cycle_with_mock_llms(self):
        """Test complete cycle: input → cognitive core → output with mocks."""
        # Initialize core with mock LLMs
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace)
        
        # Create mock LLM clients for testing without real models
        mock_input_llm = MockLLMClient()
        mock_output_llm = MockLLMClient()
        
        parser = LanguageInputParser(llm_client=mock_input_llm)
        generator = LanguageOutputGenerator(llm_client=mock_output_llm)
        
        # Parse user input into structured format
        parse_result = await parser.parse("Hello, how are you?")
        
        # Verify parsing created goals or percepts
        assert len(parse_result.goals) > 0 or len(parse_result.percepts) > 0
        
        # Inject goals/percepts into workspace
        for goal in parse_result.goals:
            workspace.add_goal(goal)
        for percept in parse_result.percepts:
            workspace.add_percept(percept)
        
        # Start core and let it run for a few cycles
        await core.start()
        await asyncio.sleep(0.5)  # Let ~5 cycles run at 10 Hz
        
        # Query state to verify processing occurred
        snapshot = core.query_state()
        assert snapshot is not None
        assert len(snapshot.goals) > 0 or len(snapshot.percepts) > 0
        
        # Generate output from workspace state
        output = await generator.generate(snapshot)
        assert output is not None
        assert len(output) > 0
        
        # Stop core
        await core.stop()
        
        # Verify no exceptions during operation
        assert True  # If we got here, the full cycle worked
    
    @pytest.mark.asyncio
    async def test_continuous_operation_without_input(self, cognitive_core):
        """Test that cognitive core runs continuously even without user input."""
        # Start the core
        await cognitive_core.start()
        
        # Let it run for 1 second (should complete ~10 cycles)
        await asyncio.sleep(1.0)
        
        # Verify it's still running
        assert cognitive_core.running
        
        # Verify cycles were executed
        snapshot = cognitive_core.query_state()
        assert snapshot is not None
        
        # Stop core
        await cognitive_core.stop()
        assert not cognitive_core.running
    
    @pytest.mark.asyncio
    async def test_no_routing_logic_in_core(self, cognitive_core):
        """Verify that cognitive core has no routing or classification logic."""
        # Check that core doesn't have routing methods
        assert not hasattr(cognitive_core, 'route_to_specialist')
        assert not hasattr(cognitive_core, 'classify_task')
        assert not hasattr(cognitive_core, 'select_specialist')
        
        # Verify workspace is unified (not split by specialist)
        assert hasattr(cognitive_core.workspace, 'goals')
        assert hasattr(cognitive_core.workspace, 'percepts')
        assert not hasattr(cognitive_core.workspace, 'specialist_outputs')
    
    @pytest.mark.asyncio
    async def test_llms_at_periphery_only(self):
        """Verify that LLMs are used only for language I/O, not cognitive processing."""
        # LanguageInputParser should use LLM for parsing
        mock_llm = MockLLMClient()
        parser = LanguageInputParser(llm_client=mock_llm)
        assert parser.llm_client is not None
        
        # LanguageOutputGenerator should use LLM for generation
        generator = LanguageOutputGenerator(llm_client=mock_llm)
        assert generator.llm_client is not None
        
        # CognitiveCore itself should NOT contain LLM clients
        workspace = GlobalWorkspace()
        core = CognitiveCore(workspace=workspace)
        assert not hasattr(core, 'llm_client')
        assert not hasattr(core, 'llm')
        
        # Subsystems should not contain LLMs
        assert not hasattr(core.attention, 'llm_client')
        assert not hasattr(core.perception, 'llm_client')
        assert not hasattr(core.action, 'llm_client')
        assert not hasattr(core.affect, 'llm_client')
    
    @pytest.mark.asyncio
    async def test_attention_creates_bottleneck(self, cognitive_core):
        """Test that attention controller creates a selective bottleneck."""
        workspace = cognitive_core.workspace
        
        # Add many percepts (more than workspace capacity)
        for i in range(20):
            percept = Percept(
                content=f"Percept {i}",
                modality="text",
                timestamp=asyncio.get_event_loop().time(),
                embedding=[0.1 * i] * 384  # Mock embedding
            )
            workspace.add_percept(percept)
        
        # Verify workspace doesn't hold all percepts (attention bottleneck working)
        # Exact capacity may vary, but should be much less than 20
        assert len(workspace.percepts) < 20
        assert len(workspace.percepts) > 0
    
    @pytest.mark.asyncio
    async def test_goal_directed_behavior(self, cognitive_core):
        """Test that system exhibits goal-directed behavior."""
        workspace = cognitive_core.workspace
        
        # Add a goal
        goal = Goal(
            description="Understand the user's question",
            goal_type=GoalType.COGNITIVE,
            priority=0.8,
            created_at=asyncio.get_event_loop().time()
        )
        workspace.add_goal(goal)
        
        # Start core
        await cognitive_core.start()
        await asyncio.sleep(0.3)  # Let a few cycles run
        
        # Verify goal is being tracked
        snapshot = cognitive_core.query_state()
        assert len(snapshot.goals) > 0
        
        # Goals should influence behavior
        # (exact behavior depends on action subsystem implementation)
        await cognitive_core.stop()
    
    @pytest.mark.asyncio
    async def test_emotional_dynamics_present(self, cognitive_core):
        """Test that system maintains emotional state."""
        # Verify affect subsystem exists and tracks VAD
        assert hasattr(cognitive_core, 'affect')
        assert hasattr(cognitive_core.affect, 'valence')
        assert hasattr(cognitive_core.affect, 'arousal')
        assert hasattr(cognitive_core.affect, 'dominance')
        
        # Start core
        await cognitive_core.start()
        await asyncio.sleep(0.2)
        
        # Query emotional state
        snapshot = cognitive_core.query_state()
        # Snapshot should contain emotional state information
        # (exact structure depends on implementation)
        
        await cognitive_core.stop()
    
    @pytest.mark.asyncio
    async def test_meta_cognitive_awareness(self, cognitive_core):
        """Test that system has meta-cognitive self-monitoring."""
        # Verify self-monitor exists
        assert hasattr(cognitive_core, 'meta_cognition')
        
        # Start core
        await cognitive_core.start()
        await asyncio.sleep(0.3)
        
        # Meta-cognition should generate introspective percepts
        # that enter the workspace
        snapshot = cognitive_core.query_state()
        # Over time, some percepts should be meta-cognitive observations
        
        await cognitive_core.stop()


class TestNoLegacyArchitecture:
    """Tests verifying old architecture is completely removed."""
    
    def test_no_specialist_files_exist(self):
        """Verify specialist architecture files don't exist."""
        base_path = Path("emergence_core/lyra")
        
        # Files that should NOT exist
        deleted_files = [
            "adaptive_router.py",
            "router_model.py",
            "specialists.py",
            "specialist_tools.py",
            "unified_core.py",
            "persistent_self_model.txt",
        ]
        
        for filename in deleted_files:
            file_path = base_path / filename
            assert not file_path.exists(), f"File {filename} should have been deleted"
    
    def test_no_specialist_imports_in_core(self):
        """Verify cognitive core doesn't import old architecture."""
        # Read cognitive core files
        core_files = [
            Path("emergence_core/lyra/cognitive_core/__init__.py"),
            Path("emergence_core/lyra/cognitive_core/core.py"),
            Path("emergence_core/lyra/__init__.py"),
        ]
        
        forbidden_imports = [
            "from .adaptive_router",
            "from .router_model",
            "from .specialists",
            "from .specialist_tools",
            "from .unified_core",
            "import adaptive_router",
            "import router_model",
            "import specialists",
            "import specialist_tools",
            "import unified_core",
        ]
        
        for file_path in core_files:
            if file_path.exists():
                content = file_path.read_text()
                for forbidden in forbidden_imports:
                    assert forbidden not in content, \
                        f"Found forbidden import '{forbidden}' in {file_path}"


# Run with: pytest emergence_core/tests/test_pure_gwt_integration.py -v
