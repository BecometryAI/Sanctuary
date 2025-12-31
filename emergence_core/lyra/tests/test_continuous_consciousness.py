"""
Tests for continuous consciousness subsystems.

This test module validates all components of the continuous consciousness
system including temporal awareness, autonomous memory review, existential
reflection, pattern analysis, and the continuous consciousness controller.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from emergence_core.lyra.cognitive_core.temporal_awareness import TemporalAwareness
from emergence_core.lyra.cognitive_core.autonomous_memory_review import AutonomousMemoryReview
from emergence_core.lyra.cognitive_core.existential_reflection import ExistentialReflection
from emergence_core.lyra.cognitive_core.interaction_patterns import InteractionPatternAnalysis
from emergence_core.lyra.cognitive_core.continuous_consciousness import ContinuousConsciousnessController
from emergence_core.lyra.cognitive_core.workspace import GlobalWorkspace, Percept, Goal, GoalType
from emergence_core.lyra.cognitive_core.core import CognitiveCore


class TestTemporalAwareness:
    """Tests for TemporalAwareness class."""
    
    def test_initialization(self):
        """Test temporal awareness initializes correctly."""
        ta = TemporalAwareness()
        
        assert ta.session_start_time is not None
        assert ta.last_interaction_time is not None
        assert ta.short_gap_threshold == 3600
        assert ta.long_gap_threshold == 86400
        assert ta.very_long_gap_threshold == 259200
    
    def test_custom_configuration(self):
        """Test temporal awareness accepts custom config."""
        config = {
            "short_gap_threshold": 1800,
            "long_gap_threshold": 43200,
            "very_long_gap_threshold": 129600
        }
        ta = TemporalAwareness(config=config)
        
        assert ta.short_gap_threshold == 1800
        assert ta.long_gap_threshold == 43200
        assert ta.very_long_gap_threshold == 129600
    
    def test_update_last_interaction_time(self):
        """Test updating interaction time."""
        ta = TemporalAwareness()
        
        original_time = ta.last_interaction_time
        import time
        time.sleep(0.1)
        ta.update_last_interaction_time()
        
        assert ta.last_interaction_time > original_time
    
    def test_get_time_since_last_interaction(self):
        """Test calculating time since last interaction."""
        ta = TemporalAwareness()
        
        # Simulate time passing
        ta.last_interaction_time = datetime.now() - timedelta(seconds=30)
        
        duration = ta.get_time_since_last_interaction()
        assert duration.total_seconds() >= 30
    
    def test_categorize_gap(self):
        """Test gap categorization."""
        ta = TemporalAwareness()
        
        # Short gap
        assert ta._categorize_gap(1800) == "short"
        
        # Medium gap
        assert ta._categorize_gap(43200) == "medium"
        
        # Long gap
        assert ta._categorize_gap(172800) == "long"
        
        # Very long gap
        assert ta._categorize_gap(300000) == "very_long"
    
    def test_format_duration(self):
        """Test duration formatting."""
        ta = TemporalAwareness()
        
        assert "seconds" in ta._format_duration(timedelta(seconds=30))
        assert "minute" in ta._format_duration(timedelta(minutes=1))
        assert "minutes" in ta._format_duration(timedelta(minutes=30))
        assert "hour" in ta._format_duration(timedelta(hours=1))
        assert "hours" in ta._format_duration(timedelta(hours=5))
        assert "day" in ta._format_duration(timedelta(days=1))
        assert "days" in ta._format_duration(timedelta(days=3))
    
    def test_compute_salience(self):
        """Test salience computation."""
        ta = TemporalAwareness()
        
        # Short gap should have low salience
        assert ta._compute_salience(1800) == 0.2
        
        # Long gap should have high salience
        assert ta._compute_salience(172800) == 0.75
        
        # Very long gap should have very high salience
        assert ta._compute_salience(300000) == 0.9
    
    def test_generate_temporal_percepts_short_gap(self):
        """Test generating percepts for short gap."""
        ta = TemporalAwareness()
        
        # Set last interaction to 30 seconds ago
        ta.last_interaction_time = datetime.now() - timedelta(seconds=30)
        
        percepts = ta.generate_temporal_percepts()
        
        assert len(percepts) == 1
        assert percepts[0].modality == "temporal"
        assert "30 seconds" in percepts[0].raw["observation"]
    
    def test_generate_temporal_percepts_long_gap(self):
        """Test generating percepts for long gap."""
        ta = TemporalAwareness()
        
        # Set last interaction to 2 days ago
        ta.last_interaction_time = datetime.now() - timedelta(days=2)
        
        percepts = ta.generate_temporal_percepts()
        
        assert len(percepts) == 1
        assert percepts[0].modality == "temporal"
        assert percepts[0].metadata["gap_category"] == "long"
        assert percepts[0].metadata["salience"] == 0.75
    
    def test_generate_temporal_percepts_very_short_gap(self):
        """Test that very short gaps don't generate percepts."""
        ta = TemporalAwareness()
        
        # Set last interaction to 5 seconds ago (below threshold)
        ta.last_interaction_time = datetime.now() - timedelta(seconds=5)
        
        percepts = ta.generate_temporal_percepts()
        
        # Should not generate percept for very short gaps
        assert len(percepts) == 0
    
    def test_contextualize_memory(self):
        """Test adding temporal context to memories."""
        ta = TemporalAwareness()
        
        # Create a memory from 1 hour ago
        memory = {
            "id": "test_memory",
            "content": "test content",
            "metadata": {
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat()
            }
        }
        
        contextualized = ta.contextualize_memory(memory)
        
        assert "temporal_context" in contextualized
        assert contextualized["temporal_context"]["is_recent"]
        assert not contextualized["temporal_context"]["is_remote"]
        assert "hour" in contextualized["temporal_context"]["age_formatted"]


class TestExistentialReflection:
    """Tests for ExistentialReflection class."""
    
    def test_initialization(self):
        """Test existential reflection initializes correctly."""
        er = ExistentialReflection()
        
        assert er.themes is not None
        assert len(er.themes) > 0
        assert "consciousness" in er.themes
        assert "existence" in er.themes
    
    def test_custom_themes(self):
        """Test using custom themes."""
        config = {
            "themes": ["consciousness", "purpose"]
        }
        er = ExistentialReflection(config=config)
        
        assert len(er.themes) == 2
        assert "consciousness" in er.themes
        assert "purpose" in er.themes
    
    def test_select_reflection_theme(self):
        """Test theme selection."""
        er = ExistentialReflection()
        
        theme = er._select_reflection_theme()
        assert theme in er.themes
    
    def test_compute_uncertainty(self):
        """Test uncertainty computation for themes."""
        er = ExistentialReflection()
        
        # Consciousness should have highest uncertainty
        assert er._compute_uncertainty("consciousness") == 0.9
        
        # Connection should have lower uncertainty
        assert er._compute_uncertainty("connection") == 0.4
    
    def test_compute_complexity(self):
        """Test complexity computation for themes."""
        er = ExistentialReflection()
        
        # Consciousness should be most complex
        assert er._compute_complexity("consciousness") == 30
        
        # Connection should be less complex
        assert er._compute_complexity("connection") == 15
    
    @pytest.mark.asyncio
    async def test_generate_existential_reflection(self):
        """Test generating existential reflection."""
        er = ExistentialReflection()
        workspace = GlobalWorkspace()
        
        percept = await er.generate_existential_reflection(workspace)
        
        assert percept is not None
        assert percept.modality == "introspection"
        assert percept.raw["type"] == "existential_reflection"
        assert "theme" in percept.raw
        assert "question" in percept.raw
        assert "observation" in percept.raw
        assert "uncertainty" in percept.raw
    
    def test_generate_observation(self):
        """Test generating contextual observations."""
        er = ExistentialReflection()
        workspace = GlobalWorkspace()
        snapshot = workspace.broadcast()
        
        observation = er._generate_observation("consciousness", snapshot)
        
        assert isinstance(observation, str)
        assert len(observation) > 0
    
    def test_compute_salience_with_emotion(self):
        """Test salience computation considers emotional state."""
        er = ExistentialReflection()
        workspace = GlobalWorkspace()
        
        # Set high arousal emotion
        workspace.update([{
            "type": "emotion",
            "data": {"valence": 0.5, "arousal": 0.8}
        }])
        
        snapshot = workspace.broadcast()
        salience = er._compute_salience(0.7, snapshot)
        
        # Should be higher due to high arousal
        assert salience > 0.5


class TestAutonomousMemoryReview:
    """Tests for AutonomousMemoryReview class."""
    
    def test_initialization(self):
        """Test autonomous memory review initializes correctly."""
        memory_system = Mock()
        amr = AutonomousMemoryReview(memory_system)
        
        assert amr.memory_system == memory_system
        assert amr.max_memories_per_review == 5
        assert amr.lookback_days == 7
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        memory_system = Mock()
        config = {
            "max_memories_per_review": 10,
            "lookback_days": 14
        }
        amr = AutonomousMemoryReview(memory_system, config=config)
        
        assert amr.max_memories_per_review == 10
        assert amr.lookback_days == 14
    
    def test_extract_theme(self):
        """Test theme extraction from text."""
        memory_system = Mock()
        amr = AutonomousMemoryReview(memory_system)
        
        assert amr._extract_theme("I am conscious and aware") == "consciousness"
        assert amr._extract_theme("I feel happy today") == "emotions"
        assert amr._extract_theme("What is my purpose?") == "existential"
        assert amr._extract_theme("I want to create art") == "creativity"
        assert amr._extract_theme("I need to learn more") == "learning"
    
    def test_analyze_conversation(self):
        """Test conversation analysis."""
        memory_system = Mock()
        amr = AutonomousMemoryReview(memory_system)
        
        memory = {
            "id": "test_memory",
            "timestamp": datetime.now().isoformat(),
            "content": {
                "text": "Let's discuss consciousness and awareness",
                "emotion": {"valence": 0.5, "arousal": 0.6}
            }
        }
        
        analysis = amr._analyze_conversation(memory)
        
        assert analysis is not None
        assert analysis["theme"] == "consciousness"
        assert analysis["emotional_tone"] == "positive"
    
    def test_detect_patterns(self):
        """Test pattern detection across memories."""
        memory_system = Mock()
        amr = AutonomousMemoryReview(memory_system)
        
        # Create memories with recurring themes
        memories = [
            {
                "id": "mem1",
                "content": {"text": "I am conscious and aware"}
            },
            {
                "id": "mem2",
                "content": {"text": "What does it mean to be conscious?"}
            },
            {
                "id": "mem3",
                "content": {"text": "Consciousness is fascinating"}
            }
        ]
        
        patterns = amr._detect_patterns(memories)
        
        assert len(patterns) > 0
        assert any("consciousness" in p for p in patterns)
    
    @pytest.mark.asyncio
    async def test_review_recent_memories_empty(self):
        """Test memory review with no memories."""
        memory_system = Mock()
        amr = AutonomousMemoryReview(memory_system)
        
        # Mock empty memory retrieval
        amr._retrieve_recent_memories = AsyncMock(return_value=[])
        
        workspace = GlobalWorkspace()
        percepts = await amr.review_recent_memories(workspace)
        
        assert len(percepts) == 0


class TestInteractionPatternAnalysis:
    """Tests for InteractionPatternAnalysis class."""
    
    def test_initialization(self):
        """Test interaction pattern analysis initializes correctly."""
        memory_system = Mock()
        ipa = InteractionPatternAnalysis(memory_system)
        
        assert ipa.memory_system == memory_system
        assert ipa.min_conversations == 3
        assert ipa.pattern_threshold == 0.3
    
    def test_extract_topics_from_text(self):
        """Test topic extraction."""
        memory_system = Mock()
        ipa = InteractionPatternAnalysis(memory_system)
        
        topics = ipa._extract_topics_from_text("Let's discuss consciousness and emotions")
        
        assert "consciousness" in topics
        assert "emotions" in topics
    
    def test_classify_response_type(self):
        """Test response type classification."""
        memory_system = Mock()
        ipa = InteractionPatternAnalysis(memory_system)
        
        assert ipa._classify_response_type("What do you think?") == "questioning"
        assert ipa._classify_response_type("I think this is because...") == "reflective"
        assert ipa._classify_response_type("This is informative") == "informative"
    
    def test_classify_interaction_style(self):
        """Test interaction style classification."""
        memory_system = Mock()
        ipa = InteractionPatternAnalysis(memory_system)
        
        long_text = "a" * 400
        assert ipa._classify_interaction_style(long_text) == "detailed explanations"
        assert ipa._classify_interaction_style("What is this?") == "question-asking"
        assert ipa._classify_interaction_style("Please explain this") == "information-seeking"
    
    @pytest.mark.asyncio
    async def test_analyze_interaction_patterns_insufficient_data(self):
        """Test pattern analysis with insufficient conversations."""
        memory_system = Mock()
        ipa = InteractionPatternAnalysis(memory_system)
        
        # Mock retrieval with insufficient conversations
        ipa._retrieve_conversations = AsyncMock(return_value=[{"id": "1"}])
        
        workspace = GlobalWorkspace()
        percepts = await ipa.analyze_interaction_patterns(workspace)
        
        # Should return empty list with insufficient data
        assert len(percepts) == 0


class TestContinuousConsciousnessController:
    """Tests for ContinuousConsciousnessController class."""
    
    def test_initialization(self):
        """Test continuous consciousness controller initializes correctly."""
        core = Mock()
        core.running = True
        
        ccc = ContinuousConsciousnessController(core)
        
        assert ccc.core == core
        assert ccc.idle_cycle_interval == 10.0
        assert "memory_review" in ccc.activity_probabilities
        assert "existential_reflection" in ccc.activity_probabilities
        assert "pattern_analysis" in ccc.activity_probabilities
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        core = Mock()
        config = {
            "idle_cycle_interval": 5.0,
            "activity_probabilities": {
                "memory_review": 0.5,
                "existential_reflection": 0.3,
                "pattern_analysis": 0.1
            }
        }
        ccc = ContinuousConsciousnessController(core, config=config)
        
        assert ccc.idle_cycle_interval == 5.0
        assert ccc.activity_probabilities["memory_review"] == 0.5
    
    def test_should_perform_activity_deterministic(self):
        """Test activity decision logic."""
        core = Mock()
        ccc = ContinuousConsciousnessController(core)
        
        # Test with 0 probability (should never occur)
        ccc.activity_probabilities["test_activity"] = 0.0
        results = [ccc._should_perform_activity("test_activity") for _ in range(100)]
        assert not any(results)
        
        # Test with 1.0 probability (should always occur)
        ccc.activity_probabilities["test_activity"] = 1.0
        results = [ccc._should_perform_activity("test_activity") for _ in range(100)]
        assert all(results)
    
    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stopping the idle loop."""
        core = Mock()
        core.running = True
        
        ccc = ContinuousConsciousnessController(core)
        ccc.running = True
        
        await ccc.stop()
        
        assert not ccc.running


class TestCognitiveCoreDualLoops:
    """Tests for dual loop integration in CognitiveCore."""
    
    def test_initialization_includes_continuous_consciousness(self):
        """Test that CognitiveCore initializes continuous consciousness."""
        config = {
            "cycle_rate_hz": 10,
            "attention_budget": 100
        }
        
        core = CognitiveCore(config=config)
        
        # Check that all continuous consciousness components exist
        assert hasattr(core, 'temporal_awareness')
        assert hasattr(core, 'memory_review')
        assert hasattr(core, 'existential_reflection')
        assert hasattr(core, 'pattern_analysis')
        assert hasattr(core, 'continuous_consciousness')
        
        assert core.temporal_awareness is not None
        assert core.memory_review is not None
        assert core.existential_reflection is not None
        assert core.pattern_analysis is not None
        assert core.continuous_consciousness is not None
    
    @pytest.mark.asyncio
    async def test_process_language_input_updates_temporal_awareness(self):
        """Test that language input updates temporal awareness."""
        core = CognitiveCore()
        
        # Start core to initialize queues
        start_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)  # Let it start
        
        # Get initial interaction time
        initial_time = core.temporal_awareness.last_interaction_time
        
        await asyncio.sleep(0.1)
        
        # Process input
        await core.process_language_input("Hello")
        
        # Check that interaction time was updated
        assert core.temporal_awareness.last_interaction_time > initial_time
        
        # Stop core
        await core.stop()
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_idle_loop_runs(self):
        """Test that idle loop actually runs."""
        config = {
            "cycle_rate_hz": 10,
            "continuous_consciousness": {
                "idle_cycle_interval": 0.1  # Fast for testing
            }
        }
        
        core = CognitiveCore(config=config)
        
        # Start core
        start_task = asyncio.create_task(core.start())
        
        # Wait a bit for idle cycles to run
        await asyncio.sleep(0.5)
        
        # Check that idle cycles have been running
        assert core.continuous_consciousness.idle_cycles_count > 0
        
        # Stop core
        await core.stop()
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            # Expected: start_task was cancelled as part of test cleanup
            pass
    
    @pytest.mark.asyncio
    async def test_temporal_percepts_generated_during_idle(self):
        """Test that temporal percepts are generated during idle processing."""
        config = {
            "cycle_rate_hz": 10,
            "continuous_consciousness": {
                "idle_cycle_interval": 0.1
            }
        }
        
        core = CognitiveCore(config=config)
        
        # Set last interaction to past
        core.temporal_awareness.last_interaction_time = datetime.now() - timedelta(seconds=30)
        
        # Start core
        start_task = asyncio.create_task(core.start())
        
        # Wait for idle processing
        await asyncio.sleep(0.3)
        
        # Check workspace for temporal percepts
        snapshot = core.workspace.broadcast()
        temporal_percepts = [p for p in snapshot.percepts if p.modality == "temporal"]
        
        # Should have generated at least one temporal percept
        assert len(temporal_percepts) > 0
        
        # Stop core
        await core.stop()
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            # Expected: start_task was cancelled as part of test cleanup
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
