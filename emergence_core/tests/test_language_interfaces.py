"""
Comprehensive tests for Phase 3 Language Interfaces.

Tests cover:
- LLM client functionality (mock and real)
- Structured formats validation
- Fallback handlers
- Input parsing with LLM integration
- Output generation with LLM integration
- Error handling and circuit breaker
- End-to-end conversation flow
"""

import pytest
import asyncio
from unittest.mock import AsyncMock

from lyra.cognitive_core.llm_client import (
    LLMClient,
    MockLLMClient,
    GemmaClient,
    LlamaClient,
    LLMError
)
from lyra.cognitive_core.structured_formats import (
    LLMInputParseRequest,
    LLMInputParseResponse,
    Intent,
    Goal,
    Entities,
    ConversationContext,
    workspace_snapshot_to_dict
)
from lyra.cognitive_core.fallback_handlers import (
    FallbackInputParser,
    FallbackOutputGenerator,
    CircuitBreaker,
    CircuitState
)
from lyra.cognitive_core.language_input import (
    LanguageInputParser,
    IntentType,
    ParseResult
)
from lyra.cognitive_core.language_output import LanguageOutputGenerator
from lyra.cognitive_core.perception import PerceptionSubsystem
from lyra.cognitive_core.workspace import (
    GlobalWorkspace,
    WorkspaceSnapshot,
    Goal as WorkspaceGoal,
    GoalType,
    Percept
)


class TestMockLLMClient:
    """Test MockLLMClient functionality."""
    
    @pytest.mark.asyncio
    async def test_mock_generate(self):
        """Test mock text generation."""
        client = MockLLMClient()
        response = await client.generate("Test prompt")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "mock" in response.lower()
    
    @pytest.mark.asyncio
    async def test_mock_generate_structured(self):
        """Test mock structured generation."""
        client = MockLLMClient()
        schema = {"intent": "dict", "goals": "list"}
        response = await client.generate_structured("Test prompt", schema)
        
        assert isinstance(response, dict)
        assert "intent" in response
        assert "goals" in response
        assert isinstance(response["goals"], list)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting works."""
        client = MockLLMClient({"rate_limit": 10})
        
        # Make 5 requests quickly - should not hit limit
        tasks = [client.generate(f"Test {i}") for i in range(5)]
        await asyncio.gather(*tasks)
        
        # Verify metrics
        assert client.metrics["total_requests"] == 5


class TestStructuredFormats:
    """Test Pydantic structured formats."""
    
    def test_intent_validation(self):
        """Test Intent model validation."""
        intent = Intent(type="question", confidence=0.95, metadata={})
        assert intent.type == "question"
        assert intent.confidence == 0.95
        
        # Test validation bounds
        with pytest.raises(Exception):
            Intent(type="question", confidence=1.5, metadata={})
    
    def test_goal_validation(self):
        """Test Goal model validation."""
        goal = Goal(
            type="respond_to_user",
            description="Respond to user",
            priority=0.9,
            metadata={}
        )
        assert goal.type == "respond_to_user"
        assert goal.priority == 0.9
    
    def test_parse_request_validation(self):
        """Test LLMInputParseRequest validation."""
        context = ConversationContext(
            turn_count=1,
            recent_topics=["test"],
            user_name="User"
        )
        request = LLMInputParseRequest(
            user_text="Hello!",
            conversation_context=context
        )
        assert request.user_text == "Hello!"
        assert request.conversation_context.turn_count == 1
    
    def test_parse_response_validation(self):
        """Test LLMInputParseResponse validation."""
        intent = Intent(type="greeting", confidence=0.9, metadata={})
        goal = Goal(
            type="respond_to_user",
            description="Greet user",
            priority=0.9,
            metadata={}
        )
        entities = Entities(topics=["greeting"], emotional_tone="positive")
        
        response = LLMInputParseResponse(
            intent=intent,
            goals=[goal],
            entities=entities,
            context_updates={},
            confidence=0.9
        )
        
        assert response.intent.type == "greeting"
        assert len(response.goals) == 1
        assert response.entities.emotional_tone == "positive"
    
    def test_workspace_snapshot_conversion(self):
        """Test workspace snapshot to dict conversion."""
        workspace = GlobalWorkspace()
        workspace.emotional_state = {"valence": 0.5, "arousal": 0.7, "dominance": 0.6}
        
        workspace.add_goal(WorkspaceGoal(
            type=GoalType.RESPOND_TO_USER,
            description="Test goal",
            priority=0.8,
            progress=0.0
        ))
        
        snapshot = workspace.broadcast()
        snapshot_dict = workspace_snapshot_to_dict(snapshot)
        
        assert "emotions" in snapshot_dict
        assert snapshot_dict["emotions"]["valence"] == 0.5
        assert "active_goals" in snapshot_dict
        assert len(snapshot_dict["active_goals"]) > 0


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_initial_state(self):
        """Test circuit breaker starts closed."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)
        assert cb.state == CircuitState.CLOSED
        assert cb.can_attempt() == True
    
    def test_failure_threshold(self):
        """Test circuit opens after threshold."""
        cb = CircuitBreaker(failure_threshold=3, timeout=1.0)
        
        # Record 3 failures
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_attempt() == False
    
    @pytest.mark.asyncio
    async def test_recovery(self):
        """Test circuit breaker recovery."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.5)
        
        # Trigger circuit open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        
        # Wait for timeout
        await asyncio.sleep(0.6)
        
        # Should allow attempt (half-open)
        assert cb.can_attempt() == True
        assert cb.state == CircuitState.HALF_OPEN
        
        # Success should close circuit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


class TestFallbackInputParser:
    """Test fallback input parser."""
    
    def test_basic_parsing(self):
        """Test basic rule-based parsing."""
        parser = FallbackInputParser()
        result = parser.parse("What is your name?")
        
        assert result["intent"]["type"] == "question"
        assert len(result["goals"]) > 0
        assert result["goals"][0]["type"] == "respond_to_user"
    
    def test_greeting_detection(self):
        """Test greeting detection."""
        parser = FallbackInputParser()
        result = parser.parse("Hello!")
        
        assert result["intent"]["type"] == "greeting"
    
    def test_entity_extraction(self):
        """Test entity extraction."""
        parser = FallbackInputParser()
        result = parser.parse("Tell me about AI today")
        
        assert "topics" in result["entities"]
        assert "temporal" in result["entities"]
        assert "today" in result["entities"]["temporal"]


class TestFallbackOutputGenerator:
    """Test fallback output generator."""
    
    def test_basic_generation(self):
        """Test basic template-based generation."""
        generator = FallbackOutputGenerator()
        response = generator.generate()
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_intent_based_generation(self):
        """Test generation based on intent."""
        generator = FallbackOutputGenerator()
        
        response = generator.generate(context={"intent": "greeting"})
        assert isinstance(response, str)
        
        response = generator.generate(context={"intent": "question"})
        assert isinstance(response, str)
    
    def test_workspace_state_integration(self):
        """Test generation with workspace state."""
        generator = FallbackOutputGenerator()
        workspace_state = {
            "emotions": {"valence": 0.5, "arousal": 0.7, "dominance": 0.6},
            "active_goals": [{"description": "test goal"}],
            "attended_percepts": []
        }
        
        response = generator.generate(workspace_state=workspace_state)
        assert isinstance(response, str)
        assert len(response) > 0


class TestLanguageInputParserIntegration:
    """Test enhanced LanguageInputParser with LLM integration."""
    
    @pytest.fixture
    def perception(self):
        """Create perception subsystem."""
        return PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        return MockLLMClient()
    
    @pytest.mark.asyncio
    async def test_parse_with_mock_llm(self, perception, mock_llm):
        """Test parsing with mock LLM."""
        parser = LanguageInputParser(
            perception,
            llm_client=mock_llm,
            config={"enable_cache": False}
        )
        
        result = await parser.parse("What is consciousness?")
        
        assert isinstance(result, ParseResult)
        assert result.intent is not None
        assert len(result.goals) > 0
        assert result.percept is not None
    
    @pytest.mark.asyncio
    async def test_parse_fallback(self, perception):
        """Test parsing falls back when no LLM."""
        parser = LanguageInputParser(
            perception,
            llm_client=None,
            config={"enable_cache": False}
        )
        
        result = await parser.parse("Hello there!")
        
        assert isinstance(result, ParseResult)
        assert result.intent.type == IntentType.GREETING
    
    @pytest.mark.asyncio
    async def test_parse_caching(self, perception, mock_llm):
        """Test parse result caching."""
        parser = LanguageInputParser(
            perception,
            llm_client=mock_llm,
            config={"enable_cache": True}
        )
        
        # First parse
        result1 = await parser.parse("Test input")
        
        # Second parse should use cache
        result2 = await parser.parse("Test input")
        
        assert result1.intent.type == result2.intent.type
    
    @pytest.mark.asyncio
    async def test_error_handling(self, perception):
        """Test error handling with failing LLM."""
        # Create failing LLM
        failing_llm = AsyncMock()
        failing_llm.generate_structured = AsyncMock(side_effect=LLMError("Test error"))
        
        parser = LanguageInputParser(
            perception,
            llm_client=failing_llm,
            config={"use_fallback_on_error": True}
        )
        
        # Should fall back to rule-based parsing
        result = await parser.parse("Test input")
        
        assert isinstance(result, ParseResult)
        assert result.intent is not None


class TestLanguageOutputGeneratorIntegration:
    """Test enhanced LanguageOutputGenerator with LLM integration."""
    
    @pytest.fixture
    def workspace_snapshot(self):
        """Create workspace snapshot."""
        workspace = GlobalWorkspace()
        workspace.emotional_state = {"valence": 0.5, "arousal": 0.7, "dominance": 0.6}
        workspace.add_goal(WorkspaceGoal(
            type=GoalType.RESPOND_TO_USER,
            description="Test goal",
            priority=0.8,
            progress=0.0
        ))
        return workspace.broadcast()
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM client."""
        return MockLLMClient()
    
    @pytest.mark.asyncio
    async def test_generate_with_mock_llm(self, workspace_snapshot, mock_llm):
        """Test generation with mock LLM."""
        generator = LanguageOutputGenerator(mock_llm)
        
        response = await generator.generate(
            workspace_snapshot,
            context={"user_input": "Test"}
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_generate_fallback(self, workspace_snapshot):
        """Test generation falls back when LLM fails."""
        # Create failing LLM
        failing_llm = AsyncMock()
        failing_llm.generate = AsyncMock(side_effect=LLMError("Test error"))
        
        generator = LanguageOutputGenerator(
            failing_llm,
            config={"use_fallback_on_error": True}
        )
        
        response = await generator.generate(
            workspace_snapshot,
            context={"user_input": "Test"}
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    @pytest.mark.asyncio
    async def test_content_filtering(self, workspace_snapshot, mock_llm):
        """Test content filtering."""
        generator = LanguageOutputGenerator(mock_llm)
        
        response = await generator.generate(workspace_snapshot)
        
        # Should not contain markdown artifacts
        assert "```" not in response


class TestEndToEndConversationFlow:
    """Test complete conversation flow with language interfaces."""
    
    @pytest.fixture
    def perception(self):
        """Create perception subsystem."""
        return PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})
    
    @pytest.fixture
    def mock_input_llm(self):
        """Create mock input LLM."""
        return MockLLMClient({"temperature": 0.3})
    
    @pytest.fixture
    def mock_output_llm(self):
        """Create mock output LLM."""
        return MockLLMClient({"temperature": 0.7})
    
    @pytest.mark.asyncio
    async def test_conversation_turn(
        self, 
        perception, 
        mock_input_llm, 
        mock_output_llm
    ):
        """Test complete conversation turn."""
        # Setup
        parser = LanguageInputParser(perception, llm_client=mock_input_llm)
        generator = LanguageOutputGenerator(mock_output_llm)
        workspace = GlobalWorkspace()
        
        # User input
        user_input = "How are you feeling today?"
        
        # Parse input
        parse_result = await parser.parse(user_input)
        
        assert parse_result.intent is not None
        assert len(parse_result.goals) > 0
        
        # Add goals to workspace
        for goal in parse_result.goals:
            workspace.add_goal(goal)
        
        # Update emotions (simulated)
        workspace.emotional_state = {"valence": 0.6, "arousal": 0.5, "dominance": 0.7}
        
        # Generate response
        snapshot = workspace.broadcast()
        response = await generator.generate(
            snapshot,
            context={"user_input": user_input}
        )
        
        assert isinstance(response, str)
        assert len(response) > 10
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(
        self, 
        perception, 
        mock_input_llm, 
        mock_output_llm
    ):
        """Test multi-turn conversation with context."""
        parser = LanguageInputParser(perception, llm_client=mock_input_llm)
        generator = LanguageOutputGenerator(mock_output_llm)
        workspace = GlobalWorkspace()
        
        # Turn 1
        result1 = await parser.parse("Hello!")
        assert result1.context["turn_count"] == 1
        
        # Turn 2
        result2 = await parser.parse("Tell me about yourself")
        assert result2.context["turn_count"] == 2
        
        # Context should be preserved
        assert "recent_topics" in result2.context


class TestPerformanceBenchmarks:
    """Performance benchmarks for language interfaces."""
    
    @pytest.fixture
    def perception(self):
        """Create perception subsystem."""
        return PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})
    
    @pytest.mark.asyncio
    async def test_parse_latency(self, perception):
        """Test parsing latency is under 5 seconds."""
        parser = LanguageInputParser(
            perception,
            llm_client=MockLLMClient(),
            config={"timeout": 5.0}
        )
        
        import time
        start = time.time()
        await parser.parse("This is a test input message")
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"Parsing took {elapsed}s, should be < 5s"
    
    @pytest.mark.asyncio
    async def test_generation_latency(self, perception):
        """Test generation latency is under 10 seconds."""
        workspace = GlobalWorkspace()
        workspace.emotional_state = {"valence": 0.5, "arousal": 0.7, "dominance": 0.6}
        snapshot = workspace.broadcast()
        
        generator = LanguageOutputGenerator(
            MockLLMClient(),
            config={"timeout": 10.0}
        )
        
        import time
        start = time.time()
        await generator.generate(snapshot)
        elapsed = time.time() - start
        
        assert elapsed < 10.0, f"Generation took {elapsed}s, should be < 10s"
