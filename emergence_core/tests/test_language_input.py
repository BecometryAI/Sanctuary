"""
Unit tests for LanguageInputParser.

Tests cover:
- Intent classification for all intent types
- Entity extraction (names, topics, temporal, emotions)
- Goal generation based on intent
- Percept creation with proper metadata
- Context tracking across conversation turns
- Full parsing integration
"""

import pytest

from lyra.cognitive_core.language_input import (
    LanguageInputParser,
    IntentType,
    Intent,
    ParseResult
)
from lyra.cognitive_core.workspace import Goal, GoalType, Percept
from lyra.cognitive_core.perception import PerceptionSubsystem


class TestIntentClassification:
    """Test intent classification for different input types."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with mock perception subsystem."""
        perception = PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})
        return LanguageInputParser(perception)
    
    def test_question_intent_what(self, parser):
        """Test classification of 'what' questions."""
        intent = parser._classify_intent("What is your name?")
        assert intent.type == IntentType.QUESTION
        assert intent.confidence > 0.0
    
    def test_question_intent_how(self, parser):
        """Test classification of 'how' questions."""
        intent = parser._classify_intent("How does this work?")
        assert intent.type == IntentType.QUESTION
        assert intent.confidence > 0.0
    
    def test_question_intent_question_mark(self, parser):
        """Test classification based on question mark."""
        intent = parser._classify_intent("You understand?")
        assert intent.type == IntentType.QUESTION
        assert intent.confidence > 0.0
    
    def test_request_intent_please(self, parser):
        """Test classification of requests with 'please'."""
        intent = parser._classify_intent("Please help me with this")
        assert intent.type == IntentType.REQUEST
        assert intent.confidence > 0.0
    
    def test_request_intent_can_you(self, parser):
        """Test classification of 'can you' requests."""
        intent = parser._classify_intent("Can you explain this?")
        assert intent.type == IntentType.REQUEST
        assert intent.confidence > 0.0
    
    def test_greeting_intent_hello(self, parser):
        """Test classification of greeting 'hello'."""
        intent = parser._classify_intent("Hello there!")
        assert intent.type == IntentType.GREETING
        assert intent.confidence > 0.0
    
    def test_greeting_intent_how_are_you(self, parser):
        """Test classification of 'how are you' greeting."""
        intent = parser._classify_intent("How are you doing?")
        assert intent.type == IntentType.GREETING
        assert intent.confidence > 0.0
    
    def test_memory_request_intent(self, parser):
        """Test classification of memory requests."""
        intent = parser._classify_intent("Do you remember what we discussed earlier?")
        assert intent.type == IntentType.MEMORY_REQUEST
        assert intent.confidence > 0.0
    
    def test_introspection_request_intent(self, parser):
        """Test classification of introspection requests."""
        intent = parser._classify_intent("How do you feel about this?")
        assert intent.type == IntentType.INTROSPECTION_REQUEST
        assert intent.confidence > 0.0
    
    def test_statement_default(self, parser):
        """Test that unknown patterns default to statement."""
        intent = parser._classify_intent("This is a simple statement.")
        assert intent.type == IntentType.STATEMENT
        assert intent.confidence == 0.5  # Default confidence


class TestEntityExtraction:
    """Test entity extraction from text."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with mock perception subsystem."""
        perception = PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})
        return LanguageInputParser(perception)
    
    def test_extract_names(self, parser):
        """Test extraction of capitalized names."""
        entities = parser._extract_entities("Hello, my name is Alice")
        assert "names" in entities
        assert "Alice" in entities["names"]
    
    def test_extract_topic_about(self, parser):
        """Test extraction of topics with 'about' pattern."""
        entities = parser._extract_entities("Tell me about quantum physics")
        assert "topic" in entities
        assert "quantum physics" in entities["topic"]
    
    def test_extract_temporal_today(self, parser):
        """Test extraction of temporal reference 'today'."""
        entities = parser._extract_entities("What happened today?")
        assert "temporal" in entities
        assert entities["temporal"] == "today"
    
    def test_extract_temporal_yesterday(self, parser):
        """Test extraction of temporal reference 'yesterday'."""
        entities = parser._extract_entities("I saw that yesterday")
        assert "temporal" in entities
        assert entities["temporal"] == "yesterday"
    
    def test_extract_positive_emotion(self, parser):
        """Test extraction of positive emotional keywords."""
        entities = parser._extract_entities("I'm so happy about this!")
        assert "user_emotion" in entities
        assert entities["user_emotion"]["valence"] == "positive"
        assert entities["user_emotion"]["keyword"] == "happy"
    
    def test_extract_negative_emotion(self, parser):
        """Test extraction of negative emotional keywords."""
        entities = parser._extract_entities("I'm feeling quite frustrated")
        assert "user_emotion" in entities
        assert entities["user_emotion"]["valence"] == "negative"
        assert entities["user_emotion"]["keyword"] == "frustrated"
    
    def test_no_entities(self, parser):
        """Test with text containing no extractable entities."""
        entities = parser._extract_entities("just some text")
        assert len(entities) == 0


class TestGoalGeneration:
    """Test goal generation based on intent."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with mock perception subsystem."""
        perception = PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})
        return LanguageInputParser(perception)
    
    def test_always_generates_response_goal(self, parser):
        """Test that all intents generate a RESPOND_TO_USER goal."""
        intent = Intent(type=IntentType.STATEMENT, confidence=0.8, metadata={})
        goals = parser._generate_goals("Test input", intent, {})
        
        assert len(goals) >= 1
        assert any(g.type == GoalType.RESPOND_TO_USER for g in goals)
    
    def test_memory_request_generates_retrieve_goal(self, parser):
        """Test that memory requests generate RETRIEVE_MEMORY goal."""
        intent = Intent(type=IntentType.MEMORY_REQUEST, confidence=0.9, metadata={})
        goals = parser._generate_goals("Do you remember?", intent, {})
        
        assert len(goals) == 2  # Response + Retrieve
        assert any(g.type == GoalType.RETRIEVE_MEMORY for g in goals)
    
    def test_introspection_request_generates_introspect_goal(self, parser):
        """Test that introspection requests generate INTROSPECT goal."""
        intent = Intent(type=IntentType.INTROSPECTION_REQUEST, confidence=0.9, metadata={})
        goals = parser._generate_goals("How do you feel?", intent, {})
        
        assert len(goals) == 2  # Response + Introspect
        assert any(g.type == GoalType.INTROSPECT for g in goals)
    
    def test_question_with_memory_keywords(self, parser):
        """Test that questions with memory keywords generate RETRIEVE_MEMORY goal."""
        intent = Intent(type=IntentType.QUESTION, confidence=0.9, metadata={})
        goals = parser._generate_goals("What did we discuss earlier?", intent, {})
        
        assert len(goals) == 2  # Response + Retrieve
        assert any(g.type == GoalType.RETRIEVE_MEMORY for g in goals)
    
    def test_simple_question_no_memory_goal(self, parser):
        """Test that simple questions don't generate memory retrieval goal."""
        intent = Intent(type=IntentType.QUESTION, confidence=0.9, metadata={})
        goals = parser._generate_goals("What is 2+2?", intent, {})
        
        # Should only have response goal
        assert len(goals) == 1
        assert goals[0].type == GoalType.RESPOND_TO_USER


class TestPerceptCreation:
    """Test percept creation with metadata."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with real perception subsystem."""
        perception = PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})
        return LanguageInputParser(perception)
    
    @pytest.mark.asyncio
    async def test_percept_has_embedding(self, parser):
        """Test that created percept has embedding."""
        intent = Intent(type=IntentType.STATEMENT, confidence=0.8, metadata={})
        percept = await parser._create_percept("Test text", intent, {})
        
        assert percept.embedding is not None
        assert len(percept.embedding) > 0
    
    @pytest.mark.asyncio
    async def test_percept_has_intent_metadata(self, parser):
        """Test that percept includes intent in metadata."""
        intent = Intent(type=IntentType.QUESTION, confidence=0.9, metadata={})
        percept = await parser._create_percept("What is this?", intent, {})
        
        assert "intent" in percept.metadata
        assert percept.metadata["intent"] == IntentType.QUESTION
        assert "intent_confidence" in percept.metadata
        assert percept.metadata["intent_confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_percept_has_entities_metadata(self, parser):
        """Test that percept includes entities in metadata."""
        intent = Intent(type=IntentType.STATEMENT, confidence=0.8, metadata={})
        entities = {"names": ["Alice"], "topic": "physics"}
        percept = await parser._create_percept("Test", intent, entities)
        
        assert "entities" in percept.metadata
        assert percept.metadata["entities"] == entities
    
    @pytest.mark.asyncio
    async def test_question_increases_complexity(self, parser):
        """Test that questions have higher complexity."""
        intent_statement = Intent(type=IntentType.STATEMENT, confidence=0.8, metadata={})
        intent_question = Intent(type=IntentType.QUESTION, confidence=0.8, metadata={})
        
        percept_statement = await parser._create_percept("Statement", intent_statement, {})
        percept_question = await parser._create_percept("Question?", intent_question, {})
        
        # Question should have higher complexity
        assert percept_question.complexity > percept_statement.complexity
    
    @pytest.mark.asyncio
    async def test_introspection_increases_complexity(self, parser):
        """Test that introspection requests have highest complexity."""
        intent_statement = Intent(type=IntentType.STATEMENT, confidence=0.8, metadata={})
        intent_introspection = Intent(type=IntentType.INTROSPECTION_REQUEST, confidence=0.8, metadata={})
        
        percept_statement = await parser._create_percept("Statement", intent_statement, {})
        percept_introspection = await parser._create_percept("How do you feel?", intent_introspection, {})
        
        # Introspection should have highest complexity
        assert percept_introspection.complexity > percept_statement.complexity


class TestContextTracking:
    """Test conversation context tracking."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with real perception subsystem."""
        perception = PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})
        return LanguageInputParser(perception)
    
    @pytest.mark.asyncio
    async def test_turn_count_increments(self, parser):
        """Test that turn count increments with each parse."""
        result1 = await parser.parse("First message")
        result2 = await parser.parse("Second message")
        result3 = await parser.parse("Third message")
        
        assert result1.context["turn_count"] == 1
        assert result2.context["turn_count"] == 2
        assert result3.context["turn_count"] == 3
    
    @pytest.mark.asyncio
    async def test_topic_tracking(self, parser):
        """Test that topics are tracked across turns."""
        await parser.parse("Tell me about physics")
        result = await parser.parse("Tell me about chemistry")
        
        assert "physics" in result.context["recent_topics"]
        assert "chemistry" in result.context["recent_topics"]
    
    @pytest.mark.asyncio
    async def test_topic_list_limited_to_five(self, parser):
        """Test that topic list is limited to 5 most recent."""
        for i in range(7):
            await parser.parse(f"Tell me about topic{i}")
        
        result = await parser.parse("Tell me about topic7")
        
        # Should only have last 5 topics (3-7)
        assert len(result.context["recent_topics"]) <= 5
        assert "topic7" in result.context["recent_topics"]
    
    @pytest.mark.asyncio
    async def test_user_name_extraction(self, parser):
        """Test that user name is extracted and stored."""
        result = await parser.parse("Hi, I'm Alice")
        
        assert result.context["user_name"] == "Alice"
    
    @pytest.mark.asyncio
    async def test_context_merging(self, parser):
        """Test that additional context can be merged."""
        custom_context = {"custom_key": "custom_value"}
        result = await parser.parse("Hello", context=custom_context)
        
        assert "custom_key" in result.context
        assert result.context["custom_key"] == "custom_value"


class TestFullParsing:
    """Test complete parsing integration."""
    
    @pytest.fixture
    def parser(self):
        """Create parser with real perception subsystem."""
        perception = PerceptionSubsystem(config={"text_model": "all-MiniLM-L6-v2"})
        return LanguageInputParser(perception)
    
    @pytest.mark.asyncio
    async def test_parse_returns_all_components(self, parser):
        """Test that parse returns complete ParseResult."""
        result = await parser.parse("What is your name?")
        
        assert isinstance(result, ParseResult)
        assert result.goals is not None
        assert result.percept is not None
        assert result.intent is not None
        assert result.entities is not None
        assert result.context is not None
    
    @pytest.mark.asyncio
    async def test_parse_question_complete(self, parser):
        """Test complete parsing of a question."""
        result = await parser.parse("What is quantum physics?")
        
        # Check intent
        assert result.intent.type == IntentType.QUESTION
        
        # Check goals
        assert len(result.goals) >= 1
        assert any(g.type == GoalType.RESPOND_TO_USER for g in result.goals)
        
        # Check percept
        assert result.percept is not None
        assert result.percept.modality == "text"
        assert result.percept.embedding is not None
        
        # Check context
        assert result.context["turn_count"] > 0
    
    @pytest.mark.asyncio
    async def test_parse_memory_request_complete(self, parser):
        """Test complete parsing of a memory request."""
        result = await parser.parse("Do you remember our last conversation?")
        
        # Check intent
        assert result.intent.type == IntentType.MEMORY_REQUEST
        
        # Check goals - should have both response and retrieve
        assert len(result.goals) >= 2
        assert any(g.type == GoalType.RESPOND_TO_USER for g in result.goals)
        assert any(g.type == GoalType.RETRIEVE_MEMORY for g in result.goals)
        
        # Check percept has metadata
        assert "intent" in result.percept.metadata
        assert result.percept.metadata["intent"] == IntentType.MEMORY_REQUEST
    
    @pytest.mark.asyncio
    async def test_parse_with_entities_complete(self, parser):
        """Test complete parsing with entity extraction."""
        result = await parser.parse("Tell me about machine learning, Alice")
        
        # Check entities
        assert "names" in result.entities
        assert "Alice" in result.entities["names"]
        assert "topic" in result.entities
        assert "machine learning" in result.entities["topic"]
        
        # Check entities in percept metadata
        assert "entities" in result.percept.metadata
        assert result.percept.metadata["entities"] == result.entities
    
    @pytest.mark.asyncio
    async def test_parse_multiple_turns(self, parser):
        """Test parsing across multiple conversation turns."""
        result1 = await parser.parse("Hello!")
        result2 = await parser.parse("Tell me about AI")
        result3 = await parser.parse("How do you feel about that?")
        
        # Each should have incremented turn count
        assert result1.context["turn_count"] == 1
        assert result2.context["turn_count"] == 2
        assert result3.context["turn_count"] == 3
        
        # Should have appropriate intents
        assert result1.intent.type == IntentType.GREETING
        assert result3.intent.type == IntentType.INTROSPECTION_REQUEST
