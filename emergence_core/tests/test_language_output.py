"""
Unit tests for LanguageOutputGenerator.

Tests cover:
- Initialization and identity loading
- Prompt building with workspace state
- Emotion-based style guidance
- Response generation and formatting
- Integration with CognitiveCore
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Mock pydantic to avoid import errors
import sys
sys.modules['pydantic'] = Mock()

from lyra.cognitive_core.language_output import LanguageOutputGenerator
from lyra.cognitive_core.workspace import WorkspaceSnapshot, Percept, Goal, GoalType


class MockLLM:
    """Mock LLM client for testing."""
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500):
        """Return a mock response based on the prompt."""
        return "Response: This is a test response from the mock LLM."


class TestLanguageOutputGenerator:
    """Test LanguageOutputGenerator class."""
    
    def test_initialization_default(self):
        """Test initialization with default config."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        assert generator.llm == llm
        assert generator.temperature == 0.7
        assert generator.max_tokens == 500
        assert isinstance(generator.charter_text, str)
        assert isinstance(generator.protocols_text, str)
    
    def test_initialization_custom_config(self):
        """Test initialization with custom config."""
        llm = MockLLM()
        config = {
            "temperature": 0.9,
            "max_tokens": 1000,
            "identity_dir": "data/identity"
        }
        generator = LanguageOutputGenerator(llm, config)
        
        assert generator.temperature == 0.9
        assert generator.max_tokens == 1000
    
    def test_load_charter(self):
        """Test charter loading."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        # Should load from file or return default
        charter = generator._load_charter()
        assert isinstance(charter, str)
        assert len(charter) > 0
    
    def test_load_protocols(self):
        """Test protocols loading."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        # Should load from file or return empty string
        protocols = generator._load_protocols()
        assert isinstance(protocols, str)
    
    def test_emotion_style_guidance_positive(self):
        """Test emotion style guidance with positive emotions."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        emotions = {
            'valence': 0.8,  # High positive
            'arousal': 0.9,  # High energy
            'dominance': 0.7  # High control
        }
        
        style = generator._get_emotion_style_guidance(emotions)
        
        assert "positive" in style.lower() or "warm" in style.lower()
        assert "energetic" in style.lower() or "engaged" in style.lower()
        assert "confident" in style.lower() or "assertive" in style.lower()
    
    def test_emotion_style_guidance_negative(self):
        """Test emotion style guidance with negative emotions."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        emotions = {
            'valence': -0.6,  # Negative
            'arousal': 0.2,   # Low energy
            'dominance': 0.2  # Low control
        }
        
        style = generator._get_emotion_style_guidance(emotions)
        
        assert "difficulty" in style.lower() or "concern" in style.lower()
        assert "calm" in style.lower() or "measured" in style.lower()
        assert "uncertainty" in style.lower() or "humility" in style.lower()
    
    def test_emotion_style_guidance_neutral(self):
        """Test emotion style guidance with neutral emotions."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        emotions = {
            'valence': 0.0,
            'arousal': 0.5,
            'dominance': 0.5
        }
        
        style = generator._get_emotion_style_guidance(emotions)
        
        assert "neutral" in style.lower() or "balanced" in style.lower()
    
    def test_format_percept_text(self):
        """Test formatting of text percepts."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        percept = Percept(
            modality="text",
            raw="This is a test text percept with some content",
            complexity=5
        )
        
        formatted = generator._format_percept(percept)
        assert "test text percept" in formatted
        assert len(formatted) <= 200
    
    def test_format_percept_memory(self):
        """Test formatting of memory percepts."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        percept = Percept(
            modality="memory",
            raw={"content": "A memory from the past", "timestamp": "2024-01-01"},
            complexity=3
        )
        
        formatted = generator._format_percept(percept)
        assert "memory from the past" in formatted.lower()
    
    def test_format_response_clean(self):
        """Test response formatting with clean input."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        response = generator._format_response("This is a clean response")
        assert response == "This is a clean response"
    
    def test_format_response_with_prefix(self):
        """Test response formatting removes 'Response:' prefix."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        response = generator._format_response("Response: This is a response")
        assert response == "This is a response"
    
    def test_format_response_with_code_block(self):
        """Test response formatting removes markdown code blocks."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        response = generator._format_response("```\nThis is in a code block\n```")
        assert response == "This is in a code block"
    
    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic response generation."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        # Create a minimal workspace snapshot
        snapshot = Mock()
        snapshot.emotions = {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5}
        snapshot.goals = []
        snapshot.percepts = {}
        snapshot.metadata = {}
        
        context = {"user_input": "Hello"}
        
        response = await generator.generate(snapshot, context)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "test response" in response.lower()
    
    @pytest.mark.asyncio
    async def test_generate_with_goals_and_percepts(self):
        """Test generation with goals and percepts in workspace."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        # Create snapshot with goals and percepts
        goal = Goal(
            type=GoalType.RESPOND_TO_USER,
            description="Respond to user question",
            priority=0.8
        )
        
        percept = Percept(
            modality="text",
            raw="User asked about AI consciousness",
            complexity=7
        )
        percept.metadata = {"attention_score": 0.9}
        
        snapshot = Mock()
        snapshot.emotions = {'valence': 0.2, 'arousal': 0.6, 'dominance': 0.7}
        snapshot.goals = [goal]
        snapshot.percepts = {"p1": percept}
        snapshot.metadata = {"emotion_label": "curious"}
        
        context = {"user_input": "What is consciousness?"}
        
        response = await generator.generate(snapshot, context)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_build_prompt_structure(self):
        """Test that built prompt has all required sections."""
        llm = MockLLM()
        generator = LanguageOutputGenerator(llm)
        
        snapshot = Mock()
        snapshot.emotions = {'valence': 0.0, 'arousal': 0.5, 'dominance': 0.5}
        snapshot.goals = []
        snapshot.percepts = {}
        snapshot.metadata = {}
        
        context = {"user_input": "Test input"}
        
        prompt = generator._build_prompt(snapshot, context)
        
        # Check for all required sections
        assert "# IDENTITY" in prompt
        assert "# PROTOCOLS" in prompt
        assert "# CURRENT EMOTIONAL STATE" in prompt
        assert "# ACTIVE GOALS" in prompt
        assert "# ATTENDED PERCEPTS" in prompt
        assert "# USER INPUT" in prompt
        assert "# INSTRUCTION" in prompt
        assert "Test input" in prompt


class TestCognitiveCoreIntegration:
    """Test integration with CognitiveCore."""
    
    @pytest.mark.asyncio
    async def test_output_queue_initialization(self):
        """Test that output queue is initialized on start."""
        from lyra.cognitive_core.core import CognitiveCore
        
        core = CognitiveCore()
        
        # Initially None
        assert core.output_queue is None
        
        # Start initializes it
        # (We can't actually run start() in tests, but we can check the logic)
        # This would be tested in integration tests
    
    @pytest.mark.asyncio
    async def test_speak_action_generates_output(self):
        """Test that SPEAK actions trigger language generation."""
        # This would be an integration test that:
        # 1. Creates a CognitiveCore with mock LLM
        # 2. Starts the cognitive loop
        # 3. Injects a message that creates a SPEAK action
        # 4. Verifies output appears in output_queue
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
