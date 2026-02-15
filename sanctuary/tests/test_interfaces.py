"""
Unit tests for language interfaces placeholder classes.

Tests cover:
- Proper initialization of interface classes
- Import structure and module organization
- Type hints and docstring presence
- Data model validation
"""

import pytest
from pathlib import Path

from mind.interfaces.language_input import LanguageInputParser, InputIntent, ParsedInput
from mind.interfaces.language_output import LanguageOutputGenerator, OutputMode, GeneratedOutput


class TestLanguageInputParser:
    """Test LanguageInputParser class initialization and data models"""
    
    def test_parser_initialization_default(self):
        """Test creating LanguageInputParser with default parameters"""
        parser = LanguageInputParser()
        assert parser is not None
        assert isinstance(parser, LanguageInputParser)
    
    def test_parser_initialization_custom(self):
        """Test creating LanguageInputParser with custom parameters"""
        parser = LanguageInputParser(
            llm_model_name="test-model",
            context_window_size=20,
            max_input_length=1024
        )
        assert parser is not None
    
    def test_input_intent_enum(self):
        """Test InputIntent enum values"""
        assert InputIntent.QUERY.value == "query"
        assert InputIntent.COMMAND.value == "command"
        assert InputIntent.STATEMENT.value == "statement"
        assert InputIntent.SOCIAL.value == "social"
        assert InputIntent.META.value == "meta"
    
    def test_parsed_input_creation(self):
        """Test creating ParsedInput data class"""
        parsed = ParsedInput(
            raw_text="Hello, how are you?",
            intent=InputIntent.SOCIAL,
            goals=["greet"],
            facts=[],
            emotional_tone=0.5,
            entities=["you"],
            urgency=0.3
        )
        assert parsed is not None
        assert parsed.raw_text == "Hello, how are you?"
        assert parsed.intent == InputIntent.SOCIAL
        assert len(parsed.goals) == 1
        assert parsed.metadata is not None
    
    def test_parsed_input_defaults(self):
        """Test ParsedInput with minimal parameters"""
        parsed = ParsedInput(
            raw_text="Test",
            intent=InputIntent.QUERY,
            goals=[],
            facts=[],
            emotional_tone=0.0,
            entities=[],
            urgency=0.5
        )
        assert parsed.metadata is not None
        assert isinstance(parsed.metadata, dict)
    
    def test_parser_has_docstring(self):
        """Test that LanguageInputParser has comprehensive docstring"""
        assert LanguageInputParser.__doc__ is not None
        assert len(LanguageInputParser.__doc__) > 100
        assert "parse" in LanguageInputParser.__doc__.lower()
        assert "peripheral" in LanguageInputParser.__doc__.lower()


class TestLanguageOutputGenerator:
    """Test LanguageOutputGenerator class initialization and data models"""
    
    def test_generator_initialization_default(self):
        """Test creating LanguageOutputGenerator with default parameters"""
        generator = LanguageOutputGenerator()
        assert generator is not None
        assert isinstance(generator, LanguageOutputGenerator)
    
    def test_generator_initialization_custom(self):
        """Test creating LanguageOutputGenerator with custom parameters"""
        generator = LanguageOutputGenerator(
            llm_model_name="test-model",
            personality_template="Custom template",
            default_mode=OutputMode.REFLECTIVE,
            max_output_length=1024
        )
        assert generator is not None
    
    def test_output_mode_enum(self):
        """Test OutputMode enum values"""
        assert OutputMode.DIRECT.value == "direct"
        assert OutputMode.REFLECTIVE.value == "reflective"
        assert OutputMode.CREATIVE.value == "creative"
        assert OutputMode.TECHNICAL.value == "technical"
        assert OutputMode.CONVERSATIONAL.value == "conversational"
    
    def test_generated_output_creation(self):
        """Test creating GeneratedOutput data class"""
        output = GeneratedOutput(
            text="This is a generated response.",
            mode=OutputMode.CONVERSATIONAL,
            workspace_snapshot={"goals": ["respond"]},
            confidence=0.85,
            alternative_responses=["Another response"],
            generation_time=0.5
        )
        assert output is not None
        assert output.text == "This is a generated response."
        assert output.mode == OutputMode.CONVERSATIONAL
        assert output.confidence == 0.85
        assert output.metadata is not None
    
    def test_generated_output_defaults(self):
        """Test GeneratedOutput with minimal parameters"""
        output = GeneratedOutput(
            text="Test output",
            mode=OutputMode.DIRECT,
            workspace_snapshot={},
            confidence=1.0,
            alternative_responses=[],
            generation_time=0.1
        )
        assert output.metadata is not None
        assert isinstance(output.metadata, dict)
    
    def test_generator_has_docstring(self):
        """Test that LanguageOutputGenerator has comprehensive docstring"""
        assert LanguageOutputGenerator.__doc__ is not None
        assert len(LanguageOutputGenerator.__doc__) > 100
        assert "generate" in LanguageOutputGenerator.__doc__.lower() or "output" in LanguageOutputGenerator.__doc__.lower()
        assert "peripheral" in LanguageOutputGenerator.__doc__.lower()


class TestInterfacesIntegration:
    """Test integration between input and output interfaces"""
    
    def test_interfaces_can_be_instantiated_together(self):
        """Test that both interfaces can be created simultaneously"""
        parser = LanguageInputParser()
        generator = LanguageOutputGenerator()
        assert parser is not None
        assert generator is not None
    
    def test_input_output_data_flow(self):
        """Test that ParsedInput could theoretically feed workspace for GeneratedOutput"""
        # Create a parsed input
        parsed = ParsedInput(
            raw_text="What is consciousness?",
            intent=InputIntent.QUERY,
            goals=["explain consciousness"],
            facts=[],
            emotional_tone=0.0,
            entities=["consciousness"],
            urgency=0.5
        )
        
        # Simulate workspace state from parsed input
        workspace_snapshot = {
            "goals": parsed.goals,
            "entities": parsed.entities,
            "emotional_context": parsed.emotional_tone
        }
        
        # Create a generated output
        output = GeneratedOutput(
            text="Consciousness is...",
            mode=OutputMode.REFLECTIVE,
            workspace_snapshot=workspace_snapshot,
            confidence=0.9,
            alternative_responses=[],
            generation_time=1.2
        )
        
        assert output is not None
        assert output.workspace_snapshot["goals"] == parsed.goals


class TestTypeHints:
    """Test that all classes have proper type hints"""
    
    def test_language_input_parser_type_hints(self):
        """Test LanguageInputParser has type hints"""
        import inspect
        sig = inspect.signature(LanguageInputParser.__init__)
        # Check that parameters have annotations (may be str or 'str' with __future__ annotations)
        assert sig.parameters['llm_model_name'].annotation in (str, 'str')
        assert sig.parameters['context_window_size'].annotation in (int, 'int')
        assert sig.parameters['max_input_length'].annotation in (int, 'int')
        assert sig.return_annotation is None or sig.return_annotation == 'None'

    def test_language_output_generator_type_hints(self):
        """Test LanguageOutputGenerator has type hints"""
        import inspect
        sig = inspect.signature(LanguageOutputGenerator.__init__)
        # Check that parameters have annotations (may be str or 'str' with __future__ annotations)
        assert sig.parameters['llm_model_name'].annotation in (str, 'str')
        assert sig.parameters['max_output_length'].annotation in (int, 'int')
        assert sig.return_annotation is None or sig.return_annotation == 'None'
