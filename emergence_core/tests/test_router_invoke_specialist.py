"""
Comprehensive tests for the refactored _invoke_specialist method.

Tests cover:
- Input validation
- Specialist name mapping
- Error handling
- Timeout behavior
- Type safety
- Edge cases
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lyra.router import AdaptiveRouter, SpecialistResponse


@pytest.fixture
def mock_router():
    """Create a mock router with specialists."""
    router = Mock(spec=AdaptiveRouter)
    router.specialists = {
        "philosopher": Mock(),
        "pragmatist": Mock(),
        "artist": Mock(),
        "voice": Mock(),
    }
    # Bind the actual method
    router._invoke_specialist = AdaptiveRouter._invoke_specialist.__get__(router, AdaptiveRouter)
    return router


class TestInvokeSpecialistValidation:
    """Test input validation for _invoke_specialist."""
    
    @pytest.mark.asyncio
    async def test_empty_specialist_name(self, mock_router):
        """Test with empty specialist name."""
        response = await mock_router._invoke_specialist("")
        
        assert response.content.startswith("Error")
        assert response.metadata["error"] == "invalid_specialist_name"
    
    @pytest.mark.asyncio
    async def test_none_specialist_name(self, mock_router):
        """Test with None as specialist name."""
        response = await mock_router._invoke_specialist(None)
        
        assert response.content.startswith("Error")
        assert response.metadata["error"] == "invalid_specialist_name"
    
    @pytest.mark.asyncio
    async def test_non_string_specialist_name(self, mock_router):
        """Test with non-string specialist name."""
        response = await mock_router._invoke_specialist(123)
        
        assert response.content.startswith("Error")
        assert response.metadata["error"] == "invalid_specialist_name"
    
    @pytest.mark.asyncio
    async def test_unknown_specialist(self, mock_router):
        """Test with unknown specialist name."""
        response = await mock_router._invoke_specialist("unknown_specialist")
        
        assert response.content.startswith("Error")
        assert "not found" in response.content
        assert response.metadata["error"] == "specialist_not_found"
        assert "available_specialists" in response.metadata


class TestSpecialistNameMapping:
    """Test legacy specialist name mapping."""
    
    @pytest.mark.asyncio
    async def test_creator_maps_to_artist(self, mock_router):
        """Test that 'creator' maps to 'artist'."""
        mock_specialist = Mock()
        mock_specialist.process = AsyncMock(return_value=Mock(
            content="result",
            metadata={},
            thought_process="thinking",
            confidence=0.9
        ))
        mock_router.specialists["artist"] = mock_specialist
        
        response = await mock_router._invoke_specialist("creator", message="test")
        
        assert mock_specialist.process.called
        assert response.source == "artist"
    
    @pytest.mark.asyncio
    async def test_logician_maps_to_philosopher(self, mock_router):
        """Test that 'logician' maps to 'philosopher'."""
        mock_specialist = Mock()
        mock_specialist.process = AsyncMock(return_value=Mock(
            content="result",
            metadata={},
            thought_process="thinking",
            confidence=0.9
        ))
        mock_router.specialists["philosopher"] = mock_specialist
        
        response = await mock_router._invoke_specialist("logician", message="test")
        
        assert mock_specialist.process.called
        assert response.source == "philosopher"
    
    @pytest.mark.asyncio
    async def test_case_insensitive_names(self, mock_router):
        """Test that specialist names are case-insensitive."""
        mock_specialist = Mock()
        mock_specialist.process = AsyncMock(return_value=Mock(
            content="result",
            metadata={},
            thought_process="thinking",
            confidence=0.9
        ))
        mock_router.specialists["pragmatist"] = mock_specialist
        
        # Try uppercase
        response = await mock_router._invoke_specialist("PRAGMATIST", message="test")
        assert mock_specialist.process.called
        
        # Try mixed case
        mock_specialist.process.reset_mock()
        response = await mock_router._invoke_specialist("PragMatist", message="test")
        assert mock_specialist.process.called


class TestParameterHandling:
    """Test parameter extraction and context building."""
    
    @pytest.mark.asyncio
    async def test_message_parameter(self, mock_router):
        """Test that message parameter is extracted correctly."""
        mock_specialist = Mock()
        captured_args = {}
        
        async def capture_call(message, context):
            captured_args["message"] = message
            captured_args["context"] = context
            return Mock(content="result", metadata={}, thought_process="", confidence=0.9)
        
        mock_specialist.process = capture_call
        mock_router.specialists["pragmatist"] = mock_specialist
        
        await mock_router._invoke_specialist("pragmatist", message="Hello")
        
        assert captured_args["message"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_query_parameter_fallback(self, mock_router):
        """Test that query parameter works as fallback for message."""
        mock_specialist = Mock()
        captured_args = {}
        
        async def capture_call(message, context):
            captured_args["message"] = message
            return Mock(content="result", metadata={}, thought_process="", confidence=0.9)
        
        mock_specialist.process = capture_call
        mock_router.specialists["pragmatist"] = mock_specialist
        
        await mock_router._invoke_specialist("pragmatist", query="Test query")
        
        assert captured_args["message"] == "Test query"
    
    @pytest.mark.asyncio
    async def test_context_parameter(self, mock_router):
        """Test that context parameter is passed correctly."""
        mock_specialist = Mock()
        captured_args = {}
        
        async def capture_call(message, context):
            captured_args["context"] = context
            return Mock(content="result", metadata={}, thought_process="", confidence=0.9)
        
        mock_specialist.process = capture_call
        mock_router.specialists["pragmatist"] = mock_specialist
        
        test_context = {"mood": "happy", "user": "Alice"}
        await mock_router._invoke_specialist("pragmatist", message="test", context=test_context)
        
        assert captured_args["context"]["mood"] == "happy"
        assert captured_args["context"]["user"] == "Alice"
    
    @pytest.mark.asyncio
    async def test_extra_kwargs_merged_into_context(self, mock_router):
        """Test that extra kwargs are merged into context."""
        mock_specialist = Mock()
        captured_args = {}
        
        async def capture_call(message, context):
            captured_args["context"] = context
            return Mock(content="result", metadata={}, thought_process="", confidence=0.9)
        
        mock_specialist.process = capture_call
        mock_router.specialists["pragmatist"] = mock_specialist
        
        await mock_router._invoke_specialist(
            "pragmatist", 
            message="test",
            ritual="morning",
            desires=["learn", "grow"]
        )
        
        assert captured_args["context"]["ritual"] == "morning"
        assert captured_args["context"]["desires"] == ["learn", "grow"]
    
    @pytest.mark.asyncio
    async def test_non_dict_context_handling(self, mock_router):
        """Test handling when context is not a dict."""
        mock_specialist = Mock()
        captured_args = {}
        
        async def capture_call(message, context):
            captured_args["context"] = context
            return Mock(content="result", metadata={}, thought_process="", confidence=0.9)
        
        mock_specialist.process = capture_call
        mock_router.specialists["pragmatist"] = mock_specialist
        
        # Pass a non-dict context
        await mock_router._invoke_specialist("pragmatist", message="test", context="not a dict")
        
        # Should be wrapped in dict
        assert isinstance(captured_args["context"], dict)
        assert "original_context" in captured_args["context"]
    
    @pytest.mark.asyncio
    async def test_non_string_message_conversion(self, mock_router):
        """Test that non-string messages are converted to strings."""
        mock_specialist = Mock()
        captured_args = {}
        
        async def capture_call(message, context):
            captured_args["message"] = message
            return Mock(content="result", metadata={}, thought_process="", confidence=0.9)
        
        mock_specialist.process = capture_call
        mock_router.specialists["pragmatist"] = mock_specialist
        
        # Pass numeric message
        await mock_router._invoke_specialist("pragmatist", message=123)
        
        assert captured_args["message"] == "123"
        assert isinstance(captured_args["message"], str)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_uninitialized_specialist(self, mock_router):
        """Test handling of uninitialized (None) specialist."""
        mock_router.specialists["philosopher"] = None
        
        response = await mock_router._invoke_specialist("philosopher", message="test")
        
        assert "not initialized" in response.content
        assert response.metadata["error"] == "specialist_not_initialized"
    
    @pytest.mark.asyncio
    async def test_specialist_process_exception(self, mock_router):
        """Test handling when specialist.process raises exception."""
        mock_specialist = Mock()
        mock_specialist.process = AsyncMock(side_effect=ValueError("Test error"))
        mock_router.specialists["pragmatist"] = mock_specialist
        
        response = await mock_router._invoke_specialist("pragmatist", message="test")
        
        assert "Error during" in response.content
        assert "Test error" in response.content
        assert "error_type" in response.metadata
    
    @pytest.mark.asyncio
    async def test_specialist_timeout(self, mock_router):
        """Test timeout handling for slow specialist."""
        mock_specialist = Mock()
        
        async def slow_process(message, context):
            await asyncio.sleep(35)  # Longer than 30s timeout
            return Mock(content="result", metadata={}, thought_process="", confidence=0.9)
        
        mock_specialist.process = slow_process
        mock_router.specialists["philosopher"] = mock_specialist
        
        response = await mock_router._invoke_specialist("philosopher", message="test")
        
        assert "timed out" in response.content.lower()
        assert response.metadata["error"] == "timeout"
    
    @pytest.mark.asyncio
    async def test_invalid_specialist_output(self, mock_router):
        """Test handling when specialist returns invalid output."""
        mock_specialist = Mock()
        mock_specialist.process = AsyncMock(return_value="not a proper output object")
        mock_router.specialists["pragmatist"] = mock_specialist
        
        response = await mock_router._invoke_specialist("pragmatist", message="test")
        
        # Should handle gracefully and return string
        assert isinstance(response.content, str)


class TestResponseConversion:
    """Test conversion from SpecialistOutput to SpecialistResponse."""
    
    @pytest.mark.asyncio
    async def test_successful_conversion(self, mock_router):
        """Test successful conversion of specialist output."""
        mock_specialist = Mock()
        specialist_output = Mock(
            content="This is the result",
            metadata={"key": "value"},
            thought_process="Deep thinking",
            confidence=0.95
        )
        mock_specialist.process = AsyncMock(return_value=specialist_output)
        mock_router.specialists["philosopher"] = mock_specialist
        
        response = await mock_router._invoke_specialist("philosopher", message="test")
        
        assert response.content == "This is the result"
        assert response.metadata["key"] == "value"
        assert response.metadata["thought_process"] == "Deep thinking"
        assert response.metadata["confidence"] == 0.95
        assert response.source == "philosopher"
    
    @pytest.mark.asyncio
    async def test_metadata_preservation(self, mock_router):
        """Test that all metadata is preserved in conversion."""
        mock_specialist = Mock()
        specialist_output = Mock(
            content="result",
            metadata={"role": "test", "custom": "data", "nested": {"key": "val"}},
            thought_process="process",
            confidence=0.8
        )
        mock_specialist.process = AsyncMock(return_value=specialist_output)
        mock_router.specialists["voice"] = mock_specialist
        
        response = await mock_router._invoke_specialist("voice", message="test")
        
        # All original metadata should be present plus thought_process and confidence
        assert response.metadata["role"] == "test"
        assert response.metadata["custom"] == "data"
        assert response.metadata["nested"]["key"] == "val"
        assert response.metadata["thought_process"] == "process"
        assert response.metadata["confidence"] == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
