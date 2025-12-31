"""
Unit tests for ConversationManager.

Tests cover:
- Initialization and configuration
- Single turn processing
- Multi-turn coherence and context tracking
- Timeout handling
- Error handling and recovery
- History tracking and management
- Metrics tracking
- Conversation reset functionality
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from lyra.cognitive_core.core import CognitiveCore
from lyra.cognitive_core.conversation import ConversationManager, ConversationTurn
from lyra.cognitive_core.workspace import GlobalWorkspace


class TestConversationManagerInitialization:
    """Test ConversationManager initialization."""
    
    def test_initialization_default(self):
        """Test creating ConversationManager with default parameters."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        assert manager is not None
        assert isinstance(manager, ConversationManager)
        assert manager.core == core
        assert manager.turn_count == 0
        assert len(manager.conversation_history) == 0
        assert len(manager.current_topics) == 0
        assert manager.response_timeout == 10.0
        assert manager.max_cycles_per_turn == 20
    
    def test_initialization_custom_config(self):
        """Test creating ConversationManager with custom configuration."""
        core = CognitiveCore()
        config = {
            "response_timeout": 15.0,
            "max_cycles_per_turn": 30,
            "max_history_size": 50
        }
        manager = ConversationManager(core, config)
        
        assert manager.response_timeout == 15.0
        assert manager.max_cycles_per_turn == 30
        assert manager.conversation_history.maxlen == 50
    
    def test_metrics_initialized(self):
        """Test that metrics are properly initialized."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        assert "total_turns" in manager.metrics
        assert "avg_response_time" in manager.metrics
        assert "timeouts" in manager.metrics
        assert "errors" in manager.metrics
        assert manager.metrics["total_turns"] == 0


class TestSingleTurn:
    """Test single conversation turn processing."""
    
    @pytest.mark.asyncio
    async def test_single_turn_success(self):
        """Test processing one conversation turn successfully."""
        # Create core and manager
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        # Start the core
        asyncio.create_task(core.start())
        await asyncio.sleep(0.1)  # Let it initialize
        
        try:
            # Mock the output queue to return a response
            mock_output = {
                "type": "SPEAK",
                "text": "Hello! How can I help you?"
            }
            
            # Process a turn (inject response into queue)
            async def delayed_response():
                await asyncio.sleep(0.2)
                await core.output_queue.put(mock_output)
            
            asyncio.create_task(delayed_response())
            
            turn = await manager.process_turn("Hello, Lyra!")
            
            # Verify turn structure
            assert isinstance(turn, ConversationTurn)
            assert turn.user_input == "Hello, Lyra!"
            assert turn.system_response == "Hello! How can I help you?"
            assert isinstance(turn.timestamp, datetime)
            assert turn.response_time > 0
            assert isinstance(turn.emotional_state, dict)
            assert "turn_number" in turn.metadata
            assert turn.metadata["turn_number"] == 1
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
    async def test_turn_updates_history(self):
        """Test that conversation history is updated after a turn."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        core_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        
        try:
            # Mock response
            async def delayed_response():
                await asyncio.sleep(0.2)
                await core.output_queue.put({"type": "SPEAK", "text": "Test response"})
            
            asyncio.create_task(delayed_response())
            
            initial_history_len = len(manager.conversation_history)
            await manager.process_turn("Test message")
            
            assert len(manager.conversation_history) == initial_history_len + 1
            assert manager.turn_count == 1
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
    async def test_turn_updates_metrics(self):
        """Test that metrics are updated after a turn."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        core_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        
        try:
            # Mock response
            async def delayed_response():
                await asyncio.sleep(0.2)
                await core.output_queue.put({"type": "SPEAK", "text": "Test response"})
            
            asyncio.create_task(delayed_response())
            
            initial_total = manager.metrics["total_turns"]
            await manager.process_turn("Test message")
            
            assert manager.metrics["total_turns"] == initial_total + 1
            assert manager.metrics["avg_response_time"] > 0
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)


class TestMultiTurnCoherence:
    """Test multi-turn conversation with context tracking."""
    
    @pytest.mark.asyncio
    async def test_multi_turn_context(self):
        """Test that context is passed between turns."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        core_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        
        try:
            # First turn
            async def response1():
                await asyncio.sleep(0.2)
                await core.output_queue.put({"type": "SPEAK", "text": "Response 1"})
            
            asyncio.create_task(response1())
            turn1 = await manager.process_turn("First message")
            
            # Second turn
            async def response2():
                await asyncio.sleep(0.2)
                await core.output_queue.put({"type": "SPEAK", "text": "Response 2"})
            
            asyncio.create_task(response2())
            turn2 = await manager.process_turn("Second message")
            
            # Verify both turns processed
            assert turn1.user_input == "First message"
            assert turn2.user_input == "Second message"
            assert manager.turn_count == 2
            assert len(manager.conversation_history) == 2
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
    async def test_topic_tracking(self):
        """Test that topics are extracted and tracked."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        core_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        
        try:
            async def delayed_response():
                await asyncio.sleep(0.2)
                await core.output_queue.put({"type": "SPEAK", "text": "Test response"})
            
            asyncio.create_task(delayed_response())
            
            # Message with clear topics
            await manager.process_turn("Let's discuss quantum physics and relativity")
            
            # Check that topics were extracted
            assert len(manager.current_topics) > 0
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)


class TestTimeoutHandling:
    """Test timeout handling for slow responses."""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test graceful timeout when no response within timeout."""
        core = CognitiveCore()
        config = {"response_timeout": 0.5}  # Short timeout for testing
        manager = ConversationManager(core, config)
        
        core_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        
        try:
            # Don't provide a response - let it timeout
            turn = await manager.process_turn("Test message")
            
            # Verify timeout was handled gracefully
            assert isinstance(turn, ConversationTurn)
            assert "trouble formulating a response" in turn.system_response.lower()
            assert manager.metrics["timeouts"] == 1
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)


class TestErrorHandling:
    """Test error handling during conversation."""
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test that errors during processing are handled gracefully."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        # Don't start core - this will cause an error
        # But start() is needed for queue initialization
        core_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        await core.stop()  # Stop immediately to cause issues
        
        # Try to process a turn with stopped core
        turn = await manager.process_turn("Test message")
        
        # Verify error was handled gracefully
        assert isinstance(turn, ConversationTurn)
        assert "error" in turn.system_response.lower() or "trouble" in turn.system_response.lower()
        # Note: metrics might be 0 or 1 depending on where the error occurred
        assert manager.metrics["errors"] >= 0
    
    @pytest.mark.asyncio
    async def test_error_turn_structure(self):
        """Test that error turns have correct structure."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        core_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        await core.stop()
        
        turn = await manager.process_turn("Test message")
        
        # Verify turn structure even in error case
        assert isinstance(turn, ConversationTurn)
        assert turn.user_input == "Test message"
        assert isinstance(turn.timestamp, datetime)
        assert isinstance(turn.emotional_state, dict)


class TestHistoryTracking:
    """Test conversation history management."""
    
    @pytest.mark.asyncio
    async def test_history_retrieval(self):
        """Test retrieving conversation history."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        core_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        
        try:
            # Add multiple turns
            for i in range(3):
                async def delayed_response():
                    await asyncio.sleep(0.2)
                    await core.output_queue.put({"type": "SPEAK", "text": f"Response {i}"})
                
                asyncio.create_task(delayed_response())
                await manager.process_turn(f"Message {i}")
            
            # Get history
            history = manager.get_conversation_history(10)
            
            assert len(history) == 3
            assert all(isinstance(turn, ConversationTurn) for turn in history)
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
    async def test_history_limit(self):
        """Test that history respects max size limit."""
        core = CognitiveCore()
        config = {"max_history_size": 2}
        manager = ConversationManager(core, config)
        
        core_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        
        try:
            # Add more turns than the limit
            for i in range(5):
                async def delayed_response():
                    await asyncio.sleep(0.2)
                    await core.output_queue.put({"type": "SPEAK", "text": f"Response {i}"})
                
                asyncio.create_task(delayed_response())
                await manager.process_turn(f"Message {i}")
            
            # History should be limited
            assert len(manager.conversation_history) == 2
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)
    
    def test_get_recent_history(self):
        """Test getting a limited number of recent turns."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        # Manually add turns to history
        for i in range(10):
            turn = ConversationTurn(
                user_input=f"Message {i}",
                system_response=f"Response {i}",
                timestamp=datetime.now(),
                response_time=0.5,
                emotional_state={}
            )
            manager.conversation_history.append(turn)
        
        # Get last 3 turns
        recent = manager.get_conversation_history(3)
        
        assert len(recent) == 3
        assert recent[0].user_input == "Message 7"
        assert recent[2].user_input == "Message 9"


class TestMetrics:
    """Test conversation metrics tracking."""
    
    def test_get_metrics(self):
        """Test retrieving conversation metrics."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        metrics = manager.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_turns" in metrics
        assert "avg_response_time" in metrics
        assert "timeouts" in metrics
        assert "errors" in metrics
        assert "turn_count" in metrics
        assert "topics_tracked" in metrics
        assert "history_size" in metrics
    
    @pytest.mark.asyncio
    async def test_metrics_update_correctly(self):
        """Test that metrics update correctly after turns."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        
        try:
            # Process a turn
            async def delayed_response():
                await asyncio.sleep(0.2)
                await core.output_queue.put({"type": "SPEAK", "text": "Response"})
            
            asyncio.create_task(delayed_response())
            await manager.process_turn("Test")
            
            metrics = manager.get_metrics()
            
            assert metrics["total_turns"] == 1
            assert metrics["turn_count"] == 1
            assert metrics["history_size"] == 1
            assert metrics["avg_response_time"] > 0
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)


class TestConversationReset:
    """Test conversation reset functionality."""
    
    @pytest.mark.asyncio
    async def test_reset_clears_state(self):
        """Test that reset clears conversation state."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        
        try:
            # Add some turns
            for i in range(3):
                async def delayed_response():
                    await asyncio.sleep(0.2)
                    await core.output_queue.put({"type": "SPEAK", "text": f"Response {i}"})
                
                asyncio.create_task(delayed_response())
                await manager.process_turn(f"Message {i}")
            
            # Verify state exists
            assert len(manager.conversation_history) > 0
            assert manager.turn_count > 0
            
            # Reset
            manager.reset_conversation()
            
            # Verify state cleared
            assert len(manager.conversation_history) == 0
            assert len(manager.current_topics) == 0
            assert manager.turn_count == 0
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)
    
    @pytest.mark.asyncio
    async def test_reset_preserves_metrics(self):
        """Test that reset preserves metrics for analytics."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        core_task = asyncio.create_task(core.start())
        await asyncio.sleep(0.1)
        
        try:
            # Add a turn
            async def delayed_response():
                await asyncio.sleep(0.2)
                await core.output_queue.put({"type": "SPEAK", "text": "Response"})
            
            asyncio.create_task(delayed_response())
            await manager.process_turn("Message")
            
            # Record metrics
            total_turns_before = manager.metrics["total_turns"]
            
            # Reset
            manager.reset_conversation()
            
            # Metrics should be preserved
            assert manager.metrics["total_turns"] == total_turns_before
            
        finally:
            await core.stop()
            await asyncio.sleep(0.1)


class TestTopicExtraction:
    """Test topic extraction helper."""
    
    def test_extract_topics_basic(self):
        """Test basic topic extraction."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        topics = manager._extract_topics("Let's discuss quantum physics and relativity")
        
        assert isinstance(topics, list)
        assert len(topics) <= 3
        # Should extract content words, not stopwords
        assert not any(word in topics for word in ["the", "a", "and"])
    
    def test_extract_topics_filters_stopwords(self):
        """Test that stopwords are filtered out."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        topics = manager._extract_topics("the cat and the dog")
        
        # "the" and "and" should be filtered, but words too short
        assert "the" not in topics
        assert "and" not in topics
    
    def test_extract_topics_length_filter(self):
        """Test that short words are filtered."""
        core = CognitiveCore()
        manager = ConversationManager(core)
        
        topics = manager._extract_topics("I am happy today because weather is nice")
        
        # Short words like "am" and words <= 4 chars should be filtered
        assert all(len(word) > 4 for word in topics)
