"""
Unit tests for context_manager.py

Tests cover:
- ContextWindow edge cases and performance
- ContextManager functionality
- Context shift detection accuracy
- Error handling and validation
"""

import pytest
import time
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lyra.context_manager import ContextWindow, ContextManager


class TestContextWindow:
    """Test ContextWindow class edge cases and performance."""
    
    def test_initialization_validation(self):
        """Test parameter validation on initialization."""
        # Valid initialization
        window = ContextWindow(max_size=10, decay_rate=0.5)
        assert window.max_size == 10
        assert window.decay_rate == 0.5
        
        # Invalid max_size
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            ContextWindow(max_size=0)
        
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            ContextWindow(max_size=-5)
        
        # Invalid decay_rate
        with pytest.raises(ValueError, match="decay_rate must be in"):
            ContextWindow(decay_rate=-0.1)
        
        with pytest.raises(ValueError, match="decay_rate must be in"):
            ContextWindow(decay_rate=1.5)
    
    def test_add_item_validation(self):
        """Test add() validates input types."""
        window = ContextWindow()
        
        # Valid dict
        window.add({"test": "value"})
        assert len(window.get_all()) == 1
        
        # Invalid types
        with pytest.raises(TypeError, match="Item must be dict"):
            window.add("not a dict")
        
        with pytest.raises(TypeError, match="Item must be dict"):
            window.add(123)
        
        with pytest.raises(TypeError, match="Item must be dict"):
            window.add(None)
    
    def test_max_size_enforcement(self):
        """Test that window respects max_size limit."""
        window = ContextWindow(max_size=3)
        
        for i in range(5):
            window.add({"item": i})
        
        # Should only keep last 3 items
        items = window.get_all()
        assert len(items) == 3
        assert items[0]["item"] == 2
        assert items[1]["item"] == 3
        assert items[2]["item"] == 4
    
    def test_relevance_decay(self):
        """Test that relevance decays over time."""
        window = ContextWindow(max_size=10, decay_rate=0.5)
        
        # Add item
        window.add({"test": "data"})
        
        # Initial relevance should be 1.0
        items = window.get_all()
        assert items[0]["relevance"] == 1.0
        
        # Simulate time passing (1 minute)
        items[0]["added_at"] = (datetime.now() - timedelta(minutes=1)).isoformat()
        
        # Get relevant items to trigger decay calculation
        relevant = window.get_relevant(threshold=0.0)
        
        # Relevance should have decayed (e^(-0.5 * 1) ≈ 0.606)
        assert relevant[0]["relevance"] < 1.0
        assert relevant[0]["relevance"] > 0.5
    
    def test_get_relevant_threshold_validation(self):
        """Test get_relevant() validates threshold parameter."""
        window = ContextWindow()
        window.add({"test": "item"})
        
        # Valid thresholds
        window.get_relevant(threshold=0.0)
        window.get_relevant(threshold=0.5)
        window.get_relevant(threshold=1.0)
        
        # Invalid thresholds
        with pytest.raises(ValueError, match="threshold must be in"):
            window.get_relevant(threshold=-0.1)
        
        with pytest.raises(ValueError, match="threshold must be in"):
            window.get_relevant(threshold=1.5)
    
    def test_get_relevant_filtering(self):
        """Test get_relevant() correctly filters by threshold."""
        window = ContextWindow(max_size=10, decay_rate=0.2)
        
        # Add items at different times
        now = datetime.now()
        
        window.add({"item": 1})  # Recent, relevance ~1.0
        window.window[-1]["added_at"] = now.isoformat()
        
        window.add({"item": 2})  # 5 min ago, relevance ~0.368
        window.window[-1]["added_at"] = (now - timedelta(minutes=5)).isoformat()
        
        window.add({"item": 3})  # 10 min ago, relevance ~0.135
        window.window[-1]["added_at"] = (now - timedelta(minutes=10)).isoformat()
        
        # Threshold 0.5: should only get item 1
        relevant_high = window.get_relevant(threshold=0.5)
        assert len(relevant_high) == 1
        assert relevant_high[0]["item"] == 1
        
        # Threshold 0.2: should get items 1 and 2
        relevant_mid = window.get_relevant(threshold=0.2)
        assert len(relevant_mid) == 2
        
        # Threshold 0.0: should get all items
        relevant_all = window.get_relevant(threshold=0.0)
        assert len(relevant_all) == 3
    
    def test_clear_functionality(self):
        """Test clear() removes all items."""
        window = ContextWindow()
        
        for i in range(5):
            window.add({"item": i})
        
        assert len(window.get_all()) == 5
        
        window.clear()
        
        assert len(window.get_all()) == 0
        assert len(window.window) == 0
    
    def test_invalid_item_handling(self):
        """Test that invalid items are handled gracefully."""
        window = ContextWindow()
        
        # Add valid item
        window.add({"test": "valid"})
        
        # Manually corrupt an item (simulate data corruption)
        window.window[0]["added_at"] = "invalid_datetime"
        
        # Should handle gracefully and mark as irrelevant
        relevant = window.get_relevant(threshold=0.0)
        assert relevant[0]["relevance"] == 0.0


class TestContextManager:
    """Test ContextManager functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    def test_initialization(self, temp_dir):
        """Test ContextManager initializes correctly."""
        manager = ContextManager(persistence_dir=temp_dir)
        
        assert manager.persistence_dir.exists()
        assert manager.current_topic is None
        assert manager.interaction_count == 0
        assert manager.topic_transition_count == 0
    
    def test_update_conversation_context(self, temp_dir):
        """Test conversation context updates correctly."""
        manager = ContextManager(persistence_dir=temp_dir)
        
        manager.update_conversation_context(
            user_input="Hello",
            system_response="Hi there",
            detected_topic="greeting",
            emotional_tone=["friendly"]
        )
        
        assert manager.interaction_count == 1
        assert manager.current_topic == "greeting"
        
        # Check conversation context
        context = manager.conversation_context.get_all()
        assert len(context) == 1
        assert context[0]["user_input"] == "Hello"
        assert context[0]["topic"] == "greeting"
    
    def test_topic_transition_tracking(self, temp_dir):
        """Test topic transitions are tracked correctly."""
        manager = ContextManager(persistence_dir=temp_dir)
        
        # First topic
        manager.update_conversation_context(
            user_input="What is memory?",
            system_response="Memory is...",
            detected_topic="memory"
        )
        
        assert manager.current_topic == "memory"
        assert manager.previous_topic is None
        assert manager.topic_transition_count == 1
        
        # Second topic
        manager.update_conversation_context(
            user_input="Tell me about pizza",
            system_response="Pizza is...",
            detected_topic="food"
        )
        
        assert manager.current_topic == "food"
        assert manager.previous_topic == "memory"
        assert manager.topic_transition_count == 2
        
        # Check transition marker
        context = manager.conversation_context.get_all()
        transitions = [c for c in context if c.get("type") == "topic_transition"]
        assert len(transitions) == 1
        assert transitions[0]["from_topic"] == "memory"
        assert transitions[0]["to_topic"] == "food"
    
    def test_context_shift_detection(self, temp_dir):
        """Test context shift detection with various inputs."""
        manager = ContextManager(persistence_dir=temp_dir)
        
        # Build context on one topic
        manager.update_conversation_context("Tell me about memory", "Memory is...", "memory")
        manager.update_conversation_context("How does memory work?", "It works by...", "memory")
        
        current_context = manager.conversation_context.get_all()
        
        # Similar input (no shift)
        shift, similarity = manager.detect_context_shift(
            "What about memory systems?",
            current_context
        )
        assert not shift
        assert similarity > 0.3
        
        # Different input (shift detected)
        shift, similarity = manager.detect_context_shift(
            "Let's talk about pizza recipes",
            current_context
        )
        assert shift
        assert similarity < 0.3
    
    def test_extract_recent_inputs(self, temp_dir):
        """Test _extract_recent_inputs helper function."""
        manager = ContextManager(persistence_dir=temp_dir)
        
        # Add various context types
        context = [
            {"type": "conversation", "user_input": "input1"},
            {"type": "topic_transition", "from_topic": "a", "to_topic": "b"},
            {"type": "conversation", "user_input": "input2"},
            {"type": "conversation", "user_input": ""},  # Empty input
            {"type": "conversation", "user_input": "input3"},
        ]
        
        recent = manager._extract_recent_inputs(context, max_count=5)
        
        # Should only get conversation types with non-empty inputs
        assert len(recent) == 3
        assert "input1" in recent
        assert "input2" in recent
        assert "input3" in recent
    
    def test_calculate_word_similarity(self, temp_dir):
        """Test _calculate_word_similarity function."""
        manager = ContextManager(persistence_dir=temp_dir)
        
        # Identical texts
        similarity = manager._calculate_word_similarity(
            "hello world",
            ["hello world"]
        )
        assert similarity == 1.0
        
        # Completely different
        similarity = manager._calculate_word_similarity(
            "hello world",
            ["foo bar baz"]
        )
        assert similarity == 0.0
        
        # Partial overlap
        similarity = manager._calculate_word_similarity(
            "hello world test",
            ["hello world example"]
        )
        assert 0.0 < similarity < 1.0
        
        # Empty inputs
        similarity = manager._calculate_word_similarity("", ["test"])
        assert similarity == 1.0  # No change
        
        similarity = manager._calculate_word_similarity("test", [""])
        assert similarity == 1.0  # No change
    
    def test_get_context_summary(self, temp_dir):
        """Test get_context_summary returns correct stats."""
        manager = ContextManager(persistence_dir=temp_dir)
        
        # Add some interactions
        for i in range(3):
            manager.update_conversation_context(
                user_input=f"message {i}",
                system_response=f"response {i}",
                detected_topic="test"
            )
        
        summary = manager.get_context_summary()
        
        assert summary["interaction_count"] == 3
        assert summary["current_topic"] == "test"
        assert summary["conversation_context_size"] == 3
        assert "session_duration_minutes" in summary
        assert isinstance(summary["session_duration_minutes"], float)
    
    def test_session_reset(self, temp_dir):
        """Test reset_session clears context but preserves learning."""
        manager = ContextManager(persistence_dir=temp_dir)
        
        # Add interactions and learning
        manager.update_conversation_context("test", "response", "topic1")
        manager.learn_from_interaction(engagement_level=0.8, topic="topic1")
        
        # Store original patterns
        original_patterns = manager.interaction_patterns.copy()
        
        # Reset session
        manager.reset_session()
        
        # Context should be cleared
        assert manager.interaction_count == 0
        assert manager.current_topic is None
        assert len(manager.conversation_context.get_all()) == 0
        
        # Learning should be preserved
        assert manager.interaction_patterns == original_patterns


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_context_handling(self):
        """Test handling of empty context lists."""
        manager = ContextManager()
        
        # Empty context should not cause errors
        shift, similarity = manager.detect_context_shift("test", [])
        assert not shift
        assert similarity == 1.0
        
        adapted = manager.get_adapted_context("test", max_items=10)
        assert isinstance(adapted, list)
    
    def test_large_context_window(self):
        """Test performance with large context windows."""
        window = ContextWindow(max_size=1000, decay_rate=0.1)
        
        # Add many items
        for i in range(1000):
            window.add({"item": i})
        
        # Should handle efficiently
        assert len(window.get_all()) == 1000
        
        relevant = window.get_relevant(threshold=0.5)
        assert isinstance(relevant, list)
    
    def test_concurrent_topic_transitions(self):
        """Test rapid topic changes."""
        manager = ContextManager()
        
        topics = ["topic_a", "topic_b", "topic_c", "topic_d", "topic_e"]
        
        for topic in topics:
            manager.update_conversation_context(
                user_input=f"Talk about {topic}",
                system_response="OK",
                detected_topic=topic
            )
        
        assert manager.topic_transition_count == 5
        assert manager.current_topic == "topic_e"
        assert manager.previous_topic == "topic_d"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
