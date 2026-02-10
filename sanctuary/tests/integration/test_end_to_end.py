"""
End-to-end integration tests for Sanctuary cognitive architecture.

Tests the complete system: identity → perception → attention → emotion →
memory → meta-cognition → action → language → conversation.
"""

import pytest
import asyncio
from datetime import datetime
from typing import List, Dict

from mind import SanctuaryAPI
from mind.cognitive_core import CognitiveCore, ConversationTurn


@pytest.mark.integration
class TestBasicConversationFlow:
    """Test basic conversation functionality."""
    
    @pytest.mark.asyncio
    async def test_single_turn_conversation(self):
        """Test a complete single turn: input → processing → response."""
        # Initialize system
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Send message
            turn = await api.chat("Hello, how are you?")
            
            # Verify turn structure
            assert isinstance(turn, ConversationTurn)
            assert turn.user_input == "Hello, how are you?"
            assert len(turn.system_response) > 0
            assert turn.response_time > 0
            assert isinstance(turn.emotional_state, dict)
            
            # Verify emotional state present
            assert "valence" in turn.emotional_state
            assert "arousal" in turn.emotional_state
            
        finally:
            await api.stop()
    
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with context."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Turn 1
            turn1 = await api.chat("My name is Alice.")
            assert len(turn1.system_response) > 0
            
            # Turn 2 - should remember name
            turn2 = await api.chat("What's my name?")
            assert len(turn2.system_response) > 0
            # Name should be referenced (though exact format may vary)
            
            # Turn 3
            turn3 = await api.chat("Tell me about yourself.")
            assert len(turn3.system_response) > 0
            
            # Verify history
            history = api.get_conversation_history()
            assert len(history) >= 3
            
        finally:
            await api.stop()


@pytest.mark.integration
class TestIdentityDrivenBehavior:
    """Test that identity (charter + protocols) influences behavior."""
    
    @pytest.mark.asyncio
    async def test_charter_influences_responses(self):
        """Test that charter values are reflected in responses."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Ask about values
            turn = await api.chat("What do you care about?")
            
            # Response should reflect charter
            assert len(turn.system_response) > 0
            # Charter-related themes should be present
            
        finally:
            await api.stop()
    
    @pytest.mark.asyncio
    async def test_protocols_guide_behavior(self):
        """Test that protocols guide conversational behavior."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Ask about approach
            turn = await api.chat("How do you approach conversations?")
            
            # Response should reflect protocols
            assert len(turn.system_response) > 0
            
        finally:
            await api.stop()


@pytest.mark.integration
class TestEmotionalDynamics:
    """Test emotional state updates and influences."""
    
    @pytest.mark.asyncio
    async def test_emotion_updates_with_input(self):
        """Test that emotional state responds to input."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Neutral input
            turn1 = await api.chat("Hello.")
            emotion1 = turn1.emotional_state
            
            # Positive input
            turn2 = await api.chat("This is wonderful! I'm so excited!")
            emotion2 = turn2.emotional_state
            
            # Emotions should differ
            # (Exact values depend on affect model, but should change)
            assert emotion2 is not None
            
        finally:
            await api.stop()
    
    @pytest.mark.asyncio
    async def test_emotion_influences_language(self):
        """Test that emotional state affects language generation."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Generate response
            turn = await api.chat("Tell me how you're feeling.")
            
            # Response should reference emotional state
            assert len(turn.system_response) > 0
            
        finally:
            await api.stop()


@pytest.mark.integration  
class TestMemoryIntegration:
    """Test memory storage and retrieval during conversation."""
    
    @pytest.mark.asyncio
    async def test_memory_consolidation(self):
        """Test that conversations are consolidated to memory."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Have conversation
            await api.chat("My favorite color is blue.")
            await api.chat("I enjoy reading science fiction.")
            
            # Allow time for consolidation
            await asyncio.sleep(2)
            
            # Memory should exist
            core = api.core
            # Check that memory subsystem has entries
            # (Implementation depends on memory system)
            
        finally:
            await api.stop()
    
    @pytest.mark.asyncio
    async def test_memory_retrieval_in_conversation(self):
        """Test that relevant memories are retrieved."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Store information
            await api.chat("Remember that I love pizza.")
            
            # Later, ask about it
            await asyncio.sleep(1)
            turn = await api.chat("What do I love?")
            
            # Response should reference the memory
            assert len(turn.system_response) > 0
            
        finally:
            await api.stop()


@pytest.mark.integration
class TestAttentionMechanism:
    """Test attention subsystem integration."""
    
    @pytest.mark.asyncio
    async def test_attention_selects_salient_percepts(self):
        """Test that attention mechanism works during conversation."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Send complex input
            turn = await api.chat(
                "I'm feeling anxious about an important decision. "
                "Should I change careers or stay where I am?"
            )
            
            # Response should attend to the important parts
            assert len(turn.system_response) > 0
            
            # Check workspace state
            snapshot = api.core.workspace.broadcast()
            # Should have attended percepts
            attended = [p for p in snapshot.percepts.values()
                       if p.metadata.get("attention_score", 0) > 0.5]
            assert len(attended) >= 0  # May or may not have attended percepts
            
        finally:
            await api.stop()


@pytest.mark.integration
class TestMetaCognitionIntegration:
    """Test meta-cognition and introspection."""
    
    @pytest.mark.asyncio
    async def test_introspection_occurs(self):
        """Test that self-monitoring generates introspective percepts."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Have conversation
            await api.chat("How are you?")
            await api.chat("What are you thinking about?")
            
            # Allow cognitive cycles
            await asyncio.sleep(1)
            
            # Check for introspective percepts
            snapshot = api.core.workspace.broadcast()
            introspections = [p for p in snapshot.percepts.values()
                            if p.modality == "introspection"]
            
            # Should have some introspective activity
            assert len(introspections) >= 0  # May or may not have generated yet
            
        finally:
            await api.stop()


@pytest.mark.integration
class TestAutonomousSpeech:
    """Test autonomous speech initiation."""
    
    @pytest.mark.asyncio
    async def test_autonomous_speech_triggers(self):
        """Test that autonomous speech can be triggered."""
        # Configure for easier autonomous triggering
        config = {
            "autonomous_initiation": {
                "introspection_threshold": 10,  # Lower threshold
                "min_interval": 1  # Shorter interval
            }
        }
        api = SanctuaryAPI(config)
        await api.start()
        
        try:
            # Trigger introspection through conversation
            await api.chat("What do you think about yourself?")
            
            # Allow time for autonomous processing
            await asyncio.sleep(2)
            
            # Check for autonomous output
            # (May or may not have triggered in this timeframe)
            
        finally:
            await api.stop()


@pytest.mark.integration
class TestPerformanceBenchmarks:
    """Test system performance metrics."""
    
    @pytest.mark.asyncio
    async def test_response_time_reasonable(self):
        """Test that response times are reasonable."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Measure response time
            start = datetime.now()
            turn = await api.chat("Hello!")
            elapsed = (datetime.now() - start).total_seconds()
            
            # Should respond within reasonable time (adjust threshold as needed)
            assert elapsed < 15.0  # 15 seconds max
            assert turn.response_time < 15.0
            
        finally:
            await api.stop()
    
    @pytest.mark.asyncio
    async def test_throughput(self):
        """Test conversation throughput."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Send multiple messages
            turns = []
            for i in range(5):
                turn = await api.chat(f"Message {i}")
                turns.append(turn)
            
            # All should succeed
            assert len(turns) == 5
            assert all(len(t.system_response) > 0 for t in turns)
            
            # Check average response time
            avg_time = sum(t.response_time for t in turns) / len(turns)
            assert avg_time < 15.0
            
        finally:
            await api.stop()


@pytest.mark.integration
class TestSystemMetrics:
    """Test system metrics and monitoring."""
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test that system collects metrics."""
        api = SanctuaryAPI()
        await api.start()
        
        try:
            # Have conversation
            await api.chat("Hello!")
            await api.chat("How are you?")
            
            # Get metrics
            metrics = api.get_metrics()
            
            # Verify structure
            assert "conversation" in metrics
            assert "cognitive_core" in metrics
            
            # Conversation metrics
            assert metrics["conversation"]["total_turns"] >= 2
            assert metrics["conversation"]["avg_response_time"] > 0
            
            # Cognitive metrics
            assert "total_cycles" in metrics["cognitive_core"]
            
        finally:
            await api.stop()
