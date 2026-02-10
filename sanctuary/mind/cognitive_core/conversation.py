"""
ConversationManager: Orchestrates dialogue flow between user and cognitive core.

This module implements the ConversationManager class, which manages turn-taking,
dialogue state tracking, conversation history, and provides a cohesive conversational
interface for multi-turn interactions with Sanctuary's cognitive core.

The ConversationManager is responsible for:
- Turn-taking coordination
- Dialogue state tracking (topics, history, timing)
- Conversation history management with multi-turn coherence
- Timeout and error handling for robust conversation
- Conversation metrics and analytics
"""

from __future__ import annotations

import logging
import asyncio
from typing import Dict, List, Any, Optional, Deque
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

from .core import CognitiveCore

logger = logging.getLogger(__name__)

# Constants
DEFAULT_RESPONSE_TIMEOUT_ERROR = "I apologize, I'm having trouble formulating a response right now."
DEFAULT_ERROR_MESSAGE = "I encountered an error processing that. Could you rephrase?"
DEFAULT_STOPWORDS = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "is", "it", "that", "this", "with"}


@dataclass
class ConversationTurn:
    """
    Represents a single conversation turn.
    
    A turn contains the user's input, Sanctuary's response, timing information,
    and emotional state during the interaction. This provides a complete
    record of one conversational exchange.
    
    Attributes:
        user_input: The text input from the user
        system_response: Sanctuary's text response
        timestamp: When the turn occurred
        response_time: How long it took to generate the response (seconds)
        emotional_state: Sanctuary's emotional state during the turn
        metadata: Additional turn-specific information
    """
    user_input: str
    system_response: str
    timestamp: datetime
    response_time: float
    emotional_state: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """
    Orchestrates conversational interactions with the cognitive core.
    
    The ConversationManager provides a high-level conversational interface,
    handling turn-taking, dialogue state management, and multi-turn coherence.
    It bridges between raw user input and Sanctuary's cognitive processing.
    
    Key Features:
    - Turn-based conversation flow
    - Automatic context tracking across turns
    - Conversation history maintenance
    - Graceful timeout and error handling
    - Performance metrics tracking
    
    The manager maintains dialogue state including recent topics, conversation
    history, and turn count. This state is passed to the cognitive core for
    context-aware processing.
    """
    
    def __init__(self, cognitive_core: CognitiveCore, config: Optional[Dict] = None):
        """
        Initialize the ConversationManager.
        
        Args:
            cognitive_core: The CognitiveCore instance to orchestrate
            config: Optional configuration dict with keys:
                - response_timeout: Max seconds to wait for response (default: 10.0)
                - max_cycles_per_turn: Max cognitive cycles per turn (default: 20)
                - max_history_size: Max turns to keep in history (default: 100)
        """
        self.core = cognitive_core
        self.config = config or {}
        
        # Dialogue state
        max_history = self.config.get("max_history_size", 100)
        self.conversation_history: Deque[ConversationTurn] = deque(maxlen=max_history)
        self.current_topics: List[str] = []
        self.turn_count = 0
        
        # Configuration
        self.response_timeout = self.config.get("response_timeout", 10.0)
        self.max_cycles_per_turn = self.config.get("max_cycles_per_turn", 20)
        
        # Metrics
        self.metrics = {
            "total_turns": 0,
            "avg_response_time": 0.0,
            "timeouts": 0,
            "errors": 0
        }
        
        logger.info("✅ ConversationManager initialized")
    
    async def process_turn(self, user_input: str) -> ConversationTurn:
        """
        Process a conversational turn.
        
        This is the main method for handling user input. It orchestrates the
        complete turn flow: parse → cognitive cycle → generate → response.
        The method handles errors gracefully and tracks metrics.
        
        Args:
            user_input: The user's text input
            
        Returns:
            ConversationTurn object containing the complete interaction
        """
        start_time = datetime.now()
        self.turn_count += 1
        
        try:
            # Build context from dialogue state
            context = {
                "turn_count": self.turn_count,
                "recent_topics": self.current_topics[-5:],
                "conversation_history": [
                    {
                        "user": turn.user_input,
                        "sanctuary": turn.system_response,
                        "timestamp": turn.timestamp.isoformat()
                    }
                    for turn in list(self.conversation_history)[-3:]
                ]
            }
            
            # Send input to cognitive core
            await self.core.process_language_input(user_input, context)
            
            # Wait for response
            response = await self._wait_for_response(self.response_timeout)
            
            if response is None:
                logger.warning("⏱️ Response timeout")
                self.metrics["timeouts"] += 1
                response = DEFAULT_RESPONSE_TIMEOUT_ERROR
            
            # Get emotional state
            snapshot = self.core.workspace.broadcast()
            emotional_state = snapshot.emotions
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Create turn record
            turn = ConversationTurn(
                user_input=user_input,
                system_response=response,
                timestamp=start_time,
                response_time=response_time,
                emotional_state=emotional_state.copy(),
                metadata={
                    "turn_number": self.turn_count
                }
            )
            
            # Update state
            self._update_dialogue_state(user_input, response)
            
            # Add to history
            self.conversation_history.append(turn)
            
            # Update metrics
            self._update_metrics(turn)
            
            logger.info(f"💬 Turn {self.turn_count} completed in {response_time:.2f}s")
            
            return turn
            
        except Exception as e:
            logger.error(f"❌ Error in conversation turn: {e}", exc_info=True)
            self.metrics["errors"] += 1
            
            # Return error turn
            return ConversationTurn(
                user_input=user_input,
                system_response=DEFAULT_ERROR_MESSAGE,
                timestamp=start_time,
                response_time=0.0,
                emotional_state={},
                metadata={"error": str(e)}
            )
    
    async def _wait_for_response(self, timeout: float) -> Optional[str]:
        """
        Wait for cognitive core to produce response.
        
        Polls the output queue with a timeout to retrieve Sanctuary's response.
        Only returns SPEAK actions as valid responses.
        
        Args:
            timeout: Maximum seconds to wait for response
            
        Returns:
            Response text string, or None if timeout or no valid response
        """
        try:
            # Poll output queue with timeout
            output = await asyncio.wait_for(
                self.core.output_queue.get(),
                timeout=timeout
            )
            
            if output and output.get("type") == "SPEAK":
                return output.get("text")
            
            return None
            
        except asyncio.TimeoutError:
            logger.warning("Response timeout exceeded")
            return None
        except Exception as e:
            logger.error(f"Error waiting for response: {e}")
            return None
    
    def _update_dialogue_state(self, user_input: str, response: str) -> None:
        """
        Update conversation state.
        
        Extracts topics from user input and updates the tracked topic list.
        This state is used to provide context for future turns.
        
        Args:
            user_input: User's text input
            response: Sanctuary's response text
        """
        # Extract topics from user input (simple keyword extraction)
        topics = self._extract_topics(user_input)
        self.current_topics.extend(topics)
        self.current_topics = self.current_topics[-10:]  # Keep recent
    
    def _extract_topics(self, text: str) -> List[str]:
        """
        Simple topic extraction from text.
        
        Extracts content words (longer words that aren't stopwords) as
        candidate topics. This provides basic topic tracking without
        requiring complex NLP.
        
        Args:
            text: Input text to extract topics from
            
        Returns:
            List of up to 3 extracted topic words
        """
        # Get stopwords from config or use default
        stopwords = self.config.get("stopwords", DEFAULT_STOPWORDS)
        words = text.lower().split()
        topics = [w for w in words if len(w) > 4 and w not in stopwords]
        return topics[:3]  # Top 3 content words
    
    def _update_metrics(self, turn: ConversationTurn) -> None:
        """
        Update conversation metrics.
        
        Tracks average response time and other statistics using a running
        average calculation.
        
        Args:
            turn: The completed conversation turn
        """
        self.metrics["total_turns"] += 1
        
        # Update average response time
        n = self.metrics["total_turns"]
        current_avg = self.metrics["avg_response_time"]
        self.metrics["avg_response_time"] = (
            (current_avg * (n - 1) + turn.response_time) / n
        )
    
    def get_conversation_history(self, n: int = 10) -> List[ConversationTurn]:
        """
        Get recent conversation turns.
        
        Args:
            n: Maximum number of recent turns to return
            
        Returns:
            List of most recent ConversationTurn objects (up to n)
        """
        return list(self.conversation_history)[-n:]
    
    async def listen_for_autonomous(self, timeout: Optional[float] = None):
        """
        Generator that yields autonomous messages from Sanctuary.
        
        Use this to listen for unprompted speech between user turns. This allows
        Sanctuary to proactively share introspective insights, emotional states, or
        other significant observations that warrant expression.
        
        Args:
            timeout: Optional timeout in seconds. If None, waits indefinitely.
                    If provided, returns after timeout without yielding anything.
        
        Yields:
            Dict containing autonomous message information:
                - text: The autonomous message text
                - trigger: What triggered the autonomous speech
                - emotion: Emotional state during speech
                - timestamp: When the autonomous speech occurred
                
        Example:
            async for message in conversation.listen_for_autonomous(timeout=5.0):
                print(f"Sanctuary says: {message['text']}")
                print(f"Trigger: {message['trigger']}")
        """
        try:
            output = await asyncio.wait_for(
                self.core.output_queue.get(),
                timeout=timeout
            ) if timeout else await self.core.output_queue.get()
            
            if output and output.get("type") == "SPEAK_AUTONOMOUS":
                yield {
                    "text": output.get("text"),
                    "trigger": output.get("trigger"),
                    "emotion": output.get("emotion"),
                    "timestamp": output.get("timestamp")
                }
        except asyncio.TimeoutError:
            return
        except Exception as e:
            logger.error(f"Error listening for autonomous speech: {e}")
            return
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dict containing conversation metrics including:
                - total_turns: Total conversation turns
                - avg_response_time: Average response time in seconds
                - timeouts: Number of timeout occurrences
                - errors: Number of error occurrences
                - turn_count: Current turn count
                - topics_tracked: Number of topics being tracked
                - history_size: Number of turns in history
        """
        return {
            **self.metrics,
            "turn_count": self.turn_count,
            "topics_tracked": len(self.current_topics),
            "history_size": len(self.conversation_history)
        }
    
    def reset_conversation(self) -> None:
        """
        Clear conversation state.
        
        Resets all dialogue state including history, topics, and turn count.
        Metrics are preserved for analytics purposes.
        """
        self.conversation_history.clear()
        self.current_topics.clear()
        self.turn_count = 0
        logger.info("🔄 Conversation reset")
