"""
High-level API for interacting with Sanctuary's cognitive core.

This module provides both asynchronous (SanctuaryAPI) and synchronous (Sanctuary) interfaces
for conversational interaction with Sanctuary. The API abstracts the cognitive core and
conversation management, providing simple methods for chatting and managing state.

Usage (Async):
    api = SanctuaryAPI()
    await api.start()
    turn = await api.chat("Hello!")
    print(turn.system_response)
    await api.stop()

Usage (Sync):
    sanctuary = Sanctuary()
    sanctuary.start()
    response = sanctuary.chat("Hello!")
    print(response)
    sanctuary.stop()
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from .cognitive_core import CognitiveCore
from .cognitive_core.conversation import ConversationManager, ConversationTurn

logger = logging.getLogger(__name__)


class SanctuaryAPI:
    """
    High-level asynchronous API for interacting with Sanctuary.

    Provides a clean interface for conversational interaction with Sanctuary's
    cognitive core. Handles lifecycle management and conversation state.

    The API integrates:
    - CognitiveCore: The cognitive processing engine
    - ConversationManager: Turn-taking and dialogue state management

    Methods:
        start(): Initialize and start the cognitive core
        stop(): Gracefully shut down the cognitive core
        chat(message): Send a message and get structured response
        get_conversation_history(n): Retrieve recent conversation turns
        get_metrics(): Get conversation and cognitive metrics
        reset_conversation(): Clear conversation state
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the Sanctuary API.

        Args:
            config: Optional configuration dict with keys:
                - cognitive_core: Config for CognitiveCore
                - conversation: Config for ConversationManager
        """
        config = config or {}
        
        # Initialize cognitive core
        self.core = CognitiveCore(config=config.get("cognitive_core", {}))
        
        # Initialize conversation manager
        conversation_config = config.get("conversation", {})
        self.conversation = ConversationManager(self.core, conversation_config)
        
        self._running = False
        
        logger.info("✅ SanctuaryAPI initialized")
    
    async def start(self) -> None:
        """
        Start the cognitive core.

        Must be called before using chat() or other interactive methods.
        Starts the recurrent cognitive loop in the background.
        """
        if not self._running:
            # Start cognitive core and wait for it to be ready
            await self.core.start()

            self._running = True
            logger.info("🧠 SanctuaryAPI started")
    
    async def stop(self) -> None:
        """
        Stop the cognitive core.
        
        Gracefully shuts down the cognitive loop and saves state.
        """
        if self._running:
            await self.core.stop()
            self._running = False
            logger.info("🧠 SanctuaryAPI stopped")
    
    async def chat(self, message: str) -> ConversationTurn:
        """
        Send message and get structured response.
        
        The primary method for conversational interaction. Processes the
        message through the cognitive core and conversation manager,
        returning a complete ConversationTurn with response and metadata.
        
        Args:
            message: User's text message
            
        Returns:
            ConversationTurn object containing response and metadata
            
        Raises:
            RuntimeError: If API not started yet
        """
        if not self._running:
            raise RuntimeError("SanctuaryAPI not started. Call start() first.")
        
        return await self.conversation.process_turn(message)
    
    def get_conversation_history(self, n: int = 10) -> List[ConversationTurn]:
        """
        Get recent conversation turns.
        
        Args:
            n: Maximum number of recent turns to return
            
        Returns:
            List of ConversationTurn objects
        """
        return self.conversation.get_conversation_history(n)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get conversation and cognitive metrics.
        
        Returns:
            Dict containing metrics from both conversation manager and
            cognitive core, including response times, turn counts, and
            system performance statistics.
        """
        conversation_metrics = self.conversation.get_metrics()
        cognitive_metrics = self.core.get_metrics()
        
        return {
            "conversation": conversation_metrics,
            "cognitive_core": cognitive_metrics
        }
    
    def reset_conversation(self) -> None:
        """
        Reset conversation state.
        
        Clears conversation history and dialogue state. Does not affect
        the cognitive core's memory or learning.
        """
        self.conversation.reset_conversation()


class Sanctuary:
    """
    Synchronous wrapper for SanctuaryAPI.

    Provides a blocking, synchronous interface for applications that don't
    use asyncio. Internally manages an event loop to run the async API.

    This is useful for:
    - Simple scripts and notebooks
    - Applications not using asyncio
    - Quick testing and experimentation

    Methods:
        start(): Initialize and start Sanctuary
        stop(): Gracefully shut down Sanctuary
        chat(message): Send message and get response text
        get_history(n): Get conversation history as dicts
        reset(): Clear conversation state
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the synchronous Sanctuary wrapper.

        Creates a new event loop for managing async operations.
        Note: This wrapper creates its own event loop and should not be used
        in applications that already have an active event loop.

        Args:
            config: Optional configuration dict (same as SanctuaryAPI)
        """
        self.api = SanctuaryAPI(config)
        self.loop = asyncio.new_event_loop()
        # Note: We don't set this as the global event loop to avoid interference

        logger.info("✅ Sanctuary (synchronous) initialized")

    def start(self) -> None:
        """
        Start Sanctuary.

        Initializes the cognitive core and begins processing.
        """
        self.loop.run_until_complete(self.api.start())

    def stop(self) -> None:
        """
        Stop Sanctuary.

        Gracefully shuts down the cognitive core.
        """
        self.loop.run_until_complete(self.api.stop())
        self.loop.close()

    def chat(self, message: str) -> str:
        """
        Send message and get response text.

        Args:
            message: User's text message

        Returns:
            Response as a string
        """
        turn = self.loop.run_until_complete(self.api.chat(message))
        return turn.system_response

    def get_history(self, n: int = 10) -> List[Dict]:
        """
        Get conversation history as dicts.

        Args:
            n: Maximum number of recent turns to return

        Returns:
            List of dicts with user input, response, timestamp, emotion
        """
        turns = self.api.get_conversation_history(n)
        return [
            {
                "user": t.user_input,
                "response": t.system_response,
                "timestamp": t.timestamp.isoformat(),
                "emotion": t.emotional_state
            }
            for t in turns
        ]
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get conversation and cognitive metrics.
        
        Returns:
            Dict containing system metrics
        """
        return self.api.get_metrics()
    
    def reset(self) -> None:
        """
        Reset conversation state.
        
        Clears conversation history and dialogue state.
        """
        self.api.reset_conversation()
