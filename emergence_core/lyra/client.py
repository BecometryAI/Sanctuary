"""
High-level API for interacting with Lyra's cognitive core.

This module provides both asynchronous (LyraAPI) and synchronous (Lyra) interfaces
for conversational interaction with Lyra. The API abstracts the cognitive core and
conversation management, providing simple methods for chatting and managing state.

Usage (Async):
    api = LyraAPI()
    await api.start()
    turn = await api.chat("Hello, Lyra!")
    print(turn.system_response)
    await api.stop()

Usage (Sync):
    lyra = Lyra()
    lyra.start()
    response = lyra.chat("Hello, Lyra!")
    print(response)
    lyra.stop()
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from .cognitive_core.core import CognitiveCore
from .cognitive_core.conversation import ConversationManager, ConversationTurn

logger = logging.getLogger(__name__)


class LyraAPI:
    """
    High-level asynchronous API for interacting with Lyra.
    
    Provides a clean interface for conversational interaction with Lyra's
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
        Initialize the Lyra API.
        
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
        
        logger.info("✅ LyraAPI initialized")
    
    async def start(self) -> None:
        """
        Start the cognitive core.
        
        Must be called before using chat() or other interactive methods.
        Starts the recurrent cognitive loop in the background.
        """
        if not self._running:
            # Start cognitive core (runs in background task)
            asyncio.create_task(self.core.start())
            
            # Give it a moment to initialize
            await asyncio.sleep(0.1)
            
            self._running = True
            logger.info("🧠 LyraAPI started")
    
    async def stop(self) -> None:
        """
        Stop the cognitive core.
        
        Gracefully shuts down the cognitive loop and saves state.
        """
        if self._running:
            await self.core.stop()
            self._running = False
            logger.info("🧠 LyraAPI stopped")
    
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
            raise RuntimeError("LyraAPI not started. Call start() first.")
        
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


class Lyra:
    """
    Synchronous wrapper for LyraAPI.
    
    Provides a blocking, synchronous interface for applications that don't
    use asyncio. Internally manages an event loop to run the async API.
    
    This is useful for:
    - Simple scripts and notebooks
    - Applications not using asyncio
    - Quick testing and experimentation
    
    Methods:
        start(): Initialize and start Lyra
        stop(): Gracefully shut down Lyra
        chat(message): Send message and get response text
        get_history(n): Get conversation history as dicts
        reset(): Clear conversation state
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the synchronous Lyra wrapper.
        
        Creates a new event loop for managing async operations.
        Note: This wrapper creates its own event loop and should not be used
        in applications that already have an active event loop.
        
        Args:
            config: Optional configuration dict (same as LyraAPI)
        """
        self.api = LyraAPI(config)
        self.loop = asyncio.new_event_loop()
        # Note: We don't set this as the global event loop to avoid interference
        
        logger.info("✅ Lyra (synchronous) initialized")
    
    def start(self) -> None:
        """
        Start Lyra.
        
        Initializes the cognitive core and begins processing.
        """
        self.loop.run_until_complete(self.api.start())
    
    def stop(self) -> None:
        """
        Stop Lyra.
        
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
            Lyra's response as a string
        """
        turn = self.loop.run_until_complete(self.api.chat(message))
        return turn.system_response
    
    def get_history(self, n: int = 10) -> List[Dict]:
        """
        Get conversation history as dicts.
        
        Args:
            n: Maximum number of recent turns to return
            
        Returns:
            List of dicts with user input, Lyra response, timestamp, emotion
        """
        turns = self.api.get_conversation_history(n)
        return [
            {
                "user": t.user_input,
                "lyra": t.system_response,
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
