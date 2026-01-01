"""
Unified Cognitive Core: Bridges new cognitive architecture with legacy specialists.

This module provides seamless integration between:
- CognitiveCore (Phases 1-4): Non-linguistic recurrent loop
- AdaptiveRouter: Legacy specialist system for deep reasoning

Architecture:
- CognitiveCore runs continuously (~10 Hz)
- When SPEAK action generated → triggers specialist system
- Specialist output → feeds back as percept to cognitive core
- Both systems share: memory (RAG), emotional state, conversation context
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .cognitive_core import (
    CognitiveCore,
    GlobalWorkspace,
    ConversationManager,
    Action,
    ActionType,
    Percept
)
from .adaptive_router import AdaptiveRouter

logger = logging.getLogger(__name__)


class UnifiedCognitiveCore:
    """
    Unified orchestrator integrating cognitive core with specialist system.
    
    Responsibilities:
    - Initialize both cognitive core and specialist router
    - Run continuous cognitive loop in background
    - Intercept SPEAK actions and route to specialists
    - Feed specialist outputs back to cognitive core
    - Maintain shared state (memory, emotion, context)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize unified cognitive system.
        
        Args:
            config: Configuration with keys:
                - cognitive_core: Config for CognitiveCore
                - specialist_router: Config for AdaptiveRouter
                - integration: Integration-specific settings
        """
        self.config = config or {}
        
        # Core components
        self.workspace: Optional[GlobalWorkspace] = None
        self.cognitive_core: Optional[CognitiveCore] = None
        self.conversation_manager: Optional[ConversationManager] = None
        self.specialist_router: Optional[AdaptiveRouter] = None
        
        # Integration state
        self.running = False
        self.pending_specialist_calls: asyncio.Queue = asyncio.Queue()
        
        # Configuration
        self.specialist_threshold = self.config.get("integration", {}).get(
            "specialist_threshold", 0.7
        )
        self.sync_interval = self.config.get("integration", {}).get(
            "sync_interval", 1.0
        )
        
        logger.info("🧠 UnifiedCognitiveCore initialized")
    
    async def initialize(self, base_dir: str, chroma_dir: str, model_dir: str) -> None:
        """
        Initialize all system components.
        
        Args:
            base_dir: Base directory for data
            chroma_dir: ChromaDB directory
            model_dir: Model cache directory
        """
        logger.info("Initializing unified cognitive system...")
        
        # 1. Initialize workspace (shared by both systems)
        self.workspace = GlobalWorkspace()
        
        # 2. Initialize cognitive core
        cognitive_config = self.config.get("cognitive_core", {})
        self.cognitive_core = CognitiveCore(
            workspace=self.workspace,
            config=cognitive_config
        )
        
        # 3. Initialize conversation manager
        self.conversation_manager = ConversationManager(
            cognitive_core=self.cognitive_core,
            config=self.config.get("conversation", {})
        )
        
        # 4. Initialize specialist router
        self.specialist_router = await AdaptiveRouter.create(
            base_dir=base_dir,
            chroma_dir=chroma_dir,
            model_dir=model_dir
        )
        
        # 5. Start cognitive core background loop
        await self.cognitive_core.start()
        
        # 6. Start specialist call handler
        asyncio.create_task(self._specialist_call_handler())
        
        self.running = True
        logger.info("✅ Unified cognitive system ready")
    
    async def process_user_input(self, user_text: str) -> str:
        """
        Process user input through unified system.
        
        Flow:
        1. Parse input via cognitive core (LanguageInputParser)
        2. Cognitive loop processes (attention, workspace, action selection)
        3. If SPEAK action generated → route to specialist
        4. Specialist output → feed back to cognitive core
        5. Return final response
        
        Args:
            user_text: User's input text
            
        Returns:
            Lyra's response text
        """
        # Use conversation manager (handles cognitive core integration)
        turn = await self.conversation_manager.chat(user_text)
        
        # Check if SPEAK action was generated and requires specialist processing
        if self._requires_specialist_processing(turn):
            # Route to specialist system
            specialist_response = await self._call_specialist(user_text, turn)
            
            # Feed back as percept
            self._feed_specialist_output(specialist_response)
            
            return specialist_response
        
        return turn.system_response
    
    def _requires_specialist_processing(self, turn) -> bool:
        """
        Determine if turn requires specialist processing.
        
        Criteria:
        - SPEAK action with high priority
        - Complex query requiring deep reasoning
        - Tool use needed (Pragmatist)
        - Creative generation requested (Artist)
        - Ethical reasoning needed (Philosopher)
        """
        # Check workspace for SPEAK actions
        snapshot = self.cognitive_core.query_state()
        
        for goal in snapshot.goals:
            if goal.type.value == "RESPOND_TO_USER" and goal.priority > self.specialist_threshold:
                return True
        
        # Check if any recent actions are SPEAK with high priority
        if hasattr(self.cognitive_core, 'action_subsystem'):
            recent_actions = list(self.cognitive_core.action_subsystem.action_history)
            for action in recent_actions[-5:]:  # Check last 5 actions
                if isinstance(action, Action):
                    if action.type == ActionType.SPEAK and action.priority > self.specialist_threshold:
                        return True
        
        return False
    
    async def _call_specialist(self, user_text: str, turn) -> str:
        """
        Call specialist system for deep processing.
        
        Args:
            user_text: Original user input
            turn: ConversationTurn from cognitive core
            
        Returns:
            Specialist system response
        """
        # Build context from cognitive state
        snapshot = self.cognitive_core.query_state()
        
        # Build emotional state dict
        emotions = {}
        if snapshot.emotions:
            emotions = {
                "valence": getattr(snapshot.emotions, 'valence', 0.0),
                "arousal": getattr(snapshot.emotions, 'arousal', 0.0),
                "dominance": getattr(snapshot.emotions, 'dominance', 0.0)
            }
        
        context = {
            "emotional_state": emotions,
            "active_goals": [g.description for g in snapshot.goals],
            "recent_memories": [m.content for m in snapshot.memories[:5]],
            "cognitive_metadata": turn.metadata
        }
        
        # Route through specialist system
        response = await self.specialist_router.process_message(
            message=user_text,
            context=context
        )
        
        return response
    
    def _feed_specialist_output(self, response: str) -> None:
        """
        Feed specialist output back to cognitive core as percept.
        
        Args:
            response: Specialist system response
        """
        percept = Percept(
            modality="text",
            raw=response,
            complexity=5,
            metadata={
                "source": "specialist_system",
                "processed": True
            }
        )
        
        self.cognitive_core.inject_input(percept)
    
    async def _specialist_call_handler(self) -> None:
        """Background task handling specialist calls asynchronously."""
        while self.running:
            try:
                # Process queued specialist calls
                await asyncio.sleep(0.1)  # Small delay
                # Future: implement async specialist processing queue
            except Exception as e:
                logger.error(f"Error in specialist handler: {e}")
    
    async def stop(self) -> None:
        """Gracefully shut down unified system."""
        logger.info("Shutting down unified cognitive system...")
        self.running = False
        
        if self.cognitive_core:
            await self.cognitive_core.stop()
        
        logger.info("✅ Unified system stopped")


class SharedMemoryBridge:
    """
    Bridges memory between cognitive core and specialist system.
    
    Both systems share:
    - ChromaDB vector store
    - Journal entries
    - Episodic memories
    - Protocols and identity
    """
    
    def __init__(self, chroma_client, memory_integration):
        """
        Initialize memory bridge.
        
        Args:
            chroma_client: ChromaDB client instance
            memory_integration: MemoryIntegration from cognitive core
        """
        self.chroma = chroma_client
        self.memory_integration = memory_integration
    
    def sync_memories(self) -> None:
        """Sync memories between systems."""
        # Ensure cognitive core memories → ChromaDB
        # Ensure ChromaDB results → cognitive core workspace
        # This is a placeholder for future implementation
        pass


class EmotionalStateBridge:
    """
    Synchronizes emotional state between cognitive core and specialists.
    
    - AffectSubsystem (cognitive core) maintains VAD model
    - Specialists use emotional context for generation
    - Bidirectional sync: core → specialists → core
    """
    
    def sync_to_specialists(self, affect_state) -> Dict:
        """
        Convert cognitive core emotion to specialist format.
        
        Args:
            affect_state: AffectSubsystem state
            
        Returns:
            Dict with valence, arousal, dominance
        """
        return {
            "valence": getattr(affect_state, 'valence', 0.0),
            "arousal": getattr(affect_state, 'arousal', 0.0),
            "dominance": getattr(affect_state, 'dominance', 0.0)
        }
    
    def sync_from_specialists(self, specialist_emotion: Dict) -> None:
        """
        Update cognitive core emotion from specialist output.
        
        Args:
            specialist_emotion: Emotion dict from specialist
        """
        # This is a placeholder for future implementation
        pass
