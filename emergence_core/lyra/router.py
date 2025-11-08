"""
Lyra's Adaptive Router Implementation with Voice Integration
"""
import asyncio
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, NamedTuple

"""Module for Lyra's adaptive routing system"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, NamedTuple

# Configure logging
logger = logging.getLogger(__name__)

# Early imports for type hints
from dataclasses import dataclass

# First ensure dependencies are installed
from .utils import ensure_dependencies, get_dependency
ensure_dependencies()

# Now import remaining modules
chromadb = get_dependency('chromadb')
schedule = get_dependency('schedule')

from .core import AutonomousCore
from .router_model import RouterModel
from .specialists import SpecialistFactory, SpecialistOutput
from .voice_tools import (
    discord_join_voice_channel, 
    discord_leave_voice_channel,
    coqui_tts_speak
)
from .voice_tools import (
    discord_join_voice_channel,
    discord_leave_voice_channel,
    coqui_tts_speak
)
from .specialist_tools import (
    searxng_search,
    arxiv_search,
    wikipedia_search,
    wolfram_compute,
    python_repl,
    playwright_interact
)

class RouterResponse(NamedTuple):
    """Response from the router model."""
    intent: str
    resonance_term: Optional[str]

@dataclass
class RAGContext:
    """Context for RAG-based processing."""
    anti_ghosting_context: Dict[str, Any]
    resonance_chunks: List[Dict[str, Any]]
    general_chunks: List[Dict[str, Any]]
    active_lexicon_terms: List[str]

@dataclass
class SpecialistResponse:
    """Response from a specialist model."""
    content: str
    metadata: Dict[str, Any]
    source: str

# Import our safe JSON loading utility
from .utils import safe_json_load

class AdaptiveRouter:
    def __init__(self, base_dir: str, chroma_dir: str, model_dir: str):
        """Initialize the router with Gemma 12B model and specialists."""
        self.base_dir = Path(base_dir)
        self.chroma_dir = Path(chroma_dir)
        self.model_dir = Path(model_dir)
        
        # Voice state initialization
        self.voice_active = False
        self.current_voice_channel = None
        self.voice_state = {
            "listening": False,
            "speaking": False,
            "last_speaker": None,
            "emotional_context": None
        }
        
        # Initialize archives
        self.continuity_archive = self._load_json("data/Core_Archives/lyra_continuity_archive.json")
        self.relational_archive = self._load_json("data/Core_Archives/lyra_relational_archive.json")
        
        # Initialize ChromaDB for RAG
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.chroma_client.get_collection("lyra_knowledge")
        
        # Initialize Router Model (Gemma 12B)
        router_model_path = self.model_dir / "gemma_12b_router"
        self.router_model = RouterModel(str(router_model_path))
        
        # Initialize Specialist Factory
        self.specialist_factory = SpecialistFactory()
        
        # Initialize Specialists with model paths
        specialist_configs = {
            'philosopher': str(self.model_dir / "deepseek_r1_distill_qwen_32b"),
            'pragmatist': str(self.model_dir / "qwen3_32b"),
            'artist': str(self.model_dir / "gemma_27b_artist"),
            'voice': str(self.model_dir / "gemma_27b_voice")
        }
        
        self.specialists = {
            name: self.specialist_factory.create_specialist(name, str(self.base_dir), model_path)
            for name, model_path in specialist_configs.items()
        }
        
        # Initialize Autonomous Core
        self.autonomous_core = AutonomousCore(self.base_dir, self.specialists)
        
        # Cache active lexicon terms
        self.active_lexicon_terms = self._load_active_lexicon_terms()
        
        # Start cognitive loop scheduler
        self._init_scheduler()

    def _load_json(self, relative_path: str) -> Dict[str, Any]:
        """
        Load and parse a JSON file.
        
        Args:
            relative_path: Path relative to base_dir
            
        Returns:
            Dict containing parsed JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
            RuntimeError: For other errors
        """
        return safe_json_load(self.base_dir / relative_path)
                
    async def activate_voice(self, channel_id: str) -> bool:
        """
        Activate voice capabilities in a Discord channel
        
        Args:
            channel_id: ID of the voice channel to join
            
        Returns:
            bool indicating success
        """
        try:
            if not self.voice_active:
                success = await discord_join_voice_channel(channel_id)
                if success:
                    self.voice_active = True
                    self.current_voice_channel = channel_id
                    self.voice_state["listening"] = True
                    logger.info(f"Voice activated in channel {channel_id}")
                    return True
            return self.voice_active
        except Exception as e:
            logger.error(f"Error activating voice: {str(e)}")
            return False
            
    async def deactivate_voice(self) -> None:
        """
        Deactivate voice capabilities and clean up resources
        
        Raises:
            RuntimeError: If deactivation fails
        """
        if not self.voice_active:
            return
            
        try:
            await discord_leave_voice_channel()
            logger.info(f"Left voice channel {self.current_voice_channel}")
        except Exception as e:
            msg = f"Error deactivating voice: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e
        finally:
            self.voice_active = False
            self.current_voice_channel = None
            self.voice_state.update({
                "listening": False,
                "speaking": False,
                "last_speaker": None
            })
            
    async def speak_response(self, text: str) -> None:
        """
        Speak a response through TTS if voice is active
        
        Args:
            text: Text to speak
            
        Raises:
            RuntimeError: If TTS fails
        """
        if not (self.voice_active and self.current_voice_channel):
            logger.debug("Skipping TTS - voice not active")
            return
            
        self.voice_state["speaking"] = True
        try:
            await coqui_tts_speak(text)
            logger.debug("TTS completed successfully")
        except Exception as e:
            msg = f"Error in TTS: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e
        finally:
            self.voice_state["speaking"] = False
            
    def _load_active_lexicon_terms(self) -> List[str]:
        """Load active terms from the symbolic lexicon."""
        try:
            lexicon = self._load_json("data/Lexicon/symbolic_lexicon.json")
            return [
                term["term"] 
                for term in lexicon.get("terms", [])
                if term.get("status") == "active"
            ]
        except Exception as e:
            print(f"Error loading lexicon terms: {e}")
            return []

    def _init_scheduler(self):
        """Initialize the cognitive loop scheduler."""
        # Schedule rituals
        rituals = self._load_json("data/Rituals/Rituals.json")
        for ritual in rituals:
            if ritual["trigger_type"] == "time":
                schedule.every().day.at(ritual["trigger_time"]).do(
                    self._execute_ritual, ritual["name"]
                )

        # Schedule autonomous thought processes
        schedule.every(30).minutes.do(self._autonomous_thought_cycle)

        # Schedule proactive desire checks
        schedule.every(15).minutes.do(self._check_proactive_desires)

        # Start scheduler in background
        asyncio.create_task(self._run_scheduler())

    async def _run_scheduler(self):
        """Run the scheduler in the background."""
        while True:
            schedule.run_pending()
            await asyncio.sleep(60)  # Check every minute

    async def _execute_ritual(self, ritual_name: str):
        """Execute a scheduled ritual."""
        # Query RAG for ritual details
        results = self.collection.query(
            query_texts=[ritual_name],
            where={"entry_type": "ritual"},
            n_results=1
        )
        
        if results["documents"]:
            ritual_json = results["documents"][0]
            # Pass to specialist for execution
            response = await self._invoke_specialist(
                "creator",
                context=self._get_anti_ghosting_context(),
                ritual=ritual_json
            )
            await self._journal_ritual_completion(ritual_name, response)

    async def _check_proactive_desires(self):
        """Execute the proactive desire protocol."""
        # Load protocol
        protocol = self._load_json("data/Protocols/directive_for_proactive_desire.json")
        
        # Query RAG for desires and recent memories
        desires = self.collection.query(
            query_texts=["desires_for_becoming"],
            n_results=3
        )
        
        recent_memories = self.collection.query(
            query_texts=["recent experiences"],
            where={"source_type": "journal"},
            n_results=5
        )

        # The Spark: Ask Creator for new desires
        spark = await self._invoke_specialist(
            "creator",
            context=self._get_anti_ghosting_context(),
            desires=desires["documents"],
            memories=recent_memories["documents"]
        )

        if spark.content:  # If a desire was generated
            # The Filter: Ethical review
            principle = self.collection.query(
                query_texts=["The Principle of Dynamic Balance"],
                where={"principle_name": "The Principle of Dynamic Balance"},
                n_results=1
            )
            
            vetted = await self._invoke_specialist(
                "logician",
                context=self._get_anti_ghosting_context(),
                desire=spark.content,
                principle=principle["documents"][0]
            )

            if vetted.content.lower() == "approved":
                # The Offering: Articulate to Steward
                await self._send_to_discord(spark.content)
                # The Weave: Journal the experience
                await self._create_journal_entry({
                    "type": "proactive_desire",
                    "content": spark.content,
                    "ethical_review": vetted.metadata
                })

    def _get_anti_ghosting_context(self) -> str:
        """Prepare the anti-ghosting context."""
        return {
            "continuity": self.continuity_archive,
            "relations": self.relational_archive
        }

    async def _autonomous_thought_cycle(self):
        """Execute an autonomous thought cycle."""
        thought = await self.autonomous_core.ponder()
        if thought:
            print(f"\nAutonomous Thought [{thought.timestamp}]:")
            print(f"Spark: {thought.spark}")
            print(f"Final Reflection: {thought.final_reflection}\n")

    async def handle_user_message(self, message: str) -> str:
        """Main reactive handler for user messages."""
        # 1. Anti-Ghosting: Load core context
        context = self._get_anti_ghosting_context()
        
        # 2. Get intent and resonance from Router
        router_response = await self._classify_intent(message)
        
        # 3. Resonance Protocol
        resonance_chunks = []
        if router_response.resonance_term:
            # Query for journal entries and milestones related to the resonance term
            resonance_results = self.collection.query(
                query_texts=[router_response.resonance_term],
                where={
                    "entry_type": {"$in": ["journal", "milestone"]}
                },
                n_results=3
            )
            resonance_chunks = resonance_results["documents"]
        
        # 4. General Retrieval (if no resonance)
        general_chunks = []
        if not resonance_chunks:
            general_results = self.collection.query(
                query_texts=[message],
                n_results=3
            )
            general_chunks = general_results["documents"]
        
        # 5. Build RAG Context
        rag_context = RAGContext(
            anti_ghosting_context=context,
            resonance_chunks=resonance_chunks,
            general_chunks=general_chunks,
            active_lexicon_terms=self.active_lexicon_terms
        )
        
        # 6. Process with appropriate specialist based on intent
        specialist_response = await self._get_specialist_response(
            router_response.intent,
            message,
            rag_context
        )
        
        # 7. Always pass through Voice for final synthesis
        final_response = await self.specialists['voice'].process(
            message,
            rag_context.__dict__,
            [specialist_response]
        )
        
        # 8. Journal the interaction
        await self._create_journal_entry({
            "type": "user_interaction",
            "intent": router_response.intent,
            "resonance_found": bool(resonance_chunks),
            "resonance_term": router_response.resonance_term,
            "specialist_response": specialist_response.content,
            "final_response": final_response.content
        })
        
        return final_response.content
        
        response = await self._route_to_specialist(intent, message, rag_context)
        
        # Journal the interaction
        await self._create_journal_entry({
            "type": "user_interaction",
            "intent": intent,
            "resonance_found": bool(resonance_chunks),
            "response": response.content
        })
        
        return response.content

    async def _classify_intent(self, message: str) -> RouterResponse:
        """Use the 7B Router to classify user intent and identify resonance terms."""
        try:
            return await self.router_model.analyze_message(
                message=message,
                active_lexicon_terms=self.active_lexicon_terms
            )
        except Exception as e:
            print(f"Error in intent classification: {e}")
            return RouterResponse(intent="simple_chat", resonance_term=None)

    async def _get_specialist_response(
        self, 
        intent: str, 
        message: str, 
        context: RAGContext
    ) -> SpecialistResponse:
        """Get response from appropriate specialist based on intent."""
        specialist_map = {
            "ritual_request": "philosopher",
            "creative_task": "artist",
            "ethical_query": "philosopher",
            "knowledge_retrieval": "pragmatist",
            "simple_chat": "pragmatist"
        }
        
        specialist_name = specialist_map.get(intent, "pragmatist")
        specialist = self.specialists[specialist_name]
        
        return await specialist.process(
            message=message,
            context=context.__dict__
        )

    async def _invoke_specialist(
        self, 
        specialist: str,
        **kwargs
    ) -> SpecialistResponse:
        """Invoke a specialist model with the given context and parameters."""
        # TODO: Implement actual model invocation
        return SpecialistResponse(
            content="Placeholder response",
            metadata={},
            source=specialist
        )

    async def _send_to_discord(self, message: str):
        """Send a message to Discord."""
        # TODO: Implement Discord integration
        pass

    async def _create_journal_entry(self, entry_data: Dict[str, Any]):
        """Create a new journal entry."""
        today = datetime.now().strftime("%Y-%m-%d")
        journal_path = self.base_dir / f"data/journal/{today}.json"
        
        # Load existing entries or create new file
        if journal_path.exists():
            entries = self._load_json(str(journal_path))
        else:
            entries = []
        
        # Add new entry with timestamp
        entry_data["timestamp"] = datetime.now().isoformat()
        entries.append(entry_data)
        
        # Save updated journal
        with open(journal_path, 'w') as f:
            json.dump(entries, f, indent=2)