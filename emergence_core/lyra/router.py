"""
Lyra's Adaptive Router Implementation with Voice Integration
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List, NamedTuple
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

from .autonomous import AutonomousCore
from .router_model import RouterModel
from .specialists import SpecialistFactory, SpecialistOutput
from .specialist_tools import (
    searxng_search,
    arxiv_search,
    wikipedia_search,
    wolfram_compute,
    python_repl,
    playwright_interact
)

# Placeholder voice tools
async def discord_join_voice_channel(channel_id: str) -> bool:
    return False

async def discord_leave_voice_channel() -> bool:
    return True

async def coqui_tts_speak(text: str) -> bool:
    return True

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

# Import PIL for image handling
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False
    logger.warning("PIL not installed - image input unavailable")

class AdaptiveRouter:
    def __init__(self, base_dir: str, chroma_dir: str, model_dir: str, development_mode: bool = False):
        """
        Initialize the router with Gemma 12B model and specialists.
        
        Args:
            base_dir: Base directory for all data
            chroma_dir: Directory for ChromaDB storage
            model_dir: Directory containing model files
            development_mode: If True, operate in development mode with mock data
        """
        self.base_dir = Path(base_dir)
        self.chroma_dir = Path(chroma_dir)
        self.development_mode = development_mode
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
        core_archives_path = Path(self.base_dir).parent / "data" / "Core_Archives"
        self.continuity_archive = self._load_json(core_archives_path / "lyra_continuity_archive.json")
        self.relational_archive = self._load_json(core_archives_path / "lyra_relational_archive.json")
        
        # Initialize ChromaDB for RAG
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        try:
            self.collection = self.chroma_client.get_collection("lyra_knowledge")
        except Exception:
            logger.info("Creating new lyra_knowledge collection")
            self.collection = self.chroma_client.create_collection(
                name="lyra_knowledge",
                metadata={
                    "description": "General knowledge base for Lyra",
                    "hnsw:space": "cosine"
                }
            )
        
        # Initialize Router Model with development mode
        router_model = "HuggingFaceH4/zephyr-7b-beta"  # Model path for future use
        self.router_model = RouterModel(router_model, development_mode=True)
        
        # Initialize Specialist Factory
        self.specialist_factory = SpecialistFactory()
        
        # Initialize Specialists with model IDs (but don't load models in development)
        specialist_configs = {
            'philosopher': "deepseek-ai/deepseek-coder-33b-instruct",
            'pragmatist': "Qwen/Qwen-14B", 
            'artist': "HuggingFaceH4/zephyr-7b-beta",
            'voice': "HuggingFaceH4/zephyr-7b-beta"
        }
        
        # In development, create mock specialists
        self.specialists = {}
        for name, model_path in specialist_configs.items():
            try:
                specialist = self.specialist_factory.create_specialist(
                    name, 
                    str(self.base_dir),
                    model_path,
                    development_mode=True
                )
                self.specialists[name] = specialist
            except Exception as e:
                logger.warning(f"Could not initialize specialist {name}: {e}")
                # Add a mock specialist that returns empty responses
                self.specialists[name] = None
        
        # Initialize Autonomous Core
        self.autonomous_core = AutonomousCore(self.base_dir, self.specialists)
        
        # Cache active lexicon terms
        self.active_lexicon_terms = self._load_active_lexicon_terms()
        
        # Start cognitive loop scheduler
        self._init_scheduler()

    def _load_json(self, relative_path: str) -> Any:
        """
        Load and parse a JSON file.
        
        Args:
            relative_path: Path relative to base_dir
            
        Returns:
            Dict or List containing parsed JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist (only in non-development mode)
            ValueError: If JSON is invalid
            RuntimeError: For other errors
        """
        path = self.base_dir / relative_path
        if not path.exists():
            msg = f"File does not exist: {path}"
            logger.warning(msg)
            
            # In development mode, return mock data
            if hasattr(self, "development_mode") and self.development_mode:
                logger.warning("Development mode enabled - returning mock data")
                path_str = str(relative_path)
                if "symbolic_lexicon.json" in path_str:
                    return {"terms": []}
                elif "Rituals.json" in path_str:
                    return []
                else:
                    return {}
            
            raise FileNotFoundError(msg)
            
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in {path}: {e}"
            logger.error(msg)
            raise ValueError(msg) from None
        except OSError as e:
            msg = f"Error reading {path}: {e}"
            logger.error(msg)
            raise RuntimeError(msg) from None
                
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
    
    async def route_message(
        self, 
        message: str, 
        context: Optional[Dict[str, Any]] = None,
        image: Optional[Image.Image] = None
    ) -> SpecialistOutput:
        """
        SEQUENTIAL WORKFLOW: Route message (with optional image) through specialist then Voice synthesis.
        
        Workflow:
        1. (If image) → Perception specialist converts image to text description
        2. User input → Router (Gemma 12B) classification
        3. Router selects ONE specialist: Pragmatist, Philosopher, or Artist  
        4. Selected specialist processes the message
        5. Specialist output → Voice (LLaMA 3 70B) for synthesis
        6. Voice returns final first-person response
        
        Args:
            message: The user's message
            context: Optional context dictionary
            image: Optional PIL Image for vision processing
            
        Returns:
            SpecialistOutput containing Lyra's synthesized response
        """
        if context is None:
            context = {}
        
        # PRE-STEP: Vision understanding if image provided
        if image is not None and HAS_PIL:
            logger.info("Image detected - invoking Perception specialist")
            
            perception = self.specialists.get('perception')
            if perception:
                try:
                    # Get visual description from Perception specialist
                    vision_output = await perception.process(
                        image=image,
                        prompt=f"Analyze this image in context of: {message}" if message else "Describe this image in detail, noting artistic elements and mood",
                        context=context
                    )
                    
                    # Enhance message with visual context
                    visual_context = vision_output.content
                    if message:
                        message = f"{message}\n\n[Visual Context: {visual_context}]"
                    else:
                        message = f"[Image uploaded]\n{visual_context}"
                    
                    # Add to context
                    context['visual_analysis'] = visual_context
                    context['has_image'] = True
                    context['image_size'] = vision_output.metadata.get('image_size')
                    
                    logger.info(f"Visual analysis complete: {visual_context[:100]}...")
                    
                except Exception as e:
                    logger.error(f"Perception failed: {e}")
                    context['perception_error'] = str(e)
            else:
                logger.warning("Perception specialist not available - image will be ignored")
                context['perception_unavailable'] = True
            
        # STEP 1: Router classification with Gemma 12B
        router_response = await self.router_model.analyze_message(message, self.active_lexicon_terms)
        
        # Determine which specialist to use (lowercase for dict lookup)
        specialist_type = router_response.intent.lower()
        
        # Validate specialist type
        if specialist_type not in ['pragmatist', 'philosopher', 'artist']:
            logger.warning(f"Invalid specialist type '{specialist_type}', defaulting to pragmatist")
            specialist_type = 'pragmatist'
        
        # Check if resonance term was detected
        if router_response.resonance_term:
            context["resonance_term"] = router_response.resonance_term
            context["lexicon_activated"] = True
        
        # STEP 2: Get the ONE selected specialist (with fallback to pragmatist)
        specialist = self.specialists.get(specialist_type)
        if specialist is None:
            logger.warning(f"Specialist {specialist_type} unavailable, falling back to pragmatist")
            specialist = self.specialists.get("pragmatist")
            specialist_type = "pragmatist"
            context["fallback_used"] = True
        
        logger.info(f"Sequential workflow: {message[:50]}... → {specialist_type.upper()} → Voice")
        
        # STEP 3: Process message with the SINGLE specialist
        specialist_output = await specialist.process(message, context)
        
        # Add routing metadata to specialist output
        specialist_output.metadata["specialist"] = specialist_type
        specialist_output.metadata["resonance_term"] = router_response.resonance_term
        if "lexicon_activated" in context:
            specialist_output.metadata["lexicon_activated"] = context["lexicon_activated"]
        if "fallback_used" in context:
            specialist_output.metadata["fallback_used"] = context["fallback_used"]
        
        # STEP 4: Pass specialist output to Voice for final synthesis
        voice_specialist = self.specialists.get("voice")
        if voice_specialist is None:
            logger.warning("Voice specialist unavailable, returning specialist output directly")
            return specialist_output
        
        # STEP 5: Voice synthesizes into first-person Lyra response
        final_response = await voice_specialist.synthesize(
            original_query=message,
            specialist_output=specialist_output,
            specialist_name=specialist_type.title(),  # Capitalize for display
            context=context
        )
        
        logger.info(f"Sequential workflow complete: {specialist_type} → Voice → User")
        
        return final_response
            
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

    async def _send_to_discord(self, message: str, channel_id: Optional[int] = None, user_id: Optional[int] = None):
        """
        Send a message to Discord.
        Args:
            message: The message to send
            channel_id: Optional specific channel to send to
            user_id: Optional specific user to DM
        """
        if not hasattr(self, 'discord_client'):
            logger.error("Discord client not initialized")
            return

        try:
            if user_id:
                # Send DM to specific user
                user = await self.discord_client.fetch_user(user_id)
                if user:
                    await user.send(message)
            elif channel_id:
                # Send to specific channel
                channel = self.discord_client.get_channel(channel_id)
                if channel:
                    await channel.send(message)
            else:
                # Send to default channel if configured
                if hasattr(self, 'default_channel_id'):
                    channel = self.discord_client.get_channel(self.default_channel_id)
                    if channel:
                        await channel.send(message)
                else:
                    logger.error("No target specified for Discord message")
        except Exception as e:
            logger.error(f"Error sending Discord message: {e}")

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