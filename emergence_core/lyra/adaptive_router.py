"""
Lyra's Adaptive Router Implementation
"""
from typing import Dict, Any, Optional, List
from pathlib import Path
import json
import asyncio
import time
import chromadb
from .cognitive_logger import CognitiveLogger
from .config import SystemConfig, ModelRegistry
from .rag_cache import RAGCache
from .router_model import RouterModel
from .specialists import SpecialistFactory, SpecialistOutput

class AdaptiveRouter:
    @classmethod
    async def create(cls, base_dir: str, chroma_dir: str, model_dir: str) -> 'AdaptiveRouter':
        """Async factory method to create and initialize the router."""
        instance = cls()
        await instance._initialize(base_dir, chroma_dir, model_dir)
        return instance
        
    def __init__(self):
        """Initialize empty instance for async creation."""
        pass
        
    async def _verify_model(self, specialist, spec_type: str) -> None:
        """Verify model can process a simple test input."""
        test_message = "Test message for model verification."
        test_context = {}
        try:
            await specialist.process(test_message, test_context)
        except Exception as e:
            raise RuntimeError(f"Model verification failed for {spec_type}: {str(e)}")
            
    async def _initialize(self, base_dir: str, chroma_dir: str, model_dir: str):
        """Initialize the router components."""
        # Load system configuration
        self.config = SystemConfig.from_json(str(Path(base_dir) / "config" / "system.json"))
        self.model_registry = ModelRegistry(str(Path(base_dir) / "config" / "models.json"))
        
        self.base_dir = self.config.base_dir
        self.chroma_dir = self.config.chroma_dir
        self.model_dir = self.config.model_dir
        
        # Initialize logger
        self.logger = CognitiveLogger(self.config.log_dir)
        
        # Initialize RAG cache
        self.rag_cache = RAGCache(self.config.cache_dir)
        """Initialize the router with Gemma 12B model and specialists."""
        self.base_dir = Path(base_dir)
        self.chroma_dir = Path(chroma_dir)
        self.model_dir = Path(model_dir)
        
        # Initialize ChromaDB for RAG
        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.chroma_client.get_collection("lyra_knowledge")
        
        # Initialize Router Model (Gemma 12B) with error handling
        try:
            router_model_path = self.model_dir / "gemma_12b_router"
            self.router_model = RouterModel(str(router_model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Router Model: {str(e)}")

        # Initialize Specialist Factory
        self.specialist_factory = SpecialistFactory()
        
        # Initialize Specialists with error handling and model verification
        self.specialists = {}
        specialist_types = ['philosopher', 'pragmatist', 'artist', 'voice']
        
        for spec_type in specialist_types:
            try:
                specialist = self.specialist_factory.create_specialist(spec_type, self.base_dir)
                # Verify model loading by running a small test input
                await self._verify_model(specialist, spec_type)
                self.specialists[spec_type] = specialist
            except Exception as e:
                raise RuntimeError(f"Failed to initialize {spec_type} specialist: {str(e)}")
                
        # Initialize model state cache
        self._model_health = {
            'router': {'status': 'healthy', 'last_check': None},
            'specialists': {k: {'status': 'healthy', 'last_check': None} for k in self.specialists}
        }

    def _load_json(self, relative_path: str) -> Dict[str, Any]:
        """Load and parse a JSON file."""
        file_path = self.base_dir / relative_path
        with open(file_path, 'r') as f:
            return json.load(f)

    async def _get_router_intent(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get routing intent from the Gemma 12B router model."""
        # The router model returns a JSON with 'intent' and 'resonance_term'
        return await self.router_model.process(message, context)

    async def _get_rag_context(self, message: str, resonance_term: Optional[str] = None) -> Dict[str, Any]:
        """Get relevant context from the RAG system with caching."""
        # Check cache first
        cache_key = {'message': message, 'resonance_term': resonance_term}
        cached_result = self.rag_cache.get(message, cache_key)
        if cached_result is not None:
            self.logger.log_model_performance('rag', 'cache_hit', 0, True)
            return cached_result
            
        # Query the vector store for relevant chunks
        start_time = time.time()
        try:
            results = self.collection.query(
                query_texts=[message],
                n_results=5
            )
            
            context = {
                'chunks': results['documents'][0],
                'resonance_term': resonance_term
            }
            
            if resonance_term:
                resonance_results = self.collection.query(
                    query_texts=[resonance_term],
                    n_results=3
                )
                context['resonance_chunks'] = resonance_results['documents'][0]
                
            # Cache the results
            self.rag_cache.set(message, context, cache_key)
            
            duration = (time.time() - start_time) * 1000
            self.logger.log_model_performance('rag', 'query', duration, True)
            
            return context
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            self.logger.log_model_performance('rag', 'query', duration, False)
            raise RuntimeError(f"RAG query failed: {str(e)}")

    async def _process_with_specialist(self, 
        specialist_type: str, 
        message: str, 
        context: Dict[str, Any]
    ) -> SpecialistOutput:
        """Process message with a specific specialist."""
        specialist = self.specialists[specialist_type]
        return await specialist.process(message, context)

    async def process_message(self, message: str, context: Dict[str, Any] = None) -> str:
        """Process a user message through the routing system."""
        if context is None:
            context = {}
            
        # First, get any relevant RAG context
        rag_context = await self._get_rag_context(message)
        context.update(rag_context)

        # Get routing decision from Gemma 12B
        router_output = await self._get_router_intent(message, context)
        intent = router_output['intent']
        resonance_term = router_output['resonance_term']

        # Update context with any resonance terms found
        if resonance_term:
            resonance_context = await self._get_rag_context(message, resonance_term)
            context.update(resonance_context)

        # For simple chat, route directly to Voice
        if intent == 'simple_chat':
            voice_output = await self.specialists['voice'].process(
                message=message,
                specialist_outputs={},
                context=context
            )
            return voice_output.content

        # For other intents, process with appropriate specialist first
        specialist_output = await self._process_with_specialist(
            specialist_type=intent,
            message=message,
            context=context
        )

        # Then synthesize through Voice
        voice_output = await self.specialists['voice'].process(
            message=message,
            specialist_outputs={intent: specialist_output.content},
            context=context
        )

        return voice_output.content