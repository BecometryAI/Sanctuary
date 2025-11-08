"""
Memory weaving system for integrating new memories with verification.
"""
from typing import Dict, Any, Optional, List
import logging
import json
import re
from datetime import datetime
from pathlib import Path

from .memory import MemoryManager
from .rag_engine import MindVectorDB, RAGQueryEngine

logger = logging.getLogger(__name__)

class MemoryWeaver:
    """
    Handles the integration of new memories into the system with RAG and blockchain verification.
    """
    def __init__(self, memory_manager: MemoryManager, vector_db: MindVectorDB, rag_engine: RAGQueryEngine):
        self.memory_manager = memory_manager
        self.vector_db = vector_db
        self.rag_engine = rag_engine

    async def process_memory(self, response_text: str) -> bool:
        """
        Processes a response text to extract and store memory entries.
        Returns True if memory was successfully processed and stored.
        """
        try:
            # Extract memory entry if present
            memory_entry = self._extract_memory_entry(response_text)
            if not memory_entry:
                return False

            # Store the memory with blockchain verification
            await self.store_memory(memory_entry)
            
            # Update vector store
            self.vector_db.index()
            
            logger.info("Memory successfully processed and integrated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process memory: {e}")
            return False

    def _extract_memory_entry(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts memory entry from response text using pattern matching.
        Supports both JSON blocks and structured text formats.
        """
        # Try to extract JSON block first
        json_match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                logger.debug("Found JSON-like structure but failed to parse")
                pass

        # Try to extract structured memory format
        memory_match = re.search(r'Memory:(.*?)(?=\n\n|\Z)', text, re.DOTALL)
        if memory_match:
            memory_text = memory_match.group(1).strip()
            return {
                "content": memory_text,
                "type": "extracted_memory",
                "timestamp": datetime.now().isoformat(),
                "source": "response_text"
            }

        return None

    async def store_memory(self, memory_data: Dict[str, Any]) -> bool:
        """
        Stores a new memory with proper verification and integration.
        """
        try:
            # Add metadata
            memory_data.update({
                "timestamp": datetime.now().isoformat(),
                "processing_metadata": {
                    "memory_weaver_version": "1.0",
                    "processing_time": datetime.now().isoformat()
                }
            })

            # Store in memory manager (includes blockchain verification)
            self.memory_manager.store_experience(memory_data)

            # Query similar memories for context enrichment
            similar_memories = await self.find_similar_memories(memory_data)
            if similar_memories:
                memory_data["context"] = {
                    "similar_memories": similar_memories,
                    "enrichment_time": datetime.now().isoformat()
                }
                # Update the stored memory with context
                self.memory_manager.update_experience(memory_data)

            return True

        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return False

    async def find_similar_memories(self, memory_data: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
        """
        Finds similar memories using RAG for context enrichment.
        """
        try:
            # Create search query from memory content
            query = memory_data.get("content", "") or memory_data.get("description", "")
            if not query:
                return []

            # Use RAG to find similar memories
            similar = self.memory_manager.retrieve_relevant_memories(
                query=query,
                k=k,
                use_rag=True
            )

            # Filter out the current memory if it's somehow included
            current_hash = memory_data.get("block_hash")
            if current_hash:
                similar = [m for m in similar if m.get("block_hash") != current_hash]

            return similar[:k]

        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []

    async def process_interaction(self, 
                                query: str, 
                                response: str, 
                                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a complete interaction (query + response) for memory integration.
        """
        interaction_data = {
            "query": query,
            "response": response,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "type": "interaction_memory"
        }

        # Process any embedded memories in the response
        memory_processed = await self.process_memory(response)
        
        # Store the interaction itself as a memory
        await self.store_memory(interaction_data)

        return {
            "status": "success",
            "memory_processed": memory_processed,
            "interaction_stored": True,
            "timestamp": datetime.now().isoformat()
        }