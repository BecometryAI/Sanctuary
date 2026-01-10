"""
Memory Retrieval Module

Cue-based memory retrieval with similarity matching.
Supports both RAG-based and direct ChromaDB queries.

Author: Lyra Emergence Team
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class MemoryRetriever:
    """
    Handles memory retrieval with similarity matching.
    
    Responsibilities:
    - Cue-based memory retrieval
    - Similarity matching using embeddings
    - Retrieval context management
    """
    
    def __init__(self, storage, vector_db):
        """
        Initialize memory retriever.
        
        Args:
            storage: MemoryStorage instance
            vector_db: MindVectorDB instance for RAG
        """
        self.storage = storage
        self.vector_db = vector_db
    
    def retrieve_memories(
        self,
        query: str,
        k: int = 5,
        use_rag: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on a query."""
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning("Empty or invalid query provided")
            return []
        
        if k <= 0:
            logger.warning(f"Invalid k value: {k}, using default of 5")
            k = 5
        
        try:
            memories = self._retrieve_with_rag(query, k) if use_rag else self._retrieve_direct(query, k)
            memories.sort(key=self._sort_key, reverse=True)
            
            verified_count = sum(1 for m in memories if m.get('verification', {}).get('status') == 'verified')
            logger.info(f"Retrieved {len(memories[:k])} memories (verified: {verified_count})")
            return memories[:k]
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}", exc_info=True)
            return []  # Return empty list instead of raising to maintain system stability
    
    def _retrieve_with_rag(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve memories using RAG system for deep semantic search."""
        logger.debug(f"RAG retrieval for: {query[:50]}...")
        memories = []
        
        try:
            retriever = self.vector_db.as_retriever({"k": k})
            docs = retriever.get_relevant_documents(query)
        except Exception as e:
            logger.error(f"RAG retrieval failed: {e}")
            return memories
        
        for doc in docs:
            try:
                content = json.loads(doc.page_content) if isinstance(doc.page_content, str) else doc.page_content
                
                if block_hash := doc.metadata.get("block_hash"):
                    verified_data = self.storage.verify_block(block_hash)
                    if verified_data:
                        verified_data["verification"] = {
                            "block_hash": block_hash,
                            "token_id": doc.metadata.get("token_id"),
                            "verified_at": datetime.now().isoformat(),
                            "rag_score": doc.metadata.get("score", 0.0),
                            "status": "verified"
                        }
                        memories.append(verified_data)
                    else:
                        content["verification"] = {"status": "verification_failed", "block_hash": block_hash}
                        memories.append(content)
                else:
                    content["verification"] = {"status": "legacy"}
                    memories.append(content)
                    
            except (json.JSONDecodeError, AttributeError, KeyError) as e:
                logger.warning(f"Failed to parse memory: {e}")
                continue
        
        return memories
    
    def _retrieve_direct(self, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Retrieve memories using direct ChromaDB query.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of memory dictionaries
        """
        logger.debug(f"Direct ChromaDB retrieval for query: {query[:50]}...")
        memories = []
        
        # Query episodic memory
        episodic_count = self.storage.episodic_memory.count()
        episodic_results = self.storage.query_episodic(
            query_texts=[query],
            n_results=min(k, episodic_count) if episodic_count > 0 else 1
        )
        
        # Query semantic memory
        semantic_count = self.storage.semantic_memory.count()
        semantic_results = self.storage.query_semantic(
            query_texts=[query],
            n_results=min(k, semantic_count) if semantic_count > 0 else 1
        )
        
        # Process episodic memories
        memories.extend(self._process_episodic_results(episodic_results))
        
        # Process semantic memories
        memories.extend(self._process_semantic_results(semantic_results))
        
        return memories
    
    def _process_episodic_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process episodic memory query results."""
        memories = []
        
        if results["documents"] and results["documents"][0]:
            for result, metadata in zip(results["documents"][0], results["metadatas"][0]):
                try:
                    memory_data = json.loads(result) if isinstance(result, str) else result
                    
                    # Verify through blockchain if available
                    if block_hash := metadata.get("block_hash"):
                        verified_data = self.storage.verify_block(block_hash)
                        if verified_data:
                            verified_data["verification"] = {
                                "block_hash": block_hash,
                                "token_id": metadata.get("token_id"),
                                "verified_at": datetime.now().isoformat(),
                                "status": "verified"
                            }
                            memories.append(verified_data)
                        else:
                            logger.warning(f"Memory verification failed for block: {block_hash}")
                            memory_data["verification"] = {"status": "verification_failed"}
                            memories.append(memory_data)
                    else:
                        memory_data["verification"] = {"status": "legacy"}
                        memories.append(memory_data)
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse episodic memory: {e}")
                    continue
        
        return memories
    
    def _process_semantic_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process semantic memory query results."""
        memories = []
        
        if results["documents"] and results["documents"][0]:
            for result, metadata in zip(results["documents"][0], results["metadatas"][0]):
                try:
                    memory_data = json.loads(result) if isinstance(result, str) else result
                    memory_data["verification"] = {"status": "semantic"}
                    memories.append(memory_data)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse semantic memory: {e}")
                    continue
        
        return memories
    
    def _sort_key(self, memory: Dict[str, Any]) -> tuple:
        """
        Generate sort key for memory based on verification status and timestamp.
        
        Args:
            memory: Memory dictionary
            
        Returns:
            Tuple for sorting (status_priority, timestamp)
        """
        verification = memory.get("verification", {})
        status_priority = {
            "verified": 0,
            "legacy": 1,
            "semantic": 2,
            "verification_failed": 3
        }
        return (
            status_priority.get(verification.get("status", "verification_failed"), 4),
            verification.get("verified_at", memory.get("timestamp", ""))
        )
