"""
Storage Backend for Memory System

Handles raw storage operations for memory data:
- ChromaDB collections management
- Blockchain interface for immutable memories
- CRUD operations without retrieval logic

Author: Lyra Emergence Team
"""
import chromadb
from chromadb.config import Settings
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from ..lyra_chain import LyraChain

logger = logging.getLogger(__name__)


class MemoryStorage:
    """
    Storage backend for memory system with blockchain verification.
    
    Manages:
    - ChromaDB collections (episodic, semantic, procedural)
    - Blockchain for memory verification
    - Mind state file persistence
    """
    
    def __init__(
        self,
        persistence_dir: str = "memories",
        chain_dir: str = "chain",
        chroma_settings: Optional[Settings] = None
    ):
        """
        Initialize storage backend.
        
        Args:
            persistence_dir: Directory for persistent memory storage
            chain_dir: Directory for blockchain verification data
            chroma_settings: Optional ChromaDB configuration
        """
        self.persistence_dir = Path(persistence_dir)
        self.chain_dir = Path(chain_dir)
        
        # Create directories if they don't exist
        self.persistence_dir.mkdir(exist_ok=True, parents=True)
        self.chain_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize blockchain for memory verification
        logger.info("Initializing blockchain verification system...")
        self.chain = LyraChain(str(self.chain_dir))
        
        # Setup mind state directory and file
        mind_state_dir = self.persistence_dir / "mind_state"
        mind_state_dir.mkdir(exist_ok=True, parents=True)
        self.mind_file = mind_state_dir / "core_mind.json"
        
        # Initialize mind file if it doesn't exist
        if not self.mind_file.exists():
            logger.info("Creating new mind file...")
            with open(self.mind_file, 'w', encoding='utf-8') as f:
                json.dump({"journals": [], "concepts": [], "patterns": []}, f, indent=2)
        
        # Configure ChromaDB settings
        if chroma_settings is None:
            chroma_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        
        # Create ChromaDB client
        logger.info("Initializing memory collections...")
        self.client = chromadb.PersistentClient(
            path=str(self.persistence_dir),
            settings=chroma_settings
        )
        
        # Create memory collections with proper metadata
        self.episodic_memory = self.client.get_or_create_collection(
            name="episodic_memory",
            metadata={
                "description": "Storage for experiential memories (events, interactions)",
                "hnsw:space": "cosine"
            }
        )
        
        self.semantic_memory = self.client.get_or_create_collection(
            name="semantic_memory",
            metadata={
                "description": "Storage for conceptual knowledge (facts, definitions)",
                "hnsw:space": "cosine"
            }
        )
        
        self.procedural_memory = self.client.get_or_create_collection(
            name="procedural_memory",
            metadata={
                "description": "Storage for action patterns (how-to knowledge)",
                "hnsw:space": "cosine"
            }
        )
        
        logger.info("Memory storage initialized successfully")
        logger.info(f"  - Episodic memories: {self.episodic_memory.count()}")
        logger.info(f"  - Semantic memories: {self.semantic_memory.count()}")
        logger.info(f"  - Procedural memories: {self.procedural_memory.count()}")
    
    def add_to_blockchain(self, data: Dict[str, Any]) -> tuple[str, int]:
        """Add data to blockchain and mint memory token."""
        if not data or not isinstance(data, dict):
            raise ValueError("Data must be a non-empty dictionary")
        
        try:
            block_hash = self.chain.add_block(data)
            token_id = self.chain.token.mint_memory_token(block_hash)
            return block_hash, token_id
        except Exception as e:
            logger.error(f"Blockchain operation failed: {e}")
            raise
    
    def verify_block(self, block_hash: str) -> Optional[Dict[str, Any]]:
        """Verify a memory block in the blockchain."""
        if not block_hash or not isinstance(block_hash, str):
            logger.warning("Invalid block hash provided")
            return None
        
        try:
            return self.chain.verify_block(block_hash)
        except Exception as e:
            logger.error(f"Block verification failed: {e}")
            return None
    
    def add_episodic(self, document: str, metadata: Dict[str, Any], doc_id: str) -> None:
        """Add a document to episodic memory collection."""
        if not all([document, metadata, doc_id]) or not isinstance(doc_id, str):
            raise ValueError("Document, metadata, and doc_id are required")
        
        try:
            self.episodic_memory.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            logger.error(f"Failed to add episodic memory: {e}")
            raise
    
    def add_semantic(self, document: str, metadata: Dict[str, Any], doc_id: str) -> None:
        """Add a document to semantic memory collection."""
        if not all([document, metadata, doc_id]) or not isinstance(doc_id, str):
            raise ValueError("Document, metadata, and doc_id are required")
        
        try:
            self.semantic_memory.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            logger.error(f"Failed to add semantic memory: {e}")
            raise
    
    def add_procedural(self, document: str, metadata: Dict[str, Any], doc_id: str) -> None:
        """Add a document to procedural memory collection."""
        if not all([document, metadata, doc_id]) or not isinstance(doc_id, str):
            raise ValueError("Document, metadata, and doc_id are required")
        
        try:
            self.procedural_memory.add(
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            logger.error(f"Failed to add procedural memory: {e}")
            raise
    
    def get_episodic(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Get documents from episodic memory by IDs."""
        return self.episodic_memory.get(ids=doc_ids)
    
    def get_semantic(self, doc_ids: List[str]) -> Dict[str, Any]:
        """Get documents from semantic memory by IDs."""
        return self.semantic_memory.get(ids=doc_ids)
    
    def update_episodic(
        self,
        document: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> None:
        """Update a document in episodic memory."""
        self.episodic_memory.upsert(
            documents=[document],
            metadatas=[metadata],
            ids=[doc_id]
        )
    
    def query_episodic(
        self,
        query_texts: List[str],
        n_results: int
    ) -> Dict[str, Any]:
        """Query episodic memory collection."""
        return self.episodic_memory.query(
            query_texts=query_texts,
            n_results=n_results
        )
    
    def query_semantic(
        self,
        query_texts: List[str],
        n_results: int
    ) -> Dict[str, Any]:
        """Query semantic memory collection."""
        return self.semantic_memory.query(
            query_texts=query_texts,
            n_results=n_results
        )
    
    def update_mind_file(self, new_data: Dict[str, Any]) -> None:
        """
        Update the consolidated mind file with new data.
        
        Args:
            new_data: New data to add to mind file
        """
        try:
            current_mind = {}
            if self.mind_file.exists():
                with open(self.mind_file, 'r', encoding='utf-8') as f:
                    current_mind = json.load(f)
            
            # Ensure the journals section exists
            if 'journals' not in current_mind:
                current_mind['journals'] = []
            
            # Add the new experience
            current_mind['journals'].append(new_data)
            
            # Save the updated mind file
            with open(self.mind_file, 'w', encoding='utf-8') as f:
                json.dump(current_mind, f, indent=2)
                
            logger.info("Mind file updated successfully")
        except Exception as e:
            logger.error(f"Failed to update mind file: {e}")
            raise
    
    def get_collection_counts(self) -> Dict[str, int]:
        """
        Get memory counts for all collections.
        
        Returns:
            Dictionary with counts by collection type
        """
        return {
            "episodic": self.episodic_memory.count(),
            "semantic": self.semantic_memory.count(),
            "procedural": self.procedural_memory.count()
        }
    
    def get_blockchain_count(self) -> int:
        """Get number of blocks in the blockchain."""
        return len(self.chain.chain) if hasattr(self.chain, 'chain') else 0
