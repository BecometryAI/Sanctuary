"""
Memory management system for Sanctuary's consciousness core with custom blockchain verification.

NOTE: This module was renamed from memory.py to memory_legacy.py to resolve a naming
conflict with the memory/ package directory. Python prioritizes packages over modules
with the same name, which broke imports. Import as:

    from mind.memory_legacy import MemoryManager

This module implements a three-tier persistent memory architecture:
1. Episodic Memory - Event-based experiences (journal entries, conversations)
2. Semantic Memory - Conceptual knowledge (protocols, lexicon, facts)
3. Working Memory - Short-term context for current interactions

The memory system has been refactored into focused modules (in memory/ package):
- storage: Raw storage backend (DB, blockchain interface)
- encoding: Transform experiences into storable representations
- retrieval: Cue-based retrieval with similarity matching
- consolidation: Memory strengthening, decay, reorganization
- emotional_weighting: Emotional salience affects storage/retrieval priority
- episodic: Autobiographical episodes (what, when, where)
- semantic: Facts and knowledge (context-independent)
- working: Short-term workspace buffer

Usage Example:
    Initialize and load existing data:

    >>> from mind.memory_legacy import MemoryManager
    >>> memory = MemoryManager(persistence_dir="memories", chain_dir="chain")
    >>> 
    >>> # Load all existing journals, protocols, and lexicon
    >>> results = memory.load_all_static_data(journal_limit=50)
    >>> print(f"Loaded {results['journals']} journal entries")
    >>> 
    >>> # Store a new experience
    >>> memory.store_experience({
    ...     "description": "User asked about consciousness",
    ...     "response": "I explained my understanding...",
    ...     "emotional_tone": ["thoughtful", "engaged"]
    ... })
    >>> 
    >>> # Retrieve relevant memories
    >>> memories = memory.retrieve_relevant_memories("What is Becometry?", k=5)
    >>> for mem in memories:
    ...     print(mem.get('description', mem.get('term')))
    >>> 
    >>> # Use working memory for conversation context
    >>> memory.update_working_memory("current_topic", "consciousness", ttl_seconds=1800)
    >>> context = memory.get_working_memory_context(max_items=10)
    >>> 
    >>> # Get system statistics
    >>> stats = memory.get_memory_stats()
    >>> print(f"Total memories: {stats['total_memories']}")

Architecture:
    The memory system integrates:
    - ChromaDB for vector storage and semantic search
    - Custom blockchain for memory verification and integrity
    - RAG (Retrieval-Augmented Generation) for contextual memory retrieval
    - Batch indexing for performance optimization
    
    All memories are stored in ChromaDB collections with optional blockchain
    verification. The RAG engine provides semantic search across all memory types.

Dependencies:
    - chromadb: Vector database for semantic search
    - langchain: RAG framework integration
    - Custom SanctuaryChain: Blockchain verification
    - Custom MindVectorDB: Vector store management
"""
from typing import Dict, List, Any, Optional
from chromadb.config import Settings
import logging
from datetime import datetime
from pathlib import Path

from .rag_engine import MindVectorDB
from .memory.storage import MemoryStorage
from .memory.encoding import MemoryEncoder
from .memory.retrieval import MemoryRetriever
from .memory.consolidation import MemoryConsolidator
from .memory.emotional_weighting import EmotionalWeighting
from .memory.episodic import EpisodicMemory
from .memory.semantic import SemanticMemory
from .memory.working import WorkingMemory

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Orchestrates the memory system using specialized modules.
    
    This class maintains backward compatibility while delegating
    to focused, single-responsibility modules.
    """
    
    def __init__(
        self, 
        persistence_dir: str = "memories", 
        chain_dir: str = "chain", 
        chroma_settings=None,
        auto_load_data: bool = True,
        journal_limit: Optional[int] = None
    ):
        """
        Initialize the memory management system with blockchain integration and RAG.
        
        This creates a three-tier memory architecture:
        - Episodic: Event-based memories (what happened)
        - Semantic: Conceptual knowledge (what things mean)
        - Working: Short-term workspace buffer
        
        Args:
            persistence_dir: Directory for persistent memory storage
            chain_dir: Directory for blockchain verification data
            chroma_settings: Optional ChromaDB configuration settings
            auto_load_data: If True, automatically load journals/protocols/lexicon on init
            journal_limit: Optional limit on number of journal files to load (None = all)
        
        Raises:
            RuntimeError: If memory systems fail to initialize
        """
        try:
            # Configure ChromaDB settings
            if chroma_settings is None:
                chroma_settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            
            # Initialize storage backend
            logger.info("Initializing memory storage...")
            self.storage = MemoryStorage(persistence_dir, chain_dir, chroma_settings)
            
            # Initialize vector database for RAG
            logger.info("Initializing vector database for RAG...")
            self.vector_db = MindVectorDB(
                db_path=str(Path(persistence_dir) / "vector_store"),
                mind_file=str(self.storage.mind_file),
                chain_dir=chain_dir,
                chroma_settings=chroma_settings
            )
            
            # Determine data directory
            self.data_dir = self._find_data_directory()
            logger.info(f"  - Data directory: {self.data_dir}")
            
            # Initialize subsystems
            self.encoder = MemoryEncoder(self.data_dir)
            self.retriever = MemoryRetriever(self.storage, self.vector_db)
            self.consolidator = MemoryConsolidator(self.storage, self.encoder)
            self.emotional_weighting = EmotionalWeighting()
            self.episodic = EpisodicMemory(self.storage, self.encoder, self.data_dir)
            self.semantic = SemanticMemory(self.storage, self.encoder, self.data_dir)
            self.working_memory = WorkingMemory()
            
            # Track indexing state for batch optimization
            self._pending_experiences = []
            self._last_index_time = datetime.now()
            self._index_batch_size = 10  # Trigger reindexing after N experiences
            
            # Backward compatibility: expose storage collections
            self.episodic_memory = self.storage.episodic_memory
            self.semantic_memory = self.storage.semantic_memory
            self.procedural_memory = self.storage.procedural_memory
            self.chain = self.storage.chain
            self.mind_file = self.storage.mind_file
            self.persistence_dir = Path(persistence_dir)
            self.chain_dir = Path(chain_dir)
            
            logger.info("Memory system initialized successfully")
            counts = self.storage.get_collection_counts()
            logger.info(f"  - Episodic memories: {counts['episodic']}")
            logger.info(f"  - Semantic memories: {counts['semantic']}")
            logger.info(f"  - Procedural memories: {counts['procedural']}")
            
            # Automatically load existing data if requested
            if auto_load_data:
                logger.info("Auto-loading static data (journals, protocols, lexicon)...")
                try:
                    load_results = self.load_all_static_data(journal_limit=journal_limit)
                    logger.info(f"Auto-load complete: {sum(load_results.values())} items loaded")
                except Exception as e:
                    logger.warning(f"Auto-load failed (system will continue): {e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory systems: {e}", exc_info=True)
            raise RuntimeError(f"Memory system initialization failed: {e}") from e
    
    def _find_data_directory(self) -> Path:
        """
        Locate the data directory containing journals, protocols, and lexicon.
        
        Checks:
        1. sanctuary/data/
        2. ../data/ (parent directory)
        3. ./data/ (current directory)
        
        Returns:
            Path to data directory
            
        Raises:
            FileNotFoundError: If data directory cannot be found
        """
        # Try multiple data directory locations
        candidates = [
            Path(__file__).parent.parent.parent / "data",  # project_root/data
            Path(__file__).parent.parent / "data",  # sanctuary/data
            Path("data"),  # ./data
        ]
        
        for candidate in candidates:
            if candidate.exists():
                # Check if it has expected subdirectories
                if (candidate / "journal").exists() or (candidate / "Protocols").exists():
                    logger.info(f"Found data directory: {candidate}")
                    return candidate
        
        raise FileNotFoundError(
            f"Could not find data directory with journal or Protocols. Checked: {[str(c) for c in candidates]}"
        )
    
    def load_journal_entries(self, limit: Optional[int] = None) -> int:
        """
        Load journal entries from data/journal/*.json into episodic memory.
        
        Args:
            limit: Optional limit on number of journals to load (most recent first)
            
        Returns:
            Number of journal entries loaded
        """
        return self.episodic.load_journal_entries(limit)
    
    def load_protocols(self) -> int:
        """
        Load protocol files from data/Protocols/*.json into semantic memory.
        
        Returns:
            Number of protocols loaded
        """
        return self.semantic.load_protocols()
    
    def load_lexicon(self) -> int:
        """
        Load lexicon files from data/Lexicon/*.json into semantic memory.
        
        Returns:
            Number of lexicon entries loaded
        """
        return self.semantic.load_lexicon()
    
    def load_all_static_data(self, journal_limit: Optional[int] = None) -> Dict[str, int]:
        """
        Load all static data files into memory: journals, protocols, and lexicon.
        
        Args:
            journal_limit: Optional limit on number of journal files to load
            
        Returns:
            Dictionary with counts of loaded items by type
        """
        logger.info("Loading all static data into memory...")
        results = {}
        
        try:
            results["journals"] = self.load_journal_entries(limit=journal_limit)
        except Exception as e:
            logger.error(f"Failed to load journals: {e}")
            results["journals"] = 0
        
        try:
            results["protocols"] = self.load_protocols()
        except Exception as e:
            logger.error(f"Failed to load protocols: {e}")
            results["protocols"] = 0
        
        try:
            results["lexicon"] = self.load_lexicon()
        except Exception as e:
            logger.error(f"Failed to load lexicon: {e}")
            results["lexicon"] = 0
        
        total_loaded = sum(results.values())
        logger.info(f"Static data loading complete: {total_loaded} total items")
        logger.info(f"  - Journal entries: {results['journals']}")
        logger.info(f"  - Protocols: {results['protocols']}")
        logger.info(f"  - Lexicon entries: {results['lexicon']}")
        
        # Trigger indexing to make all loaded data searchable
        if total_loaded > 0:
            logger.info("Indexing loaded data for RAG retrieval...")
            self.force_reindex()
        
        return results
    
    
    def store_experience(self, experience: Dict[str, Any], force_index: bool = False):
        """
        Store a new experience in episodic memory with blockchain verification.
        
        Args:
            experience: Dictionary containing the experience data
            force_index: If True, trigger immediate reindexing
            
        Returns:
            Dict with block_hash and token_id if successful
        """
        # Store the experience
        result = self.episodic.store_experience(experience, use_blockchain=True)
        
        # Add to pending batch for RAG indexing
        self._pending_experiences.append(experience)
        
        # Check if we should prioritize this memory for immediate indexing
        should_prioritize = self.emotional_weighting.should_prioritize_storage(experience)
        
        # Trigger RAG reindexing if conditions are met
        should_reindex = (
            force_index or
            should_prioritize or
            len(self._pending_experiences) >= self._index_batch_size or
            (datetime.now() - self._last_index_time).total_seconds() > 300  # 5 minutes
        )
        
        if should_reindex:
            logger.info(f"Triggering RAG reindexing ({len(self._pending_experiences)} pending experiences)...")
            self.vector_db.index()
            self._pending_experiences = []
            self._last_index_time = datetime.now()
        else:
            logger.debug(f"Batching experience for later indexing ({len(self._pending_experiences)}/{self._index_batch_size})")
        
        return result
    
    def _update_mind_file(self, new_data: Dict[str, Any]):
        """Update the consolidated mind file with new data"""
        self.storage.update_mind_file(new_data)
    
    def update_experience(self, experience_data: Dict[str, Any]) -> bool:
        """
        Update an existing experience with new data while maintaining blockchain integrity.
        
        Args:
            experience_data: Updated experience data
            
        Returns:
            True if successful, False otherwise
        """
        result = self.episodic.update_experience(experience_data)
        
        if result:
            # Trigger vector store reindexing
            self.vector_db.index()
        
        return result
    
    def store_concept(self, concept: Dict[str, Any]):
        """
        Store semantic knowledge.
        
        Args:
            concept: Concept data dictionary
        """
        self.semantic.store_concept(concept)
    
    def retrieve_relevant_memories(self, query: str, k: int = 5, use_rag: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on a query.
        
        Args:
            query: The search query (natural language)
            k: Number of results to return
            use_rag: Whether to use RAG system (True) or direct ChromaDB (False)
            
        Returns:
            List of memory dictionaries, sorted by relevance
        """
        return self.retriever.retrieve_memories(query, k, use_rag)
    
    def update_working_memory(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """
        Update working memory with optional time-to-live.
        
        Args:
            key: Memory key identifier
            value: Memory value (any JSON-serializable data)
            ttl_seconds: Optional time-to-live in seconds (None = no expiration)
        """
        self.working_memory.update(key, value, ttl_seconds)
    
    def get_working_memory(self, key: str) -> Any:
        """
        Retrieve from working memory.
        
        Args:
            key: Memory key to retrieve
            
        Returns:
            Memory value if exists and not expired, None otherwise
        """
        return self.working_memory.get(key)
    
    def get_working_memory_context(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent working memory as context for LLM prompts.
        
        Args:
            max_items: Maximum number of items to return
            
        Returns:
            List of memory dictionaries with key, value, and metadata
        """
        return self.working_memory.get_context(max_items)
    
    def _clean_expired_working_memory(self):
        """Remove expired entries from working memory."""
        self.working_memory._clean_expired()
    
    def consolidate_memories(self):
        """Consolidate important working memory items into long-term memory"""
        self.consolidator.consolidate_working_memory(self.working_memory)
    
    def _should_consolidate(self, key: str, value: Any) -> bool:
        """Determine if a memory should be consolidated"""
        return self.consolidator._should_consolidate(key, {"value": value})
    
    def force_reindex(self):
        """
        Force immediate reindexing of the RAG vector store.
        
        Returns:
            Number of pending experiences that were indexed
        """
        try:
            pending_count = len(self._pending_experiences)
            if pending_count > 0:
                logger.info(f"Force reindexing with {pending_count} pending experiences...")
                self.vector_db.index()
                self._pending_experiences = []
                self._last_index_time = datetime.now()
                logger.info("Reindexing complete")
                return pending_count
            else:
                logger.info("No pending experiences to index")
                return 0
        except Exception as e:
            logger.error(f"Force reindex failed: {e}", exc_info=True)
            raise
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the memory system.

        Returns:
            Dictionary containing memory counts and system stats
        """
        try:
            counts = self.storage.get_collection_counts()
            stats = {
                "episodic_count": counts["episodic"],
                "semantic_count": counts["semantic"],
                "procedural_count": counts["procedural"],
                "working_memory_size": self.working_memory.size(),
                "pending_indexing": len(self._pending_experiences),
                "last_index_time": self._last_index_time.isoformat(),
                "index_batch_size": self._index_batch_size,
                "total_memories": sum(counts.values()),
                "blockchain_blocks": self.storage.get_blockchain_count()
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}

    def close(self) -> None:
        """
        Close the memory system and release resources.

        This is important on Windows to release ChromaDB file locks.
        """
        try:
            # Close vector database first (it also uses ChromaDB)
            if hasattr(self, 'vector_db') and self.vector_db is not None:
                self.vector_db.close()
            # Then close storage
            if hasattr(self, 'storage') and self.storage is not None:
                self.storage.close()
            logger.info("Memory manager closed")
        except Exception as e:
            logger.error(f"Error closing memory manager: {e}")
