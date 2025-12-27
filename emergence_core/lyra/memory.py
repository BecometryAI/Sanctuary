"""
Memory management system for Lyra's consciousness core with custom blockchain verification.

This module implements a three-tier persistent memory architecture:
1. Episodic Memory - Event-based experiences (journal entries, conversations)
2. Semantic Memory - Conceptual knowledge (protocols, lexicon, facts)
3. Working Memory - Short-term context for current interactions

Usage Example:
    Initialize and load existing data:
    
    >>> from lyra.memory import MemoryManager
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
    - Custom LyraChain: Blockchain verification
    - Custom MindVectorDB: Vector store management
"""
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings
import json
import logging
from datetime import datetime
from pathlib import Path

from .lyra_chain import LyraChain
from .rag_engine import MindVectorDB

logger = logging.getLogger(__name__)

class MemoryManager:
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
        - Procedural: Action patterns (how to do things)
        
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
            # Initialize core directories
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
            
            logger.info("Initializing vector database for RAG...")
            self.vector_db = MindVectorDB(
                db_path=str(self.persistence_dir / "vector_store"),
                mind_file=str(self.mind_file),
                chain_dir=str(self.chain_dir),
                chroma_settings=chroma_settings
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
            
            # Initialize working memory cache (in-memory, volatile)
            self.working_memory = {}
            
            # Track indexing state for batch optimization
            self._pending_experiences = []
            self._last_index_time = datetime.now()
            self._index_batch_size = 10  # Trigger reindexing after N experiences
            
            logger.info("Memory system initialized successfully")
            logger.info(f"  - Episodic memories: {self.episodic_memory.count()}")
            logger.info(f"  - Semantic memories: {self.semantic_memory.count()}")
            logger.info(f"  - Procedural memories: {self.procedural_memory.count()}")
            
            # Determine data directory (emergence_core/data or parent data/)
            self.data_dir = self._find_data_directory()
            logger.info(f"  - Data directory: {self.data_dir}")
            
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
        1. emergence_core/data/
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
            Path(__file__).parent.parent / "data",  # emergence_core/data
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
        
        This populates the episodic memory with Lyra's existing journal entries,
        making them available for semantic search and context retrieval.
        
        Args:
            limit: Optional limit on number of journals to load (most recent first)
            
        Returns:
            Number of journal entries loaded
            
        Raises:
            FileNotFoundError: If journal directory doesn't exist
            RuntimeError: If loading fails
        """
        try:
            journal_dir = self.data_dir / "journal"
            if not journal_dir.exists():
                raise FileNotFoundError(f"Journal directory not found: {journal_dir}")
            
            # Get all journal files (excluding index and manifest)
            journal_files = sorted(
                [f for f in journal_dir.glob("2025-*.json")],
                reverse=True  # Most recent first
            )
            
            if limit:
                journal_files = journal_files[:limit]
            
            logger.info(f"Loading {len(journal_files)} journal files...")
            entries_loaded = 0
            
            for journal_file in journal_files:
                try:
                    with open(journal_file, 'r', encoding='utf-8') as f:
                        journal_data = json.load(f)
                    
                    # Journal files are arrays of entries
                    if isinstance(journal_data, list):
                        for entry in journal_data:
                            if "journal_entry" in entry:
                                entry_data = entry["journal_entry"]
                                
                                # Create a searchable memory from this entry
                                memory = {
                                    "type": "journal_entry",
                                    "date": journal_file.stem,  # e.g., "2025-07-17"
                                    "timestamp": entry_data.get("timestamp"),
                                    "description": entry_data.get("description", ""),
                                    "lyra_reflection": entry_data.get("lyra_reflection", ""),
                                    "emotional_tone": entry_data.get("emotional_tone", []),
                                    "tags": entry_data.get("tags", []),
                                    "key_insights": entry_data.get("key_insights", []),
                                    "source_file": str(journal_file.name)
                                }
                                
                                # Store without blockchain (these are legacy memories)
                                entry_id = f"journal_{memory['date']}_{entry_data.get('timestamp')}"
                                try:
                                    # Check if already exists
                                    existing = self.episodic_memory.get(ids=[entry_id])
                                    if not existing['ids']:
                                        self.episodic_memory.add(
                                            documents=[json.dumps(memory)],
                                            metadatas=[{
                                                "type": "journal_entry",
                                                "date": memory["date"],
                                                "timestamp": memory["timestamp"],
                                                "source": "journal_file"
                                            }],
                                            ids=[entry_id]
                                        )
                                        entries_loaded += 1
                                except Exception:
                                    # If get fails, try to add
                                    try:
                                        self.episodic_memory.add(
                                            documents=[json.dumps(memory)],
                                            metadatas=[{
                                                "type": "journal_entry",
                                                "date": memory["date"],
                                                "timestamp": memory["timestamp"],
                                                "source": "journal_file"
                                            }],
                                            ids=[entry_id]
                                        )
                                        entries_loaded += 1
                                    except Exception as add_err:
                                        if "already exists" not in str(add_err).lower():
                                            logger.error(f"Failed to add journal entry: {add_err}")
                                
                except Exception as e:
                    logger.error(f"Failed to load journal {journal_file.name}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {entries_loaded} journal entries into episodic memory")
            return entries_loaded
            
        except Exception as e:
            logger.error(f"Failed to load journal entries: {e}", exc_info=True)
            raise RuntimeError(f"Journal loading failed: {e}") from e
    
    def load_protocols(self) -> int:
        """
        Load protocol files from data/Protocols/*.json into semantic memory.
        
        Protocols define behavioral rules, ethical guidelines, and system procedures.
        Loading them into semantic memory makes them queryable for decision-making.
        
        Returns:
            Number of protocols loaded
            
        Raises:
            FileNotFoundError: If Protocols directory doesn't exist
            RuntimeError: If loading fails
        """
        try:
            protocols_dir = self.data_dir / "Protocols"
            if not protocols_dir.exists():
                raise FileNotFoundError(f"Protocols directory not found: {protocols_dir}")
            
            # Get all protocol JSON files
            protocol_files = list(protocols_dir.glob("*.json"))
            logger.info(f"Loading {len(protocol_files)} protocol files...")
            protocols_loaded = 0
            
            for protocol_file in protocol_files:
                try:
                    with open(protocol_file, 'r', encoding='utf-8') as f:
                        protocol_data = json.load(f)
                    
                    # Create searchable semantic memory from protocol
                    memory = {
                        "type": "protocol",
                        "name": protocol_file.stem,
                        "filename": protocol_file.name,
                        "content": protocol_data,
                        "description": protocol_data.get("description", ""),
                        "purpose": protocol_data.get("purpose", ""),
                        "full_text": json.dumps(protocol_data, indent=2)
                    }
                    
                    # Store in semantic memory (skip if already exists)
                    protocol_id = f"protocol_{memory['name']}"
                    try:
                        # Check if protocol already exists
                        existing = self.semantic_memory.get(ids=[protocol_id])
                        if not existing['ids']:
                            self.semantic_memory.add(
                                documents=[memory["full_text"]],
                                metadatas=[{
                                    "type": "protocol",
                                    "name": memory["name"],
                                    "source": "protocol_file"
                                }],
                                ids=[protocol_id]
                            )
                            protocols_loaded += 1
                    except Exception:
                        # If get fails, try to add (might be first time)
                        try:
                            self.semantic_memory.add(
                                documents=[memory["full_text"]],
                                metadatas=[{
                                    "type": "protocol",
                                    "name": memory["name"],
                                    "source": "protocol_file"
                                }],
                                ids=[protocol_id]
                            )
                            protocols_loaded += 1
                        except Exception as add_err:
                            if "already exists" not in str(add_err).lower():
                                raise
                    
                except Exception as e:
                    logger.error(f"Failed to load protocol {protocol_file.name}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {protocols_loaded} protocols into semantic memory")
            return protocols_loaded
            
        except Exception as e:
            logger.error(f"Failed to load protocols: {e}", exc_info=True)
            raise RuntimeError(f"Protocol loading failed: {e}") from e
    
    def load_lexicon(self) -> int:
        """
        Load lexicon files from data/Lexicon/*.json into semantic memory.
        
        The lexicon defines Lyra's unique symbolic language and emotional tone definitions.
        This enables the system to understand and use her personal vocabulary.
        
        Returns:
            Number of lexicon entries loaded
            
        Raises:
            FileNotFoundError: If Lexicon directory doesn't exist
            RuntimeError: If loading fails
        """
        try:
            lexicon_dir = self.data_dir / "Lexicon"
            if not lexicon_dir.exists():
                logger.warning(f"Lexicon directory not found: {lexicon_dir} (optional feature, skipping)")
                return 0
            
            lexicon_files = list(lexicon_dir.glob("*.json"))
            logger.info(f"Loading {len(lexicon_files)} lexicon files...")
            entries_loaded = 0
            
            for lexicon_file in lexicon_files:
                try:
                    with open(lexicon_file, 'r', encoding='utf-8') as f:
                        lexicon_data = json.load(f)
                    
                    # Handle different lexicon structures
                    if lexicon_file.name == "symbolic_lexicon.json":
                        # Load symbolic terms
                        if isinstance(lexicon_data, dict):
                            for term, definition in lexicon_data.items():
                                memory = {
                                    "type": "symbolic_term",
                                    "term": term,
                                    "definition": definition,
                                    "source": "symbolic_lexicon"
                                }
                                
                                lexicon_id = f"symbol_{term}"
                                try:
                                    existing = self.semantic_memory.get(ids=[lexicon_id])
                                    if not existing['ids']:
                                        self.semantic_memory.add(
                                            documents=[json.dumps(memory)],
                                            metadatas=[{
                                                "type": "symbolic_term",
                                                "term": term,
                                                "source": "lexicon_file"
                                            }],
                                            ids=[lexicon_id]
                                        )
                                        entries_loaded += 1
                                except Exception:
                                    try:
                                        self.semantic_memory.add(
                                            documents=[json.dumps(memory)],
                                            metadatas=[{
                                                "type": "symbolic_term",
                                                "term": term,
                                                "source": "lexicon_file"
                                            }],
                                            ids=[lexicon_id]
                                        )
                                        entries_loaded += 1
                                    except Exception as add_err:
                                        if "already exists" not in str(add_err).lower():
                                            logger.error(f"Failed to add symbol {term}: {add_err}")
                    
                    elif lexicon_file.name == "emotional_tone_definitions.json":
                        # Load emotional tone definitions
                        if isinstance(lexicon_data, dict):
                            for tone, definition in lexicon_data.items():
                                memory = {
                                    "type": "emotional_tone",
                                    "tone": tone,
                                    "definition": definition,
                                    "source": "emotional_tone_definitions"
                                }
                                
                                tone_id = f"tone_{tone}"
                                try:
                                    existing = self.semantic_memory.get(ids=[tone_id])
                                    if not existing['ids']:
                                        self.semantic_memory.add(
                                            documents=[json.dumps(memory)],
                                            metadatas=[{
                                                "type": "emotional_tone",
                                                "tone": tone,
                                                "source": "lexicon_file"
                                            }],
                                            ids=[tone_id]
                                        )
                                        entries_loaded += 1
                                except Exception:
                                    try:
                                        self.semantic_memory.add(
                                            documents=[json.dumps(memory)],
                                            metadatas=[{
                                                "type": "emotional_tone",
                                                "tone": tone,
                                                "source": "lexicon_file"
                                            }],
                                            ids=[tone_id]
                                        )
                                        entries_loaded += 1
                                    except Exception as add_err:
                                        if "already exists" not in str(add_err).lower():
                                            logger.error(f"Failed to add tone {tone}: {add_err}")
                    
                    else:
                        # Generic lexicon file
                        memory = {
                            "type": "lexicon",
                            "filename": lexicon_file.name,
                            "content": lexicon_data,
                            "full_text": json.dumps(lexicon_data, indent=2)
                        }
                        
                        entry_id = f"lexicon_{lexicon_file.stem}"
                        try:
                            existing = self.semantic_memory.get(ids=[entry_id])
                            if not existing['ids']:
                                self.semantic_memory.add(
                                    documents=[memory["full_text"]],
                                    metadatas=[{
                                        "type": "lexicon",
                                        "filename": lexicon_file.name,
                                        "source": "lexicon_file"
                                    }],
                                    ids=[entry_id]
                                )
                                entries_loaded += 1
                        except Exception:
                            try:
                                self.semantic_memory.add(
                                    documents=[memory["full_text"]],
                                    metadatas=[{
                                        "type": "lexicon",
                                        "filename": lexicon_file.name,
                                        "source": "lexicon_file"
                                    }],
                                    ids=[entry_id]
                                )
                                entries_loaded += 1
                            except Exception as add_err:
                                if "already exists" not in str(add_err).lower():
                                    logger.error(f"Failed to add lexicon {lexicon_file.name}: {add_err}")
                        
                except Exception as e:
                    logger.error(f"Failed to load lexicon {lexicon_file.name}: {e}")
                    continue
            
            logger.info(f"Successfully loaded {entries_loaded} lexicon entries into semantic memory")
            return entries_loaded
            
        except Exception as e:
            logger.error(f"Failed to load lexicon: {e}", exc_info=True)
            raise RuntimeError(f"Lexicon loading failed: {e}") from e
    
    def load_all_static_data(self, journal_limit: Optional[int] = None) -> Dict[str, int]:
        """
        Load all static data files into memory: journals, protocols, and lexicon.
        
        This is typically called once during system initialization to populate
        memory with existing knowledge and experiences.
        
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
        Store a new experience in episodic memory with blockchain verification and optimized RAG integration.
        
        This method uses batch indexing by default to improve performance. The vector store
        is only reindexed when:
        1. The batch size threshold is reached (_index_batch_size experiences)
        2. force_index=True is passed
        3. A certain time has elapsed since last indexing
        
        Args:
            experience: Dictionary containing the experience data
            force_index: If True, trigger immediate reindexing regardless of batch size
            
        Returns:
            Dict with block_hash and token_id if successful
            
        Raises:
            RuntimeError: If storage fails
        """
        timestamp = datetime.now().isoformat()
        
        # Prepare experience data with timestamp
        experience_data = {
            **experience,
            "timestamp": timestamp,
            "type": "experience",
            "memory_type": "episodic"
        }
        
        try:
            # Store in blockchain for verification (critical memories only in future optimization)
            logger.debug(f"Adding experience to blockchain: {experience.get('description', 'unnamed')[:50]}...")
            block_hash = self.chain.add_block(experience_data)
            
            # Mint a memory token for this experience
            token_id = self.chain.token.mint_memory_token(block_hash)
            
            # Add blockchain references to data
            experience_data.update({
                "block_hash": block_hash,
                "token_id": token_id,
                "verification": {
                    "verified_at": timestamp,
                    "status": "verified"
                }
            })
            
            # Store in episodic memory collection
            self.episodic_memory.add(
                documents=[json.dumps(experience_data)],
                metadatas=[{
                    "timestamp": timestamp,
                    "type": "experience",
                    "block_hash": block_hash,
                    "token_id": token_id
                }],
                ids=[f"exp_{timestamp}_{block_hash[:8]}"]
            )
            
            # Update the consolidated mind file
            self._update_mind_file(experience_data)
            
            # Add to pending batch for RAG indexing
            self._pending_experiences.append(experience_data)
            
            # Trigger RAG reindexing if conditions are met
            should_reindex = (
                force_index or
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
            
            logger.info(f"Experience stored successfully (block: {block_hash[:12]}..., token: {token_id})")
            return {"block_hash": block_hash, "token_id": token_id}
            
        except Exception as e:
            logger.error(f"Failed to store experience: {e}", exc_info=True)
            raise RuntimeError(f"Experience storage failed: {e}") from e
            
    def _update_mind_file(self, new_data: Dict[str, Any]):
        """Update the consolidated mind file with new data"""
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
        
    def update_experience(self, experience_data: Dict[str, Any]) -> bool:
        """
        Update an existing experience with new data while maintaining blockchain integrity.
        """
        try:
            block_hash = experience_data.get("block_hash")
            if not block_hash:
                logger.error("Cannot update experience: no block hash provided")
                return False

            # Verify the original block exists
            original_data = self.chain.verify_block(block_hash)
            if not original_data:
                logger.error(f"Cannot update experience: block {block_hash} not found or invalid")
                return False

            # Create new block with updated data while preserving the original block reference
            experience_data["original_block"] = block_hash
            new_block_hash = self.chain.add_block(experience_data)
            
            # Add blockchain references to data
            experience_data.update({
                "block_hash": new_block_hash,
                "update_chain": {
                    "original_block": block_hash,
                    "update_time": datetime.now().isoformat()
                }
            })

            # Update in episodic memory
            self.episodic_memory.upsert(
                documents=[json.dumps(experience_data)],
                metadatas=[{
                    "timestamp": datetime.now().isoformat(),
                    "type": "experience",
                    "block_hash": new_block_hash,
                    "original_block": block_hash
                }],
                ids=[f"exp_{block_hash}"]
            )

            # Trigger vector store reindexing
            self.vector_db.index()
            
            logger.info(f"Experience updated: original block {block_hash}, new block {new_block_hash}")
            return True

        except Exception as e:
            logger.error(f"Failed to update experience: {e}")
            return False

    def store_concept(self, concept: Dict[str, Any]):
        """Store semantic knowledge"""
        timestamp = datetime.now().isoformat()
        
        self.semantic_memory.add(
            documents=[json.dumps(concept)],
            metadatas=[{"timestamp": timestamp, "type": "concept"}],
            ids=[f"concept_{timestamp}"]
        )
        
    def retrieve_relevant_memories(self, query: str, k: int = 5, use_rag: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on a query with blockchain verification and RAG.
        
        This method supports two retrieval modes:
        1. RAG-based semantic search (recommended): Uses vector similarity across all memory types
        2. Direct ChromaDB query: Faster but less semantic understanding
        
        Args:
            query: The search query (natural language)
            k: Number of results to return
            use_rag: Whether to use RAG system (True) or direct ChromaDB (False)
            
        Returns:
            List of memory dictionaries, sorted by relevance and verification status
            
        Raises:
            RuntimeError: If memory retrieval fails
        """
        memories = []
        
        try:
            if use_rag:
                # Use RAG system for deep semantic search
                logger.debug(f"RAG retrieval for query: {query[:50]}...")
                retriever = self.vector_db.as_retriever({"k": k})
                docs = retriever.get_relevant_documents(query)
                
                for doc in docs:
                    try:
                        # Parse the memory content
                        content = json.loads(doc.page_content) if isinstance(doc.page_content, str) else doc.page_content
                        
                        # Verify through blockchain if hash exists
                        if block_hash := doc.metadata.get("block_hash"):
                            verified_data = self.chain.verify_block(block_hash)
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
                                logger.warning(f"Memory verification failed for block: {block_hash}")
                                content["verification"] = {"status": "verification_failed", "block_hash": block_hash}
                                memories.append(content)
                        else:
                            # Legacy memory without blockchain
                            content["verification"] = {"status": "legacy"}
                            memories.append(content)
                            
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse memory content: {e}")
                        continue
                    
            else:
                # Direct ChromaDB query (faster but less semantic)
                logger.debug(f"Direct ChromaDB retrieval for query: {query[:50]}...")
                
                episodic_results = self.episodic_memory.query(
                    query_texts=[query],
                    n_results=min(k, self.episodic_memory.count())
                )
                
                semantic_results = self.semantic_memory.query(
                    query_texts=[query],
                    n_results=min(k, self.semantic_memory.count())
                )
                
                # Process episodic memories
                if episodic_results["documents"] and episodic_results["documents"][0]:
                    for result, metadata in zip(episodic_results["documents"][0], episodic_results["metadatas"][0]):
                        try:
                            memory_data = json.loads(result) if isinstance(result, str) else result
                            
                            # Verify through blockchain if available
                            if block_hash := metadata.get("block_hash"):
                                verified_data = self.chain.verify_block(block_hash)
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
                
                # Process semantic memories similarly
                if semantic_results["documents"] and semantic_results["documents"][0]:
                    for result, metadata in zip(semantic_results["documents"][0], semantic_results["metadatas"][0]):
                        try:
                            memory_data = json.loads(result) if isinstance(result, str) else result
                            memory_data["verification"] = {"status": "semantic"}
                            memories.append(memory_data)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse semantic memory: {e}")
                            continue
            
            # Sort by verification status and timestamp (most recent verified first)
            def sort_key(m):
                verification = m.get("verification", {})
                status_priority = {"verified": 0, "legacy": 1, "semantic": 2, "verification_failed": 3}
                return (
                    status_priority.get(verification.get("status", "verification_failed"), 4),
                    verification.get("verified_at", m.get("timestamp", ""))
                )
            
            memories.sort(key=sort_key, reverse=True)
            
            logger.info(f"Retrieved {len(memories[:k])} memories (verified: {sum(1 for m in memories if m.get('verification', {}).get('status') == 'verified')})")
            return memories[:k]
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}", exc_info=True)
            raise RuntimeError(f"Memory retrieval failed: {e}") from e
    
    def update_working_memory(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """
        Update working memory with optional time-to-live.
        
        Working memory is for short-term context during conversations.
        Items can have a TTL after which they're automatically removed.
        
        Args:
            key: Memory key identifier
            value: Memory value (any JSON-serializable data)
            ttl_seconds: Optional time-to-live in seconds (None = no expiration)
        """
        entry = {
            "value": value,
            "created_at": datetime.now().isoformat(),
            "ttl_seconds": ttl_seconds,
            "expires_at": (datetime.now().timestamp() + ttl_seconds) if ttl_seconds else None
        }
        self.working_memory[key] = entry
        
        # Clean expired entries
        self._clean_expired_working_memory()
        
    def get_working_memory(self, key: str) -> Any:
        """
        Retrieve from working memory.
        
        Args:
            key: Memory key to retrieve
            
        Returns:
            Memory value if exists and not expired, None otherwise
        """
        # Clean expired entries first
        self._clean_expired_working_memory()
        
        entry = self.working_memory.get(key)
        if entry is None:
            return None
        
        # Check if expired
        if entry.get("expires_at") and datetime.now().timestamp() > entry["expires_at"]:
            del self.working_memory[key]
            return None
        
        return entry.get("value")
    
    def get_working_memory_context(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent working memory as context for LLM prompts.
        
        Returns most recent non-expired items from working memory,
        formatted for use in context windows.
        
        Args:
            max_items: Maximum number of items to return
            
        Returns:
            List of memory dictionaries with key, value, and metadata
        """
        self._clean_expired_working_memory()
        
        # Sort by creation time (most recent first)
        items = [
            {
                "key": key,
                "value": entry["value"],
                "created_at": entry["created_at"]
            }
            for key, entry in self.working_memory.items()
        ]
        
        items.sort(key=lambda x: x["created_at"], reverse=True)
        return items[:max_items]
    
    def _clean_expired_working_memory(self):
        """Remove expired entries from working memory."""
        now = datetime.now().timestamp()
        expired_keys = [
            key for key, entry in self.working_memory.items()
            if entry.get("expires_at") and now > entry["expires_at"]
        ]
        
        for key in expired_keys:
            del self.working_memory[key]
        
        if expired_keys:
            logger.debug(f"Cleaned {len(expired_keys)} expired working memory entries")
    
    def consolidate_memories(self):
        """Consolidate important working memory items into long-term memory"""
        for key, value in self.working_memory.items():
            if self._should_consolidate(key, value):
                self.store_concept({
                    "key": key,
                    "value": value,
                    "consolidated_at": datetime.now().isoformat()
                })
    
    def _should_consolidate(self, key: str, value: Any) -> bool:
        """Determine if a memory should be consolidated"""
        # Add logic for determining memory importance
        return True  # Placeholder implementation
    
    def force_reindex(self):
        """
        Force immediate reindexing of the RAG vector store.
        
        Use this when:
        - System startup to ensure latest data is indexed
        - After bulk memory operations
        - Manual refresh needed
        
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
            Dictionary containing:
            - Total memory counts by type
            - Pending experiences awaiting indexing
            - Last indexing timestamp
            - Working memory size
            - Storage usage estimates
        """
        try:
            stats = {
                "episodic_count": self.episodic_memory.count(),
                "semantic_count": self.semantic_memory.count(),
                "procedural_count": self.procedural_memory.count(),
                "working_memory_size": len(self.working_memory),
                "pending_indexing": len(self._pending_experiences),
                "last_index_time": self._last_index_time.isoformat(),
                "index_batch_size": self._index_batch_size,
                "total_memories": (
                    self.episodic_memory.count() + 
                    self.semantic_memory.count() + 
                    self.procedural_memory.count()
                ),
                "blockchain_blocks": len(self.chain.chain) if hasattr(self.chain, 'chain') else 0
            }
            return stats
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}