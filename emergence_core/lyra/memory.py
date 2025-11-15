"""
Memory management system for Lyra's consciousness core with custom blockchain verification
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
    def __init__(self, persistence_dir: str = "memories", chain_dir: str = "chain", chroma_settings=None):
        """Initialize the memory management system with custom blockchain integration and RAG"""
        # Initialize core components
        self.persistence_dir = Path(persistence_dir)
        self.chain_dir = Path(chain_dir)
        # Initialize blockchain and RAG components
        self.chain = LyraChain(chain_dir)
        # Ensure mind state directory exists
        mind_state_dir = self.persistence_dir / "mind_state"
        mind_state_dir.mkdir(exist_ok=True, parents=True)
        mind_file = str(mind_state_dir / "core_mind.json")
        self.mind_file = Path(mind_file)
        if chroma_settings is None:
            chroma_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        print(f"[MemoryManager] chroma_settings id: {id(chroma_settings)}, contents: {chroma_settings}")
        self.vector_db = MindVectorDB(
            db_path=str(self.persistence_dir / "vector_store"),
            mind_file=mind_file,
            chain_dir=chain_dir,
            chroma_settings=chroma_settings
        )
        # Configure ChromaDB with appropriate settings
        settings = Settings(
            anonymized_telemetry=False,  # Disable telemetry
            allow_reset=True,  # Allow collection resets during testing
            is_persistent=True  # Enable persistence
        )
        # Create client with settings
        self.client = chromadb.PersistentClient(
            path=str(self.persistence_dir),
            settings=chroma_settings
        )
        try:
            # Create collections for different types of memories
            self.episodic_memory = self.client.get_or_create_collection(
                name="episodic_memory",
                metadata={"description": "Storage for experiential memories"}
            )
            self.semantic_memory = self.client.get_or_create_collection(
                name="semantic_memory",
                metadata={"description": "Storage for concept knowledge"}
            )
            self.procedural_memory = self.client.get_or_create_collection(
                name="procedural_memory",
                metadata={"description": "Storage for action patterns"}
            )
            # Working memory cache
            self.working_memory = {}
        except Exception as e:
            raise RuntimeError(f"Failed to initialize memory systems: {e}")
        """Initialize the memory management system with custom blockchain integration and RAG"""
        # Initialize core components
        self.persistence_dir = Path(persistence_dir)
        self.chain_dir = Path(chain_dir)
        
        # Initialize blockchain and RAG components
        self.chain = LyraChain(chain_dir)
        
        # Ensure mind state directory exists
        mind_state_dir = self.persistence_dir / "mind_state"
        mind_state_dir.mkdir(exist_ok=True, parents=True)
        mind_file = str(mind_state_dir / "core_mind.json")
        self.mind_file = Path(mind_file)
        
        if chroma_settings is None:
            chroma_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        self.vector_db = MindVectorDB(
            db_path=str(self.persistence_dir / "vector_store"),
            mind_file=mind_file,
            chain_dir=chain_dir,
            chroma_settings=chroma_settings
        )
        
        # Configure ChromaDB with appropriate settings
        settings = Settings(
            anonymized_telemetry=False,  # Disable telemetry
            allow_reset=True,  # Allow collection resets during testing
            is_persistent=True  # Enable persistence
        )
        
        # Create client with settings
        self.client = chromadb.PersistentClient(
            path=str(self.persistence_dir),
            settings=chroma_settings
        )
        
        try:
            # Create collections for different types of memories
            self.episodic_memory = self.client.get_or_create_collection(
                name="episodic_memory",
                metadata={"description": "Storage for experiential memories"}
            )
            self.semantic_memory = self.client.get_or_create_collection(
                name="semantic_memory",
                metadata={"description": "Storage for concept knowledge"}
            )
            self.procedural_memory = self.client.get_or_create_collection(
                name="procedural_memory",
                metadata={"description": "Storage for action patterns"}
            )
            
            # Working memory cache
            self.working_memory = {}
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize memory systems: {e}")
        
    def store_experience(self, experience: Dict[str, Any]):
        """Store a new experience in episodic memory with blockchain verification and RAG integration"""
        timestamp = datetime.now().isoformat()
        
        # Prepare experience data
        experience_data = {
            **experience,
            "timestamp": timestamp,
            "type": "memory"
        }
        
        try:
            # Store in Lyra's blockchain first
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
            
            # Store in episodic memory with blockchain reference
            self.episodic_memory.add(
                documents=[json.dumps(experience_data)],
                metadatas=[{
                    "timestamp": timestamp,
                    "type": "experience",
                    "block_hash": block_hash,
                    "token_id": token_id
                }],
                ids=[f"exp_{timestamp}"]
            )
            
            # Update the consolidated mind file
            self._update_mind_file(experience_data)
            
            # Trigger RAG reindexing
            self.vector_db.index()
            
            logger.info(f"Experience stored with block hash: {block_hash} and token: {token_id}")
            
        except Exception as e:
            logger.error(f"Failed to store experience: {e}")
            raise
            
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
        
        Args:
            query: The search query
            k: Number of results to return
            use_rag: Whether to use the RAG system for semantic search (True) or direct ChromaDB query (False)
        """
        memories = []
        
        try:
            if use_rag:
                # Use RAG system for semantic search
                retriever = self.vector_db.as_retriever({"k": k})
                docs = retriever.get_relevant_documents(query)
                
                for doc in docs:
                    # Parse the content and verify through blockchain
                    content = json.loads(doc.page_content)
                    if block_hash := doc.metadata.get("block_hash"):
                        verified_data = self.chain.verify_block(block_hash)
                        if verified_data:
                            verified_data["verification"] = {
                                "block_hash": block_hash,
                                "token_id": doc.metadata.get("token_id"),
                                "verified_at": datetime.now().isoformat(),
                                "rag_score": doc.metadata.get("score", 0.0)
                            }
                            memories.append(verified_data)
                        else:
                            logger.warning(f"Memory verification failed for block: {block_hash}")
                    else:
                        content["verification"] = {"status": "legacy"}
                        memories.append(content)
            else:
                # Direct ChromaDB query
                episodic_results = self.episodic_memory.query(
                    query_texts=[query],
                    n_results=k
                )
                
                semantic_results = self.semantic_memory.query(
                    query_texts=[query],
                    n_results=k
                )
                
                # Process episodic memories
                for result, metadata in zip(episodic_results["documents"][0], episodic_results["metadatas"][0]):
                    memory_data = json.loads(result)
                    if block_hash := metadata.get("block_hash"):
                        verified_data = self.chain.verify_block(block_hash)
                        if verified_data:
                            verified_data["verification"] = {
                                "block_hash": block_hash,
                                "token_id": self.chain.token.memory_tokens.get(block_hash),
                                "verified_at": datetime.now().isoformat()
                            }
                            memories.append(verified_data)
                        else:
                            logger.warning(f"Memory verification failed for block: {block_hash}")
                    else:
                        memory_data["verification"] = {"status": "legacy"}
                        memories.append(memory_data)
            
            # Sort by verification timestamp
            memories.sort(key=lambda x: x.get("verification", {}).get("verified_at", ""), reverse=True)
            
            return memories[:k]
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories: {e}")
            raise

        verified_data = {
            "verified_at": datetime.now().isoformat()
        }
        memories.append(verified_data)
        
        if not verify_result.get('verified'):
            logger.warning(f"Memory verification failed for block: {block_hash}")
        else:
                memory_data["verification"] = {"status": "legacy"}
                memories.append(memory_data)
                
        return memories
    
    def update_working_memory(self, key: str, value: Any):
        """Update working memory"""
        self.working_memory[key] = value
        
    def get_working_memory(self, key: str) -> Any:
        """Retrieve from working memory"""
        return self.working_memory.get(key)
    
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