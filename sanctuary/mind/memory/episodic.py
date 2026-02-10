"""
Episodic Memory Module

Manages autobiographical episodes with temporal and contextual indexing.
Handles the "what, when, where" of experiential memories.

Author: Sanctuary Team
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Manages autobiographical memory (events, interactions).
    
    Responsibilities:
    - Store experiential memories
    - Temporal indexing (when did this happen)
    - Context binding (where, who, what)
    - Load journal entries
    """
    
    def __init__(self, storage, encoder, data_dir: Optional[Path] = None):
        """
        Initialize episodic memory manager.
        
        Args:
            storage: MemoryStorage instance
            encoder: MemoryEncoder instance
            data_dir: Optional data directory for loading journal files
        """
        self.storage = storage
        self.encoder = encoder
        self.data_dir = data_dir
    
    def store_experience(
        self,
        experience: Dict[str, Any],
        use_blockchain: bool = True
    ) -> Optional[Dict[str, str]]:
        """
        Store a new experience in episodic memory.
        
        Args:
            experience: Experience data dictionary
            use_blockchain: Whether to use blockchain verification
            
        Returns:
            Dict with block_hash and token_id if blockchain used, None otherwise
        """
        try:
            block_hash = None
            token_id = None
            
            # Add to blockchain if requested
            if use_blockchain:
                logger.debug(f"Adding experience to blockchain: {experience.get('description', 'unnamed')[:50]}...")
                block_hash, token_id = self.storage.add_to_blockchain(experience)
            
            # Encode the experience
            document, metadata, doc_id = self.encoder.encode_experience(
                experience,
                block_hash=block_hash,
                token_id=token_id
            )
            
            # Store in episodic memory collection
            self.storage.add_episodic(document, metadata, doc_id)
            
            # Update mind file
            experience_data = json.loads(document)
            self.storage.update_mind_file(experience_data)
            
            logger.info(
                f"Experience stored successfully"
                f"{f' (block: {block_hash[:12]}..., token: {token_id})' if block_hash else ''}"
            )
            
            if block_hash and token_id:
                return {"block_hash": block_hash, "token_id": token_id}
            return None
            
        except Exception as e:
            logger.error(f"Failed to store experience: {e}", exc_info=True)
            raise RuntimeError(f"Experience storage failed: {e}") from e
    
    def update_experience(self, experience_data: Dict[str, Any]) -> bool:
        """
        Update an existing experience while maintaining blockchain integrity.
        
        Args:
            experience_data: Updated experience data with block_hash
            
        Returns:
            True if successful, False otherwise
        """
        from datetime import datetime
        
        try:
            block_hash = experience_data.get("block_hash")
            if not block_hash:
                logger.error("Cannot update experience: no block hash provided")
                return False

            # Verify the original block exists
            original_data = self.storage.verify_block(block_hash)
            if not original_data:
                logger.error(f"Cannot update experience: block {block_hash} not found or invalid")
                return False

            # Create new block with updated data
            experience_data["original_block"] = block_hash
            new_block_hash, _ = self.storage.add_to_blockchain(experience_data)
            
            # Add update chain metadata
            experience_data.update({
                "block_hash": new_block_hash,
                "update_chain": {
                    "original_block": block_hash,
                    "update_time": datetime.now().isoformat()
                }
            })

            # Update in episodic memory
            document = json.dumps(experience_data)
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "type": "experience",
                "block_hash": new_block_hash,
                "original_block": block_hash
            }
            doc_id = f"exp_{block_hash}"
            
            self.storage.update_episodic(document, metadata, doc_id)
            
            logger.info(f"Experience updated: original block {block_hash}, new block {new_block_hash}")
            return True

        except Exception as e:
            logger.error(f"Failed to update experience: {e}")
            return False
    
    def load_journal_entries(self, limit: Optional[int] = None) -> int:
        """
        Load journal entries from data/journal/*.json into episodic memory.
        
        Args:
            limit: Optional limit on number of journals to load (most recent first)
            
        Returns:
            Number of journal entries loaded
        """
        if not self.data_dir:
            logger.warning("No data directory specified, cannot load journals")
            return 0
        
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
                                
                                # Encode the journal entry
                                document, metadata, doc_id = self.encoder.encode_journal_entry(
                                    entry_data,
                                    date=journal_file.stem,
                                    source_file=journal_file.name
                                )
                                
                                # Check if already exists
                                try:
                                    existing = self.storage.get_episodic([doc_id])
                                    if not existing['ids']:
                                        self.storage.add_episodic(document, metadata, doc_id)
                                        entries_loaded += 1
                                except Exception:
                                    # If get fails, try to add
                                    try:
                                        self.storage.add_episodic(document, metadata, doc_id)
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
