"""
Memory Consolidation Module

Handles memory strengthening, decay, and reorganization.
Transfers memories from episodic to semantic storage.
Implements biologically-inspired memory consolidation that runs during idle periods.

Author: Sanctuary Team
"""
import json
import logging
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Pattern extraction constants
MAX_PATTERN_TAGS = 3  # Maximum tags to consider for pattern identification
MAX_SOURCE_EPISODES = 10  # Maximum source episodes to track per pattern
PATTERN_CONFIDENCE_DIVISOR = 10.0  # Divisor for confidence calculation
PATTERN_EXTRACTION_WEAKENING_FACTOR = 0.8  # Factor to weaken episodes after pattern extraction


class MemoryConsolidator:
    """
    Manages memory consolidation processes.
    
    Responsibilities:
    - Memory strengthening based on retrieval frequency
    - Decay for unused memories
    - Sleep-like reorganization/compression
    - Transfer from episodic to semantic
    - Association reorganization
    - Emotional memory reprocessing
    
    Attributes:
        storage: MemoryStorage instance
        encoder: MemoryEncoder instance
        strengthening_factor: Boost per retrieval (default: 0.1)
        decay_rate: Daily decay rate (default: 0.95)
        deletion_threshold: Activation below this triggers deletion (default: 0.1)
        pattern_threshold: Episodes needed for semantic transfer (default: 3)
        association_boost: Strength increase for co-retrieval (default: 0.1)
        emotion_threshold: Emotional intensity for special processing (default: 0.7)
        retrieval_log: In-memory log of recent retrievals
    """
    
    def __init__(
        self,
        storage,
        encoder,
        strengthening_factor: float = 0.1,
        decay_rate: float = 0.95,
        deletion_threshold: float = 0.1,
        pattern_threshold: int = 3,
        association_boost: float = 0.1,
        emotion_threshold: float = 0.7,
        max_retrieval_log_size: int = 1000,
        min_retrieval_count_for_consolidation: int = 2,
        min_age_hours_for_consolidation: float = 1.0
    ):
        """
        Initialize memory consolidator.
        
        Args:
            storage: MemoryStorage instance
            encoder: MemoryEncoder instance
            strengthening_factor: Boost per retrieval (0.0-1.0)
            decay_rate: Daily decay multiplier (0.0-1.0)
            deletion_threshold: Min activation to keep (0.0-1.0)
            pattern_threshold: Episodes for semantic transfer (>= 2)
            association_boost: Co-retrieval association boost (0.0-1.0)
            emotion_threshold: Emotional intensity threshold (0.0-1.0)
            max_retrieval_log_size: Maximum entries in retrieval log (> 0)
            min_retrieval_count_for_consolidation: Minimum retrievals before consolidating (>= 1)
            min_age_hours_for_consolidation: Minimum age in hours before consolidating (> 0)
        """
        # Input validation
        if not (0.0 <= strengthening_factor <= 1.0):
            raise ValueError(f"strengthening_factor must be in [0.0, 1.0], got {strengthening_factor}")
        if not (0.0 <= decay_rate <= 1.0):
            raise ValueError(f"decay_rate must be in [0.0, 1.0], got {decay_rate}")
        if not (0.0 <= deletion_threshold <= 1.0):
            raise ValueError(f"deletion_threshold must be in [0.0, 1.0], got {deletion_threshold}")
        if pattern_threshold < 2:
            raise ValueError(f"pattern_threshold must be >= 2, got {pattern_threshold}")
        if not (0.0 <= association_boost <= 1.0):
            raise ValueError(f"association_boost must be in [0.0, 1.0], got {association_boost}")
        if not (0.0 <= emotion_threshold <= 1.0):
            raise ValueError(f"emotion_threshold must be in [0.0, 1.0], got {emotion_threshold}")
        if max_retrieval_log_size <= 0:
            raise ValueError(f"max_retrieval_log_size must be > 0, got {max_retrieval_log_size}")
        if min_retrieval_count_for_consolidation < 1:
            raise ValueError(f"min_retrieval_count_for_consolidation must be >= 1, got {min_retrieval_count_for_consolidation}")
        if min_age_hours_for_consolidation <= 0:
            raise ValueError(f"min_age_hours_for_consolidation must be > 0, got {min_age_hours_for_consolidation}")
        
        self.storage = storage
        self.encoder = encoder
        self.strengthening_factor = strengthening_factor
        self.decay_rate = decay_rate
        self.deletion_threshold = deletion_threshold
        self.pattern_threshold = pattern_threshold
        self.association_boost = association_boost
        self.emotion_threshold = emotion_threshold
        self.max_retrieval_log_size = max_retrieval_log_size
        self.min_retrieval_count = min_retrieval_count_for_consolidation
        self.min_age_hours = min_age_hours_for_consolidation
        
        # In-memory retrieval log (id, timestamp, session_id)
        self.retrieval_log: List[Dict[str, Any]] = []
        
        # Association tracking (source_id -> target_id -> strength)
        self.associations: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        logger.info(
            f"MemoryConsolidator initialized "
            f"(strengthen={strengthening_factor}, decay={decay_rate}, "
            f"threshold={deletion_threshold})"
        )
    
    def record_retrieval(
        self,
        memory_id: str,
        session_id: Optional[str] = None
    ) -> None:
        """
        Record that a memory was retrieved.
        
        Args:
            memory_id: ID of retrieved memory
            session_id: Optional session identifier for co-retrieval tracking
        """
        self.retrieval_log.append({
            "memory_id": memory_id,
            "timestamp": datetime.now(),
            "session_id": session_id or "default"
        })
        
        # Keep log bounded with configurable size
        if len(self.retrieval_log) > self.max_retrieval_log_size:
            self.retrieval_log = self.retrieval_log[-self.max_retrieval_log_size:]
        
        logger.debug(f"Recorded retrieval: {memory_id}")
    
    def strengthen_retrieved_memories(self, hours: int = 24) -> int:
        """
        Strengthen memories based on recent retrieval history.
        
        Uses logarithmic strengthening with diminishing returns.
        Batches updates for efficiency.
        
        Args:
            hours: Look back window in hours (default: 24)
            
        Returns:
            Number of memories strengthened
        """
        if not self.retrieval_log:
            logger.debug("No retrieval log, skipping strengthening")
            return 0
        
        # Get recent retrievals
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [r for r in self.retrieval_log if r["timestamp"] > cutoff]
        
        if not recent:
            logger.debug("No recent retrievals in window")
            return 0
        
        # Count retrievals per memory
        retrieval_counts = Counter(r["memory_id"] for r in recent)
        
        # Batch fetch memories to update
        memory_ids = list(retrieval_counts.keys())
        try:
            result = self.storage.episodic_memory.get(ids=memory_ids)
            if not result or not result.get("ids"):
                logger.warning("No memories found for strengthening")
                return 0
        except Exception as e:
            logger.error(f"Failed to fetch memories for strengthening: {e}")
            return 0
        
        # Prepare batch updates
        strengthened = 0
        documents_to_update = []
        metadatas_to_update = []
        ids_to_update = []
        
        for idx, (mem_id, document, metadata) in enumerate(zip(
            result["ids"], result["documents"], result["metadatas"]
        )):
            count = retrieval_counts.get(mem_id, 0)
            if count == 0:
                continue
            
            try:
                # Logarithmic strengthening (diminishing returns)
                strength_boost = math.log(1 + count) * self.strengthening_factor
                
                # Update metadata
                current_activation = float(metadata.get("base_activation", 1.0))
                metadata["base_activation"] = min(1.0, current_activation + strength_boost)
                metadata["last_accessed"] = datetime.now().isoformat()
                metadata["consolidation_count"] = metadata.get("consolidation_count", 0) + 1
                
                # Add to batch
                documents_to_update.append(document)
                metadatas_to_update.append(metadata)
                ids_to_update.append(mem_id)
                strengthened += 1
                
                logger.debug(f"Strengthened {mem_id}: {count}x retrievals, boost={strength_boost:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to prepare update for {mem_id}: {e}")
        
        # Batch update storage
        if ids_to_update:
            try:
                self.storage.episodic_memory.upsert(
                    documents=documents_to_update,
                    metadatas=metadatas_to_update,
                    ids=ids_to_update
                )
                logger.info(f"Batch strengthened {strengthened} memories")
            except Exception as e:
                logger.error(f"Failed to batch update memories: {e}")
                return 0
        
        return strengthened
    
    def apply_decay(self, threshold_days: int = 7) -> Tuple[int, int]:
        """
        Apply decay to unretrieved memories.
        
        Memories that haven't been retrieved decay exponentially.
        Very weak memories are marked for deletion.
        
        Args:
            threshold_days: Days since last access before starting decay
            
        Returns:
            Tuple of (memories_decayed, memories_pruned)
        """
        decayed = 0
        to_prune = []
        now = datetime.now()
        
        try:
            # Get all episodic memories
            all_episodic = self.storage.episodic_memory.get()
            
            if not all_episodic or not all_episodic.get("ids"):
                logger.debug("No episodic memories to decay")
                return 0, 0
            
            for mem_id, metadata in zip(
                all_episodic.get("ids", []),
                all_episodic.get("metadatas", [])
            ):
                try:
                    # Get last accessed time
                    last_accessed_str = metadata.get("last_accessed")
                    if not last_accessed_str:
                        # No access record, use creation time or now
                        last_accessed_str = metadata.get("timestamp", now.isoformat())
                    
                    last_accessed = datetime.fromisoformat(last_accessed_str)
                    days_since_access = (now - last_accessed).days
                    
                    if days_since_access > threshold_days:
                        # Apply exponential decay
                        base_activation = float(metadata.get("base_activation", 1.0))
                        decay_factor = self.decay_rate ** (days_since_access - threshold_days)
                        new_activation = base_activation * decay_factor
                        
                        # Update activation
                        metadata["base_activation"] = new_activation
                        
                        # Mark for deletion if too weak
                        if new_activation < self.deletion_threshold:
                            to_prune.append(mem_id)
                        else:
                            # Update metadata
                            self._update_memory_metadata(mem_id, metadata)
                        
                        decayed += 1
                        
                        logger.debug(
                            f"Decayed {mem_id}: "
                            f"{days_since_access}d since access, "
                            f"activation: {base_activation:.3f} -> {new_activation:.3f}"
                        )
                
                except Exception as e:
                    logger.error(f"Failed to decay memory {mem_id}: {e}")
                    continue
            
            # Prune very weak memories
            pruned = self._prune_memories(to_prune)
            
            logger.info(
                f"Decay complete: {decayed} memories decayed, {pruned} pruned"
            )
            return decayed, pruned
            
        except Exception as e:
            logger.error(f"Error during decay: {e}", exc_info=True)
            return 0, 0
    
    def transfer_to_semantic(self, days: int = 30, threshold: int = None) -> int:
        """
        Transfer repeated episodic patterns to semantic memory.
        
        Identifies patterns that repeat across episodes and extracts
        them as semantic knowledge.
        
        Args:
            days: Look back window for episodes
            threshold: Pattern repetition threshold (uses self.pattern_threshold if None)
            
        Returns:
            Number of patterns extracted
        """
        if threshold is None:
            threshold = self.pattern_threshold
        
        try:
            # Get recent episodic memories
            cutoff = datetime.now() - timedelta(days=days)
            episodes = self._get_recent_episodes(cutoff)
            
            if len(episodes) < threshold:
                logger.debug(f"Not enough episodes for pattern extraction: {len(episodes)}")
                return 0
            
            # Extract patterns (simplified: group by tags/content similarity)
            patterns = self._extract_patterns(episodes)
            
            transferred = 0
            for pattern_key, occurrences in patterns.items():
                if len(occurrences) >= threshold:
                    # Create semantic memory from pattern
                    semantic_data = self._create_semantic_from_pattern(
                        pattern_key,
                        occurrences,
                        episodes
                    )
                    
                    # Store as semantic knowledge
                    try:
                        document, metadata, doc_id = self.encoder.encode_concept(
                            semantic_data
                        )
                        self.storage.add_semantic(document, metadata, doc_id)
                        
                        # Weaken original episodes (gist extracted)
                        self._weaken_episodes(occurrences)
                        
                        transferred += 1
                        logger.info(
                            f"Transferred pattern to semantic: {pattern_key} "
                            f"({len(occurrences)} occurrences)"
                        )
                        
                    except Exception as e:
                        logger.error(f"Failed to store semantic pattern: {e}")
                        continue
            
            logger.info(f"Transferred {transferred} patterns to semantic memory")
            return transferred
            
        except Exception as e:
            logger.error(f"Error during semantic transfer: {e}", exc_info=True)
            return 0
    
    def reorganize_associations(self, hours: int = 24) -> int:
        """
        Reorganize memory associations based on co-retrieval patterns.
        
        Strengthens associations between memories retrieved together,
        and decays weak associations.
        
        Args:
            hours: Look back window for retrieval sessions
            
        Returns:
            Number of associations updated
        """
        if not self.retrieval_log:
            logger.debug("No retrieval log for association reorganization")
            return 0
        
        try:
            # Group retrievals by session
            cutoff = datetime.now() - timedelta(hours=hours)
            recent = [r for r in self.retrieval_log if r["timestamp"] > cutoff]
            
            # Group by session
            sessions = defaultdict(list)
            for r in recent:
                sessions[r["session_id"]].append(r["memory_id"])
            
            updated = 0
            
            # Strengthen co-retrieved associations
            for session_memories in sessions.values():
                if len(session_memories) < 2:
                    continue
                
                # Create associations for all pairs in session
                for i, mem1 in enumerate(session_memories):
                    for mem2 in session_memories[i+1:]:
                        self._strengthen_association(mem1, mem2)
                        updated += 1
            
            # Decay weak associations
            decayed = self._decay_weak_associations()
            
            logger.info(
                f"Association reorganization: {updated} strengthened, "
                f"{decayed} decayed"
            )
            return updated
            
        except Exception as e:
            logger.error(f"Error during association reorganization: {e}", exc_info=True)
            return 0
    
    def reprocess_emotional_memories(self, threshold: float = None) -> int:
        """
        Give extra consolidation to emotionally significant memories.
        
        High-emotion memories:
        - Resist decay
        - Form stronger associations
        - Get additional strengthening
        
        Args:
            threshold: Emotional intensity threshold (uses self.emotion_threshold if None)
            
        Returns:
            Number of emotional memories reprocessed
        """
        if threshold is None:
            threshold = self.emotion_threshold
        
        try:
            # Get memories with high emotional intensity
            emotional_memories = self._get_high_emotion_memories(threshold)
            
            if not emotional_memories:
                logger.debug("No high-emotion memories to reprocess")
                return 0
            
            reprocessed = 0
            
            for mem_id, metadata in emotional_memories:
                try:
                    emotional_intensity = float(
                        metadata.get("emotional_intensity", 0.5)
                    )
                    
                    # Increase decay resistance
                    current_activation = float(metadata.get("base_activation", 1.0))
                    bonus = emotional_intensity * 0.2  # Up to 20% bonus
                    new_activation = min(1.0, current_activation + bonus)
                    
                    metadata["base_activation"] = new_activation
                    metadata["decay_resistance"] = max(
                        metadata.get("decay_resistance", 0.0),
                        emotional_intensity * 0.5
                    )
                    
                    # Update metadata
                    self._update_memory_metadata(mem_id, metadata)
                    
                    reprocessed += 1
                    
                    logger.debug(
                        f"Reprocessed emotional memory {mem_id}: "
                        f"intensity={emotional_intensity:.2f}, "
                        f"activation boost={bonus:.3f}"
                    )
                    
                except Exception as e:
                    logger.error(f"Failed to reprocess emotional memory {mem_id}: {e}")
                    continue
            
            logger.info(f"Reprocessed {reprocessed} emotional memories")
            return reprocessed
            
        except Exception as e:
            logger.error(f"Error during emotional reprocessing: {e}", exc_info=True)
            return 0
    
    def consolidate_working_memory(self, working_memory) -> None:
        """
        Consolidate important working memory items into long-term memory.
        
        Args:
            working_memory: WorkingMemory instance
        """
        for key, entry in working_memory.memory.items():
            if self._should_consolidate(key, entry):
                concept_data = {
                    "key": key,
                    "value": entry.get("value"),
                    "consolidated_at": entry.get("created_at")
                }
                
                # Encode and store as semantic concept
                document, metadata, doc_id = self.encoder.encode_concept(concept_data)
                
                try:
                    self.storage.add_semantic(document, metadata, doc_id)
                    logger.info(f"Consolidated working memory key '{key}' to semantic memory")
                except Exception as e:
                    logger.error(f"Failed to consolidate memory key '{key}': {e}")
    
    def _should_consolidate(self, key: str, entry: Dict[str, Any]) -> bool:
        """
        Determine if a working memory entry should be consolidated.
        
        Args:
            key: Memory key
            entry: Memory entry data
            
        Returns:
            True if should consolidate, False otherwise
        """
        # Check retrieval frequency
        retrieval_count = entry.get("retrieval_count", 0)
        if retrieval_count < self.min_retrieval_count:
            return False
        
        # Check age (consolidate if older than configured minimum age)
        created_at_str = entry.get("created_at")
        if created_at_str:
            try:
                created_at = datetime.fromisoformat(created_at_str)
                age_hours = (datetime.now() - created_at).total_seconds() / 3600
                if age_hours < self.min_age_hours:
                    return False
            except (ValueError, TypeError):
                pass
        
        return True
    
    def _update_memory_activation(
        self,
        memory_id: str,
        boost: float
    ) -> None:
        """
        Update memory activation strength.
        
        Args:
            memory_id: Memory ID
            boost: Activation boost amount
        """
        try:
            # Get current memory data
            result = self.storage.episodic_memory.get(ids=[memory_id])
            
            if not result or not result.get("ids"):
                logger.warning(f"Memory {memory_id} not found for strengthening")
                return
            
            metadata = result["metadatas"][0]
            
            # Update activation
            current_activation = float(metadata.get("base_activation", 1.0))
            new_activation = min(1.0, current_activation + boost)
            
            metadata["base_activation"] = new_activation
            metadata["last_accessed"] = datetime.now().isoformat()
            metadata["consolidation_count"] = metadata.get("consolidation_count", 0) + 1
            
            # Update in storage
            self._update_memory_metadata(memory_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to update activation for {memory_id}: {e}")
    
    def _update_memory_metadata(
        self,
        memory_id: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Update memory metadata in storage.
        
        Args:
            memory_id: Memory ID
            metadata: Updated metadata
        """
        try:
            # Get current document
            result = self.storage.episodic_memory.get(ids=[memory_id])
            if not result or not result.get("ids"):
                return
            
            document = result["documents"][0]
            
            # Update using upsert
            self.storage.episodic_memory.upsert(
                documents=[document],
                metadatas=[metadata],
                ids=[memory_id]
            )
            
        except Exception as e:
            logger.error(f"Failed to update metadata for {memory_id}: {e}")
    
    def _prune_memories(self, memory_ids: List[str]) -> int:
        """
        Delete very weak memories.
        
        Args:
            memory_ids: List of memory IDs to prune
            
        Returns:
            Number of memories pruned
        """
        if not memory_ids:
            return 0
        
        pruned = 0
        for mem_id in memory_ids:
            try:
                self.storage.episodic_memory.delete(ids=[mem_id])
                pruned += 1
                logger.info(f"Pruned weak memory: {mem_id}")
            except Exception as e:
                logger.error(f"Failed to prune {mem_id}: {e}")
        
        return pruned
    
    def _get_recent_episodes(self, cutoff: datetime) -> List[Tuple[str, Dict]]:
        """
        Get episodic memories since cutoff time.
        
        Args:
            cutoff: Cutoff datetime
            
        Returns:
            List of (memory_id, metadata) tuples
        """
        episodes = []
        
        try:
            all_episodic = self.storage.episodic_memory.get()
            
            if not all_episodic or not all_episodic.get("ids"):
                return episodes
            
            for mem_id, metadata in zip(
                all_episodic.get("ids", []),
                all_episodic.get("metadatas", [])
            ):
                timestamp_str = metadata.get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        if timestamp > cutoff:
                            episodes.append((mem_id, metadata))
                    except (ValueError, TypeError):
                        continue
            
        except Exception as e:
            logger.error(f"Failed to get recent episodes: {e}")
        
        return episodes
    
    def _extract_patterns(
        self,
        episodes: List[Tuple[str, Dict]]
    ) -> Dict[str, List[str]]:
        """
        Extract repeated patterns from episodes.
        
        Simplified implementation: groups episodes by shared tags.
        
        Args:
            episodes: List of (memory_id, metadata) tuples
            
        Returns:
            Dict mapping pattern keys to lists of memory IDs
        """
        patterns = defaultdict(list)
        
        for mem_id, metadata in episodes:
            # Group by tags
            tags = metadata.get("tags", [])
            if tags and isinstance(tags, list):
                # Create pattern key from sorted tags (limited by MAX_PATTERN_TAGS)
                pattern_key = tuple(sorted(tags[:MAX_PATTERN_TAGS]))
                if pattern_key:
                    patterns[pattern_key].append(mem_id)
        
        return dict(patterns)
    
    def _create_semantic_from_pattern(
        self,
        pattern_key: tuple,
        occurrence_ids: List[str],
        all_episodes: List[Tuple[str, Dict]]
    ) -> Dict[str, Any]:
        """
        Create semantic memory data from pattern.
        
        Args:
            pattern_key: Pattern identifier
            occurrence_ids: Memory IDs with this pattern
            all_episodes: All episode data
            
        Returns:
            Semantic memory data dictionary
        """
        # Create semantic concept
        tags = list(pattern_key)
        
        return {
            "type": "pattern",
            "tags": tags,
            "pattern": " ".join(tags),
            "occurrences": len(occurrence_ids),
            "source_episodes": occurrence_ids[:MAX_SOURCE_EPISODES],
            "confidence": min(1.0, len(occurrence_ids) / PATTERN_CONFIDENCE_DIVISOR),
            "extracted_at": datetime.now().isoformat(),
            "created_from_consolidation": True
        }
    
    def _weaken_episodes(self, episode_ids: List[str]) -> None:
        """
        Weaken episode memories after pattern extraction.
        
        Args:
            episode_ids: List of episode memory IDs
        """
        for mem_id in episode_ids:
            try:
                result = self.storage.episodic_memory.get(ids=[mem_id])
                if not result or not result.get("ids"):
                    continue
                
                metadata = result["metadatas"][0]
                current_activation = float(metadata.get("base_activation", 1.0))
                metadata["base_activation"] = current_activation * PATTERN_EXTRACTION_WEAKENING_FACTOR
                
                self._update_memory_metadata(mem_id, metadata)
                
            except Exception as e:
                logger.error(f"Failed to weaken episode {mem_id}: {e}")
    
    def _strengthen_association(self, mem1_id: str, mem2_id: str) -> None:
        """
        Strengthen bidirectional association between memories.
        
        Args:
            mem1_id: First memory ID
            mem2_id: Second memory ID
        """
        # Ensure consistent ordering for bidirectional association
        if mem1_id > mem2_id:
            mem1_id, mem2_id = mem2_id, mem1_id
        
        # Update association strength
        current_strength = self.associations[mem1_id][mem2_id]
        new_strength = min(1.0, current_strength + self.association_boost)
        self.associations[mem1_id][mem2_id] = new_strength
        
        logger.debug(
            f"Association {mem1_id} <-> {mem2_id}: "
            f"{current_strength:.3f} -> {new_strength:.3f}"
        )
    
    def _decay_weak_associations(self, threshold: float = 0.1) -> int:
        """
        Remove weak associations below threshold.
        
        Args:
            threshold: Minimum strength to keep
            
        Returns:
            Number of associations removed
        """
        decayed = 0
        
        # Collect associations to remove
        to_remove = []
        for source_id, targets in self.associations.items():
            for target_id, strength in targets.items():
                if strength < threshold:
                    to_remove.append((source_id, target_id))
        
        # Remove weak associations
        for source_id, target_id in to_remove:
            del self.associations[source_id][target_id]
            decayed += 1
        
        return decayed
    
    def _get_high_emotion_memories(
        self,
        threshold: float
    ) -> List[Tuple[str, Dict]]:
        """
        Get memories with high emotional intensity.
        
        Args:
            threshold: Emotional intensity threshold
            
        Returns:
            List of (memory_id, metadata) tuples
        """
        high_emotion = []
        
        try:
            all_episodic = self.storage.episodic_memory.get()
            
            if not all_episodic or not all_episodic.get("ids"):
                return high_emotion
            
            for mem_id, metadata in zip(
                all_episodic.get("ids", []),
                all_episodic.get("metadatas", [])
            ):
                emotional_intensity = float(
                    metadata.get("emotional_intensity", 0.0)
                )
                
                if emotional_intensity >= threshold:
                    high_emotion.append((mem_id, metadata))
        
        except Exception as e:
            logger.error(f"Failed to get high-emotion memories: {e}")
        
        return high_emotion
    
    def get_association_strength(self, mem1_id: str, mem2_id: str) -> float:
        """
        Get association strength between two memories.
        
        Args:
            mem1_id: First memory ID
            mem2_id: Second memory ID
            
        Returns:
            Association strength (0.0-1.0)
        """
        # Ensure consistent ordering
        if mem1_id > mem2_id:
            mem1_id, mem2_id = mem2_id, mem1_id
        
        return self.associations[mem1_id].get(mem2_id, 0.0)
    
    def get_associated_memories(
        self,
        memory_id: str,
        threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Get memories associated with given memory.
        
        Args:
            memory_id: Memory ID
            threshold: Minimum association strength
            
        Returns:
            List of (associated_memory_id, strength) tuples
        """
        associated = []
        
        # Check as source
        for target_id, strength in self.associations.get(memory_id, {}).items():
            if strength >= threshold:
                associated.append((target_id, strength))
        
        # Check as target
        for source_id, targets in self.associations.items():
            if source_id == memory_id:
                continue
            strength = targets.get(memory_id, 0.0)
            if strength >= threshold:
                associated.append((source_id, strength))
        
        # Sort by strength (descending)
        associated.sort(key=lambda x: x[1], reverse=True)
        
        return associated
    
    # Legacy methods maintained for backward compatibility
    
    def strengthen_memory(self, memory_id: str, strength_delta: float = 0.1) -> None:
        """
        Strengthen a memory based on retrieval or rehearsal.
        
        Args:
            memory_id: ID of memory to strengthen
            strength_delta: Amount to increase strength (0.0-1.0)
        """
        self._update_memory_activation(memory_id, strength_delta)
    
    def transfer_episodic_to_semantic(self, pattern_threshold: int = 3) -> int:
        """
        Transfer repeated episodic patterns to semantic memory.
        
        Args:
            pattern_threshold: Number of similar episodes to trigger transfer
            
        Returns:
            Number of memories transferred
        """
        return self.transfer_to_semantic(threshold=pattern_threshold)
    
    def reorganize_memories(self) -> None:
        """
        Sleep-like reorganization and compression of memories.
        
        Simulates consolidation that occurs during rest periods.
        """
        self.reorganize_associations()
        self.reprocess_emotional_memories()

