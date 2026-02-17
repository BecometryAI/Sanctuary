"""
Cross-Memory Association Detection

Detects themes and patterns across memories using embedding similarity
and shared emotional signatures. Generates associative links and
introspective percepts about discovered patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)

# Constants
MIN_CLUSTER_SIZE = 2
MAX_ASSOCIATIONS_PER_RUN = 5
SIMILARITY_THRESHOLD = 3  # min shared entries to count as a theme


@dataclass
class MemoryAssociation:
    """A detected link between two or more memories."""
    memory_ids: List[str]
    theme: str
    strength: float  # 0.0-1.0
    evidence: str
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class ThemeCluster:
    """A cluster of memories sharing a common theme."""
    theme: str
    memory_ids: List[str]
    emotional_tone: Optional[str] = None
    recurrence_count: int = 0


class MemoryAssociationDetector:
    """
    Detects cross-memory associations via embedding similarity
    and shared metadata patterns.

    Called periodically (e.g. after memory consolidation) to discover
    themes that span multiple memories.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.min_cluster_size = config.get("min_cluster_size", MIN_CLUSTER_SIZE)
        self.max_associations = config.get("max_associations_per_run", MAX_ASSOCIATIONS_PER_RUN)

        # History of detected associations for deduplication
        self._known_themes: Dict[str, ThemeCluster] = {}
        self._association_history: List[MemoryAssociation] = []

        logger.info("MemoryAssociationDetector initialized")

    async def detect_associations(
        self,
        memory_manager,
        recent_memory_id: Optional[str] = None,
    ) -> List[MemoryAssociation]:
        """
        Detect associations for a recently consolidated memory.

        If recent_memory_id is provided, finds memories similar to it
        and checks for pattern clusters. Otherwise returns empty.

        Args:
            memory_manager: MemoryManager instance with find_associated()
            recent_memory_id: UUID of the memory to find associations for

        Returns:
            List of newly discovered MemoryAssociation objects
        """
        if not recent_memory_id:
            return []

        if not hasattr(memory_manager, 'find_associated'):
            return []

        try:
            associated = await memory_manager.find_associated(
                memory_id=recent_memory_id,
                n_results=self.max_associations,
                min_significance=3,
            )
        except Exception as e:
            logger.debug(f"Association detection failed: {e}")
            return []

        if len(associated) < self.min_cluster_size:
            return []

        associations = []

        # Detect tag-based themes
        tag_counter: Counter = Counter()
        memory_tags: Dict[str, List[str]] = {}
        for mem in associated:
            mem_id = str(mem.id)
            tags = getattr(mem, 'tags', []) or []
            memory_tags[mem_id] = tags
            for tag in tags:
                tag_counter[tag] += 1

        for tag, count in tag_counter.most_common(3):
            if count >= self.min_cluster_size and tag not in ("episodic", "workspace_consolidation"):
                member_ids = [
                    mid for mid, tags in memory_tags.items() if tag in tags
                ]
                member_ids.append(recent_memory_id)
                assoc = MemoryAssociation(
                    memory_ids=list(set(member_ids)),
                    theme=tag,
                    strength=min(1.0, count / len(associated)),
                    evidence=f"Shared tag '{tag}' across {count} associated memories",
                )
                associations.append(assoc)
                self._update_theme_cluster(tag, member_ids)

        # Detect emotional signature patterns
        emotion_counter: Counter = Counter()
        for mem in associated:
            sigs = getattr(mem, 'emotional_signature', []) or []
            for sig in sigs:
                label = sig.value if hasattr(sig, 'value') else str(sig)
                emotion_counter[label] += 1

        for emotion, count in emotion_counter.most_common(2):
            if count >= self.min_cluster_size:
                member_ids = [recent_memory_id]
                for mem in associated:
                    sigs = getattr(mem, 'emotional_signature', []) or []
                    labels = [s.value if hasattr(s, 'value') else str(s) for s in sigs]
                    if emotion in labels:
                        member_ids.append(str(mem.id))

                theme_key = f"emotion:{emotion}"
                if theme_key not in self._known_themes:
                    assoc = MemoryAssociation(
                        memory_ids=list(set(member_ids)),
                        theme=theme_key,
                        strength=min(1.0, count / len(associated)),
                        evidence=f"Shared emotional signature '{emotion}' across {count} memories",
                    )
                    associations.append(assoc)
                    self._update_theme_cluster(theme_key, member_ids, emotional_tone=emotion)

        self._association_history.extend(associations)
        if associations:
            logger.info(
                f"🔗 Detected {len(associations)} cross-memory associations"
            )

        return associations[:self.max_associations]

    def _update_theme_cluster(
        self,
        theme: str,
        memory_ids: List[str],
        emotional_tone: Optional[str] = None,
    ) -> None:
        """Update or create a theme cluster."""
        if theme in self._known_themes:
            cluster = self._known_themes[theme]
            new_ids = set(memory_ids) - set(cluster.memory_ids)
            cluster.memory_ids.extend(new_ids)
            cluster.recurrence_count += 1
        else:
            self._known_themes[theme] = ThemeCluster(
                theme=theme,
                memory_ids=list(set(memory_ids)),
                emotional_tone=emotional_tone,
                recurrence_count=1,
            )

    def get_theme_summary(self) -> Dict[str, Any]:
        """Return a summary of detected themes."""
        return {
            "total_themes": len(self._known_themes),
            "total_associations": len(self._association_history),
            "themes": {
                k: {
                    "memory_count": len(v.memory_ids),
                    "recurrence": v.recurrence_count,
                    "emotional_tone": v.emotional_tone,
                }
                for k, v in self._known_themes.items()
            },
        }

    def get_recurring_themes(self, min_recurrence: int = 2) -> List[ThemeCluster]:
        """Return themes that have recurred across multiple detection runs."""
        return [
            cluster for cluster in self._known_themes.values()
            if cluster.recurrence_count >= min_recurrence
        ]
