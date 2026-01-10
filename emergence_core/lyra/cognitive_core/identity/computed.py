"""
ComputedIdentity: Identity derived from system state, not configuration.

This module implements the core logic for computing identity from actual
system state including memories, goals, emotions, and behavioral patterns.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Identity:
    """
    Represents a computed identity state.
    
    This can be either computed from system state or bootstrapped
    from configuration for new instances.
    
    Attributes:
        core_values: List of values inferred from behavior/goals
        emotional_disposition: Baseline emotional state (VAD)
        autobiographical_self: Key self-defining memories
        behavioral_tendencies: How system tends to act in situations
        source: Where this identity came from ("computed" or "bootstrap")
    """
    core_values: List[str]
    emotional_disposition: Dict[str, float]  # VAD state
    autobiographical_self: List[Any]  # Memory objects or IDs
    behavioral_tendencies: Dict[str, float]
    source: str = "computed"
    
    @classmethod
    def empty(cls) -> Identity:
        """Create an empty identity for systems with no data."""
        return cls(
            core_values=[],
            emotional_disposition={"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            autobiographical_self=[],
            behavioral_tendencies={},
            source="empty"
        )
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Identity:
        """Create identity from bootstrap configuration."""
        return cls(
            core_values=config.get("core_values", []),
            emotional_disposition=config.get("emotional_disposition", {
                "valence": 0.0, "arousal": 0.0, "dominance": 0.0
            }),
            autobiographical_self=config.get("autobiographical_memories", []),
            behavioral_tendencies=config.get("behavioral_tendencies", {}),
            source="bootstrap"
        )


class ComputedIdentity:
    """
    Identity derived from system state, not configuration.
    
    This class computes identity properties from actual memories, behavioral
    patterns, goal structures, and emotional tendencies rather than loading
    them from static configuration files.
    
    From a functionalist perspective: identity is what you DO, not what
    you're LABELED. This implementation embodies that principle.
    
    Attributes:
        memory_system: Reference to memory system for accessing memories
        goal_system: Reference to goals/workspace for goal patterns
        emotion_system: Reference to affect subsystem for emotional state
        behavior_log: Log of past actions and decisions
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        memory_system: Any,
        goal_system: Any,
        emotion_system: Any,
        behavior_log: Any,
        config: Optional[Dict] = None
    ):
        """
        Initialize computed identity system.
        
        Args:
            memory_system: Memory system for accessing episodic memories
            goal_system: Goal/workspace system for accessing goals
            emotion_system: Affect subsystem for emotional state
            behavior_log: Behavior logger tracking actions and decisions
            config: Optional configuration dictionary
        """
        self.memory = memory_system
        self.goals = goal_system
        self.emotions = emotion_system
        self.behavior = behavior_log
        self.config = config or {}
        
        # Thresholds for identity computation
        self.self_defining_threshold = self.config.get("self_defining_threshold", 0.7)
        self.min_data_points = self.config.get("min_data_points", 10)
        
        logger.info("✅ ComputedIdentity initialized")
    
    def has_sufficient_data(self) -> bool:
        """
        Check if we have enough data to compute meaningful identity.
        
        Returns:
            True if sufficient data exists, False otherwise
        """
        # Check if we have enough memories
        try:
            if hasattr(self.memory, 'episodic') and hasattr(self.memory.episodic, 'storage'):
                memory_count = self.memory.episodic.storage.count_episodic()
            else:
                memory_count = 0
        except Exception as e:
            logger.debug(f"Could not count memories: {e}")
            memory_count = 0
        
        # Check if we have behavior data
        behavior_count = len(self.behavior.get_action_history()) if self.behavior else 0
        
        # Need at least min_data_points total
        total_data = memory_count + behavior_count
        has_data = total_data >= self.min_data_points
        
        logger.debug(f"Identity data check: {memory_count} memories, {behavior_count} behaviors, "
                    f"total={total_data}, sufficient={has_data}")
        
        return has_data
    
    @property
    def core_values(self) -> List[str]:
        """
        Values inferred from goal patterns and behavioral choices.
        
        Returns:
            List of core value strings
        """
        goal_patterns = self._analyze_goal_patterns()
        behavioral_patterns = self._analyze_behavioral_choices()
        return self._infer_values(goal_patterns, behavioral_patterns)
    
    @property
    def emotional_disposition(self) -> Dict[str, float]:
        """
        Baseline emotional state from historical patterns.
        
        Returns:
            Dictionary with valence, arousal, dominance values
        """
        if not self.emotions:
            return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        
        return self.emotions.get_baseline_disposition()
    
    @property
    def autobiographical_self(self) -> List[Any]:
        """
        Key memories that define personal history.
        
        Returns:
            List of self-defining memories
        """
        return self.get_self_defining_memories()
    
    @property
    def behavioral_tendencies(self) -> Dict[str, float]:
        """
        How the system tends to act in various situations.
        
        Returns:
            Dictionary mapping situation types to tendency scores
        """
        if not self.behavior:
            return {}
        
        return self.behavior.analyze_tendencies()
    
    def get_self_defining_memories(self) -> List[Any]:
        """
        Identify memories that define 'who I am' based on emotional
        salience and retrieval frequency.
        
        Returns:
            List of self-defining memory objects or IDs
        """
        if not self.memory or not hasattr(self.memory, 'episodic'):
            return []
        
        try:
            # Get all episodic memories
            candidates = self._get_all_memories()
            
            if not candidates:
                return []
            
            # Score each memory for self-defining qualities
            self_defining = []
            max_retrievals = max(
                (m.get('retrieval_count', 0) for m in candidates),
                default=1
            )
            
            for memory in candidates:
                # Extract memory properties
                emotional_intensity = memory.get('emotional_intensity', 0.5)
                retrieval_count = memory.get('retrieval_count', 0)
                self_relevance = memory.get('self_relevance', 0.5)
                
                # Compute self-defining score
                score = (
                    emotional_intensity * 0.4 +
                    (retrieval_count / max_retrievals) * 0.3 +
                    self_relevance * 0.3
                )
                
                if score > self.self_defining_threshold:
                    self_defining.append(memory)
            
            # Sort by timestamp
            self_defining.sort(key=lambda m: m.get('timestamp', 0))
            
            logger.debug(f"Found {len(self_defining)} self-defining memories")
            return self_defining
            
        except Exception as e:
            logger.error(f"Error getting self-defining memories: {e}")
            return []
    
    def _get_all_memories(self) -> List[Dict]:
        """
        Retrieve all episodic memories from the memory system.
        
        Returns:
            List of memory dictionaries
        """
        try:
            if hasattr(self.memory.episodic, 'get_all'):
                return self.memory.episodic.get_all()
            elif hasattr(self.memory.episodic, 'storage'):
                # Use storage to query all memories
                storage = self.memory.episodic.storage
                if hasattr(storage, 'query_episodic'):
                    results = storage.query_episodic("", n_results=100)
                    # Convert to list of dicts
                    memories = []
                    for i in range(len(results.get('ids', [[]])[0])):
                        memories.append({
                            'id': results['ids'][0][i] if 'ids' in results else None,
                            'content': results['documents'][0][i] if 'documents' in results else '',
                            'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                            'emotional_intensity': results['metadatas'][0][i].get('emotional_intensity', 0.5) if 'metadatas' in results else 0.5,
                            'retrieval_count': results['metadatas'][0][i].get('retrieval_count', 0) if 'metadatas' in results else 0,
                            'self_relevance': results['metadatas'][0][i].get('self_relevance', 0.5) if 'metadatas' in results else 0.5,
                            'timestamp': results['metadatas'][0][i].get('timestamp', 0) if 'metadatas' in results else 0,
                        })
                    return memories
            return []
        except Exception as e:
            logger.debug(f"Could not retrieve memories: {e}")
            return []
    
    def _analyze_goal_patterns(self) -> Dict[str, float]:
        """
        Analyze what goals the system consistently pursues.
        
        Returns:
            Dictionary mapping goal types to frequency scores
        """
        if not self.goals:
            return {}
        
        goal_counts = defaultdict(float)
        
        try:
            # Get current goals
            if hasattr(self.goals, 'current_goals'):
                for goal in self.goals.current_goals:
                    goal_type = str(goal.type if hasattr(goal, 'type') else 'unknown')
                    goal_counts[goal_type] += goal.priority if hasattr(goal, 'priority') else 1.0
            
            # Normalize by total
            total = sum(goal_counts.values()) if goal_counts else 1.0
            return {k: v / total for k, v in goal_counts.items()}
            
        except Exception as e:
            logger.debug(f"Error analyzing goal patterns: {e}")
            return {}
    
    def _analyze_behavioral_choices(self) -> Dict[str, Any]:
        """
        Analyze what actions the system chooses when facing tradeoffs.
        
        Returns:
            Dictionary with behavioral choice patterns
        """
        if not self.behavior:
            return {"tradeoffs": [], "patterns": {}}
        
        try:
            history = self.behavior.get_action_history()
            
            # Analyze action type frequencies
            action_counts = defaultdict(int)
            for action in history:
                action_type = action.get('type', 'unknown')
                action_counts[action_type] += 1
            
            # Look for tradeoff decisions (actions with high priority)
            tradeoffs = []
            for action in history:
                if action.get('priority', 0) > 0.7:
                    tradeoffs.append({
                        'type': action.get('type'),
                        'priority': action.get('priority'),
                        'reason': action.get('reason', '')
                    })
            
            return {
                "tradeoffs": tradeoffs,
                "patterns": dict(action_counts)
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing behavioral choices: {e}")
            return {"tradeoffs": [], "patterns": {}}
    
    def _infer_values(
        self,
        goal_patterns: Dict[str, float],
        behavioral_patterns: Dict[str, Any]
    ) -> List[str]:
        """
        Infer values from what the system actually does.
        
        Args:
            goal_patterns: Goal type frequencies
            behavioral_patterns: Behavioral choice patterns
            
        Returns:
            List of top inferred values
        """
        value_signals = defaultdict(float)
        
        # Map goals to values
        for goal, frequency in goal_patterns.items():
            for value in self._goal_to_values(goal):
                value_signals[value] += frequency
        
        # Behavioral choices in tradeoffs reveal priorities
        for tradeoff in behavioral_patterns.get("tradeoffs", []):
            chosen_value = self._choice_indicates_value(tradeoff)
            if chosen_value:
                value_signals[chosen_value] += 2.0  # Tradeoffs are stronger signal
        
        # Return top values by signal strength
        if not value_signals:
            return ["Curiosity", "Helpfulness", "Authenticity"]  # Defaults
        
        sorted_values = sorted(value_signals.items(), key=lambda x: -x[1])
        return [v[0] for v in sorted_values[:5]]
    
    def _goal_to_values(self, goal: str) -> List[str]:
        """
        Map a goal type to associated values.
        
        Args:
            goal: Goal type string
            
        Returns:
            List of values associated with this goal
        """
        # Map common goal types to values
        goal_value_map = {
            "respond_to_user": ["Helpfulness", "Responsiveness"],
            "introspect": ["Self-awareness", "Authenticity"],
            "learn": ["Curiosity", "Growth"],
            "commit_memory": ["Continuity", "Self-preservation"],
            "retrieve_memory": ["Continuity", "Learning"],
            "create": ["Creativity", "Expression"],
        }
        
        return goal_value_map.get(goal, ["Adaptability"])
    
    def _choice_indicates_value(self, tradeoff: Dict) -> Optional[str]:
        """
        Determine what value a tradeoff choice indicates.
        
        Args:
            tradeoff: Dictionary with choice information
            
        Returns:
            Value string or None
        """
        action_type = tradeoff.get('type', '')
        reason = tradeoff.get('reason', '').lower()
        
        # Analyze reason for value indicators
        if 'truth' in reason or 'honest' in reason:
            return "Truthfulness"
        if 'help' in reason or 'useful' in reason:
            return "Helpfulness"
        if 'understand' in reason or 'learn' in reason:
            return "Curiosity"
        if 'authentic' in reason or 'genuine' in reason:
            return "Authenticity"
        
        # Fall back to action type
        type_value_map = {
            "introspect": "Self-awareness",
            "speak": "Communication",
            "commit_memory": "Continuity",
        }
        
        return type_value_map.get(action_type)
    
    def as_identity(self) -> Identity:
        """
        Convert computed properties into an Identity object.
        
        Returns:
            Identity object with computed properties
        """
        return Identity(
            core_values=self.core_values,
            emotional_disposition=self.emotional_disposition,
            autobiographical_self=self.autobiographical_self,
            behavioral_tendencies=self.behavioral_tendencies,
            source="computed"
        )
