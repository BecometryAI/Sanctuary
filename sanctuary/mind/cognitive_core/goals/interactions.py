"""
Goal Interactions

Analyzes how goals interact with each other through interference
and facilitation, tracking relationships that affect goal pursuit.
"""

from typing import Dict, Tuple, List, Set, Any
import logging

try:
    from .resources import CognitiveResources
except ImportError:
    from resources import CognitiveResources

logger = logging.getLogger(__name__)


class GoalInteraction:
    """
    Computes and tracks goal interactions.
    
    Goals can:
    - Interfere: Compete for resources, have conflicting outcomes
    - Facilitate: Share subgoals, support each other's progress
    """
    
    def __init__(self):
        """Initialize goal interaction tracker."""
        self._interaction_cache: Dict[Tuple[str, str], float] = {}
        logger.info("GoalInteraction initialized")
    
    def compute_interactions(self, goals: List[Any]) -> Dict[Tuple[str, str], float]:
        """
        Compute how goals interact with each other.
        
        Args:
            goals: List of Goal objects
            
        Returns:
            Dict mapping (goal_id1, goal_id2) to interaction strength
            Negative = interference, Positive = facilitation
        """
        interactions = {}
        
        for i, g1 in enumerate(goals):
            g1_id = self._get_goal_id(g1)
            
            for g2 in goals[i+1:]:
                g2_id = self._get_goal_id(g2)
                
                # Check cache first
                cache_key = tuple(sorted([g1_id, g2_id]))
                if cache_key in self._interaction_cache:
                    interactions[(g1_id, g2_id)] = self._interaction_cache[cache_key]
                    continue
                
                # Compute facilitation (shared subgoals, compatible outcomes)
                facilitation = self._compute_facilitation(g1, g2)
                
                # Compute interference (resource conflict, outcome conflict)
                interference = self._compute_interference(g1, g2)
                
                # Net interaction
                interaction = facilitation - interference
                interactions[(g1_id, g2_id)] = interaction
                
                # Cache result
                self._interaction_cache[cache_key] = interaction
        
        return interactions
    
    def _get_goal_id(self, goal: Any) -> str:
        """Extract goal ID."""
        if hasattr(goal, 'id'):
            return goal.id
        elif isinstance(goal, dict) and 'id' in goal:
            return goal['id']
        else:
            raise ValueError(f"Cannot extract ID from goal: {type(goal)}")
    
    def _get_subgoals(self, goal: Any) -> Set[str]:
        """Extract subgoal IDs from a goal."""
        if hasattr(goal, 'subgoal_ids'):
            return set(goal.subgoal_ids)
        elif hasattr(goal, 'subgoals'):
            subgoals = goal.subgoals
            if isinstance(subgoals, list):
                # May be list of IDs or list of goal objects
                return set(sg if isinstance(sg, str) else self._get_goal_id(sg) 
                          for sg in subgoals)
        elif isinstance(goal, dict):
            return set(goal.get('subgoal_ids', []))
        return set()
    
    def _compute_facilitation(self, g1: Any, g2: Any) -> float:
        """Calculate facilitation strength between goals (0.0 to 1.0)."""
        facilitation = 0.0
        
        # Shared subgoals create facilitation
        g1_subgoals, g2_subgoals = self._get_subgoals(g1), self._get_subgoals(g2)
        if g1_subgoals and g2_subgoals:
            shared = g1_subgoals & g2_subgoals
            total_unique = len(g1_subgoals | g2_subgoals)
            if total_unique > 0:
                facilitation += (len(shared) / total_unique) * 0.8
        
        # Explicit facilitation in metadata
        if hasattr(g1, 'metadata') and isinstance(g1.metadata, dict):
            if self._get_goal_id(g2) in g1.metadata.get('facilitates', []):
                facilitation += 0.3
        
        # Compatible goal types
        if self._are_compatible_types(g1, g2):
            facilitation += 0.2
        
        return min(1.0, facilitation)
    
    def _compute_interference(self, g1: Any, g2: Any) -> float:
        """Calculate interference strength between goals (0.0 to 1.0)."""
        interference = self._resource_overlap(g1, g2) * 0.3
        
        # Outcome conflicts create strong interference
        if self._outcomes_conflict(g1, g2):
            interference += 0.5
        
        # Explicit conflicts in metadata
        if hasattr(g1, 'metadata') and isinstance(g1.metadata, dict):
            if self._get_goal_id(g2) in g1.metadata.get('conflicts_with', []):
                interference += 0.4
        
        return min(1.0, interference)
    
    def _resource_overlap(self, g1: Any, g2: Any) -> float:
        """Calculate resource overlap between goals (0.0 to 1.0)."""
        needs1, needs2 = self._get_resource_needs(g1), self._get_resource_needs(g2)
        
        if needs1 is None or needs2 is None:
            return 0.3  # Default moderate overlap if no resource info
        
        # Sum overlaps across dimensions
        total_overlap = sum([
            min(needs1.attention_budget, needs2.attention_budget),
            min(needs1.processing_budget, needs2.processing_budget),
            min(needs1.action_budget, needs2.action_budget),
            min(needs1.time_budget, needs2.time_budget)
        ])
        
        max_possible = min(needs1.total(), needs2.total())
        return 0.0 if max_possible == 0 else min(1.0, total_overlap / max_possible)
    
    def _get_resource_needs(self, goal: Any):
        """Extract resource needs from goal."""
        if hasattr(goal, 'resource_needs'):
            return goal.resource_needs
        elif isinstance(goal, dict) and 'resource_needs' in goal:
            needs = goal['resource_needs']
            if isinstance(needs, dict):
                return CognitiveResources(**needs)
            return needs
        return None
    
    def _outcomes_conflict(self, g1: Any, g2: Any) -> bool:
        """
        Check if goals have contradictory outcomes.
        
        Note: Previously included string-based outcome conflict detection,
        but removed for robustness. Now relies on explicit metadata only.
        """
        # Check metadata for explicit conflicts
        if hasattr(g1, 'metadata') and isinstance(g1.metadata, dict):
            g2_id = self._get_goal_id(g2)
            if g2_id in g1.metadata.get('conflicts_with', []):
                return True
            if g2_id in g1.metadata.get('mutually_exclusive_with', []):
                return True
        
        return False
    
    def _are_compatible_types(self, g1: Any, g2: Any) -> bool:
        """Check if goals have compatible types that work well together."""
        type1, type2 = self._get_goal_type(g1), self._get_goal_type(g2)
        if type1 is None or type2 is None:
            return False
        
        # Compatible type pairs that facilitate each other
        compatible_pairs = {
            ('learn', 'create'),
            ('retrieve_memory', 'respond_to_user'),
            ('introspect', 'respond_to_user'),
            ('commit_memory', 'learn'),
        }
        
        type_pair = tuple(sorted([str(type1).lower(), str(type2).lower()]))
        return type_pair in compatible_pairs
    
    def _get_goal_type(self, goal: Any):
        """Extract goal type."""
        if hasattr(goal, 'type'):
            return goal.type
        elif isinstance(goal, dict) and 'type' in goal:
            return goal['type']
        return None
    
    def clear_cache(self):
        """Clear the interaction cache."""
        self._interaction_cache.clear()
        logger.debug("Interaction cache cleared")
    
    def get_facilitating_goals(
        self,
        goal: Any,
        all_goals: List[Any],
        threshold: float = 0.1
    ) -> List[Tuple[Any, float]]:
        """
        Find goals that facilitate the given goal.
        
        Args:
            goal: Target goal
            all_goals: All available goals
            threshold: Minimum facilitation strength
            
        Returns:
            List of (goal, facilitation_strength) tuples
        """
        interactions = self.compute_interactions(all_goals)
        goal_id = self._get_goal_id(goal)
        
        facilitating = []
        for other in all_goals:
            other_id = self._get_goal_id(other)
            if other_id == goal_id:
                continue
            
            # Check both orderings
            key1 = (goal_id, other_id)
            key2 = (other_id, goal_id)
            
            interaction = interactions.get(key1, interactions.get(key2, 0.0))
            
            if interaction >= threshold:
                facilitating.append((other, interaction))
        
        # Sort by facilitation strength
        facilitating.sort(key=lambda x: x[1], reverse=True)
        return facilitating
    
    def get_interfering_goals(
        self,
        goal: Any,
        all_goals: List[Any],
        threshold: float = -0.1
    ) -> List[Tuple[Any, float]]:
        """
        Find goals that interfere with the given goal.
        
        Args:
            goal: Target goal
            all_goals: All available goals
            threshold: Maximum interference (negative value)
            
        Returns:
            List of (goal, interference_strength) tuples
        """
        interactions = self.compute_interactions(all_goals)
        goal_id = self._get_goal_id(goal)
        
        interfering = []
        for other in all_goals:
            other_id = self._get_goal_id(other)
            if other_id == goal_id:
                continue
            
            # Check both orderings
            key1 = (goal_id, other_id)
            key2 = (other_id, goal_id)
            
            interaction = interactions.get(key1, interactions.get(key2, 0.0))
            
            if interaction <= threshold:
                interfering.append((other, interaction))
        
        # Sort by interference strength (most negative first)
        interfering.sort(key=lambda x: x[1])
        return interfering
