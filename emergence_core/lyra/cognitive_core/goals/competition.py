"""
Goal Competition Dynamics

Implements activation-based competition with lateral inhibition, allowing
goals to compete for cognitive resources based on their importance, urgency,
and relationships to other goals.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

try:
    from .resources import CognitiveResources, ResourcePool
except ImportError:
    from resources import CognitiveResources, ResourcePool

logger = logging.getLogger(__name__)

# Activation calculation weights
IMPORTANCE_WEIGHT = 0.4
URGENCY_WEIGHT = 0.3
PROGRESS_WEIGHT = 0.2
EMOTION_WEIGHT = 0.1


@dataclass
class ActiveGoal:
    """
    Wrapper for a goal with allocated resources and activation level.
    
    Attributes:
        goal: The underlying goal object
        activation: Current activation level (0.0 to 1.0)
        resources: Allocated cognitive resources
        waiting_for_resources: True if goal needs resources but can't get them
    """
    goal: Any  # Goal from executive_function or workspace
    activation: float
    resources: CognitiveResources
    waiting_for_resources: bool = False
    
    def __post_init__(self):
        """Validate activation level."""
        if not 0 <= self.activation <= 1:
            raise ValueError(f"Activation must be in [0, 1], got {self.activation}")


class GoalCompetition:
    """
    Manages goal competition using activation-based dynamics.
    
    Goals compete for activation through:
    - Self-excitation based on importance and progress
    - Lateral inhibition from conflicting goals
    - Winner-take-all dynamics for resource allocation
    """
    
    def __init__(self, inhibition_strength: float = 0.3):
        """
        Initialize goal competition system.
        
        Args:
            inhibition_strength: How strongly goals inhibit each other (0.0 to 1.0)
            
        Raises:
            ValueError: If inhibition_strength is out of range
        """
        if not 0 <= inhibition_strength <= 1:
            raise ValueError(f"inhibition_strength must be in [0, 1], got {inhibition_strength}")
        
        self.inhibition_strength = inhibition_strength
        logger.info(f"GoalCompetition initialized with inhibition_strength={inhibition_strength}")
    
    def compete(self, goals: List[Any], iterations: int = 10) -> Dict[str, float]:
        """
        Run goal competition to determine activation levels.
        
        Goals compete through iterative activation dynamics where:
        - Each goal excites itself based on importance and progress
        - Goals inhibit each other based on conflict level
        - System converges to stable activation pattern
        
        Args:
            goals: List of Goal objects to compete
            iterations: Number of competition iterations (default: 10)
            
        Returns:
            Dict mapping goal IDs to final activation levels
            
        Raises:
            ValueError: If iterations < 1
        """
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        
        if not goals:
            return {}
        
        # Initialize activations
        activations = {self._get_goal_id(g): self._initial_activation(g) for g in goals}
        
        # Run competition dynamics
        for iteration in range(iterations):
            new_activations = {}
            
            for goal in goals:
                goal_id = self._get_goal_id(goal)
                
                # Self-excitation from goal importance and progress
                importance = self._get_goal_importance(goal)
                excitation = activations[goal_id] * (1 + importance * 0.1)
                
                # Lateral inhibition from competing goals
                inhibition = 0
                for other_goal in goals:
                    other_id = self._get_goal_id(other_goal)
                    if other_id != goal_id:
                        conflict = self._goal_conflict(goal, other_goal)
                        inhibition += activations[other_id] * conflict * self.inhibition_strength
                
                # Update activation (clamp to [0, 1])
                new_activations[goal_id] = max(0, min(1, excitation - inhibition))
            
            activations = new_activations
        
        logger.debug(f"Competition complete after {iterations} iterations")
        return activations
    
    def _get_goal_id(self, goal: Any) -> str:
        """Extract goal ID from various goal types."""
        if hasattr(goal, 'id'):
            return goal.id
        elif isinstance(goal, dict) and 'id' in goal:
            return goal['id']
        else:
            raise ValueError(f"Cannot extract ID from goal: {type(goal)}")
    
    def _get_goal_importance(self, goal: Any) -> float:
        """Extract importance/priority from goal."""
        if hasattr(goal, 'importance'):
            return goal.importance
        elif hasattr(goal, 'priority'):
            return goal.priority
        elif isinstance(goal, dict):
            return goal.get('importance', goal.get('priority', 0.5))
        return 0.5
    
    def _get_goal_progress(self, goal: Any) -> float:
        """Extract progress from goal."""
        if hasattr(goal, 'progress'):
            return goal.progress
        elif isinstance(goal, dict):
            return goal.get('progress', 0.0)
        return 0.0
    
    def _get_goal_urgency(self, goal: Any) -> float:
        """
        Calculate urgency based on deadline proximity.
        
        Returns:
            float: Urgency level (0.0 to 1.0)
        """
        if hasattr(goal, 'deadline') and goal.deadline:
            now = datetime.now()
            if goal.deadline <= now:
                return 1.0  # Overdue
            time_remaining = (goal.deadline - now).total_seconds()
            # More urgent as deadline approaches
            if time_remaining < 3600:  # < 1 hour
                return 0.9
            elif time_remaining < 86400:  # < 1 day
                return 0.7
            elif time_remaining < 604800:  # < 1 week
                return 0.5
            else:
                return 0.3
        return 0.5  # Default moderate urgency
    
    def _get_emotional_valence(self, goal: Any) -> float:
        """Extract emotional valence from goal metadata."""
        if hasattr(goal, 'metadata') and isinstance(goal.metadata, dict):
            return goal.metadata.get('emotional_valence', 0.0)
        elif isinstance(goal, dict):
            metadata = goal.get('metadata', {})
            return metadata.get('emotional_valence', 0.0)
        return 0.0
    
    def _initial_activation(self, goal: Any) -> float:
        """
        Calculate initial activation based on goal properties.
        
        Combines importance, urgency, progress, and emotional valence.
        
        Args:
            goal: Goal object to evaluate
            
        Returns:
            Initial activation level (0.0 to 1.0)
        """
        importance = self._get_goal_importance(goal)
        urgency = self._get_goal_urgency(goal)
        progress = self._get_goal_progress(goal)
        emotional_valence = self._get_emotional_valence(goal)
        
        # Less progress = more activation (incomplete goals need attention)
        activation = (
            importance * IMPORTANCE_WEIGHT +
            urgency * URGENCY_WEIGHT +
            (1 - progress) * PROGRESS_WEIGHT +
            abs(emotional_valence) * EMOTION_WEIGHT  # Strong emotions increase activation
        )
        
        return max(0, min(1, activation))
    
    def _goal_conflict(self, g1: Any, g2: Any) -> float:
        """
        Calculate how much two goals conflict.
        
        Args:
            g1: First goal
            g2: Second goal
            
        Returns:
            Conflict level (0.0 = compatible, 1.0 = mutually exclusive)
        """
        # Check if goals have resource needs specified
        resource_conflict = self._resource_overlap(g1, g2)
        
        # Check for logical conflicts (if specified in metadata)
        logical_conflict = self._outcome_conflict(g1, g2)
        
        return max(resource_conflict, logical_conflict)
    
    def _resource_overlap(self, g1: Any, g2: Any) -> float:
        """
        Calculate resource overlap between two goals.
        
        Returns:
            Overlap level (0.0 to 1.0)
        """
        # Get resource needs if available
        needs1 = self._get_resource_needs(g1)
        needs2 = self._get_resource_needs(g2)
        
        if needs1 is None or needs2 is None:
            # No resource information, assume moderate conflict
            return 0.3
        
        # Calculate overlap in each resource dimension
        attention_overlap = min(needs1.attention_budget, needs2.attention_budget)
        processing_overlap = min(needs1.processing_budget, needs2.processing_budget)
        action_overlap = min(needs1.action_budget, needs2.action_budget)
        time_overlap = min(needs1.time_budget, needs2.time_budget)
        
        total_overlap = (attention_overlap + processing_overlap + 
                        action_overlap + time_overlap)
        max_possible = min(needs1.total(), needs2.total())
        
        if max_possible == 0:
            return 0.0
        
        return min(1.0, total_overlap / max_possible)
    
    def _get_resource_needs(self, goal: Any) -> Optional[CognitiveResources]:
        """Extract resource needs from goal."""
        if hasattr(goal, 'resource_needs'):
            return goal.resource_needs
        elif isinstance(goal, dict) and 'resource_needs' in goal:
            needs = goal['resource_needs']
            if isinstance(needs, CognitiveResources):
                return needs
            elif isinstance(needs, dict):
                return CognitiveResources(**needs)
        return None
    
    def _outcome_conflict(self, g1: Any, g2: Any) -> float:
        """
        Check if goals have contradictory desired outcomes.
        
        Returns:
            Conflict level (0.0 to 1.0)
        """
        # Check metadata for explicit conflicts
        if hasattr(g1, 'metadata') and hasattr(g2, 'metadata'):
            conflicts_with = g1.metadata.get('conflicts_with', [])
            g2_id = self._get_goal_id(g2)
            if g2_id in conflicts_with:
                return 1.0
            
            # Check mutual exclusivity
            if g1.metadata.get('mutually_exclusive_with', []):
                if g2_id in g1.metadata['mutually_exclusive_with']:
                    return 0.8
        
        # No explicit conflicts found
        return 0.0
    
    def select_active_goals(
        self,
        goals: List[Any],
        pool: ResourcePool,
        max_active: Optional[int] = None
    ) -> List[ActiveGoal]:
        """
        Select goals to pursue based on competition and resource availability.
        
        Args:
            goals: List of candidate goals
            pool: Resource pool for allocation
            max_active: Optional limit on number of active goals
            
        Returns:
            List of ActiveGoal instances with allocated resources
        """
        if not goals:
            return []
        
        # Run competition
        activations = self.compete(goals)
        
        # Sort by activation (highest first)
        sorted_goals = sorted(
            goals,
            key=lambda g: activations[self._get_goal_id(g)],
            reverse=True
        )
        
        active_goals = []
        
        for goal in sorted_goals:
            goal_id = self._get_goal_id(goal)
            
            # Check max_active limit
            if max_active is not None and len(active_goals) >= max_active:
                break
            
            # Get resource needs
            needed = self._get_resource_needs(goal)
            if needed is None:
                # Default resource needs if not specified
                needed = CognitiveResources(
                    attention_budget=0.2,
                    processing_budget=0.2,
                    action_budget=0.2,
                    time_budget=0.2
                )
            
            # Try to allocate resources
            if pool.can_allocate(needed):
                granted = pool.allocate(goal_id, needed)
                active_goals.append(ActiveGoal(
                    goal=goal,
                    activation=activations[goal_id],
                    resources=granted,
                    waiting_for_resources=False
                ))
                logger.debug(f"Goal '{goal_id}' activated with {granted.total():.2f} resources")
            else:
                # Not enough resources - mark as waiting
                logger.debug(f"Goal '{goal_id}' waiting for resources")
                # Could track waiting goals separately if needed
        
        return active_goals
