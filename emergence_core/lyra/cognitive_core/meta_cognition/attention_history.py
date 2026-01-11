"""
Attention Allocation History: Track attention patterns and their outcomes.

This module tracks where attention is allocated, correlates allocations with
outcomes, and learns patterns to recommend future attention strategies.
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from ..workspace import WorkspaceState, Goal

logger = logging.getLogger(__name__)


@dataclass
class AttentionAllocation:
    """
    Record of where attention was allocated.
    
    Attributes:
        id: Unique identifier for this allocation
        timestamp: When allocation occurred
        allocation: What got attention and how much (item -> amount)
        total_available: Total attention budget available
        trigger: What caused this allocation decision
        workspace_state_hash: Hash of workspace state for correlation
    """
    id: str
    timestamp: datetime
    allocation: Dict[str, float]
    total_available: float
    trigger: str
    workspace_state_hash: str


@dataclass
class AttentionOutcome:
    """
    Outcome associated with an attention pattern.
    
    Attributes:
        allocation_id: ID of the attention allocation
        goal_progress: Progress made on each goal (goal_id -> progress delta)
        discoveries: What was noticed/learned
        missed: What was missed (known in retrospect)
        efficiency: How well was attention used (0.0-1.0)
    """
    allocation_id: str
    goal_progress: Dict[str, float]
    discoveries: List[str]
    missed: List[str]
    efficiency: float


@dataclass
class AttentionPattern:
    """
    A learned pattern about attention allocation.
    
    Attributes:
        pattern: Description of the attention pattern
        avg_efficiency: Average efficiency when using this pattern
        sample_size: Number of times pattern was observed
        recommendation: Actionable recommendation based on pattern
    """
    pattern: str
    avg_efficiency: float
    sample_size: int
    recommendation: str


class AttentionPatternLearner:
    """
    Learns effective attention allocation patterns.
    
    Analyzes attention-outcome pairs to identify what allocation
    strategies work well in different contexts.
    """
    
    def __init__(self):
        """Initialize attention pattern learner."""
        # Maps pattern key to list of efficiency scores
        self.pattern_outcomes: Dict[str, List[float]] = defaultdict(list)
        # Maps pattern key to list of allocation examples
        self.pattern_examples: Dict[str, List[AttentionAllocation]] = defaultdict(list)
    
    def learn(self, allocation: AttentionAllocation, outcome: AttentionOutcome):
        """
        Learn from an allocation-outcome pair.
        
        Args:
            allocation: The attention allocation
            outcome: The resulting outcome
        """
        pattern_key = self._extract_pattern_key(allocation)
        self.pattern_outcomes[pattern_key].append(outcome.efficiency)
        self.pattern_examples[pattern_key].append(allocation)
        
        logger.debug(
            f"Learned from attention pattern '{pattern_key}': "
            f"efficiency={outcome.efficiency:.2f}"
        )
    
    def _extract_pattern_key(self, allocation: AttentionAllocation) -> str:
        """
        Extract a pattern key from an allocation.
        
        Args:
            allocation: Attention allocation
            
        Returns:
            Pattern key string
        """
        # Categorize allocation by distribution pattern
        alloc_items = list(allocation.allocation.items())
        alloc_items.sort(key=lambda x: -x[1])  # Sort by amount
        
        if not alloc_items:
            return "empty_allocation"
        
        total = sum(v for _, v in alloc_items)
        if total == 0:
            return "no_attention"
        
        # Calculate concentration (is attention focused or distributed?)
        top_item_ratio = alloc_items[0][1] / total if total > 0 else 0
        
        if top_item_ratio > 0.7:
            return f"focused_on_{alloc_items[0][0]}"
        elif top_item_ratio > 0.4:
            return f"moderate_focus_{alloc_items[0][0]}"
        elif len(alloc_items) > 3:
            return "distributed_attention"
        else:
            return "balanced_attention"
    
    def get_patterns(self) -> List[AttentionPattern]:
        """
        Get learned patterns with sufficient data.
        
        Returns:
            List of attention patterns, sorted by efficiency
        """
        patterns = []
        
        for pattern_key, efficiencies in self.pattern_outcomes.items():
            if len(efficiencies) >= 5:  # Need at least 5 samples
                avg_efficiency = sum(efficiencies) / len(efficiencies)
                recommendation = self._generate_recommendation(pattern_key, efficiencies)
                
                patterns.append(AttentionPattern(
                    pattern=pattern_key,
                    avg_efficiency=avg_efficiency,
                    sample_size=len(efficiencies),
                    recommendation=recommendation
                ))
        
        # Sort by efficiency (best first)
        patterns.sort(key=lambda p: -p.avg_efficiency)
        return patterns
    
    def _generate_recommendation(
        self,
        pattern_key: str,
        efficiencies: List[float]
    ) -> str:
        """
        Generate actionable recommendation for a pattern.
        
        Args:
            pattern_key: Pattern identifier
            efficiencies: List of efficiency scores for this pattern
            
        Returns:
            Recommendation string
        """
        avg_eff = sum(efficiencies) / len(efficiencies)
        
        if avg_eff > 0.7:
            return f"Effective pattern - use when appropriate"
        elif avg_eff > 0.5:
            return f"Moderately effective - context dependent"
        else:
            return f"Low efficiency - avoid or investigate causes"
    
    def recommend(
        self,
        context: Any,  # WorkspaceState when TYPE_CHECKING
        goals: List[Any]  # List[Goal] when TYPE_CHECKING
    ) -> Dict[str, float]:
        """
        Recommend attention allocation based on learned patterns.
        
        Args:
            context: Current workspace state
            goals: Current goals
            
        Returns:
            Recommended allocation (item -> attention amount)
        """
        # Simple heuristic: allocate based on goal priorities
        # In a real implementation, this would use learned patterns more sophisticatedly
        
        if not goals:
            return {}
        
        allocation = {}
        total_priority = sum(
            getattr(g, 'priority', 0.5) for g in goals
        )
        
        if total_priority == 0:
            # Equal allocation
            per_goal = 1.0 / len(goals)
            for goal in goals:
                goal_id = getattr(goal, 'id', str(uuid.uuid4()))
                allocation[goal_id] = per_goal
        else:
            # Proportional to priority
            for goal in goals:
                goal_id = getattr(goal, 'id', str(uuid.uuid4()))
                priority = getattr(goal, 'priority', 0.5)
                allocation[goal_id] = priority / total_priority
        
        return allocation


class AttentionHistory:
    """
    Tracks attention allocation and learns from patterns.
    
    Records where attention is allocated, correlates with outcomes,
    and provides recommendations for future allocations based on
    learned effectiveness patterns.
    """
    
    # Class constant for memory management
    MAX_ALLOCATIONS = 1000
    
    def __init__(self):
        """Initialize attention history tracker."""
        self.allocations: List[AttentionAllocation] = []
        self.outcomes: Dict[str, AttentionOutcome] = {}
        self.pattern_learner = AttentionPatternLearner()
        
        logger.info("✅ AttentionHistory initialized")
    
    def record_allocation(
        self,
        allocation: Dict[str, float],
        trigger: str,
        workspace_state: Any
    ) -> str:
        """
        Record an attention allocation.
        
        Args:
            allocation: Attention allocation (item -> amount)
            trigger: What caused this allocation
            workspace_state: Current workspace state
            
        Returns:
            Allocation ID for later outcome recording
        """
        record = AttentionAllocation(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            allocation=allocation,
            total_available=sum(allocation.values()),
            trigger=trigger,
            workspace_state_hash=str(hash(str(workspace_state)))
        )
        
        self.allocations.append(record)
        
        logger.debug(
            f"Recorded attention allocation: {len(allocation)} items, "
            f"total={record.total_available:.2f}, trigger='{trigger}'"
        )
        
        # Keep only recent allocations
        if len(self.allocations) > self.MAX_ALLOCATIONS:
            # Remove old allocations and their outcomes
            old_ids = {a.id for a in self.allocations[:-self.MAX_ALLOCATIONS]}
            for old_id in old_ids:
                self.outcomes.pop(old_id, None)
            self.allocations = self.allocations[-self.MAX_ALLOCATIONS:]
        
        return record.id
    
    def record_outcome(
        self,
        allocation_id: str,
        goal_progress: Dict[str, float],
        discoveries: List[str],
        missed: List[str]
    ):
        """
        Record outcome of an attention allocation.
        
        Args:
            allocation_id: ID of the allocation
            goal_progress: Progress made on each goal
            discoveries: What was noticed/learned
            missed: What was missed
        """
        efficiency = self._compute_efficiency(goal_progress, discoveries, missed)
        
        outcome = AttentionOutcome(
            allocation_id=allocation_id,
            goal_progress=goal_progress,
            discoveries=discoveries,
            missed=missed,
            efficiency=efficiency
        )
        
        self.outcomes[allocation_id] = outcome
        
        # Learn from this pattern
        allocation = next(
            (a for a in self.allocations if a.id == allocation_id),
            None
        )
        
        if allocation:
            self.pattern_learner.learn(allocation, outcome)
            
            logger.debug(
                f"Recorded outcome for allocation {allocation_id}: "
                f"efficiency={efficiency:.2f}, discoveries={len(discoveries)}, "
                f"missed={len(missed)}"
            )
    
    def _compute_efficiency(
        self,
        goal_progress: Dict[str, float],
        discoveries: List[str],
        missed: List[str]
    ) -> float:
        """
        Compute efficiency of attention allocation.
        
        Args:
            goal_progress: Progress made on each goal
            discoveries: What was noticed/learned
            missed: What was missed
            
        Returns:
            Efficiency score (0.0-1.0)
        """
        # Base efficiency from goal progress
        if goal_progress:
            avg_progress = sum(goal_progress.values()) / len(goal_progress)
        else:
            avg_progress = 0.0
        
        # Bonus for discoveries
        discovery_bonus = min(0.2, len(discoveries) * 0.05)
        
        # Penalty for missed items
        missed_penalty = min(0.3, len(missed) * 0.1)
        
        efficiency = avg_progress + discovery_bonus - missed_penalty
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, efficiency))
    
    def get_recommended_allocation(
        self,
        context: Any,
        goals: List[Any]
    ) -> Dict[str, float]:
        """
        Get recommended attention allocation based on learned patterns.
        
        Args:
            context: Current workspace state
            goals: Current goals
            
        Returns:
            Recommended allocation (item -> attention amount)
        """
        return self.pattern_learner.recommend(context, goals)
    
    def get_attention_patterns(self) -> List[AttentionPattern]:
        """
        Get learned patterns about attention allocation.
        
        Returns:
            List of attention patterns, sorted by effectiveness
        """
        return self.pattern_learner.get_patterns()
    
    def get_recent_allocations(self, limit: int = 50) -> List[AttentionAllocation]:
        """
        Get recent attention allocations.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of recent allocations
        """
        return self.allocations[-limit:]
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about attention allocations.
        
        Returns:
            Dictionary with allocation statistics
        """
        if not self.allocations:
            return {
                "total_allocations": 0,
                "allocations_with_outcomes": 0,
                "avg_efficiency": 0.0,
                "patterns_learned": 0
            }
        
        outcomes_with_data = [
            o for o in self.outcomes.values()
            if o is not None
        ]
        
        avg_efficiency = (
            sum(o.efficiency for o in outcomes_with_data) / len(outcomes_with_data)
            if outcomes_with_data else 0.0
        )
        
        patterns = self.get_attention_patterns()
        
        return {
            "total_allocations": len(self.allocations),
            "allocations_with_outcomes": len(self.outcomes),
            "avg_efficiency": avg_efficiency,
            "patterns_learned": len(patterns),
            "best_pattern": patterns[0].pattern if patterns else None,
            "best_pattern_efficiency": patterns[0].avg_efficiency if patterns else 0.0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of attention history.
        
        Returns:
            Dictionary with all relevant statistics and patterns
        """
        stats = self.get_allocation_stats()
        patterns = self.get_attention_patterns()
        
        return {
            **stats,
            "top_patterns": [
                {
                    "pattern": p.pattern,
                    "efficiency": p.avg_efficiency,
                    "sample_size": p.sample_size,
                    "recommendation": p.recommendation
                }
                for p in patterns[:5]  # Top 5 patterns
            ]
        }
