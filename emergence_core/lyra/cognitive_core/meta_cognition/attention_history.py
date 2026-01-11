"""
Attention History: Tracks attention allocation patterns and their outcomes.

This module implements tracking of attention allocation decisions and their
outcomes, learning which attention patterns are most effective for different
contexts and goals.
"""

from __future__ import annotations

import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AttentionAllocation:
    """Record of where attention was allocated."""
    id: str
    timestamp: datetime
    allocation: Dict[str, float]  # What got attention and how much
    total_available: float
    trigger: str  # What caused this allocation
    workspace_state_hash: str  # For correlating with outcomes
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionOutcome:
    """Outcome associated with an attention pattern."""
    allocation_id: str
    goal_progress: Dict[str, float]  # Goal ID -> progress made
    discoveries: List[str]  # What was noticed/learned
    missed: List[str]  # What was missed (known in retrospect)
    efficiency: float  # How well was attention used?
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionPattern:
    """A learned pattern about attention allocation."""
    pattern: str
    avg_efficiency: float
    sample_size: int
    recommendation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttentionPatternLearner:
    """Learns effective attention allocation patterns."""
    
    def __init__(self):
        self.pattern_outcomes: Dict[str, List[float]] = defaultdict(list)
        self.pattern_contexts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.min_samples = 5
    
    def learn(self, allocation: AttentionAllocation, outcome: AttentionOutcome):
        """Learn from an allocation-outcome pair."""
        pattern_key = self._extract_pattern_key(allocation)
        self.pattern_outcomes[pattern_key].append(outcome.efficiency)
        
        # Store context for later analysis
        context = {
            "trigger": allocation.trigger,
            "total_available": allocation.total_available,
            "num_targets": len(allocation.allocation)
        }
        self.pattern_contexts[pattern_key].append(context)
    
    def _extract_pattern_key(self, allocation: AttentionAllocation) -> str:
        """Extract a pattern identifier from allocation."""
        # Categorize allocation pattern
        num_targets = len(allocation.allocation)
        
        if num_targets == 0:
            return "no_allocation"
        elif num_targets == 1:
            return "focused_single"
        elif num_targets <= 3:
            return "focused_few"
        else:
            # Check if attention is spread evenly or concentrated
            values = list(allocation.allocation.values())
            if not values:
                return "unknown"
            
            max_val = max(values)
            avg_val = sum(values) / len(values)
            
            if max_val > avg_val * 2:
                return "concentrated_many"
            else:
                return "distributed_many"
    
    def recommend(self, context: Any, goals: List[Any]) -> Dict[str, float]:
        """Get recommended attention allocation based on learned patterns."""
        # Get best performing pattern
        patterns = self.get_patterns()
        
        if not patterns:
            # No learned patterns, return default
            return {"default": 1.0}
        
        # Use highest efficiency pattern
        best_pattern = patterns[0]
        
        # Generate allocation based on pattern
        if best_pattern.pattern == "focused_single":
            # Allocate all to highest priority goal
            if goals:
                highest_priority = max(goals, 
                                     key=lambda g: getattr(g, 'priority', 0.5))
                goal_id = getattr(highest_priority, 'id', 'unknown')
                return {goal_id: 1.0}
            return {"default": 1.0}
        
        elif best_pattern.pattern == "focused_few":
            # Allocate to top 3 goals
            if goals:
                sorted_goals = sorted(goals, 
                                    key=lambda g: getattr(g, 'priority', 0.5),
                                    reverse=True)[:3]
                allocation = {}
                total_priority = sum(getattr(g, 'priority', 0.5) 
                                   for g in sorted_goals)
                for goal in sorted_goals:
                    goal_id = getattr(goal, 'id', 'unknown')
                    priority = getattr(goal, 'priority', 0.5)
                    allocation[goal_id] = priority / total_priority if total_priority > 0 else 1.0 / len(sorted_goals)
                return allocation
            return {"default": 1.0}
        
        else:
            # Distribute across all goals
            if goals:
                allocation = {}
                for goal in goals:
                    goal_id = getattr(goal, 'id', 'unknown')
                    allocation[goal_id] = 1.0 / len(goals)
                return allocation
            return {"default": 1.0}
    
    def get_patterns(self) -> List[AttentionPattern]:
        """Get patterns with sufficient data."""
        patterns = []
        
        for pattern_key, efficiencies in self.pattern_outcomes.items():
            if len(efficiencies) >= self.min_samples:
                avg_efficiency = sum(efficiencies) / len(efficiencies)
                recommendation = self._generate_recommendation(pattern_key, efficiencies)
                
                patterns.append(AttentionPattern(
                    pattern=pattern_key,
                    avg_efficiency=avg_efficiency,
                    sample_size=len(efficiencies),
                    recommendation=recommendation
                ))
        
        # Sort by efficiency
        patterns.sort(key=lambda p: -p.avg_efficiency)
        return patterns
    
    def _generate_recommendation(self, pattern_key: str, 
                                 efficiencies: List[float]) -> str:
        """Generate recommendation based on pattern performance."""
        avg_eff = sum(efficiencies) / len(efficiencies)
        
        if avg_eff > 0.7:
            return f"Pattern '{pattern_key}' is highly effective - use when possible"
        elif avg_eff > 0.5:
            return f"Pattern '{pattern_key}' is moderately effective - consider contextually"
        else:
            return f"Pattern '{pattern_key}' is less effective - avoid unless necessary"


class AttentionHistory:
    """Tracks attention allocation and learns from patterns."""
    
    # Efficiency calculation constants
    MAX_DISCOVERY_BONUS = 0.2
    DISCOVERY_BONUS_RATE = 0.05
    MAX_MISSED_PENALTY = 0.3
    MISSED_PENALTY_RATE = 0.1
    
    def __init__(self, config: Optional[Dict] = None):
        self.allocations: List[AttentionAllocation] = []
        self.outcomes: Dict[str, AttentionOutcome] = {}
        self.pattern_learner = AttentionPatternLearner()
        self.config = config or {}
        self.max_allocations = self.config.get("max_allocations", 1000)
        
        logger.info("✅ AttentionHistory initialized")
    
    def record_allocation(self, allocation: Dict[str, float],
                         trigger: str, workspace_state: Any) -> str:
        """Record an attention allocation."""
        # Input validation
        if not isinstance(allocation, dict):
            raise TypeError("allocation must be a dictionary")
        if not isinstance(trigger, str) or not trigger:
            raise ValueError("trigger must be a non-empty string")
        
        # Compute state hash efficiently
        state_hash = str(hash(str(workspace_state)))
        
        record = AttentionAllocation(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            allocation=allocation.copy(),
            total_available=sum(allocation.values()),
            trigger=trigger,
            workspace_state_hash=state_hash
        )
        
        self.allocations.append(record)
        
        # Limit memory usage
        if len(self.allocations) > self.max_allocations:
            self.allocations = self.allocations[-self.max_allocations:]
        
        logger.debug(f"👁️ Recorded attention allocation: {trigger} "
                    f"(targets={len(allocation)})")
        
        return record.id
    
    def record_outcome(self, allocation_id: str, goal_progress: Dict[str, float],
                      discoveries: List[str], missed: List[str]):
        """Record outcome of an attention allocation."""
        efficiency = self._compute_efficiency(goal_progress, discoveries, missed)
        
        outcome = AttentionOutcome(
            allocation_id=allocation_id,
            goal_progress=goal_progress.copy(),
            discoveries=discoveries.copy(),
            missed=missed.copy(),
            efficiency=efficiency
        )
        
        self.outcomes[allocation_id] = outcome
        
        # Learn from this pattern
        allocation = next((a for a in self.allocations if a.id == allocation_id), None)
        if allocation:
            self.pattern_learner.learn(allocation, outcome)
            
            logger.debug(f"📊 Recorded attention outcome: efficiency={efficiency:.2f}")
    
    def _compute_efficiency(self, goal_progress: Dict[str, float],
                           discoveries: List[str], missed: List[str]) -> float:
        """Compute efficiency of attention allocation."""
        # Base efficiency on goal progress
        progress_score = sum(goal_progress.values()) / max(1, len(goal_progress)) if goal_progress else 0.0
        
        # Bonus for discoveries
        discovery_bonus = min(self.MAX_DISCOVERY_BONUS, len(discoveries) * self.DISCOVERY_BONUS_RATE)
        
        # Penalty for missed items
        missed_penalty = min(self.MAX_MISSED_PENALTY, len(missed) * self.MISSED_PENALTY_RATE)
        
        efficiency = max(0.0, min(1.0, progress_score + discovery_bonus - missed_penalty))
        
        return efficiency
    
    def get_recommended_allocation(self, context: Any,
                                  goals: List[Any]) -> Dict[str, float]:
        """Get recommended attention allocation based on learned patterns."""
        return self.pattern_learner.recommend(context, goals)
    
    def get_attention_patterns(self) -> List[AttentionPattern]:
        """Get learned patterns about attention allocation."""
        return self.pattern_learner.get_patterns()
    
    def get_allocation_history(self, limit: int = 10) -> List[AttentionAllocation]:
        """Get recent allocation history."""
        return self.allocations[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of attention history."""
        patterns = self.get_attention_patterns()
        
        # Calculate average efficiency
        efficiencies = [o.efficiency for o in self.outcomes.values()]
        avg_efficiency = sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
        
        # Count triggers
        trigger_counts: Dict[str, int] = defaultdict(int)
        for allocation in self.allocations:
            trigger_counts[allocation.trigger] += 1
        
        return {
            "total_allocations": len(self.allocations),
            "total_outcomes": len(self.outcomes),
            "patterns_learned": len(patterns),
            "avg_efficiency": avg_efficiency,
            "top_triggers": sorted(trigger_counts.items(), 
                                  key=lambda x: -x[1])[:5],
            "best_pattern": patterns[0].pattern if patterns else None,
            "patterns": [
                {
                    "pattern": p.pattern,
                    "efficiency": p.avg_efficiency,
                    "sample_size": p.sample_size
                }
                for p in patterns
            ]
        }
