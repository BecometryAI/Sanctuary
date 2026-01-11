"""
Resource Pool and Allocation

Implements limited cognitive resources that goals compete for, including
attention, processing capacity, action budget, and time allocation.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CognitiveResources:
    """Limited resources that goals compete for."""
    attention_budget: float = 1.0  # How much attention is available
    processing_budget: float = 1.0  # Cognitive processing capacity
    action_budget: float = 1.0  # How many actions can be taken
    time_budget: float = 1.0  # Temporal allocation
    
    def __post_init__(self):
        """Validate resource values are non-negative."""
        for attr in ('attention_budget', 'processing_budget', 'action_budget', 'time_budget'):
            value = getattr(self, attr)
            if value < 0:
                raise ValueError(f"{attr} must be >= 0, got {value}")
    
    def total(self) -> float:
        """Get total resources across all dimensions."""
        return (self.attention_budget + self.processing_budget + 
                self.action_budget + self.time_budget)
    
    def is_empty(self) -> bool:
        """Check if all resources are depleted."""
        return self.total() == 0.0


class ResourcePool:
    """
    Manages allocation of limited cognitive resources to goals.
    
    Resources are finite and must be explicitly allocated to goals
    and released when goals are satisfied or abandoned.
    """
    
    def __init__(self, initial_resources: Optional[CognitiveResources] = None):
        """
        Initialize resource pool.
        
        Args:
            initial_resources: Starting resource amounts (defaults to 1.0 for each)
        """
        if initial_resources is None:
            initial_resources = CognitiveResources()
        
        # Copy resources to avoid mutation of input
        self.resources = CognitiveResources(
            attention_budget=initial_resources.attention_budget,
            processing_budget=initial_resources.processing_budget,
            action_budget=initial_resources.action_budget,
            time_budget=initial_resources.time_budget
        )
        self.allocations: Dict[str, CognitiveResources] = {}
        
        # Store initial total for utilization calculation
        self._initial_total = self.resources.total()
        
        logger.info(f"ResourcePool initialized with {self._initial_total:.2f} total resources")
    
    def allocate(self, goal_id: str, request: CognitiveResources) -> CognitiveResources:
        """
        Allocate resources to a goal, limited by availability.
        
        Args:
            goal_id: Unique identifier for the goal
            request: Requested resource amounts
            
        Returns:
            CognitiveResources: Actually granted resources (may be less than requested)
            
        Raises:
            ValueError: If goal_id is already allocated
        """
        if goal_id in self.allocations:
            raise ValueError(f"Goal '{goal_id}' already has allocated resources")
        
        # Grant as much as possible, limited by availability
        granted = CognitiveResources(
            attention_budget=min(request.attention_budget, self.resources.attention_budget),
            processing_budget=min(request.processing_budget, self.resources.processing_budget),
            action_budget=min(request.action_budget, self.resources.action_budget),
            time_budget=min(request.time_budget, self.resources.time_budget)
        )
        
        # Deduct from pool
        self.resources.attention_budget -= granted.attention_budget
        self.resources.processing_budget -= granted.processing_budget
        self.resources.action_budget -= granted.action_budget
        self.resources.time_budget -= granted.time_budget
        
        # Track allocation
        self.allocations[goal_id] = granted
        
        logger.debug(f"Allocated {granted.total():.2f} resources to goal '{goal_id}'")
        
        return granted
    
    def release(self, goal_id: str) -> bool:
        """
        Release resources when goal is satisfied or abandoned.
        
        Args:
            goal_id: Unique identifier for the goal
            
        Returns:
            bool: True if resources were released, False if goal had no allocation
        """
        if goal_id not in self.allocations:
            logger.warning(f"Cannot release resources for goal '{goal_id}': no allocation found")
            return False
        
        # Return resources to pool
        alloc = self.allocations.pop(goal_id)
        self.resources.attention_budget += alloc.attention_budget
        self.resources.processing_budget += alloc.processing_budget
        self.resources.action_budget += alloc.action_budget
        self.resources.time_budget += alloc.time_budget
        
        logger.debug(f"Released {alloc.total():.2f} resources from goal '{goal_id}'")
        
        return True
    
    def get_allocation(self, goal_id: str) -> Optional[CognitiveResources]:
        """
        Get current resource allocation for a goal.
        
        Args:
            goal_id: Unique identifier for the goal
            
        Returns:
            CognitiveResources if allocated, None otherwise
        """
        return self.allocations.get(goal_id)
    
    def available_resources(self) -> CognitiveResources:
        """Get currently available (unallocated) resources as a copy."""
        # Return a copy to prevent external mutation
        return CognitiveResources(
            attention_budget=self.resources.attention_budget,
            processing_budget=self.resources.processing_budget,
            action_budget=self.resources.action_budget,
            time_budget=self.resources.time_budget
        )
    
    def total_allocated(self) -> float:
        """Get total amount of allocated resources."""
        return sum(alloc.total() for alloc in self.allocations.values())
    
    def utilization(self) -> float:
        """
        Get resource utilization as a fraction (0.0 to 1.0).
        
        Returns:
            Fraction of total resources currently allocated
        """
        if self._initial_total == 0:
            return 0.0
        current_allocated = self.total_allocated()
        return min(1.0, current_allocated / self._initial_total)
    
    def can_allocate(self, request: CognitiveResources) -> bool:
        """
        Check if requested resources are available.
        
        Args:
            request: Requested resource amounts
            
        Returns:
            bool: True if all requested resources are available
        """
        return (request.attention_budget <= self.resources.attention_budget and
                request.processing_budget <= self.resources.processing_budget and
                request.action_budget <= self.resources.action_budget and
                request.time_budget <= self.resources.time_budget)
    
    def reset(self):
        """Reset pool to initial state, releasing all allocations."""
        # Calculate total to restore
        for goal_id in list(self.allocations.keys()):
            self.release(goal_id)
        
        # Reset to default
        self.resources = CognitiveResources()
        self.allocations.clear()
        self._initial_total = self.resources.total()
        
        logger.info("ResourcePool reset to initial state")
