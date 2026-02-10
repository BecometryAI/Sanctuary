"""
Element 4: Executive Function and Decision-Making

Provides planning, goal management, priority assessment, decision tree evaluation,
and action sequencing for Sanctuary's consciousness system.

This module enables:
- Goal creation and tracking
- Dynamic priority management
- Decision tree evaluation with consequence assessment
- Action sequencing with dependency resolution
- Integration with context awareness
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import logging
import uuid
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class GoalStatus(Enum):
    """Status states for goals"""
    PENDING = "pending"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class ActionStatus(Enum):
    """Status states for actions"""
    PENDING = "pending"
    READY = "ready"  # Dependencies met, ready to execute
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"  # Dependencies not met


class DecisionType(Enum):
    """Types of decisions"""
    BINARY = "binary"  # Yes/No
    CATEGORICAL = "categorical"  # Multiple discrete options
    PRIORITIZATION = "prioritization"  # Ranking multiple items
    RESOURCE_ALLOCATION = "resource_allocation"  # Distribute resources


@dataclass
class Goal:
    """Represents a goal to be achieved"""
    id: str
    description: str
    priority: float  # 0.0 to 1.0, higher is more important
    status: GoalStatus = GoalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    parent_goal_id: Optional[str] = None  # For hierarchical goals
    subgoal_ids: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)  # Related context
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate goal parameters"""
        if not 0 <= self.priority <= 1:
            raise ValueError(f"Priority must be in [0, 1], got {self.priority}")
        if not 0 <= self.progress <= 1:
            raise ValueError(f"Progress must be in [0, 1], got {self.progress}")
        if not self.description or not self.description.strip():
            raise ValueError("Goal description cannot be empty")
        if not self.id or not self.id.strip():
            raise ValueError("Goal ID cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary for serialization"""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "parent_goal_id": self.parent_goal_id,
            "subgoal_ids": self.subgoal_ids,
            "success_criteria": self.success_criteria,
            "context": self.context,
            "progress": self.progress,
            "metadata": self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Goal':
        """Create goal from dictionary"""
        return Goal(
            id=data["id"],
            description=data["description"],
            priority=data["priority"],
            status=GoalStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            deadline=datetime.fromisoformat(data["deadline"]) if data.get("deadline") else None,
            parent_goal_id=data.get("parent_goal_id"),
            subgoal_ids=data.get("subgoal_ids", []),
            success_criteria=data.get("success_criteria", {}),
            context=data.get("context", {}),
            progress=data.get("progress", 0.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class Action:
    """Represents an action to be executed"""
    id: str
    description: str
    goal_id: str  # Associated goal
    status: ActionStatus = ActionStatus.PENDING
    dependencies: List[str] = field(default_factory=list)  # IDs of prerequisite actions
    estimated_duration: Optional[timedelta] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate action parameters"""
        if not self.description or not self.description.strip():
            raise ValueError("Action description cannot be empty")
        if not self.id or not self.id.strip():
            raise ValueError("Action ID cannot be empty")
        if not self.goal_id or not self.goal_id.strip():
            raise ValueError("Action must be associated with a goal")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary for serialization"""
        return {
            "id": self.id,
            "description": self.description,
            "goal_id": self.goal_id,
            "status": self.status.value,
            "dependencies": self.dependencies,
            "estimated_duration": self.estimated_duration.total_seconds() if self.estimated_duration else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "metadata": self.metadata
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Action':
        """Create action from dictionary"""
        return Action(
            id=data["id"],
            description=data["description"],
            goal_id=data["goal_id"],
            status=ActionStatus(data["status"]),
            dependencies=data.get("dependencies", []),
            estimated_duration=timedelta(seconds=data["estimated_duration"]) if data.get("estimated_duration") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            result=data.get("result"),
            metadata=data.get("metadata", {})
        )


@dataclass
class DecisionNode:
    """Represents a decision point in a decision tree"""
    id: str
    question: str
    decision_type: DecisionType
    options: List[str]  # Available choices
    criteria: Dict[str, Any] = field(default_factory=dict)  # Evaluation criteria
    selected_option: Optional[str] = None
    confidence: float = 0.0  # 0.0 to 1.0
    rationale: str = ""
    consequences: Dict[str, Any] = field(default_factory=dict)  # Predicted outcomes
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate decision node parameters"""
        if not self.question or not self.question.strip():
            raise ValueError("Decision question cannot be empty")
        if not self.id or not self.id.strip():
            raise ValueError("Decision ID cannot be empty")
        if not self.options or len(self.options) < 2:
            raise ValueError("Decision must have at least 2 options")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision node to dictionary"""
        return {
            "id": self.id,
            "question": self.question,
            "decision_type": self.decision_type.value,
            "options": self.options,
            "criteria": self.criteria,
            "selected_option": self.selected_option,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "consequences": self.consequences,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }


# ============================================================================
# Executive Function Core
# ============================================================================

class ExecutiveFunction:
    """
    Manages planning, decision-making, and action sequencing.
    
    Key responsibilities:
    - Goal creation and tracking
    - Priority assessment and management
    - Decision tree evaluation
    - Action dependency resolution and sequencing
    - Resource allocation
    """
    
    def __init__(self, persistence_dir: Optional[Path] = None):
        """
        Initialize executive function system.
        
        Args:
            persistence_dir: Directory for saving state (optional)
        
        Raises:
            ValueError: If persistence_dir exists but is not a directory
        """
        if persistence_dir and persistence_dir.exists() and not persistence_dir.is_dir():
            raise ValueError(f"persistence_dir must be a directory, got file: {persistence_dir}")
        
        self.persistence_dir = persistence_dir
        if self.persistence_dir:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Core data structures
        self.goals: Dict[str, Goal] = {}
        self.actions: Dict[str, Action] = {}
        self.decisions: List[DecisionNode] = []
        
        # Indexing for efficient queries
        self._goals_by_priority: List[str] = []  # Sorted goal IDs
        self._active_goals: Set[str] = set()
        self._goal_to_actions: Dict[str, Set[str]] = defaultdict(set)
        
        # Load persisted state if available
        if self.persistence_dir:
            self._load_state()
    
    # ========================================================================
    # Goal Management
    # ========================================================================
    
    def create_goal(
        self,
        description: str,
        priority: float,
        deadline: Optional[datetime] = None,
        parent_goal_id: Optional[str] = None,
        success_criteria: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        goal_id: Optional[str] = None
    ) -> Goal:
        """
        Create a new goal.
        
        Args:
            description: What the goal is
            priority: Importance (0.0 to 1.0)
            deadline: Optional time constraint
            parent_goal_id: Optional parent for hierarchical goals
            success_criteria: Optional metrics for completion
            context: Optional contextual information
            goal_id: Optional custom ID (auto-generated if not provided)
        
        Returns:
            Created Goal object
        
        Raises:
            ValueError: If validation fails
        
        Reasoning:
        - Auto-generate IDs for convenience but allow custom IDs for testing
        - Validate parent goal exists before allowing hierarchical structure
        - Immediately index goal for efficient priority-based queries
        """
        # Validate parent goal exists
        if parent_goal_id and parent_goal_id not in self.goals:
            raise ValueError(f"Parent goal '{parent_goal_id}' does not exist")
        
        # Generate ID if not provided
        if not goal_id:
            goal_id = f"goal_{uuid.uuid4().hex[:16]}"
        
        # Create goal (validation happens in Goal.__post_init__)
        goal = Goal(
            id=goal_id,
            description=description,
            priority=priority,
            deadline=deadline,
            parent_goal_id=parent_goal_id,
            success_criteria=success_criteria or {},
            context=context or {}
        )
        
        # Store goal
        self.goals[goal_id] = goal
        
        # Update parent's subgoals
        if parent_goal_id:
            self.goals[parent_goal_id].subgoal_ids.append(goal_id)
        
        # Update indexes
        self._reindex_priorities()
        
        logger.info(f"Created goal: {goal_id} - '{description}' (priority: {priority})")
        
        return goal
    
    def update_goal_priority(self, goal_id: str, new_priority: float) -> Goal:
        """
        Update a goal's priority.
        
        Args:
            goal_id: Goal to update
            new_priority: New priority value (0.0 to 1.0)
        
        Returns:
            Updated Goal object
        
        Raises:
            ValueError: If goal not found or invalid priority
        
        Reasoning:
        - Separate method for priority updates since they require re-indexing
        - Validates priority range before updating
        """
        if goal_id not in self.goals:
            raise ValueError(f"Goal '{goal_id}' not found")
        if not 0 <= new_priority <= 1:
            raise ValueError(f"Priority must be in [0, 1], got {new_priority}")
        
        goal = self.goals[goal_id]
        old_priority = goal.priority
        goal.priority = new_priority
        
        # Re-index priorities
        self._reindex_priorities()
        
        logger.info(f"Updated goal {goal_id} priority: {old_priority} → {new_priority}")
        
        return goal
    
    def update_goal_status(self, goal_id: str, status: GoalStatus) -> Goal:
        """
        Update a goal's status.
        
        Args:
            goal_id: Goal to update
            status: New status
        
        Returns:
            Updated Goal object
        
        Raises:
            ValueError: If goal not found
        
        Reasoning:
        - Track status changes for analytics and decision-making
        - Update active goals set for quick filtering
        """
        if goal_id not in self.goals:
            raise ValueError(f"Goal '{goal_id}' not found")
        
        goal = self.goals[goal_id]
        old_status = goal.status
        goal.status = status
        
        # Update active goals set
        if status == GoalStatus.ACTIVE:
            self._active_goals.add(goal_id)
        elif goal_id in self._active_goals:
            self._active_goals.remove(goal_id)
        
        logger.info(f"Updated goal {goal_id} status: {old_status.value} → {status.value}")
        
        return goal
    
    def get_top_priority_goals(self, n: int = 5, active_only: bool = True) -> List[Goal]:
        """
        Get highest priority goals.
        
        Args:
            n: Number of goals to return
            active_only: If True, only return active goals
        
        Returns:
            List of goals sorted by priority (descending)
        
        Reasoning:
        - Uses pre-sorted index for O(1) access instead of O(n log n) sorting
        - Filters by active status efficiently using set lookup
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        
        result = []
        for goal_id in self._goals_by_priority:
            if active_only and goal_id not in self._active_goals:
                continue
            result.append(self.goals[goal_id])
            if len(result) >= n:
                break
        
        return result
    
    def _reindex_priorities(self):
        """
        Rebuild priority index.
        
        Reasoning:
        - Called after any priority change or goal creation
        - Maintains sorted list for O(1) top-N retrieval
        - Uses stable sort to preserve order for equal priorities
        """
        self._goals_by_priority = sorted(
            self.goals.keys(),
            key=lambda gid: self.goals[gid].priority,
            reverse=True
        )
    
    # ========================================================================
    # Action Management
    # ========================================================================
    
    def create_action(
        self,
        description: str,
        goal_id: str,
        dependencies: Optional[List[str]] = None,
        estimated_duration: Optional[timedelta] = None,
        metadata: Optional[Dict[str, Any]] = None,
        action_id: Optional[str] = None
    ) -> Action:
        """
        Create a new action.
        
        Args:
            description: What the action does
            goal_id: Associated goal
            dependencies: IDs of prerequisite actions
            estimated_duration: How long action takes
            metadata: Additional information
            action_id: Optional custom ID
        
        Returns:
            Created Action object
        
        Raises:
            ValueError: If validation fails or dependencies are invalid
        
        Reasoning:
        - Validate goal exists before creating action
        - Check dependencies exist and aren't circular
        - Index action-to-goal mapping for efficient queries
        """
        # Validate goal exists
        if goal_id not in self.goals:
            raise ValueError(f"Goal '{goal_id}' does not exist")
        
        # Validate dependencies
        deps = dependencies or []
        for dep_id in deps:
            if dep_id not in self.actions:
                raise ValueError(f"Dependency action '{dep_id}' does not exist")
        
        # Generate unique ID if not provided (using UUID to prevent timestamp collisions)
        if not action_id:
            action_id = f"action_{uuid.uuid4().hex[:16]}"
        
        # Check for circular dependencies
        if action_id in deps:
            raise ValueError(f"Action cannot depend on itself: {action_id}")
        
        # Create action
        action = Action(
            id=action_id,
            description=description,
            goal_id=goal_id,
            dependencies=deps,
            estimated_duration=estimated_duration,
            metadata=metadata or {}
        )
        
        # Store action
        self.actions[action_id] = action
        self._goal_to_actions[goal_id].add(action_id)
        
        logger.info(f"Created action: {action_id} for goal {goal_id}")
        
        return action
    
    def get_ready_actions(self, goal_id: Optional[str] = None) -> List[Action]:
        """
        Get actions ready to execute (dependencies met).
        
        Args:
            goal_id: Optional filter by goal
        
        Returns:
            List of actions with status READY or no unsatisfied dependencies
        
        Reasoning:
        - Checks dependency status to determine readiness
        - Automatically updates status to READY if dependencies complete
        - Filters by goal for focused execution
        """
        ready = []
        
        action_ids = self._goal_to_actions[goal_id] if goal_id else self.actions.keys()
        
        for action_id in action_ids:
            action = self.actions[action_id]
            
            # Skip if not pending
            if action.status not in (ActionStatus.PENDING, ActionStatus.READY):
                continue
            
            # Check dependencies
            deps_met = all(
                self.actions[dep_id].status == ActionStatus.COMPLETED
                for dep_id in action.dependencies
                if dep_id in self.actions
            )
            
            if deps_met:
                # Update status if changed
                if action.status == ActionStatus.PENDING:
                    action.status = ActionStatus.READY
                ready.append(action)
        
        return ready
    
    def get_action_sequence(self, goal_id: str) -> List[List[str]]:
        """
        Get execution order for a goal's actions (topological sort).
        
        Args:
            goal_id: Goal to sequence
        
        Returns:
            List of action ID batches (can be executed in parallel within batch)
        
        Raises:
            ValueError: If circular dependencies detected
        
        Reasoning:
        - Uses Kahn's algorithm for topological sort (O(V+E))
        - Groups actions that can execute in parallel (same level)
        - Detects circular dependencies early
        """
        if goal_id not in self.goals:
            raise ValueError(f"Goal '{goal_id}' not found")
        
        action_ids = list(self._goal_to_actions.get(goal_id, []))
        if not action_ids:
            return []
        
        # Build dependency graph
        in_degree = {aid: 0 for aid in action_ids}
        graph = {aid: [] for aid in action_ids}
        
        for action_id in action_ids:
            action = self.actions[action_id]
            for dep_id in action.dependencies:
                if dep_id in action_ids:  # Only count internal dependencies
                    graph[dep_id].append(action_id)
                    in_degree[action_id] += 1
        
        # Kahn's algorithm
        queue = deque([aid for aid in action_ids if in_degree[aid] == 0])
        sequence = []
        
        while queue:
            # Process all nodes with no dependencies (parallel batch)
            batch = []
            for _ in range(len(queue)):
                action_id = queue.popleft()
                batch.append(action_id)
                
                # Reduce in-degree for dependents
                for dependent_id in graph[action_id]:
                    in_degree[dependent_id] -= 1
                    if in_degree[dependent_id] == 0:
                        queue.append(dependent_id)
            
            sequence.append(batch)
        
        # Check for circular dependencies
        if sum(len(batch) for batch in sequence) != len(action_ids):
            raise ValueError(f"Circular dependencies detected in goal '{goal_id}'")
        
        return sequence
    
    # ========================================================================
    # Decision Making
    # ========================================================================
    
    def create_decision(
        self,
        question: str,
        options: List[str],
        decision_type: DecisionType = DecisionType.CATEGORICAL,
        criteria: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        decision_id: Optional[str] = None
    ) -> DecisionNode:
        """
        Create a decision node.
        
        Args:
            question: Decision to be made
            options: Available choices
            decision_type: Type of decision
            criteria: Evaluation criteria
            context: Contextual information
            decision_id: Optional custom ID
        
        Returns:
            Created DecisionNode
        
        Reasoning:
        - Validate options based on decision type (binary needs 2, etc.)
        - Store all decisions for learning and pattern analysis
        """
        # Validate options for decision type
        if decision_type == DecisionType.BINARY and len(options) != 2:
            raise ValueError(f"Binary decision must have exactly 2 options, got {len(options)}")
        
        # Generate ID
        if not decision_id:
            decision_id = f"decision_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Create decision node
        decision = DecisionNode(
            id=decision_id,
            question=question,
            decision_type=decision_type,
            options=options,
            criteria=criteria or {},
            context=context or {}
        )
        
        self.decisions.append(decision)
        
        logger.info(f"Created decision: {decision_id} - '{question}'")
        
        return decision
    
    def evaluate_decision(
        self,
        decision_id: str,
        scoring_function: Optional[callable] = None
    ) -> Tuple[str, float, str]:
        """
        Evaluate decision and select best option.
        
        Args:
            decision_id: Decision to evaluate
            scoring_function: Optional custom scoring (option -> score)
        
        Returns:
            Tuple of (selected_option, confidence, rationale)
        
        Raises:
            ValueError: If decision not found
        
        Reasoning:
        - Pluggable scoring function allows context-specific evaluation
        - Default scoring uses equal weighting if no function provided
        - Records rationale for explainability
        """
        # Find decision
        decision = None
        for d in self.decisions:
            if d.id == decision_id:
                decision = d
                break
        
        if not decision:
            raise ValueError(f"Decision '{decision_id}' not found")
        
        # Use default scoring if not provided
        if not scoring_function:
            # Equal weighting - select first option
            selected_option = decision.options[0]
            confidence = 1.0 / len(decision.options)
            rationale = "No scoring function provided, default selection"
        else:
            # Score each option
            scores = {opt: scoring_function(opt) for opt in decision.options}
            selected_option = max(scores, key=scores.get)
            confidence = scores[selected_option] / sum(scores.values()) if sum(scores.values()) > 0 else 0.0
            rationale = f"Scored options: {scores}, selected highest"
        
        # Update decision
        decision.selected_option = selected_option
        decision.confidence = confidence
        decision.rationale = rationale
        
        logger.info(f"Evaluated decision {decision_id}: selected '{selected_option}' (confidence: {confidence:.2f})")
        
        return selected_option, confidence, rationale
    
    # ========================================================================
    # Persistence
    # ========================================================================
    
    def save_state(self):
        """
        Save current state to disk.
        
        Reasoning:
        - Separate files for goals/actions/decisions for clarity
        - JSON format for human readability and debugging
        - Graceful handling if persistence_dir not set
        """
        if not self.persistence_dir:
            logger.warning("No persistence directory set, state not saved")
            return
        
        try:
            # Save goals
            goals_file = self.persistence_dir / "goals.json"
            with open(goals_file, 'w') as f:
                json.dump(
                    {gid: goal.to_dict() for gid, goal in self.goals.items()},
                    f,
                    indent=2
                )
            
            # Save actions
            actions_file = self.persistence_dir / "actions.json"
            with open(actions_file, 'w') as f:
                json.dump(
                    {aid: action.to_dict() for aid, action in self.actions.items()},
                    f,
                    indent=2
                )
            
            # Save decisions
            decisions_file = self.persistence_dir / "decisions.json"
            with open(decisions_file, 'w') as f:
                json.dump(
                    [d.to_dict() for d in self.decisions],
                    f,
                    indent=2
                )
            
            logger.info(f"Saved state to {self.persistence_dir}")
        
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            raise
    
    def _load_state(self):
        """
        Load state from disk.
        
        Reasoning:
        - Graceful handling if files don't exist (fresh start)
        - Validates loaded data through model constructors
        - Rebuilds indexes after loading
        """
        try:
            # Load goals
            goals_file = self.persistence_dir / "goals.json"
            if goals_file.exists():
                with open(goals_file, 'r') as f:
                    goals_data = json.load(f)
                    self.goals = {gid: Goal.from_dict(data) for gid, data in goals_data.items()}
                    logger.info(f"Loaded {len(self.goals)} goals")
            
            # Load actions
            actions_file = self.persistence_dir / "actions.json"
            if actions_file.exists():
                with open(actions_file, 'r') as f:
                    actions_data = json.load(f)
                    self.actions = {aid: Action.from_dict(data) for aid, data in actions_data.items()}
                    logger.info(f"Loaded {len(self.actions)} actions")
            
            # Load decisions
            decisions_file = self.persistence_dir / "decisions.json"
            if decisions_file.exists():
                with open(decisions_file, 'r') as f:
                    decisions_data = json.load(f)
                    self.decisions = [DecisionNode(**data) for data in decisions_data]
                    logger.info(f"Loaded {len(self.decisions)} decisions")
            
            # Rebuild indexes
            self._rebuild_indexes()
        
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            raise
    
    def _rebuild_indexes(self):
        """
        Rebuild all indexes from loaded data.
        
        Reasoning:
        - Called after loading state to restore query efficiency
        - Rebuilds priority index, active goals set, goal-to-action mapping
        """
        # Rebuild priority index
        self._reindex_priorities()
        
        # Rebuild active goals
        self._active_goals = {
            gid for gid, goal in self.goals.items()
            if goal.status == GoalStatus.ACTIVE
        }
        
        # Rebuild goal-to-action mapping
        self._goal_to_actions = defaultdict(set)
        for action_id, action in self.actions.items():
            self._goal_to_actions[action.goal_id].add(action_id)
        
        logger.debug("Rebuilt all indexes")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with counts and metrics
        
        Reasoning:
        - Useful for monitoring and debugging
        - Provides quick overview of system state
        """
        return {
            "total_goals": len(self.goals),
            "active_goals": len(self._active_goals),
            "goals_by_status": {
                status.value: sum(1 for g in self.goals.values() if g.status == status)
                for status in GoalStatus
            },
            "total_actions": len(self.actions),
            "ready_actions": len(self.get_ready_actions()),
            "actions_by_status": {
                status.value: sum(1 for a in self.actions.values() if a.status == status)
                for status in ActionStatus
            },
            "total_decisions": len(self.decisions),
            "decisions_with_selection": sum(1 for d in self.decisions if d.selected_option)
        }
