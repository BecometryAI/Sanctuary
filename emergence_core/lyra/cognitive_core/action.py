"""
Action Subsystem: Goal-directed behavior generation.

This module implements the ActionSubsystem class, which decides what actions to take
based on the current GlobalWorkspace state. It implements goal-directed behavior
using current goals, emotions, and percepts to select and execute appropriate actions.

The action subsystem is responsible for:
- Translating workspace state into concrete actions
- Managing action repertoire and capabilities
- Prioritizing competing action tendencies
- Coordinating multi-step action sequences
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List, Callable
from collections import deque
from enum import Enum

from pydantic import BaseModel, Field

from .workspace import WorkspaceSnapshot, GoalType

# Configure logging
logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """
    Categories of actions the system can perform.

    SPEAK: Generate language output
    COMMIT_MEMORY: Store to long-term memory
    RETRIEVE_MEMORY: Search memory
    INTROSPECT: Self-reflection
    UPDATE_GOAL: Modify goal state
    WAIT: Do nothing (valid action!)
    TOOL_CALL: Execute external tool
    """
    SPEAK = "speak"
    COMMIT_MEMORY = "commit_memory"
    RETRIEVE_MEMORY = "retrieve_memory"
    INTROSPECT = "introspect"
    UPDATE_GOAL = "update_goal"
    WAIT = "wait"
    TOOL_CALL = "tool_call"


class Action(BaseModel):
    """
    Represents a single executable action.

    An action is a concrete behavior that the system can perform in response
    to its current workspace state. Actions can range from generating language
    output to querying memory to invoking external tools.

    Attributes:
        type: Category of action
        priority: Urgency/importance of this action (0.0-1.0)
        parameters: Action-specific parameters and arguments
        reason: Why this action was selected
        metadata: Additional contextual information
    """
    type: ActionType
    priority: float = Field(ge=0.0, le=1.0, default=0.5)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ActionSubsystem:
    """
    Decides what actions to take based on current workspace state.

    The ActionSubsystem translates the declarative content of the GlobalWorkspace
    (goals, percepts, emotions) into procedural action decisions. It implements
    goal-directed behavior by evaluating which actions best serve current goals
    given the current perceptual and emotional context.

    Key Responsibilities:
    - Generate candidate actions based on workspace state
    - Evaluate action appropriateness given goals and context
    - Select between competing action tendencies
    - Execute chosen actions and monitor outcomes
    - Maintain action history for learning and adaptation
    - Handle action failures and implement fallback strategies
    - Enforce constitutional protocol constraints

    Integration Points:
    - GlobalWorkspace: Reads current goals, percepts, and emotions to guide action
    - AffectSubsystem: Emotional state influences action selection and urgency
    - PerceptionSubsystem: Action outcomes may generate new percepts
    - CognitiveCore: Actions are executed in the main cognitive loop
    - IdentityLoader: Loads protocol constraints from identity files

    Action Selection Process:
    1. Generate candidate actions from current workspace state
    2. Filter by protocol constraints
    3. Score each candidate based on:
       - Goal alignment: Does it advance current goals?
       - Emotional urgency: High arousal boosts priority
       - Resource cost: Some actions are expensive
       - Recency penalty: Avoid repeating same action
    4. Select highest-priority action(s) for execution
    5. Track in action history

    Attributes:
        protocol_constraints: Constitutional behavioral rules
        action_history: Recent actions taken (for pattern detection)
        action_stats: Performance metrics
        tool_registry: Available actions/tools
        config: Configuration dictionary
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        """
        Initialize the action subsystem.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self.protocol_constraints: List[Any] = []
        self.action_history: deque = deque(maxlen=50)
        self.action_stats: Dict[str, Any] = {
            "total_actions": 0,
            "blocked_actions": 0,
            "action_counts": {}
        }
        self.tool_registry: Dict[str, Dict[str, Any]] = {}
        
        # Load identity constraints
        self._load_protocol_constraints()
        
        logger.info("✅ ActionSubsystem initialized")
    
    def _load_protocol_constraints(self) -> None:
        """Load protocol constraints from identity files."""
        try:
            from ..identity.loader import IdentityLoader
            
            constraints = IdentityLoader.load_protocols()
            self.protocol_constraints.extend(constraints)
            logger.info(f"✅ Loaded {len(constraints)} protocol constraints")
        except Exception as e:
            logger.error(f"Error loading protocol constraints: {e}")
    
    def decide(self, snapshot: WorkspaceSnapshot) -> List[Action]:
        """
        Main decision-making method.
        
        Generates candidate actions based on workspace state,
        filters by protocol constraints, prioritizes by urgency
        and goal alignment, and returns ordered list of actions.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            List of actions to execute (ordered by priority)
        """
        # Generate candidate actions
        candidates = self._generate_candidates(snapshot)
        
        # Filter by protocol constraints
        valid_actions = []
        for action in candidates:
            if self._violates_protocols(action):
                logger.warning(f"❌ Blocked action: {action.type} (violates protocols)")
                self.action_stats["blocked_actions"] += 1
            else:
                valid_actions.append(action)
        
        # Score and prioritize
        scored_actions = [
            (self._score_action(action, snapshot), action)
            for action in valid_actions
        ]
        scored_actions.sort(reverse=True, key=lambda x: x[0])
        
        # Return top actions (limit to 3)
        selected = [action for score, action in scored_actions[:3]]
        
        # Track in history
        self.action_history.extend(selected)
        for action in selected:
            self.action_stats["total_actions"] += 1
            action_type_str = action.type.value if hasattr(action.type, 'value') else str(action.type)
            self.action_stats["action_counts"][action_type_str] = \
                self.action_stats["action_counts"].get(action_type_str, 0) + 1
        
        logger.info(f"✅ Selected {len(selected)} actions: "
                   f"{[a.type.value for a in selected]}")
        
        return selected
    
    def _generate_candidates(self, snapshot: WorkspaceSnapshot) -> List[Action]:
        """
        Generate possible actions based on workspace state.
        
        Creates candidate actions based on:
        - Active goals (respond, retrieve memory, commit memory, etc.)
        - Current percepts (user requests, introspections)
        - Emotional state (high arousal = urgent action)
        
        Args:
            snapshot: Current workspace state
            
        Returns:
            Unfiltered list of candidate actions
        """
        candidates = []
        
        # 1. Goal-driven actions
        for goal in snapshot.goals:
            if goal.type == GoalType.RESPOND_TO_USER:
                candidates.append(Action(
                    type=ActionType.SPEAK,
                    priority=0.9,  # User requests are high priority
                    parameters={"goal_id": goal.id},
                    reason="Responding to user request"
                ))
            
            elif goal.type == GoalType.COMMIT_MEMORY:
                candidates.append(Action(
                    type=ActionType.COMMIT_MEMORY,
                    priority=0.6,
                    parameters={"goal_id": goal.id},
                    reason="Committing experience to long-term memory"
                ))
            
            elif goal.type == GoalType.RETRIEVE_MEMORY:
                candidates.append(Action(
                    type=ActionType.RETRIEVE_MEMORY,
                    priority=0.7,
                    parameters={"goal_id": goal.id, "query": goal.description},
                    reason="Searching for relevant memories"
                ))
            
            elif goal.type == GoalType.INTROSPECT:
                candidates.append(Action(
                    type=ActionType.INTROSPECT,
                    priority=0.5,
                    parameters={"goal_id": goal.id},
                    reason="Self-reflection requested"
                ))
        
        # 2. Emotion-driven actions
        valence = snapshot.emotions.get("valence", 0.0)
        arousal = snapshot.emotions.get("arousal", 0.0)
        
        if arousal > 0.7:
            # High arousal = urgent action needed
            for action in candidates:
                if action.type == ActionType.SPEAK:
                    action.priority = min(action.priority * 1.3, 1.0)
        
        if valence < -0.5:
            # Negative emotion = may need introspection
            candidates.append(Action(
                type=ActionType.INTROSPECT,
                priority=0.4,
                reason="Negative emotional state detected"
            ))
        
        # 3. Percept-driven actions
        for percept_id, percept_data in snapshot.percepts.items():
            if isinstance(percept_data, dict):
                modality = percept_data.get("modality", "")
                if modality == "introspection":
                    # Meta-cognitive percepts may trigger introspection
                    candidates.append(Action(
                        type=ActionType.INTROSPECT,
                        priority=0.6,
                        parameters={"percept_id": percept_id},
                        reason="Responding to introspective percept"
                    ))
        
        # 4. Default: wait if nothing urgent
        if not candidates:
            candidates.append(Action(
                type=ActionType.WAIT,
                priority=0.1,
                reason="No urgent actions needed"
            ))
        
        return candidates
    
    def _violates_protocols(self, action: Action) -> bool:
        """
        Check if action violates constitutional protocols.
        
        Args:
            action: Action to check
            
        Returns:
            True if action should be blocked, False otherwise
        """
        for constraint in self.protocol_constraints:
            if constraint.test_fn is None:
                continue
            
            try:
                if constraint.test_fn(action):
                    logger.debug(f"Action {action.type} violates: {constraint.rule}")
                    return True
            except Exception as e:
                logger.error(f"Error testing constraint: {e}")
        
        return False
    
    def _score_action(self, action: Action, snapshot: WorkspaceSnapshot) -> float:
        """
        Score action priority.
        
        Scoring factors:
        - Goal alignment: Does it advance current goals?
        - Emotional urgency: High arousal boosts priority
        - Recency penalty: Avoid repeating same action
        - Resource cost: Some actions are expensive
        
        Args:
            action: Action to score
            snapshot: Current workspace state
            
        Returns:
            Priority score (0.0-1.0)
        """
        base_score = action.priority
        
        # 1. Goal alignment
        goal_boost = 0.0
        if "goal_id" in action.parameters:
            goal_id = action.parameters["goal_id"]
            matching_goals = [g for g in snapshot.goals if g.id == goal_id]
            if matching_goals:
                goal_boost = matching_goals[0].priority * 0.3
        
        # 2. Emotional urgency
        arousal = snapshot.emotions.get("arousal", 0.0)
        if action.type == ActionType.SPEAK and arousal > 0.7:
            base_score *= 1.2
        
        # 3. Recency penalty (avoid repetition)
        recent_same_type = sum(
            1 for a in list(self.action_history)[-5:]
            if a.type == action.type
        )
        recency_penalty = recent_same_type * 0.1
        
        # 4. Resource cost (some actions are expensive)
        cost_penalty = 0.0
        if action.type == ActionType.RETRIEVE_MEMORY:
            cost_penalty = 0.1  # Memory search is costly
        
        final_score = base_score + goal_boost - recency_penalty - cost_penalty
        return max(0.0, min(1.0, final_score))
    
    def register_tool(self, name: str, handler: Callable, description: str) -> None:
        """
        Register an action handler.
        
        Args:
            name: Tool name
            handler: Callable that executes the tool
            description: Human-readable description
        """
        self.tool_registry[name] = {
            "handler": handler,
            "description": description
        }
        logger.info(f"Registered tool: {name}")
    
    def add_constraint(self, constraint: Any) -> None:
        """
        Add protocol constraint at runtime.
        
        Args:
            constraint: ActionConstraint to add
        """
        self.protocol_constraints.append(constraint)
        logger.info(f"Added constraint: {constraint.rule}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return statistics on action selection.
        
        Returns:
            Dict containing:
            - Total actions taken
            - Actions blocked by protocols
            - Most common action types
        """
        return {
            "total_actions": self.action_stats["total_actions"],
            "blocked_actions": self.action_stats["blocked_actions"],
            "action_counts": self.action_stats["action_counts"].copy(),
            "history_size": len(self.action_history)
        }
    
    async def execute_action(self, action: Action) -> Any:
        """
        Execute an action (called by CognitiveCore).
        
        Args:
            action: Action to execute
            
        Returns:
            Result of action execution, or None on error
        """
        try:
            if action.type == ActionType.TOOL_CALL:
                tool_name = action.parameters.get("tool")
                if tool_name in self.tool_registry:
                    handler = self.tool_registry[tool_name]["handler"]
                    result = await handler(action.parameters)
                    return result
                else:
                    logger.error(f"Unknown tool: {tool_name}")
                    return None
            
            # Other action types handled by core
            return action
            
        except Exception as e:
            logger.error(f"Error executing action: {e}", exc_info=True)
            return None
