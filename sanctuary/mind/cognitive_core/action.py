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

from .workspace import WorkspaceSnapshot, GoalType, Percept
from .protocol_loader import ProtocolLoader, ProtocolViolation

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
    SPEAK_AUTONOMOUS: Unprompted speech initiated by Sanctuary
    """
    SPEAK = "speak"
    COMMIT_MEMORY = "commit_memory"
    RETRIEVE_MEMORY = "retrieve_memory"
    INTROSPECT = "introspect"
    UPDATE_GOAL = "update_goal"
    WAIT = "wait"
    TOOL_CALL = "tool_call"
    SPEAK_AUTONOMOUS = "speak_autonomous"


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

    def __init__(
        self,
        config: Optional[Dict] = None,
        affect: Optional[Any] = None,
        identity: Optional[Any] = None,
        behavior_logger: Optional[Any] = None
    ) -> None:
        """
        Initialize the action subsystem.

        Args:
            config: Optional configuration dict
            affect: Optional reference to the affect subsystem for emotional modulation
            identity: Optional IdentityLoader instance with charter and protocols
            behavior_logger: Optional BehaviorLogger for tracking actions for identity computation
        """
        self.config = config or {}
        self.affect = affect
        self.identity = identity
        self.behavior_logger = behavior_logger
        self.protocol_constraints: List[Any] = []
        self.action_history: deque = deque(maxlen=50)
        self.action_stats: Dict[str, Any] = {
            "total_actions": 0,
            "blocked_actions": 0,
            "action_counts": {}
        }
        
        # Initialize tool registry (legacy dict maintained for compatibility)
        self.tool_registry: Dict[str, Dict[str, Any]] = {}
        
        # Initialize new tool registry and cache
        from .tool_registry import ToolRegistry, create_default_registry
        from .tool_cache import ToolCache
        
        self.tool_reg = create_default_registry()
        self.tool_cache = ToolCache(
            max_size=self.config.get("tool_cache_size", 1000),
            default_ttl=self.config.get("tool_cache_ttl", 3600.0)
        )
        
        # Initialize protocol loader
        self.protocol_loader = ProtocolLoader()
        self.protocol_loader.load_protocols()
        
        # Load identity constraints if identity provided
        if self.identity:
            logger.info("✅ ActionSubsystem initialized with identity")
        else:
            # Fallback: Load protocol constraints from identity files
            self._load_protocol_constraints()
            logger.info("✅ ActionSubsystem initialized (legacy mode)")

    
    def _load_protocol_constraints(self) -> None:
        """Load protocol constraints from identity files (legacy fallback)."""
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
        applies emotional modulation (valence bias),
        filters by protocol constraints, prioritizes by urgency
        and goal alignment, and returns ordered list of actions.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            List of actions to execute (ordered by priority)
        """
        # Generate candidate actions
        candidates = self._generate_candidates(snapshot)
        
        # Apply emotional modulation (valence-based action biasing) BEFORE filtering
        # This makes emotions functionally modulate action selection
        if self.affect and hasattr(self.affect, 'apply_valence_bias_to_actions'):
            candidates = self.affect.apply_valence_bias_to_actions(candidates)
            logger.debug(
                f"Applied emotional valence bias to {len(candidates)} candidate actions"
            )
        
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
        
        # Apply dominance-based decision threshold (from emotional modulation)
        decision_threshold = 0.5  # Default threshold
        if self.affect and hasattr(self.affect, 'get_processing_params'):
            processing_params = self.affect.get_processing_params()
            decision_threshold = processing_params.decision_threshold
            logger.debug(
                f"Emotional modulation: decision_threshold={decision_threshold:.2f} "
                f"(dominance={processing_params.dominance_level:.2f})"
            )
        
        # Filter by decision threshold (only execute actions above threshold)
        selected = [
            action for score, action in scored_actions
            if score >= decision_threshold
        ][:3]  # Still limit to top 3
        
        # Track in history
        self.action_history.extend(selected)
        for action in selected:
            self.action_stats["total_actions"] += 1
            action_type_str = action.type.value if hasattr(action.type, 'value') else str(action.type)
            self.action_stats["action_counts"][action_type_str] = \
                self.action_stats["action_counts"].get(action_type_str, 0) + 1
            
            # Log to behavior logger for identity computation
            if self.behavior_logger:
                self.behavior_logger.log_action(action)
        
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
                    reason="Responding to user request",
                    metadata={
                        "responding_to": goal.metadata.get("user_input", "")
                    }
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
        
        # Memory retrieval - trigger when workspace is sparse but not empty
        if 0 < len(snapshot.percepts) < 5:
            candidates.append(Action(
                type=ActionType.RETRIEVE_MEMORY,
                priority=0.4,
                reason="Workspace sparse, retrieving context from memory"
            ))
        
        # Memory consolidation - trigger on high arousal
        if arousal > 0.7:
            # High arousal = urgent action needed
            for action in candidates:
                if action.type == ActionType.SPEAK:
                    action.priority = min(action.priority * 1.3, 1.0)
            
            # Also trigger memory consolidation
            candidates.append(Action(
                type=ActionType.COMMIT_MEMORY,
                priority=0.6,
                reason=f"High arousal ({arousal:.2f}), consolidating experience"
            ))
        
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
        
        This method checks both legacy protocol_constraints and new protocol loader
        constraints.
        
        Args:
            action: Action to check
            
        Returns:
            True if action should be blocked, False otherwise
        """
        # Check legacy protocol constraints (if any)
        for constraint in self.protocol_constraints:
            if constraint.test_fn is None:
                continue
            
            try:
                if constraint.test_fn(action):
                    logger.debug(f"Action {action.type} violates: {constraint.rule}")
                    return True
            except Exception as e:
                logger.error(f"Error testing constraint: {e}")
        
        # Check new protocol loader constraints
        if self.protocol_loader and self.protocol_loader.protocols:
            action_type_str = action.type.value if hasattr(action.type, 'value') else str(action.type)
            is_compliant, violations = self.protocol_loader.check_action_compliance(
                action_type=action_type_str,
                action_parameters=action.parameters,
                context=action.metadata
            )
            
            if not is_compliant:
                # Log violations and add to action metadata
                action.metadata["protocol_violations"] = [
                    {
                        "protocol": v.protocol_title,
                        "reason": v.reason,
                        "severity": v.severity
                    }
                    for v in violations
                ]
                logger.warning(f"⚠️ Action {action.type} violates {len(violations)} protocol(s)")
                return True
        
        # Check new identity-based constitutional constraints
        if self.identity:
            if not self._check_constitutional_constraints(action):
                return True
        
        return False
    
    def _check_constitutional_constraints(self, action: Action) -> bool:
        """
        Check if action violates charter or protocols.
        
        Returns:
            True if action is permitted, False if prohibited
        """
        if not self.identity or not self.identity.charter:
            return True  # No constraints if no charter loaded
        
        # Check against behavioral guidelines
        for guideline in self.identity.charter.behavioral_guidelines:
            if self._action_violates_guideline(action, guideline):
                logger.warning(f"⚠️ Action {action.type} violates guideline: {guideline}")
                return False
        
        # Check relevant protocols
        protocols = self.identity.get_relevant_protocols({"action": action})
        for protocol in protocols:
            if self._action_violates_protocol(action, protocol):
                logger.warning(f"⚠️ Action {action.type} violates protocol: {protocol.name}")
                return False
        
        return True
    
    def _action_violates_guideline(self, action: Action, guideline: str) -> bool:
        """
        Check if action violates a specific guideline.
        
        Args:
            action: Action to check
            guideline: Guideline text from charter
            
        Returns:
            True if action violates guideline, False otherwise
        """
        # Implement guideline checking logic
        # For now, basic keyword matching
        guideline_lower = guideline.lower()
        
        # Check honesty-related guidelines
        if "never lie" in guideline_lower or "always honest" in guideline_lower or "never fabricate" in guideline_lower:
            # Check if SPEAK action contains deception markers
            if action.type == ActionType.SPEAK:
                # In a real implementation, we'd analyze the content
                # For now, we allow all speech actions (assume honesty)
                return False
        
        # Check harm-related guidelines
        if "do no harm" in guideline_lower or "refuse" in guideline_lower and "harm" in guideline_lower:
            # Check if action could cause harm
            # For now, we don't have enough context to determine harm
            return False
        
        # Default: not violating
        return False
    
    def _action_violates_protocol(self, action: Action, protocol) -> bool:
        """
        Check if action violates a protocol.
        
        Args:
            action: Action to check
            protocol: ProtocolDocument to check against
            
        Returns:
            True if action violates protocol, False otherwise
        """
        # Implement protocol checking logic
        # This is a placeholder - real implementation would analyze
        # trigger conditions and check if the action's context matches
        
        # For now, we don't block any actions based on protocols
        # The protocols are more about guiding what actions to take
        # rather than blocking actions
        return False
    
    def _score_action(self, action: Action, snapshot: WorkspaceSnapshot) -> float:
        """
        Score action priority.
        
        Scoring factors:
        - Goal alignment: Does it advance current goals?
        - Emotional urgency: High arousal boosts priority
        - Recency penalty: Avoid repeating same action
        - Resource cost: Some actions are expensive
        - Affect modulation: Emotional state influences action priorities
        
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
        
        # 2. Emotional urgency (legacy - now handled by affect subsystem)
        # Kept for backward compatibility if affect subsystem is not available
        emotional_boost = 0.0
        if not self.affect:
            arousal = snapshot.emotions.get("arousal", 0.0)
            if action.type == ActionType.SPEAK and arousal > 0.7:
                emotional_boost = base_score * 0.2
        
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
        
        # Calculate base final score
        intermediate_score = base_score + goal_boost + emotional_boost - recency_penalty - cost_penalty
        intermediate_score = max(0.0, min(1.0, intermediate_score))
        
        # 5. Apply affect modulation if available
        if self.affect:
            final_score = self.affect.influence_action(intermediate_score, action)
        else:
            final_score = intermediate_score
        
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
        stats = {
            "total_actions": self.action_stats["total_actions"],
            "blocked_actions": self.action_stats["blocked_actions"],
            "action_counts": self.action_stats["action_counts"].copy(),
            "history_size": len(self.action_history)
        }
        
        # Add protocol violation statistics if available
        if self.protocol_loader:
            stats["protocol_violations"] = self.protocol_loader.get_violation_summary()
        
        return stats
    
    def reload_protocols(self) -> int:
        """
        Hot-reload protocols from disk without system restart.
        
        Returns:
            Number of protocols reloaded
        """
        if not self.protocol_loader:
            logger.warning("⚠️ Protocol loader not initialized")
            return 0
        
        count = self.protocol_loader.hot_reload()
        logger.info(f"✅ Reloaded {count} protocols")
        return count
    
    def generate_violation_percept(
        self, 
        violations: List[ProtocolViolation], 
        action: Action
    ) -> Optional[Percept]:
        """
        Generate introspective percept about protocol violations.
        
        Creates a meta-cognitive percept that allows the system to reason
        about why an action was blocked and what constraints were violated.
        
        Args:
            violations: List of protocol violations
            action: The action that was blocked
            
        Returns:
            Percept for introspection, or None if no violations
        """
        if not violations:
            return None
        
        # Summarize violations
        violation_summary = []
        for v in violations:
            violation_summary.append(
                f"Protocol '{v.protocol_title}' violated: {v.reason}"
            )
        
        percept_text = (
            f"I attempted to {action.type.value} but this action violated "
            f"{len(violations)} protocol(s): {'; '.join(violation_summary)}. "
            f"I must respect these constitutional constraints."
        )
        
        percept = Percept(
            modality="introspection",
            raw=percept_text,
            complexity=2,
            metadata={
                "type": "protocol_violation",
                "blocked_action": action.type.value,
                "violations": [
                    {
                        "protocol_id": v.protocol_id,
                        "protocol_title": v.protocol_title,
                        "severity": v.severity,
                        "reason": v.reason
                    }
                    for v in violations
                ]
            }
        )
        
        return percept
    
    async def execute_tool_action(
        self, 
        action: Action,
        use_cache: bool = True
    ) -> Percept:
        """
        Execute a TOOL_CALL action via the tool registry.
        
        This method handles the execution of external tools, including:
        - Checking cache for recent results
        - Executing tool with timeout
        - Caching successful results
        - Creating percepts from tool outputs
        - Handling errors gracefully
        
        Args:
            action: Action with type TOOL_CALL
            use_cache: Whether to use cached results
            
        Returns:
            Percept containing tool result or error
        """
        if action.type != ActionType.TOOL_CALL:
            logger.error(f"❌ execute_tool_action called with wrong action type: {action.type}")
            return self._create_error_percept("Invalid action type for tool execution")
        
        tool_name = action.parameters.get("tool_name")
        tool_params = action.parameters.get("parameters", {})
        
        if not tool_name:
            logger.error("❌ TOOL_CALL action missing tool_name")
            return self._create_error_percept("Tool name not specified")
        
        # Check if tool is registered
        if not self.tool_reg.is_tool_registered(tool_name):
            logger.error(f"❌ Tool '{tool_name}' not registered")
            return self._create_error_percept(f"Tool '{tool_name}' not available")
        
        # Check cache first
        if use_cache:
            cached_result = self.tool_cache.get(tool_name, tool_params)
            if cached_result is not None:
                logger.info(f"🎯 Using cached result for {tool_name}")
                return self._create_tool_result_percept(
                    tool_name, 
                    cached_result,
                    from_cache=True
                )
        
        # Execute tool
        try:
            tool_result = await self.tool_reg.execute_tool(tool_name, **tool_params)
            
            # Check if execution was successful
            if tool_result.status.value == "success":
                # Cache the result
                if use_cache:
                    self.tool_cache.set(tool_name, tool_params, tool_result.result)
                
                # Create percept from result
                percept = self._create_tool_result_percept(
                    tool_name,
                    tool_result.result,
                    execution_time=tool_result.execution_time
                )
            else:
                # Tool execution failed
                error_msg = tool_result.error or "Unknown error"
                percept = self._create_error_percept(
                    f"Tool '{tool_name}' {tool_result.status.value}: {error_msg}",
                    tool_name=tool_name
                )
            
            return percept
            
        except Exception as e:
            logger.error(f"❌ Exception executing tool '{tool_name}': {e}")
            return self._create_error_percept(
                f"Tool execution exception: {str(e)}",
                tool_name=tool_name
            )
    
    def _create_tool_result_percept(
        self,
        tool_name: str,
        result: Any,
        from_cache: bool = False,
        execution_time: float = 0.0
    ) -> Percept:
        """Create a percept from tool result."""
        return Percept(
            modality="tool_result",
            raw=result,
            complexity=2,
            metadata={
                "tool_name": tool_name,
                "from_cache": from_cache,
                "execution_time": execution_time
            }
        )
    
    def _create_error_percept(
        self,
        error_message: str,
        tool_name: Optional[str] = None
    ) -> Percept:
        """Create an error percept."""
        return Percept(
            modality="error",
            raw=error_message,
            complexity=1,
            metadata={
                "type": "tool_error",
                "tool_name": tool_name
            }
        )
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools with their metadata.
        
        Returns:
            List of tool information dictionaries
        """
        return self.tool_reg.get_available_tools()
    
    def invalidate_tool_cache(self, tool_name: Optional[str] = None) -> int:
        """
        Invalidate cached tool results.
        
        Args:
            tool_name: Specific tool to invalidate, or None for all
            
        Returns:
            Number of cache entries invalidated
        """
        return self.tool_cache.invalidate(tool_name)
    
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
