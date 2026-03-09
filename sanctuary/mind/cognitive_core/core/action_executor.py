"""
Action execution helpers for the cognitive cycle.

Handles execution of different action types including SPEAK, TOOL_CALL, etc.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Dict, Any
from datetime import datetime

from ..workspace import Percept
from ..action import ActionType

if TYPE_CHECKING:
    from .subsystem_coordinator import SubsystemCoordinator
    from .state_manager import StateManager

logger = logging.getLogger(__name__)


class ActionExecutor:
    """
    Handles execution of different action types.
    
    Responsibilities:
    - Execute SPEAK actions (generate and queue responses)
    - Execute SPEAK_AUTONOMOUS actions
    - Execute TOOL_CALL actions and return percepts
    - Handle other action types
    """
    
    def __init__(self, subsystems: 'SubsystemCoordinator', state: 'StateManager'):
        """
        Initialize action executor.
        
        Args:
            subsystems: SubsystemCoordinator instance
            state: StateManager instance
        """
        self.subsystems = subsystems
        self.state = state
    
    def _has_temporal_grounding(self) -> bool:
        """Check if temporal grounding subsystem is available."""
        return hasattr(self.subsystems, 'temporal_grounding') and self.subsystems.temporal_grounding is not None
    
    async def execute(self, action) -> None:
        """
        Execute a single action.
        
        Args:
            action: Action to execute
        """
        try:
            if action.type == ActionType.SPEAK:
                await self.execute_speak(action)
            elif action.type == ActionType.SPEAK_AUTONOMOUS:
                await self.execute_speak_autonomous(action)
            elif action.type == ActionType.COMMIT_MEMORY:
                logger.debug(f"Action COMMIT_MEMORY: {action.reason}")
            elif action.type == ActionType.RETRIEVE_MEMORY:
                query = action.parameters.get("query", "")
                logger.debug(f"Action RETRIEVE_MEMORY: query='{query}'")
            elif action.type == ActionType.INTROSPECT:
                logger.debug(f"Action INTROSPECT: {action.reason}")
            elif action.type == ActionType.UPDATE_GOAL:
                logger.debug(f"Action UPDATE_GOAL: {action.reason}")
            elif action.type == ActionType.WAIT:
                logger.debug("Action WAIT: maintaining current state")
            elif action.type == ActionType.TOOL_CALL:
                result = await self.subsystems.action.execute_action(action)
                logger.debug(f"Action TOOL_CALL: result={result}")
        except Exception as e:
            logger.error(f"Error executing action {action.type}: {e}", exc_info=True)
    
    async def execute_speak(self, action) -> None:
        """
        Execute SPEAK action with validation.
        
        Args:
            action: Action with metadata containing response context
        """
        try:
            snapshot = self.state.workspace.broadcast()
            context = {
                "user_input": action.metadata.get("responding_to", "") if hasattr(action, 'metadata') else ""
            }
            
            # Generate response using language output generator
            response = await self.subsystems.language_output.generate(snapshot, context)
            
            # Validate response before queueing
            if not response or not isinstance(response, str):
                logger.warning(f"Invalid response generated: {type(response)}")
                response = "..." # Fallback response
            
            # Queue response for external retrieval
            await self.state.queue_output({
                "type": "SPEAK",
                "text": response,
                "emotion": snapshot.emotions,
                "timestamp": datetime.now()
            })
            logger.info(f"🗣️ Sanctuary: {response[:100]}...")
            
            # Record output time if temporal grounding available
            if self._has_temporal_grounding():
                self.subsystems.temporal_grounding.record_output()
            
            # Record output in communication drive system
            if hasattr(self.subsystems, 'communication_drives'):
                self.subsystems.communication_drives.record_output()

            # Post-hoc reflection: "Was that the right thing to say?"
            self._reflect_on_output(response, "SPEAK", snapshot)
        except Exception as e:
            logger.error(f"Failed to execute SPEAK action: {e}", exc_info=True)

    async def execute_speak_autonomous(self, action) -> None:
        """
        Execute SPEAK_AUTONOMOUS action with validation.
        
        Args:
            action: Action with metadata containing trigger and content
        """
        try:
            snapshot = self.state.workspace.broadcast()
            context = {
                "autonomous": True,
                "trigger": action.metadata.get("trigger") if hasattr(action, 'metadata') else None,
                "introspection_content": action.metadata.get("introspection_content") if hasattr(action, 'metadata') else None
            }
            
            # Generate response using language output generator
            response = await self.subsystems.language_output.generate(snapshot, context)
            
            # Validate response before queueing
            if not response or not isinstance(response, str):
                logger.warning(f"Invalid autonomous response generated: {type(response)}")
                response = "..." # Fallback response
            
            # Queue autonomous response for external retrieval
            await self.state.queue_output({
                "type": "SPEAK_AUTONOMOUS",
                "text": response,
                "trigger": context.get("trigger"),
                "emotion": snapshot.emotions,
                "timestamp": datetime.now()
            })
            logger.info(f"🗣️💭 Sanctuary (autonomous): {response[:100]}...")
            
            # Record output time if temporal grounding available
            if self._has_temporal_grounding():
                self.subsystems.temporal_grounding.record_output()
            
            # Record output in communication drive system
            if hasattr(self.subsystems, 'communication_drives'):
                self.subsystems.communication_drives.record_output()

            # Post-hoc reflection: "Was that the right thing to say?"
            self._reflect_on_output(response, "SPEAK_AUTONOMOUS", snapshot)
        except Exception as e:
            logger.error(f"Failed to execute SPEAK_AUTONOMOUS action: {e}", exc_info=True)
    
    async def execute_tool(self, action) -> Optional[Percept]:
        """
        Execute a tool call action and return the result percept.
        
        Args:
            action: TOOL_CALL action with tool_name and parameters
            
        Returns:
            Percept from tool execution, or None on error
        """
        if action.type != ActionType.TOOL_CALL:
            logger.error(f"execute_tool called with non-TOOL_CALL action: {action.type}")
            return None
        
        try:
            # Extract tool information from action
            tool_name = action.parameters.get("tool_name")
            tool_params = action.parameters.get("parameters", {})
            
            if not tool_name:
                logger.error("TOOL_CALL action missing tool_name parameter")
                return None
            
            # Check if action subsystem has the new tool registry with percept generation
            if hasattr(self.subsystems.action, 'tool_reg'):
                # Use new tool registry with percept generation
                result = await self.subsystems.action.tool_reg.execute_tool_with_percept(
                    tool_name,
                    parameters=tool_params,
                    create_percept=True
                )
                
                # Log execution result
                if result.success:
                    logger.info(
                        f"✅ Tool '{tool_name}' executed: success "
                        f"({result.execution_time_ms:.1f}ms)"
                    )
                else:
                    logger.warning(
                        f"❌ Tool '{tool_name}' failed: {result.error} "
                        f"({result.execution_time_ms:.1f}ms)"
                    )
                
                # Return the generated percept
                return result.percept
            else:
                # Fallback to legacy tool execution (no percept generation)
                logger.warning("Action subsystem missing tool_reg, using legacy execution")
                result = await self.subsystems.action.execute_action(action)
                logger.debug(f"Action TOOL_CALL (legacy): result={result}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing tool action: {e}", exc_info=True)
            return None
    
    def _reflect_on_output(self, text: str, output_type: str, snapshot) -> None:
        """
        Run post-hoc communication reflection after producing output.

        Evaluates the communication and logs the reflection. Failures here
        must not disrupt the main action pipeline.
        """
        if not hasattr(self.subsystems, 'communication_reflection'):
            return

        try:
            emotional_state = self.subsystems.affect.get_state()

            # Gather decision context if available
            decision_context: Dict[str, Any] = {}
            if hasattr(self.subsystems, 'communication_decision'):
                history = self.subsystems.communication_decision.decision_history
                if history:
                    last = history[-1]
                    decision_context = {
                        "confidence": last.confidence,
                        "net_pressure": last.net_pressure,
                        "drive_level": last.drive_level,
                        "was_autonomous": output_type == "SPEAK_AUTONOMOUS"
                    }

            reflection = self.subsystems.communication_reflection.reflect(
                output_text=text,
                output_type=output_type,
                workspace_state=snapshot,
                emotional_state=emotional_state,
                decision_context=decision_context
            )

            if reflection.overall_score < 0.4:
                logger.info(
                    f"🪞 Reflection: {reflection.verdict.value} "
                    f"(score={reflection.overall_score:.2f}) — "
                    f"{'; '.join(reflection.lessons)}"
                )
        except Exception as e:
            logger.debug(f"Communication reflection failed (non-critical): {e}")

    @staticmethod
    def extract_outcome(action) -> Dict[str, Any]:
        """
        Extract outcome information from an executed action.
        
        Args:
            action: The action that was executed
            
        Returns:
            Dictionary containing outcome details for self-model update
        """
        outcome = {
            "action_type": str(action.type) if hasattr(action, 'type') else "unknown",
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "reason": ""
        }
        
        # Add action-specific details
        if hasattr(action, 'metadata'):
            outcome["metadata"] = action.metadata
        
        if hasattr(action, 'reason'):
            outcome["reason"] = action.reason
        
        # Check for failure indicators
        if hasattr(action, 'status'):
            outcome["success"] = action.status == "success"
        
        return outcome
