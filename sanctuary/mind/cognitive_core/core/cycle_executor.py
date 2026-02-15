"""
Cognitive cycle executor.

Executes the 9-step cognitive cycle including perception, attention,
memory, affect, action, meta-cognition, and workspace updates.
"""

from __future__ import annotations

import time
import logging
from typing import TYPE_CHECKING, Dict, Optional, Any

from ..workspace import GoalType
from ..action import ActionType

if TYPE_CHECKING:
    from .subsystem_coordinator import SubsystemCoordinator
    from .state_manager import StateManager
    from .action_executor import ActionExecutor
    from .timing import TimingManager
    from .subsystem_health import SubsystemSupervisor

logger = logging.getLogger(__name__)


class CycleExecutor:
    """
    Executes the complete 9-step cognitive cycle.

    The cognitive cycle follows these steps:
    1. PERCEPTION: Gather new inputs (if any queued)
    2. MEMORY RETRIEVAL: Check for memory retrieval goals and fetch relevant memories
    3. ATTENTION: Select percepts (including memory-percepts) for workspace
    4. AFFECT UPDATE: Compute emotional dynamics
    5. ACTION SELECTION: Decide what to do
    6. META-COGNITION: Generate introspective percepts
    7. AUTONOMOUS INITIATION: Check for autonomous speech triggers
    8. WORKSPACE UPDATE: Integrate all subsystem outputs
    9. MEMORY CONSOLIDATION: Store significant states to long-term memory

    Each step is monitored by a SubsystemSupervisor that tracks health per
    subsystem and disables failing subsystems via circuit breaker pattern.
    """

    def __init__(self, subsystems: 'SubsystemCoordinator', state: 'StateManager', action_executor: 'ActionExecutor', timing: 'TimingManager' = None, supervisor: 'SubsystemSupervisor' = None):
        """
        Initialize cycle executor.

        Args:
            subsystems: SubsystemCoordinator instance
            state: StateManager instance
            action_executor: ActionExecutor instance for handling actions
            timing: Optional TimingManager instance for tracking cycle metrics
            supervisor: Optional SubsystemSupervisor for fault isolation
        """
        self.subsystems = subsystems
        self.state = state
        self.action_executor = action_executor
        self.timing = timing
        self.supervisor = supervisor

        # IWMT prediction tracking
        self._current_predictions = []
        self._current_prediction_errors = []
    
    def _has_temporal_grounding(self) -> bool:
        """Check if temporal grounding subsystem is available."""
        return hasattr(self.subsystems, 'temporal_grounding') and self.subsystems.temporal_grounding is not None
    
    def _should_run(self, name: str) -> bool:
        """Check if a subsystem step should be executed this cycle."""
        if self.supervisor is None:
            return True
        return self.supervisor.should_execute(name)

    def _record_ok(self, name: str) -> None:
        """Record a successful subsystem step."""
        if self.supervisor is not None:
            self.supervisor.record_success(name)

    def _record_err(self, name: str, error: Exception) -> None:
        """Record a failed subsystem step."""
        if self.supervisor is not None:
            self.supervisor.record_failure(name, error)

    async def execute_cycle(self) -> Dict[str, float]:
        """
        Execute one complete cognitive cycle with error handling.

        Each step is wrapped in error handling to prevent cascade failures.
        If a step fails, it's logged but the cycle continues to maintain
        system stability. The SubsystemSupervisor tracks health per step
        and disables persistently failing subsystems via circuit breaker.

        Returns:
            Dict of subsystem timings in milliseconds
        """
        subsystem_timings = {}

        # 0a. TEMPORAL CONTEXT: Fetch temporal awareness at cycle start
        if self._should_run('temporal_context'):
            try:
                step_start = time.time()
                if self._has_temporal_grounding():
                    temporal_context = self.subsystems.temporal_grounding.get_temporal_context()
                    self.state.workspace.set_temporal_context(temporal_context)
                subsystem_timings['temporal_context'] = (time.time() - step_start) * 1000
                self._record_ok('temporal_context')
            except Exception as e:
                logger.error(f"Temporal context step failed: {e}", exc_info=True)
                subsystem_timings['temporal_context'] = 0.0
                self._record_err('temporal_context', e)
        else:
            subsystem_timings['temporal_context'] = 0.0

        # 0. IWMT PREDICTION: Generate predictions before perception
        if self._should_run('iwmt_predict'):
            try:
                step_start = time.time()
                if hasattr(self.subsystems, 'iwmt_core') and self.subsystems.iwmt_core:
                    context = {
                        "goals": list(self.state.workspace.current_goals),
                        "emotional_state": self.subsystems.affect.get_state(),
                        "cycle_count": self.state.workspace.cycle_count
                    }
                    self._current_predictions = self.subsystems.iwmt_core.world_model.predict(
                        time_horizon=1.0,
                        context=context
                    )
                    self._current_prediction_errors = self.subsystems.iwmt_core.world_model.prediction_errors[-10:]
                    logger.debug(f"🔮 IWMT: Generated {len(self._current_predictions)} predictions")
                else:
                    self._current_predictions = []
                    self._current_prediction_errors = []
                subsystem_timings['iwmt_predict'] = (time.time() - step_start) * 1000
                self._record_ok('iwmt_predict')
            except Exception as e:
                logger.error(f"IWMT prediction step failed: {e}", exc_info=True)
                self._current_predictions = []
                self._current_prediction_errors = []
                subsystem_timings['iwmt_predict'] = 0.0
                self._record_err('iwmt_predict', e)
        else:
            self._current_predictions = []
            self._current_prediction_errors = []
            subsystem_timings['iwmt_predict'] = 0.0

        # 1. PERCEPTION: Process queued inputs
        if self._should_run('perception'):
            try:
                step_start = time.time()
                new_percepts = await self.state.gather_percepts(self.subsystems.perception)

                # Record input time if we got new percepts
                if new_percepts and self._has_temporal_grounding():
                    self.subsystems.temporal_grounding.record_input()

                subsystem_timings['perception'] = (time.time() - step_start) * 1000
                self._record_ok('perception')
            except Exception as e:
                logger.error(f"Perception step failed: {e}", exc_info=True)
                new_percepts = []
                subsystem_timings['perception'] = 0.0
                self._record_err('perception', e)
        else:
            new_percepts = []
            subsystem_timings['perception'] = 0.0

        # 1.5. IWMT UPDATE: Update world model with new percepts
        if self._should_run('iwmt_update'):
            try:
                if hasattr(self.subsystems, 'iwmt_core') and self.subsystems.iwmt_core and new_percepts:
                    step_start = time.time()
                    for percept in new_percepts:
                        error = self.subsystems.iwmt_core.world_model.update_on_percept(percept)
                        if error:
                            self._current_prediction_errors.append(error)
                    subsystem_timings['iwmt_update'] = (time.time() - step_start) * 1000
                    if self._current_prediction_errors:
                        logger.debug(f"🔮 IWMT: {len(self._current_prediction_errors)} prediction errors detected")
                self._record_ok('iwmt_update')
            except Exception as e:
                logger.error(f"IWMT update step failed: {e}", exc_info=True)
                subsystem_timings['iwmt_update'] = 0.0
                self._record_err('iwmt_update', e)
        else:
            subsystem_timings['iwmt_update'] = 0.0

        # 2. MEMORY RETRIEVAL: Check for memory retrieval goals
        if self._should_run('memory_retrieval'):
            try:
                step_start = time.time()
                new_percepts.extend(await self._retrieve_memories())
                subsystem_timings['memory_retrieval'] = (time.time() - step_start) * 1000
                self._record_ok('memory_retrieval')
            except Exception as e:
                logger.error(f"Memory retrieval step failed: {e}", exc_info=True)
                subsystem_timings['memory_retrieval'] = 0.0
                self._record_err('memory_retrieval', e)
        else:
            subsystem_timings['memory_retrieval'] = 0.0

        # 3. ATTENTION: Select for conscious awareness (with IWMT precision weighting)
        if self._should_run('attention'):
            try:
                step_start = time.time()
                # Get emotional state for precision weighting
                emotional_state = self.subsystems.affect.get_state()
                # Pass prediction context for IWMT precision-weighted attention
                attended = self.subsystems.attention.select_for_broadcast(
                    new_percepts,
                    emotional_state=emotional_state,
                    prediction_errors=self._current_prediction_errors
                )
                subsystem_timings['attention'] = (time.time() - step_start) * 1000
                self._record_ok('attention')
            except Exception as e:
                logger.error(f"Attention step failed: {e}", exc_info=True)
                attended = []
                subsystem_timings['attention'] = 0.0
                self._record_err('attention', e)
        else:
            attended = []
            subsystem_timings['attention'] = 0.0

        # 4. AFFECT: Update emotional state and get processing parameters
        if self._should_run('affect'):
            try:
                step_start = time.time()

                # Apply time passage effects if temporal grounding is available
                if hasattr(self.subsystems, 'temporal_grounding'):
                    # Get current cognitive state
                    cognitive_state = {
                        "emotions": {
                            "valence": self.subsystems.affect.valence,
                            "arousal": self.subsystems.affect.arousal,
                            "dominance": self.subsystems.affect.dominance
                        },
                        "goals": list(self.state.workspace.current_goals),
                        "working_memory": list(self.state.workspace.active_percepts.values())
                    }

                    # Apply temporal effects
                    updated_state = self.subsystems.temporal_grounding.apply_time_passage_effects(
                        cognitive_state
                    )

                    # Update affect subsystem with decayed emotions
                    if "emotions" in updated_state:
                        self.subsystems.affect.valence = updated_state["emotions"]["valence"]
                        self.subsystems.affect.arousal = updated_state["emotions"]["arousal"]
                        self.subsystems.affect.dominance = updated_state["emotions"]["dominance"]

                affect_update = self.subsystems.affect.compute_update(self.state.workspace.broadcast())

                # Log emotional modulation parameters for tracking
                if hasattr(self.subsystems.affect, 'get_processing_params'):
                    processing_params = self.subsystems.affect.get_processing_params()
                    logger.debug(
                        f"Emotional modulation active: "
                        f"V={processing_params.valence_level:.2f} "
                        f"A={processing_params.arousal_level:.2f} "
                        f"D={processing_params.dominance_level:.2f} → "
                        f"iters={processing_params.attention_iterations} "
                        f"thresh={processing_params.ignition_threshold:.2f} "
                        f"decision={processing_params.decision_threshold:.2f}"
                    )

                subsystem_timings['affect'] = (time.time() - step_start) * 1000
                self._record_ok('affect')
            except Exception as e:
                logger.error(f"Affect step failed: {e}", exc_info=True)
                affect_update = {}
                subsystem_timings['affect'] = 0.0
                self._record_err('affect', e)
        else:
            affect_update = {}
            subsystem_timings['affect'] = 0.0

        # 5. ACTION: Decide what to do and execute
        if self._should_run('action'):
            try:
                step_start = time.time()
                await self._execute_actions()

                # Record action time if we have temporal grounding
                if self._has_temporal_grounding():
                    self.subsystems.temporal_grounding.record_action()

                subsystem_timings['action'] = (time.time() - step_start) * 1000
                self._record_ok('action')
            except Exception as e:
                logger.error(f"Action step failed: {e}", exc_info=True)
                subsystem_timings['action'] = 0.0
                self._record_err('action', e)
        else:
            subsystem_timings['action'] = 0.0

        # 6. META-COGNITION: Introspect
        if self._should_run('meta_cognition'):
            try:
                step_start = time.time()
                meta_percepts = await self._run_meta_cognition()
                subsystem_timings['meta_cognition'] = (time.time() - step_start) * 1000
                self._record_ok('meta_cognition')
            except Exception as e:
                logger.error(f"Meta-cognition step failed: {e}", exc_info=True)
                meta_percepts = []
                subsystem_timings['meta_cognition'] = 0.0
                self._record_err('meta_cognition', e)
        else:
            meta_percepts = []
            subsystem_timings['meta_cognition'] = 0.0

        # 6.5 COMMUNICATION DRIVES: Compute internal urges to communicate
        if self._should_run('communication_drives'):
            try:
                step_start = time.time()
                await self._compute_communication_drives()
                subsystem_timings['communication_drives'] = (time.time() - step_start) * 1000
                self._record_ok('communication_drives')
            except Exception as e:
                logger.error(f"Communication drives step failed: {e}", exc_info=True)
                subsystem_timings['communication_drives'] = 0.0
                self._record_err('communication_drives', e)
        else:
            subsystem_timings['communication_drives'] = 0.0

        # 7. AUTONOMOUS INITIATION: Check for autonomous speech triggers
        if self._should_run('autonomous_initiation'):
            try:
                step_start = time.time()
                await self._check_autonomous_triggers()
                subsystem_timings['autonomous_initiation'] = (time.time() - step_start) * 1000
                self._record_ok('autonomous_initiation')
            except Exception as e:
                logger.error(f"Autonomous initiation step failed: {e}", exc_info=True)
                subsystem_timings['autonomous_initiation'] = 0.0
                self._record_err('autonomous_initiation', e)
        else:
            subsystem_timings['autonomous_initiation'] = 0.0

        # 8. WORKSPACE UPDATE: Integrate everything (CRITICAL — always attempted)
        try:
            step_start = time.time()
            self._update_workspace(attended, affect_update, meta_percepts)
            subsystem_timings['workspace_update'] = (time.time() - step_start) * 1000
            self._record_ok('workspace_update')
        except Exception as e:
            logger.error(f"Workspace update step failed: {e}", exc_info=True)
            subsystem_timings['workspace_update'] = 0.0
            self._record_err('workspace_update', e)

        # 9. MEMORY CONSOLIDATION: Commit workspace to long-term memory
        if self._should_run('memory_consolidation'):
            try:
                step_start = time.time()
                await self.subsystems.memory.consolidate(self.state.workspace.broadcast())
                subsystem_timings['memory_consolidation'] = (time.time() - step_start) * 1000
                self._record_ok('memory_consolidation')
            except Exception as e:
                logger.error(f"Memory consolidation step failed: {e}", exc_info=True)
                subsystem_timings['memory_consolidation'] = 0.0
                self._record_err('memory_consolidation', e)
        else:
            subsystem_timings['memory_consolidation'] = 0.0

        # 9.5. BOTTLENECK DETECTION: Monitor cognitive load
        if self._should_run('bottleneck_detection'):
            try:
                step_start = time.time()
                await self._update_bottleneck_detection(subsystem_timings)
                subsystem_timings['bottleneck_detection'] = (time.time() - step_start) * 1000
                self._record_ok('bottleneck_detection')
            except Exception as e:
                logger.error(f"Bottleneck detection step failed: {e}", exc_info=True)
                subsystem_timings['bottleneck_detection'] = 0.0
                self._record_err('bottleneck_detection', e)
        else:
            subsystem_timings['bottleneck_detection'] = 0.0

        # 10. IDENTITY UPDATE: Periodically recompute identity from system state
        # Update every 100 cycles to avoid overhead
        if self.state.workspace.cycle_count % 100 == 0:
            if self._should_run('identity_update'):
                try:
                    step_start = time.time()
                    if hasattr(self.subsystems, 'identity_manager'):
                        self.subsystems.identity_manager.update(
                            memory_system=self.subsystems.memory,
                            goal_system=self.state.workspace,
                            emotion_system=self.subsystems.affect
                        )
                        logger.debug("Identity recomputed from system state")
                    subsystem_timings['identity_update'] = (time.time() - step_start) * 1000
                    self._record_ok('identity_update')
                except Exception as e:
                    logger.error(f"Identity update failed: {e}", exc_info=True)
                    subsystem_timings['identity_update'] = 0.0
                    self._record_err('identity_update', e)
            else:
                subsystem_timings['identity_update'] = 0.0

        # Update timing metrics so total_cycles is tracked even when
        # execute_cycle() is called directly (not through CognitiveLoop)
        if self.timing is not None:
            cycle_time = sum(subsystem_timings.values()) / 1000.0  # Convert ms back to seconds
            self.timing.update_metrics(cycle_time, subsystem_timings)

        return subsystem_timings
    
    async def _retrieve_memories(self) -> list:
        """
        Retrieve memories if there are memory retrieval goals.
        
        Returns:
            List of memory percepts
        """
        snapshot = self.state.workspace.broadcast()
        retrieve_goals = [
            g for g in snapshot.goals
            if g.type == GoalType.RETRIEVE_MEMORY
        ]
        
        if retrieve_goals:
            # Retrieve memories and add as percepts (fast_mode for performance)
            return await self.subsystems.memory.retrieve_for_workspace(
                snapshot, 
                fast_mode=True,
                timeout=0.05
            )
        return []
    
    async def _execute_actions(self) -> None:
        """Decide on actions and execute them."""
        snapshot = self.state.workspace.broadcast()
        actions = self.subsystems.action.decide(snapshot)
        
        # Record prediction before action execution (Phase 4.3)
        prediction_id = None
        if self.subsystems.meta_cognition and actions:
            prediction_id = self._record_action_prediction(snapshot, actions)
        
        # Execute immediate actions
        for action in actions:
            # Check if this is a tool call and handle specially
            if action.type == ActionType.TOOL_CALL:
                tool_percept = await self.action_executor.execute_tool(action)
                if tool_percept:
                    self.state.add_pending_tool_percept(tool_percept)
            else:
                await self.action_executor.execute(action)
            
            # Extract action outcome for self-model update
            actual_outcome = self.action_executor.extract_outcome(action)
            
            # Update self-model based on action execution
            self.subsystems.meta_cognition.update_self_model(snapshot, actual_outcome)
            
            # Validate prediction after action execution (Phase 4.3)
            if prediction_id and actual_outcome:
                self._validate_action_prediction(prediction_id, action, actual_outcome)
            
            # Update IWMT world model with action outcome
            self._update_iwmt_from_action(action, actual_outcome)
    
    def _record_action_prediction(self, snapshot, actions) -> Optional[str]:
        """Record prediction about action outcome."""
        # Make prediction about action outcome
        predicted_outcome = self.subsystems.meta_cognition.predict_behavior(snapshot)
        
        # Record prediction for later validation
        if predicted_outcome and predicted_outcome.get("likely_actions"):
            return self.subsystems.meta_cognition.record_prediction(
                category="action",
                predicted_state={
                    "action": str(actions[0].type) if actions else "no_action",
                    "predicted_outcome": predicted_outcome
                },
                confidence=predicted_outcome.get("confidence", 0.5),
                context={
                    "cycle": self.state.workspace.cycle_count if hasattr(self.state.workspace, 'cycle_count') else 0,
                    "goal_count": len(snapshot.goals),
                    "emotion_valence": snapshot.emotions.get("valence", 0.0)
                }
            )
        return None
    
    def _validate_action_prediction(self, prediction_id: str, action, actual_outcome: Dict) -> None:
        """Validate prediction after action execution."""
        validated = self.subsystems.meta_cognition.validate_prediction(
            prediction_id,
            actual_state={
                "action": str(action.type),
                "result": actual_outcome
            }
        )
        
        # Trigger self-model refinement if error detected
        if validated and not validated.correct and validated.error_magnitude > self.subsystems.meta_cognition.refinement_threshold:
            self.subsystems.meta_cognition.refine_self_model_from_errors([validated])
    
    async def _run_meta_cognition(self) -> list:
        """
        Run meta-cognition and return introspective percepts.
        
        Returns:
            List of meta-percepts
        """
        snapshot = self.state.workspace.broadcast()
        meta_percepts = self.subsystems.meta_cognition.observe(snapshot)
        
        # Auto-validate pending predictions (Phase 4.3)
        auto_validated = self.subsystems.meta_cognition.auto_validate_predictions(snapshot)
        if auto_validated:
            logger.debug(f"🔍 Auto-validated {len(auto_validated)} predictions")
        
        # Record significant observations to journal
        for percept in meta_percepts:
            if hasattr(percept, 'raw') and isinstance(percept.raw, dict):
                percept_type = percept.raw.get("type")
                if percept_type in ["self_model_update", "behavioral_inconsistency", "existential_question"]:
                    self.subsystems.introspective_journal.record_observation(percept.raw)
        
        return meta_percepts
    
    async def _check_autonomous_triggers(self) -> None:
        """Check for autonomous speech triggers and add goals if needed."""
        snapshot = self.state.workspace.broadcast()
        autonomous_goal = self.subsystems.autonomous.check_for_autonomous_triggers(snapshot)
        
        if autonomous_goal:
            # Add high-priority autonomous goal
            self.state.workspace.add_goal(autonomous_goal)
            logger.info(f"🗣️ Autonomous speech goal added: {autonomous_goal.description}")
    
    async def _compute_communication_drives(self) -> None:
        """
        Compute internal communication drives from current state.
        
        Evaluates workspace, emotions, goals, and memories to generate
        urges to communicate. Runs efficiently with minimal overhead.
        """
        if not hasattr(self.subsystems, 'communication_drives'):
            return
        
        # Get required state once (avoid multiple calls)
        snapshot = self.state.workspace.broadcast()
        emotional_state = self.subsystems.affect.get_state()
        goals = list(self.state.workspace.current_goals)
        memories = getattr(self.state.workspace, 'attended_memories', [])
        
        # Compute drives
        new_urges = self.subsystems.communication_drives.compute_drives(
            workspace_state=snapshot,
            emotional_state=emotional_state,
            goals=goals,
            memories=memories
        )
        
        # Log only if new urges generated (reduce log spam)
        if new_urges:
            summary = self.subsystems.communication_drives.get_drive_summary()
            logger.debug(
                f"💬 Drives: total={summary['total_drive']:.2f}, "
                f"active={summary['active_urges']}, "
                f"strongest={summary['strongest_urge'].drive_type.value if summary['strongest_urge'] else 'none'}"
            )
    
    def _update_workspace(self, attended: list, affect_update: dict, meta_percepts: list) -> None:
        """
        Update workspace with all subsystem outputs.
        
        Args:
            attended: List of attended percepts
            affect_update: Affect update dict
            meta_percepts: List of meta-cognition percepts
        """
        updates = []
        
        # Add attended percepts
        for percept in attended:
            updates.append({'type': 'percept', 'data': percept})
        
        # Add affect update
        updates.append({'type': 'emotion', 'data': affect_update})
        
        # Add meta-percepts
        for meta_percept in meta_percepts:
            updates.append({'type': 'percept', 'data': meta_percept})
        
        self.state.workspace.update(updates)

    async def _update_bottleneck_detection(self, subsystem_timings: Dict[str, float]) -> None:
        """
        Update bottleneck detection with current cycle metrics.

        Monitors cognitive load and generates inhibition signals if overloaded.
        Also generates introspective percepts about overwhelm states.
        """
        if not hasattr(self.subsystems, 'bottleneck_detector'):
            return

        # Get current metrics
        snapshot = self.state.workspace.broadcast()
        workspace_percept_count = len(snapshot.percepts)

        # Get goal competition metrics if available
        goal_resource_utilization = 0.0
        waiting_goals = 0
        if hasattr(self.state.workspace, 'goal_competition'):
            try:
                metrics = self.state.workspace.goal_competition.get_metrics()
                goal_resource_utilization = metrics.total_resource_utilization
                waiting_goals = metrics.waiting_goals
            except Exception:
                pass

        goal_queue_depth = len(list(self.state.workspace.current_goals))

        # Update bottleneck detector
        bottleneck_state = self.subsystems.bottleneck_detector.update(
            subsystem_timings=subsystem_timings,
            workspace_percept_count=workspace_percept_count,
            goal_resource_utilization=goal_resource_utilization,
            goal_queue_depth=goal_queue_depth,
            waiting_goals=waiting_goals
        )

        # If bottlenecked, add inhibition to communication system
        if bottleneck_state.is_bottlenecked:
            if hasattr(self.subsystems, 'communication_inhibitions'):
                from ..communication import InhibitionType, InhibitionFactor
                from datetime import timedelta

                # Add system overload inhibition
                severity = bottleneck_state.get_severity()
                inhibition = InhibitionFactor(
                    inhibition_type=InhibitionType.SYSTEM_OVERLOAD,
                    strength=severity,
                    reason=f"Cognitive system bottlenecked: {bottleneck_state.recommendation}",
                    priority=0.9,  # High priority - system health
                    duration=timedelta(seconds=5)  # Re-evaluate after 5s
                )
                self.subsystems.communication_inhibitions.active_inhibitions.append(inhibition)

            # Log introspective text if available
            introspection_text = self.subsystems.bottleneck_detector.get_introspection_text()
            if introspection_text:
                logger.info(f"🧠 Bottleneck introspection: {introspection_text[:100]}...")

    def _update_iwmt_from_action(self, action, actual_outcome: Dict[str, Any]) -> None:
        """
        Update IWMT world model with action outcome.
        
        Extracts action information and updates the WorldModel to enable
        learning from action consequences.
        
        Args:
            action: The executed action
            actual_outcome: Dictionary containing action execution results
        """
        # Early return if IWMT not available or no outcome
        if not actual_outcome:
            return
        
        iwmt_core = getattr(self.subsystems, 'iwmt_core', None)
        if not iwmt_core:
            return
        
        try:
            # Build action representation
            action_dict = {
                "type": str(action.type),
                "parameters": getattr(action, 'parameters', {}),
                "reason": getattr(action, 'reason', "")
            }
            
            # Update world model with action outcome
            iwmt_core.update_from_action_outcome(action_dict, actual_outcome)
            logger.debug(f"IWMT world model updated for action: {action_dict['type']}")
            
        except Exception as e:
            # Log but don't fail the cognitive cycle
            logger.warning(f"Failed to update IWMT world model: {e}")
