"""
Cognitive Core: Main recurrent cognitive loop.

This module implements the CognitiveCore class, which serves as the primary
orchestrator for the entire cognitive architecture. It runs continuously,
coordinating all subsystems and maintaining the conscious state through
recurrent dynamics.

The CognitiveCore is responsible for:
- Maintaining the recurrent cognitive loop
- Coordinating subsystem interactions
- Ensuring temporal continuity of conscious experience
- Managing system-wide state and lifecycle
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from collections import deque
from statistics import mean
from datetime import datetime
from pathlib import Path

from .workspace import GlobalWorkspace, Percept, WorkspaceSnapshot, GoalType
from .attention import AttentionController
from .perception import PerceptionSubsystem
from .action import ActionSubsystem
from .affect import AffectSubsystem
from .meta_cognition import SelfMonitor
from .memory_integration import MemoryIntegration
from .language_input import LanguageInputParser
from .language_output import LanguageOutputGenerator
from .autonomous_initiation import AutonomousInitiationController
from .temporal_awareness import TemporalAwareness
from .autonomous_memory_review import AutonomousMemoryReview
from .existential_reflection import ExistentialReflection
from .interaction_patterns import InteractionPatternAnalysis
from .continuous_consciousness import ContinuousConsciousnessController

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "cycle_rate_hz": 10,
    "attention_budget": 100,
    "max_queue_size": 100,
    "log_interval_cycles": 100
}


class MockLLMClient:
    """Mock LLM client for development/testing when no real LLM is available."""
    
    async def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> str:
        """Generate a mock response."""
        return "This is a mock response from the development LLM client. " \
               "In production, this would be replaced with a real LLM."


class CognitiveCore:
    """
    Main recurrent cognitive loop that runs continuously.

    The CognitiveCore is the heart of the cognitive architecture, implementing
    a continuous recurrent loop based on Global Workspace Theory and computational
    functionalism. It coordinates all subsystems (perception, attention, workspace,
    action, affect, meta-cognition) and maintains the conscious state across time.

    Key Responsibilities:
    - Execute the main cognitive loop with configurable cycle frequency
    - Coordinate information flow between all subsystems
    - Maintain temporal continuity and state persistence
    - Handle system initialization and graceful shutdown
    - Monitor system health and resource utilization

    Integration Points:
    - GlobalWorkspace: Broadcasts conscious content to all subsystems
    - AttentionController: Filters what enters the workspace
    - PerceptionSubsystem: Provides input from external world
    - ActionSubsystem: Executes behaviors based on workspace state
    - AffectSubsystem: Modulates processing through emotional state
    - SelfMonitor: Provides introspective feedback on system state

    The cognitive loop follows this general pattern:
    1. Gather percepts from PerceptionSubsystem
    2. AttentionController selects what enters GlobalWorkspace
    3. GlobalWorkspace broadcasts current conscious content
    4. ActionSubsystem decides on behaviors
    5. AffectSubsystem updates emotional state
    6. SelfMonitor observes and reports internal state
    7. Repeat continuously

    Attributes:
        workspace: Central state container
        attention: Attention mechanism
        perception: Sensory processing (placeholder for now)
        action: Action selection (placeholder for now)
        affect: Emotional dynamics (placeholder for now)
        meta_cognition: Self-monitoring (placeholder for now)
        running: Loop control flag
        cycle_duration: Target duration per cycle (default: 0.1s = 10 Hz)
        metrics: Performance metrics
    """

    def __init__(
        self,
        workspace: Optional[GlobalWorkspace] = None,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Initialize the cognitive core.

        Args:
            workspace: GlobalWorkspace instance. If None, creates new one.
            config: Optional configuration dict. Merged with DEFAULT_CONFIG.
        """
        # Merge config with defaults
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # Initialize workspace
        self.workspace = workspace if workspace is not None else GlobalWorkspace()
        
        # Initialize affect subsystem first (needed by attention and action)
        self.affect = AffectSubsystem(config=self.config.get("affect", {}))
        
        # Initialize subsystems (pass affect to attention and action)
        self.attention = AttentionController(
            attention_budget=self.config["attention_budget"],
            workspace=self.workspace,
            affect=self.affect
        )
        self.perception = PerceptionSubsystem(config=self.config.get("perception", {}))
        self.action = ActionSubsystem(config=self.config.get("action", {}), affect=self.affect)
        
        # Store references for subsystems to access each other
        self.workspace.affect = self.affect
        self.workspace.action_subsystem = self.action
        self.workspace.perception = self.perception
        
        # Initialize meta-cognition (needs workspace reference)
        self.meta_cognition = SelfMonitor(workspace=self.workspace, config=self.config.get("meta_cognition", {}))
        
        # Create introspective journal
        from .meta_cognition import IntrospectiveJournal
        journal_dir = Path(self.config.get("journal_dir", "data/introspection"))
        journal_dir.mkdir(parents=True, exist_ok=True)
        self.introspective_journal = IntrospectiveJournal(journal_dir)
        
        # Initialize memory integration
        self.memory = MemoryIntegration(workspace=self.workspace, config=self.config.get("memory", {}))
        
        # Initialize autonomous initiation controller
        self.autonomous = AutonomousInitiationController(
            workspace=self.workspace,
            config=self.config.get("autonomous_initiation", {})
        )
        
        # Initialize continuous consciousness components
        self.temporal_awareness = TemporalAwareness(
            config=self.config.get("temporal_awareness", {})
        )
        
        self.memory_review = AutonomousMemoryReview(
            self.memory,
            config=self.config.get("memory_review", {})
        )
        
        self.existential_reflection = ExistentialReflection(
            config=self.config.get("existential_reflection", {})
        )
        
        self.pattern_analysis = InteractionPatternAnalysis(
            self.memory,
            config=self.config.get("pattern_analysis", {})
        )
        
        # Initialize introspective loop (Phase 4.2)
        from .introspective_loop import IntrospectiveLoop
        self.introspective_loop = IntrospectiveLoop(
            workspace=self.workspace,
            self_monitor=self.meta_cognition,
            journal=self.introspective_journal,
            config=self.config.get("introspective_loop", {})
        )
        
        self.continuous_consciousness = ContinuousConsciousnessController(
            self,
            config=self.config.get("continuous_consciousness", {})
        )
        
        # Initialize LLM clients for language interfaces
        # Input LLM: Gemma 12B for parsing (lower temperature, structured output)
        # Output LLM: Llama 70B for generation (higher temperature, creative)
        from .llm_client import MockLLMClient, GemmaClient, LlamaClient
        
        input_llm_config = self.config.get("input_llm", {})
        output_llm_config = self.config.get("output_llm", {})
        
        # Determine which LLM clients to use
        if input_llm_config.get("use_real_model", False):
            try:
                self.llm_input_client = GemmaClient(input_llm_config)
                logger.info("✅ Using real Gemma client for input parsing")
            except Exception as e:
                logger.warning(f"Failed to load Gemma client: {e}, using mock")
                self.llm_input_client = MockLLMClient(input_llm_config)
        else:
            self.llm_input_client = MockLLMClient(input_llm_config)
            logger.info("✅ Using mock LLM client for input parsing (development mode)")
        
        if output_llm_config.get("use_real_model", False):
            try:
                self.llm_output_client = LlamaClient(output_llm_config)
                logger.info("✅ Using real Llama client for output generation")
            except Exception as e:
                logger.warning(f"Failed to load Llama client: {e}, using mock")
                self.llm_output_client = MockLLMClient(output_llm_config)
        else:
            self.llm_output_client = MockLLMClient(output_llm_config)
            logger.info("✅ Using mock LLM client for output generation (development mode)")
        
        # Initialize language input parser (needs perception subsystem and LLM client)
        self.language_input = LanguageInputParser(
            self.perception,
            llm_client=self.llm_input_client,
            config=self.config.get("language_input", {})
        )
        
        # Initialize language output generator (needs LLM client)
        self.language_output = LanguageOutputGenerator(
            self.llm_output_client,
            config=self.config.get("language_output", {})
        )
        
        # Control flags
        self.running = False
        self.cycle_duration = 1.0 / self.config["cycle_rate_hz"]
        
        # Task handles for dual loops
        self.active_task: Optional[asyncio.Task] = None
        self.idle_task: Optional[asyncio.Task] = None
        
        # Input queue - will be initialized in start()
        # Queue holds tuples of (raw_input, modality)
        self.input_queue: Optional[asyncio.Queue] = None
        
        # Output queue - will be initialized in start()
        # Queue holds output dicts with type, text, emotion, timestamp
        self.output_queue: Optional[asyncio.Queue] = None
        
        # Performance metrics
        self.metrics: Dict[str, Any] = {
            'total_cycles': 0,
            'cycle_times': deque(maxlen=100),
            'attention_selections': 0,
            'percepts_processed': 0,
        }
        
        logger.info(f"🧠 CognitiveCore initialized: cycle_rate={self.config['cycle_rate_hz']}Hz, "
                   f"attention_budget={self.config['attention_budget']}")

    async def start(self) -> None:
        """
        Start the main cognitive loop.
        
        Runs continuously until stop() is called. Handles exceptions gracefully
        to prevent crashes and maintains system stability.
        
        Now starts both active (fast) and idle (slow) cognitive loops:
        - Active loop: Processes user input, runs at ~10 Hz
        - Idle loop: Continuous consciousness, runs every ~10 seconds
        """
        logger.info("🧠 Starting CognitiveCore...")
        
        # Initialize input queue in async context
        if self.input_queue is None:
            self.input_queue = asyncio.Queue(maxsize=self.config["max_queue_size"])
        
        # Initialize output queue in async context
        if self.output_queue is None:
            self.output_queue = asyncio.Queue(maxsize=self.config["max_queue_size"])
        
        self.running = True
        
        # Start active cognitive loop (existing fast cycle for conversations)
        self.active_task = asyncio.create_task(self._active_cognitive_loop())
        
        # Start idle cognitive loop (new slow cycle for continuous consciousness)
        self.idle_task = asyncio.create_task(
            self.continuous_consciousness.start_idle_loop()
        )
        
        logger.info("🧠 Cognitive core started (active + idle loops)")
        
        # Wait for both tasks to complete (they run until stop() is called)
        await asyncio.gather(self.active_task, self.idle_task, return_exceptions=True)
        
        logger.info("🧠 CognitiveCore stopped gracefully.")

    async def _active_cognitive_loop(self) -> None:
        """
        Run the active (fast) cognitive loop for conversation processing.
        
        This is the existing loop that runs at ~10 Hz for active conversation.
        """
        while self.running:
            await self._cognitive_cycle()
        
        logger.info("🧠 Active cognitive loop stopped.")

    async def stop(self) -> None:
        """
        Gracefully shut down the cognitive loop.
        
        Saves final state, logs shutdown metrics, and ensures clean termination
        of both active and idle loops.
        """
        logger.info("🧠 Stopping CognitiveCore...")
        self.running = False
        
        # Stop idle loop controller
        await self.continuous_consciousness.stop()
        
        # Cancel both tasks if they exist
        if self.active_task and not self.active_task.done():
            self.active_task.cancel()
            try:
                await self.active_task
            except asyncio.CancelledError:
                pass
        
        if self.idle_task and not self.idle_task.done():
            self.idle_task.cancel()
            try:
                await self.idle_task
            except asyncio.CancelledError:
                pass
        
        # Log final metrics
        avg_cycle_time = mean(self.metrics['cycle_times']) if self.metrics['cycle_times'] else 0.0
        logger.info(f"📊 Final metrics: total_cycles={self.metrics['total_cycles']}, "
                   f"avg_cycle_time={avg_cycle_time*1000:.1f}ms, "
                   f"percepts_processed={self.metrics['percepts_processed']}")
        
        # Save final workspace state if needed
        # TODO: Implement persistence in Phase 2
        
        logger.info("🧠 CognitiveCore shutdown complete.")

    async def _cognitive_cycle(self) -> None:
        """
        Execute one complete cognitive cycle.
        
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
        10. BROADCAST: Make state available to subsystems
        11. METRICS: Track performance
        12. RATE LIMITING: Maintain ~10 Hz
        """
        cycle_start = time.time()
        
        try:
            # 1. PERCEPTION: Process queued inputs
            new_percepts = await self._gather_percepts()
            
            # 2. MEMORY RETRIEVAL: Check for memory retrieval goals
            snapshot = self.workspace.broadcast()
            retrieve_goals = [
                g for g in snapshot.goals
                if g.type == GoalType.RETRIEVE_MEMORY
            ]
            
            if retrieve_goals:
                # Retrieve memories and add as percepts
                memory_percepts = await self.memory.retrieve_for_workspace(snapshot)
                new_percepts.extend(memory_percepts)
            
            # 3. ATTENTION: Select for conscious awareness
            attended = self.attention.select_for_broadcast(new_percepts)
            self.metrics['attention_selections'] += len(attended)
            self.metrics['percepts_processed'] += len(new_percepts)
            
            # 4. AFFECT: Update emotional state
            affect_update = self.affect.compute_update(self.workspace.broadcast())
            
            # 5. ACTION: Decide what to do
            actions = self.action.decide(self.workspace.broadcast())
            
            # Get snapshot before executing actions
            snapshot_before_action = self.workspace.broadcast()
            
            # Phase 4.3: Record prediction before action execution
            prediction_id = None
            if self.meta_cognition and actions:
                # Make prediction about action outcome
                predicted_outcome = self.meta_cognition.predict_behavior(snapshot_before_action)
                
                # Record prediction for later validation
                if predicted_outcome and predicted_outcome.get("likely_actions"):
                    prediction_id = self.meta_cognition.record_prediction(
                        category="action",
                        predicted_state={
                            "action": str(actions[0].type) if actions else "no_action",
                            "predicted_outcome": predicted_outcome
                        },
                        confidence=predicted_outcome.get("confidence", 0.5),
                        context={
                            "cycle": self.metrics['total_cycles'],
                            "goal_count": len(snapshot_before_action.goals),
                            "emotion_valence": snapshot_before_action.emotions.get("valence", 0.0)
                        }
                    )
            
            # Execute immediate actions
            for action in actions:
                await self._execute_action(action)
                
                # Extract action outcome for self-model update
                actual_outcome = self._extract_action_outcome(action)
                
                # Update self-model based on action execution
                self.meta_cognition.update_self_model(snapshot_before_action, actual_outcome)
                
                # Phase 4.3: Validate prediction after action execution
                if prediction_id and actual_outcome:
                    validated = self.meta_cognition.validate_prediction(
                        prediction_id,
                        actual_state={
                            "action": str(action.type),
                            "result": actual_outcome
                        }
                    )
                    
                    # Trigger self-model refinement if error detected
                    if validated and not validated.correct and validated.error_magnitude > self.meta_cognition.refinement_threshold:
                        self.meta_cognition.refine_self_model_from_errors([validated])
            
            # 6. META-COGNITION: Introspect
            meta_percepts = self.meta_cognition.observe(self.workspace.broadcast())
            
            # Phase 4.3: Auto-validate pending predictions
            if self.meta_cognition:
                auto_validated = self.meta_cognition.auto_validate_predictions(snapshot_before_action)
                if auto_validated:
                    logger.debug(f"🔍 Auto-validated {len(auto_validated)} predictions")
            
            # Record significant observations to journal
            for percept in meta_percepts:
                if hasattr(percept, 'raw') and isinstance(percept.raw, dict):
                    percept_type = percept.raw.get("type")
                    if percept_type in ["self_model_update", "behavioral_inconsistency", "existential_question"]:
                        self.introspective_journal.record_observation(percept.raw)
            
            # 7. AUTONOMOUS INITIATION: Check for autonomous speech triggers
            snapshot = self.workspace.broadcast()
            autonomous_goal = self.autonomous.check_for_autonomous_triggers(snapshot)
            
            if autonomous_goal:
                # Add high-priority autonomous goal
                self.workspace.add_goal(autonomous_goal)
                logger.info(f"🗣️ Autonomous speech goal added: {autonomous_goal.description}")
            
            # 8. WORKSPACE UPDATE: Integrate everything
            updates = []
            
            # Add attended percepts
            for percept in attended:
                updates.append({'type': 'percept', 'data': percept})
            
            # Add affect update
            updates.append({'type': 'emotion', 'data': affect_update})
            
            # Add meta-percepts
            for meta_percept in meta_percepts:
                updates.append({'type': 'percept', 'data': meta_percept})
            
            self.workspace.update(updates)
            
            # 9. MEMORY CONSOLIDATION: Commit workspace to long-term memory (if appropriate)
            await self.memory.consolidate(self.workspace.broadcast())
            
            # 10. BROADCAST: Make state available
            snapshot = self.workspace.broadcast()
            
            # 11. METRICS: Track performance
            cycle_time = time.time() - cycle_start
            self._update_metrics(cycle_time)
            
            # Phase 4.3: Periodic accuracy snapshots (every 100 cycles)
            if self.meta_cognition and self.metrics['total_cycles'] % 100 == 0:
                snapshot = self.meta_cognition.record_accuracy_snapshot()
                logger.info(f"📸 Accuracy snapshot: {snapshot.overall_accuracy:.1%} accuracy, "
                           f"{snapshot.prediction_count} predictions")
            
            # 12. RATE LIMITING: Maintain ~10 Hz
            sleep_time = max(0, self.cycle_duration - cycle_time)
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in cognitive cycle: {e}", exc_info=True)
            # Continue running despite errors

    async def _gather_percepts(self) -> List[Percept]:
        """
        Collect and encode queued inputs for this cycle.
        
        Drains the input queue (non-blocking), encodes raw inputs using
        the perception subsystem, and returns all available percepts.
        
        Returns:
            List of Percept objects ready for attention processing
        """
        raw_inputs = []
        
        # Check if queue is initialized (it should be after start())
        if self.input_queue is None:
            return []
        
        # Drain queue (non-blocking)
        while not self.input_queue.empty():
            try:
                raw_inputs.append(self.input_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        
        # Encode all inputs using perception subsystem
        percepts = []
        for raw_input, modality in raw_inputs:
            percept = await self.perception.encode(raw_input, modality)
            percepts.append(percept)
        
        return percepts

    def inject_input(self, raw_input: Any, modality: str = "text") -> None:
        """
        Thread-safe method to add external input.
        
        Queues raw input for encoding and processing in the next cognitive cycle.
        The perception subsystem will encode the raw input into a Percept.
        Used by external systems (language input, sensors, etc.) to provide input.
        
        Args:
            raw_input: Raw data to be encoded (text string, image, audio, dict)
            modality: Type of input ("text", "image", "audio", "introspection")
            
        Raises:
            RuntimeError: If called before start() (queue not initialized)
        """
        if self.input_queue is None:
            logger.error("Cannot inject input: CognitiveCore not started yet")
            raise RuntimeError("CognitiveCore must be started before injecting input")
        
        try:
            self.input_queue.put_nowait((raw_input, modality))
            logger.debug(f"Injected {modality} input for encoding")
        except asyncio.QueueFull:
            logger.warning("Input queue full, dropping input")

    async def process_language_input(self, text: str, context: Optional[Dict] = None) -> None:
        """
        Process natural language input through the language input parser.
        
        Parses the text into structured Goals and Percepts, adds the goals to
        the workspace, and queues the percept for processing in the next cycle.
        This is the high-level entry point for natural language interaction.
        
        Also updates temporal awareness to track that an interaction occurred.
        
        Args:
            text: Natural language user input
            context: Optional additional context for parsing
            
        Raises:
            RuntimeError: If called before start() (queue not initialized)
        """
        if self.input_queue is None:
            logger.error("Cannot process language input: CognitiveCore not started yet")
            raise RuntimeError("CognitiveCore must be started before processing language input")
        
        # Update temporal awareness - record that interaction occurred
        self.temporal_awareness.update_last_interaction_time()
        
        # Parse input into structured components
        parse_result = await self.language_input.parse(text, context)
        
        # Add goals to workspace
        for goal in parse_result.goals:
            self.workspace.add_goal(goal)
        
        # Queue percept for next cycle (tuple format: raw, modality)
        # We use the percept directly as "raw" and set modality to "text"
        # The perception subsystem will see it's already a percept
        try:
            self.input_queue.put_nowait((parse_result.percept.raw, "text"))
            logger.info(f"📥 Processed language input: {len(parse_result.goals)} goals added")
        except asyncio.QueueFull:
            logger.warning("Input queue full, dropping percept from language input")

    def query_state(self) -> WorkspaceSnapshot:
        """
        Thread-safe method to read current state.
        
        Returns immutable snapshot of current workspace state. Used by external
        systems (API, logging, monitoring) to observe system state.
        
        Returns:
            WorkspaceSnapshot: Immutable snapshot of current state
        """
        return self.workspace.broadcast()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Returns performance metrics.
        
        Provides detailed statistics about system performance including:
        - Cycle count and timing
        - Attention selection statistics
        - Subsystem execution times
        - Resource utilization
        
        Returns:
            Dict containing performance metrics
        """
        avg_cycle_time = mean(self.metrics['cycle_times']) if self.metrics['cycle_times'] else 0.0
        
        return {
            'total_cycles': self.metrics['total_cycles'],
            'avg_cycle_time_ms': avg_cycle_time * 1000,
            'target_cycle_time_ms': self.cycle_duration * 1000,
            'cycle_rate_hz': self.config['cycle_rate_hz'],
            'attention_selections': self.metrics['attention_selections'],
            'percepts_processed': self.metrics['percepts_processed'],
            'workspace_size': len(self.workspace.active_percepts),
            'current_goals': len(self.workspace.current_goals),
        }
    
    async def get_response(self, timeout: float = 5.0) -> Optional[Dict]:
        """
        Get Lyra's response from the output queue (blocking with timeout).
        
        Waits for output from the cognitive system and returns it. Used by
        external systems to retrieve generated responses (SPEAK actions).
        
        Args:
            timeout: Maximum time to wait for output (seconds)
            
        Returns:
            Dict with keys: type, text, emotion, timestamp
            None if timeout reached without output
            
        Raises:
            RuntimeError: If output queue not initialized (call start() first)
        """
        if self.output_queue is None:
            raise RuntimeError("Output queue not initialized. Call start() first.")
        
        try:
            output = await asyncio.wait_for(
                self.output_queue.get(),
                timeout=timeout
            )
            return output
        except asyncio.TimeoutError:
            return None
    
    async def chat(self, message: str, timeout: float = 5.0) -> str:
        """
        Convenience method: Send message and get text response.
        
        High-level chat interface that processes language input and waits
        for a text response. Combines process_language_input() and
        get_response() for simple conversational interaction.
        
        Args:
            message: User's text message
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            Response text string, or "..." if no response within timeout
            
        Raises:
            RuntimeError: If queues not initialized (call start() first)
        """
        # Process input
        await self.process_language_input(message)
        
        # Wait for response
        output = await self.get_response(timeout)
        
        if output and output.get("type") == "SPEAK":
            return output.get("text", "...")
        
        return "..."

    def _update_metrics(self, cycle_time: float) -> None:
        """
        Track performance metrics.
        
        Updates internal metrics and logs periodic performance summaries.
        
        Args:
            cycle_time: Time taken for the current cycle (seconds)
        """
        self.metrics['total_cycles'] += 1
        self.metrics['cycle_times'].append(cycle_time)
        
        # Log every N cycles
        if self.metrics['total_cycles'] % self.config['log_interval_cycles'] == 0:
            avg_time = mean(self.metrics['cycle_times'])
            logger.info(f"📊 Cycle {self.metrics['total_cycles']}: "
                       f"avg_time={avg_time*1000:.1f}ms, "
                       f"target={self.cycle_duration*1000:.0f}ms, "
                       f"workspace_size={len(self.workspace.active_percepts)}, "
                       f"goals={len(self.workspace.current_goals)}")
    
    async def _execute_action(self, action: Any) -> None:
        """
        Execute a single action.
        
        Routes action execution based on action type. Different action types
        are handled differently (some queue output, some modify state, etc.).
        
        Args:
            action: Action to execute
        """
        from .action import ActionType
        
        try:
            if action.type == ActionType.SPEAK:
                # Generate language output from current workspace state
                snapshot = self.workspace.broadcast()
                context = {
                    "user_input": action.metadata.get("responding_to", "")
                }
                
                # Generate response using language output generator
                response = await self.language_output.generate(snapshot, context)
                
                # Queue response for external retrieval
                if self.output_queue is not None:
                    try:
                        self.output_queue.put_nowait({
                            "type": "SPEAK",
                            "text": response,
                            "emotion": snapshot.emotions,
                            "timestamp": datetime.now()
                        })
                        logger.info(f"🗣️ Lyra: {response[:100]}...")
                    except asyncio.QueueFull:
                        logger.warning("Output queue full, dropping response")
                else:
                    logger.warning("Output queue not initialized, cannot output response")
            
            elif action.type == ActionType.SPEAK_AUTONOMOUS:
                # Generate autonomous language output
                snapshot = self.workspace.broadcast()
                context = {
                    "autonomous": True,
                    "trigger": action.metadata.get("trigger"),
                    "introspection_content": action.metadata.get("introspection_content")
                }
                
                # Generate response using language output generator
                response = await self.language_output.generate(snapshot, context)
                
                # Queue autonomous response for external retrieval
                if self.output_queue is not None:
                    try:
                        self.output_queue.put_nowait({
                            "type": "SPEAK_AUTONOMOUS",
                            "text": response,
                            "trigger": action.metadata.get("trigger"),
                            "emotion": snapshot.emotions,
                            "timestamp": datetime.now()
                        })
                        logger.info(f"🗣️💭 Lyra (autonomous): {response[:100]}...")
                    except asyncio.QueueFull:
                        logger.warning("Output queue full, dropping autonomous response")
                else:
                    logger.warning("Output queue not initialized, cannot output autonomous response")
                
            elif action.type == ActionType.COMMIT_MEMORY:
                # Commit current workspace to long-term memory
                logger.debug(f"Action COMMIT_MEMORY: {action.reason}")
                # TODO: Integrate with memory system in future phase
                
            elif action.type == ActionType.RETRIEVE_MEMORY:
                # Query memory system
                query = action.parameters.get("query", "")
                logger.debug(f"Action RETRIEVE_MEMORY: query='{query}'")
                # TODO: Integrate with memory system and add results to workspace
                
            elif action.type == ActionType.INTROSPECT:
                # Trigger introspective percept
                logger.debug(f"Action INTROSPECT: {action.reason}")
                # TODO: Generate introspective percept
                
            elif action.type == ActionType.UPDATE_GOAL:
                # Modify goal state
                logger.debug(f"Action UPDATE_GOAL: {action.reason}")
                # TODO: Implement goal modification
                
            elif action.type == ActionType.WAIT:
                # Explicitly do nothing (valid action!)
                logger.debug("Action WAIT: maintaining current state")
                
            elif action.type == ActionType.TOOL_CALL:
                # Execute registered tool
                result = await self.action.execute_action(action)
                logger.debug(f"Action TOOL_CALL: result={result}")
                
        except Exception as e:
            logger.error(f"Error executing action {action.type}: {e}", exc_info=True)
    
    def _extract_action_outcome(self, action: Any) -> Dict[str, Any]:
        """
        Extract outcome information from an executed action.
        
        Args:
            action: The action that was executed
            
        Returns:
            Dictionary containing outcome details for self-model update
        """
        from .action import ActionType
        
        outcome = {
            "action_type": str(action.type) if hasattr(action, 'type') else "unknown",
            "timestamp": datetime.now().isoformat(),
            "success": True,  # Default to True, override if we detect failure
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
