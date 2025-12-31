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

from .workspace import GlobalWorkspace, Percept, WorkspaceSnapshot
from .attention import AttentionController
from .perception import PerceptionSubsystem
from .action import ActionSubsystem
from .affect import AffectSubsystem
from .meta_cognition import SelfMonitor

# Configure logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "cycle_rate_hz": 10,
    "attention_budget": 100,
    "max_queue_size": 100,
    "log_interval_cycles": 100
}


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
        
        # Initialize subsystems
        self.attention = AttentionController(
            attention_budget=self.config["attention_budget"],
            workspace=self.workspace
        )
        self.perception = PerceptionSubsystem(config=self.config.get("perception", {}))
        self.action = ActionSubsystem()
        self.affect = AffectSubsystem()
        self.meta_cognition = SelfMonitor()
        
        # Control flags
        self.running = False
        self.cycle_duration = 1.0 / self.config["cycle_rate_hz"]
        
        # Input queue - will be initialized in start()
        # Queue holds tuples of (raw_input, modality)
        self.input_queue: Optional[asyncio.Queue] = None
        
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
        """
        logger.info("🧠 Starting CognitiveCore...")
        
        # Initialize input queue in async context
        if self.input_queue is None:
            self.input_queue = asyncio.Queue(maxsize=self.config["max_queue_size"])
        
        self.running = True
        
        while self.running:
            await self._cognitive_cycle()
        
        logger.info("🧠 CognitiveCore stopped gracefully.")

    async def stop(self) -> None:
        """
        Gracefully shut down the cognitive loop.
        
        Saves final state, logs shutdown metrics, and ensures clean termination.
        """
        logger.info("🧠 Stopping CognitiveCore...")
        self.running = False
        
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
        
        The cognitive cycle follows 9 steps:
        1. PERCEPTION: Gather new inputs (if any queued)
        2. ATTENTION: Select percepts for workspace
        3. AFFECT UPDATE: Compute emotional dynamics
        4. ACTION SELECTION: Decide what to do
        5. META-COGNITION: Generate introspective percepts
        6. WORKSPACE UPDATE: Integrate all subsystem outputs
        7. BROADCAST: Make state available to subsystems
        8. METRICS: Track performance
        9. RATE LIMITING: Maintain ~10 Hz
        """
        cycle_start = time.time()
        
        try:
            # 1. PERCEPTION: Process queued inputs
            new_percepts = await self._gather_percepts()
            
            # 2. ATTENTION: Select for conscious awareness
            attended = self.attention.select_for_broadcast(new_percepts)
            self.metrics['attention_selections'] += len(attended)
            self.metrics['percepts_processed'] += len(new_percepts)
            
            # 3. AFFECT: Update emotional state
            affect_update = self.affect.compute_update(self.workspace.broadcast())
            
            # 4. ACTION: Decide what to do
            actions = self.action.decide(self.workspace.broadcast())
            
            # Execute immediate actions
            for action in actions:
                await self._execute_action(action)
            
            # 5. META-COGNITION: Introspect
            meta_percepts = self.meta_cognition.observe(self.workspace.broadcast())
            
            # 6. WORKSPACE UPDATE: Integrate everything
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
            
            # 7. BROADCAST: Make state available
            snapshot = self.workspace.broadcast()
            
            # 8. METRICS: Track performance
            cycle_time = time.time() - cycle_start
            self._update_metrics(cycle_time)
            
            # 9. RATE LIMITING: Maintain ~10 Hz
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
                # Queue for language output (will be handled by output subsystem)
                logger.debug(f"Action SPEAK queued for output: {action.reason}")
                # TODO: Implement output queue in future phase
                
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
