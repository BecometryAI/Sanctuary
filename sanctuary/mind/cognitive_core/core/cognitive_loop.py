"""
Main cognitive loop orchestration.

Coordinates the ~10Hz recurrent cognitive loop and idle loop for continuous consciousness.
"""

from __future__ import annotations

import asyncio
import time
import logging
from typing import TYPE_CHECKING, Optional, Dict

if TYPE_CHECKING:
    from .subsystem_coordinator import SubsystemCoordinator
    from .state_manager import StateManager
    from .timing import TimingManager
    from .cycle_executor import CycleExecutor

logger = logging.getLogger(__name__)


class CognitiveLoop:
    """
    Orchestrates the main cognitive loop.
    
    Responsibilities:
    - Run the active (fast) ~10Hz loop for conversation processing
    - Run the idle (slow) loop for continuous consciousness
    - Coordinate timing and rate limiting
    """
    
    def __init__(
        self,
        subsystems: 'SubsystemCoordinator',
        state: 'StateManager',
        timing: 'TimingManager',
        cycle_executor: 'CycleExecutor'
    ):
        """
        Initialize cognitive loop orchestrator.
        
        Args:
            subsystems: SubsystemCoordinator instance
            state: StateManager instance
            timing: TimingManager instance
            cycle_executor: CycleExecutor instance
        """
        self.subsystems = subsystems
        self.state = state
        self.timing = timing
        self.cycle_executor = cycle_executor
    
    async def run(self) -> None:
        """
        Start both active and idle cognitive loops.
        
        Runs continuously until stop() is called via state.running flag.
        """
        # Start active cognitive loop (existing fast cycle for conversations)
        self.state.active_task = asyncio.create_task(self._active_loop())
        
        # Start idle cognitive loop (new slow cycle for continuous consciousness)
        self.state.idle_task = asyncio.create_task(
            self.subsystems.continuous_consciousness.start_idle_loop()
        )
        
        logger.info("🧠 Cognitive loops started (active + idle)")
        
        # Wait for both tasks to complete (they run until stop() is called)
        await asyncio.gather(self.state.active_task, self.state.idle_task, return_exceptions=True)
        
        logger.info("🧠 CognitiveCore stopped gracefully.")
    
    async def _active_loop(self) -> None:
        """
        Run the active (fast) cognitive loop for conversation processing.
        
        This is the main loop that runs at ~10 Hz for active conversation.
        """
        while self.state.running:
            await self._cognitive_cycle()
        
        logger.info("🧠 Active cognitive loop stopped.")
    
    async def _cognitive_cycle(self) -> None:
        """
        Execute one complete cognitive cycle with timing.
        """
        cycle_start = time.time()
        
        try:
            # Execute the 9-step cognitive cycle
            subsystem_timings = await self.cycle_executor.execute_cycle()
            
            # Calculate cycle time
            cycle_time = time.time() - cycle_start
            
            # Check timing thresholds and log warnings if needed
            self.timing.check_cycle_timing(cycle_time, self.timing.metrics['total_cycles'] + 1)
            
            # Update metrics
            self.timing.update_metrics(cycle_time, subsystem_timings)
            
            # Periodic accuracy snapshots (Phase 4.3)
            if self.timing.metrics['total_cycles'] % 100 == 0:
                self._record_accuracy_snapshot()
            
            # Rate limiting: maintain ~10 Hz
            sleep_time = self.timing.calculate_sleep_time(cycle_time)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in cognitive cycle: {e}", exc_info=True)
    
    def _record_accuracy_snapshot(self) -> None:
        """Record periodic accuracy snapshot for meta-cognition."""
        if self.subsystems.meta_cognition:
            snapshot = self.subsystems.meta_cognition.record_accuracy_snapshot()
            logger.info(
                f"📸 Accuracy snapshot: {snapshot.overall_accuracy:.1%} accuracy, "
                f"{snapshot.prediction_count} predictions"
            )
    
    async def process_language_input(self, text: str, context: Optional[Dict] = None) -> None:
        """
        Process natural language input through the language input parser.
        
        Parses the text into structured Goals and Percepts, adds the goals to
        the workspace, and queues the percept for processing in the next cycle.
        
        Args:
            text: Natural language user input
            context: Optional additional context for parsing
            
        Raises:
            RuntimeError: If called before queues initialized
        """
        if self.state.input_queue is None:
            logger.error("Cannot process language input: queues not initialized")
            raise RuntimeError("Queues must be initialized before processing language input")
        
        # Update temporal awareness - record that interaction occurred
        self.subsystems.temporal_awareness.update_last_interaction_time()
        
        # Record input in communication drive system
        if hasattr(self.subsystems, 'communication_drives'):
            self.subsystems.communication_drives.record_input()
        
        # Update new temporal grounding system
        if hasattr(self.subsystems, 'temporal_grounding'):
            temporal_context = self.subsystems.temporal_grounding.on_interaction()
            
            # If it's a new session, handle session start
            if temporal_context.is_new_session:
                logger.info(f"🔔 New session detected: #{temporal_context.session_number}")
                # Could add session start percept here if needed
        
        # Parse input into structured components
        parse_result = await self.subsystems.language_input.parse(text, context)
        
        # Add goals to workspace
        for goal in parse_result.goals:
            self.state.workspace.add_goal(goal)
        
        # Queue percept for next cycle
        try:
            self.state.input_queue.put_nowait((parse_result.percept.raw, "text"))
            logger.info(f"📥 Processed language input: {len(parse_result.goals)} goals added")
        except asyncio.QueueFull:
            logger.warning("Input queue full, dropping percept from language input")
    
    async def chat(self, message: str, timeout: float = 5.0) -> str:
        """
        Convenience method: Send message and get text response.
        
        Args:
            message: User's text message
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            Response text string, or "..." if no response within timeout
        """
        # Process input
        await self.process_language_input(message)
        
        # Wait for response
        output = await self.state.get_response(timeout)
        
        if output and output.get("type") == "SPEAK":
            return output.get("text", "...")
        
        return "..."
