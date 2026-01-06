"""
State management for the cognitive core.

Handles workspace state, input/output queues, and metrics tracking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..workspace import GlobalWorkspace, Percept, WorkspaceSnapshot

logger = logging.getLogger(__name__)


class StateManager:
    """
    Manages all state for the cognitive core.
    
    Responsibilities:
    - Workspace state management
    - Input/output queue management
    - Metrics tracking (delegated to TimingManager)
    - Pending percepts (tool feedback loop)
    """
    
    def __init__(self, workspace: Optional[GlobalWorkspace], config: Dict[str, Any]):
        """
        Initialize state manager with validated configuration.
        
        Args:
            workspace: GlobalWorkspace instance or None to create new
            config: Configuration dict with max_queue_size parameter
            
        Raises:
            ValueError: If configuration values are invalid
        """
        self.config = config
        self.workspace = workspace if workspace is not None else GlobalWorkspace()
        
        # Validate queue size
        max_queue_size = config.get("max_queue_size", 100)
        if max_queue_size <= 0:
            raise ValueError(f"max_queue_size must be positive, got {max_queue_size}")
        
        # Control flags
        self.running = False
        
        # Task handles for dual loops
        self.active_task: Optional[asyncio.Task] = None
        self.idle_task: Optional[asyncio.Task] = None
        
        # Queues (initialized in start())
        self.input_queue: Optional[asyncio.Queue] = None
        self.output_queue: Optional[asyncio.Queue] = None
        
        # Pending tool percepts for feedback loop
        self._pending_tool_percepts: List[Percept] = []
    
    def initialize_queues(self) -> None:
        """Initialize input and output queues in async context."""
        if self.input_queue is None:
            self.input_queue = asyncio.Queue(maxsize=self.config["max_queue_size"])
        
        if self.output_queue is None:
            self.output_queue = asyncio.Queue(maxsize=self.config["max_queue_size"])
    
    def inject_input(self, raw_input: Any, modality: str = "text") -> None:
        """
        Thread-safe method to add external input.
        
        Args:
            raw_input: Raw data to be encoded
            modality: Type of input ("text", "image", "audio", "introspection")
            
        Raises:
            RuntimeError: If called before queues initialized
        """
        if self.input_queue is None:
            logger.error("Cannot inject input: queues not initialized")
            raise RuntimeError("StateManager queues must be initialized before injecting input")
        
        try:
            self.input_queue.put_nowait((raw_input, modality))
            logger.debug(f"Injected {modality} input for encoding")
        except asyncio.QueueFull:
            logger.warning("Input queue full, dropping input")
    
    async def gather_percepts(self, perception_subsystem) -> List[Percept]:
        """
        Collect and encode queued inputs for this cycle.
        
        Args:
            perception_subsystem: PerceptionSubsystem instance for encoding
            
        Returns:
            List of Percept objects ready for attention processing
        """
        raw_inputs = []
        
        # Check if queue is initialized
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
            percept = await perception_subsystem.encode(raw_input, modality)
            percepts.append(percept)
        
        # Add any pending tool result percepts from previous cycle
        if self._pending_tool_percepts:
            logger.info(f"🔄 Adding {len(self._pending_tool_percepts)} tool result percepts to cycle")
            percepts.extend(self._pending_tool_percepts)
            self._pending_tool_percepts = []
        
        return percepts
    
    def add_pending_tool_percept(self, percept: Percept) -> None:
        """
        Add a tool result percept for the next cycle (feedback loop).
        
        Args:
            percept: Tool result percept to add
        """
        self._pending_tool_percepts.append(percept)
    
    async def queue_output(self, output: Dict[str, Any]) -> None:
        """
        Queue output for external retrieval.
        
        Args:
            output: Output dict with type, text, emotion, timestamp
        """
        if self.output_queue is not None:
            try:
                self.output_queue.put_nowait(output)
            except asyncio.QueueFull:
                logger.warning("Output queue full, dropping output")
        else:
            logger.warning("Output queue not initialized, cannot queue output")
    
    async def get_response(self, timeout: float = 5.0) -> Optional[Dict]:
        """
        Get response from the output queue (blocking with timeout).
        
        Args:
            timeout: Maximum time to wait for output (seconds)
            
        Returns:
            Dict with keys: type, text, emotion, timestamp
            None if timeout reached without output
            
        Raises:
            RuntimeError: If output queue not initialized
        """
        if self.output_queue is None:
            raise RuntimeError("Output queue not initialized")
        
        try:
            output = await asyncio.wait_for(
                self.output_queue.get(),
                timeout=timeout
            )
            return output
        except asyncio.TimeoutError:
            return None
    
    def query_state(self) -> WorkspaceSnapshot:
        """
        Thread-safe method to read current state.
        
        Returns:
            WorkspaceSnapshot: Immutable snapshot of current state
        """
        return self.workspace.broadcast()
