"""
Input Queue: Non-blocking queue for human and external input.

This module implements the InputQueue class, which provides a non-blocking
interface for receiving input from external sources (human, API, etc.).
The cognitive loop checks this queue but doesn't wait for it - cognition
runs continuously regardless of whether input is available.

Key Features:
- Non-blocking queue operations
- Input events with metadata (timestamp, source)
- Support for multiple input sources
- Thread-safe operations

Author: Sanctuary Emergence Team
Phase: Communication Agency (Task #1)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import List, Optional, Any, Union, Dict
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class InputSource(str, Enum):
    """Sources of input to the cognitive system."""
    HUMAN = "human"
    API = "api"
    INTERNAL = "internal"
    SYSTEM = "system"


@dataclass
class InputEvent:
    """
    Represents an input event from an external source.
    
    Attributes:
        text: The input content (text string or structured data)
        modality: Type of input ("text", "image", "audio")
        source: Where the input came from
        timestamp: When the input was received
        metadata: Additional context about the input
    """
    text: Union[str, Dict[str, Any]]
    modality: str = "text"
    source: str = "human"
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    def __post_init__(self):
        """Initialize default values."""
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class InputQueue:
    """
    Non-blocking queue for human and external input.
    
    The cognitive loop checks this queue periodically but doesn't wait for it.
    This decouples cognition from I/O - the system can run cognitive cycles
    even when there is no external input.
    
    Key Design Principles:
    - Non-blocking: get_pending_inputs() returns immediately
    - Thread-safe: Can be called from different async tasks
    - Source tracking: Records where each input came from
    - Timestamped: All inputs have receive timestamps
    
    Attributes:
        _queue: Underlying asyncio.Queue for thread-safe operations
        max_size: Maximum number of queued inputs
        stats: Statistics about input processing
    """
    
    # Valid modality values
    VALID_MODALITIES = {"text", "image", "audio", "introspection"}
    
    def __init__(self, max_size: int = 100):
        """
        Initialize input queue.
        
        Args:
            max_size: Maximum queue size (default: 100)
            
        Raises:
            ValueError: If max_size <= 0
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
            
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.max_size = max_size
        
        # Statistics
        self.stats = {
            "total_received": 0,
            "dropped": 0,
            "by_source": {}
        }

        logger.info(f"✅ InputQueue initialized (max_size: {max_size})")

    @property
    def total_inputs_received(self) -> int:
        """Total number of inputs received (for backward compatibility)."""
        return self.stats["total_received"]
    
    async def add_input(
        self, 
        text: Union[str, Dict[str, Any]], 
        modality: str = "text",
        source: str = "human",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add input to queue (called by external interface).
        
        Args:
            text: The input content (text string or structured data)
            modality: Type of input ("text", "image", "audio", "introspection")
            source: Where the input came from
            metadata: Additional context about the input
            
        Returns:
            True if input was queued successfully, False if queue was full
            
        Raises:
            ValueError: If modality is not valid
        """
        if modality not in self.VALID_MODALITIES:
            logger.warning(f"Invalid modality '{modality}', using 'text'. "
                         f"Valid: {self.VALID_MODALITIES}")
            modality = "text"
        
        event = InputEvent(
            text=text,
            modality=modality,
            source=source,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        try:
            self._queue.put_nowait(event)
            self.stats["total_received"] += 1
            
            # Track by source
            if source not in self.stats["by_source"]:
                self.stats["by_source"][source] = 0
            self.stats["by_source"][source] += 1
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"📥 Queued input from {source}: {str(text)[:50]}...")
            return True
            
        except asyncio.QueueFull:
            self.stats["dropped"] += 1
            logger.warning(f"Input queue full ({self.max_size}), dropping input from {source}")
            return False
    
    def get_pending_inputs(self) -> List[InputEvent]:
        """
        Get all pending inputs without blocking.
        
        This is called by the cognitive loop to check for available input.
        If the queue is empty, it returns immediately with an empty list.
        This ensures the cognitive cycle never blocks waiting for input.
        
        Returns:
            List of InputEvent objects (may be empty)
        """
        inputs = []
        
        # Drain queue non-blocking
        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                inputs.append(event)
            except asyncio.QueueEmpty:
                break
        
        if inputs and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"📥 Retrieved {len(inputs)} pending inputs")
        
        return inputs
    
    def is_empty(self) -> bool:
        """
        Check if queue is empty.
        
        Returns:
            True if no inputs are queued
        """
        return self._queue.empty()
    
    def size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            Number of inputs currently queued
        """
        return self._queue.qsize()
    
    def get_stats(self) -> dict:
        """
        Get input queue statistics.
        
        Returns:
            Dict with statistics:
                - total_received: Total inputs ever received
                - dropped: Number of inputs dropped due to full queue
                - by_source: Breakdown of inputs by source
                - current_size: Current queue size
                - max_size: Maximum queue size
        """
        return {
            **self.stats,
            "current_size": self.size(),
            "max_size": self.max_size
        }
    
    def clear(self) -> int:
        """
        Clear all pending inputs from queue.
        
        Returns:
            Number of inputs that were cleared
        """
        cleared = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        
        if cleared > 0:
            logger.info(f"🗑️ Cleared {cleared} pending inputs from queue")
        
        return cleared


__all__ = [
    'InputQueue',
    'InputEvent',
    'InputSource'
]
