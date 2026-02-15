"""
Cognitive Core module.

This module provides a refactored, modular implementation of the cognitive core.
The original monolithic core.py has been split into focused, single-responsibility modules:

- subsystem_coordinator.py: Initialize and coordinate all subsystems
- state_manager.py: Workspace state, queues, and metrics
- lifecycle.py: Start/stop/checkpoint operations
- timing.py: Rate limiting and performance tracking
- cycle_executor.py: Execute the 9-step cognitive cycle
- cognitive_loop.py: Main ~10Hz recurrent loop orchestration

The CognitiveCore class remains the main public interface, but now acts as a thin
facade that delegates to these specialized modules.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from ..workspace import GlobalWorkspace, WorkspaceSnapshot

from .subsystem_coordinator import SubsystemCoordinator
from .state_manager import StateManager
from .timing import TimingManager
from .lifecycle import LifecycleManager
from .cycle_executor import CycleExecutor
from .cognitive_loop import CognitiveLoop
from .action_executor import ActionExecutor
from .subsystem_health import SubsystemSupervisor, SubsystemStatus, SubsystemHealthState

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "cycle_rate_hz": 10,
    "attention_budget": 100,
    "max_queue_size": 100,
    "log_interval_cycles": 100,
    "timing": {
        "warn_threshold_ms": 100,
        "critical_threshold_ms": 200,
        "track_slow_cycles": True,
    },
    "checkpointing": {
        "enabled": True,
        "auto_save": False,
        "auto_save_interval": 300.0,
        "checkpoint_dir": "data/checkpoints/",
        "max_checkpoints": 20,
        "compression": True,
        "checkpoint_on_shutdown": True,
    }
}


class CognitiveCore:
    """
    Main recurrent cognitive loop that runs continuously.

    The CognitiveCore is the heart of the cognitive architecture, implementing
    a continuous recurrent loop based on Global Workspace Theory and computational
    functionalism. It coordinates all subsystems and maintains the conscious state
    across time.

    This is now a thin facade that delegates to specialized modules:
    - SubsystemCoordinator: Initializes and manages all cognitive subsystems
    - StateManager: Manages workspace state, queues, and percepts
    - TimingManager: Handles timing, rate limiting, and performance metrics
    - LifecycleManager: Manages start/stop and checkpoint operations
    - CycleExecutor: Executes the 9-step cognitive cycle
    - CognitiveLoop: Orchestrates the main recurrent loop

    The public API remains unchanged for backward compatibility.
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
        
        # Initialize state manager
        self.state = StateManager(workspace, self.config)
        
        # Initialize subsystems
        self.subsystems = SubsystemCoordinator(self.state.workspace, self.config)
        
        # Initialize continuous consciousness (needs reference to self)
        self.subsystems.continuous_consciousness = self.subsystems.initialize_continuous_consciousness(self)
        
        # Initialize timing manager
        self.timing = TimingManager(self.config)

        # Initialize subsystem supervisor for fault isolation
        self.supervisor = SubsystemSupervisor(self.config.get("supervisor", {}))

        # Initialize action executor
        self.action_executor = ActionExecutor(self.subsystems, self.state)

        # Initialize cycle executor with supervisor
        self.cycle_executor = CycleExecutor(
            self.subsystems, self.state, self.action_executor,
            self.timing, self.supervisor
        )
        
        # Register subsystem reinitializers for automatic recovery
        self._register_reinitializers()

        # Initialize cognitive loop orchestrator
        self.loop = CognitiveLoop(self.subsystems, self.state, self.timing, self.cycle_executor)
        
        # Initialize lifecycle manager
        self.lifecycle = LifecycleManager(self.subsystems, self.state, self.timing, self.config)
        
        logger.info(f"🧠 CognitiveCore initialized: cycle_rate={self.config['cycle_rate_hz']}Hz, "
                   f"attention_budget={self.config['attention_budget']}")

    def _register_reinitializers(self) -> None:
        """Register subsystem reinitializers with the supervisor.

        Maps each cognitive cycle step name to a callable that
        reinitializes the corresponding subsystem.  When a subsystem
        enters recovery after being FAILED, the supervisor calls
        its reinitializer before attempting the step again.
        """
        mapping = {
            "perception": self.subsystems.reinitialize_perception,
            "attention": self.subsystems.reinitialize_attention,
            "affect": self.subsystems.reinitialize_affect,
            "action": self.subsystems.reinitialize_action,
            "meta_cognition": self.subsystems.reinitialize_meta_cognition,
            "communication_drives": self.subsystems.reinitialize_communication_drives,
            "autonomous_initiation": self.subsystems.reinitialize_autonomous_initiation,
            "bottleneck_detection": self.subsystems.reinitialize_bottleneck_detector,
            "temporal_context": self.subsystems.reinitialize_temporal_grounding,
            "memory_retrieval": self.subsystems.reinitialize_memory,
            "memory_consolidation": self.subsystems.reinitialize_memory,
            "iwmt_predict": self.subsystems.reinitialize_iwmt,
            "iwmt_update": self.subsystems.reinitialize_iwmt,
        }
        for name, callback in mapping.items():
            self.supervisor.register_reinitializer(name, callback)

        logger.info(f"Registered {len(mapping)} subsystem reinitializers")

    # ========== Lifecycle Management (delegate to LifecycleManager) ==========
    
    async def start(self, restore_latest: bool = False) -> None:
        """Start the main cognitive loop.

        Spawns the cognitive loop as a background task and returns immediately.
        Call stop() to gracefully shut down the loop.

        Args:
            restore_latest: If True, restore from the most recent checkpoint before starting
        """
        await self.lifecycle.start(restore_latest)
        # Spawn loop as background task (runs until stop() is called)
        self._loop_task = asyncio.create_task(self.loop.run())

    async def stop(self) -> None:
        """Gracefully shut down the cognitive loop."""
        await self.lifecycle.stop()
        # Wait for the loop task to complete
        if hasattr(self, '_loop_task') and self._loop_task:
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    def save_state(self, label: Optional[str] = None) -> Optional[Path]:
        """Save current workspace state to checkpoint."""
        return self.lifecycle.save_state(label)

    def restore_state(self, checkpoint_path: Path) -> bool:
        """Restore workspace from checkpoint."""
        return self.lifecycle.restore_state(checkpoint_path)

    def enable_auto_checkpoint(self, interval: float = 300.0) -> bool:
        """Enable automatic periodic checkpointing."""
        return self.lifecycle.enable_auto_checkpoint(interval)

    def disable_auto_checkpoint(self) -> bool:
        """Disable automatic periodic checkpointing."""
        return self.lifecycle.disable_auto_checkpoint()

    # ========== State Management (delegate to StateManager) ==========
    
    def inject_input(self, raw_input: Any, modality: str = "text") -> None:
        """Thread-safe method to add external input."""
        self.state.inject_input(raw_input, modality)

    def query_state(self) -> WorkspaceSnapshot:
        """Thread-safe method to read current state."""
        return self.state.query_state()

    async def get_response(self, timeout: float = 5.0) -> Optional[Dict]:
        """Get response from the output queue."""
        return await self.state.get_response(timeout)

    # ========== Language Processing (delegate to CognitiveLoop) ==========
    
    async def process_language_input(self, text: str, context: Optional[Dict] = None) -> None:
        """Process natural language input."""
        await self.loop.process_language_input(text, context)

    async def chat(self, message: str, timeout: float = 5.0) -> str:
        """Convenience method: Send message and get text response."""
        return await self.loop.chat(message, timeout)

    # ========== Metrics (delegate to TimingManager) ==========
    
    def get_metrics(self) -> Dict[str, Any]:
        """Returns performance metrics."""
        metrics = self.timing.get_metrics_summary()
        # Add state-specific metrics
        metrics['attention_selections'] = self.timing.metrics.get('attention_selections', 0)
        metrics['percepts_processed'] = self.timing.metrics.get('percepts_processed', 0)
        metrics['workspace_size'] = len(self.state.workspace.active_percepts)
        metrics['current_goals'] = len(self.state.workspace.current_goals)
        return metrics

    def get_performance_breakdown(self) -> Dict[str, Any]:
        """Get detailed performance breakdown by subsystem."""
        return self.timing.get_performance_breakdown()

    def get_health_report(self) -> Dict[str, Any]:
        """Get subsystem health report from the supervisor."""
        return self.supervisor.get_system_report()

    def get_subsystem_health(self, name: str) -> 'SubsystemHealthState':
        """Get health state for a specific subsystem."""
        return self.supervisor.get_health(name)

    def reset_subsystem(self, name: str) -> None:
        """Manually reset a failed subsystem to HEALTHY."""
        self.supervisor.reset(name)

    # ========== Direct subsystem access for backward compatibility ==========
    
    @property
    def workspace(self) -> GlobalWorkspace:
        """Access to workspace."""
        return self.state.workspace
    
    @workspace.setter
    def workspace(self, value: GlobalWorkspace) -> None:
        """Allow setting workspace (needed for checkpoint restore)."""
        self.state.workspace = value
    
    @property
    def running(self) -> bool:
        """Check if core is running."""
        return self.state.running
    
    @property
    def attention(self):
        """Access to attention subsystem."""
        return self.subsystems.attention
    
    @property
    def perception(self):
        """Access to perception subsystem."""
        return self.subsystems.perception
    
    @property
    def action(self):
        """Access to action subsystem."""
        return self.subsystems.action
    
    @property
    def affect(self):
        """Access to affect subsystem."""
        return self.subsystems.affect
    
    @property
    def meta_cognition(self):
        """Access to meta-cognition subsystem."""
        return self.subsystems.meta_cognition
    
    @property
    def memory(self):
        """Access to memory subsystem."""
        return self.subsystems.memory
    
    @property
    def autonomous(self):
        """Access to autonomous initiation controller."""
        return self.subsystems.autonomous
    
    @property
    def temporal_awareness(self):
        """Access to temporal awareness."""
        return self.subsystems.temporal_awareness
    
    @property
    def memory_review(self):
        """Access to autonomous memory review."""
        return self.subsystems.memory_review
    
    @property
    def existential_reflection(self):
        """Access to existential reflection."""
        return self.subsystems.existential_reflection
    
    @property
    def pattern_analysis(self):
        """Access to interaction pattern analysis."""
        return self.subsystems.pattern_analysis
    
    @property
    def continuous_consciousness(self):
        """Access to continuous consciousness controller."""
        return self.subsystems.continuous_consciousness
    
    @property
    def introspective_loop(self):
        """Access to introspective loop."""
        return self.subsystems.introspective_loop
    
    @property
    def introspective_journal(self):
        """Access to introspective journal."""
        return self.subsystems.introspective_journal
    
    @property
    def identity(self):
        """Access to identity loader."""
        return self.subsystems.identity
    
    @property
    def language_input(self):
        """Access to language input parser."""
        return self.subsystems.language_input
    
    @property
    def language_output(self):
        """Access to language output generator."""
        return self.subsystems.language_output
    
    @property
    def checkpoint_manager(self):
        """Access to checkpoint manager."""
        return self.subsystems.checkpoint_manager
    
    @property
    def input_queue(self):
        """Access to input queue."""
        return self.state.input_queue
    
    @property
    def output_queue(self):
        """Access to output queue."""
        return self.state.output_queue
    
    @property
    def metrics(self):
        """Access to timing metrics."""
        return self.timing.metrics


__all__ = ['CognitiveCore', 'SubsystemSupervisor', 'SubsystemStatus', 'SubsystemHealthState']
