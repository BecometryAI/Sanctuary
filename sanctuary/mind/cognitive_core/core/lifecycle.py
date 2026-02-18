"""
Lifecycle management for the cognitive core.

Handles start/stop operations, checkpoint management, and graceful shutdown.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from pathlib import Path
from statistics import mean

if TYPE_CHECKING:
    from .subsystem_coordinator import SubsystemCoordinator
    from .state_manager import StateManager
    from .timing import TimingManager

logger = logging.getLogger(__name__)


class LifecycleManager:
    """
    Manages lifecycle operations for the cognitive core.
    
    Responsibilities:
    - Start/stop operations
    - Checkpoint management
    - Auto-save configuration
    - Graceful shutdown
    """
    
    def __init__(
        self,
        subsystems: 'SubsystemCoordinator',
        state: 'StateManager',
        timing: 'TimingManager',
        config: dict
    ):
        """
        Initialize lifecycle manager.
        
        Args:
            subsystems: SubsystemCoordinator instance
            state: StateManager instance
            timing: TimingManager instance
            config: Configuration dict
        """
        self.subsystems = subsystems
        self.state = state
        self.timing = timing
        self.config = config
    
    async def start(self, restore_latest: bool = False) -> None:
        """
        Start the cognitive core.
        
        Args:
            restore_latest: If True, restore from the most recent checkpoint before starting
        """
        logger.info("🧠 Starting CognitiveCore...")
        
        # Restore from checkpoint if requested
        if restore_latest and self.subsystems.checkpoint_manager:
            await self._restore_latest_checkpoint()
        
        # Initialize queues in async context
        self.state.initialize_queues()

        # Connect device registry to input queue for multimodal perception
        self._connect_device_registry()

        # Set running flag
        self.state.running = True

        # Enable auto-GC if configured
        self._enable_memory_gc()

        # Start auto-save if enabled
        await self._start_auto_save()

        logger.info("🧠 Cognitive core started")
    
    async def stop(self, timeout: float = 30.0) -> None:
        """
        Gracefully shut down the cognitive core.

        Args:
            timeout: Maximum seconds to wait for shutdown. If exceeded, forces
                     remaining operations to cancel and logs a warning.
        """
        logger.info("🧠 Stopping CognitiveCore...")
        self.state.running = False

        try:
            await asyncio.wait_for(self._shutdown_sequence(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                f"🧠 Shutdown timed out after {timeout}s — forcing remaining tasks to cancel"
            )
            # Force-cancel any surviving tasks as a last resort
            await self._cancel_tasks()

        logger.info("🧠 CognitiveCore shutdown complete.")

    async def _shutdown_sequence(self) -> None:
        """Execute the ordered shutdown sequence (called within a timeout)."""
        # Disable GC before shutdown
        self.subsystems.memory.memory_manager.disable_auto_gc()
        logger.info("🧹 Memory garbage collection disabled")

        # Stop auto-save if running
        await self._stop_auto_save()

        # Disconnect devices and stop hot-plug monitoring
        await self._shutdown_device_registry()

        # Stop continuous consciousness
        if hasattr(self.subsystems, 'continuous_consciousness'):
            await self.subsystems.continuous_consciousness.stop()

        # Cancel tasks if they exist
        await self._cancel_tasks()

        # Log final metrics
        self._log_final_metrics()

        # Close introspective journal
        self._close_journal()

        # Save final workspace state on shutdown
        await self._save_shutdown_checkpoint()
    
    async def _shutdown_device_registry(self) -> None:
        """Disconnect all devices and stop hot-plug monitoring."""
        registry = getattr(self.subsystems, 'device_registry', None)
        if registry is None:
            return

        try:
            await registry.stop_hot_plug_monitoring()
            await registry.disconnect_all_devices()
            logger.info("📷 Device registry shut down")
        except Exception as e:
            logger.error(f"📷 Error during device registry shutdown: {e}")

    async def _restore_latest_checkpoint(self) -> None:
        """Restore from the most recent checkpoint."""
        latest = self.subsystems.checkpoint_manager.get_latest_checkpoint()
        if latest:
            try:
                logger.info(f"💾 Restoring from checkpoint: {latest.name}")
                self.state.workspace = self.subsystems.checkpoint_manager.load_checkpoint(latest)
                logger.info("✅ Workspace restored from checkpoint")
            except Exception as e:
                logger.error(f"Failed to restore checkpoint: {e}")
    
    def _connect_device_registry(self) -> None:
        """Connect device registry to input queue for multimodal data routing."""
        if not hasattr(self.subsystems, 'device_registry') or self.subsystems.device_registry is None:
            return

        connected = self.subsystems.connect_device_registry_to_input(self.state.input_queue)
        if connected:
            logger.info("📷 Device registry connected to cognitive pipeline")
        else:
            logger.debug("📷 Device registry connection skipped (no callback set)")

    def _enable_memory_gc(self) -> None:
        """Enable memory garbage collection if configured."""
        gc_config = self.config.get("memory_gc", {})
        if gc_config.get("enabled", True):
            interval = gc_config.get("collection_interval", 3600.0)
            self.subsystems.memory.memory_manager.enable_auto_gc(interval)
            logger.info(f"🧹 Memory garbage collection enabled (interval: {interval}s)")
    
    async def _start_auto_save(self) -> None:
        """Start auto-save if enabled."""
        checkpoint_config = self.config.get("checkpointing", {})
        if checkpoint_config.get("auto_save", False) and self.subsystems.checkpoint_manager:
            interval = checkpoint_config.get("auto_save_interval", 300.0)
            self.subsystems.checkpoint_manager.auto_save_task = asyncio.create_task(
                self.subsystems.checkpoint_manager.auto_save(self.state.workspace, interval)
            )
            logger.info(f"💾 Auto-save enabled: interval={interval}s")
    
    async def _stop_auto_save(self) -> None:
        """Stop auto-save if running."""
        if self.subsystems.checkpoint_manager and self.subsystems.checkpoint_manager.auto_save_task:
            self.subsystems.checkpoint_manager.stop_auto_save()
            self.subsystems.checkpoint_manager.auto_save_task.cancel()
            try:
                await self.subsystems.checkpoint_manager.auto_save_task
            except asyncio.CancelledError:
                pass
    
    async def _cancel_tasks(self) -> None:
        """Cancel active and idle tasks."""
        if self.state.active_task and not self.state.active_task.done():
            self.state.active_task.cancel()
            try:
                await self.state.active_task
            except asyncio.CancelledError:
                pass
        
        if self.state.idle_task and not self.state.idle_task.done():
            self.state.idle_task.cancel()
            try:
                await self.state.idle_task
            except asyncio.CancelledError:
                pass
    
    def _log_final_metrics(self) -> None:
        """Log final performance metrics."""
        metrics = self.timing.metrics
        avg_cycle_time = mean(metrics['cycle_times']) if metrics['cycle_times'] else 0.0
        logger.info(
            f"📊 Final metrics: total_cycles={metrics['total_cycles']}, "
            f"avg_cycle_time={avg_cycle_time*1000:.1f}ms, "
            f"percepts_processed={metrics.get('percepts_processed', 0)}"
        )
    
    def _close_journal(self) -> None:
        """Close introspective journal to ensure all entries are saved."""
        try:
            if hasattr(self.subsystems, 'introspective_journal') and self.subsystems.introspective_journal:
                self.subsystems.introspective_journal.close()
                logger.info("💾 Introspective journal closed successfully")
        except Exception as e:
            logger.error(f"❌ Failed to close introspective journal: {e}")
    
    async def _save_shutdown_checkpoint(self) -> None:
        """Save final checkpoint on shutdown."""
        checkpoint_config = self.config.get("checkpointing", {})
        if checkpoint_config.get("checkpoint_on_shutdown", True) and self.subsystems.checkpoint_manager:
            try:
                self.subsystems.checkpoint_manager.save_checkpoint(
                    self.state.workspace,
                    metadata={"auto_save": False, "shutdown": True}
                )
                logger.info("💾 Final checkpoint saved on shutdown")
            except Exception as e:
                logger.error(f"Failed to save shutdown checkpoint: {e}")
    
    def save_state(self, label: str = None) -> Path:
        """
        Save current workspace state to checkpoint.
        
        Args:
            label: Optional user label for the checkpoint
            
        Returns:
            Path to the saved checkpoint file, or None if checkpointing disabled
        """
        if not self.subsystems.checkpoint_manager:
            logger.warning("Cannot save state: checkpointing disabled")
            return None
        
        try:
            metadata = {
                "auto_save": False,
                "manual": True,
            }
            if label:
                metadata["user_label"] = label
            
            path = self.subsystems.checkpoint_manager.save_checkpoint(self.state.workspace, metadata)
            logger.info(f"💾 State saved: {path.name}")
            return path
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return None
    
    def restore_state(self, checkpoint_path: Path) -> bool:
        """
        Restore workspace from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            True if restore succeeded, False otherwise
        """
        if not self.subsystems.checkpoint_manager:
            logger.warning("Cannot restore state: checkpointing disabled")
            return False
        
        if self.state.running:
            logger.warning("Cannot restore state while cognitive loop is running")
            return False
        
        try:
            self.state.workspace = self.subsystems.checkpoint_manager.load_checkpoint(checkpoint_path)
            logger.info(f"✅ State restored from {checkpoint_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            return False
    
    def enable_auto_checkpoint(self, interval: float = 300.0) -> bool:
        """
        Enable automatic periodic checkpointing.
        
        Args:
            interval: Time between checkpoints in seconds (default: 300 = 5 minutes)
            
        Returns:
            True if auto-checkpoint enabled, False otherwise
        """
        if not self.subsystems.checkpoint_manager:
            logger.warning("Cannot enable auto-checkpoint: checkpointing disabled")
            return False
        
        if not self.state.running:
            logger.warning("Cannot enable auto-checkpoint: cognitive loop not running")
            return False
        
        # Stop existing auto-save if running
        if self.subsystems.checkpoint_manager.auto_save_task:
            self.subsystems.checkpoint_manager.stop_auto_save()
            self.subsystems.checkpoint_manager.auto_save_task.cancel()
        
        # Start new auto-save task
        self.subsystems.checkpoint_manager.auto_save_task = asyncio.create_task(
            self.subsystems.checkpoint_manager.auto_save(self.state.workspace, interval)
        )
        logger.info(f"💾 Auto-checkpoint enabled: interval={interval}s")
        return True
    
    def disable_auto_checkpoint(self) -> bool:
        """
        Disable automatic periodic checkpointing.
        
        Returns:
            True if auto-checkpoint disabled, False if it wasn't running
        """
        if not self.subsystems.checkpoint_manager:
            return False
        
        if not self.subsystems.checkpoint_manager.auto_save_task:
            return False
        
        self.subsystems.checkpoint_manager.stop_auto_save()
        self.subsystems.checkpoint_manager.auto_save_task.cancel()
        logger.info("💾 Auto-checkpoint disabled")
        return True
