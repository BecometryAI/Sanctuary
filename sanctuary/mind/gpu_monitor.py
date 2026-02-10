"""
GPU Memory Management and Monitoring System

Implements automatic model unloading when GPU memory exceeds thresholds,
prevents OOM crashes, and provides graceful degradation mechanisms.

Author: Sanctuary Emergence Team
Date: January 2, 2026
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import torch

from .exceptions import GPUMemoryError, ModelLoadError
from .logging_config import get_logger, OperationContext

logger = get_logger(__name__)


class MemoryThreshold(Enum):
    """Memory usage threshold levels."""
    NORMAL = "normal"          # < 80%
    WARNING = "warning"        # 80-90%
    CRITICAL = "critical"      # > 90%


@dataclass
class GPUMemoryInfo:
    """GPU memory information snapshot."""
    device_id: int
    total_mb: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    utilization_percent: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def threshold_level(self) -> MemoryThreshold:
        """Determine threshold level based on utilization."""
        if self.utilization_percent >= 90:
            return MemoryThreshold.CRITICAL
        elif self.utilization_percent >= 80:
            return MemoryThreshold.WARNING
        else:
            return MemoryThreshold.NORMAL


@dataclass
class ModelMemoryTracking:
    """Track memory usage for a specific model."""
    model_name: str
    device_id: int
    allocated_mb: float
    load_time: datetime = field(default_factory=datetime.utcnow)
    last_used: datetime = field(default_factory=datetime.utcnow)
    use_count: int = 0


class GPUMonitor:
    """
    GPU Memory Monitor with automatic model unloading.
    
    Features:
    - Real-time memory tracking per GPU
    - Threshold-based alerts (warning at 80%, critical at 90%)
    - Automatic model unloading on critical threshold
    - Per-model memory tracking
    - Graceful degradation support
    
    Example:
        monitor = GPUMonitor()
        
        # Check before loading model
        if monitor.can_load_model(estimated_size_mb=2000):
            model = load_model()
            monitor.register_model("gpt-neo", model, device_id=0)
        
        # Automatic unloading if threshold exceeded
        monitor.check_and_manage_memory()
    """
    
    def __init__(
        self,
        warning_threshold: float = 80.0,
        critical_threshold: float = 90.0,
        check_interval: float = 10.0,
        auto_unload: bool = True,
        on_warning: Optional[Callable[[GPUMemoryInfo], None]] = None,
        on_critical: Optional[Callable[[GPUMemoryInfo], None]] = None
    ):
        """
        Initialize GPU monitor.
        
        Args:
            warning_threshold: Memory utilization % for warning (default: 80%)
            critical_threshold: Memory utilization % for critical alert (default: 90%)
            check_interval: Seconds between automatic checks (default: 10.0)
            auto_unload: Automatically unload models on critical threshold
            on_warning: Callback function called on warning threshold
            on_critical: Callback function called on critical threshold
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        self.auto_unload = auto_unload
        self.on_warning = on_warning
        self.on_critical = on_critical
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        
        # Model tracking
        self.models: Dict[str, ModelMemoryTracking] = {}
        self.model_objects: Dict[str, Any] = {}  # Store model references
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Memory history for trends
        self.memory_history: List[GPUMemoryInfo] = []
        self.max_history_size = 100
        
        if self.cuda_available:
            logger.info(
                f"GPU Monitor initialized: {self.device_count} GPU(s) available. "
                f"Thresholds: warning={warning_threshold}%, critical={critical_threshold}%"
            )
        else:
            logger.warning("GPU Monitor initialized but CUDA not available - CPU only mode")
    
    def get_memory_info(self, device_id: int = 0) -> GPUMemoryInfo:
        """
        Get current GPU memory information.
        
        Args:
            device_id: GPU device ID
        
        Returns:
            GPUMemoryInfo object with current memory state
        
        Raises:
            GPUMemoryError: If CUDA not available or device invalid
        """
        if not self.cuda_available:
            raise GPUMemoryError("CUDA not available", context={"device_id": device_id})
        
        if device_id >= self.device_count:
            raise GPUMemoryError(
                f"Invalid device ID {device_id}, only {self.device_count} devices available",
                context={"device_id": device_id, "device_count": self.device_count}
            )
        
        # Get memory stats
        total = torch.cuda.get_device_properties(device_id).total_memory
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        
        # Convert to MB
        total_mb = total / (1024 ** 2)
        allocated_mb = allocated / (1024 ** 2)
        reserved_mb = reserved / (1024 ** 2)
        free_mb = total_mb - allocated_mb
        
        # Calculate utilization
        utilization = (allocated_mb / total_mb) * 100 if total_mb > 0 else 0
        
        return GPUMemoryInfo(
            device_id=device_id,
            total_mb=total_mb,
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
            free_mb=free_mb,
            utilization_percent=utilization
        )
    
    def can_load_model(
        self,
        estimated_size_mb: float,
        device_id: int = 0,
        safety_margin_mb: float = 500.0
    ) -> bool:
        """
        Check if there's enough memory to load a model.
        
        Args:
            estimated_size_mb: Estimated model size in MB
            device_id: Target GPU device ID
            safety_margin_mb: Additional safety margin in MB
        
        Returns:
            True if model can be loaded safely
        """
        if not self.cuda_available:
            logger.info("CUDA not available - assuming CPU mode can load model")
            return True
        
        try:
            info = self.get_memory_info(device_id)
            required_mb = estimated_size_mb + safety_margin_mb
            
            can_load = info.free_mb >= required_mb
            
            if not can_load:
                logger.warning(
                    f"Insufficient GPU memory: need {required_mb:.0f}MB, "
                    f"have {info.free_mb:.0f}MB free"
                )
            
            return can_load
        
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return False
    
    def register_model(
        self,
        model_name: str,
        model: Any,
        device_id: int = 0,
        estimated_size_mb: Optional[float] = None
    ):
        """
        Register a loaded model for tracking.
        
        Args:
            model_name: Unique name for the model
            model: Model object
            device_id: GPU device ID where model is loaded
            estimated_size_mb: Optional estimated size, calculated if not provided
        
        Note:
            If estimated_size_mb is not provided, the current GPU memory allocation
            will be used as an approximation. For accurate tracking, measure memory
            before and after model loading and pass the difference as estimated_size_mb.
        """
        with self._lock:
            # Calculate model size if not provided
            # Note: This is an approximation - total allocated memory on device
            # For precise tracking, measure memory before/after model load
            if estimated_size_mb is None and self.cuda_available:
                try:
                    # Use current allocated memory as approximation
                    estimated_size_mb = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
                except Exception as e:
                    logger.warning(f"Could not estimate model size: {e}")
                    estimated_size_mb = 0.0
            elif estimated_size_mb is None:
                estimated_size_mb = 0.0
            
            tracking = ModelMemoryTracking(
                model_name=model_name,
                device_id=device_id,
                allocated_mb=estimated_size_mb
            )
            
            self.models[model_name] = tracking
            self.model_objects[model_name] = model
            
            logger.info(
                f"Registered model '{model_name}' on device {device_id}, "
                f"estimated size: {estimated_size_mb:.0f}MB"
            )
    
    def unregister_model(self, model_name: str):
        """
        Unregister and unload a model.
        
        Args:
            model_name: Name of model to unregister
        """
        with self._lock:
            if model_name in self.models:
                tracking = self.models[model_name]
                
                # Delete model object
                if model_name in self.model_objects:
                    del self.model_objects[model_name]
                
                # Clear GPU cache
                if self.cuda_available:
                    torch.cuda.empty_cache()
                
                del self.models[model_name]
                
                logger.info(
                    f"Unregistered model '{model_name}' from device {tracking.device_id}, "
                    f"freed ~{tracking.allocated_mb:.0f}MB"
                )
            else:
                logger.warning(f"Model '{model_name}' not found in registry")
    
    def get_least_used_model(self, device_id: Optional[int] = None) -> Optional[str]:
        """
        Get the least recently used model for unloading.
        
        Args:
            device_id: Optional filter by device ID
        
        Returns:
            Model name or None if no models loaded
        """
        with self._lock:
            candidates = [
                (name, tracking)
                for name, tracking in self.models.items()
                if device_id is None or tracking.device_id == device_id
            ]
            
            if not candidates:
                return None
            
            # Sort by last used time (oldest first)
            candidates.sort(key=lambda x: x[1].last_used)
            return candidates[0][0]
    
    def check_and_manage_memory(self, device_id: int = 0) -> bool:
        """
        Check memory and automatically manage if needed.
        
        Args:
            device_id: GPU device ID to check
        
        Returns:
            True if action was taken, False otherwise
        """
        if not self.cuda_available:
            return False
        
        try:
            with OperationContext(operation="memory_check", device_id=device_id):
                info = self.get_memory_info(device_id)
                
                # Add to history
                self.memory_history.append(info)
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history.pop(0)
                
                # Check thresholds
                if info.threshold_level == MemoryThreshold.CRITICAL:
                    logger.error(
                        f"CRITICAL GPU memory: {info.utilization_percent:.1f}% "
                        f"({info.allocated_mb:.0f}MB / {info.total_mb:.0f}MB)"
                    )
                    
                    # Call critical callback
                    if self.on_critical:
                        try:
                            self.on_critical(info)
                        except Exception as e:
                            logger.error(f"Error in critical callback: {e}")
                    
                    # Auto-unload if enabled
                    if self.auto_unload:
                        model_name = self.get_least_used_model(device_id)
                        if model_name:
                            logger.warning(f"Auto-unloading model '{model_name}' to free memory")
                            self.unregister_model(model_name)
                            return True
                        else:
                            logger.warning("No models available to unload")
                
                elif info.threshold_level == MemoryThreshold.WARNING:
                    logger.warning(
                        f"WARNING GPU memory: {info.utilization_percent:.1f}% "
                        f"({info.allocated_mb:.0f}MB / {info.total_mb:.0f}MB)"
                    )
                    
                    # Call warning callback
                    if self.on_warning:
                        try:
                            self.on_warning(info)
                        except Exception as e:
                            logger.error(f"Error in warning callback: {e}")
                
                return False
        
        except Exception as e:
            logger.error(f"Error checking GPU memory: {e}")
            return False
    
    def start_monitoring(self):
        """Start background monitoring thread."""
        if self._monitoring:
            logger.warning("Monitoring already started")
            return
        
        if not self.cuda_available:
            logger.info("GPU monitoring not started - CUDA not available")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="gpu-monitor"
        )
        self._monitor_thread.start()
        logger.info(f"GPU monitoring started (interval: {self.check_interval}s)")
    
    def stop_monitoring(self):
        """Stop background monitoring thread."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                for device_id in range(self.device_count):
                    self.check_and_manage_memory(device_id)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(self.check_interval)
    
    def get_status_summary(self) -> Dict[str, Any]:
        """
        Get summary of GPU status and loaded models.
        
        Returns:
            Dictionary with status information
        """
        if not self.cuda_available:
            return {
                "cuda_available": False,
                "message": "CUDA not available - CPU only mode"
            }
        
        devices = []
        for device_id in range(self.device_count):
            try:
                info = self.get_memory_info(device_id)
                devices.append({
                    "device_id": device_id,
                    "total_mb": info.total_mb,
                    "allocated_mb": info.allocated_mb,
                    "free_mb": info.free_mb,
                    "utilization_percent": info.utilization_percent,
                    "threshold_level": info.threshold_level.value
                })
            except Exception as e:
                logger.error(f"Error getting info for device {device_id}: {e}")
        
        models = [
            {
                "name": name,
                "device_id": tracking.device_id,
                "allocated_mb": tracking.allocated_mb,
                "use_count": tracking.use_count,
                "last_used": tracking.last_used.isoformat()
            }
            for name, tracking in self.models.items()
        ]
        
        return {
            "cuda_available": True,
            "device_count": self.device_count,
            "devices": devices,
            "loaded_models": models,
            "monitoring_active": self._monitoring
        }


# Global GPU monitor instance
_global_monitor: Optional[GPUMonitor] = None


def get_global_monitor() -> GPUMonitor:
    """
    Get or create global GPU monitor instance.
    
    Returns:
        Global GPUMonitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = GPUMonitor()
    return _global_monitor


def initialize_gpu_monitoring(auto_start: bool = True, **kwargs) -> GPUMonitor:
    """
    Initialize global GPU monitoring.
    
    Args:
        auto_start: Automatically start background monitoring
        **kwargs: Additional arguments for GPUMonitor
    
    Returns:
        Initialized GPUMonitor instance
    """
    global _global_monitor
    _global_monitor = GPUMonitor(**kwargs)
    
    if auto_start:
        _global_monitor.start_monitoring()
    
    return _global_monitor
