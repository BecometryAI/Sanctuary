"""
Test Suite for GPU Monitor

Tests GPU memory monitoring, threshold alerts, and automatic model unloading.
"""
import os
import pytest

if os.environ.get("CI"):
    pytest.skip("Requires GPU/torch — skipping in CI", allow_module_level=True)

import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from mind.gpu_monitor import (
    GPUMonitor,
    GPUMemoryInfo,
    MemoryThreshold,
    ModelMemoryTracking,
    get_global_monitor,
    initialize_gpu_monitoring
)
from mind.exceptions import GPUMemoryError


class TestGPUMemoryInfo:
    """Test GPUMemoryInfo dataclass."""
    
    def test_memory_info_creation(self):
        """Test creating memory info."""
        info = GPUMemoryInfo(
            device_id=0,
            total_mb=8000.0,
            allocated_mb=6400.0,
            reserved_mb=6500.0,
            free_mb=1600.0,
            utilization_percent=80.0
        )
        
        assert info.device_id == 0
        assert info.total_mb == 8000.0
        assert info.utilization_percent == 80.0
        assert info.threshold_level == MemoryThreshold.WARNING
    
    def test_memory_threshold_normal(self):
        """Test normal threshold detection."""
        info = GPUMemoryInfo(
            device_id=0,
            total_mb=8000.0,
            allocated_mb=4000.0,
            reserved_mb=4100.0,
            free_mb=4000.0,
            utilization_percent=50.0
        )
        
        assert info.threshold_level == MemoryThreshold.NORMAL
    
    def test_memory_threshold_warning(self):
        """Test warning threshold detection."""
        info = GPUMemoryInfo(
            device_id=0,
            total_mb=8000.0,
            allocated_mb=6800.0,
            reserved_mb=7000.0,
            free_mb=1200.0,
            utilization_percent=85.0
        )
        
        assert info.threshold_level == MemoryThreshold.WARNING
    
    def test_memory_threshold_critical(self):
        """Test critical threshold detection."""
        info = GPUMemoryInfo(
            device_id=0,
            total_mb=8000.0,
            allocated_mb=7300.0,
            reserved_mb=7500.0,
            free_mb=700.0,
            utilization_percent=91.25
        )
        
        assert info.threshold_level == MemoryThreshold.CRITICAL


class TestGPUMonitor:
    """Test GPUMonitor class."""
    
    def test_monitor_initialization_no_cuda(self):
        """Test monitor initialization when CUDA not available."""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = GPUMonitor()
            
            assert not monitor.cuda_available
            assert monitor.device_count == 0
            assert monitor.warning_threshold == 80.0
            assert monitor.critical_threshold == 90.0
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_monitor_initialization_with_cuda(self):
        """Test monitor initialization with CUDA available."""
        monitor = GPUMonitor(warning_threshold=75.0, critical_threshold=85.0)
        
        assert monitor.cuda_available
        assert monitor.device_count > 0
        assert monitor.warning_threshold == 75.0
        assert monitor.critical_threshold == 85.0
    
    def test_get_memory_info_no_cuda(self):
        """Test getting memory info when CUDA not available."""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = GPUMonitor()
            
            with pytest.raises(GPUMemoryError) as exc_info:
                monitor.get_memory_info(0)
            
            assert "CUDA not available" in str(exc_info.value)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_info_with_cuda(self):
        """Test getting memory info with CUDA."""
        monitor = GPUMonitor()
        info = monitor.get_memory_info(0)
        
        assert isinstance(info, GPUMemoryInfo)
        assert info.device_id == 0
        assert info.total_mb > 0
        assert info.allocated_mb >= 0
        assert info.free_mb >= 0
        assert 0 <= info.utilization_percent <= 100
    
    def test_can_load_model_no_cuda(self):
        """Test can_load_model when CUDA not available."""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = GPUMonitor()
            
            # Should return True for CPU mode
            assert monitor.can_load_model(1000.0)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_can_load_model_with_cuda(self):
        """Test can_load_model with CUDA."""
        monitor = GPUMonitor()
        
        # Test with small size (should succeed)
        assert monitor.can_load_model(100.0)
        
        # Test with huge size (should fail)
        info = monitor.get_memory_info(0)
        assert not monitor.can_load_model(info.total_mb * 2)
    
    def test_register_model(self):
        """Test model registration."""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = GPUMonitor()
            
            model = Mock()
            monitor.register_model("test-model", model, device_id=0, estimated_size_mb=1000.0)
            
            assert "test-model" in monitor.models
            tracking = monitor.models["test-model"]
            assert tracking.model_name == "test-model"
            assert tracking.allocated_mb == 1000.0
            assert tracking.device_id == 0
    
    def test_unregister_model(self):
        """Test model unregistration."""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = GPUMonitor()
            
            model = Mock()
            monitor.register_model("test-model", model, device_id=0, estimated_size_mb=1000.0)
            
            assert "test-model" in monitor.models
            
            monitor.unregister_model("test-model")
            
            assert "test-model" not in monitor.models
            assert "test-model" not in monitor.model_objects
    
    def test_get_least_used_model(self):
        """Test getting least used model."""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = GPUMonitor()
            
            # Register multiple models
            model1 = Mock()
            model2 = Mock()
            monitor.register_model("model-1", model1, device_id=0, estimated_size_mb=1000.0)
            monitor.register_model("model-2", model2, device_id=0, estimated_size_mb=1500.0)
            
            # Should return first model (oldest)
            least_used = monitor.get_least_used_model()
            assert least_used == "model-1"
    
    def test_check_and_manage_memory_no_cuda(self):
        """Test memory check when CUDA not available."""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = GPUMonitor()
            
            # Should return False (no action taken)
            result = monitor.check_and_manage_memory()
            assert result is False
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_auto_unload_on_critical(self):
        """Test automatic model unloading on critical threshold."""
        monitor = GPUMonitor(auto_unload=True, critical_threshold=0.0)  # Force critical
        
        # Register a model
        model = Mock()
        monitor.register_model("test-model", model, device_id=0, estimated_size_mb=100.0)
        
        # Mock get_memory_info to return critical
        with patch.object(monitor, 'get_memory_info') as mock_get_info:
            mock_get_info.return_value = GPUMemoryInfo(
                device_id=0,
                total_mb=8000.0,
                allocated_mb=7300.0,
                reserved_mb=7500.0,
                free_mb=700.0,
                utilization_percent=91.25
            )
            
            result = monitor.check_and_manage_memory()
            
            # Should have unloaded the model
            assert result is True
            assert "test-model" not in monitor.models
    
    def test_callback_on_warning(self):
        """Test warning callback invocation."""
        callback_called = []
        
        def warning_callback(info):
            callback_called.append(info)
        
        monitor = GPUMonitor(on_warning=warning_callback)
        
        # Mock get_memory_info to return warning
        with patch.object(monitor, 'get_memory_info') as mock_get_info:
            mock_get_info.return_value = GPUMemoryInfo(
                device_id=0,
                total_mb=8000.0,
                allocated_mb=6800.0,
                reserved_mb=7000.0,
                free_mb=1200.0,
                utilization_percent=85.0
            )
            
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.device_count', return_value=1):
                    monitor.cuda_available = True
                    monitor.device_count = 1
                    monitor.check_and_manage_memory()
            
            assert len(callback_called) == 1
            assert callback_called[0].threshold_level == MemoryThreshold.WARNING
    
    def test_get_status_summary_no_cuda(self):
        """Test status summary when CUDA not available."""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = GPUMonitor()
            status = monitor.get_status_summary()
            
            assert status["cuda_available"] is False
            assert "message" in status
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_status_summary_with_cuda(self):
        """Test status summary with CUDA."""
        monitor = GPUMonitor()
        
        # Register a model
        model = Mock()
        monitor.register_model("test-model", model, device_id=0, estimated_size_mb=1000.0)
        
        status = monitor.get_status_summary()
        
        assert status["cuda_available"] is True
        assert status["device_count"] > 0
        assert len(status["devices"]) > 0
        assert len(status["loaded_models"]) == 1
        assert status["loaded_models"][0]["name"] == "test-model"


class TestGlobalMonitor:
    """Test global monitor instance."""
    
    def test_get_global_monitor(self):
        """Test getting global monitor instance."""
        monitor1 = get_global_monitor()
        monitor2 = get_global_monitor()
        
        # Should return same instance
        assert monitor1 is monitor2
    
    def test_initialize_gpu_monitoring(self):
        """Test initializing GPU monitoring."""
        monitor = initialize_gpu_monitoring(
            auto_start=False,
            warning_threshold=75.0
        )
        
        assert isinstance(monitor, GPUMonitor)
        assert monitor.warning_threshold == 75.0
