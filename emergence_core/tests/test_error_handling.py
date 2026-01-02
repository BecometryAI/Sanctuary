"""
Test Suite for Error Handling and Retry Logic

Tests exception hierarchy, retry decorators, and error handling mechanisms.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
import time

from emergence_core.lyra.exceptions import (
    LyraBaseException,
    ModelLoadError,
    MemoryError,
    ConsciousnessError,
    RouterError,
    GPUMemoryError,
    ValidationError,
    RateLimitError,
    ConcurrencyError
)
from emergence_core.lyra.utils.retry import (
    retry_with_backoff,
    retry_on_exception,
    RetryContext
)


class TestExceptions:
    """Test exception hierarchy."""
    
    def test_lyra_base_exception(self):
        """Test base exception creation."""
        exc = LyraBaseException(
            "Test error",
            context={"key": "value"},
            recoverable=True
        )
        
        assert str(exc) == "Test error (context: key=value)"
        assert exc.context == {"key": "value"}
        assert exc.recoverable is True
        
        exc_dict = exc.to_dict()
        assert exc_dict["type"] == "LyraBaseException"
        assert exc_dict["message"] == "Test error"
        assert exc_dict["recoverable"] is True
    
    def test_model_load_error(self):
        """Test ModelLoadError."""
        exc = ModelLoadError(
            "Failed to load model",
            model_name="gpt-neo"
        )
        
        assert "Failed to load model" in str(exc)
        assert exc.context["model_name"] == "gpt-neo"
        assert exc.recoverable is True
    
    def test_memory_error(self):
        """Test MemoryError."""
        exc = MemoryError(
            "ChromaDB operation failed",
            operation="add_documents"
        )
        
        assert "ChromaDB operation failed" in str(exc)
        assert exc.context["operation"] == "add_documents"
    
    def test_consciousness_error(self):
        """Test ConsciousnessError."""
        exc = ConsciousnessError(
            "Emotion update failed",
            subsystem="emotion_simulator"
        )
        
        assert "Emotion update failed" in str(exc)
        assert exc.context["subsystem"] == "emotion_simulator"
    
    def test_router_error(self):
        """Test RouterError."""
        exc = RouterError(
            "Specialist selection failed",
            specialist="code_specialist"
        )
        
        assert "Specialist selection failed" in str(exc)
        assert exc.context["specialist"] == "code_specialist"
    
    def test_gpu_memory_error(self):
        """Test GPUMemoryError."""
        exc = GPUMemoryError(
            "GPU memory exhausted",
            memory_used=7500,
            memory_total=8000
        )
        
        assert "GPU memory exhausted" in str(exc)
        assert exc.context["memory_used_mb"] == 7500
        assert exc.context["memory_total_mb"] == 8000
    
    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError(
            "Invalid field value",
            field="embedding",
            value="not_a_list"
        )
        
        assert "Invalid field value" in str(exc)
        assert exc.context["field"] == "embedding"
        assert exc.recoverable is False
    
    def test_rate_limit_error(self):
        """Test RateLimitError."""
        exc = RateLimitError(
            "Rate limit exceeded",
            service="wolfram",
            retry_after=60.0
        )
        
        assert "Rate limit exceeded" in str(exc)
        assert exc.context["service"] == "wolfram"
        assert exc.context["retry_after_seconds"] == 60.0
    
    def test_concurrency_error(self):
        """Test ConcurrencyError."""
        exc = ConcurrencyError(
            "Lock timeout",
            resource="memory_lock"
        )
        
        assert "Lock timeout" in str(exc)
        assert exc.context["resource"] == "memory_lock"


class TestRetryDecorator:
    """Test retry decorator functionality."""
    
    def test_retry_success_first_attempt(self):
        """Test successful execution on first attempt."""
        call_count = []
        
        @retry_with_backoff(max_retries=3)
        def successful_func():
            call_count.append(1)
            return "success"
        
        result = successful_func()
        
        assert result == "success"
        assert len(call_count) == 1
    
    def test_retry_success_after_failures(self):
        """Test successful execution after retries."""
        call_count = []
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def flaky_func():
            call_count.append(1)
            if len(call_count) < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = flaky_func()
        
        assert result == "success"
        assert len(call_count) == 3
    
    def test_retry_all_attempts_fail(self):
        """Test when all retry attempts fail."""
        call_count = []
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def failing_func():
            call_count.append(1)
            raise ValueError("Persistent failure")
        
        with pytest.raises(ValueError) as exc_info:
            failing_func()
        
        assert "Persistent failure" in str(exc_info.value)
        assert len(call_count) == 4  # Initial + 3 retries
    
    def test_retry_specific_exceptions(self):
        """Test retrying only specific exceptions."""
        call_count = []
        
        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            exceptions=(ConnectionError, TimeoutError)
        )
        def selective_func():
            call_count.append(1)
            if len(call_count) == 1:
                raise ConnectionError("Retry this")
            elif len(call_count) == 2:
                raise ValueError("Don't retry this")
            return "success"
        
        with pytest.raises(ValueError):
            selective_func()
        
        # Should only retry once (ConnectionError), then fail on ValueError
        assert len(call_count) == 2
    
    @pytest.mark.asyncio
    async def test_async_retry_success(self):
        """Test async retry success."""
        call_count = []
        
        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def async_flaky_func():
            call_count.append(1)
            if len(call_count) < 2:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await async_flaky_func()
        
        assert result == "success"
        assert len(call_count) == 2
    
    @pytest.mark.asyncio
    async def test_async_retry_failure(self):
        """Test async retry failure."""
        call_count = []
        
        @retry_with_backoff(max_retries=2, base_delay=0.01)
        async def async_failing_func():
            call_count.append(1)
            raise ValueError("Persistent failure")
        
        with pytest.raises(ValueError):
            await async_failing_func()
        
        assert len(call_count) == 3  # Initial + 2 retries
    
    def test_retry_callback(self):
        """Test on_retry callback."""
        callback_calls = []
        
        def retry_callback(exc, attempt):
            callback_calls.append((exc, attempt))
        
        @retry_with_backoff(
            max_retries=3,
            base_delay=0.01,
            on_retry=retry_callback
        )
        def flaky_func():
            if len(callback_calls) < 2:
                raise ValueError("Fail")
            return "success"
        
        result = flaky_func()
        
        assert result == "success"
        assert len(callback_calls) == 2
        assert all(attempt > 0 for _, attempt in callback_calls)
    
    def test_exponential_backoff(self):
        """Test exponential backoff timing."""
        call_times = []
        
        @retry_with_backoff(
            max_retries=3,
            base_delay=0.1,
            exponential_base=2.0
        )
        def timing_func():
            call_times.append(time.time())
            if len(call_times) < 4:
                raise ValueError("Retry")
            return "success"
        
        result = timing_func()
        
        assert result == "success"
        assert len(call_times) == 4
        
        # Check delays are approximately exponential
        delays = [call_times[i+1] - call_times[i] for i in range(len(call_times)-1)]
        
        # First delay should be ~0.1s
        assert 0.05 < delays[0] < 0.15
        
        # Second delay should be ~0.2s (2^1 * 0.1)
        assert 0.15 < delays[1] < 0.25
        
        # Third delay should be ~0.4s (2^2 * 0.1)
        assert 0.35 < delays[2] < 0.5


class TestRetryContext:
    """Test RetryContext context manager."""
    
    @pytest.mark.asyncio
    async def test_retry_context_success(self):
        """Test RetryContext successful execution."""
        call_count = []
        
        async def flaky_operation():
            call_count.append(1)
            if len(call_count) < 2:
                raise ValueError("Fail")
            return "success"
        
        async with RetryContext(max_retries=3, base_delay=0.01) as retry:
            result = await retry.execute(flaky_operation)
        
        assert result == "success"
        assert len(call_count) == 2
    
    @pytest.mark.asyncio
    async def test_retry_context_failure(self):
        """Test RetryContext when all retries fail."""
        async def failing_operation():
            raise ValueError("Always fail")
        
        with pytest.raises(ValueError):
            async with RetryContext(max_retries=2, base_delay=0.01) as retry:
                await retry.execute(failing_operation)


class TestRetryOnException:
    """Test retry_on_exception convenience function."""
    
    def test_retry_on_specific_exception(self):
        """Test retrying on specific exception type."""
        call_count = []
        
        @retry_on_exception(ConnectionError, max_retries=3, base_delay=0.01)
        def network_func():
            call_count.append(1)
            if len(call_count) < 2:
                raise ConnectionError("Network error")
            return "success"
        
        result = network_func()
        
        assert result == "success"
        assert len(call_count) == 2
