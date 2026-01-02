"""
Test Suite for Rate Limiting and Concurrency

Tests rate limiters, locks, and concurrency primitives.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock

from emergence_core.lyra.utils.rate_limiter import (
    RateLimiter,
    ServiceRateLimiter,
    RateLimitConfig,
    get_global_limiter
)
from emergence_core.lyra.utils.locks import (
    TimeoutLock,
    AsyncRWLock,
    synchronized,
    async_synchronized,
    Semaphore,
    ResourcePool
)
from emergence_core.lyra.exceptions import RateLimitError, ConcurrencyError


class TestRateLimiter:
    """Test RateLimiter class."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(calls_per_minute=60)
        
        assert limiter.calls_per_minute == 60
        assert limiter.burst_size == 60
        assert limiter.refill_rate == 1.0  # 60 calls / 60 seconds
    
    def test_rate_limiter_custom_burst(self):
        """Test rate limiter with custom burst size."""
        limiter = RateLimiter(calls_per_minute=60, burst_size=100)
        
        assert limiter.burst_size == 100
    
    @pytest.mark.asyncio
    async def test_acquire_tokens(self):
        """Test acquiring tokens."""
        limiter = RateLimiter(calls_per_minute=600)  # Fast refill for testing
        
        # Should be able to acquire immediately
        result = await limiter.acquire(timeout=1.0)
        assert result is True
        
        # Tokens should be reduced
        assert limiter.get_available_tokens() < limiter.burst_size
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """Test that rate limiting actually limits."""
        limiter = RateLimiter(calls_per_minute=60, burst_size=2)
        
        # Consume all tokens
        await limiter.acquire()
        await limiter.acquire()
        
        # Next acquire should timeout quickly
        start = time.time()
        try:
            await limiter.acquire(timeout=0.5)
            # Should have waited
            elapsed = time.time() - start
            assert elapsed > 0.3  # Should wait for refill
        except RateLimitError:
            # Or timeout
            elapsed = time.time() - start
            assert elapsed >= 0.5
    
    def test_sync_acquire(self):
        """Test synchronous token acquisition."""
        limiter = RateLimiter(calls_per_minute=600)
        
        result = limiter.acquire_sync(timeout=1.0)
        assert result is True
    
    def test_try_acquire_success(self):
        """Test try_acquire when tokens available."""
        limiter = RateLimiter(calls_per_minute=60)
        
        result = limiter.try_acquire()
        assert result is True
    
    def test_try_acquire_failure(self):
        """Test try_acquire when no tokens available."""
        limiter = RateLimiter(calls_per_minute=60, burst_size=1)
        
        # Consume token
        limiter.try_acquire()
        
        # Should fail
        result = limiter.try_acquire()
        assert result is False
    
    def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(calls_per_minute=600)  # 10 tokens/sec
        
        # Consume tokens
        limiter.try_acquire()
        limiter.try_acquire()
        
        initial_tokens = limiter.get_available_tokens()
        
        # Wait for refill
        time.sleep(0.2)
        
        refilled_tokens = limiter.get_available_tokens()
        
        # Should have more tokens
        assert refilled_tokens > initial_tokens


class TestServiceRateLimiter:
    """Test ServiceRateLimiter class."""
    
    def test_service_limiter_initialization(self):
        """Test service rate limiter initialization."""
        limiter = ServiceRateLimiter()
        
        # Should have default services registered
        assert "wolfram" in limiter.limiters
        assert "arxiv" in limiter.limiters
        assert "default" in limiter.limiters
    
    def test_register_custom_service(self):
        """Test registering custom service."""
        limiter = ServiceRateLimiter()
        
        limiter.register_service("custom_api", calls_per_minute=30)
        
        assert "custom_api" in limiter.limiters
        assert limiter.limiters["custom_api"].calls_per_minute == 30
    
    @pytest.mark.asyncio
    async def test_acquire_for_service(self):
        """Test acquiring token for specific service."""
        limiter = ServiceRateLimiter()
        
        result = await limiter.acquire("wolfram", timeout=1.0)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_acquire_for_unknown_service(self):
        """Test acquiring token for unknown service uses default."""
        limiter = ServiceRateLimiter()
        
        # Should use default limiter
        result = await limiter.acquire("unknown_service", timeout=1.0)
        assert result is True
    
    def test_get_status(self):
        """Test getting rate limiter status."""
        limiter = ServiceRateLimiter()
        
        status = limiter.get_status()
        
        assert "wolfram" in status
        assert "available_tokens" in status["wolfram"]
        assert "calls_per_minute" in status["wolfram"]


class TestTimeoutLock:
    """Test TimeoutLock class."""
    
    def test_lock_acquire_and_release(self):
        """Test basic lock acquisition and release."""
        lock = TimeoutLock(timeout=1.0)
        
        with lock.acquire():
            # Lock held
            pass
        
        # Lock should be released
        assert lock.try_acquire()
        lock.release()
    
    def test_lock_timeout(self):
        """Test lock timeout."""
        lock = TimeoutLock(timeout=0.5)
        
        # Acquire lock
        lock.try_acquire()
        
        # Try to acquire again (should timeout)
        with pytest.raises(ConcurrencyError) as exc_info:
            with lock.acquire(timeout=0.5):
                pass
        
        assert "Failed to acquire lock" in str(exc_info.value)
        
        # Release
        lock.release()
    
    def test_lock_name(self):
        """Test lock with custom name."""
        lock = TimeoutLock(name="test_lock")
        
        assert lock.name == "test_lock"


class TestAsyncRWLock:
    """Test AsyncRWLock class."""
    
    @pytest.mark.asyncio
    async def test_multiple_readers(self):
        """Test multiple readers can hold lock simultaneously."""
        lock = AsyncRWLock()
        results = []
        
        async def reader(reader_id):
            async with lock.read():
                results.append(f"reader-{reader_id}-start")
                await asyncio.sleep(0.1)
                results.append(f"reader-{reader_id}-end")
        
        # Start multiple readers
        await asyncio.gather(
            reader(1),
            reader(2),
            reader(3)
        )
        
        # All readers should have started before any finished
        start_count = len([r for r in results if "start" in r])
        assert start_count == 3
    
    @pytest.mark.asyncio
    async def test_writer_exclusivity(self):
        """Test writer has exclusive access."""
        lock = AsyncRWLock()
        results = []
        
        async def writer(writer_id):
            async with lock.write():
                results.append(f"writer-{writer_id}-start")
                await asyncio.sleep(0.1)
                results.append(f"writer-{writer_id}-end")
        
        # Start multiple writers
        await asyncio.gather(
            writer(1),
            writer(2)
        )
        
        # Writers should have run sequentially
        assert results[0] == "writer-1-start"
        assert results[1] == "writer-1-end"
        assert results[2] == "writer-2-start"
        assert results[3] == "writer-2-end"
    
    @pytest.mark.asyncio
    async def test_read_write_lock_timeout(self):
        """Test read/write lock timeout."""
        lock = AsyncRWLock()
        
        # Acquire write lock
        async with lock.write():
            # Try to acquire read lock (should timeout)
            with pytest.raises(ConcurrencyError):
                async with lock.read(timeout=0.5):
                    pass


class TestSynchronized:
    """Test synchronized decorator."""
    
    def test_synchronized_decorator(self):
        """Test synchronized decorator prevents concurrent access."""
        lock = TimeoutLock()
        call_order = []
        
        @synchronized(lock)
        def critical_section(thread_id):
            call_order.append(f"{thread_id}-start")
            time.sleep(0.05)
            call_order.append(f"{thread_id}-end")
        
        import threading
        
        t1 = threading.Thread(target=critical_section, args=(1,))
        t2 = threading.Thread(target=critical_section, args=(2,))
        
        t1.start()
        time.sleep(0.01)  # Ensure t1 starts first
        t2.start()
        
        t1.join()
        t2.join()
        
        # Should have run sequentially
        assert call_order[0] == "1-start"
        assert call_order[1] == "1-end"
        assert call_order[2] == "2-start"
        assert call_order[3] == "2-end"


class TestAsyncSynchronized:
    """Test async_synchronized decorator."""
    
    @pytest.mark.asyncio
    async def test_async_synchronized_read(self):
        """Test async synchronized decorator with read mode."""
        lock = AsyncRWLock()
        results = []
        
        @async_synchronized(lock, mode="read")
        async def read_operation(reader_id):
            results.append(f"reader-{reader_id}")
            await asyncio.sleep(0.05)
        
        # Multiple readers should work
        await asyncio.gather(
            read_operation(1),
            read_operation(2)
        )
        
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_async_synchronized_write(self):
        """Test async synchronized decorator with write mode."""
        lock = AsyncRWLock()
        results = []
        
        @async_synchronized(lock, mode="write")
        async def write_operation(writer_id):
            results.append(f"writer-{writer_id}-start")
            await asyncio.sleep(0.05)
            results.append(f"writer-{writer_id}-end")
        
        # Writers should be sequential
        await asyncio.gather(
            write_operation(1),
            write_operation(2)
        )
        
        assert results[0] == "writer-1-start"
        assert results[1] == "writer-1-end"


class TestSemaphore:
    """Test Semaphore class."""
    
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent(self):
        """Test semaphore limits concurrent operations."""
        sem = Semaphore(max_concurrent=2, timeout=1.0)
        active_count = []
        max_concurrent = 0
        
        async def operation(op_id):
            nonlocal max_concurrent
            async with sem.acquire():
                active_count.append(op_id)
                max_concurrent = max(max_concurrent, len(active_count))
                await asyncio.sleep(0.05)
                active_count.remove(op_id)
        
        # Start 5 operations
        await asyncio.gather(*[operation(i) for i in range(5)])
        
        # Max concurrent should be 2
        assert max_concurrent == 2
    
    @pytest.mark.asyncio
    async def test_semaphore_timeout(self):
        """Test semaphore timeout."""
        sem = Semaphore(max_concurrent=1, timeout=0.3)
        
        async def long_operation():
            async with sem.acquire():
                await asyncio.sleep(1.0)
        
        async def quick_operation():
            async with sem.acquire(timeout=0.3):
                pass
        
        # Start long operation
        task1 = asyncio.create_task(long_operation())
        await asyncio.sleep(0.05)  # Let it acquire
        
        # Quick operation should timeout
        with pytest.raises(ConcurrencyError):
            await quick_operation()
        
        task1.cancel()


class TestResourcePool:
    """Test ResourcePool class."""
    
    @pytest.mark.asyncio
    async def test_resource_pool_creation(self):
        """Test resource pool creates resources."""
        created_resources = []
        
        async def create_resource():
            resource = Mock()
            created_resources.append(resource)
            return resource
        
        pool = ResourcePool(create_resource, max_size=3)
        
        async with pool.acquire() as resource:
            assert resource is not None
            assert len(created_resources) == 1
    
    @pytest.mark.asyncio
    async def test_resource_pool_reuse(self):
        """Test resource pool reuses resources."""
        created_count = [0]
        
        async def create_resource():
            created_count[0] += 1
            return Mock()
        
        pool = ResourcePool(create_resource, max_size=2)
        
        # Acquire and release
        async with pool.acquire():
            pass
        
        # Acquire again - should reuse
        async with pool.acquire():
            pass
        
        # Should only create once
        assert created_count[0] == 1
    
    @pytest.mark.asyncio
    async def test_resource_pool_max_size(self):
        """Test resource pool respects max size."""
        created_count = [0]
        
        async def create_resource():
            created_count[0] += 1
            return Mock()
        
        pool = ResourcePool(create_resource, max_size=2, timeout=0.5)
        
        async def use_resource(delay):
            async with pool.acquire():
                await asyncio.sleep(delay)
        
        # Start 3 operations (max 2 concurrent)
        tasks = [
            asyncio.create_task(use_resource(0.2)),
            asyncio.create_task(use_resource(0.2)),
            asyncio.create_task(use_resource(0.1))
        ]
        
        await asyncio.gather(*tasks)
        
        # Should create at most 2 resources
        assert created_count[0] <= 2


class TestGlobalLimiter:
    """Test global rate limiter instance."""
    
    def test_get_global_limiter(self):
        """Test getting global limiter instance."""
        limiter1 = get_global_limiter()
        limiter2 = get_global_limiter()
        
        # Should return same instance
        assert limiter1 is limiter2
