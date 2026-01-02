"""
Concurrency Primitives for Lyra Emergence

Thread-safe locks with timeouts and read-write locks for async code.

Author: Lyra Emergence Team
Date: January 2, 2026
"""

import asyncio
import threading
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Optional
import functools
import logging

from ..exceptions import ConcurrencyError
from ..logging_config import get_logger

logger = get_logger(__name__)


class TimeoutLock:
    """
    Threading lock with automatic timeout to prevent deadlocks.
    
    Example:
        lock = TimeoutLock(timeout=5.0)
        
        with lock.acquire():
            # Critical section
            pass
    """
    
    def __init__(self, timeout: float = 10.0, name: Optional[str] = None):
        """
        Initialize timeout lock.
        
        Args:
            timeout: Default timeout in seconds
            name: Optional name for debugging
        """
        self._lock = threading.Lock()
        self.timeout = timeout
        self.name = name or f"lock-{id(self)}"
    
    @contextmanager
    def acquire(self, timeout: Optional[float] = None):
        """
        Acquire lock with timeout (context manager).
        
        Args:
            timeout: Timeout in seconds (uses default if None)
        
        Raises:
            ConcurrencyError: If timeout exceeded
        """
        timeout = timeout if timeout is not None else self.timeout
        acquired = self._lock.acquire(timeout=timeout)
        
        if not acquired:
            raise ConcurrencyError(
                f"Failed to acquire lock '{self.name}' within {timeout}s",
                resource=self.name
            )
        
        try:
            yield
        finally:
            self._lock.release()
    
    def try_acquire(self) -> bool:
        """
        Try to acquire lock without waiting.
        
        Returns:
            True if lock acquired, False otherwise
        """
        return self._lock.acquire(blocking=False)
    
    def release(self):
        """Release the lock."""
        self._lock.release()


class AsyncRWLock:
    """
    Async read-write lock allowing multiple readers or one writer.
    
    Features:
    - Multiple readers can hold the lock simultaneously
    - Only one writer can hold the lock
    - Writers have priority over readers
    
    Example:
        lock = AsyncRWLock()
        
        # Read access
        async with lock.read():
            data = read_shared_data()
        
        # Write access
        async with lock.write():
            write_shared_data(new_data)
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize async read-write lock.
        
        Args:
            name: Optional name for debugging
        """
        self.name = name or f"rwlock-{id(self)}"
        self._readers = 0
        self._writers = 0
        self._read_ready = asyncio.Condition()
        self._write_ready = asyncio.Condition()
    
    @asynccontextmanager
    async def read(self, timeout: float = 10.0):
        """
        Acquire read lock (context manager).
        
        Args:
            timeout: Timeout in seconds
        
        Raises:
            ConcurrencyError: If timeout exceeded
        """
        try:
            await asyncio.wait_for(self._acquire_read(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ConcurrencyError(
                f"Failed to acquire read lock '{self.name}' within {timeout}s",
                resource=self.name
            )
        
        try:
            yield
        finally:
            await self._release_read()
    
    @asynccontextmanager
    async def write(self, timeout: float = 10.0):
        """
        Acquire write lock (context manager).
        
        Args:
            timeout: Timeout in seconds
        
        Raises:
            ConcurrencyError: If timeout exceeded
        """
        try:
            await asyncio.wait_for(self._acquire_write(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ConcurrencyError(
                f"Failed to acquire write lock '{self.name}' within {timeout}s",
                resource=self.name
            )
        
        try:
            yield
        finally:
            await self._release_write()
    
    async def _acquire_read(self):
        """Acquire read lock (internal)."""
        async with self._read_ready:
            # Wait for writers to finish
            while self._writers > 0:
                await self._read_ready.wait()
            self._readers += 1
    
    async def _release_read(self):
        """Release read lock (internal)."""
        async with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                # Notify waiting writers
                async with self._write_ready:
                    self._write_ready.notify()
    
    async def _acquire_write(self):
        """Acquire write lock (internal)."""
        async with self._write_ready:
            # Wait for readers and writers to finish
            while self._readers > 0 or self._writers > 0:
                await self._write_ready.wait()
            self._writers += 1
    
    async def _release_write(self):
        """Release write lock (internal)."""
        async with self._write_ready:
            self._writers -= 1
            # Notify all waiting readers and writers
            self._write_ready.notify_all()
            async with self._read_ready:
                self._read_ready.notify_all()


def synchronized(lock: TimeoutLock):
    """
    Decorator to synchronize function access with a lock.
    
    Args:
        lock: TimeoutLock instance to use
    
    Example:
        lock = TimeoutLock()
        
        @synchronized(lock)
        def critical_section():
            # Only one thread can execute this at a time
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock.acquire():
                return func(*args, **kwargs)
        return wrapper
    return decorator


def async_synchronized(lock: AsyncRWLock, mode: str = "write"):
    """
    Decorator to synchronize async function access with a read-write lock.
    
    Args:
        lock: AsyncRWLock instance to use
        mode: "read" or "write" mode
    
    Example:
        lock = AsyncRWLock()
        
        @async_synchronized(lock, mode="read")
        async def read_data():
            # Multiple readers can execute simultaneously
            pass
        
        @async_synchronized(lock, mode="write")
        async def write_data():
            # Only one writer can execute at a time
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if mode == "read":
                async with lock.read():
                    return await func(*args, **kwargs)
            else:
                async with lock.write():
                    return await func(*args, **kwargs)
        return wrapper
    return decorator


class Semaphore:
    """
    Async semaphore with timeout support.
    
    Limits number of concurrent operations.
    
    Example:
        sem = Semaphore(max_concurrent=5)
        
        async with sem.acquire():
            await perform_operation()
    """
    
    def __init__(self, max_concurrent: int, timeout: float = 10.0):
        """
        Initialize semaphore.
        
        Args:
            max_concurrent: Maximum number of concurrent operations
            timeout: Default timeout in seconds
        """
        self._sem = asyncio.Semaphore(max_concurrent)
        self.timeout = timeout
        self.max_concurrent = max_concurrent
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire semaphore (context manager).
        
        Args:
            timeout: Timeout in seconds (uses default if None)
        
        Raises:
            ConcurrencyError: If timeout exceeded
        """
        timeout = timeout if timeout is not None else self.timeout
        
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            raise ConcurrencyError(
                f"Failed to acquire semaphore within {timeout}s",
                resource=f"semaphore-{self.max_concurrent}"
            )
        
        try:
            yield
        finally:
            self._sem.release()


class ResourcePool:
    """
    Thread-safe resource pool with automatic cleanup.
    
    Example:
        pool = ResourcePool(create_connection, max_size=10)
        
        async with pool.acquire() as conn:
            await conn.execute(query)
    """
    
    def __init__(
        self,
        create_func,
        max_size: int = 10,
        timeout: float = 10.0,
        cleanup_func=None
    ):
        """
        Initialize resource pool.
        
        Args:
            create_func: Function to create new resources
            max_size: Maximum pool size
            timeout: Acquisition timeout
            cleanup_func: Optional cleanup function for resources
        """
        self.create_func = create_func
        self.cleanup_func = cleanup_func
        self.max_size = max_size
        self.timeout = timeout
        
        self._pool = []
        self._in_use = set()
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
    
    @asynccontextmanager
    async def acquire(self, timeout: Optional[float] = None):
        """
        Acquire a resource from the pool.
        
        Args:
            timeout: Timeout in seconds
        
        Yields:
            Resource from pool
        
        Raises:
            ConcurrencyError: If timeout exceeded
        """
        timeout = timeout if timeout is not None else self.timeout
        start_time = time.time()
        
        async with self._condition:
            while True:
                # Try to get from pool
                if self._pool:
                    resource = self._pool.pop()
                    self._in_use.add(id(resource))
                    break
                
                # Create new if under limit
                if len(self._in_use) < self.max_size:
                    resource = await self.create_func()
                    self._in_use.add(id(resource))
                    break
                
                # Wait for available resource
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise ConcurrencyError(
                        f"Failed to acquire resource within {timeout}s",
                        resource="resource-pool"
                    )
                
                remaining = timeout - elapsed
                try:
                    await asyncio.wait_for(
                        self._condition.wait(),
                        timeout=remaining
                    )
                except asyncio.TimeoutError:
                    raise ConcurrencyError(
                        f"Failed to acquire resource within {timeout}s",
                        resource="resource-pool"
                    )
        
        try:
            yield resource
        finally:
            # Return to pool
            async with self._condition:
                self._in_use.discard(id(resource))
                self._pool.append(resource)
                self._condition.notify()
    
    async def close(self):
        """Close pool and cleanup all resources."""
        async with self._lock:
            if self.cleanup_func:
                for resource in self._pool:
                    try:
                        await self.cleanup_func(resource)
                    except Exception as e:
                        logger.error(f"Error cleaning up resource: {e}")
            self._pool.clear()
            self._in_use.clear()
