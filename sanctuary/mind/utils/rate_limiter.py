"""
Rate Limiting System for Lyra Emergence

Token bucket rate limiter with per-service limits to prevent API abuse
and respect external API rate limits.

Author: Lyra Emergence Team
Date: January 2, 2026
"""

import asyncio
import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional
import logging

from ..exceptions import RateLimitError
from ..logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for a rate-limited service."""
    calls_per_minute: int
    burst_size: Optional[int] = None  # Max burst, defaults to calls_per_minute
    
    def __post_init__(self):
        if self.burst_size is None:
            self.burst_size = self.calls_per_minute


class RateLimiter:
    """
    Token bucket rate limiter with per-service limits.
    
    Implements the token bucket algorithm where:
    - Tokens are added at a constant rate (refill_rate)
    - Each request consumes one token
    - Requests wait if no tokens available
    - Bucket has maximum capacity (burst_size)
    
    Example:
        limiter = RateLimiter(calls_per_minute=60)
        
        # Synchronous usage
        limiter.acquire_sync(timeout=10.0)
        make_api_call()
        
        # Asynchronous usage
        await limiter.acquire(timeout=10.0)
        await make_api_call()
    """
    
    def __init__(
        self,
        calls_per_minute: int,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_minute: Number of calls allowed per minute
            burst_size: Maximum burst size (defaults to calls_per_minute)
        """
        self.calls_per_minute = calls_per_minute
        self.burst_size = burst_size if burst_size is not None else calls_per_minute
        
        # Token bucket state
        self.tokens = float(self.burst_size)
        self.last_refill = time.time()
        
        # Calculate refill rate (tokens per second)
        self.refill_rate = calls_per_minute / 60.0
        
        # Locks for thread safety (both sync and async)
        self._sync_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        
        logger.info(
            f"Rate limiter initialized: {calls_per_minute} calls/min, "
            f"burst: {self.burst_size}, refill rate: {self.refill_rate:.2f} tokens/sec"
        )
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.burst_size, self.tokens + new_tokens)
        self.last_refill = now
    
    async def acquire(self, timeout: float = 10.0) -> bool:
        """
        Acquire a token, waiting if necessary (async).
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if token acquired, False if timeout
        
        Raises:
            RateLimitError: If timeout exceeded
        """
        start_time = time.time()
        
        async with self._async_lock:
            while True:
                self._refill_tokens()
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise RateLimitError(
                        f"Rate limit timeout after {timeout}s",
                        retry_after=1.0 / self.refill_rate
                    )
                
                # Wait for next token
                wait_time = min(1.0 / self.refill_rate, timeout - elapsed)
                await asyncio.sleep(wait_time)
    
    def acquire_sync(self, timeout: float = 10.0) -> bool:
        """
        Acquire a token, waiting if necessary (synchronous).
        
        Args:
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if token acquired
        
        Raises:
            RateLimitError: If timeout exceeded
        """
        start_time = time.time()
        
        with self._sync_lock:
            while True:
                self._refill_tokens()
                
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise RateLimitError(
                        f"Rate limit timeout after {timeout}s",
                        retry_after=1.0 / self.refill_rate
                    )
                
                # Release lock while waiting to allow other threads
                self._sync_lock.release()
                wait_time = min(1.0 / self.refill_rate, timeout - elapsed)
                time.sleep(wait_time)
                self._sync_lock.acquire()
    
    def try_acquire(self) -> bool:
        """
        Try to acquire a token without waiting.
        
        Returns:
            True if token acquired, False otherwise
        """
        with self._sync_lock:
            self._refill_tokens()
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            
            return False
    
    def get_available_tokens(self) -> float:
        """
        Get number of available tokens.
        
        Returns:
            Number of tokens currently available
        """
        with self._sync_lock:
            self._refill_tokens()
            return self.tokens


class ServiceRateLimiter:
    """
    Manages rate limiters for multiple services.
    
    Example:
        limiter = ServiceRateLimiter()
        limiter.register_service("wolfram", calls_per_minute=60)
        limiter.register_service("arxiv", calls_per_minute=30)
        
        await limiter.acquire("wolfram")
        result = await call_wolfram_api()
    """
    
    # Default rate limits for known services
    DEFAULT_LIMITS = {
        "wolfram": RateLimitConfig(calls_per_minute=60),
        "arxiv": RateLimitConfig(calls_per_minute=30),
        "wikipedia": RateLimitConfig(calls_per_minute=100),
        "chromadb": RateLimitConfig(calls_per_minute=1000),
        "default": RateLimitConfig(calls_per_minute=60)
    }
    
    def __init__(self):
        """Initialize service rate limiter."""
        self.limiters: Dict[str, RateLimiter] = {}
        
        # Register default limiters
        for service, config in self.DEFAULT_LIMITS.items():
            self.register_service(
                service,
                calls_per_minute=config.calls_per_minute,
                burst_size=config.burst_size
            )
        
        logger.info(f"Service rate limiter initialized with {len(self.limiters)} services")
    
    def register_service(
        self,
        service_name: str,
        calls_per_minute: int,
        burst_size: Optional[int] = None
    ):
        """
        Register a service with rate limiting.
        
        Args:
            service_name: Name of the service
            calls_per_minute: Number of calls allowed per minute
            burst_size: Maximum burst size
        """
        self.limiters[service_name] = RateLimiter(
            calls_per_minute=calls_per_minute,
            burst_size=burst_size
        )
        logger.info(f"Registered rate limiter for service '{service_name}'")
    
    async def acquire(self, service_name: str, timeout: float = 10.0) -> bool:
        """
        Acquire a token for a service (async).
        
        Args:
            service_name: Name of the service
            timeout: Maximum time to wait
        
        Returns:
            True if token acquired
        
        Raises:
            RateLimitError: If timeout exceeded
        """
        limiter = self.limiters.get(service_name)
        
        if limiter is None:
            # Use default limiter for unknown services
            logger.warning(
                f"No rate limiter for service '{service_name}', using default"
            )
            limiter = self.limiters.get("default")
            
            if limiter is None:
                # Create default if not exists
                self.register_service("default", calls_per_minute=60)
                limiter = self.limiters["default"]
        
        return await limiter.acquire(timeout)
    
    def acquire_sync(self, service_name: str, timeout: float = 10.0) -> bool:
        """
        Acquire a token for a service (sync).
        
        Args:
            service_name: Name of the service
            timeout: Maximum time to wait
        
        Returns:
            True if token acquired
        
        Raises:
            RateLimitError: If timeout exceeded
        """
        limiter = self.limiters.get(service_name)
        
        if limiter is None:
            logger.warning(
                f"No rate limiter for service '{service_name}', using default"
            )
            limiter = self.limiters.get("default")
            
            if limiter is None:
                self.register_service("default", calls_per_minute=60)
                limiter = self.limiters["default"]
        
        return limiter.acquire_sync(timeout)
    
    def get_status(self) -> Dict[str, Dict[str, float]]:
        """
        Get status of all rate limiters.
        
        Returns:
            Dictionary mapping service names to status info
        """
        return {
            service: {
                "available_tokens": limiter.get_available_tokens(),
                "calls_per_minute": limiter.calls_per_minute,
                "burst_size": limiter.burst_size
            }
            for service, limiter in self.limiters.items()
        }


# Global service rate limiter instance
_global_limiter: Optional[ServiceRateLimiter] = None


def get_global_limiter() -> ServiceRateLimiter:
    """
    Get or create global service rate limiter.
    
    Returns:
        Global ServiceRateLimiter instance
    """
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = ServiceRateLimiter()
    return _global_limiter
