"""
Rate limiting implementation for API calls
"""
import time
import asyncio
from typing import Dict, Any, Callable, Awaitable
from functools import wraps

class RateLimiter:
    def __init__(self, calls: int, period: float):
        """
        Initialize rate limiter
        
        Args:
            calls: Number of calls allowed in period
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.timestamps = []
    
    async def acquire(self):
        """Wait until a call is allowed"""
        now = time.time()
        
        # Remove old timestamps
        self.timestamps = [ts for ts in self.timestamps if now - ts <= self.period]
        
        # If at limit, wait until oldest call expires
        if len(self.timestamps) >= self.calls:
            wait_time = self.timestamps[0] + self.period - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                return await self.acquire()
        
        # Add new timestamp and proceed
        self.timestamps.append(now)
        return True

def rate_limit(calls: int, period: float):
    """
    Decorator for rate limiting async functions
    
    Args:
        calls: Number of calls allowed in period
        period: Time period in seconds
    """
    limiter = RateLimiter(calls, period)
    
    def decorator(func: Callable[..., Awaitable[Any]]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            await limiter.acquire()
            return await func(*args, **kwargs)
        return wrapper
    return decorator