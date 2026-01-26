"""
Retry Decorator with Exponential Backoff

Provides retry logic for transient failures with configurable backoff strategies.
Useful for network operations, API calls, and other potentially flaky operations.

Author: Lyra Emergence Team
Date: January 2, 2026
"""

import asyncio
import functools
import logging
import time
from typing import Callable, Tuple, Type, Optional, Any

from ..exceptions import LyraBaseException

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator that retries a function with exponential backoff on failure.
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay between retries (default: 60.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback function called on each retry with (exception, attempt_number)
    
    Returns:
        Decorated function that implements retry logic
    
    Example:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def flaky_operation():
            # May fail transiently
            pass
        
        @retry_with_backoff(
            max_retries=5,
            exceptions=(ConnectionError, TimeoutError)
        )
        async def async_api_call():
            # Async operation with retries
            pass
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            """Wrapper for async functions."""
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Don't retry on last attempt
                    if attempt >= max_retries:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Log retry attempt
                    context = {}
                    if isinstance(e, LyraBaseException):
                        context = e.context
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s... Context: {context}"
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception as callback_error:
                            logger.error(f"Error in retry callback: {callback_error}")
                    
                    # Wait before retry
                    await asyncio.sleep(delay)
            
            # All retries exhausted
            logger.error(
                f"All {max_retries + 1} attempts failed for {func.__name__}. "
                f"Last error: {last_exception}"
            )
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            """Wrapper for synchronous functions."""
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    # Don't retry on last attempt
                    if attempt >= max_retries:
                        break
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Log retry attempt
                    context = {}
                    if isinstance(e, LyraBaseException):
                        context = e.context
                    
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s... Context: {context}"
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception as callback_error:
                            logger.error(f"Error in retry callback: {callback_error}")
                    
                    # Wait before retry
                    time.sleep(delay)
            
            # All retries exhausted
            logger.error(
                f"All {max_retries + 1} attempts failed for {func.__name__}. "
                f"Last error: {last_exception}"
            )
            if last_exception is not None:
                raise last_exception
            else:
                raise RuntimeError(
                    f"All {max_retries + 1} attempts failed for {func.__name__}, "
                    "but no exception was captured on the final attempt."
                )
        
        # Return appropriate wrapper based on whether function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def retry_on_exception(
    exception_type: Type[Exception],
    max_retries: int = 3,
    base_delay: float = 1.0
):
    """
    Simplified retry decorator for a specific exception type.
    
    Args:
        exception_type: The specific exception type to catch and retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
    
    Example:
        @retry_on_exception(ConnectionError, max_retries=5)
        def network_call():
            # Network operation
            pass
    """
    return retry_with_backoff(
        max_retries=max_retries,
        base_delay=base_delay,
        exceptions=(exception_type,)
    )


class RetryContext:
    """
    Context manager for retry logic with custom error handling.
    
    Example:
        async with RetryContext(max_retries=3) as retry:
            result = await retry.execute(async_operation)
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.exceptions = exceptions
        self.attempt = 0
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with retry logic."""
        decorated = retry_with_backoff(
            max_retries=self.max_retries,
            base_delay=self.base_delay,
            exceptions=self.exceptions
        )(func)
        
        if asyncio.iscoroutinefunction(func):
            return await decorated(*args, **kwargs)
        else:
            return decorated(*args, **kwargs)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False
