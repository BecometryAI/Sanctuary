"""
Exception Hierarchy for Lyra Emergence System

Structured exception classes for all subsystems to enable proper error handling,
recovery, and debugging throughout the system.

Author: Lyra Emergence Team
Date: January 2, 2026
"""

from typing import Optional, Dict, Any


class LyraBaseException(Exception):
    """
    Base exception for all Lyra errors.
    
    All Lyra-specific exceptions should inherit from this class to enable
    systematic error handling and recovery strategies.
    
    Attributes:
        message: Human-readable error description
        context: Additional context about the error (dict)
        recoverable: Whether this error can be recovered from
    """
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False
    ):
        self.message = message
        self.context = context or {}
        self.recoverable = recoverable
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """String representation with context."""
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (context: {ctx_str})"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "context": self.context,
            "recoverable": self.recoverable
        }


class ModelLoadError(LyraBaseException):
    """
    Model loading/initialization failures.
    
    Raised when:
    - Model files cannot be loaded
    - Model initialization fails
    - GPU memory is insufficient
    - Model architecture is incompatible
    """
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        context = context or {}
        if model_name:
            context["model_name"] = model_name
        super().__init__(message, context, recoverable)


class MemoryError(LyraBaseException):
    """
    Memory system errors (ChromaDB, consolidation).
    
    Raised when:
    - ChromaDB operations fail
    - Memory consolidation errors occur
    - Vector store corruption detected
    - Backup/restore operations fail
    """
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        context = context or {}
        if operation:
            context["operation"] = operation
        super().__init__(message, context, recoverable)


class ConsciousnessError(LyraBaseException):
    """
    Consciousness subsystem errors.
    
    Raised when:
    - Emotion state updates fail
    - Goal management encounters errors
    - Cognitive processing failures occur
    - Self-awareness systems malfunction
    """
    
    def __init__(
        self,
        message: str,
        subsystem: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        context = context or {}
        if subsystem:
            context["subsystem"] = subsystem
        super().__init__(message, context, recoverable)


class RouterError(LyraBaseException):
    """
    Routing and specialist selection errors.
    
    Raised when:
    - Specialist selection fails
    - Model routing encounters errors
    - Load balancing failures occur
    - Specialist invocation errors
    """
    
    def __init__(
        self,
        message: str,
        specialist: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        context = context or {}
        if specialist:
            context["specialist"] = specialist
        super().__init__(message, context, recoverable)


class GPUMemoryError(ModelLoadError):
    """
    GPU memory-specific errors.
    
    Raised when:
    - GPU memory is exhausted
    - Memory allocation fails
    - OOM conditions occur
    """
    
    def __init__(
        self,
        message: str,
        memory_used: Optional[int] = None,
        memory_total: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        context = context or {}
        if memory_used is not None:
            context["memory_used_mb"] = memory_used
        if memory_total is not None:
            context["memory_total_mb"] = memory_total
        super().__init__(message, context=context, recoverable=True)


class ValidationError(LyraBaseException):
    """
    Data validation errors.
    
    Raised when:
    - Memory entry validation fails
    - Schema validation errors occur
    - Invalid data format detected
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False
    ):
        context = context or {}
        if field:
            context["field"] = field
        if value is not None:
            context["value"] = str(value)
        super().__init__(message, context, recoverable)


class RateLimitError(LyraBaseException):
    """
    Rate limiting errors.
    
    Raised when:
    - API rate limits are exceeded
    - Request quota exhausted
    - Throttling required
    """
    
    def __init__(
        self,
        message: str,
        service: Optional[str] = None,
        retry_after: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        context = context or {}
        if service:
            context["service"] = service
        if retry_after is not None:
            context["retry_after_seconds"] = retry_after
        super().__init__(message, context, recoverable)


class ConcurrencyError(LyraBaseException):
    """
    Concurrency and locking errors.
    
    Raised when:
    - Lock acquisition timeouts
    - Deadlock detected
    - Race condition encountered
    """
    
    def __init__(
        self,
        message: str,
        resource: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True
    ):
        context = context or {}
        if resource:
            context["resource"] = resource
        super().__init__(message, context, recoverable)
