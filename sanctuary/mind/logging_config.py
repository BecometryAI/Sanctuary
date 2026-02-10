"""
Enhanced Logging Configuration for Sanctuary Emergence System

Provides structured logging with JSON output, context tracking, and log rotation.
Designed for debugging, monitoring, and error tracking in production environments.

Author: Sanctuary Emergence Team
Date: January 2, 2026
"""

import json
import logging
import logging.handlers
import sys
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import traceback

# Context variable for storing operation context across async boundaries
operation_context: ContextVar[Dict[str, Any]] = ContextVar('operation_context', default={})


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs log records as JSON for easier parsing and aggregation.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add operation context if available
        ctx = operation_context.get({})
        if ctx:
            log_data["context"] = ctx
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                'name', 'msg', 'args', 'created', 'filename', 'funcName',
                'levelname', 'levelno', 'lineno', 'module', 'msecs',
                'message', 'pathname', 'process', 'processName',
                'relativeCreated', 'thread', 'threadName', 'exc_info',
                'exc_text', 'stack_info', 'taskName'
            ]:
                log_data[key] = value
        
        return json.dumps(log_data)


class ContextualFormatter(logging.Formatter):
    """
    Human-readable formatter with context information.
    
    For console output and development.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with context."""
        # Get base formatted message
        base_msg = super().format(record)
        
        # Add operation context if available
        ctx = operation_context.get({})
        if ctx:
            ctx_str = " | ".join(f"{k}={v}" for k, v in ctx.items())
            base_msg = f"{base_msg} [{ctx_str}]"
        
        return base_msg


class OperationContext:
    """
    Context manager for tracking operations across async boundaries.
    
    Example:
        async with OperationContext(operation="memory_store", user_id="123"):
            await store_memory(entry)
            # All logs within this context will include operation and user_id
    """
    
    def __init__(self, **context_data):
        self.context_data = context_data
        self.token = None
        self.previous_context = None
    
    def __enter__(self):
        """Enter context and set operation context."""
        self.previous_context = operation_context.get({})
        # Merge with existing context
        new_context = {**self.previous_context, **self.context_data}
        self.token = operation_context.set(new_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous context."""
        if self.token:
            operation_context.reset(self.token)
        return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    enable_json: bool = False,
    enable_console: bool = True,
    enable_file: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 7
) -> logging.Logger:
    """
    Configure logging for Sanctuary system.
    
    Args:
        log_dir: Directory for log files (default: ./logs)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Use JSON formatting for file logs
        enable_console: Enable console output
        enable_file: Enable file output with rotation
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured root logger
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler (human-readable)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ContextualFormatter(
            fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        # Main log file
        log_file = log_dir / "sanctuary.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        if enable_json:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = ContextualFormatter(
                fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Error-only log file
        error_file = log_dir / "sanctuary_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        if enable_json:
            error_formatter = StructuredFormatter()
        else:
            error_formatter = ContextualFormatter(
                fmt='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Initialize default logging configuration
if not logging.getLogger().handlers:
    setup_logging()
