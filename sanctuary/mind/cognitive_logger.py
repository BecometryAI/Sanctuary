"""
Logging configuration for Sanctuary's cognitive system
"""
import logging
import logging.handlers
from pathlib import Path
import json
from typing import Dict, Any

class CognitiveLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self._handlers = []  # Track handlers for cleanup
        self._setup_loggers()

    def _setup_loggers(self):
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup routing logger
        routing_logger = logging.getLogger('sanctuary.routing')
        routing_logger.setLevel(logging.INFO)
        # Remove any existing file handlers to prevent duplicates
        for h in routing_logger.handlers[:]:
            if isinstance(h, logging.handlers.RotatingFileHandler):
                h.close()
                routing_logger.removeHandler(h)
        routing_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'routing.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        routing_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        routing_logger.addHandler(routing_handler)
        self._handlers.append((routing_logger, routing_handler))

        # Setup model performance logger
        perf_logger = logging.getLogger('sanctuary.performance')
        perf_logger.setLevel(logging.INFO)
        # Remove any existing file handlers to prevent duplicates
        for h in perf_logger.handlers[:]:
            if isinstance(h, logging.handlers.RotatingFileHandler):
                h.close()
                perf_logger.removeHandler(h)
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / 'model_performance.log',
            maxBytes=10485760,
            backupCount=5
        )
        perf_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        perf_logger.addHandler(perf_handler)
        self._handlers.append((perf_logger, perf_handler))
        
    def log_routing_decision(self, message: str, intent: str, resonance_term: str, context: Dict[str, Any]):
        """Log routing decisions with context."""
        logger = logging.getLogger('sanctuary.routing')
        log_entry = {
            'message': message,
            'intent': intent,
            'resonance_term': resonance_term,
            'context_summary': self._summarize_context(context)
        }
        logger.info(f"Routing Decision: {json.dumps(log_entry, indent=2)}")
        
    def log_model_performance(self, model_type: str, operation: str, duration: float, success: bool):
        """Log model performance metrics."""
        logger = logging.getLogger('sanctuary.performance')
        log_entry = {
            'model': model_type,
            'operation': operation,
            'duration_ms': duration,
            'success': success
        }
        logger.info(f"Model Performance: {json.dumps(log_entry)}")
        
    def _summarize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summarized version of context for logging."""
        summary = {}
        for key, value in context.items():
            if isinstance(value, list):
                summary[key] = f"{len(value)} items"
            elif isinstance(value, dict):
                summary[key] = f"{len(value)} keys"
            else:
                summary[key] = str(value)[:100]  # Truncate long values
        return summary

    def close(self):
        """Close all logging handlers to release file locks."""
        for logger, handler in self._handlers:
            handler.close()
            logger.removeHandler(handler)
        self._handlers.clear()