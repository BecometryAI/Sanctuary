"""
Tests for cognitive logging system
"""
import pytest
import json
import logging
from pathlib import Path
from mind.cognitive_logger import CognitiveLogger

def test_logger_initialization(test_data_dir):
    """Test logger initialization and file creation."""
    logger = CognitiveLogger(test_data_dir / "logs")

    try:
        # Check log files were created
        assert (test_data_dir / "logs" / "routing.log").exists()
        assert (test_data_dir / "logs" / "model_performance.log").exists()
    finally:
        logger.close()  # Release file locks

def test_routing_decision_logging(test_logger):
    """Test logging of routing decisions."""
    message = "Test message"
    intent = "philosopher"
    resonance_term = "test_term"
    context = {"key": "value"}
    
    test_logger.log_routing_decision(message, intent, resonance_term, context)
    
    # Read log file
    log_file = Path(test_logger.log_dir) / "routing.log"
    with open(log_file, 'r') as f:
        log_content = f.read()
        
    assert "Test message" in log_content
    assert "philosopher" in log_content
    assert "test_term" in log_content

def test_model_performance_logging(test_logger):
    """Test logging of model performance metrics."""
    test_logger.log_model_performance("router", "classification", 100.5, True)
    
    # Read log file
    log_file = Path(test_logger.log_dir) / "model_performance.log"
    with open(log_file, 'r') as f:
        log_content = f.read()
        
    assert "router" in log_content
    assert "classification" in log_content
    assert "100.5" in log_content
    assert "true" in log_content.lower()

def test_context_summarization(test_logger):
    """Test context summarization for logging."""
    context = {
        "list_data": [1, 2, 3, 4, 5],
        "dict_data": {"key1": "value1", "key2": "value2"},
        "long_string": "x" * 200
    }

    test_logger.log_routing_decision("test", "test", None, context)

    # Read log file
    log_file = Path(test_logger.log_dir) / "routing.log"
    with open(log_file, 'r') as f:
        log_content = f.read()

    assert "5 items" in log_content  # List summary
    assert "2 keys" in log_content   # Dict summary
    # Check that long strings are truncated (100 char limit in _summarize_context)
    # The full 200 x's should NOT appear, only first 100
    assert "x" * 200 not in log_content
    assert "x" * 100 in log_content  # Truncated to 100 chars