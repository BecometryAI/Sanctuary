"""Common utilities for Sanctuary's autonomous systems"""

import importlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

DEPENDENCIES = {
    'chromadb': '0.4.0',
    'schedule': '1.2.0'
}

def get_dependency(name: str) -> Any:
    """
    Get a dependency module, installing if needed
    
    Args:
        name: Name of the dependency
        
    Returns:
        The imported module
        
    Raises:
        ImportError if module cannot be imported
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        version = DEPENDENCIES.get(name, '')
        req = f"{name}>={version}" if version else name
        logger.info("Installing %s", req)
        subprocess.check_call([sys.executable, "-m", "pip", "install", req])
        return importlib.import_module(name)
        
def ensure_dependencies() -> None:
    """Ensure all required dependencies are installed"""
    for name in DEPENDENCIES:
        try:
            get_dependency(name)
        except Exception as e:
            logger.error("Failed to install %s: %s", name, e)
            raise

def safe_json_load(path: Path | str) -> Dict[str, Any]:
    """Load JSON with proper error handling"""
    if isinstance(path, str):
        path = Path(path)
        
    if not path.exists():
        msg = f"File does not exist: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
        
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            msg = f"Expected JSON object, got {type(data).__name__}"
            logger.error(msg)
            raise ValueError(msg)
        return data
        
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in {path}: {e}"
        logger.error(msg)
        raise ValueError(msg) from None
        
    except OSError as e:
        msg = f"Error reading {path}: {e}"
        logger.error(msg)
        raise RuntimeError(msg) from None