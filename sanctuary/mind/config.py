"""
Configuration management for Sanctuary's cognitive system
"""
from pathlib import Path
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Model configuration settings."""
    path: str
    device: str = "auto"
    dtype: str = "float16"
    max_length: int = 2048
    temperature: float = 0.7

@dataclass
class SystemConfig:
    """System-wide configuration settings."""
    base_dir: Path
    chroma_dir: Path
    model_dir: Path
    cache_dir: Path
    log_dir: Path
    
    @classmethod
    def from_json(cls, config_path: str, project_root: Optional[Path] = None) -> 'SystemConfig':
        """Load system configuration from JSON file.
        
        Args:
            config_path: Path to the system.json configuration file
            project_root: Optional project root directory for resolving relative paths.
                         If None, uses the parent directory of config_path.
        
        Environment variables (if set) override config file values:
            - SANCTUARY_BASE_DIR
            - SANCTUARY_CHROMA_DIR
            - SANCTUARY_MODEL_DIR
            - SANCTUARY_CACHE_DIR
            - SANCTUARY_LOG_DIR
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Determine project root for resolving relative paths
        if project_root is None:
            # Get the directory containing the config file
            config_path_obj = Path(config_path).resolve()
            current = config_path_obj.parent
            
            # Search upward for project root markers
            project_markers = ['pyproject.toml', 'setup.py', '.git', 'uv.lock']
            max_levels = 5  # Limit upward search to prevent going too far
            
            for _ in range(max_levels):
                # Check if any project marker exists in current directory
                if any((current / marker).exists() for marker in project_markers):
                    project_root = current
                    break
                
                # Move up one level
                parent = current.parent
                if parent == current:  # Reached filesystem root
                    break
                current = parent
            
            # If no project root found, fall back to config directory's parent
            if project_root is None:
                config_dir = config_path_obj.parent
                if config_dir.name == 'config':
                    project_root = config_dir.parent
                else:
                    project_root = config_dir
        
        def resolve_path(env_var: str, config_key: str) -> Path:
            """Resolve a path from environment variable or config file."""
            # First check environment variable
            env_value = os.environ.get(env_var)
            if env_value:
                path = Path(env_value)
            else:
                path = Path(config[config_key])
            
            # Resolve relative paths against project root
            if not path.is_absolute():
                path = (project_root / path).resolve()
            
            return path
        
        return cls(
            base_dir=resolve_path('SANCTUARY_BASE_DIR', 'base_dir'),
            chroma_dir=resolve_path('SANCTUARY_CHROMA_DIR', 'chroma_dir'),
            model_dir=resolve_path('SANCTUARY_MODEL_DIR', 'model_dir'),
            cache_dir=resolve_path('SANCTUARY_CACHE_DIR', 'cache_dir'),
            log_dir=resolve_path('SANCTUARY_LOG_DIR', 'log_dir')
        )

class ModelRegistry:
    """Registry of model configurations."""
    def __init__(self, config_path: str):
        self.models: Dict[str, ModelConfig] = {}
        self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        """Load model configurations from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        for model_id, settings in config['models'].items():
            self.models[model_id] = ModelConfig(**settings)
    
    def get_model_config(self, model_id: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_id not in self.models:
            raise KeyError(f"Model {model_id} not found in registry")
        return self.models[model_id]
    
    def register_model(self, model_id: str, config: ModelConfig):
        """Register a new model configuration."""
        self.models[model_id] = config