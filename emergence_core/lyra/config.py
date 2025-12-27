"""
Configuration management for Lyra's cognitive system
"""
from pathlib import Path
import json
from typing import Dict, Any
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
    def from_json(cls, config_path: str) -> 'SystemConfig':
        """Load system configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        return cls(
            base_dir=Path(config['base_dir']),
            chroma_dir=Path(config['chroma_dir']),
            model_dir=Path(config['model_dir']),
            cache_dir=Path(config['cache_dir']),
            log_dir=Path(config['log_dir'])
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