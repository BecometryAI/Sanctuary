"""
Test configuration loading with environment variables and relative paths.
"""
import pytest
from pathlib import Path
import json
import os
from mind.config import SystemConfig


class TestSystemConfig:
    """Test SystemConfig loading and path resolution."""
    
    def test_load_from_json_with_relative_paths(self, tmp_path):
        """Test loading config with relative paths."""
        # Create a temporary config file
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "system.json"
        
        config_data = {
            "base_dir": ".",
            "chroma_dir": "./data/chroma",
            "model_dir": "./models",
            "cache_dir": "./data/cache",
            "log_dir": "./logs"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load config
        config = SystemConfig.from_json(str(config_file), project_root=tmp_path)
        
        # Verify paths are absolute and resolved correctly
        assert config.base_dir.is_absolute()
        assert config.chroma_dir.is_absolute()
        assert config.model_dir.is_absolute()
        assert config.cache_dir.is_absolute()
        assert config.log_dir.is_absolute()
        
        # Verify paths are under the project root
        assert str(config.base_dir).startswith(str(tmp_path))
        assert str(config.chroma_dir).startswith(str(tmp_path))
    
    def test_load_from_json_with_absolute_paths(self, tmp_path):
        """Test loading config with absolute paths."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "system.json"
        
        # Create absolute paths
        base = tmp_path / "absolute_base"
        chroma = tmp_path / "absolute_chroma"
        
        config_data = {
            "base_dir": str(base),
            "chroma_dir": str(chroma),
            "model_dir": str(tmp_path / "models"),
            "cache_dir": str(tmp_path / "cache"),
            "log_dir": str(tmp_path / "logs")
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Load config
        config = SystemConfig.from_json(str(config_file))
        
        # Verify absolute paths are preserved
        assert config.base_dir == base
        assert config.chroma_dir == chroma
    
    def test_environment_variable_override(self, tmp_path, monkeypatch):
        """Test that environment variables override config file values."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "system.json"
        
        config_data = {
            "base_dir": ".",
            "chroma_dir": "./data/chroma",
            "model_dir": "./models",
            "cache_dir": "./data/cache",
            "log_dir": "./logs"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Set environment variables
        env_base = tmp_path / "env_base"
        env_chroma = tmp_path / "env_chroma"
        
        monkeypatch.setenv('SANCTUARY_BASE_DIR', str(env_base))
        monkeypatch.setenv('SANCTUARY_CHROMA_DIR', str(env_chroma))
        
        # Load config
        config = SystemConfig.from_json(str(config_file), project_root=tmp_path)
        
        # Verify environment variables took precedence
        assert config.base_dir == env_base.resolve()
        assert config.chroma_dir == env_chroma.resolve()
        
        # Verify non-overridden paths still work
        assert config.model_dir.is_absolute()
        assert "models" in str(config.model_dir)
    
    def test_environment_variable_with_relative_path(self, tmp_path, monkeypatch):
        """Test that relative paths in environment variables are resolved."""
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        config_file = config_dir / "system.json"
        
        config_data = {
            "base_dir": ".",
            "chroma_dir": "./data/chroma",
            "model_dir": "./models",
            "cache_dir": "./data/cache",
            "log_dir": "./logs"
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Set environment variable with relative path
        monkeypatch.setenv('SANCTUARY_BASE_DIR', './custom_base')
        
        # Load config
        config = SystemConfig.from_json(str(config_file), project_root=tmp_path)
        
        # Verify relative path was resolved against project root
        assert config.base_dir.is_absolute()
        assert config.base_dir == (tmp_path / "custom_base").resolve()
    
    def test_real_config_file_loads(self):
        """Test that the actual project config files load successfully."""
        # Get the project root (assuming test is in sanctuary/tests)
        project_root = Path(__file__).parent.parent.parent
        
        # Test root config file
        root_config = project_root / "config" / "system.json"
        if root_config.exists():
            config = SystemConfig.from_json(str(root_config), project_root=project_root)
            assert config.base_dir.is_absolute()
            assert config.chroma_dir.is_absolute()
            assert config.model_dir.is_absolute()
            assert config.cache_dir.is_absolute()
            assert config.log_dir.is_absolute()
        
        # Test sanctuary config file
        ec_config = project_root / "sanctuary" / "config" / "system.json"
        if ec_config.exists():
            config = SystemConfig.from_json(str(ec_config), project_root=project_root)
            assert config.base_dir.is_absolute()
            assert config.chroma_dir.is_absolute()
