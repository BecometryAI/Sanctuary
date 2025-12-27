"""
Model setup script for Lyra's cognitive system.
This script will download and configure all required models when run.
"""
import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, Any

def ensure_directories(base_dir: Path):
    """Create required directories if they don't exist."""
    dirs = [
        base_dir / "model_cache" / "models",
        base_dir / "model_cache" / "chroma_db",
        base_dir / "model_cache" / "cache",
        base_dir / "model_cache" / "voices",
        base_dir / "logs"
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

def load_model_config() -> Dict[str, Any]:
    """Load model configuration."""
    config_path = Path(__file__).parent.parent / "config" / "models.json"
    with open(config_path, 'r') as f:
        return json.load(f)

def download_model(model_id: str, save_path: Path, config: Dict[str, Any]):
    """Download and save a model."""
    print(f"Downloading {model_id}...")
    
    try:
        # Download tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(config["path"])
        tokenizer.save_pretrained(save_path)
        
        # Download model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            config["path"],
            torch_dtype=getattr(torch, config["dtype"]),
            device_map=config["device"]
        )
        model.save_pretrained(save_path)
        
        print(f"Successfully downloaded {model_id}")
        
    except Exception as e:
        print(f"Error downloading {model_id}: {str(e)}")
        raise

def setup_models(base_dir: Path):
    """Set up all required models."""
    config = load_model_config()
    model_dir = base_dir / "model_cache" / "models"
    
    # Calculate total required space
    print("Checking storage requirements...")
    required_space = {
        "router": 24,  # GB
        "pragmatist": 64,
        "philosopher": 64,
        "artist": 54,
        "voice": 54,
        "speech_recognition": 1,
        "text_to_speech": 1,
        "emotion_recognition": 1
    }
    
    total_space = sum(required_space.values())
    print(f"Total required space: {total_space}GB")
    
    # Get available space
    try:
        free_space = os.statvfs(model_dir).f_frsize * os.statvfs(model_dir).f_bavail / (1024**3)
        print(f"Available space: {free_space:.1f}GB")
        
        if free_space < total_space * 1.2:  # Include 20% buffer
            raise RuntimeError(f"Insufficient disk space. Need {total_space * 1.2:.1f}GB, have {free_space:.1f}GB")
    except AttributeError:
        # Windows doesn't have statvfs, use a different method
        import shutil
        free_space = shutil.disk_usage(model_dir).free / (1024**3)
        print(f"Available space: {free_space:.1f}GB")
        
        if free_space < total_space * 1.2:
            raise RuntimeError(f"Insufficient disk space. Need {total_space * 1.2:.1f}GB, have {free_space:.1f}GB")
    
    # Download models
    for model_id, model_config in config["models"].items():
        model_path = model_dir / model_id
        if not model_path.exists():
            download_model(model_id, model_path, model_config)
        else:
            print(f"Model {model_id} already exists, skipping...")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    
    print("Creating directories...")
    ensure_directories(base_dir)
    
    print("\nThis script will download all required models.")
    print("Please ensure you have sufficient disk space and a stable internet connection.")
    response = input("Do you want to continue? [y/N]: ")
    
    if response.lower() == 'y':
        setup_models(base_dir)
        print("\nModel setup complete!")
    else:
        print("\nSetup cancelled.")