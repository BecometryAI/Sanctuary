"""
Setup script for SearXNG instance
"""
import os
import sys
import subprocess
import json
from pathlib import Path

def setup_searxng():
    """Set up SearXNG using Docker"""
    print("Setting up SearXNG...")
    
    # Create Docker compose file
    compose_content = """
version: '3.7'

services:
  searxng:
    container_name: searxng
    image: searxng/searxng
    ports:
      - "8080:8080"
    volumes:
      - ./searxng:/etc/searxng
    environment:
      - INSTANCE_NAME=lyra-searxng
      - BASE_URL=http://localhost:8080/
      - ULTRASECRET_KEY=${SEARXNG_KEY}
    restart: unless-stopped
    """
    
    # Create directory structure
    base_dir = Path(__file__).parent.parent
    searxng_dir = base_dir / "searxng"
    searxng_dir.mkdir(exist_ok=True)
    
    # Write Docker compose file
    with open(base_dir / "docker-compose.yml", "w") as f:
        f.write(compose_content.strip())
    
    # Generate secret key
    import secrets
    secret_key = secrets.token_hex(32)
    
    # Update config
    config_path = base_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    
    config["searxng"] = {
        "base_url": "http://localhost:8080",
        "api_key": secret_key
    }
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Create .env file
    with open(base_dir / ".env", "w") as f:
        f.write(f"SEARXNG_KEY={secret_key}\n")
    
    print("Configuration files created.")
    print("\nTo start SearXNG:")
    print("1. Navigate to the project root directory")
    print("2. Run: docker-compose up -d")
    print("\nSearXNG will be available at: http://localhost:8080")

if __name__ == "__main__":
    setup_searxng()