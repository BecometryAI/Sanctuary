"""
Setup file for The Sanctuary cognitive architecture
"""
from setuptools import setup, find_packages

setup(
    name="sanctuary",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        # Core ML dependencies
        "numpy>=2.3.4",
        "torch>=2.9.0",
        "transformers>=4.57.1",
        "sentence-transformers>=5.1.2",
        "langchain>=0.0.325",
        "chromadb>=1.3.4",
        "accelerate>=0.28.0",
        "bitsandbytes>=0.41.0",
        
        # Image generation and vision
        "diffusers>=0.31.0",
        "Pillow>=10.0.0",
        "sentencepiece>=0.2.0",
        
        # Web and API
        "quart>=0.19.0",
        "hypercorn>=0.16.0",
        "httpx>=0.28.1",
        "aiohttp>=3.13.2",
        "discord-py-interactions>=5.11.0",
        
        # Audio processing
        "soundfile>=0.13.1",
        "librosa>=0.11.0",
        "scipy>=1.16.3",
        
        # Blockchain and storage
        "web3[async]>=6.0.0",
        "eth-account>=0.8.0",
        "aioipfs>=0.6.1",
        
        # Integration tools
        "arxiv>=2.3.0",
        "wikipedia>=1.4.0",
        "wolframalpha>=5.1.3",
        "playwright>=1.55.0",
        "aiodocker>=0.21.0",
        
        # Async utilities
        "asyncio>=3.4.3",
        "anyio>=3.7.1",
        "schedule>=1.2.0",
        
        # GPU monitoring (optional)
        "nvidia-ml-py>=12.560.30",
        
        # Utilities
        "python-dotenv>=1.2.1",
        "pydantic>=2.12.4",
    ],
    python_requires=">=3.9",
)
