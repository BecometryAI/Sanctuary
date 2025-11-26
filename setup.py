"""
Setup file for Lyra Emergence package
"""
from setuptools import setup, find_packages

#TODO switch docker to async friendly ver (think aiodocker), Switch to `Quart` from uvicorn/fastapi
#TODO switch from discord to `interactions`
setup(
    name="lyra-emergence",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "emergence_core"},
    install_requires=[
        "numpy>=1.24.0",
        "chromadb>=0.4.15",
        "pydantic>=2.5.2",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-dotenv>=1.0.0",
        "langchain>=0.0.325",
        "sentence-transformers>=2.2.2",
        "torch>=2.1.0",
        "transformers>=4.34.0",
        "accelerate>=0.28.0",
        "bitsandbytes>=0.41.0",
        "schedule>=1.2.0",
        "discord.py>=2.3.2",
        "aiohttp>=3.9.0",
        "asyncio>=3.4.3",
        "httpx>=0.24.1",
        "arxiv>=1.4.7",
        "wikipedia>=1.4.0",
        "wolframalpha>=5.0.0",
        "playwright>=1.39.0",
        "docker>=6.1.3",
        "web3>=6.11.1",
        "eth-account>=0.9.0",
        "ipfs-http-client>=1.0.0"
    ],
    python_requires=">=3.9",
)
