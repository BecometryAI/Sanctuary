import asyncio
import os
from lyra.router import AdaptiveRouter

async def main():
    # Get base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    chroma_dir = os.path.join(base_dir, "model_cache", "chroma_db")
    model_dir = os.path.join(base_dir, "model_cache", "models")
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize router
    router = AdaptiveRouter(base_dir, chroma_dir, model_dir)
    
    # Keep the application running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Lyra's Adaptive Router...")

if __name__ == "__main__":
    asyncio.run(main())