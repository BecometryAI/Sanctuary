"""
Entry point for unified Lyra-Emergence system.

Runs both cognitive core and specialist system together.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add emergence_core to path
sys.path.insert(0, str(Path(__file__).parent))

from lyra.unified_core import UnifiedCognitiveCore


async def main():
    """Main entry point for unified system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    base_dir = str(Path(__file__).parent)
    chroma_dir = str(Path(base_dir) / "model_cache" / "chroma_db")
    model_dir = str(Path(base_dir) / "model_cache" / "models")
    
    config = {
        "cognitive_core": {
            "cycle_rate_hz": 10,
            "attention_budget": 100
        },
        "specialist_router": {
            "development_mode": False
        },
        "integration": {
            "specialist_threshold": 0.7,  # Priority threshold for specialist routing
            "sync_interval": 1.0  # Seconds between state syncs
        }
    }
    
    # Initialize unified system
    unified = UnifiedCognitiveCore(config=config)
    
    try:
        await unified.initialize(base_dir, chroma_dir, model_dir)
        
        # Interactive loop
        print("✅ Lyra unified system online. Type 'quit' to exit.")
        print()
        
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() in ["quit", "exit"]:
                    break
                
                if not user_input.strip():
                    continue
                
                response = await unified.process_user_input(user_input)
                print(f"Lyra: {response}")
                print()
                
            except KeyboardInterrupt:
                break
            except EOFError:
                break
        
    finally:
        await unified.stop()
        print("👋 Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Goodbye!")
