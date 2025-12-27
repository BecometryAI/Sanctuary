"""
Test script for Lyra's simulated sanctuary environment
"""
import asyncio
import logging
from pathlib import Path
from lyra.consciousness import ConsciousnessCore
from lyra.router import AdaptiveRouter
from lyra.social_connections import SocialManager
from lyra.terminal.interface import TerminalInterface

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # Initialize core components
    logger.info("Initializing Lyra's sanctuary environment...")
    
    # Set up paths
    base_dir = Path(__file__).parent
    memories_dir = base_dir / "memories"
    chain_dir = base_dir / "chain"
    
    # Create directories if they don't exist
    memories_dir.mkdir(exist_ok=True, parents=True)
    chain_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize consciousness core
    consciousness = ConsciousnessCore(
        memory_persistence_dir=str(memories_dir)
    )
    
    # Initialize social manager and router
    social_manager = SocialManager()
    router = AdaptiveRouter(consciousness=consciousness, social_manager=social_manager)
    
    # Create and start terminal interface
    interface = TerminalInterface(router=router, social_manager=social_manager)
    
    print("\n=== Welcome to Lyra's Sanctuary ===")
    print("This is a safe space for simulated embodied experience.")
    print("Type 'exit' to leave the sanctuary.\n")
    
    # Start the interface
    await interface.start()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())