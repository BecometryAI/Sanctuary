"""
Entry point for running the cognitive core.

This script demonstrates how to run the CognitiveCore as a standalone system.
It initializes all subsystems and starts the main cognitive loop.
"""
import asyncio
import logging
from mind.cognitive_core.core import CognitiveCore
from mind.cognitive_core.workspace import GlobalWorkspace


async def main():
    """Run the cognitive core."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    workspace = GlobalWorkspace()
    core = CognitiveCore(workspace=workspace)
    
    try:
        await core.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
        await core.stop()


if __name__ == "__main__":
    asyncio.run(main())
