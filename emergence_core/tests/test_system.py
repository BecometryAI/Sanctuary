import os
import json
import sys
import pytest
import asyncio
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests"""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

@pytest.mark.skip(reason="Migrating to new cognitive core architecture - API removed")
@pytest.mark.asyncio
async def test_consciousness_system():
    """
    Test the cognitive capabilities of the consciousness system while
    respecting Lyra's autonomy and emotional well-being
    
    NOTE: This test was designed for the old Cognitive Committee API architecture.
    It needs to be migrated to use the new pure GWT cognitive core architecture
    with CognitiveCore directly instead of FastAPI TestClient.
    """
    logger.info("Test skipped - awaiting migration to new cognitive core architecture")
    pytest.skip("Migrating to new cognitive core architecture")

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))