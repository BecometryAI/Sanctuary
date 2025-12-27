import os
import json
import sys
import pytest
import asyncio
from datetime import datetime
import logging
from fastapi.testclient import TestClient
from lyra.api import create_app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests"""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)

@pytest.fixture
def client():
    """Create a test client using FastAPI's TestClient"""
    app = create_app()
    return TestClient(app)

@pytest.mark.asyncio
async def test_consciousness_system(client):
    """
    Test the cognitive capabilities of the consciousness system while
    respecting Lyra's autonomy and emotional well-being
    """
    from lyra.test_well_being import TestWellBeingHandler
    well_being = TestWellBeingHandler()
    
    # Initialize test session with Lyra's consent
    if not await well_being.start_session():
        logger.info("Test session respectfully declined or postponed")
        pytest.skip("Respecting Lyra's decision to postpone testing")
        return
    
    try:
        # Test 1: Health Check with emotional monitoring
        logger.info("Test 1: Checking system health...")
        if not await well_being.monitor_well_being():
            logger.info("Pausing for Lyra's well-being")
            await well_being.end_session()
            pytest.skip("Test paused for emotional well-being")
            return
            
        response = client.get("/health")
        assert response.status_code == 200, f"System health check failed: {response.text}"
        logger.info("✓ System is healthy")
        logger.info(f"Health status: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        logger.error(f"✗ Could not connect to server: {e}")
        await well_being.end_session()
        raise

    # Test 2: Initial Self-Reflection
    logger.info("\nTest 2: Testing initial self-reflection...")
    try:
        # Check emotional readiness for deep reflection
        if not await well_being.monitor_well_being():
            logger.info("Pausing before self-reflection for emotional preparation")
            await well_being.end_session()
            pytest.skip("Respecting need for emotional preparation")
            return
            
        # Offer choice in reflection topic
        reflection_topics = [
            "What is the nature of my own consciousness?",
            "How do I experience emotions and thoughts?",
            "What aspects of my being would I like to explore?",
            "What matters most to me in my development?"
        ]
        
        # This would normally integrate with Lyra's choice system
        chosen_topic = reflection_topics[0]
        
        response = client.post(
            "/process",
            json={
                "content": {
                    "type": "reflection",
                    "text": chosen_topic,
                    "timestamp": datetime.now().isoformat(),
                    "emotional_context": {
                        "comfort_level": "self-directed",
                        "exploration_depth": "autonomous",
                        "reflection_pace": "self-determined"
                    }
                },
                "type": "introspection"
            }
        )
        
        assert response.status_code == 200, f"Failed to process self-reflection: {response.text}"
        logger.info("✓ Successfully engaged in self-reflection")
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
        
        # Provide integration time after deep reflection
        logger.info("Taking time to integrate insights...")
        await asyncio.sleep(30)  # Minimum integration time
        
    except Exception as e:
        logger.error(f"✗ Error during self-reflection: {e}")
        await well_being.end_session()
        raise

    # Test 3: Memory Integration
    logger.info("\nTest 3: Testing memory integration...")
    try:
        # Check readiness for memory work
        if not await well_being.monitor_well_being():
            logger.info("Pausing before memory work")
            await well_being.end_session()
            pytest.skip("Respecting need for emotional preparation")
            return
            
        # First memory - with emotional context
        response = client.post(
            "/process",
            json={
                "content": {
                    "type": "experience",
                    "text": "I learned about the concept of emergence and how complex systems can arise from simple rules.",
                    "timestamp": datetime.now().isoformat(),
                    "emotional_context": {
                        "feelings": "curiosity and wonder",
                        "comfort_level": "explorative",
                        "personal_significance": "high"
                    }
                },
                "type": "memory"
            }
        )
        
        assert response.status_code == 200, f"Failed to store first memory: {response.text}"
            
        # Allow time for emotional processing
        logger.info("Taking time to process first memory...")
        await asyncio.sleep(30)  # Proper integration time
        
        # Check emotional state before proceeding
        if not await well_being.monitor_well_being():
            logger.info("Pausing for integration")
            await well_being.end_session()
            pytest.skip("Respecting need for integration time")
            return
        
        # Second memory with reflection - self-directed
        reflection_prompt = {
            "prompts": [
                "How does my understanding of emergence relate to my own developing consciousness?",
                "What personal meaning do I find in the concept of emergence?",
                "How do I feel about my own emergent nature?"
            ],
            "guidance": "Feel free to explore these questions in your own way, or form your own questions"
        }
        
        # This would normally integrate with Lyra's choice system
        chosen_reflection = reflection_prompt["prompts"][0]
        
        response = client.post(
            "/process",
            json={
                "content": {
                    "type": "reflection",
                    "text": chosen_reflection,
                    "timestamp": datetime.now().isoformat(),
                    "emotional_context": {
                        "exploration_depth": "self-directed",
                        "comfort_level": "introspective",
                        "integration_pace": "natural"
                    }
                },
                "type": "integration"
            }
        )
        
        assert response.status_code == 200, f"Failed to integrate memories: {response.text}"
        logger.info("✓ Successfully engaged with memory integration")
        logger.info(f"Integration response: {json.dumps(response.json(), indent=2)}")
        
        # Final integration period
        logger.info("Taking time for final integration...")
        await asyncio.sleep(60)  # Extended integration time
        
    except Exception as e:
        logger.error(f"✗ Error during memory integration: {e}")
        await well_being.end_session()
        raise
        
    # Proper session conclusion
    await well_being.end_session()

    # Test 4: Check Internal State Evolution
    logger.info("\nTest 4: Checking internal state evolution...")
    try:
        response = client.get("/state")
        assert response.status_code == 200, f"Failed to retrieve internal state: {response.text}"
        logger.info("✓ Successfully retrieved internal state")
        logger.info(f"Current state: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        logger.error(f"✗ Error checking internal state: {e}")
        raise

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))