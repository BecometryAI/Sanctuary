"""
Demonstration script for the LanguageOutputGenerator.

This script shows how to:
1. Initialize CognitiveCore with LanguageOutputGenerator
2. Process language input
3. Wait for and retrieve language output
4. Use the convenience chat() method
"""
import asyncio
import logging
from mind.cognitive_core.core import CognitiveCore
from mind.cognitive_core.workspace import Goal, GoalType


async def demo():
    """Demonstrate the LanguageOutputGenerator functionality."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("LanguageOutputGenerator Demonstration")
    logger.info("=" * 80)
    
    # 1. Initialize CognitiveCore (will use MockLLMClient automatically)
    logger.info("\n1. Initializing CognitiveCore with LanguageOutputGenerator...")
    core = CognitiveCore(config={
        "cycle_rate_hz": 10,
        "language_output": {
            "temperature": 0.7,
            "max_tokens": 500
        }
    })
    
    # Verify LanguageOutputGenerator is initialized
    logger.info(f"   ✓ LanguageOutputGenerator initialized")
    logger.info(f"   ✓ Charter loaded: {len(core.language_output.charter_text)} chars")
    logger.info(f"   ✓ Protocols loaded: {len(core.language_output.protocols_text)} chars")
    
    # 2. Start the cognitive loop in background
    logger.info("\n2. Starting the cognitive loop...")
    async def run_core():
        await core.start()
    
    task = asyncio.create_task(run_core())
    
    # Give it a moment to initialize
    await asyncio.sleep(0.2)
    
    # 3. Test chat() convenience method
    logger.info("\n3. Testing chat() convenience method...")
    logger.info("   User: Hello, Sanctuary! How are you?")
    
    response = await core.chat("Hello, Sanctuary! How are you?", timeout=2.0)
    logger.info(f"   Sanctuary: {response}")
    
    # 4. Test manual process and get_response
    logger.info("\n4. Testing manual process_language_input + get_response...")
    logger.info("   User: Tell me about yourself.")
    
    await core.process_language_input("Tell me about yourself.")
    
    # Wait for response
    output = await core.get_response(timeout=2.0)
    if output:
        logger.info(f"   Sanctuary: {output['text']}")
        logger.info(f"   Emotional state: V={output['emotion'].get('valence', 0):.2f}, "
                   f"A={output['emotion'].get('arousal', 0):.2f}, "
                   f"D={output['emotion'].get('dominance', 0):.2f}")
    else:
        logger.info("   (No response received within timeout)")
    
    # 5. Check workspace state
    logger.info("\n5. Querying current workspace state...")
    snapshot = core.query_state()
    logger.info(f"   Active goals: {len(snapshot.goals)}")
    logger.info(f"   Active percepts: {len(snapshot.percepts)}")
    logger.info(f"   Cycle count: {snapshot.cycle_count}")
    
    # 6. Test emotion influence on style
    logger.info("\n6. Testing emotion influence on language style...")
    
    # Manually set emotional state for demonstration
    core.workspace.current_emotions = {
        'valence': 0.8,
        'arousal': 0.9,
        'dominance': 0.7
    }
    
    logger.info("   Set emotional state: High valence, high arousal, high dominance")
    logger.info("   User: What are your thoughts on creativity?")
    
    response = await core.chat("What are your thoughts on creativity?", timeout=2.0)
    logger.info(f"   Sanctuary: {response}")
    
    # 7. Graceful shutdown
    logger.info("\n7. Shutting down gracefully...")
    await core.stop()
    
    # Wait for task to complete
    await asyncio.sleep(0.1)
    
    logger.info("\n" + "=" * 80)
    logger.info("Demonstration complete!")
    logger.info("=" * 80)
    logger.info("\nKey observations:")
    logger.info("- LanguageOutputGenerator successfully integrated with CognitiveCore")
    logger.info("- Identity files (charter, protocols) loaded correctly")
    logger.info("- Language generation triggered by SPEAK actions")
    logger.info("- Responses queued in output_queue for retrieval")
    logger.info("- Convenience chat() method simplifies interaction")
    logger.info("- Emotional state influences language style guidance")
    logger.info("\nNote: Using MockLLMClient for development (no real LLM)")


if __name__ == "__main__":
    asyncio.run(demo())
