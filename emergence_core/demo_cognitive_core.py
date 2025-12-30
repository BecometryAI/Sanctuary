"""
Demonstration script for the CognitiveCore.

This script shows how to:
1. Initialize the CognitiveCore
2. Inject percepts
3. Run the cognitive loop for a few cycles
4. Query the state
5. Get performance metrics
6. Gracefully shut down
"""
import asyncio
import logging
from lyra.cognitive_core.core import CognitiveCore
from lyra.cognitive_core.workspace import GlobalWorkspace, Percept, Goal, GoalType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo():
    """Demonstrate the CognitiveCore functionality."""
    logger.info("=" * 80)
    logger.info("CognitiveCore Demonstration")
    logger.info("=" * 80)
    
    # 1. Initialize
    logger.info("\n1. Initializing GlobalWorkspace and CognitiveCore...")
    workspace = GlobalWorkspace(capacity=10)
    core = CognitiveCore(workspace, config={"cycle_rate_hz": 10})
    
    # 2. Add a goal
    logger.info("\n2. Adding a goal to the workspace...")
    goal = Goal(
        type=GoalType.RESPOND_TO_USER,
        description="Process incoming sensory inputs",
        priority=0.8
    )
    workspace.add_goal(goal)
    
    # 3. Start the cognitive loop in background
    logger.info("\n3. Starting the cognitive loop...")
    async def run_core():
        await core.start()
    
    task = asyncio.create_task(run_core())
    
    # 4. Inject some percepts
    logger.info("\n4. Injecting percepts over time...")
    
    await asyncio.sleep(0.15)  # Let it run a cycle
    
    percept1 = Percept(
        modality="text",
        raw="Hello, I am observing the world",
        embedding=[0.1, 0.2, 0.3] * 128,  # 384-dim embedding
        complexity=5
    )
    core.inject_input(percept1)
    logger.info(f"   Injected percept 1: {percept1.raw[:30]}...")
    
    await asyncio.sleep(0.15)  # Let it process
    
    percept2 = Percept(
        modality="text",
        raw="This is a novel input about consciousness",
        embedding=[0.9, 0.1, 0.5] * 128,
        complexity=8
    )
    core.inject_input(percept2)
    logger.info(f"   Injected percept 2: {percept2.raw[:30]}...")
    
    await asyncio.sleep(0.15)
    
    percept3 = Percept(
        modality="text",
        raw="Meta-cognitive observation: system is functioning",
        embedding=[0.3, 0.7, 0.2] * 128,
        complexity=10
    )
    core.inject_input(percept3)
    logger.info(f"   Injected percept 3: {percept3.raw[:30]}...")
    
    # 5. Let it run for a bit
    logger.info("\n5. Running cognitive cycles...")
    await asyncio.sleep(0.5)  # Run ~5 more cycles
    
    # 6. Query state
    logger.info("\n6. Querying current workspace state...")
    snapshot = core.query_state()
    logger.info(f"   Current cycle: {snapshot.cycle_count}")
    logger.info(f"   Active goals: {len(snapshot.goals)}")
    logger.info(f"   Active percepts: {len(snapshot.percepts)}")
    logger.info(f"   Emotional state: valence={snapshot.emotions.get('valence', 0):.2f}, "
               f"arousal={snapshot.emotions.get('arousal', 0):.2f}")
    
    # 7. Get metrics
    logger.info("\n7. Getting performance metrics...")
    metrics = core.get_metrics()
    logger.info(f"   Total cycles: {metrics['total_cycles']}")
    logger.info(f"   Avg cycle time: {metrics['avg_cycle_time_ms']:.2f}ms "
               f"(target: {metrics['target_cycle_time_ms']:.0f}ms)")
    logger.info(f"   Cycle rate: {metrics['cycle_rate_hz']}Hz")
    logger.info(f"   Percepts processed: {metrics['percepts_processed']}")
    logger.info(f"   Attention selections: {metrics['attention_selections']}")
    logger.info(f"   Workspace size: {metrics['workspace_size']}")
    
    # 8. Graceful shutdown
    logger.info("\n8. Shutting down gracefully...")
    await core.stop()
    
    # Wait for task to complete
    await asyncio.sleep(0.1)
    
    logger.info("\n" + "=" * 80)
    logger.info("Demonstration complete!")
    logger.info("=" * 80)
    logger.info("\nKey observations:")
    logger.info("- The cognitive loop ran continuously at ~10 Hz")
    logger.info("- Percepts were injected asynchronously and processed in cycles")
    logger.info("- Attention selected percepts based on complexity and salience")
    logger.info("- Workspace maintained state across cycles")
    logger.info("- System shut down gracefully with all metrics logged")


if __name__ == "__main__":
    asyncio.run(demo())
