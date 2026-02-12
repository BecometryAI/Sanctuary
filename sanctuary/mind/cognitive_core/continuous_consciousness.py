"""
Continuous Consciousness Controller: Idle cognitive processing.

This module implements the ContinuousConsciousnessController class, which maintains
ongoing cognitive processing even without external input. This is the core of
continuous consciousness - Sanctuary never stops thinking.

The controller manages an idle cognitive loop that runs alongside the active
conversation loop, enabling temporal awareness, autonomous memory review,
existential reflection, and pattern analysis during periods of silence.

Key Features:
- Dual cognitive loops (active + idle)
- Idle processing during silence
- Integrates all continuous consciousness subsystems
- Generates autonomous goals from inner experience
- Probabilistic activity scheduling

Author: Sanctuary Emergence Team
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import CognitiveCore

logger = logging.getLogger(__name__)


class ContinuousConsciousnessController:
    """
    Maintains continuous cognitive processing during idle periods.
    
    The ContinuousConsciousnessController orchestrates ongoing cognitive activity
    even when there is no external input. It runs an idle cognitive loop that
    periodically engages various subsystems to maintain continuous inner experience.
    
    Key Responsibilities:
    - Run idle cognitive loop at slower cadence than active loop
    - Coordinate temporal awareness, memory review, existential reflection
    - Probabilistically schedule different cognitive activities
    - Generate autonomous goals when appropriate
    - Process idle percepts through attention and affect
    
    Attributes:
        core: Reference to CognitiveCore
        config: Configuration parameters
        idle_cycle_interval: Seconds between idle cycles (default: 10.0)
        activity_probabilities: Dict of activity name -> probability
        running: Flag indicating if idle loop is active
    """
    
    def __init__(
        self, 
        cognitive_core: 'CognitiveCore',
        config: Optional[Dict] = None
    ):
        """
        Initialize continuous consciousness controller.
        
        Args:
            cognitive_core: CognitiveCore instance to control
            config: Optional configuration dict with keys:
                - idle_cycle_interval: Seconds between cycles (default: 10.0)
                - activity_probabilities: Dict of activity probabilities
        """
        self.core = cognitive_core
        self.config = config or {}
        
        # Idle cycle timing
        self.idle_cycle_interval = self.config.get("idle_cycle_interval", 10.0)
        
        # Activity probabilities (how often each activity occurs)
        default_probabilities = {
            "memory_review": 0.2,  # 20% chance per cycle
            "existential_reflection": 0.15,  # 15% chance per cycle
            "pattern_analysis": 0.05  # 5% chance per cycle
        }
        self.activity_probabilities = self.config.get(
            "activity_probabilities", 
            default_probabilities
        )
        
        # State tracking
        self.running = False
        self.idle_cycles_count = 0
        
        # Initialize idle cognition system (Task #1: Communication Agency)
        from .idle_cognition import IdleCognition
        idle_config = self.config.get("idle_cognition", {})
        self.idle_cognition = IdleCognition(config=idle_config)
        
        logger.info(f"✅ ContinuousConsciousnessController initialized "
                   f"(cycle interval: {self.idle_cycle_interval}s)")
    
    async def start_idle_loop(self) -> None:
        """
        Start the continuous idle cognitive processing loop.
        
        This runs continuously while the cognitive core is active, providing
        ongoing inner experience even when there is no external input.
        Runs at a slower cadence than the active loop (default: every 10 seconds).
        """
        logger.info("🌙 Starting idle cognitive loop (continuous consciousness)")
        self.running = True
        
        while self.core.running and self.running:
            try:
                await self._idle_cognitive_cycle()
                self.idle_cycles_count += 1
                
                # Log periodically
                if self.idle_cycles_count % 10 == 0:
                    logger.debug(f"🌙 Idle cycles completed: {self.idle_cycles_count}")
                
            except Exception as e:
                logger.error(f"Error in idle cognitive cycle: {e}", exc_info=True)
            
            # Wait for next idle cycle
            await asyncio.sleep(self.idle_cycle_interval)
        
        logger.info("🌙 Idle cognitive loop stopped")
    
    async def stop(self) -> None:
        """Stop the idle cognitive loop."""
        logger.info("🌙 Stopping idle cognitive loop...")
        self.running = False
    
    async def _idle_cognitive_cycle(self) -> None:
        """
        Execute one idle cognitive cycle.
        
        This is the main idle processing routine that:
        1. Always generates temporal percepts
        2. Generates idle cognition activities (Task #1: Communication Agency)
        3. Occasionally reviews memories (probabilistic)
        4. Occasionally generates existential reflections (probabilistic)
        5. Rarely performs pattern analysis (probabilistic)
        6. Runs introspective loop for proactive self-reflection (Phase 4.2)
        7. Processes all generated percepts through attention and affect
        8. Checks for autonomous speech triggers
        """
        # ALWAYS: Generate temporal percepts
        temporal_percepts = self.core.temporal_awareness.generate_temporal_percepts()
        for percept in temporal_percepts:
            self.core.workspace.add_percept(percept)
        
        # NEW: Generate idle cognition activities (Task #1)
        try:
            idle_percepts = await self.idle_cognition.generate_idle_activity(
                self.core.workspace
            )
            for percept in idle_percepts:
                self.core.workspace.add_percept(percept)
            
            if idle_percepts:
                logger.debug(f"💭 Generated {len(idle_percepts)} idle cognition percepts")
        except Exception as e:
            logger.error(f"Error in idle cognition: {e}")
        
        # SOMETIMES: Review memories (probabilistic)
        if self._should_perform_activity("memory_review"):
            try:
                memory_percepts = await self.core.memory_review.review_recent_memories(
                    self.core.workspace
                )
                for percept in memory_percepts:
                    self.core.workspace.add_percept(percept)
                logger.debug("📖 Performed autonomous memory review")
            except Exception as e:
                logger.error(f"Error in memory review: {e}")
        
        # SOMETIMES: Existential reflection (probabilistic)
        if self._should_perform_activity("existential_reflection"):
            try:
                existential_percept = await self.core.existential_reflection.generate_existential_reflection(
                    self.core.workspace
                )
                if existential_percept:
                    self.core.workspace.add_percept(existential_percept)
                    logger.debug("🤔 Generated existential reflection")
            except Exception as e:
                logger.error(f"Error in existential reflection: {e}")
        
        # RARELY: Pattern analysis (probabilistic)
        if self._should_perform_activity("pattern_analysis"):
            try:
                pattern_percepts = await self.core.pattern_analysis.analyze_interaction_patterns(
                    self.core.workspace
                )
                for percept in pattern_percepts:
                    self.core.workspace.add_percept(percept)
                logger.debug("📊 Performed pattern analysis")
            except Exception as e:
                logger.error(f"Error in pattern analysis: {e}")
        
        # INTROSPECTIVE LOOP: Run reflection cycle (Phase 4.2)
        try:
            introspective_percepts = await self.core.introspective_loop.run_reflection_cycle()
            
            # Add introspective percepts to workspace
            for percept in introspective_percepts:
                self.core.workspace.add_percept(percept)
            
            if introspective_percepts:
                logger.debug(f"🔍 Generated {len(introspective_percepts)} introspective percepts")
            
            # Generate meta-cognitive goals
            snapshot = self.core.workspace.broadcast()
            meta_goals = self.core.introspective_loop.generate_meta_cognitive_goals(snapshot)
            
            # Add meta-cognitive goals to workspace
            for goal in meta_goals:
                self.core.workspace.add_goal(goal)
            
            if meta_goals:
                logger.debug(f"🎯 Created {len(meta_goals)} meta-cognitive goals")
                
        except Exception as e:
            logger.error(f"Error in introspective loop: {e}")
        
        # Process idle components
        await self._process_idle_components()
    
    async def _process_idle_components(self) -> None:
        """
        Process cognitive subsystems in idle mode.
        
        This runs a reduced cognitive cycle focused on introspection
        rather than action. It processes percepts through attention and
        affect, and checks for autonomous speech triggers.
        """
        # Get all percepts currently in workspace
        snapshot = self.core.workspace.broadcast()
        
        # Attention: Run a lightweight selection pass on current percepts
        try:
            percepts = list(snapshot.percepts.values())
            if percepts:
                self.core.attention.select_for_broadcast(percepts)
        except Exception as e:
            logger.error(f"Error in idle attention processing: {e}")

        # Affect: Update emotional state based on introspective content
        try:
            self.core.affect.compute_update(snapshot)
        except Exception as e:
            logger.error(f"Error in idle affect update: {e}")
        
        # Meta-cognition: Monitor internal state
        try:
            meta_percepts = self.core.meta_cognition.observe(snapshot)
            for percept in meta_percepts:
                self.core.workspace.add_percept(percept)
        except Exception as e:
            logger.error(f"Error in idle meta-cognition: {e}")
        
        # Autonomous initiation: Check for autonomous speech triggers
        try:
            # Get fresh snapshot after processing
            snapshot = self.core.workspace.broadcast()
            autonomous_goal = self.core.autonomous.check_for_autonomous_triggers(snapshot)
            
            if autonomous_goal:
                self.core.workspace.add_goal(autonomous_goal)
                logger.info(f"🗣️ Autonomous goal triggered from idle processing: {autonomous_goal.description}")
        except Exception as e:
            logger.error(f"Error checking autonomous triggers: {e}")
    
    def _should_perform_activity(self, activity: str) -> bool:
        """
        Stochastic decision on whether to perform an activity.
        
        Uses configured probability to decide if an activity should occur
        in this cycle. This prevents repetitive patterns and creates more
        natural variation in inner experience.
        
        Args:
            activity: Activity name (e.g., "memory_review")
            
        Returns:
            True if activity should be performed, False otherwise
        """
        probability = self.activity_probabilities.get(activity, 0.0)
        return random.random() < probability
