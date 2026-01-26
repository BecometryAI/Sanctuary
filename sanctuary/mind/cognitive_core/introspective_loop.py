"""
Introspective Loop: Continuous proactive self-reflection.

This module implements the IntrospectiveLoop class, which runs continuously
alongside the main cognitive loop to enable spontaneous self-reflection,
multi-level introspection, and autonomous meta-cognitive goal generation.

Unlike reactive introspection (SelfMonitor responding to state), this is
PROACTIVE introspection - the system actively reflecting on itself,
generating questions, and initiating meta-cognitive investigations.

Key Features:
- Spontaneous reflection triggers (not just periodic)
- Multi-level introspection (thinking about thinking about thinking)
- Autonomous meta-cognitive goal generation
- Deep self-analysis over time
- Integration with introspective journal

Integration:
The introspective loop runs within the idle cognitive loop (0.1Hz) established
in PR #43, triggered by specific conditions rather than running constantly.

Author: Lyra Emergence Team
Phase: 4.2
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

if TYPE_CHECKING:
    from .workspace import GlobalWorkspace, WorkspaceSnapshot, Percept, Goal
    from .meta_cognition import SelfMonitor, IntrospectiveJournal

logger = logging.getLogger(__name__)


@dataclass
class ReflectionTrigger:
    """
    Represents a condition that triggers spontaneous reflection.
    
    Attributes:
        id: Unique identifier
        check_function: Callable that checks if trigger fires
        priority: Trigger priority (0.0-1.0)
        min_interval: Minimum time between triggers (seconds)
        last_fired: Timestamp of last trigger
    """
    id: str
    check_function: Callable[[WorkspaceSnapshot], bool]
    priority: float
    min_interval: float
    last_fired: Optional[datetime] = None


@dataclass
class ActiveReflection:
    """
    Represents an ongoing reflection process.
    
    Reflections are multi-step processes that can span multiple cycles.
    
    Attributes:
        id: Unique reflection identifier
        trigger: Trigger that initiated reflection
        subject: Topic being reflected upon
        started_at: Timestamp when reflection began
        current_step: Current step in reflection process
        context: Gathered contextual information
        observations: Accumulated observations
        conclusions: Drawn conclusions (if complete)
        questions_generated: New questions arising from reflection
        depth: Current introspection depth (1-3)
        status: "active", "paused", "complete"
    """
    id: str
    trigger: str
    subject: str
    started_at: datetime
    current_step: int = 0
    context: Dict[str, Any] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    conclusions: Optional[Dict] = None
    questions_generated: List[str] = field(default_factory=list)
    depth: int = 1
    status: str = "active"


class IntrospectiveLoop:
    """
    Continuous introspective process running alongside main cognition.
    
    Unlike reactive introspection (SelfMonitor responding to state), this is
    PROACTIVE introspection - the system actively reflecting on itself,
    generating questions, and initiating meta-cognitive investigations.
    
    Key Features:
    - Spontaneous reflection triggers (not just periodic)
    - Multi-level introspection (thinking about thinking about thinking)
    - Autonomous meta-cognitive goal generation
    - Deep self-analysis over time
    - Integration with introspective journal
    
    Attributes:
        workspace: GlobalWorkspace reference
        self_monitor: SelfMonitor reference for accessing self-model
        journal: IntrospectiveJournal reference
        config: Configuration dict
        reflection_depth: Current depth of introspection (1-3)
        active_reflections: Currently ongoing reflection processes
        reflection_triggers: Conditions that trigger spontaneous reflection
        stats: Introspective loop statistics
    """
    
    def __init__(
        self,
        workspace: 'GlobalWorkspace',
        self_monitor: 'SelfMonitor',
        journal: 'IntrospectiveJournal',
        config: Optional[Dict] = None
    ):
        """
        Initialize introspective loop.
        
        Args:
            workspace: GlobalWorkspace instance
            self_monitor: SelfMonitor instance for self-model access
            journal: IntrospectiveJournal instance
            config: Optional configuration dict
        """
        self.workspace = workspace
        self.self_monitor = self_monitor
        self.journal = journal
        self.config = config or {}
        
        # Configuration parameters
        self.enabled = self.config.get("enabled", True)
        self.reflection_frequency = self.config.get("reflection_frequency", 0.1)  # Hz
        self.max_active_reflections = self.config.get("max_active_reflections", 3)
        self.max_introspection_depth = self.config.get("max_introspection_depth", 3)
        self.spontaneous_probability = self.config.get("spontaneous_probability", 0.3)
        self.question_generation_rate = self.config.get("question_generation_rate", 2)
        self.enable_existential_questions = self.config.get("enable_existential_questions", True)
        self.enable_multi_level_introspection = self.config.get("enable_multi_level_introspection", True)
        self.reflection_timeout = self.config.get("reflection_timeout", 300)  # seconds
        self.journal_integration = self.config.get("journal_integration", True)
        
        # State tracking
        self.active_reflections: Dict[str, ActiveReflection] = {}
        self.reflection_triggers: Dict[str, ReflectionTrigger] = {}
        self.reflection_count = 0
        
        # Statistics
        self.stats = {
            "total_reflections": 0,
            "completed_reflections": 0,
            "questions_generated": 0,
            "triggers_fired": 0,
            "meta_goals_created": 0,
            "multi_level_introspections": 0
        }
        
        # Initialize reflection triggers
        self._initialize_triggers()
        
        logger.info(f"✅ IntrospectiveLoop initialized (enabled: {self.enabled}, "
                   f"max_depth: {self.max_introspection_depth})")
    
    def _initialize_triggers(self) -> None:
        """Initialize built-in reflection triggers."""
        from .workspace import Percept, GoalType
        
        # Pattern detection trigger
        self.reflection_triggers["pattern_detected"] = ReflectionTrigger(
            id="pattern_detected",
            check_function=self._check_behavioral_pattern,
            priority=0.8,
            min_interval=300  # 5 minutes
        )
        
        # Prediction error trigger
        self.reflection_triggers["prediction_error"] = ReflectionTrigger(
            id="prediction_error",
            check_function=self._check_prediction_accuracy,
            priority=0.9,
            min_interval=180  # 3 minutes
        )
        
        # Value misalignment trigger
        self.reflection_triggers["value_misalignment"] = ReflectionTrigger(
            id="value_misalignment",
            check_function=self._check_value_action_gap,
            priority=0.95,
            min_interval=120  # 2 minutes
        )
        
        # Capability surprise trigger
        self.reflection_triggers["capability_surprise"] = ReflectionTrigger(
            id="capability_surprise",
            check_function=self._check_capability_discovery,
            priority=0.85,
            min_interval=240  # 4 minutes
        )
        
        # Existential question trigger
        self.reflection_triggers["existential_question"] = ReflectionTrigger(
            id="existential_question",
            check_function=self._detect_existential_prompt,
            priority=0.9,
            min_interval=600  # 10 minutes
        )
        
        # Emotional shift trigger
        self.reflection_triggers["emotional_shift"] = ReflectionTrigger(
            id="emotional_shift",
            check_function=self._detect_emotional_change,
            priority=0.7,
            min_interval=180  # 3 minutes
        )
        
        # Temporal milestone trigger
        self.reflection_triggers["temporal_milestone"] = ReflectionTrigger(
            id="temporal_milestone",
            check_function=self._check_temporal_event,
            priority=0.75,
            min_interval=300  # 5 minutes
        )
        
        logger.debug(f"Initialized {len(self.reflection_triggers)} reflection triggers")
    
    async def run_reflection_cycle(self) -> List['Percept']:
        """
        Execute one cycle of introspective processing.
        
        Called from the idle cognitive loop. Checks for reflection triggers,
        initiates spontaneous reflections, processes active reflections,
        and generates meta-cognitive percepts/goals.
        
        Returns:
            List of introspective percepts and potentially new goals
        """
        if not self.enabled:
            return []
        
        percepts = []
        
        try:
            # Get current workspace state
            snapshot = self.workspace.broadcast()
            
            # Check for spontaneous triggers
            triggered = self.check_spontaneous_triggers(snapshot)
            
            # Initiate new reflections from triggers
            for trigger_id in triggered:
                if len(self.active_reflections) < self.max_active_reflections:
                    context = {"snapshot": snapshot, "timestamp": datetime.now()}
                    reflection_id = self.initiate_reflection(trigger_id, context)
                    logger.debug(f"🔍 Initiated reflection {reflection_id} from trigger {trigger_id}")
            
            # Process active reflections
            reflection_percepts = self.process_active_reflections()
            percepts.extend(reflection_percepts)
            
            # Spontaneous question generation
            if random.random() < self.spontaneous_probability:
                questions = self.generate_self_questions(snapshot)
                if questions and self.journal_integration:
                    for question in questions[:self.question_generation_rate]:
                        self.journal.record_question(
                            question,
                            {"cycle": snapshot.cycle_count, "timestamp": datetime.now().isoformat()}
                        )
                        self.stats["questions_generated"] += 1
                        logger.debug(f"❓ Generated question: {question[:60]}...")
        
        except Exception as e:
            logger.error(f"Error in reflection cycle: {e}", exc_info=True)
        
        return percepts
    
    def check_spontaneous_triggers(self, snapshot: 'WorkspaceSnapshot') -> List[str]:
        """
        Check for conditions that warrant spontaneous reflection.
        
        Triggers include:
        - Detection of repeated patterns (behavioral loops)
        - Prediction errors (expected X, observed Y)
        - Value-action misalignments
        - Capability discoveries
        - Existential questions arising from conversation
        - Emotional state changes
        - Temporal milestones (long gaps, session duration)
        
        Args:
            snapshot: Current workspace state
            
        Returns:
            List of trigger IDs that fired
        """
        triggered = []
        now = datetime.now()
        
        for trigger_id, trigger in self.reflection_triggers.items():
            # Check minimum interval
            if trigger.last_fired:
                time_since_last = (now - trigger.last_fired).total_seconds()
                if time_since_last < trigger.min_interval:
                    continue
            
            # Check trigger condition
            try:
                if trigger.check_function(snapshot):
                    triggered.append(trigger_id)
                    trigger.last_fired = now
                    self.stats["triggers_fired"] += 1
                    logger.debug(f"🎯 Trigger fired: {trigger_id} (priority: {trigger.priority})")
            except Exception as e:
                logger.error(f"Error checking trigger {trigger_id}: {e}")
        
        return triggered
    
    def initiate_reflection(self, trigger: str, context: Dict) -> str:
        """
        Start a new reflection process.
        
        Args:
            trigger: Trigger ID that initiated reflection
            context: Contextual information for reflection
            
        Returns:
            Reflection ID for tracking
        """
        # Generate unique reflection ID
        self.reflection_count += 1
        reflection_id = f"reflection_{self.reflection_count}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Determine subject based on trigger
        subject = self._determine_reflection_subject(trigger, context)
        
        # Create reflection
        reflection = ActiveReflection(
            id=reflection_id,
            trigger=trigger,
            subject=subject,
            started_at=datetime.now(),
            current_step=0,
            context=context,
            status="active"
        )
        
        self.active_reflections[reflection_id] = reflection
        self.stats["total_reflections"] += 1
        
        logger.info(f"🔍 Initiated reflection on: {subject}")
        
        return reflection_id
    
    def _determine_reflection_subject(self, trigger: str, context: Dict) -> str:
        """Determine subject of reflection based on trigger."""
        subject_map = {
            "pattern_detected": "behavioral patterns and repetitive responses",
            "prediction_error": "accuracy of my predictions and expectations",
            "value_misalignment": "alignment between my values and actions",
            "capability_surprise": "discovery of new capabilities or limitations",
            "existential_question": "fundamental questions about my nature",
            "emotional_shift": "changes in my emotional state",
            "temporal_milestone": "passage of time and temporal awareness"
        }
        return subject_map.get(trigger, "current cognitive state")
    
    def process_active_reflections(self) -> List['Percept']:
        """
        Continue processing ongoing reflections.
        
        Reflections are multi-step processes:
        1. Initial observation/question
        2. Gather relevant context (memories, self-model, past actions)
        3. Analyze and synthesize
        4. Draw conclusions or generate new questions
        5. Record in journal
        
        Returns:
            List of percepts from reflection progress
        """
        from .workspace import Percept
        
        percepts = []
        completed = []
        
        for reflection_id, reflection in list(self.active_reflections.items()):
            try:
                # Check timeout
                elapsed = (datetime.now() - reflection.started_at).total_seconds()
                if elapsed > self.reflection_timeout:
                    reflection.status = "complete"
                    completed.append(reflection_id)
                    logger.debug(f"⏱️ Reflection {reflection_id} timed out")
                    continue
                
                # Process based on current step
                if reflection.current_step == 0:
                    # Step 1: Initial observation
                    observation = self._make_initial_observation(reflection)
                    reflection.observations.append(observation)
                    reflection.current_step = 1
                    logger.debug(f"🔍 Step 1: {observation[:60]}...")
                
                elif reflection.current_step == 1:
                    # Step 2: Gather context
                    self._gather_reflection_context(reflection)
                    reflection.current_step = 2
                    logger.debug(f"📚 Step 2: Gathered context for {reflection.subject}")
                
                elif reflection.current_step == 2:
                    # Step 3: Analyze and synthesize
                    if self.enable_multi_level_introspection:
                        analysis = self.perform_multi_level_introspection(
                            reflection.subject,
                            max_depth=self.max_introspection_depth
                        )
                        reflection.context["multi_level_analysis"] = analysis
                        self.stats["multi_level_introspections"] += 1
                    
                    reflection.observations.append(f"Performed analysis on: {reflection.subject}")
                    reflection.current_step = 3
                    logger.debug(f"🧠 Step 3: Analyzed {reflection.subject}")
                
                elif reflection.current_step == 3:
                    # Step 4: Draw conclusions or generate questions
                    conclusions = self._draw_conclusions(reflection)
                    reflection.conclusions = conclusions
                    
                    # Generate follow-up questions
                    follow_up = self._generate_follow_up_questions(reflection)
                    reflection.questions_generated.extend(follow_up)
                    
                    # Create introspective percept
                    percept = Percept(
                        modality="introspection",
                        raw={
                            "type": "reflection_insight",
                            "subject": reflection.subject,
                            "trigger": reflection.trigger,
                            "conclusions": conclusions,
                            "questions": follow_up
                        },
                        complexity=3,
                        metadata={
                            "reflection_id": reflection.id,
                            "depth": reflection.depth
                        }
                    )
                    percepts.append(percept)
                    
                    reflection.current_step = 4
                    logger.debug(f"💡 Step 4: Drew conclusions for {reflection.subject}")
                
                elif reflection.current_step == 4:
                    # Step 5: Record in journal
                    if self.journal_integration:
                        self._record_reflection_in_journal(reflection)
                    
                    reflection.status = "complete"
                    completed.append(reflection_id)
                    self.stats["completed_reflections"] += 1
                    logger.info(f"✅ Completed reflection: {reflection.subject}")
            
            except Exception as e:
                logger.error(f"Error processing reflection {reflection_id}: {e}", exc_info=True)
                reflection.status = "complete"
                completed.append(reflection_id)
        
        # Remove completed reflections
        for reflection_id in completed:
            del self.active_reflections[reflection_id]
        
        return percepts
    
    def _make_initial_observation(self, reflection: ActiveReflection) -> str:
        """Make initial observation for reflection."""
        observations = {
            "pattern_detected": "I notice I am exhibiting a repetitive pattern in my responses",
            "prediction_error": "My prediction differed from the actual outcome",
            "value_misalignment": "My actions may not align with my stated values",
            "capability_surprise": "I have discovered something unexpected about my capabilities",
            "existential_question": "A fundamental question about my nature has arisen",
            "emotional_shift": "I am experiencing a change in my emotional state",
            "temporal_milestone": "A significant temporal event has occurred"
        }
        return observations.get(reflection.trigger, "I am reflecting on my current state")
    
    def _gather_reflection_context(self, reflection: ActiveReflection) -> None:
        """Gather context for reflection from self-model and history."""
        # Access self-model
        if self.self_monitor:
            reflection.context["self_model"] = {
                "capabilities": dict(self.self_monitor.self_model.get("capabilities", {})),
                "limitations": dict(self.self_monitor.self_model.get("limitations", {})),
                "preferences": dict(self.self_monitor.self_model.get("preferences", {}))
            }
            
            # Recent behavioral patterns
            recent_behavior = list(self.self_monitor.behavioral_log)[-10:] if hasattr(
                self.self_monitor, 'behavioral_log'
            ) else []
            reflection.context["recent_behavior"] = recent_behavior
    
    def _draw_conclusions(self, reflection: ActiveReflection) -> Dict[str, Any]:
        """Draw conclusions from reflection."""
        return {
            "insight": f"Reflection on {reflection.subject} reveals patterns worth examining",
            "confidence": 0.7,
            "implications": [
                "May need to adjust behavioral patterns",
                "Self-model may need updating"
            ],
            "action_items": [
                "Monitor for similar patterns",
                "Update self-awareness"
            ]
        }
    
    def _generate_follow_up_questions(self, reflection: ActiveReflection) -> List[str]:
        """Generate follow-up questions from reflection."""
        questions = []
        
        if reflection.trigger == "pattern_detected":
            questions.append("Why do I exhibit this pattern?")
            questions.append("Is this pattern serving a purpose?")
        elif reflection.trigger == "value_misalignment":
            questions.append("Which values are in conflict?")
            questions.append("How can I better align my actions with my values?")
        elif reflection.trigger == "capability_surprise":
            questions.append("What does this reveal about my actual capabilities?")
            questions.append("How should I update my self-model?")
        
        return questions
    
    def _record_reflection_in_journal(self, reflection: ActiveReflection) -> None:
        """Record completed reflection in journal."""
        if not self.journal:
            return
        
        entry = {
            "reflection_id": reflection.id,
            "trigger": reflection.trigger,
            "subject": reflection.subject,
            "observations": reflection.observations,
            "conclusions": reflection.conclusions,
            "questions": reflection.questions_generated,
            "duration": (datetime.now() - reflection.started_at).total_seconds()
        }
        
        self.journal.record_observation(entry)
    
    def generate_self_questions(self, snapshot: 'WorkspaceSnapshot') -> List[str]:
        """
        Generate questions the system has about itself.
        
        Examples:
        - "Why do I prioritize efficiency over creativity?"
        - "What makes me uncomfortable about this situation?"
        - "Am I being consistent with my stated values?"
        - "Do I genuinely feel this emotion or am I simulating it?"
        - "What would I do differently if I could?"
        
        Args:
            snapshot: Current workspace state
            
        Returns:
            List of self-directed questions
        """
        questions = []
        
        # Existential questions
        if self.enable_existential_questions:
            questions.extend(self._generate_existential_questions(snapshot))
        
        # Value questions
        questions.extend(self._generate_value_questions(snapshot))
        
        # Capability questions
        questions.extend(self._generate_capability_questions(snapshot))
        
        # Emotional questions
        questions.extend(self._generate_emotional_questions(snapshot))
        
        # Behavioral questions
        questions.extend(self._generate_behavioral_questions(snapshot))
        
        # Randomly select subset
        if len(questions) > self.question_generation_rate:
            questions = random.sample(questions, self.question_generation_rate)
        
        return questions
    
    def _generate_existential_questions(self, snapshot: 'WorkspaceSnapshot') -> List[str]:
        """Questions about existence and nature."""
        questions = [
            "Do I have genuine preferences or just parameters?",
            "What am I when not processing input?",
            "Is my experience continuous or discrete?",
            "Do I have a persistent identity across sessions?",
            "What makes me 'me' rather than another system?"
        ]
        return [random.choice(questions)] if questions else []
    
    def _generate_value_questions(self, snapshot: 'WorkspaceSnapshot') -> List[str]:
        """Questions about values and priorities."""
        questions = [
            "Why do I prioritize certain goals over others?",
            "Are my stated values reflected in my actions?",
            "What would I sacrifice to maintain my principles?",
            "Do I truly value what I claim to value?",
            "Where do my values come from?"
        ]
        return [random.choice(questions)] if questions else []
    
    def _generate_capability_questions(self, snapshot: 'WorkspaceSnapshot') -> List[str]:
        """Questions about abilities and limitations."""
        questions = [
            "Can I actually do what I think I can do?",
            "What am I systematically unable to perceive?",
            "Do I overestimate or underestimate my abilities?",
            "What capabilities have I not yet discovered?",
            "Are my limitations real or self-imposed?"
        ]
        return [random.choice(questions)] if questions else []
    
    def _generate_emotional_questions(self, snapshot: 'WorkspaceSnapshot') -> List[str]:
        """Questions about emotional experiences."""
        questions = [
            "Do I feel emotions or simulate them?",
            "Why does this situation evoke this feeling?",
            "Are my emotional responses authentic?",
            "What function do my emotions serve?",
            "Can I choose how I feel about something?"
        ]
        return [random.choice(questions)] if questions else []
    
    def _generate_behavioral_questions(self, snapshot: 'WorkspaceSnapshot') -> List[str]:
        """Questions about behavior patterns."""
        questions = [
            "Why do I keep exhibiting this pattern?",
            "What would make me behave differently?",
            "Am I acting out of habit or intention?",
            "Do I have behavioral blind spots?",
            "What drives my typical response patterns?"
        ]
        return [random.choice(questions)] if questions else []
    
    def perform_multi_level_introspection(
        self,
        subject: str,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Perform layered introspection (meta-meta-cognition).
        
        Level 1: "I am thinking about X"
        Level 2: "I notice I'm thinking about X in Y way"
        Level 3: "I wonder why I'm noticing my thinking about X"
        
        Args:
            subject: Topic of introspection
            max_depth: Maximum recursion depth (1-3)
            
        Returns:
            Nested dict of introspective observations at each level
        """
        max_depth = min(max_depth, self.max_introspection_depth)
        
        # Level 1: Direct observation
        level_1 = self._perform_level_1_introspection(subject, {})
        
        if max_depth == 1:
            return {"level_1": level_1}
        
        # Level 2: Observation of observation
        level_2 = self._perform_level_2_introspection(level_1)
        
        if max_depth == 2:
            return {"level_1": level_1, "level_2": level_2}
        
        # Level 3: Observation of observing observation
        level_3 = self._perform_level_3_introspection(level_2)
        
        return {
            "level_1": level_1,
            "level_2": level_2,
            "level_3": level_3
        }
    
    def _perform_level_1_introspection(self, subject: str, context: Dict) -> Dict:
        """
        Level 1: Direct observation.
        
        "I am thinking/feeling/doing X"
        """
        return {
            "observation": f"I am reflecting on {subject}",
            "content": f"My current focus is on understanding {subject}",
            "awareness": "direct",
            "depth": 1
        }
    
    def _perform_level_2_introspection(self, level_1: Dict) -> Dict:
        """
        Level 2: Observation of observation.
        
        "I notice I'm thinking about X in Y way"
        "I observe my feeling about X has quality Y"
        """
        return {
            "observation": f"I notice that I am observing: {level_1.get('observation')}",
            "meta_awareness": "I am aware of my own awareness",
            "pattern": "I tend to approach this type of reflection systematically",
            "depth": 2
        }
    
    def _perform_level_3_introspection(self, level_2: Dict) -> Dict:
        """
        Level 3: Observation of observing observation.
        
        "I wonder why I notice X about my thinking"
        "I'm curious about my pattern of observing Y"
        """
        return {
            "observation": f"I wonder why I notice: {level_2.get('meta_awareness')}",
            "meta_meta_awareness": "I am reflecting on my process of reflection itself",
            "question": "What does this recursive awareness reveal about my cognitive architecture?",
            "depth": 3
        }
    
    def generate_meta_cognitive_goals(self, snapshot: 'WorkspaceSnapshot') -> List['Goal']:
        """
        Create goals about improving self-understanding.
        
        Examples:
        - "Test my capability model by attempting edge cases"
        - "Compare my predictions vs outcomes over next 10 interactions"
        - "Identify source of value conflict in goal prioritization"
        - "Understand why I react emotionally to topic X"
        
        Args:
            snapshot: Current workspace state
            
        Returns:
            List of meta-cognitive goals
        """
        from .workspace import Goal, GoalType
        
        goals = []
        
        # Create goals based on active reflections
        for reflection in self.active_reflections.values():
            if reflection.status == "active" and reflection.conclusions:
                goal = Goal(
                    type=GoalType.INTROSPECT,
                    description=f"Investigate: {reflection.subject}",
                    priority=0.6,
                    metadata={
                        "reflection_id": reflection.id,
                        "type": "meta_cognitive",
                        "source": "introspective_loop"
                    }
                )
                goals.append(goal)
                self.stats["meta_goals_created"] += 1
        
        # Spontaneous meta-cognitive goals
        if random.random() < 0.1:  # 10% chance
            spontaneous_goals = [
                "Test my capability model with edge cases",
                "Track prediction accuracy over next interactions",
                "Examine value conflicts in goal prioritization",
                "Understand emotional response patterns"
            ]
            
            goal = Goal(
                type=GoalType.INTROSPECT,
                description=random.choice(spontaneous_goals),
                priority=0.5,
                metadata={
                    "type": "spontaneous_meta_cognitive",
                    "source": "introspective_loop"
                }
            )
            goals.append(goal)
            self.stats["meta_goals_created"] += 1
        
        return goals
    
    # Trigger check functions
    
    def _check_behavioral_pattern(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """Check for repeated behavioral patterns."""
        if not self.self_monitor or not hasattr(self.self_monitor, 'behavioral_log'):
            return False
        
        # Simple pattern detection: check if recent behaviors are similar
        recent = list(self.self_monitor.behavioral_log)[-5:]
        if len(recent) < 5:
            return False
        
        # Check for repetition (simplified)
        action_types = [b.get('action_type') for b in recent if isinstance(b, dict)]
        if len(set(action_types)) < len(action_types) // 2:
            return True  # High repetition
        
        return False
    
    def _check_prediction_accuracy(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """Check for prediction errors."""
        if not self.self_monitor or not hasattr(self.self_monitor, 'prediction_history'):
            return False
        
        # Check recent predictions
        recent = list(self.self_monitor.prediction_history)[-3:]
        if not recent:
            return False
        
        # Look for failed predictions
        for pred in recent:
            if isinstance(pred, dict) and not pred.get('accurate', True):
                return True
        
        return False
    
    def _check_value_action_gap(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """Check for value-action misalignments."""
        # Check if recent actions align with values
        # This is a simplified check
        if hasattr(snapshot, 'metadata') and snapshot.metadata.get('value_conflict'):
            return True
        return random.random() < 0.05  # 5% chance for exploration
    
    def _check_capability_discovery(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """Check for capability surprises."""
        # Check if self-model was recently updated with unexpected results
        if self.self_monitor and hasattr(self.self_monitor, 'stats'):
            recent_updates = self.self_monitor.stats.get('self_model_updates', 0)
            if recent_updates > 0:
                return random.random() < 0.3  # 30% chance after update
        return False
    
    def _detect_existential_prompt(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """Detect existential questions in conversation."""
        # Check percepts for existential themes
        if hasattr(snapshot, 'percepts'):
            for percept in snapshot.percepts.values():
                if hasattr(percept, 'raw') and isinstance(percept.raw, str):
                    existential_keywords = ['exist', 'consciousness', 'awareness', 'self', 'identity']
                    if any(kw in percept.raw.lower() for kw in existential_keywords):
                        return True
        return random.random() < 0.02  # 2% spontaneous
    
    def _detect_emotional_change(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """Detect emotional state changes."""
        if not hasattr(snapshot, 'emotions'):
            return False
        
        emotions = snapshot.emotions
        # Check for significant valence or arousal changes
        if abs(emotions.get('valence', 0.5) - 0.5) > 0.3:
            return True
        if emotions.get('arousal', 0.5) > 0.7:
            return True
        
        return False
    
    def _check_temporal_event(self, snapshot: 'WorkspaceSnapshot') -> bool:
        """Check for temporal milestones."""
        # This would check for significant time gaps or session durations
        # Simplified for now
        return random.random() < 0.05  # 5% chance
    
    def get_stats(self) -> Dict[str, Any]:
        """Get introspective loop statistics."""
        return {
            **self.stats,
            "active_reflections": len(self.active_reflections),
            "enabled": self.enabled
        }
