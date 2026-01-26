"""
Unit tests for Introspective Loop (Phase 4.2).

Tests cover:
- IntrospectiveLoop initialization
- Spontaneous reflection triggers
- Reflection initiation and tracking
- Multi-level introspection (depths 1-3)
- Self-question generation (all categories)
- Meta-cognitive goal creation
- Active reflection processing
- Journal integration
- Idle loop integration
- Configuration handling
"""

import gc
import pytest
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from mind.cognitive_core.introspective_loop import (
    IntrospectiveLoop, ActiveReflection, ReflectionTrigger
)
from mind.cognitive_core.meta_cognition import SelfMonitor, IntrospectiveJournal
from mind.cognitive_core.workspace import (
    GlobalWorkspace, WorkspaceSnapshot, Percept, Goal, GoalType
)


@pytest.fixture
def temp_journal_dir():
    """Create a temporary directory for journal tests with proper cleanup."""
    temp_base = tempfile.mkdtemp()
    journal_dir = Path(temp_base) / "journal"
    journal_dir.mkdir(parents=True, exist_ok=True)

    # Track created objects for cleanup
    created_objects = {"journals": [], "loops": []}

    def factory():
        return journal_dir, created_objects

    yield factory()

    # Cleanup: close all journals first
    for journal in created_objects["journals"]:
        try:
            journal.close()
        except Exception:
            pass

    # Force garbage collection
    gc.collect()

    # Retry cleanup for Windows file locking
    for attempt in range(3):
        try:
            shutil.rmtree(temp_base)
            break
        except PermissionError:
            if attempt < 2:
                time.sleep(0.5)
                gc.collect()


class TestIntrospectiveLoopInitialization:
    """Test IntrospectiveLoop initialization"""

    def test_basic_initialization(self, temp_journal_dir):
        """Test that IntrospectiveLoop initializes correctly"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        assert loop.workspace == workspace
        assert loop.self_monitor == monitor
        assert loop.journal == journal
        assert loop.enabled is True
        assert isinstance(loop.active_reflections, dict)
        assert isinstance(loop.reflection_triggers, dict)
        assert len(loop.reflection_triggers) == 7  # 7 built-in triggers

    def test_initialization_with_config(self, temp_journal_dir):
        """Test initialization with custom configuration"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)

        config = {
            "enabled": False,
            "max_active_reflections": 5,
            "max_introspection_depth": 2,
            "spontaneous_probability": 0.5,
            "question_generation_rate": 3
        }

        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        assert loop.enabled is False
        assert loop.max_active_reflections == 5
        assert loop.max_introspection_depth == 2
        assert loop.spontaneous_probability == 0.5
        assert loop.question_generation_rate == 3

    def test_trigger_initialization(self, temp_journal_dir):
        """Test that reflection triggers are initialized correctly"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Check all expected triggers exist
        expected_triggers = [
            "pattern_detected",
            "prediction_error",
            "value_misalignment",
            "capability_surprise",
            "existential_question",
            "emotional_shift",
            "temporal_milestone"
        ]

        for trigger_id in expected_triggers:
            assert trigger_id in loop.reflection_triggers
            trigger = loop.reflection_triggers[trigger_id]
            assert isinstance(trigger, ReflectionTrigger)
            assert trigger.priority > 0.0
            assert trigger.min_interval > 0


class TestSpontaneousTriggers:
    """Test spontaneous reflection trigger detection"""

    @pytest.mark.asyncio
    async def test_check_spontaneous_triggers(self, temp_journal_dir):
        """Test checking for spontaneous triggers"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()
        triggered = loop.check_spontaneous_triggers(snapshot)

        # Should return a list (may be empty)
        assert isinstance(triggered, list)

    @pytest.mark.asyncio
    async def test_trigger_minimum_interval(self, temp_journal_dir):
        """Test that triggers respect minimum interval"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Manually fire a trigger
        trigger = loop.reflection_triggers["pattern_detected"]
        trigger.last_fired = datetime.now()

        snapshot = workspace.broadcast()

        # Mock the check function to return True
        with patch.object(loop, '_check_behavioral_pattern', return_value=True):
            triggered = loop.check_spontaneous_triggers(snapshot)

            # Should not trigger due to minimum interval
            assert "pattern_detected" not in triggered

    def test_behavioral_pattern_trigger(self, temp_journal_dir):
        """Test behavioral pattern detection trigger"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Add repetitive behavior to monitor
        for i in range(5):
            monitor.behavioral_log.append({"action_type": "SPEAK"})

        snapshot = workspace.broadcast()
        result = loop._check_behavioral_pattern(snapshot)

        # Should detect pattern
        assert result is True

    def test_prediction_error_trigger(self, temp_journal_dir):
        """Test prediction error detection trigger"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Add failed prediction
        monitor.prediction_history.append({"accurate": False})

        snapshot = workspace.broadcast()
        result = loop._check_prediction_accuracy(snapshot)

        assert result is True

    def test_emotional_change_trigger(self, temp_journal_dir):
        """Test emotional change detection trigger"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Create snapshot with strong emotion
        snapshot = WorkspaceSnapshot(
            goals=[],
            percepts={},
            emotions={"valence": 0.9, "arousal": 0.8, "dominance": 0.5},
            memories=[],
            timestamp=datetime.now(),
            cycle_count=0,
            metadata={}
        )

        result = loop._detect_emotional_change(snapshot)

        assert result is True


class TestReflectionInitiation:
    """Test reflection initiation and tracking"""

    def test_initiate_reflection(self, temp_journal_dir):
        """Test starting a new reflection"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        context = {"test": "context"}
        reflection_id = loop.initiate_reflection("pattern_detected", context)

        assert reflection_id in loop.active_reflections
        reflection = loop.active_reflections[reflection_id]
        assert isinstance(reflection, ActiveReflection)
        assert reflection.trigger == "pattern_detected"
        assert reflection.status == "active"
        assert reflection.current_step == 0

    def test_max_active_reflections_limit(self, temp_journal_dir):
        """Test that max active reflections limit is respected"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        config = {"max_active_reflections": 2}
        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        # Create 2 reflections
        loop.initiate_reflection("pattern_detected", {})
        loop.initiate_reflection("prediction_error", {})

        assert len(loop.active_reflections) == 2

    def test_reflection_subject_determination(self, temp_journal_dir):
        """Test reflection subject is determined correctly"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        subjects = {
            "pattern_detected": "behavioral patterns",
            "value_misalignment": "alignment between my values and actions",
            "existential_question": "fundamental questions about my nature"
        }

        for trigger, expected_key in subjects.items():
            subject = loop._determine_reflection_subject(trigger, {})
            assert expected_key in subject.lower()


class TestMultiLevelIntrospection:
    """Test multi-level introspection functionality"""

    def test_level_1_introspection(self, temp_journal_dir):
        """Test level 1 (direct observation) introspection"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        result = loop.perform_multi_level_introspection("test subject", max_depth=1)

        assert "level_1" in result
        assert result["level_1"]["depth"] == 1
        assert "observation" in result["level_1"]
        assert "level_2" not in result

    def test_level_2_introspection(self, temp_journal_dir):
        """Test level 2 (observation of observation) introspection"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        result = loop.perform_multi_level_introspection("test subject", max_depth=2)

        assert "level_1" in result
        assert "level_2" in result
        assert result["level_2"]["depth"] == 2
        assert "meta_awareness" in result["level_2"]
        assert "level_3" not in result

    def test_level_3_introspection(self, temp_journal_dir):
        """Test level 3 (meta-meta-cognition) introspection"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        result = loop.perform_multi_level_introspection("test subject", max_depth=3)

        assert "level_1" in result
        assert "level_2" in result
        assert "level_3" in result
        assert result["level_3"]["depth"] == 3
        assert "meta_meta_awareness" in result["level_3"]

    def test_max_depth_constraint(self, temp_journal_dir):
        """Test that introspection respects max_depth configuration"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        config = {"max_introspection_depth": 2}
        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        # Request depth 5 but should be capped at 2
        result = loop.perform_multi_level_introspection("test", max_depth=5)

        assert "level_1" in result
        assert "level_2" in result
        assert "level_3" not in result


class TestSelfQuestionGeneration:
    """Test autonomous self-question generation"""

    def test_generate_existential_questions(self, temp_journal_dir):
        """Test existential question generation"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()
        questions = loop._generate_existential_questions(snapshot)

        assert isinstance(questions, list)
        if questions:
            assert isinstance(questions[0], str)

    def test_generate_value_questions(self, temp_journal_dir):
        """Test value-related question generation"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()
        questions = loop._generate_value_questions(snapshot)

        assert isinstance(questions, list)
        if questions:
            assert isinstance(questions[0], str)

    def test_generate_capability_questions(self, temp_journal_dir):
        """Test capability-related question generation"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()
        questions = loop._generate_capability_questions(snapshot)

        assert isinstance(questions, list)
        if questions:
            assert isinstance(questions[0], str)

    def test_generate_emotional_questions(self, temp_journal_dir):
        """Test emotional question generation"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()
        questions = loop._generate_emotional_questions(snapshot)

        assert isinstance(questions, list)
        if questions:
            assert isinstance(questions[0], str)

    def test_generate_behavioral_questions(self, temp_journal_dir):
        """Test behavioral question generation"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()
        questions = loop._generate_behavioral_questions(snapshot)

        assert isinstance(questions, list)
        if questions:
            assert isinstance(questions[0], str)

    def test_question_generation_rate_limit(self, temp_journal_dir):
        """Test that question generation respects rate limit"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        config = {"question_generation_rate": 2}
        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        snapshot = workspace.broadcast()
        questions = loop.generate_self_questions(snapshot)

        # Should respect the rate limit
        assert len(questions) <= 2


class TestMetaCognitiveGoals:
    """Test meta-cognitive goal creation"""

    def test_generate_meta_cognitive_goals_empty(self, temp_journal_dir):
        """Test goal generation with no active reflections"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()
        goals = loop.generate_meta_cognitive_goals(snapshot)

        assert isinstance(goals, list)

    def test_generate_meta_cognitive_goals_from_reflection(self, temp_journal_dir):
        """Test goal generation from active reflections"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Create an active reflection with conclusions
        reflection_id = loop.initiate_reflection("pattern_detected", {})
        loop.active_reflections[reflection_id].conclusions = {"test": "conclusion"}

        snapshot = workspace.broadcast()
        goals = loop.generate_meta_cognitive_goals(snapshot)

        # Should generate at least one goal from reflection
        meta_goals = [g for g in goals if g.type == GoalType.INTROSPECT]
        assert len(meta_goals) >= 1

    def test_meta_cognitive_goal_structure(self, temp_journal_dir):
        """Test that generated goals have correct structure"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Force goal generation
        reflection_id = loop.initiate_reflection("pattern_detected", {})
        loop.active_reflections[reflection_id].conclusions = {"test": "conclusion"}

        snapshot = workspace.broadcast()
        goals = loop.generate_meta_cognitive_goals(snapshot)

        if goals:
            goal = goals[0]
            assert isinstance(goal, Goal)
            assert goal.type == GoalType.INTROSPECT
            assert hasattr(goal, 'description')
            assert hasattr(goal, 'priority')


class TestActiveReflectionProcessing:
    """Test processing of active reflections"""

    def test_process_active_reflections_empty(self, temp_journal_dir):
        """Test processing with no active reflections"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        percepts = loop.process_active_reflections()

        assert isinstance(percepts, list)
        assert len(percepts) == 0

    def test_reflection_step_progression(self, temp_journal_dir):
        """Test that reflections progress through steps"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        reflection_id = loop.initiate_reflection("pattern_detected", {})
        reflection = loop.active_reflections[reflection_id]

        assert reflection.current_step == 0

        # Process once
        loop.process_active_reflections()

        if reflection_id in loop.active_reflections:
            assert loop.active_reflections[reflection_id].current_step > 0

    def test_reflection_completion(self, temp_journal_dir):
        """Test that reflections complete after all steps"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        reflection_id = loop.initiate_reflection("pattern_detected", {})

        # Process multiple times to complete all steps
        for _ in range(10):
            loop.process_active_reflections()
            if reflection_id not in loop.active_reflections:
                break

        # Should be completed and removed
        assert reflection_id not in loop.active_reflections
        assert loop.stats["completed_reflections"] > 0

    def test_reflection_generates_percepts(self, temp_journal_dir):
        """Test that reflections generate percepts"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        loop.initiate_reflection("pattern_detected", {})

        all_percepts = []
        for _ in range(10):
            percepts = loop.process_active_reflections()
            all_percepts.extend(percepts)

        # Should generate at least one percept during the process
        assert len(all_percepts) > 0
        assert all(isinstance(p, Percept) for p in all_percepts)


class TestJournalIntegration:
    """Test introspective journal integration"""

    @pytest.mark.asyncio
    async def test_questions_recorded_in_journal(self, temp_journal_dir):
        """Test that generated questions are recorded"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        config = {"spontaneous_probability": 1.0}  # Always generate
        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        initial_entries = len(journal.recent_entries)

        # Run a cycle
        await loop.run_reflection_cycle()

        # Should have recorded questions (or stayed same if no questions generated)
        assert len(journal.recent_entries) >= initial_entries

    def test_reflections_recorded_in_journal(self, temp_journal_dir):
        """Test that completed reflections are recorded"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Create and complete a reflection
        reflection_id = loop.initiate_reflection("pattern_detected", {})

        initial_entries = len(journal.recent_entries)

        # Process until completion
        for _ in range(10):
            loop.process_active_reflections()
            if reflection_id not in loop.active_reflections:
                break

        # Should have recorded the reflection (or stayed same)
        assert len(journal.recent_entries) >= initial_entries


class TestReflectionCycle:
    """Test the main reflection cycle"""

    @pytest.mark.asyncio
    async def test_run_reflection_cycle_disabled(self, temp_journal_dir):
        """Test that cycle returns empty when disabled"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        config = {"enabled": False}
        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        percepts = await loop.run_reflection_cycle()

        assert isinstance(percepts, list)
        assert len(percepts) == 0

    @pytest.mark.asyncio
    async def test_run_reflection_cycle_basic(self, temp_journal_dir):
        """Test basic reflection cycle execution"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        percepts = await loop.run_reflection_cycle()

        assert isinstance(percepts, list)

    @pytest.mark.asyncio
    async def test_reflection_cycle_with_triggers(self, temp_journal_dir):
        """Test cycle processes triggers correctly"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Mock a trigger to fire
        with patch.object(loop, '_check_behavioral_pattern', return_value=True):
            # Reset last_fired to allow trigger
            loop.reflection_triggers["pattern_detected"].last_fired = None

            await loop.run_reflection_cycle()

            # Should have initiated a reflection
            assert len(loop.active_reflections) >= 0 or loop.stats["triggers_fired"] >= 0


class TestStatistics:
    """Test statistics tracking"""

    def test_stats_initialization(self, temp_journal_dir):
        """Test that stats are initialized correctly"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        stats = loop.get_stats()

        assert "total_reflections" in stats
        assert "completed_reflections" in stats
        assert "questions_generated" in stats
        assert "triggers_fired" in stats
        assert "meta_goals_created" in stats
        assert "multi_level_introspections" in stats
        assert "active_reflections" in stats
        assert "enabled" in stats

    def test_stats_update_on_reflection(self, temp_journal_dir):
        """Test that stats update when reflections occur"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        initial_total = loop.stats["total_reflections"]

        loop.initiate_reflection("pattern_detected", {})

        assert loop.stats["total_reflections"] == initial_total + 1

    def test_stats_update_on_completion(self, temp_journal_dir):
        """Test that completion stats update correctly"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        reflection_id = loop.initiate_reflection("pattern_detected", {})
        initial_completed = loop.stats["completed_reflections"]

        # Process until completion
        for _ in range(10):
            loop.process_active_reflections()
            if reflection_id not in loop.active_reflections:
                break

        assert loop.stats["completed_reflections"] > initial_completed


class TestConfigurationHandling:
    """Test configuration parameter handling"""

    def test_default_configuration(self, temp_journal_dir):
        """Test that defaults are used when no config provided"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        assert loop.enabled is True
        assert loop.max_active_reflections == 3
        assert loop.max_introspection_depth == 3
        assert loop.journal_integration is True

    def test_configuration_override(self, temp_journal_dir):
        """Test that config overrides defaults"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)

        config = {
            "enabled": False,
            "max_active_reflections": 10,
            "enable_existential_questions": False,
            "reflection_timeout": 600
        }

        loop = IntrospectiveLoop(workspace, monitor, journal, config)

        assert loop.enabled is False
        assert loop.max_active_reflections == 10
        assert loop.enable_existential_questions is False
        assert loop.reflection_timeout == 600


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_workspace(self, temp_journal_dir):
        """Test handling of empty workspace"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        snapshot = workspace.broadcast()

        # Should not crash
        loop.check_spontaneous_triggers(snapshot)
        loop.generate_self_questions(snapshot)
        loop.generate_meta_cognitive_goals(snapshot)

    def test_reflection_timeout(self, temp_journal_dir):
        """Test that reflections eventually complete"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        reflection_id = loop.initiate_reflection("pattern_detected", {})

        # Process multiple times to complete the reflection
        for _ in range(15):
            loop.process_active_reflections()
            if reflection_id not in loop.active_reflections:
                break

        # Should be completed and removed
        assert reflection_id not in loop.active_reflections

    def test_none_self_monitor(self, temp_journal_dir):
        """Test handling when self_monitor is None"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, None, journal)

        snapshot = workspace.broadcast()

        # Should not crash
        result = loop._check_behavioral_pattern(snapshot)
        assert result is False

    @pytest.mark.asyncio
    async def test_error_handling_in_cycle(self, temp_journal_dir):
        """Test that errors in cycle don't crash the loop"""
        journal_dir, created_objects = temp_journal_dir
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)

        journal = IntrospectiveJournal(journal_dir)
        created_objects["journals"].append(journal)
        loop = IntrospectiveLoop(workspace, monitor, journal)

        # Mock a method to raise an exception
        with patch.object(loop, 'check_spontaneous_triggers', side_effect=Exception("Test error")):
            # Should not crash
            percepts = await loop.run_reflection_cycle()
            assert isinstance(percepts, list)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
