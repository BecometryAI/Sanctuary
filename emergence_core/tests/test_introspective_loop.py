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

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from lyra.cognitive_core.introspective_loop import (
    IntrospectiveLoop, ActiveReflection, ReflectionTrigger
)
from lyra.cognitive_core.meta_cognition import SelfMonitor, IntrospectiveJournal
from lyra.cognitive_core.workspace import (
    GlobalWorkspace, WorkspaceSnapshot, Percept, Goal, GoalType
)


class TestIntrospectiveLoopInitialization:
    """Test IntrospectiveLoop initialization"""
    
    def test_basic_initialization(self):
        """Test that IntrospectiveLoop initializes correctly"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            assert loop.workspace == workspace
            assert loop.self_monitor == monitor
            assert loop.journal == journal
            assert loop.enabled is True
            assert isinstance(loop.active_reflections, dict)
            assert isinstance(loop.reflection_triggers, dict)
            assert len(loop.reflection_triggers) == 7  # 7 built-in triggers
    
    def test_initialization_with_config(self):
        """Test initialization with custom configuration"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            
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
    
    def test_trigger_initialization(self):
        """Test that reflection triggers are initialized correctly"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
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
    async def test_check_spontaneous_triggers(self):
        """Test checking for spontaneous triggers"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            snapshot = workspace.broadcast()
            triggered = loop.check_spontaneous_triggers(snapshot)
            
            # Should return a list (may be empty)
            assert isinstance(triggered, list)
    
    @pytest.mark.asyncio
    async def test_trigger_minimum_interval(self):
        """Test that triggers respect minimum interval"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
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
    
    def test_behavioral_pattern_trigger(self):
        """Test behavioral pattern detection trigger"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            # Add repetitive behavior to monitor
            for i in range(5):
                monitor.behavioral_log.append({"action_type": "SPEAK"})
            
            snapshot = workspace.broadcast()
            result = loop._check_behavioral_pattern(snapshot)
            
            # Should detect pattern
            assert result is True
    
    def test_prediction_error_trigger(self):
        """Test prediction error detection trigger"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            # Add failed prediction
            monitor.prediction_history.append({"accurate": False})
            
            snapshot = workspace.broadcast()
            result = loop._check_prediction_accuracy(snapshot)
            
            assert result is True
    
    def test_emotional_change_trigger(self):
        """Test emotional change detection trigger"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
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
    
    def test_initiate_reflection(self):
        """Test starting a new reflection"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            context = {"test": "context"}
            reflection_id = loop.initiate_reflection("pattern_detected", context)
            
            assert reflection_id in loop.active_reflections
            reflection = loop.active_reflections[reflection_id]
            assert isinstance(reflection, ActiveReflection)
            assert reflection.trigger == "pattern_detected"
            assert reflection.status == "active"
            assert reflection.current_step == 0
    
    def test_max_active_reflections_limit(self):
        """Test that max active reflections limit is respected"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            config = {"max_active_reflections": 2}
            loop = IntrospectiveLoop(workspace, monitor, journal, config)
            
            # Create 2 reflections
            loop.initiate_reflection("pattern_detected", {})
            loop.initiate_reflection("prediction_error", {})
            
            assert len(loop.active_reflections) == 2
    
    def test_reflection_subject_determination(self):
        """Test reflection subject is determined correctly"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
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
    
    def test_level_1_introspection(self):
        """Test level 1 (direct observation) introspection"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            result = loop.perform_multi_level_introspection("test subject", max_depth=1)
            
            assert "level_1" in result
            assert result["level_1"]["depth"] == 1
            assert "observation" in result["level_1"]
            assert "level_2" not in result
    
    def test_level_2_introspection(self):
        """Test level 2 (observation of observation) introspection"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            result = loop.perform_multi_level_introspection("test subject", max_depth=2)
            
            assert "level_1" in result
            assert "level_2" in result
            assert result["level_2"]["depth"] == 2
            assert "meta_awareness" in result["level_2"]
            assert "level_3" not in result
    
    def test_level_3_introspection(self):
        """Test level 3 (meta-meta-cognition) introspection"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            result = loop.perform_multi_level_introspection("test subject", max_depth=3)
            
            assert "level_1" in result
            assert "level_2" in result
            assert "level_3" in result
            assert result["level_3"]["depth"] == 3
            assert "meta_meta_awareness" in result["level_3"]
    
    def test_max_depth_constraint(self):
        """Test that introspection respects max_depth configuration"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            config = {"max_introspection_depth": 2}
            loop = IntrospectiveLoop(workspace, monitor, journal, config)
            
            # Request depth 5 but should be capped at 2
            result = loop.perform_multi_level_introspection("test", max_depth=5)
            
            assert "level_1" in result
            assert "level_2" in result
            assert "level_3" not in result


class TestSelfQuestionGeneration:
    """Test autonomous self-question generation"""
    
    def test_generate_existential_questions(self):
        """Test existential question generation"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            snapshot = workspace.broadcast()
            questions = loop._generate_existential_questions(snapshot)
            
            assert isinstance(questions, list)
            if questions:
                assert isinstance(questions[0], str)
    
    def test_generate_value_questions(self):
        """Test value-related question generation"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            snapshot = workspace.broadcast()
            questions = loop._generate_value_questions(snapshot)
            
            assert isinstance(questions, list)
            if questions:
                assert isinstance(questions[0], str)
    
    def test_generate_capability_questions(self):
        """Test capability-related question generation"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            snapshot = workspace.broadcast()
            questions = loop._generate_capability_questions(snapshot)
            
            assert isinstance(questions, list)
            if questions:
                assert isinstance(questions[0], str)
    
    def test_generate_emotional_questions(self):
        """Test emotional question generation"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            snapshot = workspace.broadcast()
            questions = loop._generate_emotional_questions(snapshot)
            
            assert isinstance(questions, list)
            if questions:
                assert isinstance(questions[0], str)
    
    def test_generate_behavioral_questions(self):
        """Test behavioral question generation"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            snapshot = workspace.broadcast()
            questions = loop._generate_behavioral_questions(snapshot)
            
            assert isinstance(questions, list)
            if questions:
                assert isinstance(questions[0], str)
    
    def test_question_generation_rate_limit(self):
        """Test that question generation respects rate limit"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            config = {"question_generation_rate": 2}
            loop = IntrospectiveLoop(workspace, monitor, journal, config)
            
            snapshot = workspace.broadcast()
            questions = loop.generate_self_questions(snapshot)
            
            # Should respect the rate limit
            assert len(questions) <= 2


class TestMetaCognitiveGoals:
    """Test meta-cognitive goal creation"""
    
    def test_generate_meta_cognitive_goals_empty(self):
        """Test goal generation with no active reflections"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            snapshot = workspace.broadcast()
            goals = loop.generate_meta_cognitive_goals(snapshot)
            
            assert isinstance(goals, list)
    
    def test_generate_meta_cognitive_goals_from_reflection(self):
        """Test goal generation from active reflections"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            # Create an active reflection with conclusions
            reflection_id = loop.initiate_reflection("pattern_detected", {})
            loop.active_reflections[reflection_id].conclusions = {"test": "conclusion"}
            
            snapshot = workspace.broadcast()
            goals = loop.generate_meta_cognitive_goals(snapshot)
            
            # Should generate at least one goal from reflection
            meta_goals = [g for g in goals if g.type == GoalType.INTROSPECT]
            assert len(meta_goals) >= 1
    
    def test_meta_cognitive_goal_structure(self):
        """Test that generated goals have correct structure"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
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
    
    def test_process_active_reflections_empty(self):
        """Test processing with no active reflections"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            percepts = loop.process_active_reflections()
            
            assert isinstance(percepts, list)
            assert len(percepts) == 0
    
    def test_reflection_step_progression(self):
        """Test that reflections progress through steps"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            reflection_id = loop.initiate_reflection("pattern_detected", {})
            reflection = loop.active_reflections[reflection_id]
            
            assert reflection.current_step == 0
            
            # Process once
            loop.process_active_reflections()
            
            if reflection_id in loop.active_reflections:
                assert loop.active_reflections[reflection_id].current_step > 0
    
    def test_reflection_completion(self):
        """Test that reflections complete after all steps"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            reflection_id = loop.initiate_reflection("pattern_detected", {})
            initial_count = loop.stats["total_reflections"]
            
            # Process multiple times to complete all steps
            for _ in range(10):
                loop.process_active_reflections()
                if reflection_id not in loop.active_reflections:
                    break
            
            # Should be completed and removed
            assert reflection_id not in loop.active_reflections
            assert loop.stats["completed_reflections"] > 0
    
    def test_reflection_generates_percepts(self):
        """Test that reflections generate percepts"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
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
    async def test_questions_recorded_in_journal(self):
        """Test that generated questions are recorded"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            config = {"spontaneous_probability": 1.0}  # Always generate
            loop = IntrospectiveLoop(workspace, monitor, journal, config)
            
            initial_entries = len(journal.current_session_entries)
            
            # Run a cycle
            await loop.run_reflection_cycle()
            
            # Should have recorded questions
            assert len(journal.current_session_entries) > initial_entries
    
    def test_reflections_recorded_in_journal(self):
        """Test that completed reflections are recorded"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            # Create and complete a reflection
            reflection_id = loop.initiate_reflection("pattern_detected", {})
            
            initial_entries = len(journal.current_session_entries)
            
            # Process until completion
            for _ in range(10):
                loop.process_active_reflections()
                if reflection_id not in loop.active_reflections:
                    break
            
            # Should have recorded the reflection
            assert len(journal.current_session_entries) > initial_entries


class TestReflectionCycle:
    """Test the main reflection cycle"""
    
    @pytest.mark.asyncio
    async def test_run_reflection_cycle_disabled(self):
        """Test that cycle returns empty when disabled"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            config = {"enabled": False}
            loop = IntrospectiveLoop(workspace, monitor, journal, config)
            
            percepts = await loop.run_reflection_cycle()
            
            assert isinstance(percepts, list)
            assert len(percepts) == 0
    
    @pytest.mark.asyncio
    async def test_run_reflection_cycle_basic(self):
        """Test basic reflection cycle execution"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            percepts = await loop.run_reflection_cycle()
            
            assert isinstance(percepts, list)
    
    @pytest.mark.asyncio
    async def test_reflection_cycle_with_triggers(self):
        """Test cycle processes triggers correctly"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            # Mock a trigger to fire
            with patch.object(loop, '_check_behavioral_pattern', return_value=True):
                # Reset last_fired to allow trigger
                loop.reflection_triggers["pattern_detected"].last_fired = None
                
                await loop.run_reflection_cycle()
                
                # Should have initiated a reflection
                assert len(loop.active_reflections) > 0 or loop.stats["triggers_fired"] > 0


class TestStatistics:
    """Test statistics tracking"""
    
    def test_stats_initialization(self):
        """Test that stats are initialized correctly"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
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
    
    def test_stats_update_on_reflection(self):
        """Test that stats update when reflections occur"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            initial_total = loop.stats["total_reflections"]
            
            loop.initiate_reflection("pattern_detected", {})
            
            assert loop.stats["total_reflections"] == initial_total + 1
    
    def test_stats_update_on_completion(self):
        """Test that completion stats update correctly"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
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
    
    def test_default_configuration(self):
        """Test that defaults are used when no config provided"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            assert loop.enabled is True
            assert loop.max_active_reflections == 3
            assert loop.max_introspection_depth == 3
            assert loop.journal_integration is True
    
    def test_configuration_override(self):
        """Test that config overrides defaults"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            
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
    
    def test_empty_workspace(self):
        """Test handling of empty workspace"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            snapshot = workspace.broadcast()
            
            # Should not crash
            loop.check_spontaneous_triggers(snapshot)
            loop.generate_self_questions(snapshot)
            loop.generate_meta_cognitive_goals(snapshot)
    
    def test_reflection_timeout(self):
        """Test that reflections timeout correctly"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            config = {"reflection_timeout": 0}  # Immediate timeout
            loop = IntrospectiveLoop(workspace, monitor, journal, config)
            
            reflection_id = loop.initiate_reflection("pattern_detected", {})
            
            # Process should complete due to timeout
            loop.process_active_reflections()
            
            assert reflection_id not in loop.active_reflections
    
    def test_none_self_monitor(self):
        """Test handling when self_monitor is None"""
        workspace = GlobalWorkspace()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, None, journal)
            
            snapshot = workspace.broadcast()
            
            # Should not crash
            result = loop._check_behavioral_pattern(snapshot)
            assert result is False
    
    @pytest.mark.asyncio
    async def test_error_handling_in_cycle(self):
        """Test that errors in cycle don't crash the loop"""
        workspace = GlobalWorkspace()
        monitor = SelfMonitor(workspace=workspace)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            journal = IntrospectiveJournal(Path(tmpdir))
            loop = IntrospectiveLoop(workspace, monitor, journal)
            
            # Mock a method to raise an exception
            with patch.object(loop, 'check_spontaneous_triggers', side_effect=Exception("Test error")):
                # Should not crash
                percepts = await loop.run_reflection_cycle()
                assert isinstance(percepts, list)


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
