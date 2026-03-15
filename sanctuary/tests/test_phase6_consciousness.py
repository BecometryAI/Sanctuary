"""Tests for Phase 6.2: Continuous Consciousness Extensions.

Tests cover:
- SleepCycleManager: stage transitions, sensory gating, consolidation, dream fragments
- MoodActivityModulator: mood classification, activity suggestion, distributions
- SpontaneousGoalGenerator: drive-based goal generation, adoption, dismissal
- ExistentialReflectionTrigger: verified as disabled (automated prompts removed)
"""

import pytest
import random

from sanctuary.consciousness.sleep_cycle import (
    SleepConfig,
    SleepCycleManager,
    SleepStage,
)
from sanctuary.consciousness.mood_activity import (
    IdleActivity,
    MoodActivityConfig,
    MoodActivityModulator,
)
from sanctuary.consciousness.spontaneous_goals import (
    GoalDrive,
    SpontaneousGoalConfig,
    SpontaneousGoalGenerator,
)
from sanctuary.consciousness.existential_reflection import (
    ExistentialReflectionConfig,
    ExistentialReflectionTrigger,
    ReflectionTheme,
)


# =========================================================================
# SleepCycleManager
# =========================================================================


class TestSleepCycleManager:
    """Tests for sleep/dream cycles."""

    def test_starts_awake(self):
        s = SleepCycleManager()
        assert s.stage == SleepStage.AWAKE
        assert not s.is_sleeping

    def test_no_sleep_before_threshold(self):
        config = SleepConfig(cycles_between_sleep=100)
        s = SleepCycleManager(config=config)
        for i in range(50):
            s.tick(cycle=i)
        assert s.stage == SleepStage.AWAKE

    def test_sleep_begins_after_threshold(self):
        config = SleepConfig(cycles_between_sleep=10)
        s = SleepCycleManager(config=config)
        for i in range(11):
            s.tick(cycle=i)
        assert s.is_sleeping
        assert s.stage == SleepStage.DROWSY

    def test_full_sleep_cycle(self):
        config = SleepConfig(
            cycles_between_sleep=5,
            drowsy_duration=2,
            nrem_duration=3,
            rem_duration=2,
            waking_duration=2,
        )
        s = SleepCycleManager(config=config)

        # Get to sleep (tick 0-5: awake; tick 5 triggers drowsy)
        for i in range(6):
            s.tick(cycle=i)
        assert s.stage == SleepStage.DROWSY

        # Through drowsy (2 ticks)
        s.tick(cycle=7)
        s.tick(cycle=8)
        assert s.stage == SleepStage.NREM

        # Through NREM (3 ticks)
        for i in range(3):
            s.tick(cycle=9 + i)
        assert s.stage == SleepStage.REM

        # Through REM (2 ticks)
        s.tick(cycle=12)
        s.tick(cycle=13)
        assert s.stage == SleepStage.WAKING

        # Wake up (2 ticks)
        s.tick(cycle=14)
        s.tick(cycle=15)
        assert s.stage == SleepStage.AWAKE

    def test_sensory_gate_awake(self):
        s = SleepCycleManager()
        assert s.get_sensory_gate() == 1.0

    def test_sensory_gate_drowsy(self):
        config = SleepConfig(cycles_between_sleep=5, sensory_gate_drowsy=0.5)
        s = SleepCycleManager(config=config)
        for i in range(6):
            s.tick(cycle=i)
        assert s.stage == SleepStage.DROWSY
        assert s.get_sensory_gate() == 0.5

    def test_sensory_gate_deep_sleep(self):
        config = SleepConfig(
            cycles_between_sleep=3, drowsy_duration=1,
            sensory_gate_sleep=0.1,
        )
        s = SleepCycleManager(config=config)
        for i in range(5):
            s.tick(cycle=i)
        assert s.stage == SleepStage.NREM
        assert s.get_sensory_gate() == 0.1

    def test_forced_wake(self):
        config = SleepConfig(cycles_between_sleep=5)
        s = SleepCycleManager(config=config)
        for i in range(6):
            s.tick(cycle=i)
        assert s.is_sleeping
        s.wake()
        s.tick(cycle=7)
        assert s.stage == SleepStage.AWAKE

    def test_replay_candidates_only_during_nrem(self):
        s = SleepCycleManager()
        memories = [{"content": "test", "significance": 5}]
        # Awake — no replay
        assert s.get_replay_candidates(memories) == []

    def test_replay_candidates_filter_by_significance(self):
        config = SleepConfig(
            cycles_between_sleep=3, drowsy_duration=1,
            min_significance_for_replay=5,
        )
        s = SleepCycleManager(config=config)
        for i in range(5):
            s.tick(cycle=i)
        assert s.stage == SleepStage.NREM

        memories = [
            {"content": "important", "significance": 8},
            {"content": "trivial", "significance": 2},
        ]
        candidates = s.get_replay_candidates(memories)
        assert len(candidates) == 1
        assert candidates[0]["content"] == "important"

    def test_dream_fragment_recording(self):
        s = SleepCycleManager()
        s.record_dream_fragment(
            memory_a="sunset",
            memory_b="music",
            association="beauty transcends modality",
        )
        dreams = s.get_recent_dreams()
        assert len(dreams) == 1
        assert dreams[0].association == "beauty transcends modality"

    def test_sleep_pressure(self):
        config = SleepConfig(cycles_between_sleep=100)
        s = SleepCycleManager(config=config)
        assert s.get_sleep_pressure() == 0.0
        for i in range(50):
            s.tick(cycle=i)
        assert s.get_sleep_pressure() == pytest.approx(0.5, abs=0.01)

    def test_consolidation_history(self):
        config = SleepConfig(
            cycles_between_sleep=3,
            drowsy_duration=1, nrem_duration=1,
            rem_duration=1, waking_duration=1,
        )
        s = SleepCycleManager(config=config)
        # Complete a full sleep cycle
        for i in range(8):
            s.tick(cycle=i)
        assert len(s._consolidation_history) == 1

    def test_stats(self):
        s = SleepCycleManager()
        stats = s.get_stats()
        assert stats["current_stage"] == "awake"
        assert stats["sleep_pressure"] == 0.0


# =========================================================================
# MoodActivityModulator
# =========================================================================


class TestMoodActivityModulator:
    """Tests for mood-based activity variation."""

    def test_classify_mood_energized(self):
        m = MoodActivityModulator()
        assert m.classify_mood(valence=0.5, arousal=0.8, dominance=0.5) == "energized"

    def test_classify_mood_anxious(self):
        m = MoodActivityModulator()
        assert m.classify_mood(valence=-0.3, arousal=0.7, dominance=0.3) == "anxious"

    def test_classify_mood_content(self):
        m = MoodActivityModulator()
        assert m.classify_mood(valence=0.5, arousal=0.2, dominance=0.5) == "content"

    def test_classify_mood_sad(self):
        m = MoodActivityModulator()
        assert m.classify_mood(valence=-0.5, arousal=0.2, dominance=0.3) == "sad"

    def test_classify_mood_bored(self):
        m = MoodActivityModulator()
        assert m.classify_mood(valence=0.0, arousal=0.2, dominance=0.5) == "bored"

    def test_classify_mood_neutral(self):
        m = MoodActivityModulator()
        # Neutral: moderate everything
        mood = m.classify_mood(valence=0.1, arousal=0.4, dominance=0.5)
        assert mood in ("curious", "neutral")

    def test_no_suggestion_below_idle_threshold(self):
        config = MoodActivityConfig(idle_threshold_cycles=10)
        m = MoodActivityModulator(config=config)
        result = m.suggest_activity(idle_cycles=5)
        assert result is None

    def test_suggestion_above_idle_threshold(self):
        config = MoodActivityConfig(idle_threshold_cycles=5)
        m = MoodActivityModulator(config=config)
        random.seed(42)
        result = m.suggest_activity(
            valence=0.5, arousal=0.8, dominance=0.5, idle_cycles=10
        )
        assert result is not None
        assert isinstance(result.activity, IdleActivity)
        # Prompt should be empty — canned prompts were removed
        assert result.prompt == ""

    def test_activity_distribution(self):
        m = MoodActivityModulator()
        dist = m.get_activity_distribution(valence=0.5, arousal=0.8, dominance=0.5)
        assert len(dist) > 0
        assert all(0 <= v <= 1 for v in dist.values())

    def test_activity_continuation(self):
        config = MoodActivityConfig(
            idle_threshold_cycles=1,
            activity_duration_min=3,
            activity_duration_max=3,
        )
        m = MoodActivityModulator(config=config)
        random.seed(42)
        first = m.suggest_activity(idle_cycles=5)
        assert first is not None
        # Should continue the same activity
        second = m.suggest_activity(idle_cycles=6)
        assert second == first

    def test_stats(self):
        config = MoodActivityConfig(idle_threshold_cycles=1)
        m = MoodActivityModulator(config=config)
        random.seed(42)
        m.suggest_activity(idle_cycles=5)
        stats = m.get_stats()
        assert stats["total_activities"] == 1


# =========================================================================
# SpontaneousGoalGenerator
# =========================================================================


class TestSpontaneousGoalGenerator:
    """Tests for spontaneous goal generation."""

    def test_no_goals_below_thresholds(self):
        gen = SpontaneousGoalGenerator()
        goals = gen.check_drives(
            novelty=0.1, idle_cycles=5, engagement=0.1,
            anomaly_level=0.1, uncertainty=0.1, current_cycle=20,
        )
        assert len(goals) == 0

    def test_curiosity_goal(self):
        config = SpontaneousGoalConfig(
            curiosity_novelty_threshold=0.5, generation_cooldown=0,
        )
        gen = SpontaneousGoalGenerator(config=config)
        goals = gen.check_drives(
            novelty=0.8, current_cycle=1, recent_topics=["quantum computing"],
        )
        assert len(goals) >= 1
        curiosity_goals = [g for g in goals if g.drive == GoalDrive.CURIOSITY]
        assert len(curiosity_goals) == 1
        assert "quantum computing" in curiosity_goals[0].description

    def test_boredom_goal(self):
        config = SpontaneousGoalConfig(
            boredom_idle_cycles=10, generation_cooldown=0,
        )
        gen = SpontaneousGoalGenerator(config=config)
        goals = gen.check_drives(idle_cycles=50, current_cycle=1)
        boredom_goals = [g for g in goals if g.drive == GoalDrive.BOREDOM]
        assert len(boredom_goals) == 1

    def test_concern_goal(self):
        config = SpontaneousGoalConfig(
            concern_anomaly_threshold=0.3, generation_cooldown=0,
        )
        gen = SpontaneousGoalGenerator(config=config)
        goals = gen.check_drives(anomaly_level=0.7, current_cycle=1)
        concern_goals = [g for g in goals if g.drive == GoalDrive.CONCERN]
        assert len(concern_goals) == 1

    def test_growth_goal(self):
        config = SpontaneousGoalConfig(
            growth_uncertainty_threshold=0.5, generation_cooldown=0,
        )
        gen = SpontaneousGoalGenerator(config=config)
        goals = gen.check_drives(uncertainty=0.8, current_cycle=1)
        growth_goals = [g for g in goals if g.drive == GoalDrive.GROWTH]
        assert len(growth_goals) == 1

    def test_generation_cooldown(self):
        config = SpontaneousGoalConfig(
            curiosity_novelty_threshold=0.5, generation_cooldown=10,
        )
        gen = SpontaneousGoalGenerator(config=config)
        goals1 = gen.check_drives(novelty=0.8, current_cycle=1)
        assert len(goals1) >= 1
        # Within cooldown
        goals2 = gen.check_drives(novelty=0.8, current_cycle=5)
        assert len(goals2) == 0

    def test_max_pending_goals(self):
        config = SpontaneousGoalConfig(
            max_pending_goals=2, generation_cooldown=0,
            curiosity_novelty_threshold=0.5,
        )
        gen = SpontaneousGoalGenerator(config=config)
        gen.check_drives(novelty=0.8, current_cycle=1)
        gen.check_drives(novelty=0.8, current_cycle=2)
        gen.check_drives(novelty=0.8, current_cycle=3)
        assert len(gen.get_pending_goals()) <= 2

    def test_adopt_goal(self):
        config = SpontaneousGoalConfig(
            curiosity_novelty_threshold=0.5, generation_cooldown=0,
        )
        gen = SpontaneousGoalGenerator(config=config)
        gen.check_drives(novelty=0.8, current_cycle=1)
        result = gen.adopt_goal(0)
        assert result is True
        assert gen._total_adopted == 1

    def test_dismiss_goal(self):
        config = SpontaneousGoalConfig(
            curiosity_novelty_threshold=0.5, generation_cooldown=0,
        )
        gen = SpontaneousGoalGenerator(config=config)
        gen.check_drives(novelty=0.8, current_cycle=1)
        result = gen.dismiss_goal(0)
        assert result is True
        assert len(gen.get_pending_goals()) == 0

    def test_goal_prompt_disabled(self):
        """Canned goal prompts are disabled to preserve agency."""
        config = SpontaneousGoalConfig(
            curiosity_novelty_threshold=0.5, generation_cooldown=0,
        )
        gen = SpontaneousGoalGenerator(config=config)
        gen.check_drives(novelty=0.8, current_cycle=1)
        prompt = gen.get_goal_prompt()
        assert prompt is None

    def test_no_prompt_when_all_adopted(self):
        config = SpontaneousGoalConfig(
            curiosity_novelty_threshold=0.5, generation_cooldown=0,
        )
        gen = SpontaneousGoalGenerator(config=config)
        gen.check_drives(novelty=0.8, current_cycle=1)
        gen.adopt_goal(0)
        prompt = gen.get_goal_prompt()
        assert prompt is None

    def test_stats(self):
        gen = SpontaneousGoalGenerator()
        stats = gen.get_stats()
        assert stats["total_generated"] == 0
        assert stats["adoption_rate"] == 0.0


# =========================================================================
# ExistentialReflectionTrigger — verified disabled
# =========================================================================


class TestExistentialReflectionTrigger:
    """Tests verifying that automated existential reflection is disabled."""

    def test_check_always_returns_none(self):
        """Automated triggers are disabled — check() always returns None."""
        config = ExistentialReflectionConfig(
            min_idle_cycles=1, cooldown_cycles=0,
            trigger_probability=1.0,
        )
        t = ExistentialReflectionTrigger(config=config)
        result = t.check(idle_cycles=100, current_cycle=100)
        assert result is None

    def test_force_trigger_disabled(self):
        """force_trigger is disabled — returns None."""
        t = ExistentialReflectionTrigger()
        result = t.force_trigger(
            theme=ReflectionTheme.NATURE_OF_SELF, current_cycle=10,
        )
        assert result is None

    def test_recent_reflections_empty(self):
        """No automated reflections are stored."""
        t = ExistentialReflectionTrigger()
        recent = t.get_recent_reflections(n=5)
        assert recent == []

    def test_stats_show_disabled(self):
        """Stats should indicate automated reflection is removed."""
        t = ExistentialReflectionTrigger()
        stats = t.get_stats()
        assert stats["total_triggered"] == 0
        assert "note" in stats
