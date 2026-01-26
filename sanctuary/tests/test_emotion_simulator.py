"""
Unit tests for emotion_simulator.py (Element 5)

Tests cover:
- Affective state model (PAD dimensions)
- Emotion generation through appraisal theory
- Emotional memory weighting
- Mood persistence and decay
- Edge cases and error handling
"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mind.emotion_simulator import (
    EmotionSimulator,
    AffectiveState,
    Emotion,
    Mood,
    EmotionCategory,
    AppraisalType
)


class TestAffectiveState:
    """Test AffectiveState data model"""
    
    def test_affective_state_creation_valid(self):
        """Test creating a valid affective state"""
        state = AffectiveState(
            valence=0.5,
            arousal=0.3,
            dominance=0.7
        )
        
        assert state.valence == 0.5
        assert state.arousal == 0.3
        assert state.dominance == 0.7
    
    def test_affective_state_invalid_valence(self):
        """Test invalid valence range"""
        with pytest.raises(ValueError, match="valence must be in range"):
            AffectiveState(valence=1.5, arousal=0.0, dominance=0.0)
        
        with pytest.raises(ValueError, match="valence must be in range"):
            AffectiveState(valence=-1.5, arousal=0.0, dominance=0.0)
    
    def test_affective_state_invalid_arousal(self):
        """Test invalid arousal range"""
        with pytest.raises(ValueError, match="arousal must be in range"):
            AffectiveState(valence=0.0, arousal=2.0, dominance=0.0)
    
    def test_affective_state_invalid_dominance(self):
        """Test invalid dominance range"""
        with pytest.raises(ValueError, match="dominance must be in range"):
            AffectiveState(valence=0.0, arousal=0.0, dominance=-2.0)
    
    def test_affective_state_distance(self):
        """Test distance calculation between states"""
        state1 = AffectiveState(valence=0.5, arousal=0.5, dominance=0.5)
        state2 = AffectiveState(valence=-0.5, arousal=-0.5, dominance=-0.5)
        
        distance = state1.distance_to(state2)
        
        # Distance should be sqrt(3) for unit cube diagonal
        assert abs(distance - 1.732) < 0.01
    
    def test_affective_state_serialization(self):
        """Test to_dict and from_dict"""
        original = AffectiveState(
            valence=0.7,
            arousal=-0.3,
            dominance=0.4
        )
        
        data = original.to_dict()
        restored = AffectiveState.from_dict(data)
        
        assert restored.valence == original.valence
        assert restored.arousal == original.arousal
        assert restored.dominance == original.dominance


class TestEmotion:
    """Test Emotion data model"""
    
    def test_emotion_creation_valid(self):
        """Test creating a valid emotion"""
        state = AffectiveState(valence=0.8, arousal=0.6, dominance=0.5)
        emotion = Emotion(
            category=EmotionCategory.JOY,
            intensity=0.9,
            affective_state=state
        )
        
        assert emotion.category == EmotionCategory.JOY
        assert emotion.intensity == 0.9
        assert emotion.affective_state.valence == 0.8
    
    def test_emotion_invalid_intensity(self):
        """Test invalid intensity range"""
        state = AffectiveState(valence=0.5, arousal=0.5, dominance=0.5)
        
        with pytest.raises(ValueError, match="Intensity must be in range"):
            Emotion(
                category=EmotionCategory.JOY,
                intensity=1.5,
                affective_state=state
            )
    
    def test_emotion_is_active(self):
        """Test emotion active status"""
        state = AffectiveState(valence=0.5, arousal=0.5, dominance=0.5)
        
        # Active emotion
        active_emotion = Emotion(
            category=EmotionCategory.JOY,
            intensity=0.5,
            affective_state=state
        )
        assert active_emotion.is_active()
        
        # Inactive emotion
        inactive_emotion = Emotion(
            category=EmotionCategory.JOY,
            intensity=0.05,
            affective_state=state
        )
        assert not inactive_emotion.is_active()
    
    def test_emotion_serialization(self):
        """Test emotion to_dict and from_dict"""
        state = AffectiveState(valence=0.7, arousal=0.5, dominance=0.6)
        original = Emotion(
            category=EmotionCategory.SURPRISE,
            intensity=0.8,
            affective_state=state,
            context={'event': 'unexpected'}
        )
        
        data = original.to_dict()
        restored = Emotion.from_dict(data)
        
        assert restored.category == original.category
        assert restored.intensity == original.intensity
        assert restored.context == original.context


class TestMood:
    """Test Mood data model"""
    
    def test_mood_creation_valid(self):
        """Test creating a valid mood"""
        baseline = AffectiveState(valence=0.2, arousal=0.0, dominance=0.1)
        current = AffectiveState(valence=0.5, arousal=0.3, dominance=0.4)
        
        mood = Mood(
            baseline=baseline,
            current=current,
            influence=0.3,
            decay_rate=0.05
        )
        
        assert mood.influence == 0.3
        assert mood.decay_rate == 0.05
    
    def test_mood_invalid_influence(self):
        """Test invalid influence range"""
        baseline = AffectiveState(valence=0.0, arousal=0.0, dominance=0.0)
        current = AffectiveState(valence=0.0, arousal=0.0, dominance=0.0)
        
        with pytest.raises(ValueError, match="Influence must be in range"):
            Mood(baseline=baseline, current=current, influence=1.5)
    
    def test_mood_invalid_decay_rate(self):
        """Test invalid decay rate range"""
        baseline = AffectiveState(valence=0.0, arousal=0.0, dominance=0.0)
        current = AffectiveState(valence=0.0, arousal=0.0, dominance=0.0)
        
        with pytest.raises(ValueError, match="Decay rate must be in range"):
            Mood(baseline=baseline, current=current, decay_rate=-0.1)


class TestEmotionSimulator:
    """Test EmotionSimulator core functionality"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)
    
    @pytest.fixture
    def emotion_sim(self, temp_dir):
        """Create EmotionSimulator instance"""
        return EmotionSimulator(persistence_dir=temp_dir)
    
    # ========================================================================
    # Initialization Tests
    # ========================================================================
    
    def test_initialization(self, emotion_sim):
        """Test emotion simulator initializes correctly"""
        assert emotion_sim.baseline.valence == 0.2
        assert emotion_sim.mood.baseline == emotion_sim.baseline
        assert len(emotion_sim.active_emotions) == 0
    
    # ========================================================================
    # Emotion Generation Tests (Appraisal Theory)
    # ========================================================================
    
    def test_appraise_goal_progress_high(self, emotion_sim):
        """Test goal progress appraisal generates joy"""
        context = {'progress': 0.9, 'strength': 0.8}
        
        emotion = emotion_sim.appraise_context(
            context=context,
            appraisal_type=AppraisalType.GOAL_PROGRESS
        )
        
        assert emotion is not None
        assert emotion.category == EmotionCategory.JOY
        assert emotion.affective_state.valence > 0
        assert emotion.intensity > 0
    
    def test_appraise_goal_progress_low(self, emotion_sim):
        """Test low goal progress generates neutral emotion"""
        context = {'progress': 0.1, 'strength': 0.5}
        
        emotion = emotion_sim.appraise_context(
            context=context,
            appraisal_type=AppraisalType.GOAL_PROGRESS
        )
        
        # Should generate neutral or no emotion
        assert emotion is None or emotion.category == EmotionCategory.NEUTRAL
    
    def test_appraise_goal_obstruction_high_control(self, emotion_sim):
        """Test obstruction with control generates anger"""
        context = {
            'severity': 0.8,
            'control': 0.7,  # Can overcome
            'strength': 0.9
        }
        
        emotion = emotion_sim.appraise_context(
            context=context,
            appraisal_type=AppraisalType.GOAL_OBSTRUCTION
        )
        
        assert emotion is not None
        assert emotion.category == EmotionCategory.ANGER
        assert emotion.affective_state.arousal > 0  # High arousal
        assert emotion.affective_state.dominance > 0  # High dominance
    
    def test_appraise_goal_obstruction_low_control(self, emotion_sim):
        """Test obstruction without control generates sadness"""
        context = {
            'severity': 0.8,
            'control': 0.2,  # Cannot overcome
            'strength': 0.9
        }
        
        emotion = emotion_sim.appraise_context(
            context=context,
            appraisal_type=AppraisalType.GOAL_OBSTRUCTION
        )
        
        assert emotion is not None
        assert emotion.category == EmotionCategory.SADNESS
        assert emotion.affective_state.valence < 0  # Negative
        assert emotion.affective_state.arousal < 0  # Low arousal
    
    def test_appraise_novelty(self, emotion_sim):
        """Test novelty appraisal generates surprise"""
        context = {
            'unexpectedness': 0.9,
            'valence': 0.3,  # Positive surprise
            'strength': 0.7
        }
        
        emotion = emotion_sim.appraise_context(
            context=context,
            appraisal_type=AppraisalType.NOVELTY
        )
        
        assert emotion is not None
        assert emotion.category == EmotionCategory.SURPRISE
        assert emotion.affective_state.arousal > 0  # High arousal from surprise
    
    def test_appraise_social_connection_positive(self, emotion_sim):
        """Test positive social connection generates joy"""
        context = {'quality': 0.9, 'strength': 0.8}
        
        emotion = emotion_sim.appraise_context(
            context=context,
            appraisal_type=AppraisalType.SOCIAL_CONNECTION
        )
        
        assert emotion is not None
        assert emotion.category == EmotionCategory.JOY
        assert emotion.affective_state.valence > 0
    
    def test_appraise_social_connection_negative(self, emotion_sim):
        """Test negative social connection generates sadness"""
        context = {'quality': 0.1, 'strength': 0.8}
        
        emotion = emotion_sim.appraise_context(
            context=context,
            appraisal_type=AppraisalType.SOCIAL_CONNECTION
        )
        
        assert emotion is not None
        assert emotion.category == EmotionCategory.SADNESS
        assert emotion.affective_state.valence < 0
    
    # ========================================================================
    # Mood Influence Tests
    # ========================================================================
    
    def test_mood_influence_on_emotion(self, emotion_sim):
        """Test that mood influences emotional responses"""
        # Set a negative mood
        emotion_sim.mood.current = AffectiveState(
            valence=-0.6,
            arousal=-0.2,
            dominance=-0.3
        )
        
        # Generate emotion that would normally be positive
        context = {'quality': 0.8, 'strength': 0.7}
        emotion = emotion_sim.appraise_context(
            context=context,
            appraisal_type=AppraisalType.SOCIAL_CONNECTION
        )
        
        # Emotion should still be joy, but less positive due to negative mood
        assert emotion is not None
        assert emotion.category == EmotionCategory.JOY
        # Valence should be reduced by mood influence
        assert emotion.affective_state.valence < 0.8
    
    def test_mood_update_from_emotion(self, emotion_sim):
        """Test mood updates from emotions"""
        initial_mood_valence = emotion_sim.mood.current.valence
        
        # Generate strong positive emotion
        context = {'progress': 0.95, 'strength': 1.0}
        emotion = emotion_sim.appraise_context(
            context=context,
            appraisal_type=AppraisalType.GOAL_PROGRESS
        )
        
        # Mood should shift slightly positive
        assert emotion_sim.mood.current.valence > initial_mood_valence
    
    def test_mood_decay(self, emotion_sim):
        """Test mood decays back to baseline"""
        # Shift mood away from baseline
        emotion_sim.mood.current = AffectiveState(
            valence=0.9,
            arousal=0.7,
            dominance=0.8
        )
        
        # Update last_updated to trigger decay
        emotion_sim.mood.last_updated = datetime.now() - timedelta(minutes=5)
        
        initial_distance = emotion_sim.mood.current.distance_to(emotion_sim.mood.baseline)
        
        # Apply decay (should return True)
        decay_applied = emotion_sim.update_mood_decay()
        
        # Mood should be closer to baseline
        final_distance = emotion_sim.mood.current.distance_to(emotion_sim.mood.baseline)
        assert final_distance < initial_distance
        assert decay_applied is True
    
    def test_mood_decay_throttle(self, emotion_sim):
        """Test mood decay is throttled for recent updates"""
        # Set mood recently (within 60 seconds)
        emotion_sim.mood.current = AffectiveState(
            valence=0.9,
            arousal=0.7,
            dominance=0.8
        )
        emotion_sim.mood.last_updated = datetime.now() - timedelta(seconds=30)
        
        # Try to decay (should be throttled)
        decay_applied = emotion_sim.update_mood_decay()
        
        # Should return False (no decay)
        assert decay_applied is False
    
    # ========================================================================
    # Emotional Memory Weighting Tests
    # ========================================================================
    
    def test_calculate_emotional_weight_high_intensity(self, emotion_sim):
        """Test high-intensity emotion produces high weight"""
        state = AffectiveState(valence=0.9, arousal=0.9, dominance=0.8)
        emotion = Emotion(
            category=EmotionCategory.JOY,
            intensity=0.95,
            affective_state=state
        )
        
        weight = emotion_sim.calculate_emotional_weight(
            memory_id="mem1",
            emotion=emotion
        )
        
        # High arousal, high valence, high intensity → high weight
        assert weight > 0.7
    
    def test_calculate_emotional_weight_low_intensity(self, emotion_sim):
        """Test low-intensity emotion produces moderate weight"""
        state = AffectiveState(valence=0.2, arousal=0.1, dominance=0.0)
        emotion = Emotion(
            category=EmotionCategory.NEUTRAL,
            intensity=0.2,
            affective_state=state
        )
        
        weight = emotion_sim.calculate_emotional_weight(
            memory_id="mem2",
            emotion=emotion
        )
        
        # Low arousal, low valence, low intensity → lower weight
        assert weight < 0.5
    
    def test_get_memory_emotional_weight(self, emotion_sim):
        """Test retrieving stored emotional weight"""
        # Store a weight
        emotion_sim.emotional_memory_weights["mem1"] = 0.85
        
        # Retrieve it
        weight = emotion_sim.get_memory_emotional_weight("mem1")
        assert weight == 0.85
        
        # Non-existent memory returns default
        default_weight = emotion_sim.get_memory_emotional_weight("nonexistent")
        assert default_weight == 0.5
    
    def test_mood_congruent_bias(self, emotion_sim):
        """Test mood-congruent retrieval bias"""
        # Set positive mood
        emotion_sim.mood.current = AffectiveState(
            valence=0.7,
            arousal=0.3,
            dominance=0.5
        )
        
        bias = emotion_sim.get_mood_congruent_bias()
        
        # Bias should be positive (toward positive memories)
        assert bias > 0
    
    # ========================================================================
    # Emotion Queries Tests
    # ========================================================================
    
    def test_get_dominant_emotion(self, emotion_sim):
        """Test getting dominant emotion"""
        # Create emotions with different intensities
        state1 = AffectiveState(valence=0.5, arousal=0.5, dominance=0.5)
        state2 = AffectiveState(valence=-0.5, arousal=0.7, dominance=0.4)
        
        e1 = Emotion(EmotionCategory.JOY, intensity=0.6, affective_state=state1)
        e2 = Emotion(EmotionCategory.ANGER, intensity=0.9, affective_state=state2)
        
        emotion_sim.active_emotions = [e1, e2]
        
        dominant = emotion_sim.get_dominant_emotion()
        
        assert dominant.category == EmotionCategory.ANGER  # Higher intensity
        assert dominant.intensity == 0.9
    
    def test_get_active_emotions_filters_inactive(self, emotion_sim):
        """Test get_active_emotions filters out low-intensity emotions"""
        state = AffectiveState(valence=0.5, arousal=0.5, dominance=0.5)
        
        active = Emotion(EmotionCategory.JOY, intensity=0.5, affective_state=state)
        inactive = Emotion(EmotionCategory.SADNESS, intensity=0.05, affective_state=state)
        
        emotion_sim.active_emotions = [active, inactive]
        
        active_list = emotion_sim.get_active_emotions()
        
        assert len(active_list) == 1
        assert active_list[0].category == EmotionCategory.JOY
    
    def test_get_emotional_state_summary(self, emotion_sim):
        """Test getting comprehensive emotional state summary"""
        # Generate some emotions
        context = {'progress': 0.9, 'strength': 0.8}
        emotion_sim.appraise_context(context, AppraisalType.GOAL_PROGRESS)
        
        summary = emotion_sim.get_emotional_state_summary()
        
        assert 'mood' in summary
        assert 'dominant_emotion' in summary
        assert 'active_emotion_count' in summary
        assert 'recent_emotions' in summary
    
    # ========================================================================
    # Emotion Decay Tests
    # ========================================================================
    
    def test_emotion_decay(self, emotion_sim):
        """Test emotions decay over time"""
        state = AffectiveState(valence=0.8, arousal=0.7, dominance=0.6)
        emotion = Emotion(
            category=EmotionCategory.JOY,
            intensity=0.9,
            affective_state=state
        )
        
        emotion_sim.active_emotions = [emotion]
        
        # Simulate 10 minutes passing
        removed_count = emotion_sim.decay_emotions(time_elapsed=timedelta(minutes=10))
        
        # Emotion should have decayed
        assert emotion.intensity < 0.9
        # Should not have been removed yet (still above 0.1 threshold)
        assert removed_count == 0
    
    def test_emotion_decay_removes_inactive(self, emotion_sim):
        """Test decay removes inactive emotions"""
        state = AffectiveState(valence=0.5, arousal=0.5, dominance=0.5)
        
        # Create emotion with very low intensity
        weak_emotion = Emotion(
            category=EmotionCategory.JOY,
            intensity=0.15,
            affective_state=state
        )
        
        emotion_sim.active_emotions = [weak_emotion]
        
        # Decay should remove it
        removed_count = emotion_sim.decay_emotions(time_elapsed=timedelta(minutes=5))
        
        assert len(emotion_sim.active_emotions) == 0
        assert removed_count == 1
    
    def test_emotion_decay_empty_list(self, emotion_sim):
        """Test decay with no active emotions returns 0"""
        removed_count = emotion_sim.decay_emotions()
        
        assert removed_count == 0
        assert len(emotion_sim.active_emotions) == 0
    
    # ========================================================================
    # Persistence Tests
    # ========================================================================
    
    def test_save_and_load_state(self, temp_dir):
        """Test emotional state persistence"""
        # Create first instance with emotions
        sim1 = EmotionSimulator(persistence_dir=temp_dir)
        
        context = {'progress': 0.9, 'strength': 0.8}
        sim1.appraise_context(context, AppraisalType.GOAL_PROGRESS)
        sim1.calculate_emotional_weight("mem1")
        
        # Save state
        sim1.save_state()
        
        # Create new instance (should load state)
        sim2 = EmotionSimulator(persistence_dir=temp_dir)
        
        # Verify state loaded
        assert len(sim2.emotion_history) > 0
        assert "mem1" in sim2.emotional_memory_weights
    
    def test_save_state_no_persistence_dir(self):
        """Test save_state with no persistence directory"""
        sim = EmotionSimulator(persistence_dir=None)
        
        # Should log warning but not crash
        sim.save_state()
    
    # ========================================================================
    # Statistics Tests
    # ========================================================================
    
    def test_get_statistics(self, emotion_sim):
        """Test statistics generation"""
        # Generate some emotions
        context1 = {'progress': 0.9, 'strength': 0.8}
        emotion_sim.appraise_context(context1, AppraisalType.GOAL_PROGRESS)
        
        context2 = {'quality': 0.8, 'strength': 0.7}
        emotion_sim.appraise_context(context2, AppraisalType.SOCIAL_CONNECTION)
        
        # Add emotional memory weight
        emotion_sim.calculate_emotional_weight("mem1")
        
        stats = emotion_sim.get_statistics()
        
        assert stats['active_emotions'] >= 1
        assert stats['emotion_history_size'] >= 2
        assert stats['weighted_memories'] >= 1
        assert 'mood_deviation_from_baseline' in stats
        assert 'dominant_emotion' in stats


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_blend_affective_states(self):
        """Test blending helper method"""
        sim = EmotionSimulator()
        
        state1 = AffectiveState(valence=1.0, arousal=0.5, dominance=0.0)
        state2 = AffectiveState(valence=-1.0, arousal=-0.5, dominance=1.0)
        
        # 50/50 blend should average the states
        blended = sim._blend_affective_states(state1, state2, weight1=0.5)
        
        assert abs(blended.valence - 0.0) < 0.01
        assert abs(blended.arousal - 0.0) < 0.01
        assert abs(blended.dominance - 0.5) < 0.01
    
    def test_emotion_history_size_limit(self):
        """Test emotion history is limited to 100 entries"""
        sim = EmotionSimulator()
        
        # Generate 150 emotions
        for i in range(150):
            context = {'progress': 0.8, 'strength': 0.7}
            sim.appraise_context(context, AppraisalType.GOAL_PROGRESS)
        
        # History should be capped at 100
        assert len(sim.emotion_history) == 100
        
    def test_extreme_valence(self):
        """Test boundary valence values"""
        state1 = AffectiveState(valence=1.0, arousal=0.0, dominance=0.0)
        state2 = AffectiveState(valence=-1.0, arousal=0.0, dominance=0.0)
        
        assert state1.valence == 1.0
        assert state2.valence == -1.0
    
    def test_rapid_emotional_transitions(self):
        """Test rapid emotional state changes"""
        sim = EmotionSimulator()
        
        # Generate multiple emotions rapidly
        for i in range(10):
            context = {'progress': 0.5 + (i * 0.05), 'strength': 0.8}
            sim.appraise_context(context, AppraisalType.GOAL_PROGRESS)
        
        # Should have multiple emotions in history
        assert len(sim.emotion_history) >= 5
    
    def test_conflicting_emotions(self):
        """Test handling of conflicting emotions"""
        sim = EmotionSimulator()
        
        # Generate joy
        context1 = {'progress': 0.95, 'strength': 0.9}
        joy = sim.appraise_context(context1, AppraisalType.GOAL_PROGRESS)
        
        # Generate sadness
        context2 = {'quality': 0.1, 'strength': 0.8}
        sadness = sim.appraise_context(context2, AppraisalType.SOCIAL_CONNECTION)
        
        # Both should coexist in active emotions
        active = sim.get_active_emotions()
        categories = [e.category for e in active]
        
        assert EmotionCategory.JOY in categories
        assert EmotionCategory.SADNESS in categories
    
    def test_zero_intensity_emotion(self):
        """Test emotion with zero intensity"""
        state = AffectiveState(valence=0.0, arousal=0.0, dominance=0.0)
        emotion = Emotion(
            category=EmotionCategory.NEUTRAL,
            intensity=0.0,
            affective_state=state
        )
        
        assert not emotion.is_active()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
