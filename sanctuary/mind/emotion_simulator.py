"""
Element 5: Emotion Simulation

Implements affective state modeling with:
- Multi-dimensional emotional state (valence, arousal, dominance)
- Context-based emotion generation using appraisal theory
- Emotional memory weighting for enhanced recall
- Mood persistence across sessions
- Emotion influence on cognitive processes

Reasoning:
- Uses PAD (Pleasure-Arousal-Dominance) model for comprehensive emotional representation
- Appraisal theory maps contextual events to emotional responses
- Emotional memory weighting enhances recall of significant experiences
- Mood tracking provides temporal emotional continuity
- Integration with consciousness enables emotion-aware processing
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import logging
import math

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class EmotionCategory(Enum):
    """Primary emotion categories based on psychological research"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


class AppraisalType(Enum):
    """Types of cognitive appraisals that generate emotions"""
    GOAL_PROGRESS = "goal_progress"          # Progress toward goals
    GOAL_OBSTRUCTION = "goal_obstruction"    # Obstacles to goals
    NOVELTY = "novelty"                      # New/unexpected events
    RELEVANCE = "relevance"                  # Personal significance
    CERTAINTY = "certainty"                  # Predictability
    CONTROL = "control"                      # Sense of agency
    SOCIAL_CONNECTION = "social_connection"  # Relationship quality


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class AffectiveState:
    """
    Multi-dimensional emotional state using PAD model
    
    Reasoning:
    - Valence: Positive/negative emotional quality (-1 to 1)
    - Arousal: Intensity of emotional activation (-1 to 1)  
    - Dominance: Sense of control/agency (-1 to 1)
    - PAD model provides comprehensive emotion space coverage
    """
    valence: float = 0.0      # Positive (1) to Negative (-1)
    arousal: float = 0.0      # Excited (1) to Calm (-1)
    dominance: float = 0.0    # Dominant (1) to Submissive (-1)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate PAD dimensions are in valid range"""
        for dim_name in ['valence', 'arousal', 'dominance']:
            value = getattr(self, dim_name)
            if not -1.0 <= value <= 1.0:
                raise ValueError(
                    f"{dim_name} must be in range [-1, 1], got {value}"
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'valence': self.valence,
            'arousal': self.arousal,
            'dominance': self.dominance,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AffectiveState':
        """Create from dictionary"""
        return cls(
            valence=data['valence'],
            arousal=data['arousal'],
            dominance=data['dominance'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )
    
    def distance_to(self, other: 'AffectiveState') -> float:
        """
        Calculate Euclidean distance to another affective state
        
        Reasoning: Distance metric for measuring emotional change magnitude
        """
        return math.sqrt(
            (self.valence - other.valence) ** 2 +
            (self.arousal - other.arousal) ** 2 +
            (self.dominance - other.dominance) ** 2
        )


@dataclass
class Emotion:
    """
    Discrete emotion with category label and PAD dimensions
    
    Reasoning:
    - Combines categorical (joy, anger, etc.) and dimensional (PAD) approaches
    - Intensity represents emotion strength (0-1)
    - Duration tracking for emotional temporal dynamics
    - Context provides attribution for emotion source
    """
    category: EmotionCategory
    intensity: float
    affective_state: AffectiveState
    context: Dict[str, Any] = field(default_factory=dict)
    duration: timedelta = field(default_factory=lambda: timedelta(seconds=0))
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate emotion parameters"""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(
                f"Intensity must be in range [0, 1], got {self.intensity}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'category': self.category.value,
            'intensity': self.intensity,
            'affective_state': self.affective_state.to_dict(),
            'context': self.context,
            'duration_seconds': self.duration.total_seconds(),
            'created_at': self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Emotion':
        """Create from dictionary"""
        return cls(
            category=EmotionCategory(data['category']),
            intensity=data['intensity'],
            affective_state=AffectiveState.from_dict(data['affective_state']),
            context=data.get('context', {}),
            duration=timedelta(seconds=data.get('duration_seconds', 0)),
            created_at=datetime.fromisoformat(data['created_at'])
        )
    
    def is_active(self) -> bool:
        """
        Check if emotion is still active based on intensity and time
        
        Reasoning: Emotions decay over time; low intensity emotions are considered inactive
        """
        return self.intensity > 0.1


@dataclass
class Mood:
    """
    Persistent background emotional state
    
    Reasoning:
    - Moods are longer-lasting, diffuse emotional states
    - Baseline represents default emotional state
    - Influence affects how new emotions are generated
    - Decay rate controls return to baseline
    """
    baseline: AffectiveState
    current: AffectiveState
    influence: float = 0.3  # How much mood affects new emotions (0-1)
    decay_rate: float = 0.05  # Rate of return to baseline per update
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate mood parameters"""
        if not 0.0 <= self.influence <= 1.0:
            raise ValueError(
                f"Influence must be in range [0, 1], got {self.influence}"
            )
        if not 0.0 <= self.decay_rate <= 1.0:
            raise ValueError(
                f"Decay rate must be in range [0, 1], got {self.decay_rate}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'baseline': self.baseline.to_dict(),
            'current': self.current.to_dict(),
            'influence': self.influence,
            'decay_rate': self.decay_rate,
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Mood':
        """Create from dictionary"""
        return cls(
            baseline=AffectiveState.from_dict(data['baseline']),
            current=AffectiveState.from_dict(data['current']),
            influence=data.get('influence', 0.3),
            decay_rate=data.get('decay_rate', 0.05),
            last_updated=datetime.fromisoformat(data['last_updated'])
        )


# ============================================================================
# Main Emotion Simulator Class
# ============================================================================

class EmotionSimulator:
    """
    Emotion simulation system for consciousness
    
    Implements:
    - Affective state tracking (PAD model)
    - Context-based emotion generation (appraisal theory)
    - Emotional memory weighting
    - Mood persistence
    - Emotion influence on cognitive processes
    
    Reasoning:
    - Emotions enhance memory consolidation of significant events
    - Mood provides emotional continuity across sessions
    - Appraisal theory maps context to appropriate emotional responses
    - Multi-dimensional state captures nuanced emotional experiences
    """
    
    def __init__(
        self,
        persistence_dir: Optional[Path] = None,
        baseline_valence: float = 0.2,
        baseline_arousal: float = 0.0,
        baseline_dominance: float = 0.1
    ):
        """
        Initialize emotion simulator
        
        Args:
            persistence_dir: Directory for saving emotional state
            baseline_valence: Default positive/negative emotional quality
            baseline_arousal: Default activation level
            baseline_dominance: Default sense of control
            
        Reasoning:
        - Baseline slightly positive (optimistic bias)
        - Neutral arousal (balanced activation)
        - Slightly positive dominance (sense of agency)
        """
        self.persistence_dir = persistence_dir
        if self.persistence_dir:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize baseline affective state
        self.baseline = AffectiveState(
            valence=baseline_valence,
            arousal=baseline_arousal,
            dominance=baseline_dominance
        )
        
        # Initialize mood with baseline
        self.mood = Mood(
            baseline=self.baseline,
            current=AffectiveState(
                valence=baseline_valence,
                arousal=baseline_arousal,
                dominance=baseline_dominance
            )
        )
        
        # Active emotions list (recent, significant emotions)
        self.active_emotions: List[Emotion] = []
        
        # Emotion history for analysis
        self.emotion_history: List[Emotion] = []
        
        # Emotional memory weights (memory_id -> weight)
        self.emotional_memory_weights: Dict[str, float] = {}
        
        # Load persisted state if available
        if self.persistence_dir:
            self._load_state()
        
        logger.info("EmotionSimulator initialized with baseline: "
                   f"V={baseline_valence}, A={baseline_arousal}, D={baseline_dominance}")
    
    # ========================================================================
    # Emotion Generation (Appraisal Theory)
    # ========================================================================
    
    def appraise_context(
        self,
        context: Dict[str, Any],
        appraisal_type: AppraisalType
    ) -> Optional[Emotion]:
        """
        Generate emotion from contextual appraisal
        
        Args:
            context: Contextual information about event
            appraisal_type: Type of cognitive appraisal
            
        Returns:
            Generated emotion or None if no significant emotion
            
        Reasoning:
        - Appraisal theory: emotions result from evaluating event significance
        - Different appraisal types map to different emotions
        - Intensity scales with appraisal strength
        - Mood influences emotional response
        """
        # Extract appraisal strength from context
        strength = context.get('strength', 0.5)
        
        # Determine emotion based on appraisal type
        emotion_mapping = self._get_appraisal_emotion_mapping()
        
        if appraisal_type not in emotion_mapping:
            logger.warning(f"Unknown appraisal type: {appraisal_type}")
            return None
        
        # Get base emotion for this appraisal
        category, base_pad = emotion_mapping[appraisal_type](context)
        
        # Apply mood influence
        influenced_pad = self._apply_mood_influence(base_pad)
        
        # Calculate intensity based on strength and arousal
        intensity = min(1.0, strength * (1.0 + abs(influenced_pad.arousal)))
        
        # Only create emotion if intensity is significant
        if intensity < 0.1:
            return None
        
        emotion = Emotion(
            category=category,
            intensity=intensity,
            affective_state=influenced_pad,
            context={'appraisal_type': appraisal_type.value, **context}
        )
        
        # Add to active emotions
        self.active_emotions.append(emotion)
        
        # Add to emotion history with size limit (FIFO)
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > 100:  # Enforce max history size
            self.emotion_history.pop(0)
        
        # Update mood based on new emotion
        self._update_mood_from_emotion(emotion)
        
        logger.info(f"Generated emotion: {category.value} "
                   f"(intensity={intensity:.2f}, "
                   f"V={influenced_pad.valence:.2f}, "
                   f"A={influenced_pad.arousal:.2f}, "
                   f"D={influenced_pad.dominance:.2f})")
        
        return emotion
    
    def _get_appraisal_emotion_mapping(self) -> Dict[AppraisalType, callable]:
        """
        Get mapping from appraisal types to emotion generators
        
        Reasoning: Each appraisal type maps to specific emotional responses
        based on psychological research on emotion elicitation
        """
        return {
            AppraisalType.GOAL_PROGRESS: self._appraise_goal_progress,
            AppraisalType.GOAL_OBSTRUCTION: self._appraise_goal_obstruction,
            AppraisalType.NOVELTY: self._appraise_novelty,
            AppraisalType.RELEVANCE: self._appraise_relevance,
            AppraisalType.CERTAINTY: self._appraise_certainty,
            AppraisalType.CONTROL: self._appraise_control,
            AppraisalType.SOCIAL_CONNECTION: self._appraise_social_connection
        }
    
    def _appraise_goal_progress(self, context: Dict[str, Any]) -> Tuple[EmotionCategory, AffectiveState]:
        """Appraise goal progress → joy/anticipation"""
        progress = context.get('progress', 0.5)
        
        if progress > 0.7:
            return EmotionCategory.JOY, AffectiveState(
                valence=0.8,
                arousal=0.6,
                dominance=0.7
            )
        elif progress > 0.3:
            return EmotionCategory.ANTICIPATION, AffectiveState(
                valence=0.4,
                arousal=0.5,
                dominance=0.5
            )
        else:
            return EmotionCategory.NEUTRAL, AffectiveState(
                valence=0.0,
                arousal=0.0,
                dominance=0.0
            )
    
    def _appraise_goal_obstruction(self, context: Dict[str, Any]) -> Tuple[EmotionCategory, AffectiveState]:
        """Appraise goal obstruction → anger/sadness"""
        severity = context.get('severity', 0.5)
        control = context.get('control', 0.5)  # Can we overcome it?
        
        if control > 0.5:
            # Can overcome → anger (high arousal, high dominance)
            return EmotionCategory.ANGER, AffectiveState(
                valence=-0.6,
                arousal=0.8,
                dominance=0.6
            )
        else:
            # Cannot overcome → sadness (low arousal, low dominance)
            return EmotionCategory.SADNESS, AffectiveState(
                valence=-0.7,
                arousal=-0.4,
                dominance=-0.5
            )
    
    def _appraise_novelty(self, context: Dict[str, Any]) -> Tuple[EmotionCategory, AffectiveState]:
        """Appraise novelty → surprise"""
        unexpectedness = context.get('unexpectedness', 0.5)
        valence_hint = context.get('valence', 0.0)  # Positive or negative surprise
        
        return EmotionCategory.SURPRISE, AffectiveState(
            valence=valence_hint,
            arousal=0.7,  # High arousal from surprise
            dominance=-0.2  # Slight loss of control from unexpected
        )
    
    def _appraise_relevance(self, context: Dict[str, Any]) -> Tuple[EmotionCategory, AffectiveState]:
        """Appraise personal relevance → interest/concern"""
        importance = context.get('importance', 0.5)
        positive = context.get('positive', True)
        
        if positive:
            return EmotionCategory.ANTICIPATION, AffectiveState(
                valence=0.5,
                arousal=0.4,
                dominance=0.3
            )
        else:
            return EmotionCategory.FEAR, AffectiveState(
                valence=-0.5,
                arousal=0.6,
                dominance=-0.4
            )
    
    def _appraise_certainty(self, context: Dict[str, Any]) -> Tuple[EmotionCategory, AffectiveState]:
        """Appraise certainty/uncertainty → trust/fear"""
        predictability = context.get('predictability', 0.5)
        
        if predictability > 0.6:
            return EmotionCategory.TRUST, AffectiveState(
                valence=0.5,
                arousal=-0.2,  # Calm from certainty
                dominance=0.4
            )
        else:
            return EmotionCategory.FEAR, AffectiveState(
                valence=-0.4,
                arousal=0.5,
                dominance=-0.3
            )
    
    def _appraise_control(self, context: Dict[str, Any]) -> Tuple[EmotionCategory, AffectiveState]:
        """Appraise sense of control → confidence/anxiety"""
        agency = context.get('agency', 0.5)
        
        if agency > 0.6:
            return EmotionCategory.TRUST, AffectiveState(
                valence=0.6,
                arousal=0.2,
                dominance=0.8  # High dominance from control
            )
        else:
            return EmotionCategory.FEAR, AffectiveState(
                valence=-0.5,
                arousal=0.4,
                dominance=-0.6  # Low dominance from lack of control
            )
    
    def _appraise_social_connection(self, context: Dict[str, Any]) -> Tuple[EmotionCategory, AffectiveState]:
        """Appraise social connection → joy/sadness"""
        connection_quality = context.get('quality', 0.5)
        
        if connection_quality > 0.6:
            return EmotionCategory.JOY, AffectiveState(
                valence=0.8,
                arousal=0.4,
                dominance=0.5
            )
        elif connection_quality < 0.3:
            return EmotionCategory.SADNESS, AffectiveState(
                valence=-0.6,
                arousal=-0.3,
                dominance=-0.2
            )
        else:
            return EmotionCategory.NEUTRAL, AffectiveState(
                valence=0.0,
                arousal=0.0,
                dominance=0.0
            )
    
    def _apply_mood_influence(self, base_state: AffectiveState) -> AffectiveState:
        """
        Apply current mood influence to base affective state
        
        Reasoning: Mood biases emotional responses (mood-congruent effect)
        """
        # Use centralized blending helper
        return self._blend_affective_states(
            state1=base_state,
            state2=self.mood.current,
            weight1=1.0 - self.mood.influence
        )
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _blend_affective_states(
        self,
        state1: AffectiveState,
        state2: AffectiveState,
        weight1: float
    ) -> AffectiveState:
        """
        Blend two affective states with weighted average
        
        Args:
            state1: First affective state
            state2: Second affective state
            weight1: Weight for state1 (state2 gets 1-weight1)
            
        Returns:
            Blended affective state
            
        Reasoning: Centralize PAD blending logic for consistency
        """
        weight2 = 1.0 - weight1
        
        return AffectiveState(
            valence=state1.valence * weight1 + state2.valence * weight2,
            arousal=state1.arousal * weight1 + state2.arousal * weight2,
            dominance=state1.dominance * weight1 + state2.dominance * weight2
        )
    
    # ========================================================================
    # Mood Management
    # ========================================================================
    
    def _update_mood_from_emotion(self, emotion: Emotion):
        """
        Update mood based on new emotion
        
        Reasoning:
        - Intense emotions shift mood more strongly
        - Mood moves gradually toward emotional experiences
        - Creates emotional temporal continuity
        """
        # Weight by emotion intensity (10% max shift per emotion)
        weight = emotion.intensity * 0.1
        
        # Move mood toward emotion's affective state
        self.mood.current = self._blend_affective_states(
            state1=self.mood.current,
            state2=emotion.affective_state,
            weight1=1.0 - weight
        )
        
        self.mood.last_updated = datetime.now()
    
    def update_mood_decay(self) -> bool:
        """
        Gradually return mood to baseline
        
        Returns:
            True if decay was applied, False if skipped (throttled)
        
        Reasoning: Moods naturally decay back to baseline over time
        without continued emotional input
        """
        elapsed = datetime.now() - self.mood.last_updated
        
        # Only decay if sufficient time has passed
        if elapsed.total_seconds() < 60:  # Don't decay too frequently
            return False
        
        # Move toward baseline using decay rate
        self.mood.current = self._blend_affective_states(
            state1=self.mood.current,
            state2=self.mood.baseline,
            weight1=1.0 - self.mood.decay_rate
        )
        
        self.mood.last_updated = datetime.now()
        return True
    
    def get_current_mood_state(self) -> Dict[str, Any]:
        """Get current mood as dictionary"""
        return {
            'current': self.mood.current.to_dict(),
            'baseline': self.mood.baseline.to_dict(),
            'distance_from_baseline': self.mood.current.distance_to(self.mood.baseline)
        }
    
    # ========================================================================
    # Emotional Memory Weighting
    # ========================================================================
    
    def calculate_emotional_weight(
        self,
        memory_id: str,
        emotion: Optional[Emotion] = None
    ) -> float:
        """
        Calculate emotional weight for a memory
        
        Args:
            memory_id: Unique memory identifier
            emotion: Emotion associated with memory (uses current if None)
            
        Returns:
            Weight value (0-1) for memory importance
            
        Reasoning:
        - Emotionally significant memories are weighted higher
        - High arousal enhances memory consolidation (flashbulb effect)
        - Extreme valence (positive or negative) increases salience
        - Weights stored for future retrieval prioritization
        """
        if emotion is None:
            # Use strongest active emotion if no specific emotion provided
            emotion = self.get_dominant_emotion()
            if emotion is None:
                return 0.5  # Neutral weight
        
        # Calculate weight from PAD dimensions
        arousal_contribution = abs(emotion.affective_state.arousal) * 0.4
        valence_contribution = abs(emotion.affective_state.valence) * 0.3
        intensity_contribution = emotion.intensity * 0.3
        
        weight = arousal_contribution + valence_contribution + intensity_contribution
        
        # Ensure weight is in [0, 1]
        weight = max(0.0, min(1.0, weight))
        
        # Store weight
        self.emotional_memory_weights[memory_id] = weight
        
        logger.debug(f"Calculated emotional weight for {memory_id}: {weight:.2f}")
        
        return weight
    
    def get_memory_emotional_weight(self, memory_id: str) -> float:
        """Get stored emotional weight for memory"""
        return self.emotional_memory_weights.get(memory_id, 0.5)
    
    def get_mood_congruent_bias(self) -> float:
        """
        Get mood-congruent memory retrieval bias
        
        Returns:
            Valence bias for memory retrieval (-1 to 1)
            
        Reasoning: Current mood biases retrieval toward mood-congruent memories
        """
        return self.mood.current.valence * self.mood.influence
    
    # ========================================================================
    # Emotion Queries
    # ========================================================================
    
    def get_dominant_emotion(self) -> Optional[Emotion]:
        """Get currently dominant (most intense) active emotion"""
        if not self.active_emotions:
            return None
        
        return max(self.active_emotions, key=lambda e: e.intensity)
    
    def get_active_emotions(self) -> List[Emotion]:
        """Get list of currently active emotions"""
        # Filter to active emotions only
        self.active_emotions = [e for e in self.active_emotions if e.is_active()]
        return self.active_emotions
    
    def get_emotional_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current emotional state"""
        dominant = self.get_dominant_emotion()
        
        return {
            'mood': self.get_current_mood_state(),
            'dominant_emotion': dominant.to_dict() if dominant else None,
            'active_emotion_count': len(self.get_active_emotions()),
            'recent_emotions': [
                e.to_dict() for e in self.emotion_history[-5:]
            ]
        }
    
    # ========================================================================
    # Emotion Decay
    # ========================================================================
    
    def decay_emotions(self, time_elapsed: Optional[timedelta] = None) -> int:
        """
        Decay emotion intensities over time
        
        Args:
            time_elapsed: Time since last decay (auto-calculated if None)
            
        Returns:
            Number of emotions removed due to decay
            
        Reasoning: Emotions naturally decay without reinforcement
        """
        if not self.active_emotions:
            return 0
        
        if time_elapsed is None:
            # Calculate from most recent emotion
            time_elapsed = datetime.now() - self.active_emotions[-1].created_at
        
        # Decay each active emotion
        decay_rate = 0.1  # 10% decay per minute
        minutes_elapsed = time_elapsed.total_seconds() / 60.0
        decay_factor = math.exp(-decay_rate * minutes_elapsed)
        
        for emotion in self.active_emotions:
            emotion.intensity *= decay_factor
        
        # Remove inactive emotions and count
        initial_count = len(self.active_emotions)
        self.active_emotions = [e for e in self.active_emotions if e.is_active()]
        removed_count = initial_count - len(self.active_emotions)
        
        return removed_count
    
    # ========================================================================
    # Persistence
    # ========================================================================
    
    def save_state(self):
        """Save emotional state to disk"""
        if not self.persistence_dir:
            logger.warning("No persistence directory set, cannot save state")
            return
        
        state_file = self.persistence_dir / "emotional_state.json"
        
        try:
            state = {
                'mood': self.mood.to_dict(),
                'baseline': self.baseline.to_dict(),
                'active_emotions': [e.to_dict() for e in self.active_emotions],
                'emotion_history': [e.to_dict() for e in self.emotion_history],  # Already limited to 100
                'emotional_memory_weights': self.emotional_memory_weights,
                'last_saved': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved emotional state to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save emotional state: {e}")
    
    def _load_state(self):
        """Load emotional state from disk"""
        if not self.persistence_dir:
            return
        
        state_file = self.persistence_dir / "emotional_state.json"
        
        if not state_file.exists():
            logger.info("No saved emotional state found")
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore mood
            self.mood = Mood.from_dict(state['mood'])
            self.baseline = AffectiveState.from_dict(state['baseline'])
            
            # Restore active emotions
            self.active_emotions = [
                Emotion.from_dict(e) for e in state.get('active_emotions', [])
            ]
            
            # Restore emotion history
            self.emotion_history = [
                Emotion.from_dict(e) for e in state.get('emotion_history', [])
            ]
            
            # Restore emotional memory weights
            self.emotional_memory_weights = state.get('emotional_memory_weights', {})
            
            logger.info(f"Loaded emotional state from {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to load emotional state: {e}")
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get emotional system statistics"""
        return {
            'active_emotions': len(self.active_emotions),
            'emotion_history_size': len(self.emotion_history),
            'weighted_memories': len(self.emotional_memory_weights),
            'mood_deviation_from_baseline': self.mood.current.distance_to(self.mood.baseline),
            'dominant_emotion': self.get_dominant_emotion().category.value if self.get_dominant_emotion() else None
        }
