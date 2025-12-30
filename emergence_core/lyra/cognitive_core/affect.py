"""
Affect Subsystem: Emotional state management.

This module implements the AffectSubsystem class, which maintains and updates
emotional state using a 3-dimensional model (valence, arousal, dominance). Emotions
influence attention, memory retrieval, and action selection, providing adaptive
modulation of cognitive processing.

The affect subsystem is responsible for:
- Tracking current emotional state in a continuous space
- Updating emotions based on appraisals of events and states
- Influencing other subsystems through emotional biasing
- Maintaining emotional history and detecting patterns
"""

from __future__ import annotations

from typing import List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from numpy.typing import NDArray


@dataclass
class EmotionalState:
    """
    Represents emotional state in 3D space (VAD model).

    Uses the Valence-Arousal-Dominance (VAD) model, a widely-used framework
    for representing emotional states in a continuous space:

    - Valence: Pleasantness vs. unpleasantness (-1.0 to +1.0)
    - Arousal: Activation level, calm vs. excited (-1.0 to +1.0)
    - Dominance: Sense of control, submissive vs. dominant (-1.0 to +1.0)

    Attributes:
        valence: Emotional valence (-1.0 = negative, +1.0 = positive)
        arousal: Activation level (-1.0 = calm, +1.0 = excited)
        dominance: Sense of control (-1.0 = submissive, +1.0 = dominant)
        timestamp: When this emotional state was recorded
        intensity: Overall emotional intensity (0.0-1.0)
        labels: Optional categorical emotion labels (e.g., "joy", "fear")
    """
    valence: float
    arousal: float
    dominance: float
    timestamp: datetime = None
    intensity: float = 0.0
    labels: List[str] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.labels is None:
            self.labels = []
        # Calculate intensity as distance from neutral (0, 0, 0)
        self.intensity = np.sqrt(self.valence**2 + self.arousal**2 + self.dominance**2) / np.sqrt(3)

    def to_vector(self) -> NDArray[np.float32]:
        """Convert to numpy vector for calculations."""
        return np.array([self.valence, self.arousal, self.dominance], dtype=np.float32)


class AffectSubsystem:
    """
    Maintains and updates emotional state using a 3-dimensional model.

    The AffectSubsystem implements computational emotion using the Valence-Arousal-
    Dominance (VAD) model. It continuously updates emotional state based on appraisals
    of events, percepts, and internal states, and modulates other cognitive processes
    through emotional biasing.

    Key Responsibilities:
    - Maintain current emotional state in continuous VAD space
    - Appraise events and percepts for emotional significance
    - Update emotional state through dynamics (decay, transitions)
    - Influence attention by amplifying emotionally salient content
    - Bias memory retrieval toward mood-congruent memories
    - Modulate action selection based on emotional state
    - Track emotional history and detect patterns over time

    Integration Points:
    - AttentionController: Emotional salience influences attention allocation
    - GlobalWorkspace: Current emotion is part of conscious content
    - ActionSubsystem: Emotion influences action selection and urgency
    - PerceptionSubsystem: Emotion affects interpretation of percepts
    - SelfMonitor: Emotional state is part of self-monitoring
    - CognitiveCore: Emotions are updated in each cognitive cycle

    Emotional Dynamics:
    1. Appraisal: Events are evaluated for emotional significance
       - Goal relevance: Does this help or hinder current goals?
       - Novelty: Is this unexpected or surprising?
       - Control: Do I have agency over this situation?
    2. Update: Emotional state shifts based on appraisal
       - Positive events increase valence
       - Novel/intense events increase arousal
       - Success/control increases dominance
    3. Decay: Emotions gradually return toward baseline (emotional regulation)
    4. Influence: Current emotion modulates other cognitive processes

    The subsystem can represent both simple emotions (happiness, fear, anger)
    and complex emotional states through combinations of VAD dimensions.

    Attributes:
        current_state: Current emotional state in VAD space
        baseline_state: Neutral/resting emotional state (target for decay)
        decay_rate: Rate of return to baseline (emotional regulation)
        emotional_history: Recent emotional states for pattern detection
    """

    def __init__(
        self,
        baseline_valence: float = 0.0,
        baseline_arousal: float = 0.0,
        baseline_dominance: float = 0.0,
        decay_rate: float = 0.1,
        history_size: int = 1000,
    ) -> None:
        """
        Initialize the affect subsystem.

        Args:
            baseline_valence: Resting valence level (-1.0 to +1.0)
            baseline_arousal: Resting arousal level (-1.0 to +1.0)
            baseline_dominance: Resting dominance level (-1.0 to +1.0)
            decay_rate: Rate of return to baseline per update (0.0-1.0)
                Higher values mean faster emotional regulation
            history_size: Number of emotional states to maintain in history
        """
        # Placeholder implementation - will be fully implemented in Phase 2
        self.baseline_valence = baseline_valence
        self.baseline_arousal = baseline_arousal
        self.baseline_dominance = baseline_dominance
        self.decay_rate = decay_rate
        self.history_size = history_size
        
        # Initialize current state at baseline
        self.current_state = EmotionalState(
            valence=baseline_valence,
            arousal=baseline_arousal,
            dominance=baseline_dominance
        )
        self.emotional_history: List[EmotionalState] = []
    
    def compute_update(self, snapshot: Any) -> Dict[str, float]:
        """
        Placeholder: will be implemented in Phase 2.
        
        Computes emotional state update based on workspace snapshot.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            Dict with valence, arousal, dominance values
        """
        return {
            "valence": self.current_state.valence,
            "arousal": self.current_state.arousal,
            "dominance": self.current_state.dominance
        }
