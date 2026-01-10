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

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque
from enum import Enum
import logging
import numpy as np
from numpy.typing import NDArray

from .workspace import WorkspaceSnapshot, Goal
from .action import Action, ActionType
from .emotional_modulation import EmotionalModulation, ProcessingParams

# Configure logging
logger = logging.getLogger(__name__)


class EmotionCategory(Enum):
    """
    Primary emotion categories mapped from VAD space.
    
    Based on Plutchik's wheel and VAD emotion mappings:
    - JOY: High valence, high arousal (happy, excited)
    - SADNESS: Low valence, low arousal (sad, melancholy)
    - ANGER: Low valence, high arousal, high dominance (angry, furious)
    - FEAR: Low valence, high arousal, low dominance (afraid, anxious)
    - SURPRISE: Neutral valence, high arousal (surprised, astonished)
    - DISGUST: Low valence, low arousal (disgusted, repulsed)
    - CONTENTMENT: Mid-high valence, low arousal (calm, peaceful)
    - ANTICIPATION: Mid valence, mid arousal, high dominance (expectant)
    """
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CONTENTMENT = "contentment"
    ANTICIPATION = "anticipation"


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
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance
        }


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
        config: Optional[Dict] = None
    ) -> None:
        """
        Initialize the affect subsystem.

        Args:
            config: Optional configuration dict with:
                - baseline: Dict with valence, arousal, dominance baseline values
                - decay_rate: Rate of return to baseline (0.0-1.0)
                - sensitivity: How strongly events affect emotions (0.0-1.0)
                - history_size: Number of emotional states to maintain
                - enable_modulation: Whether to enable emotional modulation (default: True)
        """
        self.config = config or {}
        
        # Baseline emotional state (slightly positive, mild activation, moderate agency)
        self.baseline = self.config.get("baseline", {
            "valence": 0.1,   # Slightly positive default
            "arousal": 0.3,   # Mild activation
            "dominance": 0.6  # Moderate agency
        })
        
        # Current emotional state (start at baseline)
        self.valence = self.baseline["valence"]
        self.arousal = self.baseline["arousal"]
        self.dominance = self.baseline["dominance"]
        
        # Parameters
        self.decay_rate = self.config.get("decay_rate", 0.05)  # 5% per cycle
        self.sensitivity = self.config.get("sensitivity", 0.3)
        
        # History tracking (using deque for efficient append/pop)
        history_size = self.config.get("history_size", 100)
        self.emotion_history: deque = deque(maxlen=history_size)
        
        # Initialize emotional modulation system
        enable_modulation = self.config.get("enable_modulation", True)
        self.emotional_modulation = EmotionalModulation(enabled=enable_modulation)
        
        logger.info(f"✅ AffectSubsystem initialized with baseline: {self.baseline}, modulation: {enable_modulation}")

    
    def compute_update(self, snapshot: WorkspaceSnapshot) -> Dict[str, float]:
        """
        Compute emotional state update for current cycle.
        
        Calculates emotional changes based on:
        - Goal progress (success/failure)
        - Percept content (emotional stimuli)
        - Action outcomes
        - Meta-cognitive percepts
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            Dict with updated valence, arousal, dominance values
        """
        # Calculate deltas from different sources
        goal_deltas = self._update_from_goals(snapshot.goals)
        percept_deltas = self._update_from_percepts(snapshot.percepts)
        
        # Extract recent actions from metadata if available
        recent_actions = []
        if hasattr(snapshot, 'metadata') and isinstance(snapshot.metadata, dict):
            recent_actions = snapshot.metadata.get("recent_actions", [])
        action_deltas = self._update_from_actions(recent_actions)
        
        # Combine deltas (weighted)
        total_delta = {
            "valence": (
                goal_deltas["valence"] * 0.4 +
                percept_deltas["valence"] * 0.4 +
                action_deltas["valence"] * 0.2
            ) * self.sensitivity,
            
            "arousal": (
                goal_deltas["arousal"] * 0.3 +
                percept_deltas["arousal"] * 0.5 +
                action_deltas["arousal"] * 0.2
            ) * self.sensitivity,
            
            "dominance": (
                goal_deltas["dominance"] * 0.3 +
                percept_deltas["dominance"] * 0.2 +
                action_deltas["dominance"] * 0.5
            ) * self.sensitivity
        }
        
        # Apply deltas
        self.valence = np.clip(self.valence + total_delta["valence"], -1.0, 1.0)
        self.arousal = np.clip(self.arousal + total_delta["arousal"], 0.0, 1.0)
        self.dominance = np.clip(self.dominance + total_delta["dominance"], 0.0, 1.0)
        
        # Apply decay toward baseline
        self._apply_decay()
        
        # Record state
        state = EmotionalState(
            valence=self.valence,
            arousal=self.arousal,
            dominance=self.dominance,
            timestamp=datetime.now()
        )
        self.emotion_history.append(state)
        
        logger.debug(f"Emotion update: V={self.valence:.2f}, "
                    f"A={self.arousal:.2f}, D={self.dominance:.2f} "
                    f"({self.get_emotion_label()})")
        
        return state.to_dict()
    
    def _update_from_goals(self, goals: List[Goal]) -> Dict[str, float]:
        """
        Compute emotional impact of goal states.
        
        Enhanced appraisal including:
        - Goal achievement → joy
        - Goal failure → sadness
        - Goal progress tracking → anticipation/disappointment
        
        Args:
            goals: List of current goals
            
        Returns:
            Dict with valence, arousal, dominance deltas
        """
        deltas = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        
        if not goals:
            # No goals = slight decrease in arousal and dominance
            deltas["arousal"] = -0.1
            deltas["dominance"] = -0.05
            return deltas
        
        # Goal progress
        avg_progress = np.mean([g.progress for g in goals])
        deltas["valence"] = (avg_progress - 0.5) * 0.3  # Progress = positive
        
        # Goal quantity
        num_goals = len(goals)
        if num_goals > 3:
            deltas["arousal"] = 0.2  # Many goals = high arousal
        
        # High-priority goals
        high_priority_goals = [g for g in goals if g.priority > 0.8]
        if high_priority_goals:
            deltas["arousal"] += 0.15
            deltas["dominance"] += 0.1  # Important goals = agency
        
        # Goal achievement (progress = 1.0) → JOY
        completed = [g for g in goals if g.progress >= 1.0]
        if completed:
            # Joy: high valence, high arousal, high dominance
            deltas["valence"] += 0.4 * len(completed)
            deltas["arousal"] += 0.3 * len(completed)
            deltas["dominance"] += 0.25 * len(completed)
        
        # Goal failure (progress decreased) → SADNESS
        # Check metadata for failed goals
        failed_goals = [g for g in goals if g.metadata.get("failed", False)]
        if failed_goals:
            # Sadness: low valence, low arousal, low dominance
            deltas["valence"] -= 0.4 * len(failed_goals)
            deltas["arousal"] -= 0.2 * len(failed_goals)
            deltas["dominance"] -= 0.2 * len(failed_goals)
        
        return deltas
    
    def _update_from_percepts(self, percepts: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute emotional impact of percepts.
        
        Enhanced appraisal including:
        - Novelty detection → surprise
        - Social feedback → various emotions
        - Value alignment → positive/negative affect
        
        Args:
            percepts: Dict of current percepts (keyed by ID)
            
        Returns:
            Dict with valence, arousal, dominance deltas
        """
        deltas = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        
        if not percepts:
            return deltas
        
        # Emotional keyword detection
        emotional_keywords = {
            "joy": ["happy", "joy", "delighted", "wonderful", "excellent"],
            "sadness": ["sad", "depressed", "melancholy", "disappointed"],
            "anger": ["angry", "furious", "outraged", "irritated"],
            "fear": ["afraid", "anxious", "worried", "scared", "terrified"],
            "surprise": ["surprising", "unexpected", "shocked", "astonished"],
            "disgust": ["disgusting", "repulsive", "revolting", "awful"],
            "positive": ["love", "great", "amazing", "fantastic"],
            "negative": ["terrible", "horrible", "bad", "awful"],
            "high_arousal": ["urgent", "crisis", "emergency", "panic", "exciting"],
            "low_dominance": ["helpless", "overwhelmed", "lost", "confused"]
        }
        
        for percept_id, percept_data in percepts.items():
            # Handle both Percept objects and dict representations
            if isinstance(percept_data, dict):
                raw = percept_data.get("raw", "")
                modality = percept_data.get("modality", "")
                complexity = percept_data.get("complexity", 0)
                metadata = percept_data.get("metadata", {})
            else:
                # Assume it's a Percept object
                raw = getattr(percept_data, "raw", "")
                modality = getattr(percept_data, "modality", "")
                complexity = getattr(percept_data, "complexity", 0)
                metadata = getattr(percept_data, "metadata", {})
            
            text = str(raw).lower()
            
            # Check for specific emotions
            if any(kw in text for kw in emotional_keywords["joy"]):
                deltas["valence"] += 0.3
                deltas["arousal"] += 0.2
            
            if any(kw in text for kw in emotional_keywords["sadness"]):
                deltas["valence"] -= 0.3
                deltas["arousal"] -= 0.1
                deltas["dominance"] -= 0.1
            
            if any(kw in text for kw in emotional_keywords["anger"]):
                deltas["valence"] -= 0.3
                deltas["arousal"] += 0.3
                deltas["dominance"] += 0.2
            
            if any(kw in text for kw in emotional_keywords["fear"]):
                deltas["valence"] -= 0.3
                deltas["arousal"] += 0.4
                deltas["dominance"] -= 0.3
            
            if any(kw in text for kw in emotional_keywords["surprise"]):
                # SURPRISE: neutral valence, very high arousal
                deltas["arousal"] += 0.5
            
            if any(kw in text for kw in emotional_keywords["disgust"]):
                deltas["valence"] -= 0.3
                deltas["arousal"] += 0.1
            
            # General positive/negative
            if any(kw in text for kw in emotional_keywords["positive"]):
                deltas["valence"] += 0.2
            
            if any(kw in text for kw in emotional_keywords["negative"]):
                deltas["valence"] -= 0.2
                deltas["arousal"] += 0.1
            
            if any(kw in text for kw in emotional_keywords["high_arousal"]):
                deltas["arousal"] += 0.3
            
            if any(kw in text for kw in emotional_keywords["low_dominance"]):
                deltas["dominance"] -= 0.2
            
            # Social feedback appraisal
            if "praise" in text or "well done" in text or "good job" in text:
                # Positive social feedback → joy
                deltas["valence"] += 0.3
                deltas["arousal"] += 0.2
                deltas["dominance"] += 0.2
            
            if "criticism" in text or "you failed" in text or "wrong" in text:
                # Negative social feedback → sadness/anger
                deltas["valence"] -= 0.2
                deltas["arousal"] += 0.15
            
            # Novelty detection → SURPRISE
            if metadata.get("novelty", 0) > 0.7 or "unexpected" in text:
                deltas["arousal"] += 0.4
            
            # Value alignment detection
            if metadata.get("value_aligned", False):
                deltas["valence"] += 0.2
                deltas["dominance"] += 0.1
            
            # Introspective percepts
            if modality == "introspection":
                if isinstance(raw, dict):
                    if raw.get("type") == "value_conflict":
                        # Value conflict → disgust/anger
                        deltas["valence"] -= 0.3
                        deltas["arousal"] += 0.2
                        deltas["dominance"] -= 0.1
                    elif raw.get("type") == "protocol_violation":
                        # Protocol violation → shame/fear
                        deltas["valence"] -= 0.25
                        deltas["arousal"] += 0.15
                        deltas["dominance"] -= 0.2
            
            # High complexity percepts increase arousal
            if complexity > 30:
                deltas["arousal"] += 0.1
        
        return deltas
    
    def _update_from_actions(self, actions: List[Action]) -> Dict[str, float]:
        """
        Compute emotional impact of actions taken.
        
        Args:
            actions: List of recent actions
            
        Returns:
            Dict with valence, arousal, dominance deltas
        """
        deltas = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        
        if not actions:
            # No actions = decreased dominance
            deltas["dominance"] = -0.05
            return deltas
        
        for action in actions:
            # Get action type - handle both Action objects and dicts
            if isinstance(action, dict):
                action_type = action.get("type")
                metadata = action.get("metadata", {})
            else:
                action_type = getattr(action, "type", None)
                metadata = getattr(action, "metadata", {})
            
            # Convert string to ActionType if needed
            if isinstance(action_type, str):
                try:
                    action_type = ActionType(action_type)
                except (ValueError, TypeError):
                    continue
            
            # Successful actions
            if action_type == ActionType.SPEAK:
                deltas["arousal"] += 0.05
                deltas["dominance"] += 0.1
            
            elif action_type == ActionType.COMMIT_MEMORY:
                deltas["valence"] += 0.05  # Consolidation = positive
                deltas["dominance"] += 0.05
            
            elif action_type == ActionType.INTROSPECT:
                deltas["arousal"] += 0.1
                deltas["valence"] -= 0.05  # Introspection often follows problems
            
            # Blocked actions (from metadata)
            if metadata.get("blocked"):
                deltas["dominance"] -= 0.15
                deltas["valence"] -= 0.1
        
        return deltas
    
    def _apply_decay(self) -> None:
        """Gradually return emotions to baseline."""
        self.valence = (
            self.valence * (1 - self.decay_rate) +
            self.baseline["valence"] * self.decay_rate
        )
        self.arousal = (
            self.arousal * (1 - self.decay_rate) +
            self.baseline["arousal"] * self.decay_rate
        )
        self.dominance = (
            self.dominance * (1 - self.decay_rate) +
            self.baseline["dominance"] * self.decay_rate
        )
    
    def get_emotion_label(self) -> str:
        """
        Convert VAD to emotion label using primary emotion categories.
        
        Maps continuous VAD state to categorical emotions using
        distance-based classification in VAD space.
        
        Returns:
            String label for current emotional state
        """
        categories = self.get_emotion_categories()
        if categories:
            return categories[0].value  # Return primary emotion
        return "neutral"
    
    def get_emotion_categories(self) -> List[EmotionCategory]:
        """
        Get primary emotion categories from current VAD state.
        
        Maps VAD coordinates to one or more emotion categories.
        Multiple emotions can be active if the state is between categories.
        
        Returns:
            List of EmotionCategory enums, sorted by relevance
        """
        v, a, d = self.valence, self.arousal, self.dominance
        
        # Define emotion prototypes in VAD space
        prototypes = {
            EmotionCategory.JOY: (0.8, 0.7, 0.7),           # High valence, high arousal, high dominance
            EmotionCategory.SADNESS: (-0.6, 0.2, 0.3),      # Low valence, low arousal, low dominance
            EmotionCategory.ANGER: (-0.7, 0.8, 0.8),        # Low valence, high arousal, high dominance
            EmotionCategory.FEAR: (-0.7, 0.8, 0.2),         # Low valence, high arousal, low dominance
            EmotionCategory.SURPRISE: (0.0, 0.9, 0.5),      # Neutral valence, very high arousal
            EmotionCategory.DISGUST: (-0.6, 0.3, 0.6),      # Low valence, low arousal, medium dominance
            EmotionCategory.CONTENTMENT: (0.5, 0.2, 0.6),   # Mid-high valence, low arousal
            EmotionCategory.ANTICIPATION: (0.3, 0.6, 0.7),  # Positive valence, mid arousal, high dominance
        }
        
        # Calculate distances to each prototype
        current = np.array([v, a, d])
        distances = []
        
        for category, prototype in prototypes.items():
            prototype_vec = np.array(prototype)
            distance = np.linalg.norm(current - prototype_vec)
            distances.append((distance, category))
        
        # Sort by distance (closest first)
        distances.sort(key=lambda x: x[0])
        
        # Return emotions within threshold of closest
        threshold = 1.0  # Only include emotions close to current state
        closest_distance = distances[0][0]
        
        active_emotions = [
            category for distance, category in distances
            if distance <= closest_distance + threshold
        ]
        
        return active_emotions if active_emotions else [EmotionCategory.CONTENTMENT]
    
    def get_state(self) -> Dict[str, Any]:
        """
        Return current emotional state with metadata.
        
        Returns:
            Dict containing:
            - VAD values
            - Emotion label
            - History statistics
        """
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "label": self.get_emotion_label(),
            "history_size": len(self.emotion_history),
            "baseline": self.baseline.copy()
        }
    
    def influence_attention(self, base_score: float, percept: Any) -> float:
        """
        Modify attention score based on emotion.
        
        Args:
            base_score: Original attention score
            percept: Percept being scored (Percept object or dict)
            
        Returns:
            Modified attention score
        """
        modifier = 1.0
        
        # Extract percept properties
        if isinstance(percept, dict):
            raw = percept.get("raw", "")
            modality = percept.get("modality", "")
            complexity = percept.get("complexity", 0)
        else:
            raw = getattr(percept, "raw", "")
            modality = getattr(percept, "modality", "")
            complexity = getattr(percept, "complexity", 0)
        
        text = str(raw).lower()
        
        # High arousal boosts urgent/emotional percepts
        if self.arousal > 0.7:
            if complexity > 30:
                modifier *= 1.3
            if "urgent" in text:
                modifier *= 1.4
        
        # Negative valence boosts introspective percepts
        if self.valence < -0.3:
            if modality == "introspection":
                modifier *= 1.2
        
        # Low dominance boosts supportive percepts
        if self.dominance < 0.3:
            if any(kw in text for kw in ["help", "support", "guide"]):
                modifier *= 1.2
        
        return base_score * modifier
    
    def influence_action(self, base_priority: float, action: Any) -> float:
        """
        Modify action priority based on emotion.
        
        Args:
            base_priority: Original action priority
            action: Action being scored (Action object or dict)
            
        Returns:
            Modified action priority
        """
        modifier = 1.0
        
        # Extract action type
        if isinstance(action, dict):
            action_type = action.get("type")
        else:
            action_type = getattr(action, "type", None)
        
        # Convert string to ActionType if needed
        if isinstance(action_type, str):
            try:
                action_type = ActionType(action_type)
            except (ValueError, TypeError):
                return base_priority
        
        # High arousal boosts immediate actions
        if self.arousal > 0.7:
            if action_type in [ActionType.SPEAK, ActionType.TOOL_CALL]:
                modifier *= 1.3
        
        # Low dominance boosts introspection
        if self.dominance < 0.4:
            if action_type == ActionType.INTROSPECT:
                modifier *= 1.4
        
        # Negative valence may delay non-urgent actions
        if self.valence < -0.4:
            if action_type == ActionType.WAIT:
                modifier *= 1.2
        
        return base_priority * modifier
    
    def get_processing_params(self) -> ProcessingParams:
        """
        Get processing parameters modulated by current emotional state.
        
        This method makes emotions functionally efficacious by directly affecting
        cognitive processing parameters BEFORE any LLM invocation.
        
        Returns:
            ProcessingParams with emotionally-modulated values
        """
        return self.emotional_modulation.modulate_processing(
            arousal=self.arousal,
            valence=self.valence,
            dominance=self.dominance
        )
    
    def apply_valence_bias_to_actions(self, actions: List[Any]) -> List[Any]:
        """
        Apply valence-based approach/avoidance bias to actions.
        
        This makes valence functionally modulate action selection BEFORE LLM scoring.
        
        Args:
            actions: List of action objects or dicts
            
        Returns:
            Actions with valence-biased priorities
        """
        return self.emotional_modulation.bias_action_selection(
            actions=actions,
            valence=self.valence
        )
    
    def get_modulation_metrics(self) -> Dict[str, Any]:
        """
        Get metrics tracking emotional modulation effects.
        
        Returns:
            Dictionary of metrics showing how emotions are modulating processing
        """
        return self.emotional_modulation.get_metrics()

