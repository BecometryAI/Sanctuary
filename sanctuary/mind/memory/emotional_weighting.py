"""
Emotional Weighting Module

Emotional salience affects storage and retrieval priority.
High-emotion memories get preferential treatment.

Author: Sanctuary Team
"""
import logging
import math
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class EmotionalWeighting:
    """
    Manages emotional salience in memory operations.
    
    Responsibilities:
    - Emotional salience scoring
    - High-emotion memories get storage priority
    - Emotional state biases retrieval
    """
    
    def __init__(self):
        """Initialize emotional weighting system."""
        # Emotional intensity weights (can be tuned)
        self.emotion_weights = {
            "joy": 0.8,
            "surprise": 0.9,
            "fear": 1.0,
            "anger": 0.9,
            "sadness": 0.8,
            "trust": 0.7,
            "anticipation": 0.6,
            "disgust": 0.7,
            # Extended tones
            "curious": 0.6,
            "thoughtful": 0.7,
            "engaged": 0.7,
            "reflective": 0.8,
            "excited": 0.9,
            "concerned": 0.8,
            "hopeful": 0.7,
            "grateful": 0.8,
        }
    
    def calculate_salience(self, memory: Dict[str, Any]) -> float:
        """Calculate emotional salience score for a memory."""
        if not memory:
            return 0.5
        
        emotional_tones = memory.get("emotional_tone", [])
        if not emotional_tones or not isinstance(emotional_tones, list):
            return 0.5
        
        # Calculate average weight, filtering invalid entries
        weights = [
            self.emotion_weights.get(tone.lower(), 0.5)
            for tone in emotional_tones
            if isinstance(tone, str) and tone.strip()
        ]
        
        if not weights:
            return 0.5
        
        salience = sum(weights) / len(weights)
        logger.debug(f"Salience {salience:.2f} for tones: {emotional_tones}")
        return salience
    
    def should_prioritize_storage(self, memory: Dict[str, Any], threshold: float = 0.7) -> bool:
        """Determine if a memory should get prioritized storage."""
        if not memory or not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
            return False
        
        salience = self.calculate_salience(memory)
        should_prioritize = salience >= threshold
        
        if should_prioritize:
            logger.info(f"Memory prioritized (salience: {salience:.2f})")
        
        return should_prioritize
    
    def weight_retrieval_results(
        self,
        memories: List[Dict[str, Any]],
        current_emotional_state: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Bias retrieval results based on emotional congruence."""
        if not memories or not current_emotional_state:
            return memories
        
        # Filter and normalize current state
        current_state_lower = [
            tone.lower() for tone in current_emotional_state 
            if isinstance(tone, str) and tone.strip()
        ]
        
        if not current_state_lower:
            return memories
        
        # Filter to only valid dict memories
        valid_memories = [m for m in memories if isinstance(m, dict)]

        # Calculate emotional congruence for each memory
        for memory in valid_memories:
            memory_tones = [
                tone.lower() for tone in memory.get("emotional_tone", [])
                if isinstance(tone, str) and tone.strip()
            ]

            # Calculate overlap
            overlap = len(set(current_state_lower) & set(memory_tones))
            memory["emotional_congruence"] = overlap / len(current_state_lower) if overlap > 0 else 0.0

        # Sort by congruence and timestamp
        valid_memories.sort(
            key=lambda m: (m.get("emotional_congruence", 0), m.get("timestamp", "")),
            reverse=True
        )

        return valid_memories
    
    def get_emotion_weight(self, emotion: str) -> float:
        """
        Get the salience weight for a specific emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Weight value (0.0-1.0)
        """
        return self.emotion_weights.get(emotion.lower(), 0.5)
    
    def update_emotion_weight(self, emotion: str, weight: float) -> None:
        """
        Update the salience weight for an emotion.
        
        Args:
            emotion: Emotion name
            weight: New weight value (0.0-1.0)
        """
        if 0.0 <= weight <= 1.0:
            self.emotion_weights[emotion.lower()] = weight
            logger.info(f"Updated emotion weight: {emotion} -> {weight}")
        else:
            logger.warning(f"Invalid weight value {weight} for emotion {emotion}, must be 0.0-1.0")
    
    def emotional_congruence_pad(
        self,
        current_state: Dict[str, float],
        memory_state: Optional[Dict[str, float]]
    ) -> float:
        """
        Calculate emotional congruence using PAD (Pleasure-Arousal-Dominance) model.
        
        Memories encoded in similar emotional states are easier to retrieve.
        Based on Euclidean distance in PAD space.
        
        Args:
            current_state: Current PAD state with keys 'valence', 'arousal', 'dominance'
            memory_state: Memory's PAD state (same keys), or None if unavailable
            
        Returns:
            Congruence score (0.0-1.0, higher = more congruent)
        """
        if not memory_state or not current_state:
            return 0.5  # Neutral if no state available
        
        # Extract PAD dimensions (with fallbacks)
        current_pleasure = current_state.get("valence", 0.0)
        current_arousal = current_state.get("arousal", 0.0)
        current_dominance = current_state.get("dominance", 0.0)
        
        memory_pleasure = memory_state.get("valence", 0.0)
        memory_arousal = memory_state.get("arousal", 0.0)
        memory_dominance = memory_state.get("dominance", 0.0)
        
        # Calculate Euclidean distance in PAD space
        distance = math.sqrt(
            (current_pleasure - memory_pleasure) ** 2 +
            (current_arousal - memory_arousal) ** 2 +
            (current_dominance - memory_dominance) ** 2
        )
        
        # Max distance in normalized PAD space (each dimension ranges ~-1 to 1)
        # Assuming range is roughly [-1, 1] for valence and [0, 1] for arousal/dominance
        # Conservative estimate: sqrt(2^2 + 1^2 + 1^2) = sqrt(6) ≈ 2.45
        max_distance = math.sqrt(6.0)
        
        # Convert distance to similarity (1.0 = identical, 0.0 = maximally different)
        congruence = 1.0 - (distance / max_distance)
        congruence = max(0.0, min(1.0, congruence))  # Clamp to [0, 1]
        
        logger.debug(f"PAD congruence: {congruence:.3f} (distance: {distance:.3f})")
        return congruence
