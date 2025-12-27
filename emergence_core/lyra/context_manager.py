"""
Context management system for adaptive consciousness.

This module implements context tracking, adaptation, and seamless transitions
to support Element 3: Context Setting and Adaptation.

The system maintains multiple context dimensions:
- Conversation context (topic flow, history)
- Emotional context (current emotional state)
- Task context (current goal/activity)
- Temporal context (time of day, session duration)
- Interaction patterns (user preferences, communication style)
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import json
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ContextWindow:
    """
    Manages a sliding window of recent context with automatic relevance decay.
    
    Attributes:
        max_size: Maximum number of items to keep
        window: Deque storing context items
        decay_rate: How quickly older items lose relevance (0-1)
    """
    
    def __init__(self, max_size: int = 20, decay_rate: float = 0.1):
        """Initialize ContextWindow with validation."""
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if not 0 <= decay_rate <= 1:
            raise ValueError(f"decay_rate must be in [0, 1], got {decay_rate}")
        
        self.max_size = max_size
        self.window = deque(maxlen=max_size)
        self.decay_rate = decay_rate
        self._last_update_time = None  # Cache for optimization
        
    def add(self, item: Dict[str, Any]):
        """Add a new context item with timestamp and initial relevance."""
        if not isinstance(item, dict):
            raise TypeError(f"Item must be dict, got {type(item).__name__}")
        
        now = datetime.now()
        item["added_at"] = now.isoformat()
        item["relevance"] = 1.0
        self.window.append(item)
        self._last_update_time = now  # Cache current time
        self._update_relevance(now)
    
    def _update_relevance(self, now: Optional[datetime] = None):
        """Decay relevance of older items based on time elapsed.
        
        Args:
            now: Current datetime (cached for efficiency)
        """
        if now is None:
            now = datetime.now()
        
        # Skip if no time passed since last update (optimization)
        if self._last_update_time and (now - self._last_update_time).total_seconds() < 1:
            return
        
        self._last_update_time = now
        
        for item in self.window:
            try:
                added_at = datetime.fromisoformat(item["added_at"])
                age_seconds = (now - added_at).total_seconds()
                # Exponential decay: relevance = e^(-decay_rate * age_minutes)
                item["relevance"] = np.exp(-self.decay_rate * (age_seconds / 60))
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid item in context window: {e}")
                item["relevance"] = 0.0  # Mark invalid items as irrelevant
    
    def get_relevant(self, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Get items above relevance threshold.
        
        Args:
            threshold: Minimum relevance score (0-1)
            
        Returns:
            List of context items with relevance >= threshold
        """
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        
        now = datetime.now()
        self._update_relevance(now)
        return [item for item in self.window if item.get("relevance", 0) >= threshold]
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all items in window."""
        return list(self.window)
    
    def clear(self):
        """Clear the context window."""
        self.window.clear()


class ContextManager:
    """
    Manages multi-dimensional context for adaptive consciousness.
    
    Tracks and adapts to:
    - Conversation flow and topic changes
    - Emotional state evolution
    - Task/goal progression
    - Temporal patterns
    - User interaction preferences
    """
    
    def __init__(self, persistence_dir: str = "context_state"):
        """
        Initialize context management system.
        
        Args:
            persistence_dir: Directory to save/load context state
        """
        self.persistence_dir = Path(persistence_dir)
        self.persistence_dir.mkdir(exist_ok=True, parents=True)
        
        # Multi-dimensional context tracking
        self.conversation_context = ContextWindow(max_size=20, decay_rate=0.1)
        self.emotional_context = ContextWindow(max_size=10, decay_rate=0.05)
        self.task_context = ContextWindow(max_size=5, decay_rate=0.2)
        
        # Current context state
        self.current_topic = None
        self.previous_topic = None
        self.topic_transition_count = 0
        
        # Interaction patterns (learning from experience)
        self.interaction_patterns = {
            "preferred_detail_level": "moderate",  # brief, moderate, detailed
            "communication_style": "balanced",  # formal, casual, balanced
            "topic_preferences": {},  # topic -> engagement_score
            "successful_contexts": [],  # contexts that led to good interactions
        }
        
        # Session tracking
        self.session_start = datetime.now()
        self.interaction_count = 0
        
        logger.info("Context manager initialized")
    
    def update_conversation_context(
        self, 
        user_input: str, 
        system_response: str,
        detected_topic: Optional[str] = None,
        emotional_tone: Optional[List[str]] = None
    ):
        """
        Update conversation context with new interaction.
        
        Args:
            user_input: User's message
            system_response: System's response
            detected_topic: Detected topic/theme of conversation
            emotional_tone: Detected emotional tones
        """
        # Create conversation entry
        entry = {
            "type": "conversation",
            "user_input": user_input,
            "system_response": system_response,
            "topic": detected_topic or self.current_topic,
            "emotional_tone": emotional_tone or [],
            "timestamp": datetime.now().isoformat(),
            "interaction_number": self.interaction_count
        }
        
        # Add to conversation window
        self.conversation_context.add(entry)
        
        # Detect topic change
        if detected_topic and detected_topic != self.current_topic:
            self._handle_topic_transition(detected_topic)
        
        # Update emotional context if provided
        if emotional_tone:
            self.emotional_context.add({
                "type": "emotion",
                "tones": emotional_tone,
                "timestamp": datetime.now().isoformat()
            })
        
        self.interaction_count += 1
        logger.debug(f"Updated conversation context (interaction {self.interaction_count})")
    
    def _handle_topic_transition(self, new_topic: str):
        """
        Handle smooth transition between topics.
        
        Args:
            new_topic: The new topic being discussed
        """
        self.previous_topic = self.current_topic
        self.current_topic = new_topic
        self.topic_transition_count += 1
        
        logger.info(f"Topic transition: '{self.previous_topic}' -> '{new_topic}' (transition #{self.topic_transition_count})")
        
        # Add transition marker to conversation context
        self.conversation_context.add({
            "type": "topic_transition",
            "from_topic": self.previous_topic,
            "to_topic": new_topic,
            "timestamp": datetime.now().isoformat()
        })
    
    def detect_context_shift(
        self, 
        new_input: str, 
        current_context: List[Dict[str, Any]],
        shift_threshold: float = 0.3
    ) -> Tuple[bool, float]:
        """
        Detect if a significant context shift has occurred.
        
        Uses word overlap similarity between new input and recent context.
        For production, consider semantic embeddings for better accuracy.
        
        Args:
            new_input: New user input
            current_context: Recent conversation context
            shift_threshold: Similarity below this triggers shift detection
            
        Returns:
            Tuple of (shift_detected: bool, similarity_score: float)
        """
        if not new_input or not current_context:
            return False, 1.0
        
        # Extract recent conversation inputs (last 5)
        recent_inputs = self._extract_recent_inputs(current_context, max_count=5)
        if not recent_inputs:
            return False, 1.0
        
        # Calculate similarity using word overlap
        similarity = self._calculate_word_similarity(new_input, recent_inputs)
        
        # Detect shift based on threshold
        shift_detected = similarity < shift_threshold
        
        if shift_detected:
            logger.info(f"Context shift detected (similarity: {similarity:.2f})")
        
        return shift_detected, similarity
    
    def _extract_recent_inputs(self, context: List[Dict[str, Any]], max_count: int = 5) -> List[str]:
        """Extract recent user inputs from context.
        
        Args:
            context: List of context items
            max_count: Maximum number of inputs to extract
            
        Returns:
            List of recent user input strings
        """
        recent = [
            ctx.get("user_input", "") 
            for ctx in context[-max_count:] 
            if ctx.get("type") == "conversation" and ctx.get("user_input")
        ]
        return [inp for inp in recent if inp.strip()]  # Filter empty strings
    
    def _calculate_word_similarity(self, new_text: str, reference_texts: List[str]) -> float:
        """Calculate word overlap similarity (Jaccard index).
        
        Args:
            new_text: New text to compare
            reference_texts: List of reference texts
            
        Returns:
            Similarity score (0-1), where 1 is identical
        """
        # Tokenize and normalize
        new_words = set(new_text.lower().split())
        reference_words = set(" ".join(reference_texts).lower().split())
        
        # Handle empty sets
        if not new_words or not reference_words:
            return 1.0  # Consider empty as "no change"
        
        # Jaccard similarity: intersection / union
        intersection = len(new_words & reference_words)
        union = len(new_words | reference_words)
        
        return intersection / union if union > 0 else 0.0
    
    def get_adapted_context(
        self, 
        query: str,
        context_type: str = "conversation",
        max_items: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get context adapted to current situation.
        
        Filters and weights context based on:
        - Relevance decay over time
        - Similarity to current query
        - Context type priority
        
        Args:
            query: Current query/input
            context_type: Type of context to prioritize
            max_items: Maximum items to return
            
        Returns:
            List of relevant context items
        """
        # Get relevant items from appropriate context window
        if context_type == "conversation":
            items = self.conversation_context.get_relevant(threshold=0.2)
        elif context_type == "emotional":
            items = self.emotional_context.get_relevant(threshold=0.3)
        elif context_type == "task":
            items = self.task_context.get_relevant(threshold=0.4)
        else:
            # Combine all contexts
            items = (
                self.conversation_context.get_relevant(threshold=0.2) +
                self.emotional_context.get_relevant(threshold=0.3) +
                self.task_context.get_relevant(threshold=0.4)
            )
        
        # Sort by relevance
        items.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        
        return items[:max_items]
    
    def learn_from_interaction(
        self, 
        user_feedback: Optional[str] = None,
        engagement_level: Optional[float] = None,
        topic: Optional[str] = None
    ):
        """
        Learn and adapt from interaction feedback.
        
        Updates interaction patterns based on:
        - Explicit user feedback
        - Implicit engagement signals
        - Topic preferences
        
        Args:
            user_feedback: Explicit feedback (positive/negative)
            engagement_level: Implicit engagement score (0-1)
            topic: Topic of the interaction
        """
        # Update topic preferences
        if topic and engagement_level is not None:
            if topic not in self.interaction_patterns["topic_preferences"]:
                self.interaction_patterns["topic_preferences"][topic] = []
            
            self.interaction_patterns["topic_preferences"][topic].append(engagement_level)
            
            # Keep only recent scores (last 10)
            self.interaction_patterns["topic_preferences"][topic] = \
                self.interaction_patterns["topic_preferences"][topic][-10:]
        
        # Store successful contexts for future reference
        if engagement_level and engagement_level > 0.7:
            current_state = {
                "topic": self.current_topic,
                "emotional_tones": [
                    item.get("tones", []) 
                    for item in self.emotional_context.get_relevant()
                ],
                "conversation_length": len(self.conversation_context.get_all()),
                "engagement": engagement_level,
                "timestamp": datetime.now().isoformat()
            }
            self.interaction_patterns["successful_contexts"].append(current_state)
            
            # Keep only recent successful contexts
            self.interaction_patterns["successful_contexts"] = \
                self.interaction_patterns["successful_contexts"][-20:]
        
        logger.debug(f"Learning updated: engagement={engagement_level}, topic={topic}")
    
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current context state.
        
        Returns:
            Dictionary with context statistics and state
        """
        return {
            "current_topic": self.current_topic,
            "previous_topic": self.previous_topic,
            "topic_transitions": self.topic_transition_count,
            "session_duration_minutes": (datetime.now() - self.session_start).total_seconds() / 60,
            "interaction_count": self.interaction_count,
            "conversation_context_size": len(self.conversation_context.get_all()),
            "emotional_context_size": len(self.emotional_context.get_all()),
            "task_context_size": len(self.task_context.get_all()),
            "learned_topic_preferences": {
                topic: np.mean(scores) 
                for topic, scores in self.interaction_patterns["topic_preferences"].items()
                if scores
            },
            "interaction_patterns": {
                "preferred_detail_level": self.interaction_patterns["preferred_detail_level"],
                "communication_style": self.interaction_patterns["communication_style"]
            }
        }
    
    def save_context_state(self):
        """Save current context state to disk."""
        state_file = self.persistence_dir / "context_state.json"
        
        try:
            state = {
                "current_topic": self.current_topic,
                "previous_topic": self.previous_topic,
                "topic_transition_count": self.topic_transition_count,
                "session_start": self.session_start.isoformat(),
                "interaction_count": self.interaction_count,
                "interaction_patterns": self.interaction_patterns,
                "conversation_context": self.conversation_context.get_all(),
                "emotional_context": self.emotional_context.get_all(),
                "task_context": self.task_context.get_all(),
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Context state saved to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save context state: {e}")
    
    def load_context_state(self):
        """Load context state from disk."""
        state_file = self.persistence_dir / "context_state.json"
        
        if not state_file.exists():
            logger.info("No saved context state found")
            return
        
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.current_topic = state.get("current_topic")
            self.previous_topic = state.get("previous_topic")
            self.topic_transition_count = state.get("topic_transition_count", 0)
            self.session_start = datetime.fromisoformat(state.get("session_start"))
            self.interaction_count = state.get("interaction_count", 0)
            self.interaction_patterns = state.get("interaction_patterns", self.interaction_patterns)
            
            # Restore context windows
            for item in state.get("conversation_context", []):
                self.conversation_context.window.append(item)
            for item in state.get("emotional_context", []):
                self.emotional_context.window.append(item)
            for item in state.get("task_context", []):
                self.task_context.window.append(item)
            
            logger.info(f"Context state loaded from {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to load context state: {e}")
    
    def reset_session(self):
        """Reset context for new session while preserving learned patterns."""
        logger.info("Resetting session context")
        
        # Save current state before reset
        self.save_context_state()
        
        # Clear context windows
        self.conversation_context.clear()
        self.emotional_context.clear()
        self.task_context.clear()
        
        # Reset session tracking
        self.current_topic = None
        self.previous_topic = None
        self.topic_transition_count = 0
        self.session_start = datetime.now()
        self.interaction_count = 0
        
        # Keep learned interaction patterns
        # (don't reset interaction_patterns)
