"""
Attention Controller: Selective attention mechanism.

This module implements the AttentionController class, which decides what information
gains access to the limited-capacity GlobalWorkspace. It implements selective attention
based on goal relevance, novelty, emotional salience, and resource constraints.

The attention mechanism is crucial for:
- Creating the selective nature of consciousness
- Managing cognitive resource allocation
- Prioritizing information based on multiple factors
- Implementing both top-down (goal-driven) and bottom-up (stimulus-driven) attention
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from .workspace import GlobalWorkspace, Percept

# Configure logging
logger = logging.getLogger(__name__)


# Scoring weights (configurable)
SCORING_WEIGHTS = {
    "goal_relevance": 0.4,
    "novelty": 0.3,
    "emotional_salience": 0.2,
    "recency": 0.1
}


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0.0-1.0, where 1.0 is identical, 0.0 is orthogonal/opposite)
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    # Convert to numpy arrays and reshape for sklearn
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    
    # Use sklearn's cosine_similarity (returns values in [-1, 1])
    similarity = sklearn_cosine(v1, v2)[0][0]
    
    # Clamp to [0, 1] range - negative similarities become 0
    # This makes sense for attention: opposing directions shouldn't get negative scores
    return max(0.0, float(similarity))


def keyword_overlap(text1: str, text2: str) -> float:
    """
    Simple keyword overlap score (0.0-1.0) using Jaccard similarity.
    
    Args:
        text1: First text string
        text2: Second text string
        
    Returns:
        Jaccard similarity score (0.0-1.0)
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple tokenization (lowercase and split)
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    
    # Remove common stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}
    tokens1 = tokens1 - stopwords
    tokens2 = tokens2 - stopwords
    
    # Compute Jaccard similarity
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return intersection / union if union > 0 else 0.0


class AttentionMode(Enum):
    """
    Different modes of attention allocation.

    FOCUSED: Narrow, goal-driven attention on specific targets
    DIFFUSE: Broad, exploratory attention across multiple inputs
    VIGILANT: Heightened alertness for threat or novelty detection
    RELAXED: Low-intensity monitoring during low-demand periods
    """
    FOCUSED = "focused"
    DIFFUSE = "diffuse"
    VIGILANT = "vigilant"
    RELAXED = "relaxed"


@dataclass
class AttentionScore:
    """
    Scores for different factors contributing to attention.

    Attributes:
        goal_relevance: How relevant to current goals (0.0-1.0)
        novelty: How novel or unexpected (0.0-1.0)
        emotional_salience: Emotional importance (0.0-1.0)
        urgency: Time-sensitivity of the information (0.0-1.0)
        total: Weighted sum of all factors
    """
    goal_relevance: float
    novelty: float
    emotional_salience: float
    urgency: float
    total: float


class AttentionController:
    """
    Decides what information enters the GlobalWorkspace based on salience.

    The AttentionController implements selective attention, acting as a gatekeeper
    for the limited-capacity GlobalWorkspace. It evaluates incoming information
    from multiple sources (percepts, memories, emotions, internal states) and
    assigns attention scores based on multiple factors.

    Key Responsibilities:
    - Evaluate attention worthiness of candidate information
    - Implement both top-down (goal-driven) and bottom-up (stimulus-driven) attention
    - Manage attention resources and prevent cognitive overload
    - Track attention history to detect novelty and habituation
    - Dynamically adjust attention mode based on context

    Integration Points:
    - GlobalWorkspace: Selects what content enters the workspace
    - PerceptionSubsystem: Evaluates salience of incoming percepts
    - AffectSubsystem: Uses emotional state to modulate attention
    - CognitiveCore: Receives attention mode adjustments based on system state
    - SelfMonitor: Can redirect attention to internal states when needed

    Attention Mechanisms:
    1. Goal-Driven (Top-Down): Prioritizes information relevant to active goals
    2. Stimulus-Driven (Bottom-Up): Responds to novel, intense, or unexpected stimuli
    3. Emotional Salience: Amplifies attention for emotionally significant content
    4. Habituation: Reduces attention to repeated, non-threatening stimuli
    5. Resource Management: Prevents overload by limiting concurrent attention targets

    The controller uses a weighted scoring system that can be dynamically adjusted
    based on current context, emotional state, and cognitive load. Different attention
    modes shift these weights to support different cognitive strategies.

    Attributes:
        mode: Current attention allocation strategy
        goal_weight: Weight for goal-relevance in scoring (0.0-1.0)
        novelty_weight: Weight for novelty in scoring (0.0-1.0)
        emotion_weight: Weight for emotional salience in scoring (0.0-1.0)
        urgency_weight: Weight for urgency in scoring (0.0-1.0)
    """

    def __init__(
        self,
        attention_budget: int = 100,
        workspace: Optional[GlobalWorkspace] = None,
        affect: Optional[Any] = None,
        initial_mode: AttentionMode = AttentionMode.FOCUSED,
        goal_weight: float = 0.4,
        novelty_weight: float = 0.3,
        emotion_weight: float = 0.2,
        urgency_weight: float = 0.1,
    ) -> None:
        """
        Initialize the attention controller.

        Args:
            attention_budget: Total attention units available per cycle
            workspace: Reference to the workspace for context
            affect: Reference to the affect subsystem for emotional modulation
            initial_mode: Starting attention mode (focused, diffuse, vigilant, relaxed)
            goal_weight: Importance of goal-relevance in attention (0.0-1.0)
            novelty_weight: Importance of novelty in attention (0.0-1.0)
            emotion_weight: Importance of emotional salience in attention (0.0-1.0)
            urgency_weight: Importance of urgency in attention (0.0-1.0)

        Note: Weights should sum to approximately 1.0 for balanced scoring.
        """
        self.attention_budget = attention_budget
        self.initial_budget = attention_budget
        self.workspace = workspace
        self.affect = affect
        self.mode = initial_mode
        
        # Scoring weights
        self.goal_weight = goal_weight
        self.novelty_weight = novelty_weight
        self.emotion_weight = emotion_weight
        self.urgency_weight = urgency_weight
        
        # History tracking
        self.recent_percepts: deque = deque(maxlen=50)
        self.attention_history: List[Dict[str, Any]] = []
        
        # Performance optimization: Relevance cache
        self._relevance_cache: Dict[Tuple[str, str], float] = {}  # (percept_id, goal_id) -> score
        self._cache_max_size = 1000
        
        logger.info(f"AttentionController initialized with budget={attention_budget}, mode={initial_mode.value}")

    def select_for_broadcast(self, candidates: List[Percept]) -> List[Percept]:
        """
        Scores all candidate percepts and selects top-scoring ones within budget.
        
        Optimizations:
        - Early termination when budget is exhausted
        - Batch goal relevance computation for performance
        - Heuristic pre-filtering for large candidate sets
        
        Args:
            candidates: List of candidate percepts to evaluate
            
        Returns:
            Sorted list of selected percepts (highest scoring first)
        """
        if not candidates:
            logger.debug("No candidates to select from")
            return []
        
        # For large candidate sets, use batch goal relevance computation
        if len(candidates) > 10:
            goal_relevance_scores = self._compute_goal_relevance_batch(candidates)
        else:
            goal_relevance_scores = {p.id: self._compute_goal_relevance(p) for p in candidates}
        
        # Score all candidates
        scored_percepts = []
        for percept in candidates:
            # Use pre-computed goal relevance for efficiency
            goal_rel = goal_relevance_scores.get(percept.id, 0.5)
            novelty = self._compute_novelty(percept)
            emotion_sal = self._compute_emotional_salience(percept)
            
            # Recency bonus: newer percepts get slight boost
            time_diff = (datetime.now() - percept.timestamp).total_seconds()
            recency = 0.2 if time_diff < 1.0 else 0.1 if time_diff < 5.0 else 0.0
            
            # Weighted average
            base_score = (
                goal_rel * self.goal_weight +
                novelty * self.novelty_weight +
                emotion_sal * self.emotion_weight +
                recency * self.urgency_weight
            )
            
            # Tool result boost: Tool results get attention priority
            if percept.modality == "tool_result":
                # Base boost for all tool results
                base_score += 0.30
                
                # Additional boost for failed tools (errors need attention)
                if percept.metadata.get("tool_success") is False:
                    base_score += 0.20
            
            # Apply affect modulation if affect subsystem is available
            if self.affect:
                total_score = self.affect.influence_attention(base_score, percept)
            else:
                total_score = base_score
            
            scored_percepts.append((percept, total_score))
        
        # Sort by score (highest first)
        scored_percepts.sort(key=lambda x: x[1], reverse=True)
        
        # Select percepts that fit within budget (with early termination)
        selected = []
        budget_used = 0
        rejected_budget = []
        
        for percept, score in scored_percepts:
            if budget_used + percept.complexity <= self.attention_budget:
                selected.append(percept)
                budget_used += percept.complexity
                
                # Add to recent percepts for novelty detection
                if percept.embedding:
                    self.recent_percepts.append(percept.embedding)
                
                logger.debug(f"Selected percept: {percept.id} (score: {score:.3f}, complexity: {percept.complexity})")
            else:
                # Budget exhausted
                rejected_budget.append((percept.id, score))
                logger.debug(f"Budget exhausted: rejected {percept.id} (score: {score:.3f})")
        
        # Log selection decision
        decision = {
            "timestamp": datetime.now(),
            "total_candidates": len(candidates),
            "selected_count": len(selected),
            "budget_used": budget_used,
            "budget_available": self.attention_budget,
            "rejected_budget": len(rejected_budget)
        }
        self.attention_history.append(decision)
        
        logger.info(f"Selected {len(selected)}/{len(candidates)} percepts, budget used: {budget_used}/{self.attention_budget}")
        
        return selected

    def _score(self, percept: Percept) -> float:
        """
        Calculates relevance score for a single percept.
        
        Score components:
        - Goal relevance (0.0-1.0): Cosine similarity with current goals
        - Novelty (0.0-1.0): How different from recent percepts
        - Emotional salience (0.0-1.0): Matches emotional themes
        - Recency bonus (0.0-0.2): Slight boost for very recent percepts
        - Affect modulation: Emotional state influences attention
        
        Args:
            percept: The percept to score
            
        Returns:
            Float score (0.0-1.0+)
        """
        goal_rel = self._compute_goal_relevance(percept)
        novelty = self._compute_novelty(percept)
        emotion_sal = self._compute_emotional_salience(percept)
        
        # Recency bonus: newer percepts get slight boost
        time_diff = (datetime.now() - percept.timestamp).total_seconds()
        recency = 0.2 if time_diff < 1.0 else 0.1 if time_diff < 5.0 else 0.0
        
        # Weighted average
        base_score = (
            goal_rel * self.goal_weight +
            novelty * self.novelty_weight +
            emotion_sal * self.emotion_weight +
            recency * self.urgency_weight
        )
        
        # Tool result boost: Tool results get attention priority
        if percept.modality == "tool_result":
            # Base boost for all tool results
            base_score += 0.30
            
            # Additional boost for failed tools (errors need attention)
            if percept.metadata.get("tool_success") is False:
                base_score += 0.20
        
        # Apply affect modulation if affect subsystem is available
        if self.affect:
            total_score = self.affect.influence_attention(base_score, percept)
        else:
            total_score = base_score
        
        logger.debug(f"Scored percept {percept.id}: total={total_score:.3f}, "
                    f"base={base_score:.3f}, goal_rel={goal_rel:.2f}, "
                    f"novelty={novelty:.2f}, emotion={emotion_sal:.2f}, recency={recency:.2f}")
        
        return total_score

    def _compute_goal_relevance(self, percept: Percept) -> float:
        """
        Compute goal relevance score for percept.
        
        Args:
            percept: The percept to evaluate
            
        Returns:
            Score 0.0-1.0 indicating relevance to current goals
        """
        if not self.workspace or not self.workspace.current_goals:
            return 0.5  # Neutral score if no goals
        
        max_relevance = 0.0
        
        for goal in self.workspace.current_goals:
            # Check cache first for performance
            cache_key = (percept.id, goal.id)
            if cache_key in self._relevance_cache:
                max_relevance = max(max_relevance, self._relevance_cache[cache_key])
                continue
            
            # Try embedding-based similarity if available
            if percept.embedding and goal.metadata.get('embedding'):
                similarity = cosine_similarity(percept.embedding, goal.metadata['embedding'])
                max_relevance = max(max_relevance, similarity)
                
                # Cache result with eviction if needed
                if len(self._relevance_cache) >= self._cache_max_size:
                    # Evict oldest entry (FIFO)
                    self._relevance_cache.pop(next(iter(self._relevance_cache)))
                self._relevance_cache[cache_key] = similarity
            else:
                # Fall back to keyword matching
                percept_text = str(percept.raw) if not isinstance(percept.raw, str) else percept.raw
                overlap = keyword_overlap(percept_text, goal.description)
                max_relevance = max(max_relevance, overlap)
        
        return max_relevance
    
    def _compute_goal_relevance_batch(self, percepts: List[Percept]) -> Dict[str, float]:
        """
        Compute goal relevance scores using batched operations for performance.
        
        Uses numpy vectorization when embeddings are available for faster computation.
        
        Args:
            percepts: List of percepts to evaluate
            
        Returns:
            Dict mapping percept IDs to relevance scores (0.0-1.0)
        """
        if not percepts or not self.workspace or not self.workspace.current_goals:
            return {p.id: 0.5 for p in percepts}
        
        scores = {}
        
        # Separate percepts with and without embeddings
        percepts_with_embeddings = []
        percepts_without_embeddings = []
        
        for p in percepts:
            if p.embedding and any(g.metadata.get('embedding') for g in self.workspace.current_goals):
                percepts_with_embeddings.append(p)
            else:
                percepts_without_embeddings.append(p)
        
        # Batch process percepts with embeddings using numpy
        if percepts_with_embeddings:
            goals_with_embeddings = [g for g in self.workspace.current_goals if g.metadata.get('embedding')]
            
            # Extract embeddings as numpy arrays
            percept_embeddings = np.array([p.embedding for p in percepts_with_embeddings])  # Shape: (N, D)
            goal_embeddings = np.array([g.metadata['embedding'] for g in goals_with_embeddings])  # Shape: (M, D)
            
            # Batch cosine similarity: (N, D) @ (D, M) = (N, M)
            # Normalize embeddings
            percept_norms = np.linalg.norm(percept_embeddings, axis=1, keepdims=True)
            goal_norms = np.linalg.norm(goal_embeddings, axis=1, keepdims=True)
            
            # Avoid division by zero
            percept_norms = np.where(percept_norms == 0, 1, percept_norms)
            goal_norms = np.where(goal_norms == 0, 1, goal_norms)
            
            percept_embeddings_norm = percept_embeddings / percept_norms
            goal_embeddings_norm = goal_embeddings / goal_norms
            
            # Compute similarities
            similarities = percept_embeddings_norm @ goal_embeddings_norm.T
            
            # Clamp to [0, 1] range
            similarities = np.maximum(0.0, similarities)
            
            # Max similarity for each percept across all goals
            max_similarities = np.max(similarities, axis=1)
            
            for p, score in zip(percepts_with_embeddings, max_similarities):
                scores[p.id] = float(score)
        
        # Process percepts without embeddings using keyword matching
        for p in percepts_without_embeddings:
            scores[p.id] = self._compute_goal_relevance(p)
        
        return scores

    def _compute_novelty(self, percept: Percept) -> float:
        """
        Compute novelty score for percept.
        
        High novelty if dissimilar to recent percepts.
        
        Args:
            percept: The percept to evaluate
            
        Returns:
            Score 0.0-1.0 (1.0 = completely novel)
        """
        if not percept.embedding or not self.recent_percepts:
            return 1.0  # Completely novel if no embedding or no history
        
        # Compute similarity to all recent percepts
        similarities = []
        for recent_embedding in self.recent_percepts:
            sim = cosine_similarity(percept.embedding, list(recent_embedding))
            similarities.append(sim)
        
        # Novelty is inverse of maximum similarity
        if similarities:
            max_similarity = max(similarities)
            novelty = 1.0 - max_similarity
        else:
            novelty = 1.0
        
        return novelty

    def _compute_emotional_salience(self, percept: Percept) -> float:
        """
        Compute emotional salience score for percept.
        
        High salience if matches current emotional state intensity.
        
        Args:
            percept: The percept to evaluate
            
        Returns:
            Score 0.0-1.0
        """
        if not self.workspace:
            return 0.0
        
        # Check for emotion keywords in percept metadata or raw content
        emotion_keywords = {
            'positive': ['happy', 'joy', 'excited', 'pleased', 'good', 'great', 'love'],
            'negative': ['sad', 'angry', 'fear', 'anxious', 'bad', 'terrible', 'hate'],
            'neutral': ['calm', 'peaceful', 'neutral', 'okay']
        }
        
        percept_text = str(percept.raw).lower() if percept.raw else ""
        
        # Check metadata for emotion tags
        if 'emotion' in percept.metadata:
            emotion_tag = percept.metadata['emotion']
            # Match with workspace emotional state
            valence = self.workspace.emotional_state.get('valence', 0.0)
            if emotion_tag in emotion_keywords['positive'] and valence > 0.3:
                return 0.8
            elif emotion_tag in emotion_keywords['negative'] and valence < -0.3:
                return 0.8
            else:
                return 0.5
        
        # Check for emotion keywords in text
        for emotion_type, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in percept_text:
                    # Boost salience if matches current emotional state
                    valence = self.workspace.emotional_state.get('valence', 0.0)
                    arousal = self.workspace.emotional_state.get('arousal', 0.0)
                    
                    if emotion_type == 'positive' and valence > 0.3:
                        return min(1.0, 0.7 + abs(arousal) * 0.3)
                    elif emotion_type == 'negative' and valence < -0.3:
                        return min(1.0, 0.7 + abs(arousal) * 0.3)
                    else:
                        return 0.5
        
        # Default: low emotional salience
        return 0.2

    def reset_budget(self) -> None:
        """
        Resets attention budget to initial value.
        
        Called at the start of each cognitive cycle.
        """
        self.attention_budget = self.initial_budget
        logger.debug(f"Attention budget reset to {self.attention_budget}")

    def get_attention_report(self) -> Dict[str, Any]:
        """
        Returns summary of recent attention decisions.
        
        Returns:
            Dict with attention statistics including:
            - total_candidates: Total percepts evaluated
            - selected_count: Number of percepts selected
            - rejection_reasons: Breakdown of why percepts were rejected
            - budget_usage: Average budget utilization
        """
        if not self.attention_history:
            return {
                "total_decisions": 0,
                "total_candidates": 0,
                "selected_count": 0,
                "avg_budget_usage": 0.0,
                "rejection_reasons": {
                    "low_score": 0,
                    "budget_exhausted": 0
                }
            }
        
        total_candidates = sum(d['total_candidates'] for d in self.attention_history)
        selected_count = sum(d['selected_count'] for d in self.attention_history)
        total_rejected_low = sum(d['rejected_low_score'] for d in self.attention_history)
        total_rejected_budget = sum(d['rejected_budget'] for d in self.attention_history)
        avg_budget = sum(d['budget_used'] for d in self.attention_history) / len(self.attention_history)
        
        return {
            "total_decisions": len(self.attention_history),
            "total_candidates": total_candidates,
            "selected_count": selected_count,
            "avg_budget_usage": avg_budget,
            "avg_budget_available": self.initial_budget,
            "rejection_reasons": {
                "low_score": total_rejected_low,
                "budget_exhausted": total_rejected_budget
            }
        }
