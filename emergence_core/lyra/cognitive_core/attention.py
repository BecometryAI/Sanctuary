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
from dataclasses import dataclass, field
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


@dataclass
class CompetitionMetrics:
    """
    Metrics tracking competitive dynamics during attention selection.
    
    Attributes:
        inhibition_events: Number of inhibition interactions processed
        suppressed_percepts: IDs of percepts suppressed below threshold
        activation_spread_before: Standard deviation of activations before competition
        activation_spread_after: Standard deviation of activations after competition
        winner_ids: IDs of percepts that exceeded the ignition threshold
        coalition_formations: Dict mapping percept IDs to their coalition partners
    """
    inhibition_events: int = 0
    suppressed_percepts: List[str] = field(default_factory=list)
    activation_spread_before: float = 0.0
    activation_spread_after: float = 0.0
    winner_ids: List[str] = field(default_factory=list)
    coalition_formations: Dict[str, List[str]] = field(default_factory=dict)


class CompetitiveAttention:
    """
    Implements competitive attention dynamics with lateral inhibition.
    
    Based on Global Workspace Theory, this implements genuine competition
    between percepts where high-activation percepts inhibit lower-activation
    competitors. Only percepts that exceed an ignition threshold after
    competition enter the workspace.
    
    Key Features:
    - Lateral inhibition: High-activation percepts suppress competitors
    - Ignition threshold: Percepts must exceed threshold to enter workspace
    - Coalition formation: Related percepts support each other
    - Competition metrics: Track inhibition, suppression, and dynamics
    
    Attributes:
        inhibition_strength: Strength of lateral inhibition (0.0-1.0)
        ignition_threshold: Minimum activation required for workspace entry
        iterations: Number of competition iterations to run
        coalition_boost: Boost factor for related percepts in coalitions
    """
    
    def __init__(
        self,
        inhibition_strength: float = 0.3,
        ignition_threshold: float = 0.5,
        iterations: int = 10,
        coalition_boost: float = 0.2,
    ) -> None:
        """
        Initialize competitive attention mechanism.
        
        Args:
            inhibition_strength: How strongly percepts inhibit each other (0.0-1.0)
            ignition_threshold: Activation level required for workspace entry (0.0-1.0)
            iterations: Number of competition iterations to run (1-100)
            coalition_boost: Bonus activation for percepts in same coalition (0.0-1.0)
            
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate and clamp parameters
        if not (0.0 <= inhibition_strength <= 1.0):
            raise ValueError(f"inhibition_strength must be in [0, 1], got {inhibition_strength}")
        if not (0.0 <= ignition_threshold <= 1.0):
            raise ValueError(f"ignition_threshold must be in [0, 1], got {ignition_threshold}")
        if not (1 <= iterations <= 100):
            raise ValueError(f"iterations must be in [1, 100], got {iterations}")
        if not (0.0 <= coalition_boost <= 1.0):
            raise ValueError(f"coalition_boost must be in [0, 1], got {coalition_boost}")
            
        self.inhibition_strength = inhibition_strength
        self.ignition_threshold = ignition_threshold
        self.iterations = iterations
        self.coalition_boost = coalition_boost
        
        # Track activations during competition
        self.activations: Dict[str, float] = {}
        
        logger.debug(
            f"CompetitiveAttention initialized: "
            f"inhibition={self.inhibition_strength:.2f}, "
            f"threshold={self.ignition_threshold:.2f}, "
            f"iterations={self.iterations}"
        )
    
    def _compute_relatedness(self, p1: Percept, p2: Percept) -> float:
        """
        Compute how related two percepts are for coalition formation.
        
        Percepts are related if they:
        - Have similar embeddings (semantic similarity)
        - Share the same modality
        - Have overlapping keywords
        
        Args:
            p1: First percept
            p2: Second percept
            
        Returns:
            Relatedness score (0.0-1.0, where 1.0 is highly related)
        """
        relatedness = 0.0
        
        # Embedding similarity (strongest signal)
        if p1.embedding and p2.embedding:
            similarity = cosine_similarity(p1.embedding, p2.embedding)
            relatedness = max(relatedness, similarity)
        
        # Modality match (weak signal, but counts)
        if p1.modality == p2.modality:
            relatedness = max(relatedness, 0.2)
        
        # Keyword overlap (fallback for percepts without embeddings)
        if not p1.embedding or not p2.embedding:
            p1_text = str(p1.raw) if not isinstance(p1.raw, str) else p1.raw
            p2_text = str(p2.raw) if not isinstance(p2.raw, str) else p2.raw
            overlap = keyword_overlap(p1_text, p2_text)
            relatedness = max(relatedness, overlap)
        
        return relatedness
    
    def _form_coalitions(self, percepts: List[Percept], relatedness_threshold: float = 0.6) -> Dict[str, List[str]]:
        """
        Form coalitions of related percepts that support each other.
        
        Args:
            percepts: List of percepts to form coalitions from
            relatedness_threshold: Minimum relatedness to form coalition
            
        Returns:
            Dict mapping percept IDs to lists of coalition partner IDs
        """
        coalitions: Dict[str, List[str]] = {p.id: [] for p in percepts}
        
        # Find related percepts
        for i, p1 in enumerate(percepts):
            for p2 in percepts[i + 1:]:
                relatedness = self._compute_relatedness(p1, p2)
                
                if relatedness >= relatedness_threshold:
                    # Mutual coalition membership
                    coalitions[p1.id].append(p2.id)
                    coalitions[p2.id].append(p1.id)
                    
                    logger.debug(
                        f"Coalition formed: {p1.id[:8]} <-> {p2.id[:8]} "
                        f"(relatedness={relatedness:.2f})"
                    )
        
        return coalitions
    
    def compete(
        self,
        percepts: List[Percept],
        base_scores: Dict[str, float],
    ) -> Tuple[List[Percept], CompetitionMetrics]:
        """
        Run competitive dynamics where percepts inhibit each other.
        
        High-activation percepts inhibit lower-activation competitors.
        Related percepts form coalitions and support each other.
        
        Args:
            percepts: List of percepts competing for attention
            base_scores: Initial attention scores for each percept
            
        Returns:
            Tuple of (sorted percepts by final activation, competition metrics)
        """
        if not percepts:
            return [], CompetitionMetrics()
        
        # Initialize activations efficiently
        self.activations = {
            p.id: max(0.0, min(1.0, base_scores.get(p.id, 0.5)))
            for p in percepts
        }
        
        # Track initial activation spread
        initial_activations = list(self.activations.values())
        activation_spread_before = float(np.std(initial_activations)) if len(initial_activations) > 1 else 0.0
        
        # Form coalitions once (not per iteration) - optimization
        coalitions = self._form_coalitions(percepts)
        
        # Pre-compute coalition sets for faster lookup
        coalition_sets = {p.id: set(coalitions[p.id]) for p in percepts}
        
        # Run competition iterations
        for iteration in range(self.iterations):
            new_activations = {}
            
            for p in percepts:
                # Self-excitation (activation persists with decay factor)
                excitation = self.activations[p.id] * 1.1
                
                # Coalition support (related percepts boost each other)
                coalition_support = 0.0
                if coalition_sets[p.id]:
                    # Average activation of coalition partners - optimized
                    partner_sum = sum(self.activations[pid] for pid in coalition_sets[p.id])
                    coalition_support = (partner_sum / len(coalition_sets[p.id])) * self.coalition_boost
                
                # Lateral inhibition from competing percepts (optimized)
                inhibition = sum(
                    self.activations[other.id] * self.inhibition_strength
                    for other in percepts
                    if other.id != p.id and other.id not in coalition_sets[p.id]
                )
                
                # Update activation with bounds
                new_activation = excitation + coalition_support - inhibition
                new_activations[p.id] = max(0.0, min(1.0, new_activation))
            
            self.activations = new_activations
        
        # Track final activation spread
        final_activations = list(self.activations.values())
        activation_spread_after = float(np.std(final_activations)) if len(final_activations) > 1 else 0.0
        
        # Identify winners (exceeded threshold) and suppressed percepts
        winner_ids = [p.id for p in percepts if self.activations[p.id] >= self.ignition_threshold]
        suppressed_percepts = [p.id for p in percepts if self.activations[p.id] < self.ignition_threshold]
        
        # Sort by final activation (descending)
        sorted_percepts = sorted(
            percepts,
            key=lambda p: self.activations[p.id],
            reverse=True
        )
        
        # Calculate actual inhibition events accounting for coalitions
        # Each percept inhibits all non-coalition members, across all iterations
        inhibition_events = 0
        for p_id, coalition_partners in coalition_sets.items():
            # Number of percepts this one inhibits (all except self and coalition)
            num_inhibited = len(percepts) - 1 - len(coalition_partners)
            inhibition_events += num_inhibited * self.iterations
        
        # Create metrics
        metrics = CompetitionMetrics(
            inhibition_events=inhibition_events,
            suppressed_percepts=suppressed_percepts,
            activation_spread_before=activation_spread_before,
            activation_spread_after=activation_spread_after,
            winner_ids=winner_ids,
            coalition_formations=coalitions,
        )
        
        logger.debug(
            f"Competition complete: {len(winner_ids)}/{len(percepts)} winners, "
            f"{len(suppressed_percepts)} suppressed"
        )
        
        return sorted_percepts, metrics
    
    def select_for_workspace(
        self,
        percepts: List[Percept],
        base_scores: Dict[str, float],
    ) -> Tuple[List[Percept], CompetitionMetrics]:
        """
        Select percepts for workspace using competitive dynamics.
        
        Only percepts that survive competition AND exceed the ignition
        threshold enter the workspace. This is fundamentally different
        from top-N selection.
        
        Args:
            percepts: Candidate percepts
            base_scores: Initial attention scores
            
        Returns:
            Tuple of (selected percepts sorted by activation, competition metrics)
        """
        sorted_percepts, metrics = self.compete(percepts, base_scores)
        
        # Only percepts that exceeded threshold (convert to set for O(1) lookup)
        winner_ids_set = set(metrics.winner_ids)
        selected = [
            p for p in sorted_percepts
            if p.id in winner_ids_set
        ]
        
        logger.debug(
            f"Selected {len(selected)}/{len(percepts)} percepts "
            f"(threshold={self.ignition_threshold:.2f})"
        )
        
        return selected, metrics


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
    
    **Competitive Dynamics (New):**
    The controller now supports genuine competitive attention dynamics based on
    Global Workspace Theory. When enabled with `use_competition=True`, percepts
    actively compete for workspace access through lateral inhibition, ignition
    thresholds, and coalition formation. This is fundamentally different from
    simple top-N selection.
    
    **Default Behavior:**
    By default, `use_competition=True` to enable genuine competitive attention
    dynamics compliant with Global Workspace Theory. For legacy behavior using
    simple top-N selection, set `use_competition=False` when initializing the
    controller.

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
    6. Competitive Dynamics (Optional): Lateral inhibition and coalition formation

    The controller uses a weighted scoring system that can be dynamically adjusted
    based on current context, emotional state, and cognitive load. Different attention
    modes shift these weights to support different cognitive strategies.

    Attributes:
        mode: Current attention allocation strategy
        goal_weight: Weight for goal-relevance in scoring (0.0-1.0)
        novelty_weight: Weight for novelty in scoring (0.0-1.0)
        emotion_weight: Weight for emotional salience in scoring (0.0-1.0)
        urgency_weight: Weight for urgency in scoring (0.0-1.0)
        use_competition: Whether to use competitive dynamics (default: False)
        competitive_attention: CompetitiveAttention instance if enabled
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
        use_competition: bool = True,
        inhibition_strength: float = 0.3,
        ignition_threshold: float = 0.5,
        competition_iterations: int = 10,
        coalition_boost: float = 0.2,
    ) -> None:
        """
        Initialize the attention controller.

        Args:
            attention_budget: Total attention units available per cycle (positive integer)
            workspace: Reference to the workspace for context
            affect: Reference to the affect subsystem for emotional modulation
            initial_mode: Starting attention mode (focused, diffuse, vigilant, relaxed)
            goal_weight: Importance of goal-relevance in attention (0.0-1.0)
            novelty_weight: Importance of novelty in attention (0.0-1.0)
            emotion_weight: Importance of emotional salience in attention (0.0-1.0)
            urgency_weight: Importance of urgency in attention (0.0-1.0)
            use_competition: Enable GWT-compliant competitive dynamics (default: True)
            inhibition_strength: Strength of lateral inhibition (0.0-1.0)
            ignition_threshold: Activation threshold for workspace entry (0.0-1.0)
            competition_iterations: Number of competition iterations (1-100)
            coalition_boost: Boost factor for coalition members (0.0-1.0)

        Note: Weights should sum to approximately 1.0 for balanced scoring.
        """
        # Validate inputs
        if attention_budget <= 0:
            raise ValueError(f"attention_budget must be positive, got {attention_budget}")
        
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
        
        # Competitive attention settings (GWT-compliant by default)
        self.use_competition = use_competition
        self.competitive_attention = CompetitiveAttention(
            inhibition_strength=inhibition_strength,
            ignition_threshold=ignition_threshold,
            iterations=competition_iterations,
            coalition_boost=coalition_boost,
        ) if use_competition else None
        
        # History tracking
        self.recent_percepts: deque = deque(maxlen=50)
        self.attention_history: List[Dict[str, Any]] = []
        self.competition_metrics_history: List[CompetitionMetrics] = []
        
        # Performance optimization: Relevance cache
        self._relevance_cache: Dict[Tuple[str, str], float] = {}
        self._cache_max_size = 1000
        
        logger.info(
            f"AttentionController initialized: budget={attention_budget}, "
            f"mode={initial_mode.value}, GWT_competitive={use_competition}"
        )

    def select_for_broadcast(self, candidates: List[Percept]) -> List[Percept]:
        """
        Scores candidate percepts and selects top-scoring ones within budget.
        
        Uses GWT-compliant competitive dynamics by default (use_competition=True),
        or legacy top-N selection if disabled.
        
        Args:
            candidates: List of candidate percepts to evaluate
            
        Returns:
            Sorted list of selected percepts (highest scoring first)
            
        Raises:
            TypeError: If candidates is not a list or contains non-Percept items
        """
        # Validate inputs
        if not isinstance(candidates, list):
            raise TypeError(f"candidates must be a list, got {type(candidates)}")
        
        if not candidates:
            logger.debug("No candidates to select from")
            return []
        
        # Validate all items are Percepts
        if not all(isinstance(p, Percept) for p in candidates):
            raise TypeError("All candidates must be Percept instances")
        
        # Compute goal relevance scores (batched for efficiency)
        if len(candidates) > 10:
            goal_relevance_scores = self._compute_goal_relevance_batch(candidates)
        else:
            goal_relevance_scores = {p.id: self._compute_goal_relevance(p) for p in candidates}
        
        # Score all candidates
        base_scores = {}
        for percept in candidates:
            # Compute base score from multiple factors
            goal_rel = goal_relevance_scores.get(percept.id, 0.5)
            novelty = self._compute_novelty(percept)
            emotion_sal = self._compute_emotional_salience(percept)
            
            # Recency bonus for recent percepts
            time_diff = (datetime.now() - percept.timestamp).total_seconds()
            recency = 0.2 if time_diff < 1.0 else 0.1 if time_diff < 5.0 else 0.0
            
            # Weighted combination
            base_score = (
                goal_rel * self.goal_weight +
                novelty * self.novelty_weight +
                emotion_sal * self.emotion_weight +
                recency * self.urgency_weight
            )
            
            # Tool result boost
            if percept.modality == "tool_result":
                base_score += 0.30
                if percept.metadata.get("tool_success") is False:
                    base_score += 0.20  # Failed tools need attention
            
            # Apply affect modulation if available
            if self.affect:
                total_score = self.affect.influence_attention(base_score, percept)
            else:
                total_score = base_score
            
            base_scores[percept.id] = total_score
        
        # Select using competitive dynamics (GWT) or legacy mode
        if self.use_competition and self.competitive_attention:
            # Apply emotional modulation to competition parameters BEFORE selection
            if self.affect and hasattr(self.affect, 'get_processing_params'):
                processing_params = self.affect.get_processing_params()
                # Modulate competition iterations and ignition threshold based on arousal
                self.competitive_attention.iterations = processing_params.attention_iterations
                self.competitive_attention.ignition_threshold = processing_params.ignition_threshold
                logger.debug(
                    f"Emotional modulation: iterations={processing_params.attention_iterations}, "
                    f"threshold={processing_params.ignition_threshold:.2f} "
                    f"(arousal={processing_params.arousal_level:.2f})"
                )
            
            selected, metrics = self._select_with_competition(candidates, base_scores)
            self.competition_metrics_history.append(metrics)
        else:
            selected = self._select_legacy(candidates, base_scores)
        
        # Record decision for tracking
        decision = {
            "timestamp": datetime.now(),
            "total_candidates": len(candidates),
            "selected_count": len(selected),
            "budget_used": sum(p.complexity for p in selected),
            "budget_available": self.attention_budget,
            "rejected_count": len(candidates) - len(selected),
            "gwt_competitive": self.use_competition,
        }
        self.attention_history.append(decision)
        
        logger.debug(
            f"Selected {len(selected)}/{len(candidates)} percepts "
            f"(budget: {decision['budget_used']}/{self.attention_budget}, "
            f"GWT: {self.use_competition})"
        )
        
        return selected
    
    def _select_with_competition(
        self,
        candidates: List[Percept],
        base_scores: Dict[str, float],
    ) -> Tuple[List[Percept], CompetitionMetrics]:
        """
        Select percepts using competitive dynamics.
        
        Args:
            candidates: Candidate percepts
            base_scores: Initial attention scores
            
        Returns:
            Tuple of (selected percepts, competition metrics)
        """
        # Run competition to get percepts sorted by final activation
        competed_percepts, metrics = self.competitive_attention.select_for_workspace(
            candidates, base_scores
        )
        
        # Apply budget constraint (select as many as fit)
        selected = []
        budget_used = 0
        
        for percept in competed_percepts:
            if budget_used + percept.complexity <= self.attention_budget:
                selected.append(percept)
                budget_used += percept.complexity
                
                # Add to recent percepts for novelty detection
                if percept.embedding:
                    self.recent_percepts.append(percept.embedding)
                
                activation = self.competitive_attention.activations.get(percept.id, 0.0)
                logger.debug(
                    f"Selected percept: {percept.id} "
                    f"(activation: {activation:.3f}, complexity: {percept.complexity})"
                )
            else:
                logger.debug(
                    f"Budget exhausted: rejected {percept.id} "
                    f"(complexity: {percept.complexity})"
                )
        
        return selected, metrics
    
    def _select_legacy(
        self,
        candidates: List[Percept],
        base_scores: Dict[str, float],
    ) -> List[Percept]:
        """
        Legacy selection using simple top-N sorting (backward compatible).
        
        Args:
            candidates: Candidate percepts
            base_scores: Attention scores
            
        Returns:
            Selected percepts within budget
        """
        # Sort by score (highest first)
        scored_percepts = [(p, base_scores[p.id]) for p in candidates]
        scored_percepts.sort(key=lambda x: x[1], reverse=True)
        
        # Select percepts that fit within budget
        selected = []
        budget_used = 0
        
        for percept, score in scored_percepts:
            if budget_used + percept.complexity <= self.attention_budget:
                selected.append(percept)
                budget_used += percept.complexity
                
                # Add to recent percepts for novelty detection
                if percept.embedding:
                    self.recent_percepts.append(percept.embedding)
                
                logger.debug(
                    f"Selected percept: {percept.id} "
                    f"(score: {score:.3f}, complexity: {percept.complexity})"
                )
            else:
                logger.debug(
                    f"Budget exhausted: rejected {percept.id} "
                    f"(score: {score:.3f})"
                )
        
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
            - competition_stats: Statistics from competitive dynamics (if enabled)
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
                },
                "competition_enabled": self.use_competition,
            }
        
        total_candidates = sum(d['total_candidates'] for d in self.attention_history)
        selected_count = sum(d['selected_count'] for d in self.attention_history)
        total_rejected_budget = sum(d['rejected_budget'] for d in self.attention_history)
        avg_budget = sum(d['budget_used'] for d in self.attention_history) / len(self.attention_history)
        
        report = {
            "total_decisions": len(self.attention_history),
            "total_candidates": total_candidates,
            "selected_count": selected_count,
            "avg_budget_usage": avg_budget,
            "avg_budget_available": self.initial_budget,
            "rejection_reasons": {
                "low_score": 0,  # Legacy metric, kept for compatibility
                "budget_exhausted": total_rejected_budget
            },
            "competition_enabled": self.use_competition,
        }
        
        # Add competition statistics if available
        if self.use_competition and self.competition_metrics_history:
            report["competition_stats"] = self._summarize_competition_metrics()
        
        return report
    
    def _summarize_competition_metrics(self) -> Dict[str, Any]:
        """
        Summarize competition metrics from history.
        
        Returns:
            Dict with competition statistics
        """
        if not self.competition_metrics_history:
            return {}
        
        total_inhibition_events = sum(m.inhibition_events for m in self.competition_metrics_history)
        total_suppressed = sum(len(m.suppressed_percepts) for m in self.competition_metrics_history)
        total_winners = sum(len(m.winner_ids) for m in self.competition_metrics_history)
        
        avg_spread_before = np.mean([m.activation_spread_before for m in self.competition_metrics_history])
        avg_spread_after = np.mean([m.activation_spread_after for m in self.competition_metrics_history])
        
        # Count coalition formations
        total_coalitions = 0
        for m in self.competition_metrics_history:
            for partners in m.coalition_formations.values():
                if partners:
                    total_coalitions += len(partners)
        
        return {
            "total_inhibition_events": total_inhibition_events,
            "total_suppressed_percepts": total_suppressed,
            "total_winners": total_winners,
            "avg_activation_spread_before": float(avg_spread_before),
            "avg_activation_spread_after": float(avg_spread_after),
            "total_coalition_links": total_coalitions // 2,  # Divide by 2 since links are bidirectional
        }
    
    def get_competition_metrics(self) -> List[CompetitionMetrics]:
        """
        Get detailed competition metrics from recent attention cycles.
        
        Returns:
            List of CompetitionMetrics from recent cycles
        """
        return self.competition_metrics_history
