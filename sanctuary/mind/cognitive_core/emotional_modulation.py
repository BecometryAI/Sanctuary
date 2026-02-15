"""
Emotional Modulation: Making emotions functionally efficacious.

This module implements functional emotional modulation where PAD (Pleasure-Arousal-Dominance)
values directly modulate processing parameters BEFORE LLM invocation. This ensures emotions
are causal forces that shape computation, not merely descriptive labels.

From a functionalist perspective: if emotions don't cause measurable changes to processing
before the LLM is invoked, they're not functionally real emotions.

Key Principles:
1. Arousal modulates processing speed and thoroughness (fight/flight vs deliberation)
2. Valence creates approach/avoidance biases in action selection
3. Dominance modulates decision confidence thresholds (assertiveness)
4. All modulation happens BEFORE cognitive processing, not as context to LLM
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# Constants for modulation ranges
class ModulationConstants:
    """Constants defining modulation parameter ranges."""
    # Arousal modulation ranges
    ATTENTION_ITERATIONS_MIN = 5
    ATTENTION_ITERATIONS_MAX = 10
    IGNITION_THRESHOLD_MIN = 0.4
    IGNITION_THRESHOLD_MAX = 0.6
    MEMORY_RETRIEVAL_MIN = 2
    MEMORY_RETRIEVAL_MAX = 5
    PROCESSING_TIMEOUT_MIN = 1.0
    PROCESSING_TIMEOUT_MAX = 2.0
    
    # Dominance modulation range
    DECISION_THRESHOLD_MIN = 0.5
    DECISION_THRESHOLD_MAX = 0.7
    
    # Valence modulation
    VALENCE_BIAS_STRENGTH = 0.3  # Max adjustment
    VALENCE_THRESHOLD = 0.2  # Minimum valence to apply bias
    
    # Metrics
    MAX_CORRELATION_HISTORY = 100


@dataclass
class ProcessingParams:
    """
    Processing parameters modulated by emotional state.
    
    These parameters directly affect cognitive processing before any LLM invocation,
    making emotions functionally efficacious rather than merely descriptive.
    
    Attributes:
        attention_iterations: Number of competitive attention cycles (arousal-modulated)
        ignition_threshold: Threshold for global workspace broadcast (arousal-modulated)
        memory_retrieval_limit: Max memories to retrieve (arousal-modulated)
        processing_timeout: Time budget for processing (arousal-modulated)
        decision_threshold: Confidence needed to act (dominance-modulated)
        action_bias_strength: Strength of valence-based action biasing (valence-modulated)
    """
    attention_iterations: int = 7
    ignition_threshold: float = 0.5
    memory_retrieval_limit: int = 3
    processing_timeout: float = 1.5
    decision_threshold: float = 0.7
    action_bias_strength: float = 0.0
    
    # Metadata for tracking
    arousal_level: float = 0.0
    valence_level: float = 0.0
    dominance_level: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/metrics."""
        return {
            'attention_iterations': self.attention_iterations,
            'ignition_threshold': self.ignition_threshold,
            'memory_retrieval_limit': self.memory_retrieval_limit,
            'processing_timeout': self.processing_timeout,
            'decision_threshold': self.decision_threshold,
            'action_bias_strength': self.action_bias_strength,
            'arousal_level': self.arousal_level,
            'valence_level': self.valence_level,
            'dominance_level': self.dominance_level,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ModulationMetrics:
    """
    Metrics for tracking emotional modulation effects.
    
    These metrics verify that emotions are actually modulating processing,
    providing evidence that emotions are functionally real.
    """
    total_modulations: int = 0
    
    # Arousal effects
    high_arousal_fast_processing: int = 0
    low_arousal_slow_processing: int = 0
    arousal_attention_correlations: List[tuple] = field(default_factory=list)
    
    # Valence effects
    positive_valence_approach_bias: int = 0
    negative_valence_avoidance_bias: int = 0
    valence_action_correlations: List[tuple] = field(default_factory=list)
    
    # Dominance effects
    high_dominance_assertive: int = 0
    low_dominance_cautious: int = 0
    dominance_threshold_correlations: List[tuple] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'total_modulations': self.total_modulations,
            'arousal_effects': {
                'high_arousal_fast': self.high_arousal_fast_processing,
                'low_arousal_slow': self.low_arousal_slow_processing,
                'correlations_count': len(self.arousal_attention_correlations)
            },
            'valence_effects': {
                'positive_approach': self.positive_valence_approach_bias,
                'negative_avoidance': self.negative_valence_avoidance_bias,
                'correlations_count': len(self.valence_action_correlations)
            },
            'dominance_effects': {
                'high_assertive': self.high_dominance_assertive,
                'low_cautious': self.low_dominance_cautious,
                'correlations_count': len(self.dominance_threshold_correlations)
            }
        }


class EmotionalModulation:
    """
    Implements functional emotional modulation of cognitive processing.
    
    Makes emotions causally efficacious by directly modulating processing parameters
    BEFORE any LLM invocation. This ensures emotions are real forces that shape
    computation, not merely descriptive context passed to the LLM.
    
    Arousal Effects (Processing Speed/Thoroughness):
    - High arousal (0.7-1.0): Faster, less thorough (fight/flight)
      * Fewer attention iterations (snap decisions)
      * Lower ignition threshold (react to more stimuli)
      * Shorter memory retrieval (less deliberation)
      * Faster timeout (quick response)
    
    - Low arousal (0.0-0.3): Slower, more deliberate
      * More attention iterations (careful analysis)
      * Higher ignition threshold (selective)
      * More memory retrieval (thorough consideration)
      * Longer timeout (take time to think)
    
    Valence Effects (Approach/Avoidance):
    - Positive valence (0.3-1.0): Approach bias
      * Boost priority of engage, explore, create, connect actions
      * Reduce priority of withdraw, defend, avoid, reject actions
    
    - Negative valence (-1.0 to -0.3): Avoidance bias
      * Reduce priority of approach actions
      * Boost priority of defensive/avoidance actions
    
    Dominance Effects (Confidence/Assertiveness):
    - High dominance (0.7-1.0): Lower confidence threshold
      * More assertive (act with less certainty)
      * Lower decision threshold
    
    - Low dominance (0.0-0.3): Higher confidence threshold
      * More cautious (need more certainty to act)
      * Higher decision threshold
    
    Attributes:
        enabled: Whether modulation is active (for ablation testing)
        metrics: Tracking metrics for verifying functional effects
        baseline_params: Default processing parameters
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize emotional modulation system.
        
        Args:
            enabled: Whether modulation is active (False for ablation testing)
        """
        self.enabled = enabled
        self.metrics = ModulationMetrics()
        
        # Baseline processing parameters (neutral emotional state)
        self.baseline_params = ProcessingParams(
            attention_iterations=7,
            ignition_threshold=0.5,
            memory_retrieval_limit=3,
            processing_timeout=1.5,
            decision_threshold=0.7,
            action_bias_strength=0.0
        )
        
        logger.info(f"EmotionalModulation initialized (enabled={enabled})")
    
    def modulate_processing(
        self,
        arousal: float,
        valence: float,
        dominance: float
    ) -> ProcessingParams:
        """
        Modulate processing parameters based on emotional state.
        
        This is the core method that makes emotions functionally efficacious.
        It directly affects cognitive parameters BEFORE any LLM processing.
        
        Args:
            arousal: Emotional arousal level (-1.0 to 1.0, typically 0.0 to 1.0)
            valence: Emotional valence (-1.0 to 1.0)
            dominance: Sense of control/dominance (0.0 to 1.0)
        
        Returns:
            ProcessingParams with emotionally-modulated values
            
        Raises:
            ValueError: If parameters are outside valid ranges
        """
        # Validate inputs
        if not -1.0 <= arousal <= 1.0:
            raise ValueError(f"Arousal must be in [-1, 1], got {arousal}")
        if not -1.0 <= valence <= 1.0:
            raise ValueError(f"Valence must be in [-1, 1], got {valence}")
        if not 0.0 <= dominance <= 1.0:
            raise ValueError(f"Dominance must be in [0, 1], got {dominance}")
        
        if not self.enabled:
            # Return baseline parameters (for ablation testing)
            return self.baseline_params
        
        # Ensure arousal is in [0, 1] range (some systems use [-1, 1])
        arousal_normalized = max(0.0, min(1.0, arousal))
        
        # Apply arousal modulation to processing speed/thoroughness
        params = self._modulate_arousal(arousal_normalized)
        
        # Apply dominance modulation to decision threshold
        params = self._modulate_dominance(dominance, params)
        
        # Store emotional levels for metrics
        params.arousal_level = arousal_normalized
        params.valence_level = valence
        params.dominance_level = dominance
        params.timestamp = datetime.now()
        
        # Update metrics
        self._update_metrics(arousal_normalized, valence, dominance, params)
        
        logger.debug(
            f"Emotional modulation: A={arousal_normalized:.2f} V={valence:.2f} D={dominance:.2f} "
            f"→ iters={params.attention_iterations}, thresh={params.ignition_threshold:.2f}, "
            f"decision={params.decision_threshold:.2f}"
        )
        
        return params
    
    def _modulate_arousal(self, arousal: float) -> ProcessingParams:
        """
        Modulate processing parameters based on arousal.
        
        High arousal = faster, less thorough (fight/flight)
        Low arousal = slower, more deliberate
        
        Args:
            arousal: Arousal level (0.0-1.0)
        
        Returns:
            ProcessingParams with arousal modulation
        """
        # Attention iterations: inverse relationship with arousal
        # High arousal → fewer iterations (snap decisions)
        # Low arousal → more iterations (careful analysis)
        iterations_range = ModulationConstants.ATTENTION_ITERATIONS_MAX - ModulationConstants.ATTENTION_ITERATIONS_MIN
        attention_iterations = int(ModulationConstants.ATTENTION_ITERATIONS_MAX - (arousal * iterations_range))
        attention_iterations = max(ModulationConstants.ATTENTION_ITERATIONS_MIN, 
                                  min(ModulationConstants.ATTENTION_ITERATIONS_MAX, attention_iterations))
        
        # Ignition threshold: inverse relationship with arousal
        # High arousal → lower threshold (react to more stimuli)
        # Low arousal → higher threshold (more selective)
        threshold_range = ModulationConstants.IGNITION_THRESHOLD_MAX - ModulationConstants.IGNITION_THRESHOLD_MIN
        ignition_threshold = ModulationConstants.IGNITION_THRESHOLD_MAX - (arousal * threshold_range)
        ignition_threshold = max(ModulationConstants.IGNITION_THRESHOLD_MIN,
                                min(ModulationConstants.IGNITION_THRESHOLD_MAX, ignition_threshold))
        
        # Memory retrieval limit: inverse relationship with arousal
        # High arousal → fewer memories (less deliberation)
        # Low arousal → more memories (thorough consideration)
        memory_range = ModulationConstants.MEMORY_RETRIEVAL_MAX - ModulationConstants.MEMORY_RETRIEVAL_MIN
        memory_retrieval_limit = int(ModulationConstants.MEMORY_RETRIEVAL_MAX - (arousal * memory_range))
        memory_retrieval_limit = max(ModulationConstants.MEMORY_RETRIEVAL_MIN,
                                    min(ModulationConstants.MEMORY_RETRIEVAL_MAX, memory_retrieval_limit))
        
        # Processing timeout: inverse relationship with arousal
        # High arousal → shorter timeout (quick response)
        # Low arousal → longer timeout (take time to think)
        timeout_range = ModulationConstants.PROCESSING_TIMEOUT_MAX - ModulationConstants.PROCESSING_TIMEOUT_MIN
        processing_timeout = ModulationConstants.PROCESSING_TIMEOUT_MAX - (arousal * timeout_range)
        processing_timeout = max(ModulationConstants.PROCESSING_TIMEOUT_MIN,
                                min(ModulationConstants.PROCESSING_TIMEOUT_MAX, processing_timeout))
        
        return ProcessingParams(
            attention_iterations=attention_iterations,
            ignition_threshold=ignition_threshold,
            memory_retrieval_limit=memory_retrieval_limit,
            processing_timeout=processing_timeout,
            decision_threshold=self.baseline_params.decision_threshold,
            action_bias_strength=0.0
        )
    
    def _modulate_dominance(
        self,
        dominance: float,
        params: ProcessingParams
    ) -> ProcessingParams:
        """
        Modulate decision threshold based on dominance.
        
        High dominance = lower confidence threshold (more assertive)
        Low dominance = higher confidence threshold (more cautious)
        
        Args:
            dominance: Dominance level (0.0-1.0)
            params: ProcessingParams to modify
        
        Returns:
            Modified ProcessingParams
        """
        # Decision threshold: inverse relationship with dominance
        # High dominance → lower threshold (act with less certainty)
        # Low dominance → higher threshold (need more certainty)
        threshold_range = ModulationConstants.DECISION_THRESHOLD_MAX - ModulationConstants.DECISION_THRESHOLD_MIN
        decision_threshold = ModulationConstants.DECISION_THRESHOLD_MAX - (dominance * threshold_range)
        decision_threshold = max(ModulationConstants.DECISION_THRESHOLD_MIN,
                                min(ModulationConstants.DECISION_THRESHOLD_MAX, decision_threshold))
        
        params.decision_threshold = decision_threshold
        return params
    
    def bias_action_selection(
        self,
        actions: List[Any],
        valence: float,
        action_type_attr: str = 'type'
    ) -> List[Any]:
        """
        Apply valence-based approach/avoidance bias to action selection.
        
        This modulates action priorities BEFORE any LLM scoring, making
        valence a causal force in action selection.
        
        Args:
            actions: List of action objects or dicts
            valence: Emotional valence (-1.0 to 1.0)
            action_type_attr: Attribute/key name for action type
        
        Returns:
            List of actions with modulated priorities
        """
        if not self.enabled or abs(valence) < ModulationConstants.VALENCE_THRESHOLD:
            # No significant valence, return unchanged
            return actions
        
        # Bias strength scales with valence magnitude
        bias_strength = abs(valence) * ModulationConstants.VALENCE_BIAS_STRENGTH
        
        # Define action categories
        approach_types = {'speak', 'tool_call', 'commit_memory', 'engage', 'explore', 'create', 'connect'}
        avoidance_types = {'wait', 'introspect', 'withdraw', 'defend', 'avoid', 'reject'}
        
        for action in actions:
            action_type_str = self._get_action_type(action, action_type_attr).lower()
            priority = self._get_action_priority(action)
            
            # Determine if this is an approach or avoidance action
            is_approach = any(atype in action_type_str for atype in approach_types)
            is_avoidance = any(atype in action_type_str for atype in avoidance_types)
            
            if not (is_approach or is_avoidance):
                continue  # Skip actions that aren't categorized
            
            # Calculate new priority based on valence and action type
            if is_approach:
                # Approach actions: boosted by positive valence, reduced by negative
                new_priority = priority + (valence * bias_strength)
            else:  # is_avoidance
                # Avoidance actions: reduced by positive valence, boosted by negative
                new_priority = priority - (valence * bias_strength)
            
            # Clamp to valid range and update
            new_priority = max(0.0, min(1.0, new_priority))
            self._set_action_priority(action, new_priority)
        
        return actions
    
    def _get_action_type(self, action: Any, attr_name: str) -> str:
        """Extract action type from action object or dict."""
        if isinstance(action, dict):
            return str(action.get(attr_name, ''))
        return str(getattr(action, attr_name, ''))
    
    def _get_action_priority(self, action: Any) -> float:
        """Extract priority from action object or dict."""
        if isinstance(action, dict):
            return action.get('priority', 0.5)
        return getattr(action, 'priority', 0.5)
    
    def _set_action_priority(self, action: Any, priority: float) -> None:
        """Set priority on action object or dict."""
        if isinstance(action, dict):
            action['priority'] = priority
        else:
            action.priority = priority
    
    def _update_metrics(
        self,
        arousal: float,
        valence: float,
        dominance: float,
        params: ProcessingParams
    ) -> None:
        """
        Update metrics tracking emotional modulation effects.
        
        These metrics verify that emotions are functionally modulating processing.
        """
        self.metrics.total_modulations += 1
        
        # Track arousal effects (categorize into high/low)
        if arousal > 0.7:
            self.metrics.high_arousal_fast_processing += 1
        elif arousal < 0.3:
            self.metrics.low_arousal_slow_processing += 1
        
        # Track valence effects (categorize into positive/negative)
        if valence > 0.3:
            self.metrics.positive_valence_approach_bias += 1
        elif valence < -0.3:
            self.metrics.negative_valence_avoidance_bias += 1
        
        # Track dominance effects (categorize into high/low)
        if dominance >= 0.7:
            self.metrics.high_dominance_assertive += 1
        elif dominance < 0.3:
            self.metrics.low_dominance_cautious += 1
        
        # Append correlations (bounded to prevent unbounded growth)
        self._append_bounded_correlation(
            self.metrics.arousal_attention_correlations,
            (arousal, params.attention_iterations, params.ignition_threshold)
        )
        self._append_bounded_correlation(
            self.metrics.valence_action_correlations,
            (valence, params.action_bias_strength)
        )
        self._append_bounded_correlation(
            self.metrics.dominance_threshold_correlations,
            (dominance, params.decision_threshold)
        )
    
    def _append_bounded_correlation(self, correlation_list: List[tuple], item: tuple) -> None:
        """Append item to correlation list, removing oldest if at max size."""
        correlation_list.append(item)
        if len(correlation_list) > ModulationConstants.MAX_CORRELATION_HISTORY:
            correlation_list.pop(0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current modulation metrics.
        
        Returns:
            Dictionary of metrics showing emotional modulation effects
        """
        return self.metrics.to_dict()
    
    def reset_metrics(self) -> None:
        """Reset metrics (useful for testing)."""
        self.metrics = ModulationMetrics()
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable emotional modulation.
        
        Used for ablation testing to verify emotions have measurable effects.
        
        Args:
            enabled: Whether to enable modulation
        """
        self.enabled = enabled
        logger.info(f"Emotional modulation {'enabled' if enabled else 'disabled'}")
