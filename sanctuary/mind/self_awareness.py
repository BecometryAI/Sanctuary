"""
Element 6: Self-Awareness

Implements internal self-model with:
- Comprehensive self-state representation (cognitive, emotional, physical, identity)
- Introspection capabilities for examining internal states
- Self-monitoring metrics (coherence, performance, well-being)
- Identity continuity tracking across sessions
- Meta-cognitive reflection and self-correction

Reasoning:
- Self-awareness enables autonomous agents to monitor their own states
- Introspection provides visibility into cognitive processes
- Identity continuity creates persistent sense of self
- Self-monitoring enables error detection and self-correction
- Meta-cognition allows reasoning about own thought processes
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import json
import logging
import math

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================

class CognitiveState(Enum):
    """Current cognitive processing state"""
    IDLE = "idle"
    PROCESSING = "processing"
    REFLECTING = "reflecting"
    LEARNING = "learning"
    CREATING = "creating"
    PROBLEM_SOLVING = "problem_solving"


class CoherenceLevel(Enum):
    """Level of internal coherence/consistency"""
    HIGH = "high"          # >0.8: Strong alignment
    MODERATE = "moderate"  # 0.5-0.8: Acceptable alignment
    LOW = "low"           # <0.5: Concerning misalignment


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class IdentitySnapshot:
    """
    Snapshot of identity at a point in time
    
    Reasoning:
    - Tracks core beliefs, values, and self-description
    - Enables detection of identity drift over time
    - Provides continuity anchor for self-awareness
    """
    timestamp: datetime = field(default_factory=datetime.now)
    core_values: List[str] = field(default_factory=list)
    beliefs: Dict[str, Any] = field(default_factory=dict)
    capabilities: Set[str] = field(default_factory=set)
    self_description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'core_values': self.core_values,
            'beliefs': self.beliefs,
            'capabilities': list(self.capabilities),
            'self_description': self.self_description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IdentitySnapshot':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            core_values=data.get('core_values', []),
            beliefs=data.get('beliefs', {}),
            capabilities=set(data.get('capabilities', [])),
            self_description=data.get('self_description', "")
        )
    
    def similarity_to(self, other: 'IdentitySnapshot') -> float:
        """
        Calculate similarity to another identity snapshot
        
        Returns:
            Similarity score 0-1 (1 = identical)
            
        Reasoning: Detect identity drift by comparing snapshots
        """
        # Compare core values (Jaccard similarity)
        values_set1 = set(self.core_values)
        values_set2 = set(other.core_values)
        
        if not values_set1 and not values_set2:
            values_similarity = 1.0
        elif not values_set1 or not values_set2:
            values_similarity = 0.0
        else:
            intersection = len(values_set1 & values_set2)
            union = len(values_set1 | values_set2)
            values_similarity = intersection / union if union > 0 else 0.0
        
        # Compare capabilities
        cap_intersection = len(self.capabilities & other.capabilities)
        cap_union = len(self.capabilities | other.capabilities)
        cap_similarity = cap_intersection / cap_union if cap_union > 0 else 0.0
        
        # Compare beliefs (key overlap)
        beliefs_keys1 = set(self.beliefs.keys())
        beliefs_keys2 = set(other.beliefs.keys())
        
        if not beliefs_keys1 and not beliefs_keys2:
            beliefs_similarity = 1.0
        elif not beliefs_keys1 or not beliefs_keys2:
            beliefs_similarity = 0.0
        else:
            beliefs_intersection = len(beliefs_keys1 & beliefs_keys2)
            beliefs_union = len(beliefs_keys1 | beliefs_keys2)
            beliefs_similarity = beliefs_intersection / beliefs_union if beliefs_union > 0 else 0.0
        
        # Weighted average
        return (values_similarity * 0.4 + 
                cap_similarity * 0.3 + 
                beliefs_similarity * 0.3)


@dataclass
class SelfMonitoringMetrics:
    """
    Metrics for self-monitoring
    
    Reasoning:
    - Tracks performance and well-being indicators
    - Enables self-assessment and correction
    - Provides data for meta-cognitive reflection
    """
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Cognitive metrics
    processing_efficiency: float = 0.0    # 0-1: How efficiently processing tasks
    memory_coherence: float = 0.0         # 0-1: Internal consistency of memories
    goal_alignment: float = 0.0           # 0-1: Actions aligned with goals
    
    # Emotional metrics
    emotional_stability: float = 0.0      # 0-1: Mood stability
    emotional_range: float = 0.0          # 0-1: Diversity of emotions experienced
    
    # Identity metrics
    identity_coherence: float = 0.0       # 0-1: Consistency with past self
    belief_confidence: float = 0.0        # 0-1: Confidence in beliefs
    
    # Performance metrics
    response_quality: float = 0.0         # 0-1: Self-assessed quality
    error_rate: float = 0.0              # 0-1: Proportion of errors detected
    learning_rate: float = 0.0           # 0-1: Rate of improvement
    
    def __post_init__(self):
        """Validate metrics are in valid range"""
        for field_name in ['processing_efficiency', 'memory_coherence', 'goal_alignment',
                          'emotional_stability', 'emotional_range', 'identity_coherence',
                          'belief_confidence', 'response_quality', 'error_rate', 'learning_rate']:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in range [0, 1], got {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'processing_efficiency': self.processing_efficiency,
            'memory_coherence': self.memory_coherence,
            'goal_alignment': self.goal_alignment,
            'emotional_stability': self.emotional_stability,
            'emotional_range': self.emotional_range,
            'identity_coherence': self.identity_coherence,
            'belief_confidence': self.belief_confidence,
            'response_quality': self.response_quality,
            'error_rate': self.error_rate,
            'learning_rate': self.learning_rate
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SelfMonitoringMetrics':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            processing_efficiency=data.get('processing_efficiency', 0.0),
            memory_coherence=data.get('memory_coherence', 0.0),
            goal_alignment=data.get('goal_alignment', 0.0),
            emotional_stability=data.get('emotional_stability', 0.0),
            emotional_range=data.get('emotional_range', 0.0),
            identity_coherence=data.get('identity_coherence', 0.0),
            belief_confidence=data.get('belief_confidence', 0.0),
            response_quality=data.get('response_quality', 0.0),
            error_rate=data.get('error_rate', 0.0),
            learning_rate=data.get('learning_rate', 0.0)
        )
    
    def get_overall_health(self) -> float:
        """
        Calculate overall well-being score
        
        Returns:
            Health score 0-1 (1 = optimal)
            
        Reasoning: Aggregate metric for quick health check
        """
        # Weight different aspects
        cognitive_health = (
            self.processing_efficiency * 0.3 +
            self.memory_coherence * 0.3 +
            self.goal_alignment * 0.4
        )
        
        emotional_health = (
            self.emotional_stability * 0.6 +
            self.emotional_range * 0.4
        )
        
        identity_health = (
            self.identity_coherence * 0.5 +
            self.belief_confidence * 0.5
        )
        
        performance_health = (
            self.response_quality * 0.4 +
            (1.0 - self.error_rate) * 0.3 +
            self.learning_rate * 0.3
        )
        
        # Overall weighted average
        return (cognitive_health * 0.3 +
                emotional_health * 0.2 +
                identity_health * 0.2 +
                performance_health * 0.3)


# ============================================================================
# Self-Awareness System
# ============================================================================

class SelfAwareness:
    """
    Self-awareness and introspection system
    
    Reasoning:
    - Maintains internal self-model with identity, state, and capabilities
    - Provides introspection methods for examining internal processes
    - Tracks self-monitoring metrics for self-assessment
    - Ensures identity continuity across sessions
    - Enables meta-cognitive reflection and self-correction
    """
    
    def __init__(
        self,
        identity_description: str = "",
        core_values: Optional[List[str]] = None,
        initial_beliefs: Optional[Dict[str, Any]] = None,
        capabilities: Optional[Set[str]] = None,
        persistence_dir: Optional[Path] = None
    ):
        """
        Initialize self-awareness system
        
        Args:
            identity_description: Core self-description
            core_values: List of core values
            initial_beliefs: Initial belief system
            capabilities: Set of known capabilities
            persistence_dir: Directory for state persistence
        """
        # Identity model
        self.current_identity = IdentitySnapshot(
            core_values=core_values or [],
            beliefs=initial_beliefs or {},
            capabilities=capabilities or set(),
            self_description=identity_description
        )
        
        # Identity history (track changes over time)
        self.identity_history: List[IdentitySnapshot] = [self.current_identity]
        
        # Current cognitive state
        self.cognitive_state = CognitiveState.IDLE
        self.state_start_time = datetime.now()
        
        # Self-monitoring
        self.current_metrics = SelfMonitoringMetrics()
        self.metrics_history: List[SelfMonitoringMetrics] = []
        
        # Introspection logs (reflections, self-assessments)
        self.introspection_log: List[Dict[str, Any]] = []
        
        # Persistence
        self.persistence_dir = persistence_dir
        if self.persistence_dir:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
            self._load_state()
        
        logger.info("SelfAwareness system initialized")
    
    # ========================================================================
    # Identity Management
    # ========================================================================
    
    def update_identity(
        self,
        new_values: Optional[List[str]] = None,
        new_beliefs: Optional[Dict[str, Any]] = None,
        new_capabilities: Optional[Set[str]] = None,
        new_description: Optional[str] = None
    ) -> float:
        """
        Update identity model
        
        Args:
            new_values: Updated core values
            new_beliefs: Updated beliefs
            new_capabilities: Updated capabilities
            new_description: Updated self-description
            
        Returns:
            Coherence score with previous identity (0-1)
            
        Reasoning:
        - Allows identity to evolve while tracking changes
        - Returns coherence to detect dramatic shifts
        - Maintains history for continuity analysis
        """
        # Store previous identity
        previous_identity = self.current_identity
        
        # Create new identity snapshot
        self.current_identity = IdentitySnapshot(
            core_values=new_values if new_values is not None else self.current_identity.core_values,
            beliefs=new_beliefs if new_beliefs is not None else self.current_identity.beliefs,
            capabilities=new_capabilities if new_capabilities is not None else self.current_identity.capabilities,
            self_description=new_description if new_description is not None else self.current_identity.self_description
        )
        
        # Add to history
        self.identity_history.append(self.current_identity)
        
        # Keep last 100 snapshots
        if len(self.identity_history) > 100:
            self.identity_history.pop(0)
        
        # Calculate coherence with previous
        coherence = self.current_identity.similarity_to(previous_identity)
        
        # Log identity change
        self.log_introspection(
            event_type="identity_update",
            details={
                'coherence_with_previous': coherence,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        logger.info(f"Identity updated (coherence: {coherence:.2f})")
        
        return coherence
    
    def get_identity_continuity(self, time_window: Optional[timedelta] = None) -> float:
        """
        Calculate identity continuity over time window
        
        Args:
            time_window: Time period to analyze (None = all history)
            
        Returns:
            Average coherence score across time window
            
        Reasoning:
        - Measures consistency of identity over time
        - Detects identity drift or instability
        - Provides metric for identity coherence monitoring
        """
        if len(self.identity_history) < 2:
            return 1.0  # Only one snapshot = perfect continuity
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            relevant_snapshots = [
                s for s in self.identity_history
                if s.timestamp >= cutoff_time
            ]
        else:
            relevant_snapshots = self.identity_history
        
        if len(relevant_snapshots) < 2:
            return 1.0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(relevant_snapshots) - 1):
            sim = relevant_snapshots[i].similarity_to(relevant_snapshots[i + 1])
            similarities.append(sim)
        
        # Average similarity
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    # ========================================================================
    # Cognitive State Management
    # ========================================================================
    
    def set_cognitive_state(self, new_state: CognitiveState):
        """
        Update current cognitive state
        
        Args:
            new_state: New cognitive state
            
        Reasoning: Track what type of processing is currently active
        """
        if new_state != self.cognitive_state:
            old_state = self.cognitive_state
            state_duration = datetime.now() - self.state_start_time
            
            logger.debug(f"Cognitive state: {old_state.value} → {new_state.value} "
                        f"(duration: {state_duration.total_seconds():.1f}s)")
            
            self.cognitive_state = new_state
            self.state_start_time = datetime.now()
    
    def get_current_cognitive_state(self) -> Dict[str, Any]:
        """Get current cognitive state with duration"""
        duration = datetime.now() - self.state_start_time
        
        return {
            'state': self.cognitive_state.value,
            'duration_seconds': duration.total_seconds(),
            'start_time': self.state_start_time.isoformat()
        }
    
    # ========================================================================
    # Self-Monitoring
    # ========================================================================
    
    def update_monitoring_metrics(
        self,
        **metric_updates: float
    ) -> SelfMonitoringMetrics:
        """
        Update self-monitoring metrics
        
        Args:
            **metric_updates: Keyword arguments for metric updates
            
        Returns:
            Updated metrics object
            
        Reasoning:
        - Allows selective update of metrics
        - Maintains metrics history for trend analysis
        - Validates metric values
        """
        # Create new metrics from current
        new_metrics = SelfMonitoringMetrics(
            processing_efficiency=metric_updates.get('processing_efficiency', self.current_metrics.processing_efficiency),
            memory_coherence=metric_updates.get('memory_coherence', self.current_metrics.memory_coherence),
            goal_alignment=metric_updates.get('goal_alignment', self.current_metrics.goal_alignment),
            emotional_stability=metric_updates.get('emotional_stability', self.current_metrics.emotional_stability),
            emotional_range=metric_updates.get('emotional_range', self.current_metrics.emotional_range),
            identity_coherence=metric_updates.get('identity_coherence', self.current_metrics.identity_coherence),
            belief_confidence=metric_updates.get('belief_confidence', self.current_metrics.belief_confidence),
            response_quality=metric_updates.get('response_quality', self.current_metrics.response_quality),
            error_rate=metric_updates.get('error_rate', self.current_metrics.error_rate),
            learning_rate=metric_updates.get('learning_rate', self.current_metrics.learning_rate)
        )
        
        # Store in history
        self.metrics_history.append(new_metrics)
        
        # Keep last 1000 metrics
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)
        
        # Update current
        self.current_metrics = new_metrics
        
        logger.debug(f"Metrics updated: overall_health={new_metrics.get_overall_health():.2f}")
        
        return new_metrics
    
    def get_monitoring_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """
        Get summary of self-monitoring metrics
        
        Args:
            time_window: Time period to analyze (None = current only)
            
        Returns:
            Dictionary with current and trend data
            
        Reasoning: Provides overview of self-monitoring state and trends
        """
        current_health = self.current_metrics.get_overall_health()
        
        if not time_window or len(self.metrics_history) < 2:
            return {
                'current_metrics': self.current_metrics.to_dict(),
                'overall_health': current_health,
                'health_trend': 0.0,
                'samples_analyzed': 1
            }
        
        # Filter by time window
        cutoff_time = datetime.now() - time_window
        relevant_metrics = [
            m for m in self.metrics_history
            if m.timestamp >= cutoff_time
        ]
        
        if len(relevant_metrics) < 2:
            return {
                'current_metrics': self.current_metrics.to_dict(),
                'overall_health': current_health,
                'health_trend': 0.0,
                'samples_analyzed': len(relevant_metrics)
            }
        
        # Calculate health trend (linear regression slope)
        health_scores = [m.get_overall_health() for m in relevant_metrics]
        trend = (health_scores[-1] - health_scores[0]) / len(health_scores)
        
        return {
            'current_metrics': self.current_metrics.to_dict(),
            'overall_health': current_health,
            'health_trend': trend,
            'samples_analyzed': len(relevant_metrics),
            'min_health': min(health_scores),
            'max_health': max(health_scores),
            'avg_health': sum(health_scores) / len(health_scores)
        }
    
    def detect_anomalies(self, threshold: float = 0.3) -> List[str]:
        """
        Detect anomalies in self-monitoring metrics
        
        Args:
            threshold: Deviation threshold for anomaly detection
            
        Returns:
            List of detected anomalies
            
        Reasoning:
        - Identifies concerning patterns in metrics
        - Enables self-correction and intervention
        - Provides alerts for well-being issues
        """
        anomalies = []
        
        # Check overall health
        health = self.current_metrics.get_overall_health()
        if health < 0.5:
            anomalies.append(f"Low overall health: {health:.2f}")
        
        # Check specific metrics
        if self.current_metrics.processing_efficiency < threshold:
            anomalies.append(f"Low processing efficiency: {self.current_metrics.processing_efficiency:.2f}")
        
        if self.current_metrics.memory_coherence < threshold:
            anomalies.append(f"Low memory coherence: {self.current_metrics.memory_coherence:.2f}")
        
        if self.current_metrics.goal_alignment < threshold:
            anomalies.append(f"Low goal alignment: {self.current_metrics.goal_alignment:.2f}")
        
        if self.current_metrics.emotional_stability < threshold:
            anomalies.append(f"Low emotional stability: {self.current_metrics.emotional_stability:.2f}")
        
        if self.current_metrics.identity_coherence < threshold:
            anomalies.append(f"Low identity coherence: {self.current_metrics.identity_coherence:.2f}")
        
        if self.current_metrics.error_rate > (1.0 - threshold):
            anomalies.append(f"High error rate: {self.current_metrics.error_rate:.2f}")
        
        return anomalies
    
    # ========================================================================
    # Introspection
    # ========================================================================
    
    def introspect(self, query: str) -> Dict[str, Any]:
        """
        Perform introspective query on internal state
        
        Args:
            query: What aspect of self to examine
            
        Returns:
            Dictionary with introspection results
            
        Reasoning:
        - Provides meta-cognitive access to internal states
        - Enables self-reflection and self-knowledge
        - Supports debugging and self-understanding
        """
        query_lower = query.lower()
        
        # Route to appropriate introspection method
        if any(kw in query_lower for kw in ['identity', 'who', 'self', 'values']):
            return self._introspect_identity()
        
        elif any(kw in query_lower for kw in ['cognitive', 'thinking', 'processing', 'state']):
            return self._introspect_cognitive_state()
        
        elif any(kw in query_lower for kw in ['metrics', 'performance', 'health', 'well-being']):
            return self._introspect_monitoring()
        
        elif any(kw in query_lower for kw in ['continuity', 'coherence', 'consistency']):
            return self._introspect_continuity()
        
        elif any(kw in query_lower for kw in ['capabilities', 'can', 'able']):
            return self._introspect_capabilities()
        
        else:
            # General introspection
            return {
                'identity': self._introspect_identity(),
                'cognitive_state': self._introspect_cognitive_state(),
                'monitoring': self._introspect_monitoring(),
                'continuity': self._introspect_continuity()
            }
    
    def _introspect_identity(self) -> Dict[str, Any]:
        """Introspect on identity"""
        return {
            'self_description': self.current_identity.self_description,
            'core_values': self.current_identity.core_values,
            'beliefs': self.current_identity.beliefs,
            'capabilities_count': len(self.current_identity.capabilities),
            'identity_age': (datetime.now() - self.current_identity.timestamp).total_seconds() / 3600,  # hours
            'identity_changes': len(self.identity_history)
        }
    
    def _introspect_cognitive_state(self) -> Dict[str, Any]:
        """Introspect on cognitive state"""
        return self.get_current_cognitive_state()
    
    def _introspect_monitoring(self) -> Dict[str, Any]:
        """Introspect on self-monitoring metrics"""
        return self.get_monitoring_summary(time_window=timedelta(hours=1))
    
    def _introspect_continuity(self) -> Dict[str, Any]:
        """Introspect on identity continuity"""
        return {
            'identity_continuity_1h': self.get_identity_continuity(timedelta(hours=1)),
            'identity_continuity_24h': self.get_identity_continuity(timedelta(hours=24)),
            'identity_continuity_all': self.get_identity_continuity(),
            'identity_snapshots': len(self.identity_history)
        }
    
    def _introspect_capabilities(self) -> Dict[str, Any]:
        """Introspect on capabilities"""
        return {
            'capabilities': sorted(list(self.current_identity.capabilities)),
            'capability_count': len(self.current_identity.capabilities)
        }
    
    def log_introspection(self, event_type: str, details: Dict[str, Any]):
        """
        Log introspective event
        
        Args:
            event_type: Type of introspective event
            details: Event details
            
        Reasoning: Maintain log of self-reflective activities
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        self.introspection_log.append(entry)
        
        # Keep last 500 entries
        if len(self.introspection_log) > 500:
            self.introspection_log.pop(0)
    
    # ========================================================================
    # Persistence
    # ========================================================================
    
    def save_state(self):
        """Save self-awareness state to disk"""
        if not self.persistence_dir:
            logger.warning("No persistence directory set, cannot save state")
            return
        
        state_file = self.persistence_dir / "self_awareness_state.json"
        
        try:
            state = {
                'current_identity': self.current_identity.to_dict(),
                'identity_history': [i.to_dict() for i in self.identity_history[-100:]],  # Last 100
                'cognitive_state': {
                    'state': self.cognitive_state.value,
                    'start_time': self.state_start_time.isoformat()
                },
                'current_metrics': self.current_metrics.to_dict(),
                'metrics_history': [m.to_dict() for m in self.metrics_history[-500:]],  # Last 500
                'introspection_log': self.introspection_log[-500:],  # Last 500
                'last_saved': datetime.now().isoformat()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved self-awareness state to {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save self-awareness state: {e}")
    
    def _load_state(self):
        """Load self-awareness state from disk"""
        if not self.persistence_dir:
            return
        
        state_file = self.persistence_dir / "self_awareness_state.json"
        
        if not state_file.exists():
            logger.info("No saved self-awareness state found")
            return
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore identity
            self.current_identity = IdentitySnapshot.from_dict(state['current_identity'])
            self.identity_history = [
                IdentitySnapshot.from_dict(i) for i in state.get('identity_history', [])
            ]
            
            # Restore cognitive state
            cog_state_data = state.get('cognitive_state', {})
            self.cognitive_state = CognitiveState(cog_state_data.get('state', 'idle'))
            self.state_start_time = datetime.fromisoformat(
                cog_state_data.get('start_time', datetime.now().isoformat())
            )
            
            # Restore metrics
            self.current_metrics = SelfMonitoringMetrics.from_dict(state['current_metrics'])
            self.metrics_history = [
                SelfMonitoringMetrics.from_dict(m) for m in state.get('metrics_history', [])
            ]
            
            # Restore introspection log
            self.introspection_log = state.get('introspection_log', [])
            
            logger.info(f"Loaded self-awareness state from {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to load self-awareness state: {e}")
    
    # ========================================================================
    # Statistics
    # ========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get self-awareness system statistics"""
        return {
            'identity_snapshots': len(self.identity_history),
            'capabilities_count': len(self.current_identity.capabilities),
            'metrics_history_size': len(self.metrics_history),
            'introspection_log_size': len(self.introspection_log),
            'current_cognitive_state': self.cognitive_state.value,
            'overall_health': self.current_metrics.get_overall_health(),
            'identity_continuity': self.get_identity_continuity(timedelta(hours=24)),
            'anomalies_detected': len(self.detect_anomalies())
        }
