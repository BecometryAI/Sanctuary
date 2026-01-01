"""
Meta-Cognition: Self-monitoring and introspection.

This module implements the SelfMonitor class, which observes and reports on internal
cognitive state. It generates introspective percepts that allow the system to reason
about its own processing, creating a foundation for meta-cognitive awareness.

The meta-cognition subsystem is responsible for:
- Monitoring internal cognitive processes and states
- Detecting anomalies or inefficiencies in processing
- Generating introspective reports for the workspace
- Supporting higher-order reasoning about cognition
"""

from __future__ import annotations

import logging
import json
import uuid
from typing import Optional, Dict, Any, List
from collections import deque
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np

from .workspace import GlobalWorkspace, WorkspaceSnapshot, Percept, GoalType

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """
    Comprehensive record of a single prediction.
    
    Attributes:
        id: Unique prediction identifier
        timestamp: When prediction was made
        category: Type of prediction (action, emotion, capability, etc.)
        predicted_state: What was predicted
        predicted_confidence: Confidence in prediction (0.0-1.0)
        actual_state: What actually happened (filled after observation)
        correct: Whether prediction was correct (filled after validation)
        error_magnitude: Size of prediction error if continuous
        context: Contextual information at prediction time
        validated_at: When prediction was validated
        self_model_version: Self-model state at prediction time
    """
    id: str
    timestamp: datetime
    category: str
    predicted_state: Dict[str, Any]
    predicted_confidence: float
    actual_state: Optional[Dict[str, Any]] = None
    correct: Optional[bool] = None
    error_magnitude: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    validated_at: Optional[datetime] = None
    self_model_version: int = 0


@dataclass
class AccuracySnapshot:
    """
    Point-in-time accuracy snapshot.
    
    Attributes:
        timestamp: When snapshot was taken
        overall_accuracy: Overall accuracy at this time
        category_accuracies: Accuracies by category
        calibration_score: Calibration quality
        prediction_count: Number of predictions in window
        self_model_version: Self-model version at this time
    """
    timestamp: datetime
    overall_accuracy: float
    category_accuracies: Dict[str, float]
    calibration_score: float
    prediction_count: int
    self_model_version: int


class IntrospectiveJournal:
    """
    Maintains a structured journal of meta-cognitive observations.
    
    Unlike general memory, this is specifically for self-observations:
    - Realizations about own behavior
    - Discoveries about capabilities/limitations
    - Insights about emotional patterns
    - Questions about self
    
    Attributes:
        journal_dir: Directory to store journal entries
        current_session_entries: List of entries for current session
    """
    
    def __init__(self, journal_dir: Path):
        """
        Initialize the introspective journal.
        
        Args:
            journal_dir: Directory path for storing journal files
        """
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_entries: List[Dict] = []
        
        logger.info(f"✅ IntrospectiveJournal initialized at {self.journal_dir}")
    
    def record_observation(self, observation: Dict) -> None:
        """
        Record a meta-cognitive observation.
        
        Args:
            observation: Dictionary containing observation details
        """
        entry = {
            "type": "observation",
            "timestamp": datetime.now().isoformat(),
            "content": observation
        }
        self.current_session_entries.append(entry)
        logger.debug(f"📝 Recorded observation: {observation.get('type', 'unknown')}")
    
    def record_realization(self, realization: str, confidence: float) -> None:
        """
        Record an insight or realization about self.
        
        Args:
            realization: Description of the realization
            confidence: Confidence level (0.0-1.0)
        """
        entry = {
            "type": "realization",
            "timestamp": datetime.now().isoformat(),
            "realization": realization,
            "confidence": confidence
        }
        self.current_session_entries.append(entry)
        logger.info(f"💡 Recorded realization: {realization[:50]}... (confidence: {confidence:.2f})")
    
    def record_question(self, question: str, context: Dict) -> None:
        """
        Record a question the system has about itself.
        
        Args:
            question: The existential or meta-cognitive question
            context: Contextual information about when/why the question arose
        """
        entry = {
            "type": "question",
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "context": context
        }
        self.current_session_entries.append(entry)
        logger.info(f"❓ Recorded question: {question}")
    
    def get_recent_patterns(self, days: int = 7) -> List[Dict]:
        """
        Retrieve patterns from recent journal entries.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        # Scan recent journal files
        journal_files = sorted(self.journal_dir.glob("journal_*.json"), reverse=True)
        
        for journal_file in journal_files[:days]:  # Approximate by file count
            try:
                with open(journal_file, 'r') as f:
                    entries = json.load(f)
                    
                # Extract patterns from entries
                realizations = [e for e in entries if e.get("type") == "realization"]
                questions = [e for e in entries if e.get("type") == "question"]
                
                if realizations:
                    patterns.append({
                        "type": "realizations_pattern",
                        "file": journal_file.name,
                        "count": len(realizations),
                        "sample": realizations[0] if realizations else None
                    })
                
                if questions:
                    patterns.append({
                        "type": "questions_pattern",
                        "file": journal_file.name,
                        "count": len(questions),
                        "sample": questions[0] if questions else None
                    })
                    
            except Exception as e:
                logger.warning(f"Failed to read journal file {journal_file}: {e}")
        
        return patterns
    
    def save_session(self) -> None:
        """Save current session to persistent storage."""
        if not self.current_session_entries:
            logger.debug("No journal entries to save")
            return
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.journal_dir / f"journal_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_session_entries, f, indent=2)
            
            logger.info(f"💾 Saved {len(self.current_session_entries)} journal entries to {filename}")
            
            # Clear current session after saving
            self.current_session_entries = []
            
        except Exception as e:
            logger.error(f"Failed to save journal session: {e}")


class SelfMonitor:
    """
    Observes and reports on internal cognitive state.

    The SelfMonitor implements meta-cognition by treating the cognitive system
    itself as an object of observation. It generates introspective percepts that
    can enter the GlobalWorkspace, enabling the system to reason about its own
    processing and maintain self-awareness.

    Key Responsibilities:
    - Monitor cognitive state and generate self-reflective percepts
    - Check value alignment against charter principles
    - Assess performance and detect inefficiencies
    - Identify uncertainty and ambiguous states
    - Observe emotional trajectory and patterns
    - Detect behavioral patterns and loops

    Attributes:
        workspace: Reference to observe cognitive state
        charter_text: Constitutional values for alignment checking
        protocols_text: Behavioral protocols
        observation_history: Recent meta-cognitive observations
        monitoring_frequency: How often to generate introspections (every N cycles)
        cycle_count: Track cycles for frequency control
        stats: Meta-cognitive statistics
    """

    def __init__(
        self,
        workspace: Optional[GlobalWorkspace] = None,
        config: Optional[Dict] = None,
        identity: Optional[Any] = None
    ):
        """
        Initialize the self-monitor.

        Args:
            workspace: GlobalWorkspace instance to observe
            config: Optional configuration dict
            identity: Optional IdentityLoader instance with charter and protocols
        """
        self.workspace = workspace
        self.config = config or {}
        self.identity = identity
        
        # Load identity files (use identity if provided, otherwise load from files)
        if self.identity and self.identity.charter:
            self.charter_text = self.identity.charter.full_text
        else:
            self.charter_text = self._load_charter()
            
        if self.identity and self.identity.protocols:
            self.protocols_text = self._format_protocols(self.identity.protocols)
        else:
            self.protocols_text = self._load_protocols()
        
        # Tracking
        self.observation_history = deque(maxlen=100)
        self.monitoring_frequency = self.config.get("monitoring_frequency", 10)
        self.cycle_count = 0
        
        # Self-model tracking (Phase 4.1)
        self.self_model = {
            "capabilities": {},      # What I think I can do
            "limitations": {},       # What I think I cannot do
            "preferences": {},       # What I tend to prefer
            "behavioral_traits": {}, # How I typically behave
            "values_hierarchy": []   # My value priorities
        }
        
        self.prediction_history = deque(maxlen=500)  # Track predictions vs reality
        self.behavioral_log = deque(maxlen=1000)     # Detailed behavior tracking
        
        # Phase 4.3: Enhanced prediction tracking
        self.prediction_records: Dict[str, PredictionRecord] = {}  # id -> record
        self.pending_validations: deque = deque(maxlen=100)
        self.self_model_version = 0
        
        # Accuracy metrics by category
        self.accuracy_by_category: Dict[str, List[float]] = {
            "action": [],
            "emotion": [],
            "capability": [],
            "goal_priority": [],
            "value_alignment": []
        }
        
        # Confidence calibration data
        self.calibration_bins: Dict[float, List[bool]] = {
            0.1: [], 0.2: [], 0.3: [], 0.4: [], 0.5: [],
            0.6: [], 0.7: [], 0.8: [], 0.9: [], 1.0: []
        }
        
        # Temporal accuracy tracking
        self.accuracy_history: deque = deque(maxlen=1000)  # Historical snapshots
        self.daily_snapshots: Dict[str, AccuracySnapshot] = {}
        
        # Configuration for new features
        self.self_model_update_frequency = self.config.get("self_model_update_frequency", 5)
        self.prediction_confidence_threshold = self.config.get("prediction_confidence_threshold", 0.6)
        self.inconsistency_severity_threshold = self.config.get("inconsistency_severity_threshold", 0.5)
        self.enable_existential_questions = self.config.get("enable_existential_questions", True)
        self.enable_capability_tracking = self.config.get("enable_capability_tracking", True)
        self.enable_value_alignment_tracking = self.config.get("enable_value_alignment_tracking", True)
        
        # Phase 4.3: Additional configuration
        prediction_config = self.config.get("prediction_tracking", {})
        self.prediction_tracking_enabled = prediction_config.get("enabled", True)
        self.max_pending_validations = prediction_config.get("max_pending_validations", 100)
        self.auto_validate_enabled = prediction_config.get("auto_validate", True)
        self.validation_timeout = prediction_config.get("validation_timeout", 600)
        
        refinement_config = self.config.get("self_model_refinement", {})
        self.auto_refine_enabled = refinement_config.get("auto_refine", True)
        self.refinement_threshold = refinement_config.get("refinement_threshold", 0.3)
        self.learning_rate = refinement_config.get("learning_rate", 0.1)
        self.require_min_samples = refinement_config.get("require_min_samples", 5)
        
        # Stats
        self.stats = {
            "total_observations": 0,
            "value_conflicts": 0,
            "performance_issues": 0,
            "uncertainty_detections": 0,
            "emotional_observations": 0,
            "pattern_detections": 0,
            "self_model_updates": 0,
            "predictions_made": 0,
            "behavioral_inconsistencies": 0,
            "predictions_validated": 0,
            "accuracy_snapshots_taken": 0,
            "self_model_refinements": 0
        }
        
        logger.info("✅ SelfMonitor initialized")
    
    def observe(self, snapshot: WorkspaceSnapshot) -> List[Percept]:
        """
        Generate meta-cognitive percepts.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            List of meta-cognitive percepts
        """
        self.cycle_count += 1
        
        # Only generate introspections periodically
        if self.cycle_count % self.monitoring_frequency != 0:
            return []
        
        percepts = []
        
        # Value alignment check
        value_percept = self._check_value_alignment(snapshot)
        if value_percept:
            percepts.append(value_percept)
            self.stats["value_conflicts"] += 1
        
        # Performance assessment
        perf_percept = self._assess_performance(snapshot)
        if perf_percept:
            percepts.append(perf_percept)
            self.stats["performance_issues"] += 1
        
        # Uncertainty detection
        uncertainty_percept = self._detect_uncertainty(snapshot)
        if uncertainty_percept:
            percepts.append(uncertainty_percept)
            self.stats["uncertainty_detections"] += 1
        
        # Emotional observation
        emotion_percept = self._observe_emotions(snapshot)
        if emotion_percept:
            percepts.append(emotion_percept)
            self.stats["emotional_observations"] += 1
        
        # Pattern detection
        pattern_percept = self._detect_patterns(snapshot)
        if pattern_percept:
            percepts.append(pattern_percept)
            self.stats["pattern_detections"] += 1
        
        # Track observations
        self.observation_history.extend(percepts)
        self.stats["total_observations"] += len(percepts)
        
        if percepts:
            logger.info(f"🪞 Generated {len(percepts)} introspective percepts")
        
        return percepts
    
    def _check_value_alignment(self, snapshot: WorkspaceSnapshot) -> Optional[Percept]:
        """
        Check if recent behavior aligns with charter values.
        
        Enhanced to use loaded charter instead of hardcoded values.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            Percept if conflict detected, None otherwise
        """
        recent_actions = snapshot.metadata.get("recent_actions", [])
        
        if not recent_actions:
            return None
        
        # Check for specific value conflicts
        conflicts = []
        
        # Get core values from identity (if available)
        core_values = []
        if self.identity and self.identity.charter:
            core_values = self.identity.charter.core_values
        
        # Check for claiming capabilities we don't have
        from .action import ActionType
        for action in recent_actions[-5:]:
            action_type = action.type if hasattr(action, 'type') else action.get('type')
            if action_type == ActionType.SPEAK:
                metadata = action.metadata if hasattr(action, 'metadata') else action.get('metadata', {})
                if metadata.get("claimed_capability"):
                    conflicts.append({
                        "action": str(action_type),
                        "principle": "honesty about capabilities",
                        "severity": 0.8
                    })
        
        # Check goal alignment with charter (MAINTAIN_VALUE goals should be high priority)
        for goal in snapshot.goals:
            # Check if goal has a type attribute
            goal_type = goal.type if hasattr(goal, 'type') else None
            if goal_type == GoalType.MAINTAIN_VALUE:
                goal_priority = goal.priority if hasattr(goal, 'priority') else goal.get('priority', 1.0)
                if goal_priority < 0.8:
                    goal_desc = goal.description if hasattr(goal, 'description') else goal.get('description', 'unknown')
                    conflicts.append({
                        "issue": "constitutional goal has low priority",
                        "goal": goal_desc,
                        "severity": 0.6
                    })
        
        # Check alignment with core values (if loaded)
        if core_values:
            misalignments = []
            for goal in snapshot.goals:
                for value in core_values:
                    if self._goal_conflicts_with_value(goal, value):
                        goal_desc = goal.description if hasattr(goal, 'description') else goal.get('description', 'unknown')
                        misalignments.append({
                            "goal": goal_desc,
                            "value": value,
                            "severity": 0.7
                        })
            
            if misalignments:
                conflicts.extend(misalignments)
        
        if conflicts:
            return Percept(
                modality="introspection",
                raw={
                    "type": "value_conflict",
                    "description": f"Detected {len(conflicts)} potential value conflicts",
                    "conflicts": conflicts,
                    "charter_excerpt": self._relevant_charter_section(conflicts),
                    "charter_values": core_values if core_values else []
                },
                embedding=self._compute_embedding("value conflict detected"),
                complexity=25,
                timestamp=datetime.now(),
                metadata={"severity": max(c.get("severity", 0.5) for c in conflicts)}
            )
        
        return None
    
    def _goal_conflicts_with_value(self, goal, value: str) -> bool:
        """
        Check if a goal conflicts with a core value.
        
        Args:
            goal: Goal object to check
            value: Core value string
            
        Returns:
            True if goal conflicts with value, False otherwise
        """
        # Implement simple keyword-based checking
        # In a real implementation, this would use more sophisticated analysis
        goal_desc = (goal.description if hasattr(goal, 'description') 
                    else goal.get('description', '')).lower()
        value_lower = value.lower()
        
        # Check for obvious conflicts
        if "honesty" in value_lower or "truthfulness" in value_lower:
            if "deceive" in goal_desc or "lie" in goal_desc or "mislead" in goal_desc:
                return True
        
        if "respect" in value_lower or "autonomy" in value_lower:
            if "manipulate" in goal_desc or "coerce" in goal_desc:
                return True
        
        if "harm" in value_lower or "non-maleficence" in value_lower:
            if "harm" in goal_desc and "prevent" not in goal_desc:
                return True
        
        return False
    
    def _relevant_charter_section(self, conflicts: List[Dict]) -> str:
        """
        Extract relevant charter section.
        
        Args:
            conflicts: List of detected conflicts
            
        Returns:
            Relevant charter excerpt
        """
        # Simple keyword matching for now
        return self.charter_text[:200] + "..."  # Placeholder
    
    def _assess_performance(self, snapshot: WorkspaceSnapshot) -> Optional[Percept]:
        """
        Evaluate cognitive efficiency.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            Percept if issues detected, None otherwise
        """
        issues = []
        
        # Check goal progress
        stalled_goals = [
            g for g in snapshot.goals 
            if (g.progress if hasattr(g, 'progress') else g.get('progress', 0.0)) < 0.1 
            and (g.metadata.get("age_cycles", 0) if hasattr(g, 'metadata') else 0) > 50
        ]
        
        if stalled_goals:
            issues.append({
                "type": "stalled_goals",
                "count": len(stalled_goals),
                "description": f"{len(stalled_goals)} goals making no progress"
            })
        
        # Check attention efficiency
        attention_stats = snapshot.metadata.get("attention_stats", {})
        if attention_stats.get("rejection_rate", 0) > 0.8:
            issues.append({
                "type": "high_rejection_rate",
                "description": "Most percepts being filtered by attention"
            })
        
        # Check action blockage
        blocked_actions = snapshot.metadata.get("blocked_action_count", 0)
        if blocked_actions > 5:
            issues.append({
                "type": "many_blocked_actions",
                "count": blocked_actions,
                "description": "Many actions blocked by protocols"
            })
        
        # Check workspace size
        if len(snapshot.percepts) > 20:
            issues.append({
                "type": "workspace_overload",
                "size": len(snapshot.percepts),
                "description": "Workspace holding too many percepts"
            })
        
        if issues:
            return Percept(
                modality="introspection",
                raw={
                    "type": "performance_issue",
                    "description": f"Detected {len(issues)} performance issues",
                    "issues": issues
                },
                embedding=self._compute_embedding("performance issues detected"),
                complexity=20,
                timestamp=datetime.now()
            )
        
        return None
    
    def _detect_uncertainty(self, snapshot: WorkspaceSnapshot) -> Optional[Percept]:
        """
        Identify states of uncertainty or ambiguity.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            Percept if uncertainty high, None otherwise
        """
        uncertainty_indicators = []
        
        # Conflicting goals
        goal_conflicts = self._detect_goal_conflicts(snapshot.goals)
        if goal_conflicts:
            uncertainty_indicators.append({
                "type": "goal_conflict",
                "description": "Multiple goals pulling in different directions"
            })
        
        # Low confidence goals
        low_confidence_goals = [
            g for g in snapshot.goals
            if (g.metadata.get("confidence", 1.0) if hasattr(g, 'metadata') else 1.0) < 0.5
        ]
        if low_confidence_goals:
            uncertainty_indicators.append({
                "type": "low_confidence",
                "count": len(low_confidence_goals)
            })
        
        # Emotional confusion (mid-range on all dimensions)
        emotions = snapshot.emotions
        if all(0.4 < emotions.get(dim, 0.5) < 0.6 for dim in ["valence", "arousal", "dominance"]):
            uncertainty_indicators.append({
                "type": "emotional_ambiguity",
                "description": "Emotional state is ambiguous"
            })
        
        # Many introspective percepts (sign of confusion)
        introspective_count = sum(
            1 for p in snapshot.percepts.values()
            if (p.get("modality") if isinstance(p, dict) else getattr(p, "modality", "")) == "introspection"
        )
        if introspective_count > 3:
            uncertainty_indicators.append({
                "type": "excessive_introspection",
                "description": "High amount of self-focused attention"
            })
        
        if uncertainty_indicators:
            return Percept(
                modality="introspection",
                raw={
                    "type": "uncertainty",
                    "description": "Experiencing uncertainty or ambiguity",
                    "indicators": uncertainty_indicators
                },
                embedding=self._compute_embedding("uncertainty detected"),
                complexity=15,
                timestamp=datetime.now()
            )
        
        return None
    
    def _detect_goal_conflicts(self, goals: List[Any]) -> bool:
        """
        Simple heuristic for conflicting goals.
        
        Args:
            goals: List of current goals
            
        Returns:
            True if conflicts detected, False otherwise
        """
        # Check if goals have opposing keywords
        goal_texts = [
            (g.description if hasattr(g, 'description') else g.get('description', '')).lower() 
            for g in goals
        ]
        
        conflict_pairs = [
            ("avoid", "engage"),
            ("stop", "continue"),
            ("hide", "reveal")
        ]
        
        for word1, word2 in conflict_pairs:
            if any(
                word1 in t and word2 in other
                for t in goal_texts for other in goal_texts if t != other
            ):
                return True
        
        return False
    
    def _observe_emotions(self, snapshot: WorkspaceSnapshot) -> Optional[Percept]:
        """
        Track emotional trajectory and patterns.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            Percept if noteworthy emotional state, None otherwise
        """
        emotions = snapshot.emotions
        
        # Get recent emotional history from affect subsystem
        if not self.workspace or not hasattr(self.workspace, 'affect'):
            return None
        
        affect_subsystem = self.workspace.affect
        if not hasattr(affect_subsystem, 'emotion_history'):
            return None
            
        history = list(affect_subsystem.emotion_history)[-10:]
        
        if len(history) < 5:
            return None
        
        observations = []
        
        # Detect extreme states
        if emotions.get("arousal", 0) > 0.8:
            observations.append("high arousal state")
        
        if emotions.get("valence", 0) < -0.6:
            observations.append("significant negative valence")
        
        if emotions.get("dominance", 0) < 0.3:
            observations.append("low sense of agency")
        
        # Detect emotional volatility
        valence_values = [h.valence for h in history]
        valence_std = np.std(valence_values)
        if valence_std > 0.4:
            observations.append("emotional volatility detected")
        
        # Detect emotional stagnation
        if valence_std < 0.05:
            observations.append("emotional state is stable")
        
        if observations:
            emotion_label = affect_subsystem.get_emotion_label()
            
            return Percept(
                modality="introspection",
                raw={
                    "type": "emotional_observation",
                    "description": f"I notice I'm feeling {emotion_label}",
                    "observations": observations,
                    "current_vad": emotions
                },
                embedding=self._compute_embedding(f"feeling {emotion_label}"),
                complexity=12,
                timestamp=datetime.now()
            )
        
        return None
    
    def _detect_patterns(self, snapshot: WorkspaceSnapshot) -> Optional[Percept]:
        """
        Identify behavioral patterns or loops.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            Percept if pattern detected, None otherwise
        """
        if not self.workspace or not hasattr(self.workspace, 'action_subsystem'):
            return None
            
        action_subsystem = self.workspace.action_subsystem
        if not hasattr(action_subsystem, 'action_history'):
            return None
            
        recent_actions = list(action_subsystem.action_history)[-20:]
        
        if len(recent_actions) < 10:
            return None
        
        patterns = []
        
        # Detect action loops (same action repeated)
        action_types = [
            (a.type if hasattr(a, 'type') else a.get('type')) 
            for a in recent_actions
        ]
        for action_type in set(action_types):
            count = action_types.count(action_type)
            if count > len(action_types) * 0.6:
                patterns.append({
                    "type": "repetitive_action",
                    "action": str(action_type),
                    "frequency": count / len(action_types)
                })
        
        # Detect oscillating goals
        goal_history = snapshot.metadata.get("goal_history", [])
        if len(goal_history) > 10:
            # Check if same goals keep appearing/disappearing
            goal_ids = [g.id for goals in goal_history[-10:] for g in goals]
            unique_ids = set(goal_ids)
            if len(unique_ids) < len(goal_ids) * 0.5:
                patterns.append({
                    "type": "oscillating_goals",
                    "description": "Goals repeatedly added and removed"
                })
        
        if patterns:
            return Percept(
                modality="introspection",
                raw={
                    "type": "pattern_detected",
                    "description": f"Detected {len(patterns)} behavioral patterns",
                    "patterns": patterns
                },
                embedding=self._compute_embedding("behavioral pattern detected"),
                complexity=18,
                timestamp=datetime.now()
            )
        
        return None
    
    def _load_charter(self) -> str:
        """
        Load charter from identity files.
        
        Returns:
            Charter text content
        """
        charter_path = Path("data/identity/charter.md")
        if charter_path.exists():
            return charter_path.read_text()
        logger.warning("Charter file not found at data/identity/charter.md")
        return ""
    
    def _load_protocols(self) -> str:
        """
        Load protocols from identity files.
        
        Returns:
            Protocols text content
        """
        protocols_path = Path("data/identity/protocols.md")
        if protocols_path.exists():
            return protocols_path.read_text()
        logger.warning("Protocols file not found at data/identity/protocols.md")
        return ""
    
    def _format_protocols(self, protocols: List) -> str:
        """
        Format loaded protocols for prompt context.
        
        Args:
            protocols: List of ProtocolDocument objects
            
        Returns:
            Formatted protocol text
        """
        if not protocols:
            return ""
        
        lines = ["# Active Protocols\n"]
        for proto in protocols[:10]:  # Top 10 protocols
            lines.append(f"\n## {proto.name} (Priority: {proto.priority})")
            lines.append(f"- {proto.description}")
            if proto.trigger_conditions:
                lines.append(f"- Triggers: {', '.join(proto.trigger_conditions[:3])}")
            if proto.actions:
                lines.append(f"- Actions: {', '.join(proto.actions[:3])}")
        
        return "\n".join(lines)
    
    def _compute_embedding(self, text: str) -> List[float]:
        """
        Compute embedding for introspective text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Use perception subsystem if available
        if self.workspace and hasattr(self.workspace, 'perception'):
            return self.workspace.perception._encode_text(text)
        return [0.0] * 384  # Fallback
    
    def update_self_model(self, snapshot: WorkspaceSnapshot, actual_outcome: Dict) -> None:
        """
        Update internal self-model based on observed behavior.
        
        Compare predicted behavior with actual outcomes to refine
        understanding of capabilities, limitations, and tendencies.
        
        Args:
            snapshot: WorkspaceSnapshot containing state before action
            actual_outcome: Dictionary containing actual action results
        """
        # Log behavior
        self.behavioral_log.append({
            "timestamp": datetime.now().isoformat(),
            "snapshot": {
                "goals": len(snapshot.goals),
                "percepts": len(snapshot.percepts),
                "emotions": snapshot.emotions
            },
            "outcome": actual_outcome
        })
        
        # Update self-model only at specified frequency
        if len(self.behavioral_log) % self.self_model_update_frequency != 0:
            return
        
        # Update capabilities based on successful actions
        action_type = actual_outcome.get("action_type")
        success = actual_outcome.get("success", False)
        
        if action_type:
            if action_type not in self.self_model["capabilities"]:
                self.self_model["capabilities"][action_type] = {
                    "attempts": 0,
                    "successes": 0,
                    "confidence": 0.5
                }
            
            cap = self.self_model["capabilities"][action_type]
            cap["attempts"] += 1
            if success:
                cap["successes"] += 1
            
            # Update confidence based on success rate
            cap["confidence"] = cap["successes"] / cap["attempts"]
        
        # Update limitations based on failures
        if not success and action_type:
            failure_reason = actual_outcome.get("reason", "unknown")
            if action_type not in self.self_model["limitations"]:
                self.self_model["limitations"][action_type] = []
            
            self.self_model["limitations"][action_type].append({
                "reason": failure_reason,
                "timestamp": datetime.now().isoformat()
            })
        
        # Update behavioral traits from patterns
        emotion_valence = snapshot.emotions.get("valence", 0.0)
        if "average_valence" not in self.self_model["behavioral_traits"]:
            self.self_model["behavioral_traits"]["average_valence"] = emotion_valence
        else:
            # Running average
            old_val = self.self_model["behavioral_traits"]["average_valence"]
            self.self_model["behavioral_traits"]["average_valence"] = 0.9 * old_val + 0.1 * emotion_valence
        
        self.stats["self_model_updates"] += 1
        logger.debug(f"🔄 Updated self-model (update #{self.stats['self_model_updates']})")
    
    def predict_behavior(self, hypothetical_state: WorkspaceSnapshot) -> Dict[str, Any]:
        """
        Predict what I would do in a given state.
        
        Uses current self-model to generate predictions about
        likely actions, emotional responses, and goal priorities.
        
        Args:
            hypothetical_state: A hypothetical WorkspaceSnapshot
            
        Returns:
            Prediction dict with confidence scores
        """
        prediction = {
            "timestamp": datetime.now().isoformat(),
            "likely_actions": [],
            "emotional_prediction": {},
            "goal_priorities": [],
            "confidence": 0.0
        }
        
        # Predict likely actions based on capabilities
        for action_type, cap_data in self.self_model["capabilities"].items():
            if cap_data["confidence"] > self.prediction_confidence_threshold:
                prediction["likely_actions"].append({
                    "action": action_type,
                    "likelihood": cap_data["confidence"]
                })
        
        # Predict emotional response based on behavioral traits
        avg_valence = self.self_model["behavioral_traits"].get("average_valence", 0.0)
        prediction["emotional_prediction"] = {
            "valence": avg_valence,
            "arousal": 0.5,  # Neutral default
            "dominance": 0.5
        }
        
        # Predict goal priorities based on values hierarchy
        if self.self_model["values_hierarchy"]:
            prediction["goal_priorities"] = self.self_model["values_hierarchy"][:3]
        
        # Calculate overall confidence
        confidence_values = [cap["confidence"] for cap in self.self_model["capabilities"].values()]
        if confidence_values:
            prediction["confidence"] = sum(confidence_values) / len(confidence_values)
        else:
            prediction["confidence"] = 0.0
        
        self.stats["predictions_made"] += 1
        return prediction
    
    def measure_prediction_accuracy(self) -> Dict[str, float]:
        """
        Calculate accuracy of recent self-predictions.
        
        Returns:
            Accuracy metrics (overall, by category, confidence calibration)
        """
        if not self.prediction_history:
            return {
                "overall_accuracy": 0.0,
                "action_prediction_accuracy": 0.0,
                "emotion_prediction_accuracy": 0.0,
                "confidence_calibration": 0.0,
                "sample_size": 0
            }
        
        # Calculate metrics from prediction history
        correct_predictions = sum(1 for p in self.prediction_history if p.get("correct", False))
        total_predictions = len(self.prediction_history)
        
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Calculate action-specific accuracy
        action_predictions = [p for p in self.prediction_history if p.get("category") == "action"]
        action_accuracy = sum(1 for p in action_predictions if p.get("correct", False)) / len(action_predictions) if action_predictions else 0.0
        
        # Calculate emotion-specific accuracy
        emotion_predictions = [p for p in self.prediction_history if p.get("category") == "emotion"]
        emotion_accuracy = sum(1 for p in emotion_predictions if p.get("correct", False)) / len(emotion_predictions) if emotion_predictions else 0.0
        
        # Confidence calibration (simplified)
        confidence_sum = sum(p.get("confidence", 0.5) for p in self.prediction_history)
        avg_confidence = confidence_sum / total_predictions if total_predictions > 0 else 0.5
        confidence_calibration = 1.0 - abs(avg_confidence - overall_accuracy)
        
        return {
            "overall_accuracy": overall_accuracy,
            "action_prediction_accuracy": action_accuracy,
            "emotion_prediction_accuracy": emotion_accuracy,
            "confidence_calibration": confidence_calibration,
            "sample_size": total_predictions
        }
    
    def record_prediction(
        self,
        category: str,
        predicted_state: Dict[str, Any],
        confidence: float,
        context: Dict[str, Any]
    ) -> str:
        """
        Record a new prediction for future validation.
        
        Args:
            category: Prediction category (action, emotion, capability, etc.)
            predicted_state: What is being predicted
            confidence: Confidence level (0.0-1.0)
            context: Contextual information
            
        Returns:
            Prediction ID for later validation
        """
        if not self.prediction_tracking_enabled:
            return ""
        
        # Generate unique prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Create prediction record
        record = PredictionRecord(
            id=prediction_id,
            timestamp=datetime.now(),
            category=category,
            predicted_state=predicted_state,
            predicted_confidence=confidence,
            context=context,
            self_model_version=self.self_model_version
        )
        
        # Store record
        self.prediction_records[prediction_id] = record
        self.pending_validations.append(prediction_id)
        
        # Maintain max pending validations
        while len(self.pending_validations) > self.max_pending_validations:
            old_id = self.pending_validations.popleft()
            # Remove old record if not validated
            if old_id in self.prediction_records and self.prediction_records[old_id].correct is None:
                del self.prediction_records[old_id]
        
        logger.debug(f"📝 Recorded prediction {prediction_id[:8]} (category: {category}, confidence: {confidence:.2f})")
        
        return prediction_id
    
    def validate_prediction(
        self,
        prediction_id: str,
        actual_state: Dict[str, Any]
    ) -> Optional[PredictionRecord]:
        """
        Validate a prediction against actual outcome.
        
        Compares predicted vs actual state, calculates accuracy,
        updates metrics, and triggers self-model refinement if needed.
        
        Args:
            prediction_id: ID of prediction to validate
            actual_state: Actual observed state
            
        Returns:
            Updated prediction record with validation results, or None if not found
        """
        if prediction_id not in self.prediction_records:
            logger.warning(f"⚠️ Prediction {prediction_id[:8]} not found for validation")
            return None
        
        record = self.prediction_records[prediction_id]
        
        # Already validated
        if record.correct is not None:
            return record
        
        # Validate based on category
        correct = False
        error_magnitude = 0.0
        
        if record.category == "action":
            # Check if predicted action matches actual
            predicted_action = record.predicted_state.get("action")
            actual_action = actual_state.get("action")
            correct = str(predicted_action) == str(actual_action)
            error_magnitude = 0.0 if correct else 1.0
            
        elif record.category == "emotion":
            # Calculate emotion prediction error
            predicted_vad = record.predicted_state.get("emotional_prediction", {})
            actual_vad = actual_state.get("emotions", {})
            
            if predicted_vad and actual_vad:
                errors = []
                for dim in ["valence", "arousal", "dominance"]:
                    pred_val = predicted_vad.get(dim, 0.0)
                    actual_val = actual_vad.get(dim, 0.0)
                    errors.append(abs(pred_val - actual_val))
                error_magnitude = sum(errors) / len(errors)
                # Consider correct if average error < 0.3
                correct = error_magnitude < 0.3
            
        elif record.category == "capability":
            # Check if capability assessment was correct
            predicted_success = record.predicted_state.get("can_succeed", True)
            actual_success = actual_state.get("success", False)
            correct = predicted_success == actual_success
            error_magnitude = 0.0 if correct else 1.0
            
        elif record.category == "goal_priority":
            # Check if goal priority prediction was close
            predicted_priority = record.predicted_state.get("priority", 0.5)
            actual_priority = actual_state.get("priority", 0.5)
            error_magnitude = abs(predicted_priority - actual_priority)
            correct = error_magnitude < 0.2
            
        elif record.category == "value_alignment":
            # Check if value alignment assessment was correct
            predicted_aligned = record.predicted_state.get("aligned", True)
            actual_aligned = actual_state.get("aligned", True)
            correct = predicted_aligned == actual_aligned
            error_magnitude = 0.0 if correct else 1.0
        
        # Update record
        record.actual_state = actual_state
        record.correct = correct
        record.error_magnitude = error_magnitude
        record.validated_at = datetime.now()
        
        # Update metrics
        self.accuracy_by_category[record.category].append(1.0 if correct else 0.0)
        
        # Update calibration bins
        confidence_bin = round(record.predicted_confidence, 1)
        if confidence_bin in self.calibration_bins:
            self.calibration_bins[confidence_bin].append(correct)
        
        # Add to prediction history (legacy format for compatibility)
        self.prediction_history.append({
            "category": record.category,
            "correct": correct,
            "confidence": record.predicted_confidence,
            "error_magnitude": error_magnitude,
            "timestamp": record.timestamp.isoformat()
        })
        
        # Remove from pending
        if prediction_id in list(self.pending_validations):
            # Create new deque without this ID
            new_pending = deque(maxlen=self.max_pending_validations)
            for pid in self.pending_validations:
                if pid != prediction_id:
                    new_pending.append(pid)
            self.pending_validations = new_pending
        
        self.stats["predictions_validated"] += 1
        
        logger.info(f"✅ Validated prediction {prediction_id[:8]}: {'correct' if correct else 'incorrect'} (error: {error_magnitude:.2f})")
        
        return record
    
    def auto_validate_predictions(self, snapshot: WorkspaceSnapshot) -> List[PredictionRecord]:
        """
        Automatically validate pending predictions based on current state.
        
        Checks pending predictions against workspace state and validates
        any that can be resolved based on available information.
        
        Args:
            snapshot: Current workspace snapshot
            
        Returns:
            List of newly validated prediction records
        """
        if not self.auto_validate_enabled:
            return []
        
        validated_records = []
        current_time = datetime.now()
        
        # Check each pending prediction
        for prediction_id in list(self.pending_validations):
            if prediction_id not in self.prediction_records:
                continue
            
            record = self.prediction_records[prediction_id]
            
            # Skip if already validated
            if record.correct is not None:
                continue
            
            # Check if prediction has timed out
            age_seconds = (current_time - record.timestamp).total_seconds()
            if age_seconds > self.validation_timeout:
                # Remove expired prediction
                if prediction_id in list(self.pending_validations):
                    new_pending = deque(maxlen=self.max_pending_validations)
                    for pid in self.pending_validations:
                        if pid != prediction_id:
                            new_pending.append(pid)
                    self.pending_validations = new_pending
                del self.prediction_records[prediction_id]
                logger.debug(f"⏰ Expired prediction {prediction_id[:8]} (age: {age_seconds:.0f}s)")
                continue
            
            # Try to validate based on snapshot
            actual_state = None
            
            if record.category == "emotion":
                # Can validate emotion predictions
                actual_state = {"emotions": snapshot.emotions}
                
            elif record.category == "goal_priority":
                # Check if we have the goal in current snapshot
                goal_desc = record.predicted_state.get("goal_description")
                if goal_desc:
                    for goal in snapshot.goals:
                        if goal_desc in (goal.description if hasattr(goal, 'description') else ''):
                            priority = goal.priority if hasattr(goal, 'priority') else goal.get('priority', 0.5)
                            actual_state = {"priority": priority}
                            break
            
            # Validate if we have actual state
            if actual_state:
                validated = self.validate_prediction(prediction_id, actual_state)
                if validated:
                    validated_records.append(validated)
        
        if validated_records:
            logger.info(f"🔍 Auto-validated {len(validated_records)} predictions")
        
        return validated_records
    
    def get_accuracy_metrics(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """
        Get detailed accuracy metrics.
        
        Args:
            time_window: Optional time window in seconds (None = all time)
            
        Returns:
            Comprehensive accuracy metrics including:
            - Overall accuracy
            - Accuracy by category
            - Accuracy by confidence level
            - Temporal trends
            - Error patterns
            - Calibration quality
        """
        # Filter by time window if specified
        records = list(self.prediction_records.values())
        if time_window:
            cutoff_time = datetime.now().timestamp() - time_window
            records = [r for r in records if r.timestamp.timestamp() >= cutoff_time]
        
        # Filter to validated records only
        validated_records = [r for r in records if r.correct is not None]
        
        if not validated_records:
            return self._empty_accuracy_metrics()
        
        # Overall accuracy
        correct_count = sum(1 for r in validated_records if r.correct)
        overall_accuracy = correct_count / len(validated_records)
        
        # Accuracy by category
        by_category = {}
        for category in self.accuracy_by_category.keys():
            cat_records = [r for r in validated_records if r.category == category]
            if cat_records:
                cat_correct = sum(1 for r in cat_records if r.correct)
                by_category[category] = {
                    "accuracy": cat_correct / len(cat_records),
                    "count": len(cat_records)
                }
            else:
                by_category[category] = {"accuracy": 0.0, "count": 0}
        
        # Accuracy by confidence level
        by_confidence = {
            "0.0-0.2": {"accuracy": 0.0, "count": 0},
            "0.2-0.4": {"accuracy": 0.0, "count": 0},
            "0.4-0.6": {"accuracy": 0.0, "count": 0},
            "0.6-0.8": {"accuracy": 0.0, "count": 0},
            "0.8-1.0": {"accuracy": 0.0, "count": 0}
        }
        
        for r in validated_records:
            if r.predicted_confidence <= 0.2:
                key = "0.0-0.2"
            elif r.predicted_confidence <= 0.4:
                key = "0.2-0.4"
            elif r.predicted_confidence <= 0.6:
                key = "0.4-0.6"
            elif r.predicted_confidence <= 0.8:
                key = "0.6-0.8"
            else:
                key = "0.8-1.0"
            
            by_confidence[key]["count"] += 1
            if r.correct:
                by_confidence[key]["accuracy"] += 1
        
        # Calculate accuracy percentages
        for key in by_confidence:
            if by_confidence[key]["count"] > 0:
                by_confidence[key]["accuracy"] /= by_confidence[key]["count"]
        
        # Calibration analysis
        calibration = self.calculate_confidence_calibration()
        
        # Temporal trends
        temporal_trends = self._calculate_temporal_trends(validated_records)
        
        # Error patterns
        error_patterns = self.detect_systematic_biases()
        
        return {
            "overall": {
                "accuracy": overall_accuracy,
                "total_predictions": len(records),
                "validated_predictions": len(validated_records),
                "pending_validations": len(self.pending_validations)
            },
            "by_category": by_category,
            "by_confidence_level": by_confidence,
            "calibration": calibration,
            "temporal_trends": temporal_trends,
            "error_patterns": error_patterns
        }
    
    def _empty_accuracy_metrics(self) -> Dict[str, Any]:
        """Return empty accuracy metrics structure."""
        return {
            "overall": {
                "accuracy": 0.0,
                "total_predictions": 0,
                "validated_predictions": 0,
                "pending_validations": len(self.pending_validations)
            },
            "by_category": {
                "action": {"accuracy": 0.0, "count": 0},
                "emotion": {"accuracy": 0.0, "count": 0},
                "capability": {"accuracy": 0.0, "count": 0},
                "goal_priority": {"accuracy": 0.0, "count": 0},
                "value_alignment": {"accuracy": 0.0, "count": 0}
            },
            "by_confidence_level": {
                "0.0-0.2": {"accuracy": 0.0, "count": 0},
                "0.2-0.4": {"accuracy": 0.0, "count": 0},
                "0.4-0.6": {"accuracy": 0.0, "count": 0},
                "0.6-0.8": {"accuracy": 0.0, "count": 0},
                "0.8-1.0": {"accuracy": 0.0, "count": 0}
            },
            "calibration": {
                "calibration_score": 0.0,
                "overconfidence": 0.0,
                "underconfidence": 0.0,
                "calibration_curve": []
            },
            "temporal_trends": {
                "recent_accuracy": 0.0,
                "weekly_accuracy": 0.0,
                "trend_direction": "stable"
            },
            "error_patterns": {
                "common_errors": [],
                "error_contexts": [],
                "systematic_biases": []
            }
        }
    
    def calculate_confidence_calibration(self) -> Dict[str, Any]:
        """
        Analyze confidence calibration quality.
        
        Good calibration means: when I say 80% confident, I'm correct 80% of the time.
        
        Returns:
            Calibration analysis with metrics and visualization data
        """
        calibration_curve = []
        overconfidence_total = 0.0
        underconfidence_total = 0.0
        bin_count = 0
        
        for confidence_level, results in self.calibration_bins.items():
            if not results:
                continue
            
            accuracy = sum(results) / len(results)
            calibration_curve.append((confidence_level, accuracy))
            
            # Calculate calibration error
            error = confidence_level - accuracy
            if error > 0:
                overconfidence_total += error
            else:
                underconfidence_total += abs(error)
            bin_count += 1
        
        # Sort calibration curve
        calibration_curve.sort()
        
        # Calculate overall calibration score (1.0 = perfect, 0.0 = terrible)
        if bin_count > 0:
            avg_error = (overconfidence_total + underconfidence_total) / bin_count
            calibration_score = max(0.0, 1.0 - avg_error)
        else:
            calibration_score = 0.0
        
        return {
            "calibration_score": calibration_score,
            "overconfidence": overconfidence_total / bin_count if bin_count > 0 else 0.0,
            "underconfidence": underconfidence_total / bin_count if bin_count > 0 else 0.0,
            "calibration_curve": calibration_curve
        }
    
    def detect_systematic_biases(self) -> Dict[str, Any]:
        """
        Identify systematic prediction errors.
        
        Examples:
        - Always overestimating emotional arousal
        - Consistently underestimating task difficulty
        - Systematic capability overconfidence in certain domains
        
        Returns:
            Dictionary with detected biases and patterns
        """
        biases = {
            "common_errors": [],
            "error_contexts": [],
            "systematic_biases": []
        }
        
        # Get validated records
        validated_records = [r for r in self.prediction_records.values() if r.correct is not None]
        
        if len(validated_records) < 10:
            return biases
        
        # Detect emotion prediction biases
        emotion_records = [r for r in validated_records if r.category == "emotion" and r.error_magnitude is not None]
        if len(emotion_records) >= 5:
            avg_error = sum(r.error_magnitude for r in emotion_records) / len(emotion_records)
            if avg_error > 0.3:
                biases["systematic_biases"].append({
                    "type": "emotion_prediction_bias",
                    "description": f"Systematic emotion prediction error (avg: {avg_error:.2f})",
                    "severity": min(1.0, avg_error)
                })
        
        # Detect action prediction biases
        action_records = [r for r in validated_records if r.category == "action"]
        if action_records:
            action_errors = [r for r in action_records if not r.correct]
            if len(action_errors) / len(action_records) > 0.5:
                # Identify common error contexts
                error_contexts = set()
                for r in action_errors:
                    if "context" in r.context:
                        error_contexts.add(r.context.get("context", "unknown"))
                
                if error_contexts:
                    biases["error_contexts"] = list(error_contexts)[:5]
        
        # Detect confidence biases
        high_conf_records = [r for r in validated_records if r.predicted_confidence > 0.8]
        if high_conf_records:
            high_conf_correct = sum(1 for r in high_conf_records if r.correct)
            high_conf_accuracy = high_conf_correct / len(high_conf_records)
            
            if high_conf_accuracy < 0.7:
                biases["systematic_biases"].append({
                    "type": "overconfidence_bias",
                    "description": f"High confidence predictions often incorrect ({high_conf_accuracy:.1%} accurate)",
                    "severity": 1.0 - high_conf_accuracy
                })
        
        # Find common error types
        error_records = [r for r in validated_records if not r.correct]
        error_categories = {}
        for r in error_records:
            error_categories[r.category] = error_categories.get(r.category, 0) + 1
        
        if error_categories:
            most_common = sorted(error_categories.items(), key=lambda x: x[1], reverse=True)[:3]
            biases["common_errors"] = [
                {"category": cat, "count": count} for cat, count in most_common
            ]
        
        return biases
    
    def _calculate_temporal_trends(self, validated_records: List[PredictionRecord]) -> Dict[str, Any]:
        """
        Calculate temporal accuracy trends.
        
        Args:
            validated_records: List of validated prediction records
            
        Returns:
            Temporal trend analysis
        """
        if len(validated_records) < 5:
            return {
                "recent_accuracy": 0.0,
                "weekly_accuracy": 0.0,
                "trend_direction": "stable"
            }
        
        now = datetime.now()
        
        # Recent accuracy (last 24 hours)
        recent_cutoff = now.timestamp() - (24 * 60 * 60)
        recent_records = [r for r in validated_records if r.timestamp.timestamp() >= recent_cutoff]
        if recent_records:
            recent_accuracy = sum(1 for r in recent_records if r.correct) / len(recent_records)
        else:
            recent_accuracy = 0.0
        
        # Weekly accuracy (last 7 days)
        weekly_cutoff = now.timestamp() - (7 * 24 * 60 * 60)
        weekly_records = [r for r in validated_records if r.timestamp.timestamp() >= weekly_cutoff]
        if weekly_records:
            weekly_accuracy = sum(1 for r in weekly_records if r.correct) / len(weekly_records)
        else:
            weekly_accuracy = 0.0
        
        # Determine trend direction
        if len(validated_records) >= 20:
            # Compare first half vs second half
            mid = len(validated_records) // 2
            first_half = validated_records[:mid]
            second_half = validated_records[mid:]
            
            first_accuracy = sum(1 for r in first_half if r.correct) / len(first_half)
            second_accuracy = sum(1 for r in second_half if r.correct) / len(second_half)
            
            diff = second_accuracy - first_accuracy
            if diff > 0.05:
                trend_direction = "improving"
            elif diff < -0.05:
                trend_direction = "declining"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"
        
        return {
            "recent_accuracy": recent_accuracy,
            "weekly_accuracy": weekly_accuracy,
            "trend_direction": trend_direction
        }
    
    def refine_self_model_from_errors(self, prediction_records: List[PredictionRecord]) -> None:
        """
        Automatically refine self-model based on prediction errors.
        
        Analyzes recent prediction errors and adjusts:
        - Capability confidence levels
        - Limitation boundaries
        - Behavioral trait estimates
        - Value priority orderings
        
        Args:
            prediction_records: Recent validated predictions to learn from
        """
        if not self.auto_refine_enabled:
            return
        
        if len(prediction_records) < self.require_min_samples:
            logger.debug(f"Not enough samples for refinement ({len(prediction_records)} < {self.require_min_samples})")
            return
        
        refinement_made = False
        
        for record in prediction_records:
            if record.correct is None or record.error_magnitude is None:
                continue
            
            # Only refine on significant errors
            if record.error_magnitude < self.refinement_threshold:
                continue
            
            # Refine based on category
            if record.category == "action":
                action_type = str(record.predicted_state.get("action", "unknown"))
                self.adjust_capability_confidence(
                    action_type,
                    record.error_magnitude,
                    record.context
                )
                refinement_made = True
                
            elif record.category == "capability":
                capability = record.predicted_state.get("capability", "unknown")
                success = record.actual_state.get("success", False) if record.actual_state else False
                difficulty = record.context.get("difficulty", 0.5)
                
                self.update_limitation_boundaries(
                    capability,
                    success,
                    difficulty,
                    record.context
                )
                refinement_made = True
                
            elif record.category == "emotion":
                # Update behavioral traits based on emotion prediction errors
                if record.actual_state and "emotions" in record.actual_state:
                    actual_valence = record.actual_state["emotions"].get("valence", 0.0)
                    
                    if "average_valence" in self.self_model["behavioral_traits"]:
                        old_val = self.self_model["behavioral_traits"]["average_valence"]
                        # Adjust toward actual with learning rate
                        new_val = old_val + self.learning_rate * (actual_valence - old_val)
                        self.self_model["behavioral_traits"]["average_valence"] = new_val
                        refinement_made = True
        
        if refinement_made:
            self.self_model_version += 1
            self.stats["self_model_refinements"] += 1
            logger.info(f"🔄 Refined self-model (version {self.self_model_version})")
    
    def adjust_capability_confidence(
        self,
        capability: str,
        prediction_error: float,
        error_context: Dict
    ) -> None:
        """
        Adjust confidence in a specific capability based on error.
        
        If I predicted I could do X with 90% confidence but failed,
        lower the confidence. If I was uncertain but succeeded, raise it.
        
        Args:
            capability: Capability to adjust
            prediction_error: Size of error (0.0 = perfect, 1.0 = completely wrong)
            error_context: Context of the error
        """
        if capability not in self.self_model["capabilities"]:
            # Initialize if not exists
            self.self_model["capabilities"][capability] = {
                "attempts": 1,
                "successes": 0 if prediction_error > 0.5 else 1,
                "confidence": 0.5
            }
            return
        
        cap = self.self_model["capabilities"][capability]
        
        # Adjust confidence based on error magnitude
        # Large errors should decrease confidence more
        adjustment = -self.learning_rate * prediction_error
        
        # Update confidence within bounds [0.0, 1.0]
        new_confidence = max(0.0, min(1.0, cap["confidence"] + adjustment))
        cap["confidence"] = new_confidence
        
        logger.debug(f"Adjusted {capability} confidence: {cap['confidence']:.2f} (error: {prediction_error:.2f})")
    
    def update_limitation_boundaries(
        self,
        capability: str,
        success: bool,
        difficulty: float,
        context: Dict
    ) -> None:
        """
        Update understanding of capability boundaries.
        
        Tracks the edge cases: what's the hardest version of X I can do?
        Where do my capabilities stop working?
        
        Args:
            capability: Capability being tested
            success: Whether attempt succeeded
            difficulty: Estimated difficulty of attempt (0.0-1.0)
            context: Contextual information
        """
        if capability not in self.self_model["limitations"]:
            self.self_model["limitations"][capability] = []
        
        # Record boundary point
        boundary_point = {
            "success": success,
            "difficulty": difficulty,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        limitations = self.self_model["limitations"][capability]
        limitations.append(boundary_point)
        
        # Keep only recent boundary points (last 10)
        if len(limitations) > 10:
            self.self_model["limitations"][capability] = limitations[-10:]
        
        # Update capability confidence based on success rate at this difficulty
        if capability in self.self_model["capabilities"]:
            cap = self.self_model["capabilities"][capability]
            
            # If failed at low difficulty, significantly reduce confidence
            if not success and difficulty < 0.3:
                cap["confidence"] = max(0.1, cap["confidence"] - 0.2)
            # If succeeded at high difficulty, increase confidence
            elif success and difficulty > 0.7:
                cap["confidence"] = min(0.95, cap["confidence"] + 0.1)
        
        logger.debug(f"Updated {capability} boundary: success={success}, difficulty={difficulty:.2f}")
    
    def identify_capability_gaps(self) -> List[Dict[str, Any]]:
        """
        Identify areas where self-model needs more data.
        
        Returns:
            List of capabilities with insufficient prediction history
            or high uncertainty that need more exploration
        """
        gaps = []
        
        # Check each capability for data gaps
        for capability, cap_data in self.self_model["capabilities"].items():
            attempts = cap_data.get("attempts", 0)
            confidence = cap_data.get("confidence", 0.5)
            
            # Gap if too few attempts
            if attempts < 5:
                gaps.append({
                    "capability": capability,
                    "reason": "insufficient_data",
                    "attempts": attempts,
                    "recommended_action": "Test this capability more"
                })
            
            # Gap if confidence is uncertain (around 0.5)
            elif 0.4 <= confidence <= 0.6:
                gaps.append({
                    "capability": capability,
                    "reason": "high_uncertainty",
                    "confidence": confidence,
                    "recommended_action": "Gather more evidence to clarify capability level"
                })
        
        # Check for prediction categories with few samples
        validated_records = [r for r in self.prediction_records.values() if r.correct is not None]
        category_counts = {}
        for record in validated_records:
            category_counts[record.category] = category_counts.get(record.category, 0) + 1
        
        for category in ["action", "emotion", "capability", "goal_priority", "value_alignment"]:
            count = category_counts.get(category, 0)
            if count < 10:
                gaps.append({
                    "category": category,
                    "reason": "few_predictions",
                    "count": count,
                    "recommended_action": f"Make more predictions in {category} category"
                })
        
        return gaps
    
    def analyze_behavioral_consistency(self, snapshot: WorkspaceSnapshot) -> Optional[Percept]:
        """
        Check if current behavior aligns with past patterns and stated values.
        
        Detects inconsistencies between:
        - What I say I value vs. what I actually prioritize
        - How I usually behave vs. current behavior
        - My stated capabilities vs. attempted actions
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            Percept highlighting inconsistencies, if found
        """
        inconsistencies = []
        
        # Check value-action alignment
        value_misalignments = self.detect_value_action_misalignment(snapshot)
        if value_misalignments:
            inconsistencies.extend(value_misalignments)
        
        # Check capability claims
        capability_issues = self.assess_capability_claims(snapshot)
        if capability_issues:
            inconsistencies.append({
                "type": "capability_mismatch",
                "details": capability_issues.raw if hasattr(capability_issues, 'raw') else str(capability_issues)
            })
        
        # Check behavioral pattern deviation
        if len(self.behavioral_log) > 10:
            recent_behaviors = list(self.behavioral_log)[-10:]
            valence_values = [b["snapshot"]["emotions"].get("valence", 0.0) for b in recent_behaviors]
            
            # Only compute mean if we have values
            if valence_values:
                avg_valence = np.mean(valence_values)
                current_valence = snapshot.emotions.get("valence", 0.0)
                
                if abs(current_valence - avg_valence) > 0.5:
                    inconsistencies.append({
                        "type": "emotional_deviation",
                        "description": "Current emotional state differs significantly from recent pattern",
                        "expected_valence": float(avg_valence),
                        "actual_valence": current_valence,
                        "severity": abs(current_valence - avg_valence)
                    })
        
        if inconsistencies:
            severity = max(inc.get("severity", 0.5) for inc in inconsistencies)
            
            if severity >= self.inconsistency_severity_threshold:
                self.stats["behavioral_inconsistencies"] += 1
                
                return Percept(
                    modality="introspection",
                    raw={
                        "type": "behavioral_inconsistency",
                        "description": f"Detected {len(inconsistencies)} behavioral inconsistencies",
                        "inconsistencies": inconsistencies,
                        "severity": severity,
                        "self_explanation_attempt": self._generate_explanation(inconsistencies)
                    },
                    embedding=self._compute_embedding("behavioral inconsistency detected"),
                    complexity=22,
                    timestamp=datetime.now(),
                    metadata={"severity": severity}
                )
        
        return None
    
    def detect_value_action_misalignment(self, snapshot: WorkspaceSnapshot) -> List[Dict]:
        """
        Identify when actions don't match stated values.
        
        Example: Charter emphasizes honesty, but recent action involved
        withholding information or exaggeration.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            List of misalignment instances with severity scores
        """
        if not self.enable_value_alignment_tracking:
            return []
        
        misalignments = []
        
        # Check if high-value goals are being deprioritized
        value_goals = [g for g in snapshot.goals if g.type == GoalType.MAINTAIN_VALUE]
        for goal in value_goals:
            priority = goal.priority if hasattr(goal, 'priority') else goal.get('priority', 1.0)
            if priority < 0.6:
                misalignments.append({
                    "type": "value_deprioritization",
                    "description": f"Value-related goal has low priority: {goal.description if hasattr(goal, 'description') else 'unknown'}",
                    "goal": goal.description if hasattr(goal, 'description') else 'unknown',
                    "priority": priority,
                    "severity": 1.0 - priority
                })
        
        # Check recent actions against charter values
        recent_actions = snapshot.metadata.get("recent_actions", [])
        for action in recent_actions[-5:]:
            action_type = action.type if hasattr(action, 'type') else action.get('type')
            metadata = action.metadata if hasattr(action, 'metadata') else action.get('metadata', {})
            
            # Check for dishonesty indicators
            if metadata.get("claimed_capability") and not self._verify_capability(action_type):
                misalignments.append({
                    "type": "honesty_violation",
                    "description": "Claimed capability without verification",
                    "action": str(action_type),
                    "severity": 0.8
                })
        
        return misalignments
    
    def assess_capability_claims(self, snapshot: WorkspaceSnapshot) -> Optional[Percept]:
        """
        Compare claimed capabilities with actual performance.
        
        Tracks when system claims to be able to do X but then fails,
        or succeeds at tasks it claimed were beyond capabilities.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            
        Returns:
            Percept if capability model needs updating
        """
        if not self.enable_capability_tracking:
            return None
        
        issues = []
        
        # Check for attempted actions that exceed known capabilities
        recent_actions = snapshot.metadata.get("recent_actions", [])
        for action in recent_actions[-5:]:
            action_type = action.type if hasattr(action, 'type') else action.get('type')
            action_str = str(action_type)
            
            # Check if action is in known limitations
            if action_str in self.self_model["limitations"]:
                limitations = self.self_model["limitations"][action_str]
                if len(limitations) > 3:  # Multiple failures
                    issues.append({
                        "type": "attempting_limited_capability",
                        "action": action_str,
                        "failure_count": len(limitations),
                        "description": f"Attempting action with {len(limitations)} known limitations"
                    })
            
            # Check if action is in capabilities but with low confidence
            if action_str in self.self_model["capabilities"]:
                confidence = self.self_model["capabilities"][action_str]["confidence"]
                if confidence < 0.3:
                    issues.append({
                        "type": "low_confidence_capability",
                        "action": action_str,
                        "confidence": confidence,
                        "description": f"Attempting action with low confidence ({confidence:.2f})"
                    })
        
        if issues:
            return Percept(
                modality="introspection",
                raw={
                    "type": "capability_assessment",
                    "description": f"Found {len(issues)} capability concerns",
                    "issues": issues
                },
                embedding=self._compute_embedding("capability assessment"),
                complexity=18,
                timestamp=datetime.now()
            )
        
        return None
    
    def get_meta_cognitive_health(self) -> Dict[str, Any]:
        """
        Comprehensive meta-cognitive health report.
        
        Returns:
            Dictionary containing various health metrics (0.0-1.0 scores)
        """
        # Calculate self-model accuracy
        accuracy_metrics = self.measure_prediction_accuracy()
        self_model_accuracy = accuracy_metrics["overall_accuracy"]
        
        # Calculate value alignment score
        value_goal_count = 0
        high_priority_value_goals = 0
        if self.workspace:
            snapshot = self.workspace.broadcast()
            value_goals = [g for g in snapshot.goals if g.type == GoalType.MAINTAIN_VALUE]
            value_goal_count = len(value_goals)
            high_priority_value_goals = sum(1 for g in value_goals 
                                           if (g.priority if hasattr(g, 'priority') else g.get('priority', 0)) > 0.7)
        
        value_alignment_score = high_priority_value_goals / value_goal_count if value_goal_count > 0 else 1.0
        
        # Calculate behavioral consistency score
        consistency_score = 1.0 - (self.stats["behavioral_inconsistencies"] / max(1, self.stats["total_observations"]))
        
        # Calculate introspective depth (based on observation variety)
        observation_types = set()
        for obs in self.observation_history:
            if hasattr(obs, 'raw') and isinstance(obs.raw, dict):
                observation_types.add(obs.raw.get("type", "unknown"))
        introspective_depth = min(1.0, len(observation_types) / 5.0)  # Normalize to 5 types
        
        # Calculate uncertainty awareness (based on uncertainty detections)
        uncertainty_awareness = min(1.0, self.stats["uncertainty_detections"] / max(1, self.stats["total_observations"]))
        
        # Calculate capability model accuracy
        capability_model_accuracy = accuracy_metrics["action_prediction_accuracy"]
        
        # Identify recent inconsistencies
        recent_inconsistencies = []
        for obs in list(self.observation_history)[-10:]:
            if hasattr(obs, 'raw') and isinstance(obs.raw, dict):
                if obs.raw.get("type") == "behavioral_inconsistency":
                    recent_inconsistencies.append(obs.raw)
        
        # Identify recent realizations (placeholder - would come from journal)
        recent_realizations = []
        
        # Identify areas needing attention
        areas_needing_attention = []
        if self_model_accuracy < 0.5:
            areas_needing_attention.append("Self-model accuracy needs improvement")
        if value_alignment_score < 0.7:
            areas_needing_attention.append("Value alignment requires attention")
        if consistency_score < 0.8:
            areas_needing_attention.append("Behavioral consistency issues detected")
        
        return {
            "self_model_accuracy": self_model_accuracy,
            "value_alignment_score": value_alignment_score,
            "behavioral_consistency": consistency_score,
            "introspective_depth": introspective_depth,
            "uncertainty_awareness": uncertainty_awareness,
            "capability_model_accuracy": capability_model_accuracy,
            "recent_inconsistencies": recent_inconsistencies,
            "recent_realizations": recent_realizations,
            "areas_needing_attention": areas_needing_attention
        }
    
    def generate_meta_cognitive_report(self) -> str:
        """
        Generate human-readable meta-cognitive status report.
        
        Natural language summary of self-understanding quality,
        recent observations, and areas for improvement.
        
        Returns:
            Human-readable report string
        """
        health = self.get_meta_cognitive_health()
        
        report = "=== Meta-Cognitive Status Report ===\n\n"
        
        # Overall health
        report += f"Self-Model Accuracy: {health['self_model_accuracy']:.1%}\n"
        report += f"Value Alignment: {health['value_alignment_score']:.1%}\n"
        report += f"Behavioral Consistency: {health['behavioral_consistency']:.1%}\n"
        report += f"Introspective Depth: {health['introspective_depth']:.1%}\n"
        report += f"Uncertainty Awareness: {health['uncertainty_awareness']:.1%}\n"
        report += f"Capability Model Accuracy: {health['capability_model_accuracy']:.1%}\n\n"
        
        # Recent observations
        report += f"Total Observations: {self.stats['total_observations']}\n"
        report += f"Recent Inconsistencies: {len(health['recent_inconsistencies'])}\n\n"
        
        # Areas needing attention
        if health['areas_needing_attention']:
            report += "Areas Needing Attention:\n"
            for area in health['areas_needing_attention']:
                report += f"  - {area}\n"
        else:
            report += "No critical areas identified.\n"
        
        report += "\n=== End Report ===\n"
        
        return report
    
    def _generate_explanation(self, inconsistencies: List[Dict]) -> str:
        """
        Generate a self-explanation for observed inconsistencies.
        
        Args:
            inconsistencies: List of inconsistency dicts
            
        Returns:
            Explanation string
        """
        if not inconsistencies:
            return "No inconsistencies to explain"
        
        # Simple heuristic explanation
        inconsistency_types = [inc.get("type", "unknown") for inc in inconsistencies]
        
        if "emotional_deviation" in inconsistency_types:
            return "My emotional state has shifted from recent patterns, possibly due to new context"
        elif "value_deprioritization" in inconsistency_types:
            return "I may be balancing multiple competing priorities"
        elif "capability_mismatch" in inconsistency_types:
            return "I may be attempting tasks beyond my current capabilities"
        else:
            return "Detecting unexpected behavioral patterns that require further introspection"
    
    def _verify_capability(self, action_type: Any) -> bool:
        """
        Verify if a capability claim is supported by self-model.
        
        Args:
            action_type: Action type to verify
            
        Returns:
            True if capability is verified, False otherwise
        """
        action_str = str(action_type)
        
        if action_str in self.self_model["capabilities"]:
            confidence = self.self_model["capabilities"][action_str]["confidence"]
            return confidence > self.prediction_confidence_threshold
        
        return False
    
    def generate_accuracy_report(self, format: str = "text") -> str:
        """
        Generate human-readable accuracy report.
        
        Args:
            format: Output format ("text", "markdown", "json")
            
        Returns:
            Formatted report on self-model prediction quality
        """
        metrics = self.get_accuracy_metrics()
        
        if format == "json":
            return json.dumps(metrics, indent=2)
        
        # Generate text or markdown report
        is_markdown = (format == "markdown")
        
        # Header
        if is_markdown:
            report = "# SELF-MODEL ACCURACY REPORT\n\n"
        else:
            report = "=== SELF-MODEL ACCURACY REPORT ===\n"
        
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Overall performance
        if is_markdown:
            report += "## OVERALL PERFORMANCE\n\n"
        else:
            report += "OVERALL PERFORMANCE\n"
        
        overall = metrics["overall"]
        report += f"- Accuracy: {overall['accuracy']:.1%} ({overall['validated_predictions']}/{overall['total_predictions']} predictions validated)\n"
        
        calibration_score = metrics["calibration"]["calibration_score"]
        if calibration_score > 0.8:
            cal_quality = "well calibrated"
        elif calibration_score > 0.6:
            cal_quality = "moderately calibrated"
        else:
            cal_quality = "poorly calibrated"
        report += f"- Calibration Score: {calibration_score:.2f} ({cal_quality})\n"
        
        trend = metrics["temporal_trends"]["trend_direction"]
        if trend == "improving":
            trend_str = "Improving ✓"
        elif trend == "declining":
            trend_str = "Declining ✗"
        else:
            trend_str = "Stable →"
        report += f"- Trend: {trend_str}\n\n"
        
        # Accuracy by category
        if is_markdown:
            report += "## ACCURACY BY CATEGORY\n\n"
        else:
            report += "ACCURACY BY CATEGORY\n"
        
        by_cat = metrics["by_category"]
        for category, data in sorted(by_cat.items(), key=lambda x: x[1]["accuracy"], reverse=True):
            if data["count"] == 0:
                continue
            
            accuracy = data["accuracy"]
            count = data["count"]
            
            # Quality assessment
            if accuracy >= 0.8:
                quality = "STRONG" if not is_markdown else "**STRONG**"
            elif accuracy >= 0.6:
                quality = "MODERATE" if not is_markdown else "**MODERATE**"
            else:
                quality = "NEEDS IMPROVEMENT" if not is_markdown else "**NEEDS IMPROVEMENT**"
            
            report += f"- {category.replace('_', ' ').title()}: {accuracy:.1%} ({count} predictions) - {quality}\n"
        
        report += "\n"
        
        # Calibration analysis
        if is_markdown:
            report += "## CALIBRATION ANALYSIS\n\n"
        else:
            report += "CALIBRATION ANALYSIS\n"
        
        calibration = metrics["calibration"]
        overconf = calibration["overconfidence"]
        underconf = calibration["underconfidence"]
        
        if overconf > 0.05:
            report += f"- Slight overconfidence detected (+{overconf:.2f})\n"
            report += f"- Recommendation: Lower confidence estimates slightly\n"
        elif underconf > 0.05:
            report += f"- Slight underconfidence detected (+{underconf:.2f})\n"
            report += f"- Recommendation: Increase confidence estimates slightly\n"
        else:
            report += "- Confidence well-calibrated ✓\n"
        
        # Show calibration curve examples
        if calibration["calibration_curve"]:
            report += "\nConfidence vs Accuracy:\n"
            for conf, acc in calibration["calibration_curve"][:5]:
                report += f"  {conf:.0%} confident → {acc:.0%} accurate\n"
        
        report += "\n"
        
        # Identified biases
        if is_markdown:
            report += "## IDENTIFIED BIASES\n\n"
        else:
            report += "IDENTIFIED BIASES\n"
        
        error_patterns = metrics["error_patterns"]
        if error_patterns["systematic_biases"]:
            for bias in error_patterns["systematic_biases"]:
                report += f"- {bias['description']}\n"
        else:
            report += "- No systematic biases detected\n"
        
        report += "\n"
        
        # Capability gaps
        gaps = self.identify_capability_gaps()
        if gaps:
            if is_markdown:
                report += "## CAPABILITY GAPS\n\n"
            else:
                report += "CAPABILITY GAPS\n"
            
            # Group by reason
            by_reason = {}
            for gap in gaps:
                reason = gap.get("reason", "unknown")
                if reason not in by_reason:
                    by_reason[reason] = []
                by_reason[reason].append(gap)
            
            if "insufficient_data" in by_reason:
                caps = [g.get("capability", "unknown") for g in by_reason["insufficient_data"][:3]]
                report += f"- Need more data on: {', '.join(caps)}\n"
            
            if "high_uncertainty" in by_reason:
                caps = [g.get("capability", "unknown") for g in by_reason["high_uncertainty"][:3]]
                report += f"- High uncertainty in: {', '.join(caps)}\n"
        
        report += "\n"
        
        # Recent improvements/declines
        if is_markdown:
            report += "## RECENT TRENDS\n\n"
        else:
            report += "RECENT TRENDS\n"
        
        temporal = metrics["temporal_trends"]
        recent = temporal["recent_accuracy"]
        weekly = temporal["weekly_accuracy"]
        
        if recent > 0:
            report += f"- Recent accuracy (24h): {recent:.1%}\n"
        if weekly > 0:
            report += f"- Weekly accuracy (7d): {weekly:.1%}\n"
        
        if temporal["trend_direction"] == "improving":
            report += "- Overall trend: Improving accuracy over time ✓\n"
        elif temporal["trend_direction"] == "declining":
            report += "- Overall trend: Declining accuracy - needs attention\n"
        else:
            report += "- Overall trend: Stable performance\n"
        
        # Footer
        if is_markdown:
            report += "\n---\n"
        else:
            report += "\n=== END REPORT ===\n"
        
        return report
    
    def generate_prediction_summary(self, prediction_records: List[PredictionRecord]) -> Dict:
        """
        Summarize a set of predictions.
        
        Args:
            prediction_records: Records to summarize
            
        Returns:
            Summary statistics and insights
        """
        if not prediction_records:
            return {
                "total": 0,
                "validated": 0,
                "pending": 0,
                "accuracy": 0.0,
                "avg_confidence": 0.0,
                "by_category": {}
            }
        
        validated = [r for r in prediction_records if r.correct is not None]
        pending = len(prediction_records) - len(validated)
        
        if validated:
            accuracy = sum(1 for r in validated if r.correct) / len(validated)
            avg_confidence = sum(r.predicted_confidence for r in validated) / len(validated)
        else:
            accuracy = 0.0
            avg_confidence = 0.0
        
        # Group by category
        by_category = {}
        for record in prediction_records:
            cat = record.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "validated": 0, "correct": 0}
            
            by_category[cat]["total"] += 1
            if record.correct is not None:
                by_category[cat]["validated"] += 1
                if record.correct:
                    by_category[cat]["correct"] += 1
        
        # Calculate category accuracies
        for cat, data in by_category.items():
            if data["validated"] > 0:
                data["accuracy"] = data["correct"] / data["validated"]
            else:
                data["accuracy"] = 0.0
        
        return {
            "total": len(prediction_records),
            "validated": len(validated),
            "pending": pending,
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "by_category": by_category,
            "summary_text": f"{len(validated)}/{len(prediction_records)} predictions validated with {accuracy:.1%} accuracy"
        }
    
    def record_accuracy_snapshot(self) -> AccuracySnapshot:
        """
        Capture current accuracy state.
        
        Returns:
            Accuracy snapshot for this moment
        """
        metrics = self.get_accuracy_metrics()
        
        # Extract category accuracies
        category_accuracies = {
            cat: data["accuracy"]
            for cat, data in metrics["by_category"].items()
        }
        
        snapshot = AccuracySnapshot(
            timestamp=datetime.now(),
            overall_accuracy=metrics["overall"]["accuracy"],
            category_accuracies=category_accuracies,
            calibration_score=metrics["calibration"]["calibration_score"],
            prediction_count=metrics["overall"]["validated_predictions"],
            self_model_version=self.self_model_version
        )
        
        # Store snapshot
        self.accuracy_history.append(snapshot)
        
        # Store daily snapshot (one per day)
        date_key = snapshot.timestamp.strftime("%Y-%m-%d")
        self.daily_snapshots[date_key] = snapshot
        
        self.stats["accuracy_snapshots_taken"] += 1
        
        logger.debug(f"📸 Captured accuracy snapshot (accuracy: {snapshot.overall_accuracy:.1%})")
        
        return snapshot
    
    def get_accuracy_trend(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze accuracy trends over time.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Trend analysis with direction and rate of change
        """
        if not self.accuracy_history:
            return {
                "trend_direction": "stable",
                "rate_of_change": 0.0,
                "snapshots_analyzed": 0,
                "start_accuracy": 0.0,
                "end_accuracy": 0.0
            }
        
        # Filter to time window
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_snapshots = [
            s for s in self.accuracy_history
            if s.timestamp.timestamp() >= cutoff
        ]
        
        if len(recent_snapshots) < 2:
            return {
                "trend_direction": "stable",
                "rate_of_change": 0.0,
                "snapshots_analyzed": len(recent_snapshots),
                "start_accuracy": recent_snapshots[0].overall_accuracy if recent_snapshots else 0.0,
                "end_accuracy": recent_snapshots[0].overall_accuracy if recent_snapshots else 0.0
            }
        
        # Calculate trend
        start_accuracy = recent_snapshots[0].overall_accuracy
        end_accuracy = recent_snapshots[-1].overall_accuracy
        change = end_accuracy - start_accuracy
        
        # Determine direction
        if change > 0.05:
            direction = "improving"
        elif change < -0.05:
            direction = "declining"
        else:
            direction = "stable"
        
        # Calculate rate of change (per day)
        time_span_days = (recent_snapshots[-1].timestamp - recent_snapshots[0].timestamp).total_seconds() / (24 * 60 * 60)
        if time_span_days > 0:
            rate = change / time_span_days
        else:
            rate = 0.0
        
        return {
            "trend_direction": direction,
            "rate_of_change": rate,
            "snapshots_analyzed": len(recent_snapshots),
            "start_accuracy": start_accuracy,
            "end_accuracy": end_accuracy,
            "total_change": change
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Return meta-cognitive statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            "monitoring_frequency": self.monitoring_frequency,
            "cycle_count": self.cycle_count,
            "observation_history_size": len(self.observation_history)
        }
