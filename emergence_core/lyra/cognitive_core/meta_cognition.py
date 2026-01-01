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
from typing import Optional, Dict, Any, List
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np

from .workspace import GlobalWorkspace, WorkspaceSnapshot, Percept, GoalType

logger = logging.getLogger(__name__)


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

    def __init__(self, workspace: Optional[GlobalWorkspace] = None, config: Optional[Dict] = None):
        """
        Initialize the self-monitor.

        Args:
            workspace: GlobalWorkspace instance to observe
            config: Optional configuration dict
        """
        self.workspace = workspace
        self.config = config or {}
        
        # Load identity files
        self.charter_text = self._load_charter()
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
        
        # Configuration for new features
        self.self_model_update_frequency = self.config.get("self_model_update_frequency", 5)
        self.prediction_confidence_threshold = self.config.get("prediction_confidence_threshold", 0.6)
        self.inconsistency_severity_threshold = self.config.get("inconsistency_severity_threshold", 0.5)
        self.enable_existential_questions = self.config.get("enable_existential_questions", True)
        self.enable_capability_tracking = self.config.get("enable_capability_tracking", True)
        self.enable_value_alignment_tracking = self.config.get("enable_value_alignment_tracking", True)
        
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
            "behavioral_inconsistencies": 0
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
        
        if conflicts:
            return Percept(
                modality="introspection",
                raw={
                    "type": "value_conflict",
                    "description": f"Detected {len(conflicts)} potential value conflicts",
                    "conflicts": conflicts,
                    "charter_excerpt": self._relevant_charter_section(conflicts)
                },
                embedding=self._compute_embedding("value conflict detected"),
                complexity=25,
                timestamp=datetime.now(),
                metadata={"severity": max(c.get("severity", 0.5) for c in conflicts)}
            )
        
        return None
    
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
