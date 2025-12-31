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
from typing import Optional, Dict, Any, List
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np

from .workspace import GlobalWorkspace, WorkspaceSnapshot, Percept, GoalType

logger = logging.getLogger(__name__)


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
        
        # Stats
        self.stats = {
            "total_observations": 0,
            "value_conflicts": 0,
            "performance_issues": 0,
            "uncertainty_detections": 0,
            "emotional_observations": 0,
            "pattern_detections": 0
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
            if goal_type and str(goal_type) == "maintain_value":
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
        
        for text in goal_texts:
            for word1, word2 in conflict_pairs:
                if any(word1 in t and word2 in other 
                       for t in goal_texts for other in goal_texts if t != other):
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
