"""
Monitor: Self-monitoring and state observation for meta-cognition.

This module handles observing internal cognitive states and generating
introspective percepts about the system's processing.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np

from ..workspace import GlobalWorkspace, WorkspaceSnapshot, Percept, GoalType

logger = logging.getLogger(__name__)


class Monitor:
    """
    Observes and reports on internal cognitive state.
    
    Generates introspective percepts by monitoring:
    - Value alignment with charter
    - Performance and efficiency
    - Uncertainty and ambiguity
    - Emotional states
    - Behavioral patterns
    """
    
    def __init__(
        self,
        workspace: Optional[GlobalWorkspace] = None,
        charter_text: str = "",
        protocols_text: str = "",
        identity: Optional[Any] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize the monitor.
        
        Args:
            workspace: GlobalWorkspace to observe
            charter_text: Constitutional charter text
            protocols_text: Behavioral protocols text
            identity: Optional IdentityLoader with charter/protocols
            config: Optional configuration dict
        """
        self.workspace = workspace
        self.charter_text = charter_text
        self.protocols_text = protocols_text
        self.identity = identity
        self.config = config or {}
        
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
        from ..action import ActionType
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
    
    @staticmethod
    def load_charter(charter_path: Optional[Path] = None) -> str:
        """
        Load charter from identity files.
        
        Args:
            charter_path: Optional path to charter file
        
        Returns:
            Charter text content
        """
        path = charter_path or Path("data/identity/charter.md")
        if path.exists():
            return path.read_text()
        logger.warning(f"Charter file not found at {path}")
        return ""
    
    @staticmethod
    def load_protocols(protocols_path: Optional[Path] = None) -> str:
        """
        Load protocols from identity files.
        
        Args:
            protocols_path: Optional path to protocols file
        
        Returns:
            Protocols text content
        """
        path = protocols_path or Path("data/identity/protocols.md")
        if path.exists():
            return path.read_text()
        logger.warning(f"Protocols file not found at {path}")
        return ""
    
    @staticmethod
    def format_protocols(protocols: List) -> str:
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
