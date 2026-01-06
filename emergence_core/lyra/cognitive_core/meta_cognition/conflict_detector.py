"""
Conflict Detector: Identifies conflicts and inconsistencies in behavior.

This module detects behavioral inconsistencies, value-action misalignments,
and capability assessment issues.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

import numpy as np

from ..workspace import WorkspaceSnapshot, Percept, GoalType
from datetime import datetime

logger = logging.getLogger(__name__)


class ConflictDetector:
    """
    Detects processing conflicts and behavioral inconsistencies.
    
    Identifies:
    - Value-action misalignments
    - Behavioral inconsistencies
    - Capability claims vs. actual performance
    """
    
    def __init__(
        self,
        self_model: Dict[str, Any],
        behavioral_log: Any,
        config: Optional[Dict] = None
    ):
        """
        Initialize conflict detector.
        
        Args:
            self_model: Reference to self-model data
            behavioral_log: Reference to behavioral log
            config: Optional configuration dict
        """
        self.self_model = self_model
        self.behavioral_log = behavioral_log
        self.config = config or {}
        
        self.enable_value_alignment_tracking = self.config.get("enable_value_alignment_tracking", True)
        self.enable_capability_tracking = self.config.get("enable_capability_tracking", True)
        self.inconsistency_severity_threshold = self.config.get("inconsistency_severity_threshold", 0.5)
        self.prediction_confidence_threshold = self.config.get("prediction_confidence_threshold", 0.6)
        
        self.stats = {
            "behavioral_inconsistencies": 0
        }
    
    def analyze_behavioral_consistency(
        self,
        snapshot: WorkspaceSnapshot,
        compute_embedding_fn
    ) -> Optional[Percept]:
        """
        Check if current behavior aligns with past patterns and stated values.
        
        Detects inconsistencies between:
        - What I say I value vs. what I actually prioritize
        - How I usually behave vs. current behavior
        - My stated capabilities vs. attempted actions
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            compute_embedding_fn: Function to compute embeddings
            
        Returns:
            Percept highlighting inconsistencies, if found
        """
        inconsistencies = []
        
        # Check value-action alignment
        value_misalignments = self.detect_value_action_misalignment(snapshot)
        if value_misalignments:
            inconsistencies.extend(value_misalignments)
        
        # Check capability claims
        capability_issues = self.assess_capability_claims(snapshot, compute_embedding_fn)
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
                    embedding=compute_embedding_fn("behavioral inconsistency detected"),
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
    
    def assess_capability_claims(
        self,
        snapshot: WorkspaceSnapshot,
        compute_embedding_fn
    ) -> Optional[Percept]:
        """
        Compare claimed capabilities with actual performance.
        
        Tracks when system claims to be able to do X but then fails,
        or succeeds at tasks it claimed were beyond capabilities.
        
        Args:
            snapshot: WorkspaceSnapshot containing current state
            compute_embedding_fn: Function to compute embeddings
            
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
                embedding=compute_embedding_fn("capability assessment"),
                complexity=18,
                timestamp=datetime.now()
            )
        
        return None
    
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
