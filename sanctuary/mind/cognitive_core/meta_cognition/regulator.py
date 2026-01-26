"""
Regulator: Self-model adjustment and capability refinement.

This module handles regulatory adjustments to the self-model based on
prediction errors and observed outcomes.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from ._shared import PredictionRecord

logger = logging.getLogger(__name__)


class Regulator:
    """
    Regulates and adjusts self-model based on observations.
    
    Handles:
    - Self-model refinement from prediction errors
    - Capability confidence adjustment
    - Limitation boundary updates
    - Capability gap identification
    """
    
    def __init__(
        self,
        self_model: Dict[str, Any],
        prediction_records: Dict[str, PredictionRecord],
        config: Optional[Dict] = None
    ):
        """
        Initialize regulator.
        
        Args:
            self_model: Reference to self-model data
            prediction_records: Reference to prediction records
            config: Optional configuration dict
        """
        self.self_model = self_model
        self.prediction_records = prediction_records
        self.config = config or {}
        
        # Configuration
        refinement_config = self.config.get("self_model_refinement", {})
        self.auto_refine_enabled = refinement_config.get("auto_refine", True)
        self.refinement_threshold = refinement_config.get("refinement_threshold", 0.3)
        self.learning_rate = refinement_config.get("learning_rate", 0.1)
        self.require_min_samples = refinement_config.get("require_min_samples", 5)
        
        self.self_model_version = 0
        self.stats = {
            "self_model_refinements": 0,
            "self_model_updates": 0
        }
    
    def update_self_model(self, snapshot: Any, actual_outcome: Dict) -> None:
        """
        Update internal self-model based on observed behavior.
        
        Compare predicted behavior with actual outcomes to refine
        understanding of capabilities, limitations, and tendencies.
        
        Args:
            snapshot: WorkspaceSnapshot containing state before action
            actual_outcome: Dictionary containing actual action results
        """
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
        
        logger.debug(f"Updated {capability} boundary: success={success}, difficulty={difficulty:.2f})")
    
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
