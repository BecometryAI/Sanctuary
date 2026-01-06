"""
Confidence Estimator: Prediction tracking and confidence calibration.

This module handles making predictions about own behavior, tracking prediction
accuracy, and calculating confidence calibration metrics.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional, Dict, Any, List
from collections import deque
from datetime import datetime

from ._shared import PredictionRecord, AccuracySnapshot

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """
    Estimates confidence levels and tracks prediction accuracy.
    
    Handles:
    - Behavioral predictions
    - Prediction validation and tracking
    - Confidence calibration
    - Systematic bias detection
    """
    
    def __init__(
        self,
        self_model: Dict[str, Any],
        prediction_history: Any,
        config: Optional[Dict] = None
    ):
        """
        Initialize confidence estimator.
        
        Args:
            self_model: Reference to self-model data
            prediction_history: Reference to prediction history
            config: Optional configuration dict
        """
        self.self_model = self_model
        self.prediction_history = prediction_history
        self.config = config or {}
        
        # Prediction tracking
        self.prediction_records: Dict[str, PredictionRecord] = {}
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
        
        # Configuration
        self.prediction_confidence_threshold = self.config.get("prediction_confidence_threshold", 0.6)
        self.prediction_tracking_enabled = self.config.get("prediction_tracking", {}).get("enabled", True)
        self.max_pending_validations = self.config.get("prediction_tracking", {}).get("max_pending_validations", 100)
        self.auto_validate_enabled = self.config.get("prediction_tracking", {}).get("auto_validate", True)
        self.validation_timeout = self.config.get("prediction_tracking", {}).get("validation_timeout", 600)
        
        self.stats = {
            "predictions_made": 0,
            "predictions_validated": 0
        }
    
    def predict_behavior(self, hypothetical_state: Any) -> Dict[str, Any]:
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
    
    def auto_validate_predictions(self, snapshot: Any) -> List[PredictionRecord]:
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
