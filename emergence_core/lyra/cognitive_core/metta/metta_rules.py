"""
MeTTa Rule Definitions for IWMT.

This module contains MeTTa rules as Python strings for the pilot implementation.
When MeTTa/Hyperon is fully integrated, these will be loaded into the atomspace.
"""

# Communication decision rules in MeTTa syntax
COMMUNICATION_DECISION_RULES = """
; IWMT-style communication decision in MeTTa
; These rules determine SPEAK/SILENCE/DEFER decisions

; Type definitions
(: Prediction Type)
(: WorldModel Type)
(: TimeHorizon Number)
(: predict (-> WorldModel TimeHorizon (List Prediction)))

; Drive and inhibition types
(: drive (-> DriveType Intensity Bool))
(: inhibition (-> InhibitionReason Strength Bool))
(: decision (-> DecisionType Reason Bool))

; High drive, low inhibition -> SPEAK
(= (should-speak ?drive-type ?drive-intensity ?inhibition-strength)
   (if (and (> ?drive-intensity 0.5)
            (< ?inhibition-strength ?drive-intensity))
       (decision SPEAK ?drive-type)
       False))

; High inhibition -> SILENCE
(= (should-silence ?inhibition-reason ?inhibition-strength)
   (if (> ?inhibition-strength 0.6)
       (decision SILENCE ?inhibition-reason)
       False))

; Moderate drive and inhibition -> DEFER
(= (should-defer ?drive-intensity ?inhibition-strength)
   (if (and (> ?drive-intensity 0.3)
            (> ?inhibition-strength 0.3)
            (< ?drive-intensity 0.7))
       (decision DEFER "balanced_pressure")
       False))
"""

# Prediction rules for world modeling
PREDICTION_RULES = """
; IWMT prediction rules in MeTTa

; Type definitions
(: SelfModel Type)
(: EnvironmentModel Type)
(: Context Type)
(: predict-self (-> SelfModel Context Prediction))
(: predict-environment (-> EnvironmentModel Context (List Prediction)))

; Self-prediction: High goal priority -> will pursue goal
(= (predict-self $self $context)
   (let* (($goals (get-goals $context))
          ($top-goal (first $goals))
          ($priority (goal-priority $top-goal)))
     (if (> $priority 0.5)
         (Prediction "will-pursue-goal" $top-goal 0.6)
         (Prediction "will-wait" "no-urgent-goals" 0.4))))

; Environment prediction: Active agents continue behavior
(= (predict-environment $env $context)
   (let* (($agents (get-agents $env))
          ($predictions (map predict-agent-continuation $agents)))
     $predictions))

; Prediction error computation
(= (compute-prediction-error $prediction $actual)
   (let* (($expected (prediction-content $prediction))
          ($confidence (prediction-confidence $prediction))
          ($match (semantic-similarity $expected $actual))
          ($error (- 1.0 $match))
          ($surprise (- (log (- 1.0 $confidence)))))
     (PredictionError $prediction $actual $error $surprise)))
"""

# Free energy minimization rules
FREE_ENERGY_RULES = """
; Free energy computation in MeTTa

; Type definitions
(: free-energy (-> WorldModel Number))
(: expected-free-energy (-> Action WorldModel Number))

; Compute current free energy
(= (free-energy $world-model)
   (let* (($errors (get-prediction-errors $world-model))
          ($avg-error (mean-magnitude $errors))
          ($avg-surprise (mean-surprise $errors))
          ($complexity (model-complexity $world-model)))
     (+ (* 1.0 (/ (+ $avg-error $avg-surprise) 2.0))
        (* 0.1 $complexity))))

; Expected free energy for actions
(= (expected-free-energy $action $world-model)
   (let* (($current-fe (free-energy $world-model))
          ($action-type (action-type $action))
          ($epistemic-gain (epistemic-value $action-type))
          ($pragmatic-gain (pragmatic-value $action-type)))
     (- $current-fe (+ $epistemic-gain $pragmatic-gain))))

; Select action with minimum expected free energy
(= (select-action $actions $world-model)
   (let* (($efes (map (lambda ($a) 
                        (cons $a (expected-free-energy $a $world-model)))
                      $actions))
          ($sorted (sort-by-second $efes)))
     (first (first $sorted))))
"""

# Precision weighting rules
PRECISION_RULES = """
; Precision-weighted attention in MeTTa

; Type definitions
(: precision (-> Percept EmotionalState PredictionError Number))
(: weighted-salience (-> Number Number Number))

; Compute precision (inverse uncertainty)
(= (precision $percept $emotional-state $prediction-error)
   (let* (($base-precision 0.5)
          ($arousal (get-arousal $emotional-state))
          ($arousal-effect (* -0.5 $arousal))
          ($error-magnitude (if $prediction-error 
                               (error-magnitude $prediction-error) 
                               0.0))
          ($error-boost (* 0.3 $error-magnitude))
          ($precision (+ $base-precision $arousal-effect $error-boost)))
     (clamp $precision 0.0 1.0)))

; Apply precision weighting to salience
(= (weighted-salience $salience $precision)
   (* $salience $precision))
"""

__all__ = [
    "COMMUNICATION_DECISION_RULES",
    "PREDICTION_RULES",
    "FREE_ENERGY_RULES",
    "PRECISION_RULES",
]
