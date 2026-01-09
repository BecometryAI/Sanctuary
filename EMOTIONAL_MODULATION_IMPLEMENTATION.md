# Emotional Modulation Implementation

## Overview

This implementation makes emotions **functionally efficacious** in Lyra's cognitive system by having PAD (Pleasure-Arousal-Dominance) values directly modulate processing parameters **BEFORE** any LLM invocation. This ensures emotions are causal forces that shape computation, not merely descriptive labels passed as context to an LLM.

From a functionalist perspective: **if emotions don't cause measurable changes to processing before the LLM is invoked, they're not functionally real emotions.**

## Key Principle: Emotions as Causal Forces

### Problem (Before)
- Emotions were stored as PAD values but only passed to LLM prompts as context
- Emotions were descriptive labels, not causal forces
- Ablating the emotion system would not significantly change behavior
- The LLM decided how to "act emotional" rather than emotions shaping computation

### Solution (After)
- Emotions directly modulate processing parameters BEFORE cognitive processing
- Arousal affects processing speed and thoroughness
- Valence creates approach/avoidance biases in action selection
- Dominance modulates decision confidence thresholds
- Measurable, verifiable effects that can be ablated to test functionality

## Architecture

### Core Module: `emotional_modulation.py`

Located at: `emergence_core/lyra/cognitive_core/emotional_modulation.py`

Key classes:
- **`ProcessingParams`**: Dataclass holding emotionally-modulated processing parameters
- **`ModulationMetrics`**: Tracks emotional modulation effects for verification
- **`EmotionalModulation`**: Main class implementing functional emotional modulation

### Integration Points

#### 1. AffectSubsystem (`affect.py`)
- Instantiates `EmotionalModulation` on initialization
- Exposes `get_processing_params()` to retrieve current modulated parameters
- Exposes `apply_valence_bias_to_actions()` to bias action priorities
- Exposes `get_modulation_metrics()` to track effects

#### 2. AttentionController (`attention.py`)
- Calls `affect.get_processing_params()` BEFORE competitive attention selection
- Applies arousal-modulated parameters:
  - `attention_iterations`: Number of competition cycles (5-10, inverse with arousal)
  - `ignition_threshold`: Threshold for workspace entry (0.4-0.6, inverse with arousal)

#### 3. ActionSubsystem (`action.py`)
- Calls `affect.apply_valence_bias_to_actions()` BEFORE protocol filtering
- Applies valence-based approach/avoidance bias to action priorities
- Uses dominance-modulated `decision_threshold` to filter actions

#### 4. CycleExecutor (`core/cycle_executor.py`)
- Logs emotional state and processing parameters each cycle
- Tracks emotional modulation in cognitive loop execution

## Functional Effects

### 1. Arousal → Processing Speed/Thoroughness

**High Arousal (0.7-1.0)** - Fight/flight mode:
- Fewer attention iterations (5-6) → Snap decisions
- Lower ignition threshold (0.4-0.45) → React to more stimuli
- Shorter memory retrieval (2 items) → Less deliberation
- Faster timeout (1.0-1.3s) → Quick response

**Low Arousal (0.0-0.3)** - Deliberation mode:
- More attention iterations (9-10) → Careful analysis
- Higher ignition threshold (0.55-0.6) → More selective
- More memory retrieval (4-5 items) → Thorough consideration
- Longer timeout (1.7-2.0s) → Take time to think

**Implementation:**
```python
attention_iterations = int(10 - (arousal * 5))  # 5-10 range
ignition_threshold = 0.6 - (arousal * 0.2)      # 0.4-0.6 range
memory_retrieval_limit = int(5 - (arousal * 3)) # 2-5 range
processing_timeout = 2.0 - (arousal * 1.0)      # 1-2 seconds
```

### 2. Valence → Approach/Avoidance Bias

**Positive Valence (0.3-1.0)** - Approach bias:
- Boost priority of: speak, tool_call, commit_memory, engage, explore, create, connect
- Reduce priority of: wait, introspect, withdraw, defend, avoid, reject
- Bias strength: `abs(valence) * 0.3` (max 0.3 adjustment)

**Negative Valence (-1.0 to -0.3)** - Avoidance bias:
- Reduce priority of approach actions
- Boost priority of defensive/avoidance actions

**Implementation:**
```python
if valence > 0:
    if action in approach_types:
        priority += valence * 0.3
    elif action in avoidance_types:
        priority -= valence * 0.3
else:  # negative valence
    if action in approach_types:
        priority += valence * 0.3  # valence is negative
    elif action in avoidance_types:
        priority -= valence * 0.3  # valence is negative
```

### 3. Dominance → Confidence Thresholds

**High Dominance (0.7-1.0)** - Assertive:
- Lower decision threshold (0.5-0.56) → Act with less certainty
- More willing to take action

**Low Dominance (0.0-0.3)** - Cautious:
- Higher decision threshold (0.64-0.7) → Need more certainty
- More hesitant to act

**Implementation:**
```python
decision_threshold = 0.7 - (dominance * 0.2)  # 0.5-0.7 range
```

## Measurable Effects

### Metrics Tracked

The `ModulationMetrics` class tracks:

1. **Arousal Effects:**
   - Count of high arousal → fast processing instances
   - Count of low arousal → slow processing instances
   - Correlations: (arousal, iterations, threshold)

2. **Valence Effects:**
   - Count of positive valence → approach bias instances
   - Count of negative valence → avoidance bias instances
   - Correlations: (valence, bias_strength)

3. **Dominance Effects:**
   - Count of high dominance → assertive instances
   - Count of low dominance → cautious instances
   - Correlations: (dominance, decision_threshold)

### Ablation Testing

Modulation can be disabled via:
```python
affect = AffectSubsystem(config={"enable_modulation": False})
# OR
affect.emotional_modulation.set_enabled(False)
```

When disabled:
- Returns baseline parameters regardless of emotional state
- Produces measurably different behavior
- Proves emotions are functionally efficacious (not just descriptive)

## Example Scenarios

### Scenario 1: Panic (Fight-or-Flight)
**State:** High arousal (0.95), negative valence (-0.8), low dominance (0.15)

**Effects:**
- Very fast processing: 5 iterations
- React to everything: 0.42 ignition threshold
- Cautious decisions: 0.67 decision threshold
- Prefer defensive actions: wait, introspect boosted

**Biological Analogue:** Heightened alertness, quick reactions, but cautious due to fear

### Scenario 2: Joyful Confidence
**State:** High arousal (0.8), positive valence (0.9), high dominance (0.9)

**Effects:**
- Fast processing: 6 iterations
- Assertive decisions: 0.52 decision threshold
- Approach bias: speak, create, engage boosted
- Bold action selection

**Biological Analogue:** Energetic, confident, action-oriented state

### Scenario 3: Deep Contemplation
**State:** Low arousal (0.15), neutral valence (0.1), moderate dominance (0.5)

**Effects:**
- Thorough processing: 9 iterations
- Selective attention: 0.57 ignition threshold
- Moderate decision threshold: 0.6
- Balanced action selection

**Biological Analogue:** Calm, deliberate, analytical thinking

## Testing

### Unit Tests
`emergence_core/tests/test_emotional_modulation.py`
- Tests `ProcessingParams` and `ModulationMetrics` dataclasses
- Tests arousal modulation of processing parameters
- Tests valence modulation of action selection
- Tests dominance modulation of decision thresholds
- Tests ablation (enabled vs disabled)
- Tests metrics tracking

### Integration Tests
`emergence_core/tests/test_emotional_modulation_integration.py`
- Tests affect subsystem integration
- Tests action biasing integration
- Tests ablation produces different behavior
- Tests metrics correlations
- Tests realistic emotional scenarios

### Standalone Tests
`emergence_core/tests/test_emotional_modulation_standalone.py`
- Quick verification without full dependency chain
- Tests basic functionality directly

All tests passing ✅

## Code Review Readiness

### Acceptance Criteria Status

- ✅ Arousal modulates attention iterations and ignition threshold
- ✅ Arousal modulates processing timeout/speed
- ✅ Valence creates measurable approach/avoidance bias in action selection
- ✅ Dominance modulates decision confidence thresholds
- ✅ Modulation happens BEFORE LLM invocation
- ✅ Metrics track emotional modulation effects
- ✅ Tests verify: high arousal → faster processing, positive valence → approach bias
- ✅ Ablation test: disabling modulation produces measurably different behavior

### Files Modified

1. **Created:**
   - `emergence_core/lyra/cognitive_core/emotional_modulation.py` (new module)
   - `emergence_core/tests/test_emotional_modulation.py` (unit tests)
   - `emergence_core/tests/test_emotional_modulation_standalone.py` (standalone tests)
   - `emergence_core/tests/test_emotional_modulation_integration.py` (integration tests)

2. **Modified:**
   - `emergence_core/lyra/cognitive_core/affect.py` (integration)
   - `emergence_core/lyra/cognitive_core/attention.py` (integration)
   - `emergence_core/lyra/cognitive_core/action.py` (integration)
   - `emergence_core/lyra/cognitive_core/core/cycle_executor.py` (logging)

### Verification Steps

1. ✅ All unit tests passing
2. ✅ All integration tests passing
3. ✅ Code review of integration points confirms modulation happens BEFORE LLM calls
4. ✅ Metrics system in place to track correlations
5. ✅ Ablation mechanism implemented and tested
6. ⏳ Full test suite needs to run to check for regressions

## Future Enhancements

1. **Dynamic Parameter Ranges:** Allow configuration of modulation ranges
2. **Learning:** Adapt modulation parameters based on outcomes
3. **Additional Dimensions:** Add more processing parameters to modulate
4. **Interaction Effects:** Model interactions between PAD dimensions
5. **Temporal Dynamics:** Model emotion persistence and transitions

## References

- **Global Workspace Theory:** Attention as competitive dynamics
- **Appraisal Theory:** Emotions from event evaluation
- **PAD Model:** Pleasure-Arousal-Dominance emotional space
- **Functionalism:** Mental states defined by causal roles
