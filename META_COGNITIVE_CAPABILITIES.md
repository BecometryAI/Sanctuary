# Meta-Cognitive Capabilities

## Overview

Three interconnected meta-cognitive systems that enable the system to observe and learn from its own processing:

1. **Processing Monitor** - Tracks cognitive processing patterns
2. **Action-Outcome Learner** - Learns what actions actually achieve
3. **Attention History** - Tracks attention allocation effectiveness

## Quick Start

```python
from lyra.cognitive_core.meta_cognition import MetaCognitiveSystem

system = MetaCognitiveSystem()

# Monitor processing
with system.monitor.observe("reasoning") as ctx:
    ctx.set_complexity(0.7)
    result = perform_reasoning()
    ctx.set_quality(0.8)

# Record action outcome
system.action_learner.record_outcome(
    action_id="a1", action_type="speak",
    intended="provide helpful response",
    actual="provided detailed response",
    context={"user_sentiment": "positive"}
)

# Track attention
allocation_id = system.attention_history.record_allocation(
    allocation={"goal1": 0.6, "goal2": 0.4},
    trigger="new_percept",
    workspace_state=current_state
)

# Get self-assessment
assessment = system.get_self_assessment()
print(assessment.identified_strengths)
print(assessment.suggested_adaptations)

# Introspect
response = system.introspect("What are my strengths?")
```

## Components

### ProcessingMonitor
Observes cognitive processes and identifies patterns (success conditions, failure modes, efficiency factors).

**Key Methods:**
- `observe(process_type)` - Context manager for monitoring
- `get_process_statistics(process_type)` - Get stats for a process
- `get_identified_patterns()` - Get detected patterns

### ActionOutcomeLearner
Tracks what actions achieve vs. intentions, builds predictive models.

**Key Methods:**
- `record_outcome(action_id, action_type, intended, actual, context)` - Record outcome
- `get_action_reliability(action_type)` - Get reliability metrics
- `predict_outcome(action_type, context)` - Predict likely outcome

### AttentionHistory
Tracks attention allocation patterns and learns effective strategies.

**Key Methods:**
- `record_allocation(allocation, trigger, workspace_state)` - Record allocation
- `record_outcome(allocation_id, goal_progress, discoveries, missed)` - Record outcome
- `get_recommended_allocation(context, goals)` - Get recommendation

### MetaCognitiveSystem
Unified interface to all subsystems.

**Key Methods:**
- `get_self_assessment()` - Comprehensive self-assessment
- `introspect(query)` - Answer questions about cognitive patterns

## Configuration

```python
config = {
    "monitor": {"max_observations": 1000},
    "action_learner": {"max_outcomes": 1000, "min_samples_for_model": 5},
    "attention_history": {"max_allocations": 1000}
}
system = MetaCognitiveSystem(config=config)
```

## Integration

Works alongside existing `SelfMonitor`. Can be used independently or integrated into the cognitive loop.

See `example_metacognitive_integration.py` for integration examples.

## Testing

```bash
pytest emergence_core/tests/test_meta_cognitive_capabilities.py -v
```
