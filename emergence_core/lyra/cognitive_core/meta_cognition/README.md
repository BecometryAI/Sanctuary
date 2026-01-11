# Meta-Cognition Module

## Overview

Implements self-monitoring and introspective capabilities enabling the system to observe processing patterns, learn from outcomes, and adapt strategies.

## Core Components

### 1. Processing Monitoring
- `ProcessingObservation`: Records process execution metrics
- `PatternDetector`: Identifies success/failure patterns
- `MetaCognitiveMonitor`: Context managers for observation

### 2. Action-Outcome Learning
- `ActionOutcomeLearner`: Tracks intended vs actual results
- `ActionModel`: Builds predictive reliability models

### 3. Attention Allocation History
- `AttentionHistory`: Tracks attention distribution
- `AttentionPatternLearner`: Learns optimal strategies

### 4. Unified Interface
- `MetaCognitiveSystem`: Single API for all subsystems

## Quick Start

```python
from lyra.cognitive_core.meta_cognition import MetaCognitiveSystem

meta = MetaCognitiveSystem()

# Monitor process
with meta.monitor.observe('reasoning') as ctx:
    ctx.input_complexity = 0.7
    result = perform_reasoning()
    ctx.output_quality = 0.9

# Record action outcome
meta.record_action_outcome(
    action_id="act_1",
    action_type="speak",
    intended="provide helpful response",
    actual="provided detailed answer",
    context={"complexity": 0.6}
)

# Get self-assessment
assessment = meta.get_self_assessment()
```

## Key Features

- **Low overhead**: ~1ms per observation
- **Auto-pruning**: 10K observations/5K outcomes/1K allocations
- **Pattern detection**: Automatic with confidence scores
- **Type safety**: Full annotations throughout
- **Thread safe**: All operations concurrent-safe

## Integration

Integrates with: Attention system, Action system, Goal system, Memory system

## Testing

Run tests: `pytest emergence_core/tests/test_metacognition*.py`

## Documentation

- `USAGE.md`: Detailed examples and patterns
- Example: `examples/metacognition_example.py`
