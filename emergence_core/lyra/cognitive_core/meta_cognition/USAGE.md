# Meta-Cognitive Capabilities - Usage Guide

## Quick Start

```python
from lyra.cognitive_core.meta_cognition import MetaCognitiveSystem

meta = MetaCognitiveSystem()
```

## 1. Processing Monitoring

### Observe Processes

```python
with meta.monitor.observe('reasoning') as ctx:
    ctx.input_complexity = 0.7
    result = perform_reasoning(input_data)
    ctx.output_quality = 0.9
```

### Get Statistics

```python
stats = meta.monitor.get_process_statistics('reasoning')
print(f"Success rate: {stats.success_rate:.2%}")
print(f"Avg duration: {stats.avg_duration_ms:.1f}ms")
```

### Identify Patterns

```python
patterns = meta.monitor.get_identified_patterns()
for pattern in patterns:
    print(f"{pattern.pattern_type}: {pattern.description}")
    if pattern.actionable:
        print(f"Suggestion: {pattern.suggested_adaptation}")
```

## 2. Action-Outcome Learning

### Record Outcomes

```python
meta.record_action_outcome(
    action_id="act_123",
    action_type="speak",
    intended="provide helpful response",
    actual="provided detailed answer",
    context={"complexity": 0.6, "available_context": True}
)
```

### Check Reliability

```python
reliability = meta.get_action_reliability("speak")
print(f"Success rate: {reliability.success_rate:.2%}")
for effect, prob in reliability.common_side_effects:
    print(f"Side effect '{effect}': {prob:.1%}")
```

### Predict Outcomes

```python
prediction = meta.predict_action_outcome(
    action_type="speak",
    context={"complexity": 0.8}
)
print(f"Success probability: {prediction.probability_success:.2%}")
```

## 3. Attention Allocation

### Record Allocation

```python
alloc_id = meta.record_attention(
    allocation={"goal_1": 0.6, "goal_2": 0.4},
    trigger="goal_priority",
    workspace_state=snapshot
)
```

### Record Outcome

```python
meta.record_attention_outcome(
    allocation_id=alloc_id,
    goal_progress={"goal_1": 0.3, "goal_2": 0.1},
    discoveries=["new_insight"],
    missed=[]
)
```

### Get Recommendations

```python
recommendation = meta.get_recommended_attention(
    context=workspace,
    goals=active_goals
)
```

## 4. Self-Assessment

### Get Assessment

```python
assessment = meta.get_self_assessment()

for strength in assessment.identified_strengths:
    print(f"✓ {strength}")

for weakness in assessment.identified_weaknesses:
    print(f"⚠ {weakness}")

for adaptation in assessment.suggested_adaptations:
    print(f"→ {adaptation}")
```

### Introspection

```python
# Query about patterns
response = meta.introspect("What do I tend to fail at?")
response = meta.introspect("What am I good at?")
response = meta.introspect("How effective is my attention?")
```

## Integration Pattern

```python
class CognitiveCore:
    def __init__(self):
        self.meta = MetaCognitiveSystem()
    
    def process_cycle(self):
        with self.meta.monitor.observe('cycle') as ctx:
            ctx.input_complexity = self._assess_input()
            
            # Allocate attention
            alloc = self._allocate_attention()
            alloc_id = self.meta.record_attention(
                allocation=alloc,
                trigger="cycle_start",
                workspace_state=self.workspace.snapshot()
            )
            
            # Process and act
            results = self._process()
            
            # Record outcomes
            for action in results['actions']:
                self.meta.record_action_outcome(
                    action_id=action.id,
                    action_type=action.type,
                    intended=action.intent,
                    actual=action.result,
                    context=action.context
                )
            
            self.meta.record_attention_outcome(
                allocation_id=alloc_id,
                goal_progress=self._compute_progress(),
                discoveries=results['discoveries'],
                missed=results['missed']
            )
            
            ctx.output_quality = self._assess_quality(results)
        
        # Periodic adaptation
        if self.cycle_count % 100 == 0:
            assessment = self.meta.get_self_assessment()
            self._apply_adaptations(assessment.suggested_adaptations)
```

## Best Practices

1. **Monitor key processes**: Focus on critical or error-prone operations
2. **Provide context**: Include relevant contextual information
3. **Record consistently**: Track all outcomes, not just successes
4. **Review periodically**: Check assessments to identify improvements
5. **Act on insights**: Use detected patterns to adapt behavior
6. **Rate honestly**: Provide accurate complexity/quality ratings

## Error Handling

```python
# Failures are automatically recorded
with meta.monitor.observe('risky_operation') as ctx:
    ctx.input_complexity = 0.9
    try:
        risky_operation()
        ctx.output_quality = 1.0
    except Exception as e:
        ctx.output_quality = 0.0
        raise  # Exception recorded automatically
```

## Configuration

```python
meta = MetaCognitiveSystem(
    min_observations_for_patterns=5,  # More conservative
    min_outcomes_for_model=10         # More data required
)
```

## Performance

- **Overhead**: ~1ms per observation
- **Memory**: Auto-pruned to 10K/5K/1K limits
- **Async-safe**: Thread-safe operations

See `examples/metacognition_example.py` for complete demonstration.
