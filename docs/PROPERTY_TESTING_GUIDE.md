# Property-Based Testing Guide

## Introduction

Property-based testing is a powerful testing approach that validates system invariants by generating hundreds or thousands of randomized test cases. Instead of writing individual test cases with specific inputs, you define **properties** that should always hold true, and the testing framework (Hypothesis) generates diverse inputs to verify these properties.

## Why Property-Based Testing?

Traditional example-based testing:
```python
def test_addition():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
```

Property-based testing:
```python
@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    """Property: Addition is commutative"""
    assert add(a, b) == add(b, a)
```

**Benefits:**
- **Comprehensive Coverage**: Tests thousands of input combinations automatically
- **Edge Case Discovery**: Finds bugs humans wouldn't think to test
- **Regression Prevention**: Hypothesis "shrinks" failing examples to minimal reproducible cases
- **Living Documentation**: Properties document system invariants clearly
- **High Confidence**: Validates that invariants hold under all conditions

## Project Structure

```
emergence_core/tests/property_tests/
├── __init__.py                      # Package initialization
├── strategies.py                    # Custom Hypothesis strategies
├── test_workspace_properties.py     # Workspace invariant tests
├── test_attention_properties.py     # Attention controller tests
├── test_memory_properties.py        # Memory system tests
└── test_emotion_properties.py       # Emotional dynamics tests
```

## Custom Strategies

Located in `strategies.py`, these generate valid instances of cognitive architecture types:

### Available Strategies

- **`embeddings(dim=384)`**: Generate normalized embedding vectors
- **`percepts()`**: Generate valid Percept objects
- **`goals()`**: Generate valid Goal objects
- **`memories()`**: Generate valid Memory objects
- **`emotional_states()`**: Generate valid EmotionalState objects
- **`percept_lists`**: Lists of percepts (0-50 items)
- **`goal_lists`**: Lists of goals (0-10 items)
- **`memory_lists`**: Lists of memories (0-20 items)

### Example Usage

```python
from hypothesis import given
from .strategies import percepts, goals

@given(percepts(), goals())
def test_workspace_integration(percept, goal):
    workspace = GlobalWorkspace()
    workspace.active_percepts[percept.id] = percept
    workspace.add_goal(goal)
    
    assert len(workspace.active_percepts) == 1
    assert len(workspace.current_goals) == 1
```

## Writing Property Tests

### 1. Import Required Modules

```python
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from .strategies import percepts, goals, emotional_states
```

### 2. Mark Tests as Property Tests

```python
@pytest.mark.property
class TestYourProperties:
    """Property-based tests for YourComponent."""
```

### 3. Define Properties

```python
@given(percepts())
@settings(max_examples=100)
def test_percept_id_unique(self, percept):
    """Property: Each percept has a unique ID."""
    assert percept.id is not None
    assert isinstance(percept.id, str)
```

### 4. Use `assume()` for Preconditions

```python
@given(percept_lists)
def test_attention_budget(self, percepts):
    assume(len(percepts) > 0)  # Skip if empty
    # ... test logic
```

## Common Property Patterns

### Invariant Properties

Conditions that must always hold:

```python
@given(emotional_states)
def test_emotion_bounded(self, state):
    """Property: Emotions are always bounded in [-1, 1]."""
    assert -1.0 <= state.valence <= 1.0
    assert -1.0 <= state.arousal <= 1.0
    assert -1.0 <= state.dominance <= 1.0
```

### Round-trip Properties

Serialization/deserialization should preserve data:

```python
@given(memories())
def test_memory_serialization(self, memory):
    """Property: Memories survive serialization round-trip."""
    data = memory.model_dump()
    restored = Memory(**data)
    assert restored.content == memory.content
```

### Ordering Properties

Verify sorting and ordering:

```python
@given(goal_lists)
def test_goals_priority_ordered(self, goals):
    """Property: Goals maintain priority ordering."""
    workspace = GlobalWorkspace()
    for goal in goals:
        workspace.add_goal(goal)
    
    priorities = [g.priority for g in workspace.current_goals]
    assert priorities == sorted(priorities, reverse=True)
```

### Budget/Constraint Properties

Verify resource constraints:

```python
@given(percept_lists, st.integers(min_value=1, max_value=20))
def test_attention_budget_respected(self, percepts, budget):
    """Property: Selected percepts never exceed budget."""
    controller = AttentionController(attention_budget=budget)
    selected = controller.select_for_broadcast(percepts)
    
    total = sum(p.complexity for p in selected)
    assert total <= budget
```

## Running Property Tests

### Run all property tests:
```bash
pytest -v -m property
```

### Run with statistics:
```bash
pytest -v -m property --hypothesis-show-statistics
```

### Run specific test file:
```bash
pytest -v emergence_core/tests/property_tests/test_workspace_properties.py
```

### Run with more examples:
```bash
pytest -v -m property --hypothesis-seed=0 \
    --hypothesis-profile=thorough
```

### Run single test:
```bash
pytest -v emergence_core/tests/property_tests/test_workspace_properties.py::TestWorkspaceProperties::test_emotional_state_bounded
```

## Interpreting Hypothesis Output

### Successful Test
```
test_emotion_bounded PASSED
  - 100 examples generated
  - 0 failing examples
```

### Failing Test
```
test_emotion_bounded FAILED
Falsifying example: test_emotion_bounded(
    state=EmotionalState(valence=1.5, arousal=0.0, dominance=0.0)
)
```

Hypothesis automatically **shrinks** the failing case to the minimal example that demonstrates the bug.

### Statistics Output
```
  - 100 examples generated in 2.3s
  - 15 examples discarded (assume() filtered)
  - Smallest example: 2 bytes
  - Largest example: 156 bytes
```

## Debugging Failing Property Tests

### 1. Reproduce with Seed
When a test fails, Hypothesis provides a seed:
```bash
pytest -v -m property --hypothesis-seed=123456789
```

### 2. Use `@example()` for Specific Cases
```python
from hypothesis import example

@given(percepts())
@example(Percept(id="test", modality="text", raw=""))  # Known edge case
def test_percept_handling(self, percept):
    # ...
```

### 3. Enable Verbose Output
```python
@settings(max_examples=100, verbosity=Verbosity.verbose)
```

### 4. Use Print Debugging
```python
@given(percepts())
def test_something(self, percept):
    print(f"Testing with: {percept.id}")
    # ...
```

## Configuration

### Global Settings (pyproject.toml)

```toml
[tool.pytest.ini_options]
markers = [
    "property: property-based tests using Hypothesis",
]
```

### Per-Test Settings

```python
@settings(
    max_examples=200,      # Run 200 examples instead of 100
    deadline=None,         # Disable deadline for slow tests
    suppress_health_check=[HealthCheck.too_slow]
)
```

### Hypothesis Profiles

Create `.hypothesis/profiles.yaml`:
```yaml
default:
  max_examples: 100
  deadline: 200

thorough:
  max_examples: 1000
  deadline: null

quick:
  max_examples: 20
  deadline: 100
```

Use with: `pytest --hypothesis-profile=thorough`

## Best Practices

### 1. Start Small
Begin with simple properties before complex ones:
- ✅ "Value is in valid range"
- ✅ "Operation doesn't crash"
- ⏭️ "Complex multi-step workflow"

### 2. Use Meaningful Property Names
```python
# ✅ Good
def test_emotional_state_bounded(self, state):

# ❌ Bad
def test_emotions(self, state):
```

### 3. Document Properties Clearly
```python
@given(percepts())
def test_percept_addition_idempotent(self, percept):
    """Property: Adding same percept twice has same effect as adding once."""
```

### 4. Use `assume()` Wisely
```python
@given(percept_lists)
def test_something(self, percepts):
    assume(len(percepts) > 5)  # Filter, don't assert
```

### 5. Keep Tests Focused
One property per test function.

### 6. Handle Floating Point Carefully
```python
# Use epsilon for floating point comparisons
assert abs(value1 - value2) < 1e-10
```

## Performance Considerations

### Limit Example Count for Slow Tests
```python
@settings(max_examples=20, deadline=None)
def test_expensive_operation(self, data):
    # ...
```

### Use Smaller Data Structures
```python
# Instead of:
percept_lists = st.lists(percepts(), max_size=1000)

# Use:
percept_lists = st.lists(percepts(), max_size=50)
```

### Profile Tests
```bash
pytest --durations=10 -m property
```

## Integration with CI/CD

Property tests run automatically in GitHub Actions:
```yaml
- name: Run property-based tests
  run: pytest -v -m property --hypothesis-show-statistics
```

Extended tests with 1000 examples run in a separate job to catch rare edge cases.

## Troubleshooting

### "Unsatisfiable: Unable to satisfy assumptions"
Too many `assume()` filters. Relax conditions or use different strategies.

### "Flaky test detected"
Non-deterministic behavior in code. Ensure tests are deterministic.

### "Deadline exceeded"
Increase deadline: `@settings(deadline=None)`

### "Data generation is slow"
Use simpler strategies or reduce `max_size` parameters.

## Examples from the Codebase

### Workspace Immutability
```python
@given(percept_lists, goal_lists, emotional_states)
def test_workspace_snapshot_immutability(self, percepts, goals, emotions):
    workspace = GlobalWorkspace()
    # ... populate workspace
    snapshot = workspace.broadcast()
    
    # Verify snapshot is frozen
    with pytest.raises(ValidationError):
        snapshot.cycle_count = 999
```

### Attention Budget Constraint
```python
@given(percept_lists, st.integers(min_value=1, max_value=20))
def test_attention_respects_budget(self, percepts, budget):
    controller = AttentionController(attention_budget=budget)
    selected = controller.select_for_broadcast(percepts)
    
    total = sum(p.complexity for p in selected)
    assert total <= budget
```

### Emotional Decay Convergence
```python
@given(emotional_states)
def test_emotion_decay_converges(self, initial_state):
    affect = AffectSubsystem(decay_rate=0.1)
    affect.emotional_state = initial_state
    
    initial_distance = distance_from_baseline(initial_state)
    
    for _ in range(10):
        affect._apply_decay()
    
    final_distance = distance_from_baseline(affect.emotional_state)
    assert final_distance <= initial_distance
```

## Resources

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Property-Based Testing Patterns](https://hypothesis.works/articles/what-is-property-based-testing/)
- [Hypothesis Best Practices](https://hypothesis.readthedocs.io/en/latest/details.html)

## Questions?

For questions about property-based testing in this project, see:
- Test examples in `emergence_core/tests/property_tests/`
- Custom strategies in `strategies.py`
- GitHub Issues for bug reports or feature requests
