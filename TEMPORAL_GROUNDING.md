# Temporal Grounding System

Implements temporal awareness with session tracking for the Lyra cognitive architecture.

## Core Components

### Temporal Awareness (`temporal/awareness.py`)
- **TemporalContext**: Tracks session state and time passage
- **TemporalAwareness**: Detects session boundaries (default: 1h gap threshold)
- **Session**: Stores session metadata, topics, and emotional arc

### Session Management (`temporal/sessions.py`)
- **SessionManager**: Handles session lifecycle and greeting contexts
- Greeting types: first_meeting, continuation, same_day, recent, long_gap

### Time Effects (`temporal/effects.py`)
- **TimePassageEffects**: Applies temporal decay
  - Emotional decay toward baseline (exponential)
  - Working memory fading
  - Goal urgency updates

### Pattern Learning (`temporal/expectations.py`)
- **TemporalExpectations**: Learns event patterns
- Predicts next occurrences with confidence
- Detects overdue expectations

### Utilities (`temporal/relative.py`)
- **RelativeTime**: Human-friendly time descriptions
- Examples: "just now", "3 hours ago", "in 2 days"

### Integration (`temporal/grounding.py`)
- **TemporalGrounding**: Unified interface
- Integrates all components
- Connected to cognitive loop and affect subsystem

## Usage

```python
from emergence_core.lyra.cognitive_core.temporal import TemporalGrounding

tg = TemporalGrounding()

# Track interaction
context = tg.on_interaction()
if context.is_new_session:
    greeting = tg.sessions.get_session_greeting_context()

# Apply time effects
state = {'emotions': {'valence': 0.9, 'arousal': 0.8}, ...}
updated = tg.apply_time_passage_effects(state)

# Record patterns
tg.record_event('daily_check')
```

## Configuration

```python
config = {
    "session_gap_threshold_seconds": 3600,  # 1 hour
    "min_observations": 3,
    "effects": {
        "emotion_decay_rate": 0.9,
        "context_fade_rate": 0.85,
        "consolidation_threshold_hours": 1.0
    }
}
```

## Integration Points

- `subsystem_coordinator.py`: Initialize with memory connection
- `cognitive_loop.py`: Track interactions, detect sessions
- `cycle_executor.py`: Apply time effects to emotions

## Files

**New:**
- `temporal/` module (7 files)
- `test_temporal_grounding.py`
- `demo_temporal_grounding.py`

**Modified:**
- `__init__.py`, `temporal_awareness.py`
- `subsystem_coordinator.py`, `cognitive_loop.py`, `cycle_executor.py`
