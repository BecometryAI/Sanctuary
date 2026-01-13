# Communication Decision System

Autonomous communication agency: decides whether to speak, stay silent, or defer.

## Components

1. **Drive System** (`drive.py`) - Internal urges to communicate
2. **Inhibition System** (`inhibition.py`) - Reasons not to communicate  
3. **Decision Loop** (`decision.py`) - SPEAK/SILENCE/DEFER decisions

## Decision Logic

Computes **net pressure** = drive - inhibition, applies thresholds:

| Net Pressure | Decision | When |
|--------------|----------|------|
| > 0.3 | **SPEAK** | Drive exceeds inhibition |
| < -0.2 | **SILENCE** | Inhibition exceeds drive |
| -0.2 to 0.3 | **DEFER/SILENCE** | Both high → defer; else silence |

## Usage

```python
from lyra.cognitive_core.communication import (
    CommunicationDecisionLoop,
    CommunicationDriveSystem,
    CommunicationInhibitionSystem,
    CommunicationDecision
)

# Initialize
drives = CommunicationDriveSystem()
inhibitions = CommunicationInhibitionSystem()
decision_loop = CommunicationDecisionLoop(drives, inhibitions)

# In cognitive cycle
result = decision_loop.evaluate(workspace, emotions, goals, memories)

if result.decision == CommunicationDecision.SPEAK:
    output = generate_output(result.urge)
    drives.record_output()
    inhibitions.record_output(output)
```

## Configuration

```python
decision_loop = CommunicationDecisionLoop(
    drive_system=drives,
    inhibition_system=inhibitions,
    config={
        "speak_threshold": 0.3,        # Net pressure to speak
        "silence_threshold": -0.2,     # Net pressure for silence
        "defer_min_drive": 0.3,        # Min drive for deferral
        "defer_min_inhibition": 0.3,   # Min inhibition for deferral
        "defer_duration_seconds": 30,  # Deferral duration
        "max_deferred": 10,            # Max queue size
        "max_defer_attempts": 3,       # Max reconsiderations
        "history_size": 100            # Decision history limit
    }
)
```

## Deferred Queue

Time-based reconsideration:
- Priority by urge intensity × priority
- Max 3 attempts before dropping
- Auto-cleanup of expired items

## Testing

```bash
pytest emergence_core/tests/test_communication_decision.py -v
```

## Related PRs

- PR #87: Decoupled cognitive loop from I/O
- PR #88: Communication drive system
- PR #89: Communication inhibition system
inhibitions = CommunicationInhibitionSystem()
decision_loop = CommunicationDecisionLoop(drives, inhibitions)

# In your cognitive cycle:
result = decision_loop.evaluate(
    workspace_state=workspace,
    emotional_state=emotions,
    goals=active_goals,
    memories=recent_memories
)

if result.decision == CommunicationDecision.SPEAK:
    output = generate_output(result.urge)
    drives.record_output()
    inhibitions.record_output(content=output)
elif result.decision == CommunicationDecision.SILENCE:
    log_silence_decision(result.reason)
elif result.decision == CommunicationDecision.DEFER:
    log_deferral(result.defer_until)
```

### Configuration

```python
decision_loop = CommunicationDecisionLoop(
    drive_system=drives,
    inhibition_system=inhibitions,
    config={
        "speak_threshold": 0.3,        # Net pressure > 0.3 = SPEAK
        "silence_threshold": -0.2,     # Net pressure < -0.2 = SILENCE
        "defer_min_drive": 0.3,        # Min drive for deferral
        "defer_min_inhibition": 0.3,   # Min inhibition for deferral
        "defer_duration_seconds": 30,  # How long to defer
        "max_deferred": 10,            # Max deferred items
        "max_defer_attempts": 3,       # Max reconsideration attempts
        "history_size": 100            # Max decision history
    }
)
```


