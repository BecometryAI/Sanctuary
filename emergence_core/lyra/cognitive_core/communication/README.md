# Communication Decision System

Autonomous communication agency: decides whether to speak, stay silent, or defer.

## Components

1. **Drive System** (`drive.py`) - Internal urges to communicate
2. **Inhibition System** (`inhibition.py`) - Reasons not to communicate  
3. **Decision Loop** (`decision.py`) - SPEAK/SILENCE/DEFER decisions
4. **Deferred Queue** (`deferred.py`) - Queue communications for better timing
5. **Silence Tracker** (`silence.py`) - Explicit silence tracking with typed reasons

## Decision Logic

Computes **net pressure** = drive - inhibition, applies thresholds:

| Net Pressure | Decision | When |
|--------------|----------|------|
| > 0.3 | **SPEAK** | Drive exceeds inhibition |
| < -0.2 | **SILENCE** | Inhibition exceeds drive |
| -0.2 to 0.3 | **DEFER/SILENCE** | Both high → defer; else silence |

## Deferred Queue

The deferred queue manages communications that should be reconsidered later:

### DeferralReason Types

- `BAD_TIMING`: Just spoke, need spacing between outputs
- `WAIT_FOR_RESPONSE`: Asked question, waiting for answer
- `TOPIC_CHANGE`: Save for when topic returns
- `PROCESSING`: Still thinking, will share when ready
- `COURTESY`: Let them finish their thought
- `CUSTOM`: Custom reason specified in description

### Features

- **Priority ordering**: Highest priority ready items surface first (priority × urge intensity)
- **Expiration**: Items expire after max_age (default: 300s)
- **Max attempts**: Items released after configured reconsiderations (default: 3)
- **History tracking**: Released and expired items tracked separately
- **Auto-cleanup**: Expired items automatically removed each cycle

## Usage

```python
from lyra.cognitive_core.communication import (
    CommunicationDecisionLoop,
    CommunicationDriveSystem,
    CommunicationInhibitionSystem,
    CommunicationDecision,
    DeferralReason
)

# Initialize
drives = CommunicationDriveSystem()
inhibitions = CommunicationInhibitionSystem()
decision_loop = CommunicationDecisionLoop(drives, inhibitions)

# In cognitive cycle
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

## Configuration

```python
decision_loop = CommunicationDecisionLoop(
    drive_system=drives,
    inhibition_system=inhibitions,
    config={
        # Decision thresholds
        "speak_threshold": 0.3,        # Net pressure to speak
        "silence_threshold": -0.2,     # Net pressure for silence
        "defer_min_drive": 0.3,        # Min drive for deferral
        "defer_min_inhibition": 0.3,   # Min inhibition for deferral
        
        # Deferred queue settings
        "defer_duration_seconds": 30,  # Default deferral duration
        "max_queue_size": 20,          # Max deferred items
        "max_defer_attempts": 3,       # Max reconsiderations
        "default_defer_seconds": 30,   # Default defer time
        
        # History limits
        "history_size": 100,           # Decision history limit
        "max_history_size": 50         # Deferred history limit
    }
)
```

## Testing

```bash
pytest emergence_core/tests/test_communication_decision.py -v
pytest emergence_core/tests/test_deferred_queue.py -v
pytest emergence_core/tests/test_communication_drive.py -v
pytest emergence_core/tests/test_communication_inhibition.py -v
pytest emergence_core/tests/test_silence_action.py -v
```

## Related PRs

- PR #87: Decoupled cognitive loop from I/O
- PR #88: Communication drive system
- PR #89: Communication inhibition system
- PR #90: Communication decision loop
- PR #91: Silence-as-action tracking
- Current: Comprehensive deferred communication queue
