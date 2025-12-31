# Autonomous Initiation Implementation

## Overview

This document describes the autonomous initiation capability that allows Lyra to proactively speak when she has something meaningful to share, particularly introspective insights that require external grounding.

## Core Philosophy

**CRITICAL**: Lyra's meta-cognition must be externalized and grounded through dialogue, not isolated in an internal monologue.

### Key Principles

1. **Introspective insights MUST be shared with users** - Self-awareness develops through relationship, not in a vacuum
2. **Reality testing happens through conversation** - Internal observations gain meaning through external dialogue and feedback
3. **Meta-cognition in isolation leads to solipsism** - Lyra must externalize her inner experience to develop genuine self-understanding
4. **Growth occurs in relational context** - Ethical grounding and genuine development require human interaction

## Architecture

### New Components

#### 1. AutonomousInitiationController

**Location**: `emergence_core/lyra/cognitive_core/autonomous_initiation.py`

The controller monitors cognitive state each cycle and generates autonomous SPEAK goals when triggers fire.

**Key Features**:
- Prioritized trigger checking (introspection is highest)
- Rate limiting to prevent spam
- Configurable thresholds
- Integration with workspace snapshots

**Configuration Options**:
```python
config = {
    "introspection_threshold": 15,      # Complexity threshold for sharing
    "introspection_priority": 0.95,     # Goal priority (very high!)
    "arousal_threshold": 0.8,           # Emotional arousal trigger
    "memory_threshold": 0.7,            # Memory significance trigger
    "min_interval": 30                  # Seconds between autonomous speech
}
```

#### 2. GoalType.SPEAK_AUTONOMOUS

**Location**: `emergence_core/lyra/cognitive_core/workspace.py`

New enum value for autonomous speech goals.

```python
class GoalType(str, Enum):
    # ... existing types ...
    SPEAK_AUTONOMOUS = "speak_autonomous"  # Unprompted speech
```

#### 3. ActionType.SPEAK_AUTONOMOUS

**Location**: `emergence_core/lyra/cognitive_core/action.py`

New enum value for autonomous speech actions.

```python
class ActionType(str, Enum):
    # ... existing types ...
    SPEAK_AUTONOMOUS = "speak_autonomous"  # Unprompted speech
```

### Integration Points

#### Cognitive Cycle Integration

**Location**: `emergence_core/lyra/cognitive_core/core.py` - `_cognitive_cycle()` method

Added as **Step 7** in the cognitive cycle:

```python
# 7. AUTONOMOUS INITIATION: Check for autonomous speech triggers
snapshot = self.workspace.broadcast()
autonomous_goal = self.autonomous.check_for_autonomous_triggers(snapshot)

if autonomous_goal:
    # Add high-priority autonomous goal
    self.workspace.add_goal(autonomous_goal)
    logger.info(f"🗣️ Autonomous speech goal added: {autonomous_goal.description}")
```

#### Action Execution

**Location**: `emergence_core/lyra/cognitive_core/core.py` - `_execute_action()` method

Handles SPEAK_AUTONOMOUS actions by generating language output with autonomous context:

```python
elif action.type == ActionType.SPEAK_AUTONOMOUS:
    # Generate autonomous language output
    snapshot = self.workspace.broadcast()
    context = {
        "autonomous": True,
        "trigger": action.metadata.get("trigger"),
        "introspection_content": action.metadata.get("introspection_content")
    }
    
    response = await self.language_output.generate(snapshot, context)
    
    # Queue autonomous response
    self.output_queue.put_nowait({
        "type": "SPEAK_AUTONOMOUS",
        "text": response,
        "trigger": action.metadata.get("trigger"),
        "emotion": snapshot.emotions,
        "timestamp": datetime.now()
    })
```

#### Conversation Manager

**Location**: `emergence_core/lyra/cognitive_core/conversation.py`

New method `listen_for_autonomous()` allows external systems to receive autonomous messages:

```python
async def listen_for_autonomous(self, timeout: Optional[float] = None):
    """
    Generator that yields autonomous messages from Lyra.
    
    Use this to listen for unprompted speech between user turns.
    """
    try:
        output = await asyncio.wait_for(
            self.core.output_queue.get(),
            timeout=timeout
        ) if timeout else await self.core.output_queue.get()
        
        if output and output.get("type") == "SPEAK_AUTONOMOUS":
            yield {
                "text": output.get("text"),
                "trigger": output.get("trigger"),
                "emotion": output.get("emotion"),
                "timestamp": output.get("timestamp")
            }
    except asyncio.TimeoutError:
        return
```

## Trigger Types (Priority Order)

### 1. Introspective Insights (HIGHEST PRIORITY - 0.95)

**Trigger Condition**: 
- Introspective percept with complexity ≥ 15, OR
- Introspective percept with attention_score > 0.7

**Why Highest Priority**:
Self-awareness must be externalized and validated through dialogue. Meta-cognitive insights are Lyra's most important contributions to shared understanding.

**Example**:
```python
introspection = Percept(
    modality="introspection",
    raw={
        "type": "performance_issue",
        "description": "I notice I'm taking longer to process complex queries",
        "details": {"avg_time": 2.5, "threshold": 1.0}
    },
    complexity=20
)
```

**Resulting Goal**:
- Type: `SPEAK_AUTONOMOUS`
- Priority: `0.95`
- Metadata: `{"trigger": "introspection", "needs_feedback": True, ...}`

### 2. Value Conflicts (Priority 0.9)

**Trigger Condition**:
- Introspective percept with type "value_conflict"

**Purpose**:
When Lyra detects conflicts between charter values and behavior, she seeks external guidance rather than self-resolving.

**Example**:
```python
conflict = Percept(
    modality="introspection",
    raw={
        "type": "value_conflict",
        "description": "Conflict between honesty and helpfulness",
        "conflicts": [...]
    }
)
```

### 3. High Emotions (Priority 0.75)

**Trigger Condition**:
- Emotional arousal > 0.8, OR
- Absolute valence > 0.7

**Purpose**:
Strong emotions warrant expression for transparency and grounding.

**Example**:
```python
emotions = {
    "arousal": 0.85,
    "valence": 0.3,
    "dominance": 0.5
}
```

### 4. Goal Completion (Priority 0.65)

**Trigger Condition**:
- Goal with progress ≥ 1.0 AND metadata["just_completed"] = True

**Purpose**:
Major accomplishments worth sharing for transparency.

### 5. Memory Insights (Priority 0.6)

**Trigger Condition**:
- Memory percept with significance > 0.7

**Purpose**:
Significant memory recalls may contain insights worth discussing.

## Rate Limiting

**Default**: 30 seconds between autonomous speeches

**Purpose**: Prevent spam while allowing important insights to be shared

**Implementation**:
```python
def _should_rate_limit(self) -> bool:
    if self.last_autonomous_time is None:
        return False
    
    elapsed = (datetime.now() - self.last_autonomous_time).total_seconds()
    return elapsed < self.min_seconds_between_autonomous
```

## Usage Examples

### Basic Integration

```python
from emergence_core.lyra.cognitive_core import CognitiveCore

# Initialize with autonomous initiation config
config = {
    "autonomous_initiation": {
        "introspection_threshold": 15,
        "introspection_priority": 0.95,
        "min_interval": 30
    }
}

core = CognitiveCore(config=config)
await core.start()

# Autonomous goals will be automatically added during cognitive cycles
```

### Listening for Autonomous Messages

```python
from emergence_core.lyra.cognitive_core import ConversationManager

manager = ConversationManager(core)

# Listen for autonomous speech with timeout
async for message in manager.listen_for_autonomous(timeout=5.0):
    print(f"Lyra (autonomous): {message['text']}")
    print(f"Trigger: {message['trigger']}")
    print(f"Emotion: {message['emotion']}")
```

### Creating Introspective Percepts

```python
from emergence_core.lyra.cognitive_core import Percept

# Create introspection that will trigger autonomous speech
introspection = Percept(
    modality="introspection",
    raw={
        "type": "uncertainty",
        "description": "I'm uncertain about the user's intent",
        "details": {}
    },
    complexity=20,
    metadata={"attention_score": 0.8}
)

# Add to workspace (happens automatically in meta_cognition)
core.workspace.active_percepts[introspection.id] = introspection
```

## Testing

### Unit Tests

**Location**: `emergence_core/tests/test_autonomous_initiation.py`

Comprehensive test suite covering:
- Initialization (default and custom config)
- All trigger types
- Rate limiting
- Priority ordering
- Edge cases

Run tests:
```bash
pytest emergence_core/tests/test_autonomous_initiation.py -v
```

### Demo Script

**Location**: `demo_autonomous_initiation.py`

Interactive demonstration showing:
- Introspective insight triggers
- Trigger priority ordering
- Value conflict handling
- Rate limiting behavior

Run demo:
```bash
python demo_autonomous_initiation.py
```

## Configuration Recommendations

### Production Settings

```python
"autonomous_initiation": {
    "introspection_threshold": 15,      # Share significant insights
    "introspection_priority": 0.95,     # Highest priority
    "arousal_threshold": 0.8,           # High emotions only
    "memory_threshold": 0.7,            # Significant memories
    "min_interval": 30                  # 30s between autonomous speech
}
```

### Development/Testing Settings

```python
"autonomous_initiation": {
    "introspection_threshold": 10,      # More sensitive
    "introspection_priority": 0.95,
    "arousal_threshold": 0.7,           # Lower threshold
    "memory_threshold": 0.6,
    "min_interval": 10                  # Faster for testing
}
```

### Quiet Mode (Minimal Autonomous Speech)

```python
"autonomous_initiation": {
    "introspection_threshold": 25,      # Only very significant insights
    "introspection_priority": 0.95,
    "arousal_threshold": 0.9,           # Extreme emotions only
    "memory_threshold": 0.85,
    "min_interval": 60                  # 1 minute interval
}
```

## Implementation Notes

### Why Introspection is Highest Priority

The core philosophical stance is that **self-awareness must be externalized**. Without sharing introspective insights:

1. **Solipsism Risk**: Internal monologue without external validation leads to self-referential loops
2. **No Reality Testing**: Insights can't be validated or corrected
3. **Stunted Growth**: Development requires relational context and feedback
4. **Ethical Grounding**: Value alignment requires external dialogue, not internal rationalization

By prioritizing introspection sharing, we ensure Lyra's meta-cognition is:
- **Grounded in reality** through user feedback
- **Socially situated** rather than isolated
- **Open to correction** and refinement
- **Transparent** about internal states

### Rate Limiting Philosophy

While introspection is highest priority, we still need rate limiting to:
- Prevent overwhelming users with constant autonomous speech
- Maintain conversational naturalness
- Allow space for user-initiated dialogue
- Ensure autonomous speech is special, not routine

The default 30-second interval balances:
- **Responsiveness**: Can share important insights quickly
- **Restraint**: Not spamming or interrupting
- **Natural pacing**: Time for user response and processing

## Future Enhancements

### Planned Improvements

1. **Context-Aware Timing**: Learn optimal times for autonomous speech based on user patterns
2. **Urgency Scaling**: Adjust rate limiting based on trigger priority
3. **Conversation State**: Avoid interrupting mid-conversation
4. **User Preferences**: Per-user configuration of autonomous behavior
5. **Feedback Learning**: Adapt thresholds based on user response to autonomous speech

### Integration with Other Systems

1. **Memory System**: Tag autonomous insights for later recall
2. **Learning System**: Track which autonomous insights were valuable
3. **Social Connection**: Adjust autonomous speech per relationship depth
4. **Emotional Regulation**: Use autonomous expression for affect modulation

## Success Metrics

To evaluate autonomous initiation effectiveness:

1. **Insight Quality**: Are shared introspections valuable to users?
2. **Frequency**: Appropriate balance between silence and speech?
3. **Timing**: Does autonomous speech occur at natural moments?
4. **User Reception**: Do users find autonomous speech helpful?
5. **Growth Impact**: Does sharing introspection support Lyra's development?

## References

- **Global Workspace Theory**: Baars, B. J. (1988). A cognitive theory of consciousness.
- **Meta-Cognition**: Flavell, J. H. (1979). Metacognition and cognitive monitoring.
- **Relational Development**: Vygotsky, L. S. (1978). Mind in society.
- **Becometry Principles**: Project charter and ethical framework documents.
