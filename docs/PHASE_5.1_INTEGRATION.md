# Phase 5.1: Full Cognitive Loop Integration

## Overview

This document describes the Phase 5.1 integration that bridges the cognitive core (Phases 1-4) with the legacy specialist system, enabling unified operation where both architectures work together cohesively.

## Architecture

```
USER INPUT
    ↓
┌─────────────────────────────────────────────────┐
│  COGNITIVE CORE (Continuous ~10 Hz Loop)        │
│                                                  │
│  ┌────────────────────────────────────────┐    │
│  │ LanguageInputParser (Gemma 12B)        │    │
│  │ - Parse natural language → Goals       │    │
│  └────────────────────────────────────────┘    │
│                ↓                                 │
│  ┌────────────────────────────────────────┐    │
│  │ GlobalWorkspace + Subsystems           │    │
│  │ - Attention, Perception, Affect        │    │
│  │ - Meta-Cognition, Memory Integration   │    │
│  │ - Goal-directed processing             │    │
│  └────────────────────────────────────────┘    │
│                ↓                                 │
│  ┌────────────────────────────────────────┐    │
│  │ ActionSubsystem                        │    │
│  │ - Generates SPEAK action when needed   │    │
│  └────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
    ↓ (SPEAK action triggers specialist system)
┌─────────────────────────────────────────────────┐
│  SPECIALIST SYSTEM (On-Demand Processing)       │
│                                                  │
│  RouterModel (Gemma 12B)                        │
│         ↓                                        │
│  Specialist Selection:                          │
│  - Pragmatist (Llama-3.3-Nemotron-49B)         │
│  - Philosopher (Jamba 52B)                      │
│  - Artist (Flux.1-schnell)                      │
│         ↓                                        │
│  Voice Synthesis (Llama 3 70B)                  │
└─────────────────────────────────────────────────┘
    ↓
USER OUTPUT
```

## Components

### 1. UnifiedCognitiveCore

**Location:** `emergence_core/lyra/unified_core.py`

The main orchestrator that integrates both systems. Key responsibilities:

- Initialize both cognitive core and specialist router
- Run continuous cognitive loop in background (~10 Hz)
- Intercept SPEAK actions and route to specialists when needed
- Feed specialist outputs back to cognitive core as percepts
- Maintain shared state (memory, emotion, context)

**Key Methods:**

- `initialize()`: Sets up both systems and starts the cognitive loop
- `process_user_input()`: Main entry point for user interactions
- `_requires_specialist_processing()`: Determines if specialist routing is needed
- `_call_specialist()`: Routes complex queries to specialist system
- `_feed_specialist_output()`: Feeds specialist responses back as percepts

**Configuration:**

```python
config = {
    "cognitive_core": {
        "cycle_rate_hz": 10,
        "attention_budget": 100
    },
    "specialist_router": {
        "development_mode": False
    },
    "integration": {
        "specialist_threshold": 0.7,  # Priority threshold for specialist routing
        "sync_interval": 1.0  # Seconds between state syncs
    }
}
```

### 2. Specialists Module

**Location:** `emergence_core/lyra/specialists.py`

Defines the specialist models:

#### PhilosopherSpecialist
- **Model:** Jamba 52B (ai21labs/Jamba-v0.1)
- **Purpose:** Ethical reflection and meta-cognition
- **Use Cases:** Philosophical questions, ethical dilemmas, self-reflection

#### PragmatistSpecialist
- **Model:** Llama-3.3-Nemotron-70B-Instruct
- **Purpose:** Tool use and practical reasoning
- **Use Cases:** API calls, web searches, practical problem-solving

#### ArtistSpecialist
- **Model:** Flux.1-schnell
- **Purpose:** Creative and visual generation
- **Use Cases:** Image generation, creative writing, artistic expression

#### VoiceSpecialist
- **Model:** Llama 3 70B
- **Purpose:** Final synthesis and personality
- **Use Cases:** All responses (final synthesis with persistent self-model)

#### PerceptionSpecialist
- **Model:** LLaVA (llava-hf/llava-v1.6-mistral-7b-hf)
- **Purpose:** Image understanding
- **Use Cases:** Analyzing images, visual question answering

**Usage:**

```python
from lyra.specialists import SpecialistFactory, SpecialistOutput

factory = SpecialistFactory(development_mode=True)
philosopher = factory.create_specialist('philosopher', base_dir)
result = await philosopher.process("What is consciousness?", context={})
```

### 3. SharedMemoryBridge

**Location:** `emergence_core/lyra/unified_core.py`

Bridges memory between cognitive core and specialist system. Both systems share:

- ChromaDB vector store
- Journal entries
- Episodic memories
- Protocols and identity

**Methods:**

- `sync_memories()`: Ensures both systems have access to the same memories

### 4. EmotionalStateBridge

**Location:** `emergence_core/lyra/unified_core.py`

Synchronizes emotional state bidirectionally:

- AffectSubsystem (cognitive core) maintains VAD (Valence-Arousal-Dominance) model
- Specialists use emotional context for generation
- Bidirectional sync: core → specialists → core

**Methods:**

- `sync_to_specialists(affect_state)`: Convert cognitive core emotion to specialist format
- `sync_from_specialists(specialist_emotion)`: Update cognitive core from specialist output

## Usage

### Running the Unified System

**Interactive Mode:**

```bash
cd emergence_core
python3 run_unified_system.py
```

This starts an interactive session where you can chat with Lyra using the full unified system.

**Programmatic Usage:**

```python
from lyra import UnifiedCognitiveCore
import asyncio

async def main():
    config = {
        "cognitive_core": {"cycle_rate_hz": 10},
        "integration": {"specialist_threshold": 0.7}
    }
    
    unified = UnifiedCognitiveCore(config=config)
    
    await unified.initialize(
        base_dir="/path/to/emergence_core",
        chroma_dir="/path/to/chroma_db",
        model_dir="/path/to/models"
    )
    
    response = await unified.process_user_input("Hello!")
    print(f"Lyra: {response}")
    
    await unified.stop()

asyncio.run(main())
```

### Development Mode

For testing without loading full models:

```python
config = {
    "specialist_router": {
        "development_mode": True
    }
}
```

In development mode, specialists return mock responses instead of loading actual models.

## Integration Flow

### 1. User Input Processing

```
User Input
    ↓
ConversationManager.chat()
    ↓
LanguageInputParser (cognitive core)
    ↓
Cognitive Loop (attention, workspace, actions)
    ↓
ActionSubsystem decides on SPEAK action
```

### 2. Specialist Routing Decision

```
Check SPEAK action priority
    ↓
If priority > threshold (0.7):
    Route to specialist system
Else:
    Return cognitive core response
```

### 3. Specialist Processing

```
Build context from cognitive state:
    - Emotional state (VAD)
    - Active goals
    - Recent memories
    ↓
AdaptiveRouter.process_message()
    ↓
RouterModel selects specialist
    ↓
Specialist processes with context
    ↓
Voice synthesizes final response
```

### 4. Feedback Loop

```
Specialist response
    ↓
Create Percept from response
    ↓
CognitiveCore.inject_input()
    ↓
Response enters cognitive loop
    ↓
Updates workspace state
```

## Testing

### Running Tests

**Minimal Tests (no heavy dependencies):**

```bash
pytest emergence_core/tests/test_unified_minimal.py -v
```

**Full Integration Tests (requires models):**

```bash
pytest emergence_core/tests/test_unified_integration.py -v
```

### Test Coverage

- **Initialization:** 3 tests
- **User input flow:** 2 tests
- **Specialist routing:** 3 tests
- **Memory sharing:** 3 tests
- **Emotional sync:** 4 tests
- **Context preservation:** 2 tests
- **Action system:** 2 tests
- **Specialist factory:** 6 tests
- **System shutdown:** 2 tests

**Total: 27 integration tests**

Plus 13 minimal tests for structure validation.

## Configuration Reference

### Cognitive Core Config

```python
"cognitive_core": {
    "cycle_rate_hz": 10,          # Cognitive loop frequency
    "attention_budget": 100,       # Attention allocation
    "max_queue_size": 100,         # Percept queue size
    "log_interval_cycles": 100     # Logging frequency
}
```

### Specialist Router Config

```python
"specialist_router": {
    "development_mode": False      # Use mock models if True
}
```

### Integration Config

```python
"integration": {
    "specialist_threshold": 0.7,   # Priority threshold for specialist routing
    "sync_interval": 1.0           # Seconds between state syncs
}
```

## Troubleshooting

### Issue: Models not loading

**Solution:** Ensure model files are in the correct directory:
```
model_cache/
    models/
        gemma_12b_router/
        llama_70b/
        etc.
```

### Issue: Import errors

**Solution:** Install dependencies:
```bash
pip install pydantic numpy scikit-learn chromadb
```

For full model support:
```bash
pip install torch transformers diffusers
```

### Issue: Specialist routing not triggering

**Solution:** Check the specialist threshold in config. Lower it to route more queries:
```python
"integration": {
    "specialist_threshold": 0.5  # Lower threshold
}
```

### Issue: Memory not shared

**Solution:** Ensure both systems point to the same ChromaDB directory:
```python
chroma_dir = "/same/path/for/both/systems"
```

## API Reference

### UnifiedCognitiveCore

```python
class UnifiedCognitiveCore:
    def __init__(self, config: Optional[Dict] = None)
    
    async def initialize(
        self,
        base_dir: str,
        chroma_dir: str,
        model_dir: str
    ) -> None
    
    async def process_user_input(self, user_text: str) -> str
    
    async def stop(self) -> None
```

### SpecialistFactory

```python
class SpecialistFactory:
    def __init__(self, development_mode: bool = False)
    
    def create_specialist(
        self,
        specialist_type: str,
        base_dir: Path
    ) -> BaseSpecialist
```

### SpecialistOutput

```python
@dataclass
class SpecialistOutput:
    content: str
    confidence: float = 1.0
    metadata: Dict[str, Any]
    specialist_type: str
```

## Future Enhancements

1. **Adaptive Routing:** Learn when to use specialists based on outcomes
2. **Parallel Processing:** Run multiple specialists simultaneously
3. **Specialist Caching:** Cache specialist outputs for similar queries
4. **Enhanced Feedback:** More sophisticated percept generation from specialist outputs
5. **Performance Metrics:** Track and optimize integration overhead
6. **Dynamic Thresholds:** Adjust specialist threshold based on context

## Related Documentation

- [Cognitive Core Documentation](./cognitive_core/README.md)
- [Specialist System Documentation](./specialists/README.md)
- [Memory Integration Guide](./memory/MEMORY_INTEGRATION.md)
- [Testing Guide](./testing/TESTING.md)

## Contributors

- Phase 5.1 Implementation: GitHub Copilot
- Architecture Design: Based on Lyra-Emergence project specifications
- Integration Pattern: Inspired by Global Workspace Theory

## License

See main project LICENSE file.
