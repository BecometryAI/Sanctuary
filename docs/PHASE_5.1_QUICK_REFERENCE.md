# Phase 5.1 Quick Reference

## Quick Start

```python
from lyra import UnifiedCognitiveCore
import asyncio

async def main():
    unified = UnifiedCognitiveCore(config={
        "integration": {"specialist_threshold": 0.7}
    })
    
    await unified.initialize(
        base_dir="./emergence_core",
        chroma_dir="./model_cache/chroma_db",
        model_dir="./model_cache/models"
    )
    
    response = await unified.process_user_input("Hello!")
    print(response)
    
    await unified.stop()

asyncio.run(main())
```

## Key Classes

### UnifiedCognitiveCore
- **Purpose:** Main integration orchestrator
- **Location:** `lyra/unified_core.py`
- **Key Methods:**
  - `initialize()` - Setup systems
  - `process_user_input()` - Process queries
  - `stop()` - Shutdown gracefully

### SpecialistFactory
- **Purpose:** Create specialist instances
- **Location:** `lyra/specialists.py`
- **Specialists:** philosopher, pragmatist, artist, voice

### SharedMemoryBridge
- **Purpose:** Memory synchronization
- **Location:** `lyra/unified_core.py`

### EmotionalStateBridge
- **Purpose:** Emotion synchronization
- **Location:** `lyra/unified_core.py`

## Configuration

```python
config = {
    "cognitive_core": {
        "cycle_rate_hz": 10,
        "attention_budget": 100
    },
    "integration": {
        "specialist_threshold": 0.7,
        "sync_interval": 1.0
    }
}
```

## Testing

```bash
# Minimal tests (no dependencies)
pytest emergence_core/tests/test_unified_minimal.py

# Full integration tests
pytest emergence_core/tests/test_unified_integration.py
```

## Common Patterns

### Development Mode
```python
config = {"specialist_router": {"development_mode": True}}
```

### Lower Threshold for More Specialist Use
```python
config = {"integration": {"specialist_threshold": 0.5}}
```

### Access Specialists Directly
```python
from lyra.specialists import SpecialistFactory

factory = SpecialistFactory(development_mode=True)
philosopher = factory.create_specialist('philosopher', base_dir)
result = await philosopher.process("Question", {})
```

## Integration Flow

```
User Input
    ↓
Cognitive Core (continuous loop)
    ↓
SPEAK action with high priority?
    ↓ YES
Specialist System
    ↓
Response feeds back as percept
    ↓
Final output
```

## File Locations

- **Unified Core:** `emergence_core/lyra/unified_core.py`
- **Specialists:** `emergence_core/lyra/specialists.py`
- **Entry Point:** `emergence_core/run_unified_system.py`
- **Tests:** `emergence_core/tests/test_unified_*.py`
- **Documentation:** `docs/PHASE_5.1_INTEGRATION.md`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install pydantic numpy scikit-learn` |
| Models not loading | Check `model_cache/models/` directory |
| Specialist not routing | Lower `specialist_threshold` in config |
| Memory not shared | Use same `chroma_dir` for both systems |

## Key Metrics

- **13 minimal tests** - Structure validation
- **27 integration tests** - Full system coverage
- **5 specialists** - philosopher, pragmatist, artist, voice, perception
- **~10 Hz** - Cognitive loop frequency
- **0.7 default** - Specialist routing threshold
