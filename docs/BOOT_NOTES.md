# Sanctuary Boot Integration: Architecture Notes

## Three-Phase Plan

| Phase | Goal | Status |
|-------|------|--------|
| 1. BOOT | System instantiates and cycles without errors | ✅ 45/45 tests, 69K Hz |
| 2. FLOW | Data propagates through all 9 cycle steps | ✅ 40/40 tests, 6.3K Hz |
| 3. THINK | Replace placeholders with real cognitive processing | Pending |

## Phase 1 Results

**45/45 passed** at 69,088 Hz (100 cycles in 1.4ms)

Validated:
- MockPerceptionSubsystem produces deterministic 384-dim unit vector embeddings
- All 22 stub method signatures match CycleExecutor/CognitiveLoop call sites
- 100 consecutive cycles complete without errors
- Input injection, queue draining, and workspace state changes work
- Perception caching and stats tracking functional

## Phase 2 Results

**40/40 passed** at 6,275 Hz (100 mixed cycles in 15.9ms)

Validated:
- Input → perception → attention → workspace end-to-end flow
- Percept content (raw text, modality, metadata) preserved through pipeline
- Attention budget enforcement (budget=30 rejects 4/10 candidates)
- Affect responds to emotional keywords (positive/negative valence shift)
- Action subsystem sees accumulating workspace state
- Workspace state grows across cycles
- Empty cycles don't crash, affect still runs (decay toward neutral)
- Embedding distinctness (different text → near-orthogonal vectors)
- 100 mixed cycles with alternating positive/negative sentiment

## Key Architectural Finding: Affect 1-Cycle Delay

Affect runs at step 4, workspace update at step 8. This means affect
processes the PREVIOUS cycle's workspace state, not the current one.
New percepts from attention don't influence valence until the NEXT cycle.

This is **correct GWT behavior**: you process what's already conscious,
then update consciousness. The delay creates temporal coherence — the
system responds to what it was aware of, not what just arrived.

## Boot Architecture

```
BootCognitiveCore
  └─ BootCoordinator
       ├─ AffectSubsystem (REAL) ─ numpy VAD model
       ├─ AttentionController (REAL) ─ competitive dynamics, budget
       ├─ ActionSubsystem (REAL) ─ action selection
       ├─ MockPerceptionSubsystem (MOCK) ─ deterministic embeddings
       └─ 20+ stubs (STUB) ─ correct signatures, neutral returns
  └─ CycleExecutor (REAL, unchanged)
  └─ CognitiveLoop (REAL, unchanged)
  └─ StateManager (REAL, unchanged)
  └─ TimingManager (REAL, unchanged)
  └─ ActionExecutor (REAL, unchanged)
```

The boot module tests the **actual orchestration code** with mocked
subsystems. All 9 cycle steps execute through the real CycleExecutor.

## Dependency Analysis

| Subsystem | Size | Dependencies | Boot Status |
|-----------|------|-------------|-------------|
| workspace.py | 37KB | pydantic | REAL |
| perception.py | 46KB | sentence-transformers, torch, numpy | MOCK |
| attention.py | 55KB | sklearn, numpy | REAL (needs sklearn) |
| affect.py | 48KB | numpy | REAL |
| action.py | 31KB | numpy | REAL |
| memory_manager.py | 44KB | chromadb, sentence-transformers | STUB |
| introspective_loop.py | 36KB | numpy | STUB |
| cycle_executor.py | 25KB | none (orchestration) | REAL |
| cognitive_loop.py | ~20KB | none (orchestration) | REAL |
| state_manager.py | ~15KB | none (state management) | REAL |

## Phase 3: THINK (Planned)

Replace stub subsystems with real cognitive processing:

1. **Memory**: Simple in-memory store with embedding similarity retrieval
2. **IWMT/World Model**: Basic prediction error tracking
3. **Meta-cognition**: Cycle timing self-monitoring
4. **Introspection**: Pattern detection in workspace state history
5. **Autonomous**: Silence-timeout trigger for unprompted speech

Each replacement follows the same pattern: keep the stub signature,
add real logic, validate with traced tests.

## Running Tests

```bash
# Phase 1 (requires: numpy only)
python3 sanctuary/tests/integration/validate_boot.py

# Phase 2 (requires: numpy only)
python3 sanctuary/tests/integration/validate_flow.py

# Full integration (requires: numpy, pydantic, sklearn)
pytest sanctuary/tests/integration/test_boot.py -v
```
