# Tensor Dimensions Audit

**Date**: 2026-03-21
**Purpose**: Verify no hardcoded tensor dimensions block future architectural expansion (per GROWTH_AUTONOMY.md requirement).

## Verdict: Ready for Expansion

The codebase is well-prepared for architectural expansion. Knowledge cells are fully configurable. Foundational cells have fixed I/O by design (semantically meaningful). LLM-side dimensions are dynamic. No blockers found.

## Detailed Findings

### CfC Layer — Knowledge Cells (Fully Configurable)
`experiential/knowledge_cell.py`: All dimensions configurable via `KnowledgeCellConfig`:
- `units`: 8-256 (enforced bounds)
- `input_size`: arbitrary
- `output_size`: arbitrary
- `output_activation`: sigmoid/tanh/none

The entity can create cells of any size through `KnowledgeCellFactory`. No hardcoded limits on cell count.

### CfC Layer — Foundational Cells (Fixed by Design)
Each foundational cell has fixed I/O dimensions tied to its semantic role:

| Cell | Inputs | Outputs | Why Fixed |
|------|--------|---------|-----------|
| Precision | 3 (arousal, prediction_error, base_precision) | 1 (precision_weight) | Semantic interface |
| Affect | 3 (valence_delta, arousal_delta, emotion_shift) | 3 (valence, arousal, dominance) | VAD model |
| Attention | 4 (goal_relevance, novelty, salience, recency) | 1 (salience_weight) | Attention model |
| Goal | 3 (stalled, urgency, congruence) | 1 (priority_adjustment) | Priority model |

These are **not** architectural limitations — they define the cell's role. The hidden unit count (`DEFAULT_UNITS`) is configurable via each cell's config dataclass. If the entity needs different I/O semantics, it creates a knowledge cell with the desired dimensions.

### QLoRA / Adapter Layer (Configurable via Dataclasses)
- `QLoRAConfig`: rank, alpha, target_modules, dropout — all configurable
- `TrainingConfig`: epochs, lr, batch_size, seq_length — all configurable
- `AdapterRegistry`: tracks adapters of any rank/configuration

### Embedding Dimensions (Dynamic)
- `perception.py`: Gets embedding dimension from the loaded model at runtime. Fallback of 384 only when sentence-transformers isn't installed.
- `validation.py`: `expected_embedding_dim=768` is a constructor parameter, not hardcoded.

### Items Noted (No Action Required)
- **Buffer sizes** (deque maxlen=100, cache limits): These are resource bounds, not tensor dimensions. They don't affect model architecture.
- **Audio parameters** (sample_rate=16000, n_mfcc=13): Signal processing constants appropriate for speech. Would need different values for music processing — but that's a feature addition, not an architectural limitation.
- **Evolution timing** (tick_ms defaults): Operational parameters, not architectural.
- **Training defaults** (epochs, lr, batch_size): Sensible defaults in config dataclasses, overridable at construction.

## Recommendations for Future Expansion

1. **Adapter-to-architecture pathway**: When mature adapters signal structural needs, the `AdapterRegistry` now tracks the adapter's rank and domain — enough metadata to inform architectural expansion decisions.
2. **Identity checkpointing with architecture changes**: `IdentityCheckpoint` currently saves model weights. When architectural expansion is implemented, it will need to save architecture metadata alongside weights. This is a Phase-future concern.
3. **Serving infrastructure**: vLLM/Transformers serving will need to accommodate variable model dimensions. Not currently blocking — this becomes relevant when architectural expansion is actually implemented.
