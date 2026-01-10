# Cue-Dependent Memory Retrieval - Quick Reference

## Overview
Implements cognitively realistic memory retrieval where what you remember depends on current context (workspace state), not just semantic similarity.

## Key Components

### 1. CueDependentRetrieval Class
Main implementation in `emergence_core/lyra/memory/retrieval.py`.

**Activation Formula:**
```
activation = similarity * 0.5 + recency * 0.2 + emotional_congruence * 0.3
```

**Key Methods:**
- `retrieve(workspace_state, limit=5)` - Main retrieval interface
- `_compute_activations()` - Calculate activation scores
- `_spread_activation()` - Spread to associated memories
- `_competitive_retrieval()` - Competition with interference
- `_strengthen_retrieved()` - Update retrieval metadata

### 2. Emotional Congruence
PAD-based (Pleasure-Arousal-Dominance) in `emotional_weighting.py`:
```python
distance = sqrt((v1-v2)² + (a1-a2)² + (d1-d2)²)
congruence = 1 - (distance / max_distance)
```

### 3. Storage Extensions
In `storage.py`:
- `update_retrieval_metadata()` - Track count and last_accessed
- `get_memory_associations()` - Get associated memories
- `add_memory_association()` - Create associations

## Configuration

**Tunable Constants (in CueDependentRetrieval):**
```python
SIMILARITY_WEIGHT = 0.5      # Semantic relevance
RECENCY_WEIGHT = 0.2         # Temporal dynamics
EMOTIONAL_WEIGHT = 0.3       # Emotional congruence
RECENCY_DECAY_RATE = 0.01    # λ (half-life ~69h)
```

**Constructor Parameters:**
- `retrieval_threshold=0.3` - Minimum activation
- `inhibition_strength=0.4` - Competition strength
- `strengthening_factor=0.05` - Learning rate
- `spread_factor=0.3` - Spreading intensity

## Usage

```python
# Initialize with storage and emotional weighting
retriever = MemoryRetriever(storage, vector_db, emotional_weighting)

# Retrieve with workspace context
workspace_state = {
    "goals": [{"description": "understand emotions"}],
    "emotions": {"valence": 0.7, "arousal": 0.6, "dominance": 0.7},
    "percepts": {"p1": {"raw": "user discussing feelings"}}
}
memories = retriever.retrieve_with_cues(workspace_state, limit=5)
```

## Metrics

Track retrieval dynamics via `get_metrics()`:
- `total_retrievals` - Number of retrieval operations
- `avg_cue_similarity` - Average embedding similarity
- `spreading_activations` - Spreading events count
- `interference_events` - Competition events count
- `strengthening_events` - Strengthening count

## Testing

Run tests: `python -m pytest emergence_core/lyra/tests/test_cue_dependent_retrieval.py`

Demo script: `python demo_cue_dependent_retrieval.py`

## Key Improvements from Refactoring

1. **Efficiency**: Eliminated code duplication with `_get_collection()` helper
2. **Readability**: Named constants replace magic numbers
3. **Simplicity**: Large methods broken into focused helpers
4. **Robustness**: Added input validation
5. **Maintainability**: Modular design, DRY principle

## References

Based on cognitive science:
- Cue-dependent memory (Tulving & Thomson, 1973)
- Spreading activation (Collins & Loftus, 1975)
- Mood-congruent memory (Bower, 1981)
- Retrieval competition (Anderson & Neely, 1996)
