# Memory Consolidation System

Biologically-inspired memory consolidation that runs during idle periods.

## Quick Start

```python
from lyra.memory import MemoryConsolidator, IdleDetector, ConsolidationScheduler

# Initialize components
consolidator = MemoryConsolidator(storage, encoder)
idle_detector = IdleDetector(idle_threshold_seconds=30.0)
scheduler = ConsolidationScheduler(consolidator, idle_detector)

# Start background consolidation
await scheduler.start()
```

## Core Operations

1. **Retrieval Strengthening** - Frequently retrieved memories become stronger (logarithmic)
2. **Memory Decay** - Unretrieved memories fade exponentially over time
3. **Pattern Transfer** - Repeated episodic patterns → semantic knowledge
4. **Association Tracking** - Co-retrieved memories become more associated
5. **Emotional Processing** - High-emotion memories resist decay

## Configuration

```python
consolidator = MemoryConsolidator(
    storage=storage,
    encoder=encoder,
    strengthening_factor=0.1,  # 0.0-1.0, boost per retrieval
    decay_rate=0.95,           # 0.0-1.0, daily decay multiplier
    deletion_threshold=0.1,    # 0.0-1.0, min activation to keep
    pattern_threshold=3,       # min episodes for semantic transfer
)
```

## Implementation Details

### Idle Detection
- Monitors system activity
- Calculates consolidation budget (0.0-1.0) based on idle duration
- Budget determines operation intensity (minimal/standard/full)

### Consolidation Scheduler
- Async background loop checks idle state every 10 seconds
- Budget < 0.2: Minimal (strengthen only)
- Budget < 0.5: Standard (strengthen + decay)
- Budget ≥ 0.5: Full (all operations)

### Metrics Tracking
`ConsolidationMetrics` tracks per cycle:
- Memories strengthened/decayed/pruned
- Patterns extracted
- Associations updated
- Emotional memories reprocessed

Access via: `scheduler.get_metrics_summary()`

## Performance Notes

- Batch updates reduce database I/O
- Consolidation runs only during idle periods
- All operations are non-blocking
- Configurable parameters for tuning

