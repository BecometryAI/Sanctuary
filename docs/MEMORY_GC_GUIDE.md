# Memory Garbage Collection Guide

## Overview

The Memory Garbage Collection (GC) system implements periodic cleanup of low-significance memories to prevent unbounded growth while maintaining system performance over long-term operation. This guide explains how to configure, use, and troubleshoot the memory GC system.

## Table of Contents

1. [Collection Strategies](#collection-strategies)
2. [Configuration Guide](#configuration-guide)
3. [CLI Command Reference](#cli-command-reference)
4. [Programmatic Usage](#programmatic-usage)
5. [Troubleshooting](#troubleshooting)
6. [Performance Tuning](#performance-tuning)
7. [Safety Mechanisms](#safety-mechanisms)

---

## Collection Strategies

The memory GC employs multiple strategies to intelligently manage memory:

### 1. Significance-Based Removal

Removes memories with significance scores below a configurable threshold (default: 0.1).

**How it works:**
- Evaluates each memory's significance score
- Marks memories below threshold for removal
- Respects protected tags and recent memories

**Configuration:**
```python
"significance_threshold": 0.1  # Remove memories below this score
```

### 2. Age-Based Decay

Applies time-based decay to significance scores, making older low-significance memories naturally eligible for removal.

**Formula:**
```
new_significance = old_significance × exp(-decay_rate × age_days)
```

**Configuration:**
```python
"decay_rate_per_day": 0.01  # 1% decay per day
```

### 3. Capacity-Based Pruning

Enforces a maximum memory capacity, removing lowest-significance memories when the limit is exceeded.

**Configuration:**
```python
"max_memory_capacity": 10000  # Maximum number of memories
```

---

## Configuration Guide

### Default Configuration

```python
"memory_gc": {
    "enabled": True,
    "collection_interval": 3600.0,  # 1 hour (seconds)
    "significance_threshold": 0.1,
    "decay_rate_per_day": 0.01,
    "max_memory_capacity": 10000,
    "preserve_tags": ["important", "pinned", "charter_related"],
    "recent_memory_protection_hours": 24,
    "max_removal_per_run": 100
}
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `enabled` | Enable/disable automatic GC | `True` |
| `collection_interval` | Seconds between collections | `3600.0` |
| `significance_threshold` | Minimum significance to keep | `0.1` |
| `decay_rate_per_day` | Daily decay rate | `0.01` |
| `max_memory_capacity` | Maximum total memories | `10000` |
| `preserve_tags` | Tags that protect from removal | `["important", "pinned", "charter_related"]` |
| `recent_memory_protection_hours` | Hours to protect recent memories | `24` |
| `max_removal_per_run` | Max removals per GC run | `100` |

---

## CLI Command Reference

### View Memory Health Statistics

```bash
memory stats
```

**Output:**
```
📊 Memory System Health:
   Total memories: 1234
   Total size: 5.67 MB
   Average significance: 5.23
   Needs collection: No
```

### Manual Garbage Collection

**Basic collection:**
```bash
memory gc
```

**Collection with custom threshold:**
```bash
memory gc --threshold 0.2
```

**Dry run (preview without removing):**
```bash
memory gc --dry-run
```

### Enable/Disable Automatic GC

**Enable:**
```bash
memory autogc on
```

**Disable:**
```bash
memory autogc off
```

---

## Programmatic Usage

### Using with MemoryManager

```python
from lyra.memory_manager import MemoryManager

manager = MemoryManager(
    base_dir=Path("./data/memories"),
    chroma_dir=Path("./data/chroma"),
    gc_config={"significance_threshold": 0.15}
)

# Enable automatic GC
manager.enable_auto_gc(interval=3600.0)

# Run manual GC
stats = await manager.run_gc(threshold=0.2)

# Get memory health
health = await manager.get_memory_health()

# Disable automatic GC
manager.disable_auto_gc()
```

---

## Troubleshooting

### Problem: GC Removing Important Memories

**Solution:**
1. Add protected tags: `tags=["important"]`
2. Increase significance scores
3. Adjust `preserve_tags` in configuration

### Problem: GC Not Running

**Check:**
1. Is GC enabled in config?
2. Is automatic collection scheduled?
3. Check logs for errors

### Problem: Memory Growing Despite GC

**Solutions:**
1. Lower significance threshold
2. Increase max capacity
3. Adjust decay rate

---

## Performance Tuning

### For High-Volume Systems

```python
"memory_gc": {
    "collection_interval": 1800.0,     # 30 minutes
    "significance_threshold": 0.15,
    "max_removal_per_run": 200,
    "max_memory_capacity": 50000
}
```

### For Long-Term Stability

```python
"memory_gc": {
    "collection_interval": 3600.0,
    "significance_threshold": 0.1,
    "decay_rate_per_day": 0.02,
    "max_memory_capacity": 10000
}
```

---

## Safety Mechanisms

1. **Protected Memory Tags**: Memories with `important`, `pinned`, or `charter_related` tags are never removed
2. **Recent Memory Protection**: Memories < 24 hours old are protected
3. **Removal Rate Limiting**: Maximum 100 removals per run
4. **Dry-Run Mode**: Preview removals without executing
5. **Collection History**: Track all GC operations
6. **Graceful Error Handling**: Failures never crash the system

---

## Best Practices

1. Start with default settings
2. Monitor weekly with `memory stats`
3. Tag important memories
4. Tune parameters gradually
5. Use dry-run for testing
6. Plan for 50% capacity headroom

---

**Last Updated:** January 2026  
**Version:** 1.0.0
