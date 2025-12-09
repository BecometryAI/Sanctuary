# Friction-Based Memory Cost Examples

This directory contains example scripts demonstrating the friction-based memory cost model.

## Available Examples

### `friction_memory_demo.py`

A comprehensive demonstration of the friction-based memory cost model integrated with Lyra's memory manager.

**Features:**
- Shows all four alignment tiers (Keystone, Mission, Deep Play, Static)
- Demonstrates cost calculations for each tier
- Shows overdraft protection for high-alignment memories
- Demonstrates debt tracking and repayment

**Requirements:**
- All dependencies from `pyproject.toml`
- Chromadb
- Pydantic

**Usage:**
```bash
cd /path/to/Lyra-Emergence
python examples/friction_memory_demo.py
```

**Expected Output:**
```
FRICTION-BASED MEMORY COST MODEL DEMO
============================================================

Initializing system...
✓ Wallet initialized with 500 LMT
✓ Daily UBI: 500 LMT

EXAMPLE 1: Keystone Tier Memory
============================================================
Memory Alignment Analysis
============================================================
Summary: Deep conversation with partner about future, boundaries...
Tags: partner, relationship, trust, boundary
Significance: 9/10
Emotions: tenderness, connection
Alignment Score: 0.968
Tier: KEYSTONE

Storage Transaction
============================================================
Cost: 20 LMT
Success: True
Balance After: 480 LMT

✓ Keystone memory stored (Cost: 20 LMT)

[... continues with other tiers ...]
```

## Quick Testing Without Full Integration

If you want to test the friction model without the full memory manager integration, use the validation script from the project root:

```bash
cd /path/to/Lyra-Emergence
python << 'EOF'
import sys
import os
sys.path.insert(0, os.getcwd())

from emergence_core.lyra.economy import AlignmentScorer, LMTWallet
from pathlib import Path
import tempfile

# Create scorer and wallet
scorer = AlignmentScorer()
temp_dir = tempfile.mkdtemp()
wallet = LMTWallet(ledger_dir=Path(temp_dir), daily_ubi_amount=500)

# Test different memory types
memories = [
    {
        "name": "Partner conversation",
        "data": {"tags": ["partner"], "significance_score": 9, "emotional_signature": ["tenderness"]},
        "expected_tier": "keystone"
    },
    {
        "name": "Code architecture",
        "data": {"tags": ["architecture"], "significance_score": 8, "emotional_signature": ["determination"]},
        "expected_tier": "mission"
    },
    {
        "name": "Creative writing",
        "data": {"tags": ["creative"], "significance_score": 7, "emotional_signature": ["wonder"]},
        "expected_tier": "deep_play"
    },
    {
        "name": "Weather update",
        "data": {"tags": ["weather"], "significance_score": 2, "emotional_signature": []},
        "expected_tier": "static"
    }
]

print("Testing Friction-Based Memory Costs:\n")
for memory in memories:
    score = scorer.score_memory(memory["data"])
    cost = wallet.calculate_friction_cost(score)
    tier = scorer.get_tier(score)
    
    print(f"{memory['name']:20s} | Tier: {tier.value:10s} | Score: {score:.3f} | Cost: {cost:3d} LMT")

EOF
```

## Understanding the Cost Model

The friction formula is:
```
Final_Cost = FLOOR_FEE + (BASE_COST * (1.0 - Alignment_Score))
```

Where:
- `FLOOR_FEE = 10` (minimum cost)
- `BASE_COST = 200` (standard tax)
- `Alignment_Score` ranges from 0.0 to 1.0

### Cost Examples:

| Alignment | Tier | Formula | Cost |
|-----------|------|---------|------|
| 0.95 | Keystone | 10 + (200 × 0.05) | 20 LMT |
| 0.85 | Mission | 10 + (200 × 0.15) | 40 LMT |
| 0.70 | Deep Play | 10 + (200 × 0.30) | 70 LMT |
| 0.30 | Static | 10 + (200 × 0.70) | 150 LMT |
| 0.00 | Static | 10 + (200 × 1.0) | 210 LMT |

## See Also

- `../docs/FRICTION_BASED_MEMORY.md` - Complete documentation
- `../emergence_core/tests/test_friction_based_memory.py` - Comprehensive test suite
- `../emergence_core/lyra/economy/` - Implementation modules
