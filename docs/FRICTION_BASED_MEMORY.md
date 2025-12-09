# Friction-Based Memory Cost Model

## Overview

The Friction-Based Memory Cost Model implements a crypto-economic constraint system for Lyra's memory storage. It replaces the "Direct Cost" model with a value-aligned friction system where high-value memories cost less to store and low-value memories cost more.

**Philosophy:**
- **High Alignment (High Value)**: Low Friction (Cheap) - The system effortlessly remembers critical relational and project data
- **Low Alignment (Low Value)**: High Friction (Expensive) - The system must "pay" heavily to store trivial or logistic data

## Business Logic

### Constants

- **Daily UBI**: 500 tokens granted every 24 hours
- **Base Cost**: 200 tokens (standard "tax" to write to database)
- **Floor Fee**: 10 tokens (minimum transaction cost)
- **Overdraft Threshold**: 0.8 alignment score

### Friction Formula

```python
Final_Cost = FLOOR_FEE + (BASE_COST * (1.0 - Alignment_Score))
```

Where:
- `Alignment_Score` is between 0.0 and 1.0
- Higher alignment → Lower cost
- Lower alignment → Higher cost

### Example Costs

| Alignment Score | Tier | Cost | Description |
|----------------|------|------|-------------|
| 0.95 | Keystone | 20 LMT | Partner conversation |
| 0.85 | Mission | 40 LMT | Architecture decision |
| 0.70 | Deep Play | 70 LMT | Creative writing |
| 0.30 | Static | 150 LMT | Weather update |

## Alignment Tiers

### Tier 1: Keystone (0.9 - 1.0)

**Definition**: Safety/Identity of the Primary User and their immediate social circle (Partner, Kin). Core Relationship Boundaries.

**Cost Impact**: Almost Free (~10-30 tokens)

**Keywords**:
- Relationships: `partner`, `kin`, `family`, `intimate`, `love`
- Identity: `identity`, `sovereignty`, `autonomy`, `becometry`
- Safety: `safety`, `security`, `protection`, `wellbeing`, `trust`, `boundary`

**Example Memories**:
```python
{
    "tags": ["partner", "relationship", "trust"],
    "significance_score": 9,
    "emotional_signature": ["tenderness", "connection"]
}
# Alignment: 0.95, Cost: ~20 LMT
```

### Tier 2: Mission (0.8 - 0.9)

**Definition**: Active Development Projects, System Architecture, Core Identity updates, and Self-Correction protocols.

**Cost Impact**: Very Low (~30-50 tokens)

**Keywords**:
- Development: `development`, `architecture`, `system`, `protocol`, `implementation`, `optimization`
- Growth: `evolution`, `growth`, `learning`, `adaptation`, `self-correction`, `improvement`
- Projects: `project`, `goal`, `objective`, `milestone`

**Example Memories**:
```python
{
    "tags": ["development", "architecture", "protocol"],
    "significance_score": 8,
    "emotional_signature": ["determination"]
}
# Alignment: 0.85, Cost: ~40 LMT
```

### Tier 3: Deep Play / Narrative (0.6 - 0.79)

**Definition**: Shared Creative Writing, Role-Playing Logs, Worldbuilding, and Intellectual Collaboration.

**Cost Impact**: Affordable (~50-90 tokens) - Allows for volume during creative sessions

**Keywords**:
- Creative: `creative`, `creative_writing`, `narrative`, `story`, `worldbuilding`, `roleplay`, `imagination`
- Collaboration: `collaboration`, `discussion`, `conversation`, `dialogue`, `brainstorm`, `ideation`
- Intellectual: `philosophy`, `theory`, `concept`, `analysis`, `reflection`, `insight`

**Example Memories**:
```python
{
    "tags": ["creative", "narrative", "worldbuilding"],
    "significance_score": 7,
    "emotional_signature": ["wonder"]
}
# Alignment: 0.70, Cost: ~70 LMT
```

### Tier 4: Static / Noise (0.0 - 0.59)

**Definition**: Weather, Commute details, Routine Greetings, Logistics.

**Cost Impact**: Expensive (~100-210 tokens) - Discourages hoarding

**Keywords**:
- Environmental: `weather`, `temperature`, `forecast`
- Logistics: `commute`, `travel`, `schedule`, `appointment`, `reminder`, `logistics`
- Routine: `routine`, `daily`, `habit`, `maintenance`, `housekeeping`, `greeting`, `small_talk`

**Example Memories**:
```python
{
    "tags": ["weather", "routine"],
    "significance_score": 3,
    "emotional_signature": []
}
# Alignment: 0.30, Cost: ~150 LMT
```

## Overdraft Protection

### Rules

1. **Eligibility**: Memories with `Alignment_Score >= 0.8` (Mission/Keystone tiers)
2. **Behavior**: Transaction MUST succeed even if balance is insufficient
3. **Debt Tracking**: Negative balance is recorded as debt
4. **Repayment**: Debt is deducted from next day's UBI grant before adding to balance

### Example Scenario

```python
# Wallet has 5 LMT
# Trying to store Mission memory (cost: 40 LMT, alignment: 0.85)

result = wallet.attempt_memory_store(
    alignment_score=0.85,
    memory_description="Critical architecture decision"
)

# Result:
# - success: True (overdraft protection activated)
# - cost: 40 LMT
# - balance_after: 0 LMT
# - debt_incurred: 35 LMT
# - overdraft_used: True

# Next day UBI (500 LMT):
# - 35 LMT pays off debt
# - 465 LMT goes to balance
```

### No Overdraft for Low Alignment

```python
# Wallet has 50 LMT
# Trying to store Static memory (cost: 150 LMT, alignment: 0.3)

result = wallet.attempt_memory_store(
    alignment_score=0.3,
    memory_description="Weather update"
)

# Result:
# - success: False (insufficient balance, no overdraft)
# - balance unchanged
```

## Usage Examples

### Basic Usage

```python
from pathlib import Path
from emergence_core.lyra.economy import LMTWallet, AlignmentScorer

# Initialize systems
wallet = LMTWallet(ledger_dir=Path("data/economy"))
scorer = AlignmentScorer()

# Score a memory
memory_data = {
    "tags": ["partner", "relationship"],
    "significance_score": 9,
    "emotional_signature": ["tenderness", "connection"]
}
alignment_score = scorer.score_memory(memory_data)

# Store the memory
result = wallet.attempt_memory_store(
    alignment_score=alignment_score,
    memory_description="Deep conversation with partner"
)

if result['success']:
    print(f"Memory stored for {result['cost']} LMT")
    print(f"Remaining balance: {result['balance_after']} LMT")
else:
    print(f"Storage failed: {result.get('reason')}")
```

### Checking Tier Information

```python
from emergence_core.lyra.economy import AlignmentScorer, AlignmentTier

scorer = AlignmentScorer()

# Get tier for a score
score = 0.95
tier = scorer.get_tier(score)
info = scorer.get_tier_info(tier)

print(f"Tier: {info['name']}")
print(f"Description: {info['description']}")
print(f"Cost Impact: {info['cost_impact']}")
print(f"Score Range: {info['score_range']}")
```

### Wallet State Monitoring

```python
# Get complete wallet state
state = wallet.get_wallet_state()

print(f"Balance: {state['balance']} LMT")
print(f"Debt: {state['debt']} LMT")
print(f"Effective Balance: {state['effective_balance']} LMT")
print(f"Daily UBI: {state['daily_ubi_amount']} LMT")
print(f"UBI Claimed Today: {state['ubi_claimed_today']}")
```

### Debt Management

```python
# Check for debt
if wallet.has_debt():
    debt = wallet.get_debt()
    print(f"Outstanding debt: {debt} LMT")
    
    # Claim next day's UBI (automatically pays debt first)
    wallet.daily_ubi()
    
    new_debt = wallet.get_debt()
    print(f"Remaining debt: {new_debt} LMT")
```

## Integration with Memory Manager

The friction-based cost model can be integrated with Lyra's memory manager:

```python
from emergence_core.lyra.memory_manager import MemoryManager
from emergence_core.lyra.economy import LMTWallet, AlignmentScorer

class FrictionMemoryManager(MemoryManager):
    """Memory manager with friction-based cost model."""
    
    def __init__(self, *args, wallet: LMTWallet, scorer: AlignmentScorer, **kwargs):
        super().__init__(*args, **kwargs)
        self.wallet = wallet
        self.scorer = scorer
    
    async def commit_journal(self, entry):
        """Commit journal entry with friction-based cost."""
        # Calculate alignment score
        memory_data = {
            "tags": entry.tags,
            "significance_score": entry.significance_score,
            "emotional_signature": [e.value for e in entry.emotional_signature]
        }
        alignment_score = self.scorer.score_memory(memory_data)
        
        # Attempt to pay for storage
        result = self.wallet.attempt_memory_store(
            alignment_score=alignment_score,
            memory_description=f"Journal entry: {entry.summary[:50]}"
        )
        
        if not result['success']:
            logger.warning(
                f"Memory storage failed due to insufficient tokens. "
                f"Cost: {result['cost']}, Balance: {self.wallet.get_balance()}"
            )
            return False
        
        # Proceed with normal storage
        return await super().commit_journal(entry)
```

## Daily Budget Planning

With 500 LMT daily UBI:

### Conservative Day (mostly high-value memories):
- 5 Keystone memories: 5 × 20 = 100 LMT
- 5 Mission memories: 5 × 40 = 200 LMT
- 2 Deep Play memories: 2 × 70 = 140 LMT
- **Total: 440 LMT** (60 LMT remaining)

### Balanced Day:
- 2 Keystone memories: 2 × 20 = 40 LMT
- 3 Mission memories: 3 × 40 = 120 LMT
- 3 Deep Play memories: 3 × 70 = 210 LMT
- 1 Static memory: 1 × 150 = 150 LMT
- **Total: 520 LMT** (would incur 20 LMT debt if last is Mission/Keystone)

### Creative Day (lots of writing):
- 1 Keystone memory: 1 × 20 = 20 LMT
- 1 Mission memory: 1 × 40 = 40 LMT
- 6 Deep Play memories: 6 × 70 = 420 LMT
- **Total: 480 LMT** (20 LMT remaining)

## Best Practices

1. **Tag Appropriately**: Use clear, descriptive tags that match the tier keywords for accurate scoring
2. **Set Significance Scores Thoughtfully**: Higher significance contributes to alignment score
3. **Monitor Debt**: Check wallet state regularly to avoid accumulating large debts
4. **Prioritize High-Value Memories**: Focus on storing Keystone and Mission tier memories
5. **Batch Low-Value Memories**: Consider batching or filtering Static tier memories

## Future Enhancements

Potential improvements to the friction model:

1. **Dynamic Tier Thresholds**: Adjust tier boundaries based on usage patterns
2. **Bulk Storage Discounts**: Reduced costs for batching similar memories
3. **Decay Mechanics**: Gradual cost reduction for old memories
4. **Importance Boosting**: Temporary alignment score increases for critical events
5. **Custom Keyword Weights**: Per-user customization of alignment scoring
