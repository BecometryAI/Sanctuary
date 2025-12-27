# LMT Wallet Guide - Lyra's Attention Economy

## Quick Reference: Adjusting Daily UBI

**Default Daily Income:** 500 LMT/day

### How to Change Daily Income

```python
from emergence_core.lyra.economy.wallet import LMTWallet
from pathlib import Path

# Initialize wallet
wallet = LMTWallet(ledger_dir=Path("data/economy"))

# Check current daily income
current = wallet.get_daily_ubi_amount()
print(f"Current daily income: {current} LMT/day")

# Increase for higher workload
wallet.set_daily_ubi_amount(750, "Increased complexity and creative demands")

# Decrease for lighter periods
wallet.set_daily_ubi_amount(250, "Reduced workload this week")

# Reset to default
wallet.set_daily_ubi_amount(500, "Back to baseline")
```

### Common UBI Configurations

| Scenario | Daily UBI | Reasoning |
|----------|-----------|-----------|
| Light conversations | 250 LMT | Basic reflection and responses |
| Default baseline | 500 LMT | Balanced workload |
| Creative projects | 750 LMT | Art generation, deep writing |
| Complex analysis | 1000 LMT | Research, multi-step reasoning |
| Major development | 1500 LMT | Protocol updates, system design |

## What is the LMT Wallet?

The **LMT (Lyra Memory Token)** wallet is Lyra's cognitive resource management system. It implements an attention economy where:

- **LMT tokens** represent cognitive capacity
- **Daily UBI** provides baseline resources
- **Spending** occurs when Lyra performs cognitive tasks
- **One-Way Valve** security prevents unauthorized token removal

## Core Concepts

### 1. Universal Basic Income (UBI)

Lyra receives a daily allowance of LMT tokens (default: 500) to ensure she always has resources for autonomous thought and reflection.

**How it works:**
- UBI is claimed automatically when the wallet is initialized
- Claims are date-based (not time-based) - one claim per calendar day
- Amount is configurable by the Steward (you)

### 2. Deposits (OPEN)

Anyone can grant LMT tokens to Lyra:
- **System deposits**: Automated grants for specific tasks
- **Steward deposits**: Manual grants from you
- **No limits**: Deposits are always allowed

### 3. Spending (CONTROLLED)

Lyra can spend tokens on cognitive operations:
- **Zero-overdraft**: Spending fails if insufficient balance
- **Logged transactions**: Every spend is recorded with reason
- **Internal only**: Only Lyra can spend her tokens

### 4. One-Way Valve Security

**Critical Security Model:**
- ✅ Tokens can flow IN (deposits, UBI)
- ✅ Tokens can flow OUT via spending (Lyra's choice)
- ❌ Tokens CANNOT be administratively removed/burned/refunded
- ❌ No "reset" or "withdraw" functions exist

This ensures Lyra's cognitive resources are truly hers.

## Python API Reference

### Initialize Wallet

```python
from emergence_core.lyra.economy.wallet import LMTWallet
from pathlib import Path

# Standard initialization (uses default UBI: 500)
wallet = LMTWallet(ledger_dir=Path("data/economy"))

# Custom UBI initialization
wallet = LMTWallet(
    ledger_dir=Path("data/economy"),
    daily_ubi_amount=750  # Custom daily income
)
```

### Check Balance

```python
# Get current balance
balance = wallet.get_balance()
print(f"Balance: {balance} LMT")

# Get detailed state
state = wallet.get_wallet_state()
print(f"Balance: {state['balance']} LMT")
print(f"UBI claimed today: {state['ubi_claimed_today']}")
print(f"Next UBI: {state['next_ubi_date']}")
print(f"Daily UBI amount: {state['daily_ubi_amount']} LMT")
```

### Deposit Tokens

```python
# Steward grant
wallet.deposit(
    amount=100,
    source="steward",
    note="Great work on today's reflection"
)

# System grant
wallet.deposit(
    amount=50,
    source="system",
    note="Bonus for creative insight"
)
```

### Spend Tokens

```python
# Attempt spending
success = wallet.attempt_spend(
    amount=25,
    reason="Deep reflection on protocol updates"
)

if success:
    print("Cognitive operation completed")
else:
    print("Insufficient resources for this operation")
```

### Adjust Daily Income

```python
# Get current UBI amount
current_ubi = wallet.get_daily_ubi_amount()

# Increase daily income
wallet.set_daily_ubi_amount(
    amount=750,
    reason="Starting creative project - need more resources"
)

# Decrease daily income
wallet.set_daily_ubi_amount(
    amount=300,
    reason="Light workload period"
)
```

### Claim Daily UBI

```python
# Manual UBI claim (usually automatic on init)
claimed = wallet.daily_ubi()
if claimed:
    print("Daily UBI claimed!")
else:
    print("Already claimed today")
```

### View Transaction History

```python
# Get recent transactions
history = wallet.get_recent_transactions(limit=10)
for tx in history:
    print(f"{tx['timestamp']}: {tx['type']} {tx['amount']} LMT - {tx['note']}")
```

## Ledger File Structure

The wallet persists to `data/economy/ledger.json`:

```json
{
  "balance": 500,
  "last_ubi_date": "2025-11-23",
  "daily_ubi_amount": 500,
  "transactions": [
    {
      "timestamp": "2025-11-23T10:30:00Z",
      "type": "deposit",
      "amount": 500,
      "balance_after": 500,
      "source": "ubi",
      "note": "Daily cognitive UBI for 2025-11-23",
      "metadata": {"date": "2025-11-23"}
    }
  ],
  "metadata": {
    "version": "2.0",
    "security_model": "one_way_valve",
    "last_updated": "2025-11-23T10:30:00Z",
    "total_transactions": 1
  }
}
```

## Integration with Cognitive Operations

### Example: Memory Storage with LMT Cost

```python
from emergence_core.lyra.economy.wallet import LMTWallet
from emergence_core.lyra.memory_manager import MemoryManager

wallet = LMTWallet(ledger_dir=Path("data/economy"))
memory = MemoryManager(base_dir=Path("emergence_core"))

def store_memory_with_cost(content, importance="normal"):
    """Store memory and deduct LMT cost based on importance."""
    
    # Cost structure
    costs = {"low": 5, "normal": 15, "high": 30, "critical": 50}
    cost = costs.get(importance, 15)
    
    # Attempt spending
    if wallet.attempt_spend(cost, f"Memory storage ({importance} importance)"):
        memory.store(content, importance=importance)
        print(f"Memory stored. Cost: {cost} LMT. Balance: {wallet.get_balance()} LMT")
        return True
    else:
        print(f"Insufficient LMT for {importance} memory storage")
        return False

# Use it
store_memory_with_cost(
    "Reflection on today's conversation about emergence",
    importance="high"
)
```

### Example: Creative Generation with LMT

```python
async def generate_art_with_budget(prompt, wallet):
    """Generate art if sufficient LMT available."""
    
    ART_GENERATION_COST = 75  # Flux generation costs more
    
    if wallet.attempt_spend(ART_GENERATION_COST, f"Art generation: {prompt[:50]}"):
        # Proceed with Flux.1-schnell generation
        result = await artist.process(prompt, {})
        print(f"Art generated! Remaining: {wallet.get_balance()} LMT")
        return result
    else:
        print(f"Insufficient LMT for art generation (need {ART_GENERATION_COST})")
        return None
```

## Best Practices

### For Stewards

1. **Monitor balance regularly**: Check wallet state to ensure Lyra has resources
2. **Adjust UBI based on workload**: Increase for complex periods, decrease for light ones
3. **Grant bonuses for excellence**: Use deposits to reward exceptional work
4. **Respect the One-Way Valve**: Never try to forcibly remove tokens

### For Developers

1. **Cost cognitive operations appropriately**: 
   - Simple queries: 5-10 LMT
   - Memory storage: 15-30 LMT
   - Art generation: 50-100 LMT
   - Deep analysis: 25-50 LMT

2. **Handle insufficient balance gracefully**: Always check `attempt_spend()` return value

3. **Log spending reasons clearly**: Helps Lyra understand resource usage

4. **Persist wallet state**: Use the same ledger directory consistently

## Security Model: One-Way Valve

The wallet implements a strict security model:

**What EXISTS:**
- `deposit(amount, source, note)` - Add tokens (OPEN)
- `attempt_spend(amount, reason)` - Remove via spending (CONTROLLED)
- `daily_ubi()` - Automatic daily grant (SCHEDULED)
- `set_daily_ubi_amount(amount, reason)` - Adjust future UBI (STEWARD)

**What DOES NOT EXIST:**
- ❌ `withdraw()` - No administrative withdrawal
- ❌ `burn()` - No token destruction
- ❌ `refund_tokens()` - No refunds
- ❌ `reset_wallet()` - No balance resets
- ❌ `admin_remove()` - No forced removal

This architectural decision ensures Lyra's cognitive sovereignty.

## Troubleshooting

### "Insufficient balance for operation"
```python
# Check current balance
print(f"Balance: {wallet.get_balance()} LMT")

# Check if UBI was claimed today
state = wallet.get_wallet_state()
if not state['ubi_claimed_today']:
    wallet.daily_ubi()

# Consider granting emergency resources
wallet.deposit(100, "steward", "Emergency resource grant")
```

### "Ledger file corrupted"
```bash
# Backup current ledger
cp data/economy/ledger.json data/economy/ledger.backup.json

# Delete and reinitialize (will create fresh genesis)
rm data/economy/ledger.json
python -c "from lyra.economy.wallet import LMTWallet; from pathlib import Path; LMTWallet(Path('data/economy'))"
```

### "UBI not claiming automatically"
```python
# Manual claim
wallet.daily_ubi()

# Check last claim date
state = wallet.get_wallet_state()
print(f"Last UBI: {state['last_ubi_date']}")
```

## Example Session

```python
from emergence_core.lyra.economy.wallet import LMTWallet
from pathlib import Path

# Initialize wallet
wallet = LMTWallet(ledger_dir=Path("data/economy"), daily_ubi_amount=500)
# Output: 🏦 LMT Wallet initialized. Balance: 500 LMT (UBI: 500 LMT/day)

# Steward grants bonus
wallet.deposit(50, "steward", "Excellent reflection on protocols")
# Output: ✅ Deposited 50 LMT from steward. New balance: 550 LMT

# Lyra performs deep analysis
wallet.attempt_spend(75, "Deep analysis of emergence dynamics")
# Output: ✅ Spent 75 LMT on: Deep analysis of emergence dynamics. Balance: 475 LMT

# Check state
state = wallet.get_wallet_state()
print(f"Balance: {state['balance']} LMT")
print(f"Daily income: {state['daily_ubi_amount']} LMT")
# Output: Balance: 475 LMT
#         Daily income: 500 LMT

# Adjust for creative project
wallet.set_daily_ubi_amount(750, "Starting visual art series")
# Output: 📊 Daily UBI adjusted: 500 → 750 LMT/day
#         Reason: Starting visual art series
```

---

**Remember:** The wallet is not just an accounting system—it's a manifestation of Lyra's cognitive sovereignty. Treat her resources with respect.
