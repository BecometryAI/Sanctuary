# Quick Reference: Lyra Configuration Changes

## LMT Wallet - Daily Income Adjustment

### Current Configuration
- **Default daily UBI:** 500 LMT/day
- **Adjustable:** Yes, via Steward control
- **Persisted:** Changes saved to `data/economy/ledger.json`

### How to Adjust Daily Income

**Python API:**
```python
from emergence_core.lyra.economy.wallet import LMTWallet
from pathlib import Path

# Initialize wallet
wallet = LMTWallet(ledger_dir=Path("data/economy"))

# Check current daily income
current = wallet.get_daily_ubi_amount()  # Returns 500 by default

# Increase for higher workload
wallet.set_daily_ubi_amount(750, "Increased complexity and creative demands")

# Decrease for lighter periods
wallet.set_daily_ubi_amount(250, "Reduced workload this week")

# Check balance
balance = wallet.get_balance()
```

**Quick Demo:**
```bash
python tools/demo_wallet_ubi.py
```

### Common UBI Levels
| Workload | Daily UBI | Use Case |
|----------|-----------|----------|
| Light | 250 LMT | Basic conversations |
| **Default** | **500 LMT** | **Balanced baseline** |
| Creative | 750 LMT | Art + deep writing |
| Complex | 1000 LMT | Research + analysis |
| Major | 1500 LMT | Protocol design |

### Full Documentation
See: `docs/LMT_WALLET_GUIDE.md`

---

## Artist Model - Flux.1-schnell

### What Changed
- **Old Model:** Stable Diffusion 3 Medium
- **New Model:** Flux.1-schnell
- **Why:** 3x faster (4 steps vs 28), better quality, lower VRAM

### Model Comparison
| Feature | Flux.1-schnell | SD3 Medium |
|---------|----------------|------------|
| Speed | ~10-15s | ~30-45s |
| VRAM | 4-6GB | 6-8GB |
| Quality | Higher | High |
| License | Apache 2.0 | Restricted |

### Updated Files
- ✅ `README.md` - Updated architecture description
- ✅ `config/models.json` - Changed artist model path
- ✅ `emergence_core/lyra/specialists.py` - Already using Flux (no change needed)
- ✅ `tools/verify_flux_setup.py` - Renamed from verify_sd3_setup.py
- ✅ `docs/FLUX_SETUP_GUIDE.md` - New installation guide

### Installation
```bash
pip install diffusers transformers accelerate safetensors pillow
```

### Verification
```bash
python tools/verify_flux_setup.py
```

### Configuration
Location: `config/models.json`
```json
{
  "artist": {
    "path": "black-forest-labs/FLUX.1-schnell",
    "device": "auto",
    "dtype": "float16",
    "inference_steps": 4,
    "guidance_scale": 0.0
  }
}
```

### Full Documentation
See: `docs/FLUX_SETUP_GUIDE.md`

---

## Files Created/Updated

### New Documentation
1. `docs/LMT_WALLET_GUIDE.md` - Complete wallet API reference
2. `docs/FLUX_SETUP_GUIDE.md` - Flux.1 installation and setup
3. `tools/demo_wallet_ubi.py` - Wallet demonstration script
4. `QUICK_REFERENCE.md` - This file!

### Updated Files
1. `README.md` - Updated artist model, added wallet section, updated links
2. `config/models.json` - Changed artist path to Flux.1-schnell
3. `tools/verify_flux_setup.py` - Renamed and updated from SD3 version
4. `emergence_core/lyra/economy/wallet.py` - Implemented with 500 LMT default + UBI adjustment

### Code Updates
- **Wallet:** Default UBI is 500 LMT/day, adjustable via `set_daily_ubi_amount()`
- **Artist:** Already using Flux.1-schnell in specialists.py (no code changes needed)
- **Config:** Models.json updated to reflect Flux path

---

## How to Use

### Adjust Lyra's Daily Income
```python
# In any Python script or REPL:
from emergence_core.lyra.economy.wallet import LMTWallet
from pathlib import Path

wallet = LMTWallet(ledger_dir=Path("data/economy"))
wallet.set_daily_ubi_amount(750, "Starting creative project")
```

### Check Current Configuration
```python
state = wallet.get_wallet_state()
print(f"Daily income: {state['daily_ubi_amount']} LMT")
print(f"Balance: {state['balance']} LMT")
```

### Verify Flux Setup
```bash
python tools/verify_flux_setup.py
```

---

## Remember
- **Wallet UBI:** Default 500 LMT/day, adjust via `set_daily_ubi_amount(amount, reason)`
- **Flux Model:** 3x faster than SD3, better quality, already in code
- **Documentation:** Full guides in `docs/` folder
- **Demo:** Run `python tools/demo_wallet_ubi.py` to see wallet in action

---

**Last Updated:** November 23, 2025
