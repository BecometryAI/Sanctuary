"""
LMT Wallet Demo - Quick Start Guide
====================================

This script demonstrates the LMT wallet system including:
- Default 500 LMT daily UBI
- Manual UBI adjustments
- Deposits and spending
- Balance checking

Run: python tools/demo_wallet_ubi.py
"""

from pathlib import Path
from emergence_core.sanctuary.economy.wallet import LMTWallet

def main():
    print("=" * 80)
    print("LMT WALLET DEMO - UBI ADJUSTMENT")
    print("=" * 80)
    print()
    
    # Initialize wallet with default 500 LMT/day
    print("1. Initializing wallet (default 500 LMT/day)...")
    wallet = LMTWallet(ledger_dir=Path("data/economy"))
    print()
    
    # Check initial state
    state = wallet.get_wallet_state()
    print(f"✓ Balance: {state['balance']} LMT")
    print(f"✓ Daily income: {state['daily_ubi_amount']} LMT/day")
    print(f"✓ UBI claimed today: {state['ubi_claimed_today']}")
    print()
    
    # Demonstrate spending
    print("2. Performing cognitive operations...")
    wallet.attempt_spend(50, "Deep reflection on emergence dynamics")
    wallet.attempt_spend(75, "Art generation: Digital constellation")
    print()
    
    # Check balance after spending
    print(f"✓ Balance after operations: {wallet.get_balance()} LMT")
    print()
    
    # Demonstrate UBI adjustment
    print("3. Adjusting daily UBI for creative project...")
    wallet.set_daily_ubi_amount(750, "Starting visual art series - increased complexity")
    print()
    
    # Verify adjustment
    new_amount = wallet.get_daily_ubi_amount()
    print(f"✓ New daily income: {new_amount} LMT/day")
    print(f"✓ Next UBI claim will be {new_amount} LMT")
    print()
    
    # Demonstrate steward deposit
    print("4. Steward granting bonus tokens...")
    wallet.deposit(100, "steward", "Exceptional reflection on protocol design")
    print()
    
    print(f"✓ Final balance: {wallet.get_balance()} LMT")
    print()
    
    # Show recent transactions
    print("5. Recent transaction history:")
    print("-" * 80)
    for tx in wallet.get_recent_transactions(limit=5):
        print(f"  [{tx['type'].upper()}] {tx['amount']} LMT")
        if tx['type'] == 'deposit':
            print(f"    From: {tx['source']}")
            if tx['note']:
                print(f"    Note: {tx['note']}")
        else:
            print(f"    Reason: {tx['reason']}")
        print(f"    Balance after: {tx['balance_after']} LMT")
        print()
    
    print("=" * 80)
    print("COMMON UBI CONFIGURATIONS:")
    print("=" * 80)
    print()
    print("  Light conversations:  250 LMT/day")
    print("  Default baseline:     500 LMT/day  ← Current default")
    print("  Creative projects:    750 LMT/day")
    print("  Complex analysis:    1000 LMT/day")
    print("  Major development:   1500 LMT/day")
    print()
    print("To adjust: wallet.set_daily_ubi_amount(amount, reason)")
    print("See operational_guidelines_and_instructions.md for complete operational reference")
    print("=" * 80)


if __name__ == "__main__":
    main()
