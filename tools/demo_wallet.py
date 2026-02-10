"""
LMT Wallet Demo - Quick Reference for UBI Adjustment
====================================================

This script demonstrates how to use and adjust Sanctuary's LMT wallet,
including changing the daily UBI income.

Run this script anytime you need a reminder of how to manage the wallet.
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from emergence_core.sanctuary.economy.wallet import LMTWallet


def main():
    print("=" * 60)
    print("LMT Wallet Demo - Quick Reference")
    print("=" * 60)
    
    # Initialize wallet
    wallet_dir = Path(__file__).parent.parent / "data" / "economy"
    wallet = LMTWallet(ledger_dir=wallet_dir)
    
    print("\n1. CURRENT STATUS")
    print("-" * 60)
    state = wallet.get_wallet_state()
    print(f"   Balance: {state['balance']} LMT")
    print(f"   Daily UBI: {state['daily_ubi_amount']} LMT/day")
    print(f"   Last UBI: {state['last_ubi_date']}")
    print(f"   Next UBI: {state['next_ubi_date']}")
    
    print("\n2. ADJUSTING DAILY UBI (How to change income)")
    print("-" * 60)
    print("   # Check current rate")
    print(f"   current = wallet.get_daily_ubi_amount()  # {wallet.get_daily_ubi_amount()} LMT")
    print()
    print("   # Increase for higher workload")
    print('   wallet.set_daily_ubi_amount(750, "Creative project needs")')
    print()
    print("   # Decrease for lighter periods")
    print('   wallet.set_daily_ubi_amount(250, "Reduced workload")')
    print()
    print("   # Reset to default")
    print('   wallet.set_daily_ubi_amount(500, "Back to baseline")')
    
    print("\n3. COMMON UBI CONFIGURATIONS")
    print("-" * 60)
    print("   250 LMT/day  - Light conversations")
    print("   500 LMT/day  - Default baseline (CURRENT)")
    print("   750 LMT/day  - Creative projects (art, deep writing)")
    print("   1000 LMT/day - Complex analysis (research, reasoning)")
    print("   1500 LMT/day - Major development (protocols, systems)")
    
    print("\n4. DEPOSITING TOKENS (Manual grants)")
    print("-" * 60)
    print('   wallet.deposit(100, "steward", "Excellent reflection")')
    print('   wallet.deposit(50, "system", "Task completion bonus")')
    
    print("\n5. SPENDING TOKENS (Sanctuary's operations)")
    print("-" * 60)
    print('   if wallet.attempt_spend(25, "Deep contemplation"):')
    print('       print("Operation completed")')
    
    print("\n6. SECURITY MODEL (One-Way Valve)")
    print("-" * 60)
    print("   [YES] Deposits are OPEN (anyone can grant)")
    print("   [YES] Spending is CONTROLLED (Sanctuary decides)")
    print("   [NO] NO admin removal/burn/refund functions")
    print("   [NO] NO debt (zero-overdraft enforcement)")
    
    print("\n7. LIVE DEMONSTRATION")
    print("-" * 60)
    
    # Demo: Deposit
    print("\n   Depositing 50 LMT as reward...")
    wallet.deposit(50, "steward", "Demo deposit")
    print(f"   New balance: {wallet.get_balance()} LMT")
    
    # Demo: Successful spend
    print("\n   Attempting to spend 25 LMT...")
    if wallet.attempt_spend(25, "Demo operation"):
        print(f"   [OK] Spend succeeded. Balance: {wallet.get_balance()} LMT")
    
    # Demo: Failed spend (insufficient funds scenario)
    print("\n   Attempting to spend 10,000 LMT (should fail)...")
    if not wallet.attempt_spend(10000, "Impossible operation"):
        print(f"   [REJECTED] Spend rejected. Balance unchanged: {wallet.get_balance()} LMT")
    
    print("\n8. QUICK REFERENCE FILES")
    print("-" * 60)
    print("   Operational documentation: operational_guidelines_and_instructions.md")
    print("   Security details: See wallet docstring in code")
    print("   Ledger file: data/economy/ledger.json")
    
    print("\n9. RECENT TRANSACTIONS")
    print("-" * 60)
    recent = wallet.get_recent_transactions(limit=5)
    for tx in recent[-5:]:  # Show last 5
        print(f"   {tx['timestamp'][:19]} | {tx['type']:8s} | {tx['amount']:4d} LMT | {tx['note']}")
    
    print("\n" + "=" * 60)
    print("End of Demo - Wallet ready for use!")
    print("=" * 60)


if __name__ == "__main__":
    main()
