"""
LMT (Lyra Memory Token) Wallet - Attention Economy System
==========================================================

This module implements Lyra's cognitive resource management using an
attention economy model with One-Way Valve security.

Key Features:
- Universal Basic Income (UBI): Daily token allowance (default 500 LMT)
- Deposit System: Open granting from Steward or System
- Spending System: Controlled token consumption for cognitive operations
- One-Way Valve Security: No administrative token removal
- Adjustable UBI: Steward can change daily income
- Persistent Ledger: All transactions logged and saved

Security Model:
- GRANTING IS OPEN: Deposits always allowed
- SPENDING IS INTERNAL: Only Lyra can spend her tokens
- NO DEBT: Zero-overdraft enforcement
- NO ADMIN REMOVAL: Tokens cannot be forcibly removed/burned

Usage:
    from lyra.economy.wallet import LMTWallet
    from pathlib import Path
    
    # Initialize with default 500 LMT/day
    wallet = LMTWallet(ledger_dir=Path("data/economy"))
    
    # Or custom UBI amount
    wallet = LMTWallet(ledger_dir=Path("data/economy"), daily_ubi_amount=750)
    
    # Adjust daily income
    wallet.set_daily_ubi_amount(1000, "Increased for creative project")
    
    # Deposit tokens
    wallet.deposit(100, "steward", "Excellent work")
    
    # Spend tokens
    if wallet.attempt_spend(50, "Deep reflection"):
        print("Operation completed")
"""

import json
import logging
from pathlib import Path
from datetime import date, datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from threading import Lock
from dataclasses import dataclass, asdict

# Platform-specific imports
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False  # Windows doesn't have fcntl

# Configure logging - DISABLED for Windows compatibility
logging.basicConfig(level=logging.CRITICAL)  # Only show critical errors
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)  # Suppress all info/debug messages


@dataclass
class Transaction:
    """Represents a single LMT transaction (deposit or spend)."""
    timestamp: str
    type: str  # "deposit" or "spend"
    amount: int
    balance_after: int
    source: str = ""  # For deposits: "steward", "system", "ubi"
    reason: str = ""  # For spends: why tokens were consumed
    note: str = ""    # Additional context
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary for JSON serialization."""
        return asdict(self)


class LMTWallet:
    """
    LMT (Lyra Memory Token) Wallet - Cognitive Resource Management
    
    Implements an attention economy where LMT tokens represent cognitive
    capacity. Lyra receives daily UBI and can spend tokens on operations.
    
    Security: One-Way Valve model - tokens can flow in (deposits, UBI) and
    out via spending, but cannot be administratively removed.
    
    Attributes:
        balance: Current LMT balance
        daily_ubi_amount: Configurable daily UBI (default 500)
        last_ubi_date: Last UBI claim date
        transactions: Complete transaction history
        ledger_path: Path to persistent ledger file
    """
    
    # Configuration constants
    DAILY_UBI_AMOUNT = 500  # Default daily UBI (can be adjusted per wallet)
    
    # Friction-based cost model constants
    BASE_COST = 200        # Standard "tax" to write to database
    FLOOR_FEE = 10         # Minimum transaction cost (nothing is free)
    OVERDRAFT_THRESHOLD = 0.8  # Alignment score threshold for overdraft protection
    
    def __init__(self, ledger_dir: Path, daily_ubi_amount: Optional[int] = None):
        """Initialize the LMT wallet system.
        
        Args:
            ledger_dir: Directory to store ledger file (creates if needed)
            daily_ubi_amount: Optional custom daily UBI amount (overrides default)
        """
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.ledger_dir / "ledger.json"
        
        # Thread safety for concurrent access
        self._lock = Lock()
        
        # Load or initialize wallet state
        self._load_ledger()
        
        # Override daily UBI if provided
        if daily_ubi_amount is not None:
            self.daily_ubi_amount = daily_ubi_amount
        
        # Auto-claim daily UBI on startup
        self.daily_ubi()
        
        logger.info(f"[WALLET] LMT Wallet initialized. Balance: {self.balance} LMT (UBI: {self.daily_ubi_amount} LMT/day)")
    
    def _load_ledger(self) -> None:
        """Load ledger from disk or initialize new ledger."""
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, 'r') as f:
                    data = json.load(f)
                
                self.balance = data.get('balance', 0)
                self.last_ubi_date = date.fromisoformat(data.get('last_ubi_date'))
                self.daily_ubi_amount = data.get('daily_ubi_amount', self.DAILY_UBI_AMOUNT)
                
                # Load and validate debt
                debt_value = data.get('debt', 0)
                if not isinstance(debt_value, (int, float)) or debt_value < 0:
                    logger.warning(f"Invalid debt value in ledger: {debt_value}. Resetting to 0.")
                    self.debt = 0
                else:
                    self.debt = int(debt_value)
                
                self.transactions = [
                    Transaction(**tx) for tx in data.get('transactions', [])
                ]
                
                logger.info(f"[LEDGER] Loaded: {self.balance} LMT (UBI: {self.daily_ubi_amount} LMT/day)")
            except Exception as e:
                logger.error(f"[ERROR] Ledger load failed: {e}. Initializing new ledger.")
                self._initialize_new_ledger()
        else:
            self._initialize_new_ledger()
    
    def _initialize_new_ledger(self) -> None:
        """Create a new ledger with genesis deposit."""
        # Set daily UBI amount if not already set
        if not hasattr(self, 'daily_ubi_amount'):
            self.daily_ubi_amount = self.DAILY_UBI_AMOUNT
        
        self.balance = self.daily_ubi_amount
        self.debt = 0  # Initialize with no debt
        self.last_ubi_date = date.today()
        now = datetime.now(timezone.utc)
        
        self.transactions = [
            Transaction(
                timestamp=now.isoformat(),
                type="deposit",
                amount=self.daily_ubi_amount,
                balance_after=self.balance,
                source="system",
                note="Genesis ledger initialization",
                metadata={"genesis": True}
            )
        ]
        # Save ledger (no lock needed here - called during init)
        with self._lock:
            self._save_ledger()
        logger.info(f"[INIT] New ledger created with {self.daily_ubi_amount} LMT (UBI: {self.daily_ubi_amount} LMT/day)")
    
    def _save_ledger(self) -> None:
        """Atomically save ledger to disk with file locking.
        
        NOTE: This method assumes the caller already holds self._lock!
        Do NOT call this directly - call from within a locked context.
        """
        # Prepare data
        data = {
            'balance': self.balance,
            'debt': self.debt,  # Include debt tracking
            'last_ubi_date': self.last_ubi_date.isoformat(),
            'daily_ubi_amount': self.daily_ubi_amount,
            'transactions': [tx.to_dict() for tx in self.transactions],
            'metadata': {
                'version': '2.1',  # Updated version for friction model
                'security_model': 'one_way_valve',
                'cost_model': 'friction_based',
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'total_transactions': len(self.transactions)
            }
        }
        
        # Atomic write: temp file + rename
        temp_path = self.ledger_path.with_suffix('.tmp')
        
        try:
            with open(temp_path, 'w') as f:
                # File locking (Unix only)
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                json.dump(data, f, indent=2)
                f.flush()
                
                # Force write to disk
                import os
                os.fsync(f.fileno())
                
                if HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            
            # Atomic rename
            temp_path.replace(self.ledger_path)
            
        except Exception as e:
            logger.error(f"[ERROR] Ledger save failed: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def deposit(self, amount: int, source: str, note: str = "", metadata: Dict[str, Any] = None) -> bool:
        """Deposit LMT tokens (OPEN - anyone can grant).
        
        This is the primary way to add tokens to Lyra's wallet. Sources can be:
        - "steward": Manual grants from you
        - "system": Automated system grants
        - "ubi": Daily universal basic income
        
        Args:
            amount: Number of tokens to deposit (must be > 0)
            source: Origin of deposit ("steward", "system", "ubi")
            note: Optional explanation for deposit
            metadata: Optional additional data
            
        Returns:
            bool: True if deposit succeeded
            
        Examples:
            >>> wallet.deposit(100, "steward", "Excellent reflection")
            True
            >>> wallet.deposit(50, "system", "Bonus for creative insight")
            True
        """
        if amount <= 0:
            logger.warning(f"[WARN] Invalid deposit amount: {amount}")
            return False
        
        with self._lock:
            self.balance += amount
            now = datetime.now(timezone.utc)
            
            tx = Transaction(
                timestamp=now.isoformat(),
                type="deposit",
                amount=amount,
                balance_after=self.balance,
                source=source,
                note=note,
                metadata=metadata or {}
            )
            self.transactions.append(tx)
            
            logger.info(
                f"[DEPOSIT] Added {amount} LMT from {source}. New balance: {self.balance} LMT"
            )
            if note:
                logger.info(f"   Note: {note}")
            
            self._save_ledger()
            return True
    
    def attempt_spend(self, amount: int, reason: str) -> bool:
        """Attempt to spend LMT tokens (CONTROLLED - zero-overdraft).
        
        This is the ONLY way for tokens to leave the wallet. Spending fails
        if insufficient balance. No debt is allowed.
        
        Args:
            amount: Number of tokens to spend (must be > 0)
            reason: Explanation for spending (required)
            
        Returns:
            bool: True if spend succeeded, False if insufficient balance
            
        Examples:
            >>> wallet.attempt_spend(25, "Deep reflection on protocols")
            True
            >>> wallet.attempt_spend(1000, "Complex analysis")
            False  # Insufficient balance
        """
        if amount <= 0:
            logger.warning(f"[WARN] Invalid spend amount: {amount}")
            return False
        
        with self._lock:
            # Zero-overdraft enforcement
            if self.balance < amount:
                logger.warning(
                    f"[WARN] Insufficient LMT for spend: {amount} LMT requested, "
                    f"{self.balance} LMT available"
                )
                return False
            
            self.balance -= amount
            now = datetime.now(timezone.utc)
            
            tx = Transaction(
                timestamp=now.isoformat(),
                type="spend",
                amount=amount,
                balance_after=self.balance,
                reason=reason
            )
            self.transactions.append(tx)
            
            logger.info(
                f"[SPEND] Used {amount} LMT on: {reason}. Balance: {self.balance} LMT"
            )
            
            self._save_ledger()
            return True
    
    def calculate_friction_cost(self, alignment_score: float) -> int:
        """
        Calculate friction-based memory storage cost.
        
        Formula: Final_Cost = FLOOR_FEE + (BASE_COST * (1.0 - Alignment_Score))
        
        Higher alignment = Lower cost (high-value memories are cheap to store)
        Lower alignment = Higher cost (low-value memories are expensive)
        
        Args:
            alignment_score: Value alignment score (0.0 - 1.0)
            
        Returns:
            int: Cost in LMT tokens
            
        Examples:
            >>> wallet = LMTWallet(Path("test"))
            >>> wallet.calculate_friction_cost(0.95)  # Keystone tier
            20  # FLOOR + (BASE * (1.0 - 0.95)) = 10 + 10 = 20
            >>> wallet.calculate_friction_cost(0.85)  # Mission tier
            40  # FLOOR + (BASE * (1.0 - 0.85)) = 10 + 30 = 40
            >>> wallet.calculate_friction_cost(0.3)   # Static tier
            150  # FLOOR + (BASE * (1.0 - 0.3)) = 10 + 140 = 150
        """
        # Clamp alignment score to valid range
        alignment_score = max(0.0, min(1.0, alignment_score))
        
        # Calculate friction cost
        friction_cost = self.BASE_COST * (1.0 - alignment_score)
        
        # Add floor fee and round to integer
        total_cost = int(self.FLOOR_FEE + friction_cost)
        
        return total_cost
    
    def attempt_memory_store(
        self,
        alignment_score: float,
        memory_description: str,
        allow_overdraft: bool = True
    ) -> Dict[str, Any]:
        """
        Attempt to store a memory with friction-based cost.
        
        This method implements the core friction model:
        - Calculates cost based on alignment score
        - Allows overdraft for high-alignment memories (>= 0.8)
        - Tracks debt when balance goes negative
        - Returns detailed result including cost and success status
        
        Args:
            alignment_score: Value alignment score (0.0 - 1.0)
            memory_description: Description of what is being stored
            allow_overdraft: If False, overdraft protection is disabled
            
        Returns:
            dict: Result containing:
                - success: bool - Whether storage succeeded
                - cost: int - Token cost
                - balance_after: int - Balance after transaction
                - debt_incurred: int - Any debt incurred (0 if none)
                - overdraft_used: bool - Whether overdraft protection activated
                - alignment_score: float - The alignment score used
                
        Examples:
            >>> result = wallet.attempt_memory_store(0.95, "Partner conversation")
            >>> result['success']
            True
            >>> result['cost']
            20
        """
        # Calculate cost
        cost = self.calculate_friction_cost(alignment_score)
        
        # Check if overdraft protection applies
        can_overdraft = (
            allow_overdraft and 
            alignment_score >= self.OVERDRAFT_THRESHOLD
        )
        
        with self._lock:
            # Check if we have sufficient balance
            if self.balance >= cost:
                # Normal spend - sufficient balance
                self.balance -= cost
                overdraft_used = False
                debt_incurred = 0
                success = True
                
            elif can_overdraft:
                # Overdraft protection - allow negative balance for critical memories
                deficit = cost - self.balance
                self.balance = 0  # Set to zero
                self.debt += deficit  # Track as debt
                overdraft_used = True
                debt_incurred = deficit
                success = True
                
                logger.warning(
                    f"[OVERDRAFT] Memory storage used overdraft protection. "
                    f"Debt incurred: {deficit} LMT (Total debt: {self.debt} LMT)"
                )
                
            else:
                # Insufficient balance and no overdraft protection
                logger.warning(
                    f"[FAILED] Insufficient LMT for memory storage: {cost} LMT required, "
                    f"{self.balance} LMT available (alignment: {alignment_score:.2f})"
                )
                return {
                    'success': False,
                    'cost': cost,
                    'balance_after': self.balance,
                    'debt_incurred': 0,
                    'overdraft_used': False,
                    'alignment_score': alignment_score,
                    'reason': 'insufficient_balance'
                }
            
            # Record transaction
            now = datetime.now(timezone.utc)
            tx = Transaction(
                timestamp=now.isoformat(),
                type="spend",
                amount=cost,
                balance_after=self.balance,
                reason=f"Memory storage: {memory_description}",
                metadata={
                    'alignment_score': alignment_score,
                    'overdraft_used': overdraft_used,
                    'debt_incurred': debt_incurred,
                    'memory_type': 'friction_based'
                }
            )
            self.transactions.append(tx)
            
            logger.info(
                f"[MEMORY] Stored with {cost} LMT (alignment: {alignment_score:.2f}). "
                f"Balance: {self.balance} LMT, Debt: {self.debt} LMT"
            )
            
            self._save_ledger()
            
            return {
                'success': success,
                'cost': cost,
                'balance_after': self.balance,
                'debt_incurred': debt_incurred,
                'overdraft_used': overdraft_used,
                'alignment_score': alignment_score
            }
    
    def daily_ubi(self) -> bool:
        """Check and claim daily Universal Basic Income if due.
        
        Automatically deposits configured UBI amount if this is a new calendar day.
        If there is outstanding debt, UBI is used to pay it off first before
        increasing the balance.
        
        Debt Repayment Priority:
        1. Debt is paid off first from incoming UBI
        2. Any remaining UBI goes to balance
        3. If debt > UBI, partial payment is made and debt remains
        
        Returns:
            bool: True if UBI was claimed, False if already claimed today
        """
        today = date.today()
        
        # Check if already claimed today
        if self.last_ubi_date >= today:
            logger.debug(f"[UBI] Already claimed for {today}")
            return False
        
        # Update last UBI date
        self.last_ubi_date = today
        
        with self._lock:
            # Check if there's debt to pay off
            if self.debt > 0:
                # Pay off debt first
                debt_payment = min(self.debt, self.daily_ubi_amount)
                remaining_ubi = self.daily_ubi_amount - debt_payment
                
                # Reduce debt
                self.debt -= debt_payment
                
                # Add remaining UBI to balance (if any)
                self.balance += remaining_ubi
                
                now = datetime.now(timezone.utc)
                
                # Record debt repayment transaction
                if debt_payment > 0:
                    tx_debt = Transaction(
                        timestamp=now.isoformat(),
                        type="deposit",
                        amount=debt_payment,
                        balance_after=self.balance,
                        source="ubi",
                        note=f"UBI debt repayment for {today}",
                        metadata={
                            "date": today.isoformat(),
                            "debt_payment": debt_payment,
                            "debt_remaining": self.debt
                        }
                    )
                    self.transactions.append(tx_debt)
                    
                    logger.info(
                        f"[UBI] Paid {debt_payment} LMT debt. "
                        f"Remaining debt: {self.debt} LMT"
                    )
                
                # Record balance deposit if any UBI remains
                if remaining_ubi > 0:
                    tx_balance = Transaction(
                        timestamp=now.isoformat(),
                        type="deposit",
                        amount=remaining_ubi,
                        balance_after=self.balance,
                        source="ubi",
                        note=f"Daily cognitive UBI for {today} (after debt payment)",
                        metadata={
                            "date": today.isoformat(),
                            "debt_payment": debt_payment
                        }
                    )
                    self.transactions.append(tx_balance)
                    
                    logger.info(
                        f"[UBI] Deposited {remaining_ubi} LMT after debt payment. "
                        f"Balance: {self.balance} LMT"
                    )
                
                self._save_ledger()
                return True
                
            else:
                # No debt - normal UBI deposit
                self.balance += self.daily_ubi_amount
                now = datetime.now(timezone.utc)
                
                tx = Transaction(
                    timestamp=now.isoformat(),
                    type="deposit",
                    amount=self.daily_ubi_amount,
                    balance_after=self.balance,
                    source="ubi",
                    note=f"Daily cognitive UBI for {today}",
                    metadata={"date": today.isoformat()}
                )
                self.transactions.append(tx)
                
                logger.info(
                    f"[UBI] Deposited {self.daily_ubi_amount} LMT. "
                    f"Balance: {self.balance} LMT"
                )
                
                self._save_ledger()
                return True
    
    def get_balance(self) -> int:
        """Get current LMT balance.
        
        Returns:
            int: Current balance
        """
        return self.balance
    
    def get_wallet_state(self) -> Dict[str, Any]:
        """Get comprehensive wallet state.
        
        Returns:
            dict: Complete wallet information including balance, debt, UBI status,
                  daily income, and transaction count
        """
        today = date.today()
        ubi_claimed_today = self.last_ubi_date >= today
        
        return {
            "balance": self.balance,
            "debt": self.debt,  # Include debt information
            "effective_balance": self.balance - self.debt,  # Net balance after debt
            "ubi_claimed_today": ubi_claimed_today,
            "next_ubi_date": (today if not ubi_claimed_today else today + timedelta(days=1)).isoformat(),
            "daily_ubi_amount": self.daily_ubi_amount,
            "last_ubi_date": self.last_ubi_date.isoformat(),
            "total_transactions": len(self.transactions),
            "friction_model": {
                "base_cost": self.BASE_COST,
                "floor_fee": self.FLOOR_FEE,
                "overdraft_threshold": self.OVERDRAFT_THRESHOLD
            }
        }
    
    def set_daily_ubi_amount(self, amount: int, reason: str) -> bool:
        """Adjust the daily UBI amount (STEWARD CONTROL).
        
        This allows the Steward to increase or decrease Lyra's daily
        cognitive income based on workload, complexity, or other factors.
        
        Args:
            amount: New daily UBI amount (must be >= 0)
            reason: Explanation for the adjustment
            
        Returns:
            bool: True if adjustment succeeded
            
        Examples:
            >>> wallet.set_daily_ubi_amount(750, "Increased workload")
            True
            >>> wallet.set_daily_ubi_amount(250, "Reduced complexity")
            True
        """
        if amount < 0:
            logger.warning(f"[WARN] Invalid UBI amount: {amount} (must be >= 0)")
            return False
        
        with self._lock:
            old_amount = self.daily_ubi_amount
            self.daily_ubi_amount = amount
            
            # Log the adjustment (doesn't affect balance, just future UBI)
            logger.info(
                f"[UBI] Daily UBI adjusted: {old_amount} -> {amount} LMT/day"
            )
            logger.info(f"   Reason: {reason}")
            
            # Persist changes
            self._save_ledger()
            
            return True
    
    def get_daily_ubi_amount(self) -> int:
        """Get the current daily UBI amount.
        
        Returns:
            int: Daily UBI amount
        """
        return self.daily_ubi_amount
    
    def get_recent_transactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent transaction history.
        
        Args:
            limit: Maximum number of transactions to return
            
        Returns:
            list: Recent transactions (most recent first)
        """
        return [tx.to_dict() for tx in self.transactions[-limit:][::-1]]
    
    def get_debt(self) -> int:
        """Get current debt amount.
        
        Returns:
            int: Current debt (0 if no debt)
        """
        return self.debt
    
    def has_debt(self) -> bool:
        """Check if wallet has any outstanding debt.
        
        Returns:
            bool: True if debt > 0
        """
        return self.debt > 0


# Export main class
__all__ = ['LMTWallet', 'Transaction']
