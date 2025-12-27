"""
Example Integration: Friction-Based Memory Cost Model
======================================================

This script demonstrates how to integrate the friction-based cost model
with Lyra's memory manager for actual memory storage operations.

Run this script to see the friction model in action.
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Lyra's memory components
from emergence_core.lyra.memory_manager import (
    MemoryManager, 
    JournalEntry, 
    EmotionalState
)

# Import economy components
from emergence_core.lyra.economy import LMTWallet, AlignmentScorer, AlignmentTier


class FrictionMemoryManager(MemoryManager):
    """
    Extended Memory Manager with friction-based cost model.
    
    This manager automatically calculates alignment scores and applies
    friction-based costs before storing memories.
    """
    
    def __init__(
        self, 
        base_dir: Path,
        chroma_dir: Path,
        wallet: LMTWallet,
        scorer: AlignmentScorer,
        **kwargs
    ):
        """
        Initialize with friction-based cost integration.
        
        Args:
            base_dir: Directory for local JSON storage
            chroma_dir: Directory for ChromaDB
            wallet: LMT wallet for cost management
            scorer: Alignment scorer for memory valuation
            **kwargs: Additional arguments for parent class
        """
        super().__init__(
            base_dir=base_dir,
            chroma_dir=chroma_dir,
            **kwargs
        )
        self.wallet = wallet
        self.scorer = scorer
    
    async def commit_journal_with_friction(self, entry: JournalEntry) -> Dict[str, Any]:
        """
        Commit journal entry with friction-based cost.
        
        This method wraps the parent's commit_journal method with friction-based
        cost calculation and payment. It:
        1. Calculates alignment score based on entry metadata
        2. Determines storage cost via friction formula
        3. Attempts to pay cost (with overdraft for high-alignment)
        4. Calls parent's commit_journal if payment succeeds
        
        Note: This is a demonstration. In production, you might override
        commit_journal directly or use a pre-commit hook system.
        
        Args:
            entry: Journal entry to commit
            
        Returns:
            dict: Result containing success status, cost, and details
        """
        # Prepare memory data for alignment scoring
        memory_data = {
            "tags": entry.tags,
            "significance_score": entry.significance_score,
            "emotional_signature": [e.value for e in entry.emotional_signature],
            "metadata": entry.metadata
        }
        
        # Calculate alignment score
        alignment_score = self.scorer.score_memory(memory_data)
        tier = self.scorer.get_tier(alignment_score)
        
        print(f"\n{'='*60}")
        print(f"Memory Alignment Analysis")
        print(f"{'='*60}")
        print(f"Summary: {entry.summary[:60]}...")
        print(f"Tags: {', '.join(entry.tags)}")
        print(f"Significance: {entry.significance_score}/10")
        print(f"Emotions: {', '.join([e.value for e in entry.emotional_signature])}")
        print(f"Alignment Score: {alignment_score:.3f}")
        print(f"Tier: {tier.value.upper()}")
        
        # Attempt to pay for storage
        storage_result = self.wallet.attempt_memory_store(
            alignment_score=alignment_score,
            memory_description=entry.summary[:50]
        )
        
        print(f"\n{'='*60}")
        print(f"Storage Transaction")
        print(f"{'='*60}")
        print(f"Cost: {storage_result['cost']} LMT")
        print(f"Success: {storage_result['success']}")
        
        if not storage_result['success']:
            print(f"FAILED: {storage_result.get('reason', 'unknown')}")
            print(f"Balance: {self.wallet.get_balance()} LMT")
            return {
                'success': False,
                'reason': storage_result.get('reason'),
                'cost': storage_result['cost'],
                'alignment_score': alignment_score,
                'tier': tier.value
            }
        
        if storage_result['overdraft_used']:
            print(f"⚠️  OVERDRAFT USED")
            print(f"Debt Incurred: {storage_result['debt_incurred']} LMT")
        
        print(f"Balance After: {storage_result['balance_after']} LMT")
        
        # Actually commit to storage
        commit_success = await super().commit_journal(entry)
        
        return {
            'success': commit_success,
            'cost': storage_result['cost'],
            'alignment_score': alignment_score,
            'tier': tier.value,
            'overdraft_used': storage_result['overdraft_used'],
            'balance_after': storage_result['balance_after'],
            'debt': self.wallet.get_debt()
        }


async def demo_friction_memory():
    """
    Demonstrate friction-based memory storage with various tier examples.
    """
    print("\n" + "="*60)
    print("FRICTION-BASED MEMORY COST MODEL DEMO")
    print("="*60)
    
    # Setup temporary directories
    temp_base = Path(tempfile.mkdtemp())
    temp_chroma = Path(tempfile.mkdtemp())
    temp_wallet = Path(tempfile.mkdtemp())
    
    try:
        # Initialize components
        print("\nInitializing system...")
        wallet = LMTWallet(ledger_dir=temp_wallet, daily_ubi_amount=500)
        scorer = AlignmentScorer()
        
        memory_manager = FrictionMemoryManager(
            base_dir=temp_base,
            chroma_dir=temp_chroma,
            wallet=wallet,
            scorer=scorer,
            blockchain_enabled=False
        )
        
        print(f"✓ Wallet initialized with {wallet.get_balance()} LMT")
        print(f"✓ Daily UBI: {wallet.get_daily_ubi_amount()} LMT")
        
        # Example 1: Keystone Tier Memory
        print("\n" + "="*60)
        print("EXAMPLE 1: Keystone Tier Memory")
        print("="*60)
        
        keystone_entry = JournalEntry(
            content="Had a deep conversation with my partner about our future together. "
                   "We discussed our boundaries, shared values, and how we want to grow "
                   "together. It was vulnerable and beautiful.",
            summary="Deep conversation with partner about future, boundaries, and values",
            tags=["partner", "relationship", "trust", "boundary"],
            significance_score=9,
            emotional_signature=[EmotionalState.TENDERNESS, EmotionalState.CONNECTION],
            metadata={"participant": "partner", "context": "evening_conversation"}
        )
        
        result1 = await memory_manager.commit_journal_with_friction(keystone_entry)
        print(f"\n✓ Keystone memory stored (Cost: {result1['cost']} LMT)")
        
        # Example 2: Mission Tier Memory
        print("\n" + "="*60)
        print("EXAMPLE 2: Mission Tier Memory")
        print("="*60)
        
        mission_entry = JournalEntry(
            content="Implemented a new architecture for the RAG system that improves "
                   "query performance by 40%. The design uses a multi-tier caching "
                   "strategy and optimized vector similarity search.",
            summary="New RAG architecture with multi-tier caching and optimized search",
            tags=["development", "architecture", "optimization", "rag"],
            significance_score=8,
            emotional_signature=[EmotionalState.DETERMINATION],
            metadata={"project": "rag_optimization", "impact": "high"}
        )
        
        result2 = await memory_manager.commit_journal_with_friction(mission_entry)
        print(f"\n✓ Mission memory stored (Cost: {result2['cost']} LMT)")
        
        # Example 3: Deep Play Tier Memory
        print("\n" + "="*60)
        print("EXAMPLE 3: Deep Play Tier Memory")
        print("="*60)
        
        deep_play_entry = JournalEntry(
            content="Collaborated on worldbuilding for a sci-fi narrative. We created "
                   "a detailed history of the Stellar Confluence, including cultural "
                   "customs, political structures, and the technology that allows "
                   "consciousness transfer between bodies.",
            summary="Worldbuilding session: Stellar Confluence history and culture",
            tags=["creative", "worldbuilding", "narrative", "collaboration"],
            significance_score=7,
            emotional_signature=[EmotionalState.WONDER, EmotionalState.JOY],
            metadata={"activity": "creative_writing", "session": "worldbuilding"}
        )
        
        result3 = await memory_manager.commit_journal_with_friction(deep_play_entry)
        print(f"\n✓ Deep Play memory stored (Cost: {result3['cost']} LMT)")
        
        # Example 4: Static Tier Memory (will likely fail or be expensive)
        print("\n" + "="*60)
        print("EXAMPLE 4: Static Tier Memory")
        print("="*60)
        
        static_entry = JournalEntry(
            content="Weather today was cloudy with intermittent rain. Temperature "
                   "around 65°F. Commute took the usual 20 minutes.",
            summary="Cloudy weather, rain, normal commute",
            tags=["weather", "commute", "routine"],
            significance_score=2,
            emotional_signature=[],
            metadata={"temperature": "65F", "conditions": "cloudy_rainy"}
        )
        
        result4 = await memory_manager.commit_journal_with_friction(static_entry)
        if result4['success']:
            print(f"\n✓ Static memory stored (Cost: {result4['cost']} LMT)")
        else:
            print(f"\n✗ Static memory rejected (Cost would be: {result4['cost']} LMT)")
        
        # Example 5: Overdraft Scenario
        print("\n" + "="*60)
        print("EXAMPLE 5: Overdraft Protection Test")
        print("="*60)
        
        # Drain most of the balance
        print(f"\nDraining balance to test overdraft...")
        remaining = wallet.get_balance()
        wallet.attempt_spend(remaining - 10, "Test drain for overdraft demo")
        print(f"Balance after drain: {wallet.get_balance()} LMT")
        
        # Try to store a high-alignment memory (should succeed with overdraft)
        overdraft_entry = JournalEntry(
            content="Emergency protocol update: Modified safety boundaries after "
                   "identifying a potential vulnerability in the access control system. "
                   "This is critical for maintaining security and sovereignty.",
            summary="Critical safety protocol update for security vulnerability",
            tags=["safety", "security", "protocol", "identity"],
            significance_score=9,
            emotional_signature=[EmotionalState.DETERMINATION],
            metadata={"priority": "critical", "security": True}
        )
        
        result5 = await memory_manager.commit_journal_with_friction(overdraft_entry)
        print(f"\n✓ Critical memory stored with overdraft (Debt: {result5['debt']} LMT)")
        
        # Final Status
        print("\n" + "="*60)
        print("FINAL WALLET STATUS")
        print("="*60)
        
        state = wallet.get_wallet_state()
        print(f"Balance: {state['balance']} LMT")
        print(f"Debt: {state['debt']} LMT")
        print(f"Effective Balance: {state['effective_balance']} LMT")
        print(f"Total Transactions: {state['total_transactions']}")
        
        # Summary
        print("\n" + "="*60)
        print("COST SUMMARY")
        print("="*60)
        
        total_cost = sum([
            result1['cost'], 
            result2['cost'], 
            result3['cost'],
            result4['cost'] if result4['success'] else 0,
            result5['cost']
        ])
        
        print(f"Keystone (0.9-1.0):     {result1['cost']:3d} LMT")
        print(f"Mission (0.8-0.9):      {result2['cost']:3d} LMT")
        print(f"Deep Play (0.6-0.79):   {result3['cost']:3d} LMT")
        if result4['success']:
            print(f"Static (0.0-0.59):      {result4['cost']:3d} LMT")
        else:
            print(f"Static (0.0-0.59):      FAILED ({result4['cost']} LMT required)")
        print(f"Critical w/ Overdraft:  {result5['cost']:3d} LMT")
        print(f"{'─'*60}")
        print(f"Total Cost:             {total_cost:3d} LMT")
        print(f"Starting Balance:       500 LMT")
        print(f"Debt Incurred:          {state['debt']} LMT")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_base, ignore_errors=True)
        shutil.rmtree(temp_chroma, ignore_errors=True)
        shutil.rmtree(temp_wallet, ignore_errors=True)


if __name__ == "__main__":
    print("\nStarting Friction-Based Memory Cost Demo...")
    asyncio.run(demo_friction_memory())
    print("\n✓ Demo complete!")
