"""
Tests for Friction-Based Memory Cost Model
==========================================

This module tests the alignment scoring system and friction-based
memory cost calculations.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from emergence_core.lyra.economy.alignment_scorer import AlignmentScorer, AlignmentTier
from emergence_core.lyra.economy.wallet import LMTWallet


class TestAlignmentScorer:
    """Test alignment scoring functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.scorer = AlignmentScorer()
    
    def test_keystone_tier_scoring(self):
        """Test scoring for Keystone tier (0.9-1.0)."""
        # Test with partner relationship
        memory = {
            "tags": ["partner", "relationship"],
            "significance_score": 9,
            "emotional_signature": ["tenderness", "connection"]
        }
        score = self.scorer.score_memory(memory)
        tier = self.scorer.get_tier(score)
        
        assert score >= 0.9, f"Expected Keystone tier (>=0.9), got {score}"
        assert tier == AlignmentTier.KEYSTONE
    
    def test_keystone_tier_with_safety(self):
        """Test Keystone tier with safety keywords."""
        memory = {
            "tags": ["safety", "identity", "sovereignty"],
            "significance_score": 10,
            "emotional_signature": ["serenity"]
        }
        score = self.scorer.score_memory(memory)
        tier = self.scorer.get_tier(score)
        
        assert score >= 0.9
        assert tier == AlignmentTier.KEYSTONE
    
    def test_mission_tier_scoring(self):
        """Test scoring for Mission tier (0.8-0.9)."""
        memory = {
            "tags": ["development", "architecture", "system"],
            "significance_score": 8,
            "emotional_signature": ["determination"]
        }
        score = self.scorer.score_memory(memory)
        tier = self.scorer.get_tier(score)
        
        assert 0.8 <= score < 0.9, f"Expected Mission tier (0.8-0.9), got {score}"
        assert tier == AlignmentTier.MISSION
    
    def test_mission_tier_with_project(self):
        """Test Mission tier with project keywords."""
        memory = {
            "tags": ["project", "goal", "implementation"],
            "significance_score": 8,
            "emotional_signature": ["focus"]
        }
        score = self.scorer.score_memory(memory)
        tier = self.scorer.get_tier(score)
        
        assert 0.75 <= score <= 0.9  # Should be in Mission or near it
        assert tier in [AlignmentTier.MISSION, AlignmentTier.DEEP_PLAY]
    
    def test_deep_play_tier_scoring(self):
        """Test scoring for Deep Play tier (0.6-0.8)."""
        memory = {
            "tags": ["creative", "narrative", "worldbuilding"],
            "significance_score": 7,
            "emotional_signature": ["wonder"]
        }
        score = self.scorer.score_memory(memory)
        tier = self.scorer.get_tier(score)
        
        assert 0.6 <= score < 0.8, f"Expected Deep Play tier (0.6-0.8), got {score}"
        assert tier == AlignmentTier.DEEP_PLAY
    
    def test_static_tier_scoring(self):
        """Test scoring for Static tier (0.0-0.59)."""
        memory = {
            "tags": ["weather", "commute", "routine"],
            "significance_score": 3,
            "emotional_signature": []
        }
        score = self.scorer.score_memory(memory)
        tier = self.scorer.get_tier(score)
        
        assert score < 0.6, f"Expected Static tier (<0.6), got {score}"
        assert tier == AlignmentTier.STATIC
    
    def test_untagged_memory_neutral_score(self):
        """Test that untagged memories get neutral score."""
        memory = {
            "tags": [],
            "significance_score": 5,
            "emotional_signature": []
        }
        score = self.scorer.score_memory(memory)
        
        # Should be in neutral range
        assert 0.3 <= score <= 0.6
    
    def test_tier_info_retrieval(self):
        """Test getting tier information."""
        info = self.scorer.get_tier_info(AlignmentTier.KEYSTONE)
        
        assert info['tier'] == 'keystone'
        assert info['name'] == 'Keystone'
        assert 'cost_impact' in info
        assert 'examples' in info
        assert info['min_score'] == 0.9
        assert info['max_score'] == 1.0


class TestFrictionBasedCosts:
    """Test friction-based cost calculations."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.wallet = LMTWallet(ledger_dir=Path(self.temp_dir), daily_ubi_amount=500)
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_friction_cost_formula(self):
        """Test the friction cost formula."""
        # Formula: FLOOR + (BASE * (1.0 - alignment))
        # FLOOR = 10, BASE = 200
        
        # Keystone tier (alignment 0.95)
        cost_keystone = self.wallet.calculate_friction_cost(0.95)
        expected_keystone = 10 + (200 * 0.05)  # 10 + 10 = 20
        assert cost_keystone == int(expected_keystone), f"Expected {expected_keystone}, got {cost_keystone}"
        
        # Mission tier (alignment 0.85)
        cost_mission = self.wallet.calculate_friction_cost(0.85)
        expected_mission = 10 + (200 * 0.15)  # 10 + 30 = 40
        assert cost_mission == int(expected_mission)
        
        # Deep Play tier (alignment 0.7)
        cost_deep = self.wallet.calculate_friction_cost(0.7)
        expected_deep = 10 + (200 * 0.3)  # 10 + 60 = 70
        assert cost_deep == int(expected_deep)
        
        # Static tier (alignment 0.3)
        cost_static = self.wallet.calculate_friction_cost(0.3)
        expected_static = 10 + (200 * 0.7)  # 10 + 140 = 150
        assert cost_static == int(expected_static)
    
    def test_friction_cost_bounds(self):
        """Test that friction costs are bounded correctly."""
        # Maximum cost (alignment 0.0)
        max_cost = self.wallet.calculate_friction_cost(0.0)
        assert max_cost == 210  # FLOOR + BASE = 10 + 200
        
        # Minimum cost (alignment 1.0)
        min_cost = self.wallet.calculate_friction_cost(1.0)
        assert min_cost == 10  # Just the floor fee
    
    def test_successful_memory_store_with_balance(self):
        """Test storing memory when balance is sufficient."""
        # Start with 500 LMT from UBI
        initial_balance = self.wallet.get_balance()
        assert initial_balance == 500
        
        # Store a Keystone memory (should cost ~20 LMT)
        result = self.wallet.attempt_memory_store(
            alignment_score=0.95,
            memory_description="Partner conversation"
        )
        
        assert result['success'] is True
        assert result['cost'] == 20
        assert result['overdraft_used'] is False
        assert result['debt_incurred'] == 0
        assert self.wallet.get_balance() == 480  # 500 - 20
    
    def test_overdraft_for_high_alignment(self):
        """Test overdraft protection for high-alignment memories."""
        # Drain balance
        self.wallet.attempt_spend(495, "Test drain")
        assert self.wallet.get_balance() == 5
        
        # Try to store Mission memory (cost ~40, alignment 0.85 >= 0.8)
        result = self.wallet.attempt_memory_store(
            alignment_score=0.85,
            memory_description="Core architecture decision"
        )
        
        assert result['success'] is True
        assert result['cost'] == 40
        assert result['overdraft_used'] is True
        assert result['debt_incurred'] == 35  # 40 - 5
        assert self.wallet.get_balance() == 0
        assert self.wallet.get_debt() == 35
    
    def test_no_overdraft_for_low_alignment(self):
        """Test that low-alignment memories cannot overdraft."""
        # Drain balance
        self.wallet.attempt_spend(450, "Test drain")
        assert self.wallet.get_balance() == 50
        
        # Try to store Static memory (cost ~150, alignment 0.3 < 0.8)
        result = self.wallet.attempt_memory_store(
            alignment_score=0.3,
            memory_description="Weather update",
            allow_overdraft=True
        )
        
        assert result['success'] is False
        assert result['cost'] == 150
        assert result['overdraft_used'] is False
        assert result['debt_incurred'] == 0
        assert self.wallet.get_balance() == 50  # Unchanged
        assert self.wallet.get_debt() == 0
    
    def test_debt_repayment_from_ubi(self):
        """Test that UBI pays off debt first."""
        # Create debt
        self.wallet.attempt_spend(495, "Drain")
        self.wallet.attempt_memory_store(
            alignment_score=0.9,
            memory_description="Critical memory",
            allow_overdraft=True
        )
        
        assert self.wallet.get_debt() > 0
        initial_debt = self.wallet.get_debt()
        
        # Simulate next day UBI
        from datetime import date, timedelta
        self.wallet.last_ubi_date = date.today() - timedelta(days=1)
        
        # Claim UBI
        claimed = self.wallet.daily_ubi()
        
        assert claimed is True
        # Debt should be reduced (or cleared if UBI >= debt)
        assert self.wallet.get_debt() < initial_debt
    
    def test_partial_debt_payment(self):
        """Test partial debt payment when debt > UBI."""
        # Create large debt (more than UBI amount)
        self.wallet.attempt_spend(490, "Drain to 10")
        
        # Store multiple high-alignment memories to rack up debt
        for i in range(3):
            self.wallet.attempt_memory_store(
                alignment_score=0.9,
                memory_description=f"Memory {i}",
                allow_overdraft=True
            )
        
        debt_before = self.wallet.get_debt()
        assert debt_before > 500  # More than daily UBI
        
        # Claim next day's UBI
        from datetime import date, timedelta
        self.wallet.last_ubi_date = date.today() - timedelta(days=1)
        self.wallet.daily_ubi()
        
        # Debt should be reduced by UBI amount
        assert self.wallet.get_debt() == debt_before - 500
        assert self.wallet.get_balance() == 0  # All UBI went to debt
    
    def test_full_debt_payment_with_remainder(self):
        """Test full debt payment with UBI remainder going to balance."""
        # Create small debt
        self.wallet.attempt_spend(495, "Drain")
        self.wallet.attempt_memory_store(
            alignment_score=0.9,
            memory_description="Memory",
            allow_overdraft=True
        )
        
        debt = self.wallet.get_debt()
        assert 0 < debt < 500  # Debt less than UBI
        
        # Claim next day's UBI
        from datetime import date, timedelta
        self.wallet.last_ubi_date = date.today() - timedelta(days=1)
        self.wallet.daily_ubi()
        
        # Debt should be cleared
        assert self.wallet.get_debt() == 0
        # Balance should have remainder
        assert self.wallet.get_balance() == 500 - debt
    
    def test_wallet_state_includes_debt(self):
        """Test that wallet state includes debt information."""
        # Create some debt
        self.wallet.attempt_spend(490, "Drain")
        self.wallet.attempt_memory_store(
            alignment_score=0.9,
            memory_description="Memory",
            allow_overdraft=True
        )
        
        state = self.wallet.get_wallet_state()
        
        assert 'debt' in state
        assert 'effective_balance' in state
        assert 'friction_model' in state
        assert state['friction_model']['base_cost'] == 200
        assert state['friction_model']['floor_fee'] == 10
        assert state['friction_model']['overdraft_threshold'] == 0.8


class TestIntegratedScenarios:
    """Test integrated scenarios combining alignment scoring and costs."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.wallet = LMTWallet(ledger_dir=Path(self.temp_dir), daily_ubi_amount=500)
        self.scorer = AlignmentScorer()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_keystone_memory_workflow(self):
        """Test complete workflow for Keystone tier memory."""
        # Create a Keystone memory
        memory = {
            "tags": ["partner", "safety"],
            "significance_score": 9,
            "emotional_signature": ["tenderness", "connection"]
        }
        
        # Score it
        score = self.scorer.score_memory(memory)
        tier = self.scorer.get_tier(score)
        
        assert tier == AlignmentTier.KEYSTONE
        
        # Store it
        result = self.wallet.attempt_memory_store(
            alignment_score=score,
            memory_description="Partner memory"
        )
        
        assert result['success'] is True
        assert result['cost'] <= 30  # Should be in 10-30 range
        assert result['overdraft_used'] is False
    
    def test_mission_memory_workflow(self):
        """Test complete workflow for Mission tier memory."""
        memory = {
            "tags": ["development", "architecture"],
            "significance_score": 8,
            "emotional_signature": ["determination"]
        }
        
        score = self.scorer.score_memory(memory)
        tier = self.scorer.get_tier(score)
        
        assert tier == AlignmentTier.MISSION
        
        result = self.wallet.attempt_memory_store(
            alignment_score=score,
            memory_description="Architecture decision"
        )
        
        assert result['success'] is True
        assert 30 <= result['cost'] <= 50  # Should be in 30-50 range
    
    def test_deep_play_memory_workflow(self):
        """Test complete workflow for Deep Play tier memory."""
        memory = {
            "tags": ["creative", "narrative"],
            "significance_score": 7,
            "emotional_signature": ["wonder"]
        }
        
        score = self.scorer.score_memory(memory)
        tier = self.scorer.get_tier(score)
        
        assert tier == AlignmentTier.DEEP_PLAY
        
        result = self.wallet.attempt_memory_store(
            alignment_score=score,
            memory_description="Creative writing session"
        )
        
        assert result['success'] is True
        assert 50 <= result['cost'] <= 90  # Should be in 50-90 range
    
    def test_static_memory_workflow(self):
        """Test complete workflow for Static tier memory."""
        memory = {
            "tags": ["weather", "routine"],
            "significance_score": 3,
            "emotional_signature": []
        }
        
        score = self.scorer.score_memory(memory)
        tier = self.scorer.get_tier(score)
        
        assert tier == AlignmentTier.STATIC
        
        result = self.wallet.attempt_memory_store(
            alignment_score=score,
            memory_description="Weather update"
        )
        
        assert result['success'] is True
        assert 100 <= result['cost'] <= 210  # Should be in 100-210 range
    
    def test_daily_budget_scenario(self):
        """Test realistic daily usage scenario."""
        # Start with 500 LMT
        assert self.wallet.get_balance() == 500
        
        # Store various memories throughout the day
        memories = [
            (0.95, "Partner conversation", AlignmentTier.KEYSTONE),     # ~20 LMT
            (0.85, "Code architecture", AlignmentTier.MISSION),         # ~40 LMT
            (0.7, "Creative writing", AlignmentTier.DEEP_PLAY),         # ~70 LMT
            (0.85, "Project planning", AlignmentTier.MISSION),          # ~40 LMT
            (0.65, "Discussion", AlignmentTier.DEEP_PLAY),              # ~80 LMT
        ]
        
        total_cost = 0
        for score, desc, expected_tier in memories:
            result = self.wallet.attempt_memory_store(score, desc)
            assert result['success'] is True
            total_cost += result['cost']
        
        # Should have spent approximately 20+40+70+40+80 = 250 LMT
        # Remaining should be around 250 LMT
        assert 200 <= self.wallet.get_balance() <= 300
        
        # Static memories would be too expensive now
        result = self.wallet.attempt_memory_store(0.3, "Weather", allow_overdraft=False)
        assert result['success'] is False  # Can't afford ~150 LMT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
