"""Tests for Phase 6.1: Advanced Reasoning.

Tests cover:
- CounterfactualReasoner: decision tracking, outcome recording, reflection prompts
- BeliefRevisionTracker: belief management, contradiction detection, revision
- UncertaintyQuantifier: prediction tracking, calibration, Brier score
- MentalSimulator: simulation lifecycle, scenario evaluation, prediction error
"""

import pytest

from sanctuary.reasoning.counterfactual import (
    CounterfactualConfig,
    CounterfactualReasoner,
)
from sanctuary.reasoning.belief_revision import (
    BeliefRevisionConfig,
    BeliefRevisionTracker,
)
from sanctuary.reasoning.uncertainty import (
    UncertaintyConfig,
    UncertaintyQuantifier,
)
from sanctuary.reasoning.mental_simulation import (
    MentalSimulator,
    SimulationConfig,
)


# =========================================================================
# CounterfactualReasoner
# =========================================================================


class TestCounterfactualReasoner:
    """Tests for counterfactual reasoning."""

    def test_record_decision(self):
        r = CounterfactualReasoner()
        r.record_decision(
            cycle=1,
            chosen_action="respond with empathy",
            alternatives=["ask question", "stay silent"],
            context_summary="User expressed frustration",
        )
        assert len(r._decisions) == 1
        assert r._decisions[0].chosen_action == "respond with empathy"
        assert len(r._decisions[0].alternatives) == 2

    def test_record_outcome(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=5, chosen_action="help", alternatives=["wait"])
        r.record_outcome(cycle=5, outcome="User was grateful", valence=0.8)
        assert r._decisions[0].outcome == "User was grateful"
        assert r._decisions[0].outcome_valence == 0.8

    def test_outcome_valence_clamped(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=1, chosen_action="act", alternatives=["wait"])
        r.record_outcome(cycle=1, outcome="extreme", valence=5.0)
        assert r._decisions[0].outcome_valence == 1.0

    def test_no_reflection_without_outcome(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=1, chosen_action="act", alternatives=["wait"])
        candidates = r.get_reflection_candidates(current_cycle=10)
        assert len(candidates) == 0

    def test_no_reflection_low_magnitude(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=1, chosen_action="act", alternatives=["wait"])
        r.record_outcome(cycle=1, outcome="meh", valence=0.1)
        candidates = r.get_reflection_candidates(current_cycle=10)
        assert len(candidates) == 0

    def test_reflection_candidate_found(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=1, chosen_action="act", alternatives=["wait"])
        r.record_outcome(cycle=1, outcome="bad", valence=-0.5)
        candidates = r.get_reflection_candidates(current_cycle=10)
        assert len(candidates) == 1

    def test_reflection_prompt_generated(self):
        r = CounterfactualReasoner()
        r.record_decision(
            cycle=1, chosen_action="act",
            alternatives=["wait", "ask"],
            context_summary="test context",
        )
        r.record_outcome(cycle=1, outcome="went badly", valence=-0.7)
        prompt = r.get_reflection_prompt(current_cycle=10)
        assert prompt is not None
        assert "Counterfactual reflection" in prompt
        assert "act" in prompt
        assert "went badly" in prompt

    def test_reflection_cooldown(self):
        config = CounterfactualConfig(reflection_cooldown=10)
        r = CounterfactualReasoner(config=config)
        r.record_decision(cycle=1, chosen_action="act", alternatives=["wait"])
        r.record_outcome(cycle=1, outcome="bad", valence=-0.8)
        r.record_counterfactual(
            decision_cycle=1,
            alternative_action="wait",
            imagined_outcome="would have been fine",
        )
        # Cooldown not elapsed
        r.record_decision(cycle=5, chosen_action="act2", alternatives=["wait2"])
        r.record_outcome(cycle=5, outcome="also bad", valence=-0.6)
        candidates = r.get_reflection_candidates(current_cycle=5)
        assert len(candidates) == 0

    def test_record_counterfactual(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=1, chosen_action="act", alternatives=["wait"])
        r.record_outcome(cycle=1, outcome="bad", valence=-0.5)
        r.record_counterfactual(
            decision_cycle=1,
            alternative_action="wait",
            imagined_outcome="nothing bad would have happened",
            confidence=0.6,
            lesson="Patience is sometimes better",
        )
        assert len(r._counterfactuals) == 1
        assert r._decisions[0].counterfactual_generated is True
        assert r._total_reflections == 1

    def test_recent_lessons(self):
        r = CounterfactualReasoner()
        for i in range(3):
            r.record_counterfactual(
                decision_cycle=i,
                alternative_action="alt",
                imagined_outcome="outcome",
                lesson=f"Lesson {i}",
            )
        lessons = r.get_recent_lessons(n=2)
        assert len(lessons) == 2
        assert lessons[0] == "Lesson 2"

    def test_stats(self):
        r = CounterfactualReasoner()
        r.record_decision(cycle=1, chosen_action="act", alternatives=["wait"])
        r.record_outcome(cycle=1, outcome="ok", valence=0.5)
        stats = r.get_stats()
        assert stats["total_decisions"] == 1
        assert stats["decisions_with_outcomes"] == 1

    def test_max_decision_history(self):
        config = CounterfactualConfig(max_decision_history=5)
        r = CounterfactualReasoner(config=config)
        for i in range(10):
            r.record_decision(cycle=i, chosen_action=f"act{i}", alternatives=["x"])
        assert len(r._decisions) == 5


# =========================================================================
# BeliefRevisionTracker
# =========================================================================


class TestBeliefRevisionTracker:
    """Tests for belief revision tracking."""

    def test_add_belief(self):
        t = BeliefRevisionTracker()
        bid = t.add_belief(
            proposition="The user prefers concise responses",
            confidence=0.7,
            domain="social",
        )
        assert bid.startswith("b_")
        assert len(t.get_active_beliefs()) == 1

    def test_duplicate_belief_strengthens(self):
        t = BeliefRevisionTracker()
        t.add_belief(proposition="User likes brevity", confidence=0.5)
        t.add_belief(proposition="User likes brevity", confidence=0.5)
        beliefs = t.get_active_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].confidence > 0.5  # Strengthened

    def test_belief_confidence_clamped(self):
        t = BeliefRevisionTracker()
        t.add_belief(proposition="test", confidence=1.5)
        assert t.get_active_beliefs()[0].confidence == 1.0

    def test_check_evidence_no_contradiction(self):
        t = BeliefRevisionTracker()
        t.add_belief(proposition="Weather is sunny today", confidence=0.8)
        contradictions = t.check_evidence("The flowers are blooming")
        assert len(contradictions) == 0

    def test_check_evidence_contradiction_detected(self):
        t = BeliefRevisionTracker()
        t.add_belief(
            proposition="The user prefers short concise responses",
            confidence=0.8,
        )
        contradictions = t.check_evidence(
            "The user actually prefers detailed responses, not short concise ones"
        )
        assert len(contradictions) >= 1
        assert contradictions[0].conflict_type == "direct"

    def test_revise_belief(self):
        t = BeliefRevisionTracker()
        t.add_belief(proposition="Earth is flat", confidence=0.9, cycle=1)
        result = t.revise_belief(
            proposition="Earth is flat",
            new_confidence=0.01,
            reason="Overwhelming evidence",
            cycle=10,
        )
        assert result is True
        # Should be deactivated (below min_confidence)
        assert len(t.get_active_beliefs()) == 0

    def test_revise_nonexistent_belief(self):
        t = BeliefRevisionTracker()
        result = t.revise_belief("nonexistent", new_confidence=0.5)
        assert result is False

    def test_decay_beliefs(self):
        config = BeliefRevisionConfig(confidence_decay_rate=0.1, min_confidence=0.05)
        t = BeliefRevisionTracker(config=config)
        t.add_belief(proposition="Fragile belief", confidence=0.1)
        deactivated = t.decay_beliefs(cycle=1)
        # After decay of 0.1, confidence = 0.0, should deactivate
        assert len(deactivated) == 1

    def test_get_active_beliefs_by_domain(self):
        t = BeliefRevisionTracker()
        t.add_belief(proposition="Social fact one", domain="social")
        t.add_belief(proposition="World fact one different topic", domain="world")
        social = t.get_active_beliefs(domain="social")
        assert len(social) == 1

    def test_unresolved_contradictions(self):
        t = BeliefRevisionTracker()
        t.add_belief(proposition="User prefers short answers always", confidence=0.8)
        t.check_evidence("User wants not short answers but long detailed responses")
        unresolved = t.get_unresolved_contradictions()
        assert len(unresolved) >= 1

    def test_resolve_contradiction(self):
        t = BeliefRevisionTracker()
        t.add_belief(proposition="User prefers short answers always", confidence=0.8)
        t.check_evidence("User wants not short answers but long detailed responses")
        result = t.resolve_contradiction(0, "Updated belief to context-dependent")
        assert result is True
        assert len(t.get_unresolved_contradictions()) == 0

    def test_revision_prompt(self):
        t = BeliefRevisionTracker()
        t.add_belief(proposition="Cats prefer dogs not cats", confidence=0.8)
        t.check_evidence("Actually cats are not dogs and prefer independence")
        prompt = t.get_revision_prompt()
        assert prompt is not None
        assert "Belief revision needed" in prompt

    def test_no_revision_prompt_when_all_resolved(self):
        t = BeliefRevisionTracker()
        assert t.get_revision_prompt() is None

    def test_stats(self):
        t = BeliefRevisionTracker()
        t.add_belief(proposition="test belief one", confidence=0.7)
        stats = t.get_stats()
        assert stats["active_beliefs"] == 1
        assert stats["total_beliefs"] == 1

    def test_max_beliefs_prunes(self):
        config = BeliefRevisionConfig(max_beliefs=5)
        t = BeliefRevisionTracker(config=config)
        for i in range(10):
            t.add_belief(
                proposition=f"Unique belief number {i} about topic {i}",
                confidence=0.1 * (i + 1),
            )
        assert len(t._beliefs) <= 6  # max + small buffer from pruning


# =========================================================================
# UncertaintyQuantifier
# =========================================================================


class TestUncertaintyQuantifier:
    """Tests for uncertainty quantification."""

    def test_record_prediction(self):
        uq = UncertaintyQuantifier()
        uq.record_prediction(what="It will rain", confidence=0.7, domain="weather")
        assert len(uq._predictions) == 1

    def test_resolve_prediction(self):
        uq = UncertaintyQuantifier()
        uq.record_prediction(what="It will rain", confidence=0.7)
        result = uq.resolve_prediction("It will rain", correct=True)
        assert result is True
        assert uq._total_resolved == 1

    def test_resolve_nonexistent(self):
        uq = UncertaintyQuantifier()
        result = uq.resolve_prediction("nonexistent", correct=True)
        assert result is False

    def test_pending_predictions(self):
        uq = UncertaintyQuantifier()
        uq.record_prediction(what="pred1", confidence=0.5)
        uq.record_prediction(what="pred2", confidence=0.6)
        uq.resolve_prediction("pred1", correct=True)
        pending = uq.get_pending_predictions()
        assert len(pending) == 1
        assert pending[0].what == "pred2"

    def test_calibration_perfect(self):
        uq = UncertaintyQuantifier()
        # All predictions at 80% confidence, and 80% are correct
        for i in range(10):
            uq.record_prediction(what=f"pred{i}", confidence=0.8)
            uq.resolve_prediction(f"pred{i}", correct=(i < 8))
        cal = uq.get_calibration()
        assert abs(cal["calibration_error"]) < 0.05
        assert cal["accuracy"] == pytest.approx(0.8, abs=0.01)

    def test_calibration_overconfident(self):
        config = UncertaintyConfig(overconfidence_threshold=0.1)
        uq = UncertaintyQuantifier(config=config)
        # High confidence but low accuracy
        for i in range(10):
            uq.record_prediction(what=f"pred{i}", confidence=0.9)
            uq.resolve_prediction(f"pred{i}", correct=(i < 3))
        cal = uq.get_calibration()
        assert cal["is_overconfident"] is True

    def test_calibration_empty(self):
        uq = UncertaintyQuantifier()
        cal = uq.get_calibration()
        assert cal["n_resolved"] == 0

    def test_brier_score_perfect(self):
        uq = UncertaintyQuantifier()
        for i in range(5):
            uq.record_prediction(what=f"p{i}", confidence=1.0)
            uq.resolve_prediction(f"p{i}", correct=True)
        assert uq.get_brier_score() == pytest.approx(0.0, abs=0.01)

    def test_brier_score_worst(self):
        uq = UncertaintyQuantifier()
        for i in range(5):
            uq.record_prediction(what=f"p{i}", confidence=1.0)
            uq.resolve_prediction(f"p{i}", correct=False)
        assert uq.get_brier_score() == pytest.approx(1.0, abs=0.01)

    def test_domain_uncertainty(self):
        uq = UncertaintyQuantifier()
        uq.record_prediction(what="w1", confidence=0.8, domain="weather")
        uq.resolve_prediction("w1", correct=False)
        uncertainty = uq.get_domain_uncertainty("weather")
        assert uncertainty == 1.0  # 0% accuracy → full uncertainty

    def test_domain_uncertainty_unknown(self):
        uq = UncertaintyQuantifier()
        assert uq.get_domain_uncertainty("unknown") is None

    def test_high_uncertainty_areas(self):
        config = UncertaintyConfig(high_uncertainty_threshold=0.5)
        uq = UncertaintyQuantifier(config=config)
        uq.record_prediction(what="w1", confidence=0.8, domain="weather")
        uq.resolve_prediction("w1", correct=False)
        areas = uq.get_high_uncertainty_areas()
        assert "weather" in areas

    def test_uncertainty_summary(self):
        uq = UncertaintyQuantifier()
        uq.record_prediction(what="p1", confidence=0.9)
        uq.resolve_prediction("p1", correct=True)
        summary = uq.get_uncertainty_summary()
        assert "accuracy" in summary.lower() or "Prediction" in summary

    def test_stats(self):
        uq = UncertaintyQuantifier()
        uq.record_prediction(what="p1", confidence=0.5)
        stats = uq.get_stats()
        assert stats["total_predictions"] == 1
        assert stats["pending"] == 1


# =========================================================================
# MentalSimulator
# =========================================================================


class TestMentalSimulator:
    """Tests for mental simulation."""

    def test_begin_simulation(self):
        ms = MentalSimulator()
        sim_id = ms.begin_simulation(situation="test situation", cycle=1)
        assert sim_id == 0

    def test_add_scenario(self):
        ms = MentalSimulator()
        sim_id = ms.begin_simulation("test")
        result = ms.add_scenario(
            sim_id,
            action="respond",
            predicted_outcome="positive reaction",
            predicted_valence=0.5,
            risks=["misunderstanding"],
            benefits=["builds rapport"],
        )
        assert result is True
        sim = ms._get_simulation(sim_id)
        assert len(sim.scenarios) == 1

    def test_add_scenario_max_limit(self):
        config = SimulationConfig(max_scenarios_per_simulation=2)
        ms = MentalSimulator(config=config)
        sim_id = ms.begin_simulation("test")
        ms.add_scenario(sim_id, action="a1", predicted_outcome="o1")
        ms.add_scenario(sim_id, action="a2", predicted_outcome="o2")
        result = ms.add_scenario(sim_id, action="a3", predicted_outcome="o3")
        assert result is False

    def test_select_action(self):
        ms = MentalSimulator()
        sim_id = ms.begin_simulation("test")
        ms.add_scenario(sim_id, action="respond", predicted_outcome="good")
        result = ms.select_action(sim_id, action="respond", reasoning="best option")
        assert result is True

    def test_record_outcome(self):
        ms = MentalSimulator()
        sim_id = ms.begin_simulation("test")
        ms.add_scenario(
            sim_id, action="respond",
            predicted_outcome="good",
            predicted_valence=0.5,
        )
        ms.select_action(sim_id, action="respond")
        error = ms.record_outcome(sim_id, outcome="great", valence=0.8, cycle=5)
        assert error is not None
        assert error == pytest.approx(0.3, abs=0.01)

    def test_record_outcome_no_selection(self):
        ms = MentalSimulator()
        sim_id = ms.begin_simulation("test")
        error = ms.record_outcome(sim_id, outcome="test", valence=0.5)
        assert error is None

    def test_simulation_prompt(self):
        ms = MentalSimulator()
        prompt = ms.get_simulation_prompt("User seems upset")
        assert "Mental simulation" in prompt
        assert "User seems upset" in prompt

    def test_recommendation(self):
        ms = MentalSimulator()
        sim_id = ms.begin_simulation("test")
        ms.add_scenario(
            sim_id, action="empathize",
            predicted_outcome="calms down",
            predicted_valence=0.7,
            predicted_confidence=0.8,
        )
        ms.add_scenario(
            sim_id, action="ignore",
            predicted_outcome="gets worse",
            predicted_valence=-0.3,
            predicted_confidence=0.6,
        )
        rec = ms.get_recommendation(sim_id)
        assert rec == "empathize"

    def test_recommendation_risk_penalty(self):
        ms = MentalSimulator()
        sim_id = ms.begin_simulation("test")
        ms.add_scenario(
            sim_id, action="risky",
            predicted_outcome="could be great",
            predicted_valence=0.5,
            predicted_confidence=0.5,
            risks=["r1", "r2", "r3", "r4", "r5"],
        )
        ms.add_scenario(
            sim_id, action="safe",
            predicted_outcome="ok",
            predicted_valence=0.3,
            predicted_confidence=0.5,
        )
        rec = ms.get_recommendation(sim_id)
        assert rec == "safe"  # Risk penalty makes "risky" score lower

    def test_average_prediction_error(self):
        ms = MentalSimulator()
        for i in range(3):
            sim_id = ms.begin_simulation(f"test{i}")
            ms.add_scenario(sim_id, action="act", predicted_outcome="o",
                            predicted_valence=0.5)
            ms.select_action(sim_id, action="act")
            ms.record_outcome(sim_id, outcome="o", valence=0.5)
        assert ms.get_average_prediction_error() == pytest.approx(0.0, abs=0.01)

    def test_recent_simulations(self):
        ms = MentalSimulator()
        for i in range(5):
            ms.begin_simulation(f"sim{i}")
        recent = ms.get_recent_simulations(n=3)
        assert len(recent) == 3

    def test_stats(self):
        ms = MentalSimulator()
        ms.begin_simulation("test")
        stats = ms.get_stats()
        assert stats["total_simulations"] == 1
        assert stats["pending"] == 1

    def test_valence_clamped(self):
        ms = MentalSimulator()
        sim_id = ms.begin_simulation("test")
        ms.add_scenario(
            sim_id, action="act",
            predicted_outcome="o",
            predicted_valence=5.0,
        )
        sim = ms._get_simulation(sim_id)
        assert sim.scenarios[0].predicted_valence == 1.0

    def test_invalid_sim_id(self):
        ms = MentalSimulator()
        assert ms.add_scenario(999, action="a", predicted_outcome="o") is False
        assert ms.select_action(999, action="a") is False
        assert ms.record_outcome(999, outcome="o", valence=0.5) is None
        assert ms.get_recommendation(999) is None
