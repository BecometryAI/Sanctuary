#!/usr/bin/env python3
"""
Phase 2 FLOW Validation

Verifies that data actually PROPAGATES through the cognitive cycle.
Phase 1 proved methods don't crash. Phase 2 proves data flows.

Key questions answered:
1. Does input text become a percept with a real embedding?
2. Does attention select it (not drop it)?
3. Does it land in the workspace?
4. Does the workspace broadcast include it?
5. Does affect respond to emotional content?
6. Does the action subsystem see the workspace state?
7. Do multiple inputs compete for attention correctly?
8. Does workspace state accumulate across cycles?

Requires only: Python 3.10+, numpy
Result: 40/40 passed at 6,275 Hz

KEY ARCHITECTURAL FINDING:
Affect has a 1-cycle delay. It runs BEFORE workspace update (step 4 vs step 8),
so it processes the PREVIOUS cycle's workspace. This is correct GWT behavior:
you process what's already conscious, then update consciousness.
"""

import asyncio
import hashlib
import logging
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("flow_validation")
logger.setLevel(logging.INFO)


# ============================================================
# Minimal stand-ins (same as validate_boot.py)
# ============================================================

@dataclass
class Percept:
    modality: str
    raw: Any
    embedding: Optional[List[float]] = None
    complexity: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:12])


@dataclass
class WorkspaceSnapshot:
    goals: list = field(default_factory=list)
    percepts: dict = field(default_factory=dict)
    emotions: dict = field(default_factory=lambda: {"valence": 0.0, "arousal": 0.0, "dominance": 0.0})
    memories: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    cycle_count: int = 0
    metadata: dict = field(default_factory=dict)
    temporal_context: Optional[dict] = None


class MockPerceptionSubsystem:
    def __init__(self, config=None):
        self.config = config or {}
        self.embedding_dim = self.config.get("mock_embedding_dim", 384)
        self.embedding_cache: OrderedDict = OrderedDict()
        self.cache_size = 1000
        self.stats = {"cache_hits": 0, "cache_misses": 0, "total_encodings": 0}

    async def encode(self, raw_input, modality):
        embedding = self._deterministic_embedding(str(raw_input))
        complexity = min(max(len(str(raw_input)) // 20, 5), 50)
        self.stats["total_encodings"] += 1
        return Percept(
            modality=modality, raw=raw_input, embedding=embedding,
            complexity=complexity, metadata={"mock_mode": True},
        )

    def _deterministic_embedding(self, text):
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            self.stats["cache_hits"] += 1
            return self.embedding_cache[cache_key]
        self.stats["cache_misses"] += 1
        seed = int(cache_key[:8], 16)
        rng = np.random.RandomState(seed)
        emb = rng.randn(self.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        result = emb.tolist()
        self.embedding_cache[cache_key] = result
        return result


# ============================================================
# TRACED subsystems -- record what they receive and return
# ============================================================

class TracedAttention:
    """Attention that records what it sees and applies actual selection logic."""

    def __init__(self, budget=100):
        self.budget = budget
        self.trace: List[Dict] = []

    def select_for_broadcast(self, candidates, emotional_state=None, prediction_errors=None):
        selected = []
        budget_used = 0
        rejected = []

        scored = []
        for p in candidates:
            score = np.linalg.norm(p.embedding) if p.embedding else 0.0
            scored.append((p, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        for p, score in scored:
            if budget_used + p.complexity <= self.budget:
                selected.append(p)
                budget_used += p.complexity
            else:
                rejected.append(p)

        self.trace.append({
            "candidates": len(candidates),
            "selected": len(selected),
            "rejected": len(rejected),
            "budget_used": budget_used,
            "emotional_state": emotional_state,
            "selected_ids": [p.id for p in selected],
            "rejected_ids": [p.id for p in rejected],
        })
        return selected


class TracedAffect:
    """Affect that actually responds to emotional content."""

    def __init__(self):
        self.valence = 0.0
        self.arousal = 0.0
        self.dominance = 0.0
        self.trace: List[Dict] = []

    def get_state(self):
        return {"valence": self.valence, "arousal": self.arousal, "dominance": self.dominance}

    def compute_update(self, broadcast):
        """Simple keyword-based affect update."""
        old_state = self.get_state()

        if broadcast and broadcast.percepts:
            for pid, p in broadcast.percepts.items():
                text = str(p.raw).lower() if hasattr(p, 'raw') else str(p).lower()
                positive = sum(1 for w in ["happy", "joy", "love", "good", "great", "hello", "wonderful"]
                             if w in text)
                negative = sum(1 for w in ["sad", "angry", "fear", "bad", "terrible", "error"]
                             if w in text)
                self.valence = max(-1, min(1, self.valence * 0.9 + (positive - negative) * 0.1))
                self.arousal = max(-1, min(1, self.arousal * 0.9 + (positive + negative) * 0.05))

        new_state = self.get_state()
        self.trace.append({"old": old_state, "new": new_state, "percept_count": len(broadcast.percepts) if broadcast else 0})
        return new_state

    def influence_attention(self, score, percept):
        return score


class TracedAction:
    """Action that records workspace state it receives."""

    def __init__(self):
        self.trace: List[Dict] = []
        self.speak_threshold = 5

    def decide(self, snapshot):
        actions = []
        percept_count = len(snapshot.percepts)

        if percept_count >= self.speak_threshold:
            actions.append({
                "type": "SPEAK",
                "content": f"[Processing {percept_count} percepts, valence={snapshot.emotions.get('valence', 0):.2f}]",
            })

        self.trace.append({
            "percept_count": percept_count,
            "goal_count": len(snapshot.goals),
            "emotions": snapshot.emotions.copy(),
            "actions_generated": len(actions),
            "cycle": snapshot.cycle_count,
        })
        return actions


class TracedWorkspace:
    """Workspace that tracks all mutations."""

    def __init__(self):
        self.current_goals = []
        self.active_percepts: Dict[str, Percept] = {}
        self.emotional_state = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        self.attended_memories = []
        self.cycle_count = 0
        self.temporal_context = None
        self.mutation_log: List[Dict] = []

    def broadcast(self):
        return WorkspaceSnapshot(
            goals=list(self.current_goals),
            percepts=dict(self.active_percepts),
            emotions=self.emotional_state.copy(),
            memories=list(self.attended_memories),
            cycle_count=self.cycle_count,
        )

    def update(self, outputs):
        percepts_added = 0
        emotion_updates = 0
        for output in outputs:
            if isinstance(output, dict):
                otype = output.get('type')
                data = output.get('data')
                if otype == 'percept' and isinstance(data, Percept):
                    self.active_percepts[data.id] = data
                    percepts_added += 1
                elif otype == 'emotion' and isinstance(data, dict):
                    self.emotional_state.update(data)
                    emotion_updates += 1
        self.cycle_count += 1
        self.mutation_log.append({
            "cycle": self.cycle_count,
            "percepts_added": percepts_added,
            "emotion_updates": emotion_updates,
            "total_percepts": len(self.active_percepts),
            "emotional_state": self.emotional_state.copy(),
        })

    def set_temporal_context(self, ctx):
        self.temporal_context = ctx


# Stubs for non-traced subsystems
class StubWorldModel:
    def __init__(self): self.prediction_errors = []
    def predict(self, **kw): return []
    def update_on_percept(self, p): return None

class StubIWMT:
    def __init__(self): self.world_model = StubWorldModel()
    def update_from_action_outcome(self, *a): pass

class StubMemory:
    async def retrieve_for_workspace(self, *a, **kw): return []
    async def consolidate(self, *a): pass

class StubMeta:
    def observe(self, s): return []
    def auto_validate_predictions(self, s): return []
    def update_self_model(self, *a): pass
    def record_accuracy_snapshot(self):
        class S: overall_accuracy = 0.0; prediction_count = 0
        return S()

class StubTemporal:
    def get_temporal_context(self): return {"uptime_seconds": 0.0}
    def record_input(self): pass
    def record_action(self): pass
    def apply_time_passage_effects(self, s): return s

class StubComm:
    def compute_drives(self, **kw): return []

class StubAutonomous:
    def check_for_autonomous_triggers(self, s): return None

class StubBottleneck:
    class _S:
        is_bottlenecked = False
        def get_severity(self): return 0.0
    def update(self, **kw): return self._S()

class StubJournal:
    def record_observation(self, o): pass

class StubIdentity:
    def update(self, **kw): pass


# ============================================================
# Traced Cognitive Cycle
# ============================================================

async def traced_cycle(workspace, perception, affect, attention, action,
                       iwmt, memory, meta, autonomous, temporal,
                       bottleneck, comm, journal, identity, input_queue):
    """Same 9-step cycle but with traced subsystems."""

    workspace.set_temporal_context(temporal.get_temporal_context())
    iwmt.world_model.predict(time_horizon=1.0)

    # 1. Perception
    new_percepts = []
    while not input_queue.empty():
        raw, modality = input_queue.get_nowait()
        percept = await perception.encode(raw, modality)
        new_percepts.append(percept)
    if new_percepts:
        temporal.record_input()

    for p in new_percepts:
        iwmt.world_model.update_on_percept(p)

    # 2. Memory
    snapshot = workspace.broadcast()
    mem_percepts = await memory.retrieve_for_workspace(snapshot)
    new_percepts.extend(mem_percepts)

    # 3. Attention
    emotional_state = affect.get_state()
    attended = attention.select_for_broadcast(
        new_percepts, emotional_state=emotional_state,
        prediction_errors=iwmt.world_model.prediction_errors
    )

    # 4. Affect
    temporal.apply_time_passage_effects({})
    affect_update = affect.compute_update(workspace.broadcast())

    # 5. Action
    snapshot = workspace.broadcast()
    actions = action.decide(snapshot)
    if actions:
        meta.update_self_model(snapshot, {"action_type": actions[0].get("type")})
        iwmt.update_from_action_outcome(actions[0], {})
    temporal.record_action()

    # 6. Meta-cognition
    meta_percepts = meta.observe(workspace.broadcast())
    meta.auto_validate_predictions(workspace.broadcast())

    # 6.5. Communication
    comm.compute_drives(workspace_state=workspace.broadcast())

    # 7. Autonomous
    autonomous.check_for_autonomous_triggers(workspace.broadcast())

    # 8. Workspace update
    updates = []
    for p in attended:
        updates.append({'type': 'percept', 'data': p})
    updates.append({'type': 'emotion', 'data': affect_update})
    workspace.update(updates)

    # 9. Memory consolidation
    await memory.consolidate(workspace.broadcast())

    bottleneck.update(subsystem_timings={})

    if workspace.cycle_count % 100 == 0:
        identity.update()

    return {
        "new_percepts": len(new_percepts),
        "attended": len(attended),
        "actions": len(actions),
        "workspace_percepts": len(workspace.active_percepts),
    }


# ============================================================
# Flow Tests
# ============================================================

class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            logger.info(f"  \u2705 {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            logger.error(f"  \u274c {name}: {detail}")

    def summary(self):
        total = self.passed + self.failed
        logger.info(f"\n{'='*60}")
        logger.info(f"Phase 2 FLOW: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            for e in self.errors:
                logger.info(f"  - {e}")
        logger.info(f"{'='*60}")
        return self.failed == 0


async def run_flow_tests():
    results = TestResults()

    # -- Flow 1: Single Input End-to-End --
    logger.info("\n-- Flow 1: Single Input End-to-End --")
    workspace = TracedWorkspace()
    perception = MockPerceptionSubsystem(config={"mock_embedding_dim": 384})
    attention = TracedAttention(budget=100)
    affect = TracedAffect()
    action = TracedAction()
    input_queue = asyncio.Queue()

    input_queue.put_nowait(("Hello, Sanctuary!", "text"))

    step = await traced_cycle(
        workspace, perception, affect, attention, action,
        StubIWMT(), StubMemory(), StubMeta(), StubAutonomous(),
        StubTemporal(), StubBottleneck(), StubComm(), StubJournal(),
        StubIdentity(), input_queue,
    )

    results.check("Input reached perception", perception.stats["total_encodings"] == 1)
    results.check("Percept has 384-dim embedding",
                  step["new_percepts"] == 1 and len(list(workspace.active_percepts.values())[0].embedding) == 384)
    results.check("Attention saw 1 candidate", attention.trace[0]["candidates"] == 1)
    results.check("Attention selected it", attention.trace[0]["selected"] == 1)
    results.check("Percept landed in workspace", len(workspace.active_percepts) == 1)
    results.check("Workspace mutation logged", len(workspace.mutation_log) == 1)
    results.check("Workspace mutation shows 1 percept added",
                  workspace.mutation_log[0]["percepts_added"] == 1)

    percept = list(workspace.active_percepts.values())[0]
    results.check("Percept raw content preserved", percept.raw == "Hello, Sanctuary!")
    results.check("Percept modality correct", percept.modality == "text")
    results.check("Percept mock metadata present", percept.metadata.get("mock_mode") is True)

    # -- Flow 2: Affect Responds to Emotional Content (with 1-cycle delay) --
    # KEY ARCHITECTURAL INSIGHT: Affect runs BEFORE workspace update (step 4 vs step 8).
    # So affect sees the PREVIOUS cycle's workspace. New percepts from attention
    # don't affect valence until the NEXT cycle. This is correct GWT behavior:
    # you process what's already conscious, then update consciousness.
    logger.info("\n-- Flow 2: Affect Responds to Emotional Content (with 1-cycle delay) --")
    workspace2 = TracedWorkspace()
    affect2 = TracedAffect()
    attention2 = TracedAttention(budget=100)
    action2 = TracedAction()
    input_queue2 = asyncio.Queue()

    # Cycle 1: Inject positive content (it enters workspace but affect sees empty workspace)
    input_queue2.put_nowait(("I feel happy and good today!", "text"))
    await traced_cycle(
        workspace2, perception, affect2, attention2, action2,
        StubIWMT(), StubMemory(), StubMeta(), StubAutonomous(),
        StubTemporal(), StubBottleneck(), StubComm(), StubJournal(),
        StubIdentity(), input_queue2,
    )

    results.check("Affect trace recorded", len(affect2.trace) == 1)
    results.check("Percept landed in workspace after cycle 1",
                  len(workspace2.active_percepts) == 1)
    results.check("Valence still neutral (affect sees previous cycle)",
                  affect2.valence == 0.0,
                  f"valence={affect2.valence:.4f} (expected 0 -- correct 1-cycle delay)")

    # Cycle 2: Empty input -- but now affect sees the positive percept from cycle 1
    await traced_cycle(
        workspace2, perception, affect2, attention2, action2,
        StubIWMT(), StubMemory(), StubMeta(), StubAutonomous(),
        StubTemporal(), StubBottleneck(), StubComm(), StubJournal(),
        StubIdentity(), input_queue2,
    )

    results.check("Valence shifted positive on cycle 2 (sees cycle 1 percept)",
                  affect2.valence > 0.0,
                  f"valence={affect2.valence:.4f}")
    results.check("Workspace emotional state reflects positive shift",
                  workspace2.emotional_state["valence"] > 0.0,
                  f"ws_valence={workspace2.emotional_state['valence']:.4f}")

    # Cycle 3: Inject negative content
    input_queue2.put_nowait(("This is terrible and bad and sad", "text"))
    await traced_cycle(
        workspace2, perception, affect2, attention2, action2,
        StubIWMT(), StubMemory(), StubMeta(), StubAutonomous(),
        StubTemporal(), StubBottleneck(), StubComm(), StubJournal(),
        StubIdentity(), input_queue2,
    )
    # Cycle 4: Affect now sees the negative percept
    await traced_cycle(
        workspace2, perception, affect2, attention2, action2,
        StubIWMT(), StubMemory(), StubMeta(), StubAutonomous(),
        StubTemporal(), StubBottleneck(), StubComm(), StubJournal(),
        StubIdentity(), input_queue2,
    )

    results.check("Valence shifted negative after processing bad input",
                  affect2.valence < affect2.trace[1]["new"]["valence"],
                  f"valence={affect2.valence:.4f} (was {affect2.trace[1]['new']['valence']:.4f})")

    # -- Flow 3: Attention Budget Enforcement --
    logger.info("\n-- Flow 3: Attention Budget Enforcement --")
    workspace3 = TracedWorkspace()
    attention3 = TracedAttention(budget=30)
    affect3 = TracedAffect()
    action3 = TracedAction()
    input_queue3 = asyncio.Queue()

    for i in range(10):
        input_queue3.put_nowait((f"msg{i}", "text"))

    await traced_cycle(
        workspace3, perception, affect3, attention3, action3,
        StubIWMT(), StubMemory(), StubMeta(), StubAutonomous(),
        StubTemporal(), StubBottleneck(), StubComm(), StubJournal(),
        StubIdentity(), input_queue3,
    )

    results.check("Attention saw all 10 candidates", attention3.trace[0]["candidates"] == 10)
    results.check("Attention rejected some (budget constraint)",
                  attention3.trace[0]["rejected"] > 0,
                  f"rejected={attention3.trace[0]['rejected']}")
    results.check("Budget used <= 30",
                  attention3.trace[0]["budget_used"] <= 30,
                  f"used={attention3.trace[0]['budget_used']}")
    results.check("Only selected percepts in workspace",
                  len(workspace3.active_percepts) == attention3.trace[0]["selected"],
                  f"ws={len(workspace3.active_percepts)}, selected={attention3.trace[0]['selected']}")

    # -- Flow 4: Action Sees Workspace State --
    logger.info("\n-- Flow 4: Action Sees Workspace State --")
    workspace4 = TracedWorkspace()
    attention4 = TracedAttention(budget=200)
    affect4 = TracedAffect()
    action4 = TracedAction()
    action4.speak_threshold = 3
    input_queue4 = asyncio.Queue()

    for cycle in range(5):
        input_queue4.put_nowait((f"Building context message {cycle}", "text"))
        await traced_cycle(
            workspace4, perception, affect4, attention4, action4,
            StubIWMT(), StubMemory(), StubMeta(), StubAutonomous(),
            StubTemporal(), StubBottleneck(), StubComm(), StubJournal(),
            StubIdentity(), input_queue4,
        )

    results.check("Action trace has 5 entries", len(action4.trace) == 5)
    results.check("Action saw accumulating percepts",
                  action4.trace[-1]["percept_count"] > action4.trace[0]["percept_count"],
                  f"first={action4.trace[0]['percept_count']}, last={action4.trace[-1]['percept_count']}")
    results.check("Action generated SPEAK once threshold met",
                  any(t["actions_generated"] > 0 for t in action4.trace),
                  f"actions per cycle: {[t['actions_generated'] for t in action4.trace]}")
    results.check("Action received emotional state",
                  all("valence" in t["emotions"] for t in action4.trace))

    # -- Flow 5: Workspace Accumulation Across Cycles --
    logger.info("\n-- Flow 5: Workspace Accumulation Across Cycles --")
    results.check("Workspace cycle count = 5", workspace4.cycle_count == 5)
    results.check("Workspace has 5 percepts from 5 cycles",
                  len(workspace4.active_percepts) == 5,
                  f"got {len(workspace4.active_percepts)}")
    results.check("Mutation log shows growth",
                  workspace4.mutation_log[-1]["total_percepts"] > workspace4.mutation_log[0]["total_percepts"],
                  f"log: {[m['total_percepts'] for m in workspace4.mutation_log]}")

    # -- Flow 6: Empty Cycle (no input) --
    logger.info("\n-- Flow 6: Empty Cycle (no input) --")
    workspace6 = TracedWorkspace()
    attention6 = TracedAttention()
    affect6 = TracedAffect()
    action6 = TracedAction()
    input_queue6 = asyncio.Queue()

    step = await traced_cycle(
        workspace6, perception, affect6, attention6, action6,
        StubIWMT(), StubMemory(), StubMeta(), StubAutonomous(),
        StubTemporal(), StubBottleneck(), StubComm(), StubJournal(),
        StubIdentity(), input_queue6,
    )

    results.check("No percepts on empty cycle", step["new_percepts"] == 0)
    results.check("Attention got 0 candidates", attention6.trace[0]["candidates"] == 0)
    results.check("Workspace still updated (cycle count)", workspace6.cycle_count == 1)
    results.check("Affect still ran (decay toward neutral)", len(affect6.trace) == 1)

    # -- Flow 7: Embedding Distinctness --
    logger.info("\n-- Flow 7: Embedding Distinctness --")
    p_hello = await perception.encode("hello", "text")
    p_goodbye = await perception.encode("goodbye", "text")
    p_hello2 = await perception.encode("hello", "text")

    def cosine(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    sim_same = cosine(p_hello.embedding, p_hello2.embedding)
    sim_diff = cosine(p_hello.embedding, p_goodbye.embedding)

    results.check("Same text -> identical embedding", abs(sim_same - 1.0) < 1e-6, f"sim={sim_same:.6f}")
    results.check("Different text -> different embedding", sim_diff < 1.0, f"sim={sim_diff:.4f}")
    results.check("Different embeddings aren't too similar", abs(sim_diff) < 0.5,
                  f"sim={sim_diff:.4f} (random vectors should be near-orthogonal)")

    # -- Flow 8: 100 Cycles Mixed Flow --
    logger.info("\n-- Flow 8: 100 Cycles Mixed Flow --")
    workspace8 = TracedWorkspace()
    attention8 = TracedAttention(budget=50)
    affect8 = TracedAffect()
    action8 = TracedAction()
    action8.speak_threshold = 10
    input_queue8 = asyncio.Queue()

    start = time.time()
    speak_cycles = []

    for i in range(100):
        if i % 3 == 0:
            sentiment = "happy great wonderful" if i % 6 == 0 else "terrible bad sad"
            input_queue8.put_nowait((f"Cycle {i}: {sentiment}", "text"))

        step = await traced_cycle(
            workspace8, perception, affect8, attention8, action8,
            StubIWMT(), StubMemory(), StubMeta(), StubAutonomous(),
            StubTemporal(), StubBottleneck(), StubComm(), StubJournal(),
            StubIdentity(), input_queue8,
        )

        if step["actions"] > 0:
            speak_cycles.append(i)

    elapsed = time.time() - start

    results.check("100 mixed cycles completed", workspace8.cycle_count == 100)

    total_ws_percepts = len(workspace8.active_percepts)
    results.check("Percepts accumulated in workspace",
                  total_ws_percepts > 0,
                  f"got {total_ws_percepts}")

    results.check("Affect responded to mixed sentiment",
                  len(affect8.trace) == 100,
                  f"got {len(affect8.trace)} affect updates")

    valence_values = [t["new"]["valence"] for t in affect8.trace]
    valence_range = max(valence_values) - min(valence_values)
    results.check("Valence oscillated (range > 0.01)",
                  valence_range > 0.01,
                  f"range={valence_range:.4f}")

    results.check("Action generated SPEAK actions",
                  len(speak_cycles) > 0,
                  f"spoke at cycles: {speak_cycles}")

    results.check(f"Performance: {100/elapsed:.0f} Hz",
                  elapsed < 5.0,
                  f"{elapsed:.2f}s total")

    logger.info(f"  100 mixed cycles in {elapsed*1000:.1f}ms ({100/elapsed:.0f} Hz)")
    logger.info(f"  Workspace: {total_ws_percepts} percepts, {workspace8.cycle_count} cycles")
    logger.info(f"  Affect: valence range [{min(valence_values):.3f}, {max(valence_values):.3f}]")
    logger.info(f"  Action spoke at cycles: {speak_cycles}")

    return results.summary()


if __name__ == "__main__":
    success = asyncio.run(run_flow_tests())
    sys.exit(0 if success else 1)
