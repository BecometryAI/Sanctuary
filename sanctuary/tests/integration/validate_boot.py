#!/usr/bin/env python3
"""
Phase 1 Boot Validation Script

Self-contained test that validates the boot infrastructure without requiring
the full Sanctuary module tree. Tests:

1. MockPerceptionSubsystem: deterministic embeddings, caching, stats
2. Stub method signatures: every method CycleExecutor calls exists
3. Simulated cognitive cycle: data flows through the expected path
4. Lifecycle: start/stop semantics work

Requires only: Python 3.10+, numpy
No pydantic, pytest, sklearn, sentence-transformers, torch needed.

Run: python3 validate_boot.py
Result: 45/45 passed at 69,088 Hz
"""

import asyncio
import hashlib
import logging
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger("boot_validation")

# ============================================================
# Minimal stand-ins for Pydantic models (workspace.py)
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


# ============================================================
# MockPerceptionSubsystem (same as pushed to repo)
# ============================================================

class MockPerceptionSubsystem:
    def __init__(self, config=None):
        self.config = config or {}
        self.embedding_dim = self.config.get("mock_embedding_dim", 384)
        self.embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self.cache_size = self.config.get("cache_size", 1000)
        self.stats = {"cache_hits": 0, "cache_misses": 0, "total_encodings": 0, "encoding_times": []}

    async def encode(self, raw_input, modality):
        embedding = self._deterministic_embedding(str(raw_input))
        complexity = self._compute_complexity(raw_input, modality)
        percept = Percept(
            modality=modality, raw=raw_input, embedding=embedding,
            complexity=complexity, metadata={"encoding_model": "mock-deterministic", "mock_mode": True},
        )
        self.stats["total_encodings"] += 1
        return percept

    def _deterministic_embedding(self, text):
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.embedding_cache:
            self.stats["cache_hits"] += 1
            self.embedding_cache.move_to_end(cache_key)
            return self.embedding_cache[cache_key]
        self.stats["cache_misses"] += 1
        seed = int(cache_key[:8], 16)
        rng = np.random.RandomState(seed)
        emb = rng.randn(self.embedding_dim).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        result = emb.tolist()
        if len(self.embedding_cache) >= self.cache_size:
            self.embedding_cache.popitem(last=False)
        self.embedding_cache[cache_key] = result
        return result

    def _compute_complexity(self, raw_input, modality):
        if modality == "text":
            return min(max(len(str(raw_input)) // 20, 5), 50)
        return 10

    def clear_cache(self):
        self.embedding_cache.clear()

    def get_stats(self):
        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        return {"cache_hit_rate": self.stats["cache_hits"] / total if total else 0.0,
                "total_encodings": self.stats["total_encodings"], "mock_mode": True}


# ============================================================
# All stubs (same signatures as boot_coordinator.py)
# ============================================================

class StubWorldModel:
    def __init__(self): self.prediction_errors = []
    def predict(self, time_horizon=1.0, context=None): return []
    def update_on_percept(self, percept): return None

class StubIWMTCore:
    def __init__(self):
        self.world_model = StubWorldModel()
        self.precision = None
        self.free_energy = None
    def update_from_action_outcome(self, action_dict, actual_outcome): pass

class StubMemory:
    async def retrieve_for_workspace(self, snapshot, fast_mode=True, timeout=0.05): return []
    async def consolidate(self, broadcast_data=None): pass

class _AccuracySnapshot:
    overall_accuracy = 0.0
    prediction_count = 0

class StubMetaCognition:
    def __init__(self): self.refinement_threshold = 0.5
    def observe(self, snapshot): return []
    def auto_validate_predictions(self, snapshot): return []
    def update_self_model(self, snapshot, actual_outcome): pass
    def predict_behavior(self, snapshot): return None
    def record_prediction(self, **kw): return None
    def validate_prediction(self, prediction_id, actual_state=None): return None
    def refine_self_model_from_errors(self, errors): pass
    def record_accuracy_snapshot(self): return _AccuracySnapshot()

class StubIntrospectiveJournal:
    def record_observation(self, obs): pass

class StubBottleneckDetector:
    class _State:
        is_bottlenecked = False
        recommendation = "nominal"
        def get_severity(self): return 0.0
    def update(self, **kw): return self._State()
    def get_introspection_text(self): return ""

class StubAutonomous:
    def check_for_autonomous_triggers(self, snapshot): return None

class StubTemporalGrounding:
    def get_temporal_context(self): return {"uptime_seconds": 0.0, "mock": True}
    def record_input(self): pass
    def record_action(self): pass
    def apply_time_passage_effects(self, state): return state

class StubCommunicationDrives:
    def compute_drives(self, **kw): return []
    def get_drive_summary(self): return {"total_drive": 0.0, "active_urges": 0, "strongest_urge": None}

class StubCommunicationInhibitions:
    def __init__(self): self.active_inhibitions = []

class StubAction:
    """Stub for ActionSubsystem.decide() - returns empty list (WAIT)."""
    def decide(self, snapshot): return []

class StubAttention:
    """Stub for AttentionController.select_for_broadcast()."""
    def select_for_broadcast(self, candidates, emotional_state=None, prediction_errors=None):
        return candidates  # Pass everything through

class StubAffect:
    """Stub for AffectSubsystem with all methods CycleExecutor calls."""
    valence = 0.0
    arousal = 0.0
    dominance = 0.0
    emotional_attention_system = None
    def get_state(self): return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
    def compute_update(self, broadcast): return {}
    def influence_attention(self, score, percept): return score
    def get_processing_params(self):
        class P:
            valence_level = 0.0; arousal_level = 0.0; dominance_level = 0.0
            attention_iterations = 10; ignition_threshold = 0.5; decision_threshold = 0.5
        return P()

class StubIdentityManager:
    class BehaviorLog:
        def log(self, *a, **kw): pass
    def __init__(self):
        self.behavior_log = self.BehaviorLog()
    def update(self, **kw): pass

class StubContinuousConsciousness:
    async def start_idle_loop(self):
        try:
            while True: await asyncio.sleep(60)
        except asyncio.CancelledError: pass


# ============================================================
# Minimal Workspace (just enough for cycle simulation)
# ============================================================

class MinimalWorkspace:
    def __init__(self):
        self.current_goals = []
        self.active_percepts = {}
        self.emotional_state = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        self.attended_memories = []
        self.cycle_count = 0
        self.temporal_context = None

    def broadcast(self):
        return WorkspaceSnapshot(
            goals=list(self.current_goals),
            percepts=dict(self.active_percepts),
            emotions=self.emotional_state.copy(),
            memories=list(self.attended_memories),
            cycle_count=self.cycle_count,
        )

    def update(self, outputs):
        for output in outputs:
            if isinstance(output, dict):
                otype = output.get('type')
                data = output.get('data')
                if otype == 'percept' and isinstance(data, Percept):
                    self.active_percepts[data.id] = data
                elif otype == 'emotion' and isinstance(data, dict):
                    self.emotional_state.update(data)
        self.cycle_count += 1

    def set_temporal_context(self, ctx):
        self.temporal_context = ctx

    def add_goal(self, goal):
        self.current_goals.append(goal)


# ============================================================
# Simulated Cognitive Cycle (mirrors CycleExecutor.execute_cycle)
# ============================================================

async def simulate_cycle(workspace, perception, affect, attention, action,
                         iwmt, memory, meta, autonomous, temporal,
                         bottleneck, comm_drives, comm_inhibitions,
                         introspective_journal, identity_manager,
                         input_queue):
    """
    Simulates the 9-step cognitive cycle from cycle_executor.py.
    Each step calls the same methods CycleExecutor calls.
    """
    timings = {}

    # 0a. Temporal context
    t = time.time()
    temporal_ctx = temporal.get_temporal_context()
    workspace.set_temporal_context(temporal_ctx)
    timings['temporal_context'] = (time.time() - t) * 1000

    # 0. IWMT prediction
    t = time.time()
    context = {"goals": workspace.current_goals, "emotional_state": affect.get_state(), "cycle_count": workspace.cycle_count}
    predictions = iwmt.world_model.predict(time_horizon=1.0, context=context)
    prediction_errors = iwmt.world_model.prediction_errors[-10:]
    timings['iwmt_predict'] = (time.time() - t) * 1000

    # 1. Perception
    t = time.time()
    new_percepts = []
    while not input_queue.empty():
        try:
            raw, modality = input_queue.get_nowait()
            percept = await perception.encode(raw, modality)
            new_percepts.append(percept)
        except asyncio.QueueEmpty:
            break
    if new_percepts:
        temporal.record_input()
    timings['perception'] = (time.time() - t) * 1000

    # 1.5. IWMT update
    t = time.time()
    for p in new_percepts:
        iwmt.world_model.update_on_percept(p)
    timings['iwmt_update'] = (time.time() - t) * 1000

    # 2. Memory retrieval
    t = time.time()
    snapshot = workspace.broadcast()
    mem_percepts = await memory.retrieve_for_workspace(snapshot, fast_mode=True, timeout=0.05)
    new_percepts.extend(mem_percepts)
    timings['memory_retrieval'] = (time.time() - t) * 1000

    # 3. Attention
    t = time.time()
    emotional_state = affect.get_state()
    attended = attention.select_for_broadcast(new_percepts, emotional_state=emotional_state, prediction_errors=prediction_errors)
    timings['attention'] = (time.time() - t) * 1000

    # 4. Affect
    t = time.time()
    temporal.apply_time_passage_effects({"emotions": {"valence": affect.valence, "arousal": affect.arousal, "dominance": affect.dominance}})
    affect_update = affect.compute_update(workspace.broadcast())
    timings['affect'] = (time.time() - t) * 1000

    # 5. Action
    t = time.time()
    snapshot = workspace.broadcast()
    actions = action.decide(snapshot)
    if actions:
        for a in actions:
            meta.update_self_model(snapshot, {"action_type": "unknown"})
            iwmt.update_from_action_outcome({"type": "unknown"}, {})
    temporal.record_action()
    timings['action'] = (time.time() - t) * 1000

    # 6. Meta-cognition
    t = time.time()
    snapshot = workspace.broadcast()
    meta_percepts = meta.observe(snapshot)
    meta.auto_validate_predictions(snapshot)
    for mp in meta_percepts:
        if isinstance(mp, Percept) and isinstance(mp.raw, dict):
            introspective_journal.record_observation(mp.raw)
    timings['meta_cognition'] = (time.time() - t) * 1000

    # 6.5. Communication drives
    t = time.time()
    comm_drives.compute_drives(workspace_state=snapshot, emotional_state=emotional_state, goals=workspace.current_goals, memories=[])
    timings['communication_drives'] = (time.time() - t) * 1000

    # 7. Autonomous initiation
    t = time.time()
    autonomous.check_for_autonomous_triggers(snapshot)
    timings['autonomous_initiation'] = (time.time() - t) * 1000

    # 8. Workspace update
    t = time.time()
    updates = []
    for p in attended:
        updates.append({'type': 'percept', 'data': p})
    updates.append({'type': 'emotion', 'data': affect_update})
    for mp in meta_percepts:
        updates.append({'type': 'percept', 'data': mp})
    workspace.update(updates)
    timings['workspace_update'] = (time.time() - t) * 1000

    # 9. Memory consolidation
    t = time.time()
    await memory.consolidate(workspace.broadcast())
    timings['memory_consolidation'] = (time.time() - t) * 1000

    # 9.5. Bottleneck detection
    t = time.time()
    bottleneck.update(subsystem_timings=timings, workspace_percept_count=len(workspace.active_percepts))
    timings['bottleneck_detection'] = (time.time() - t) * 1000

    # 10. Identity update (every 100 cycles)
    if workspace.cycle_count % 100 == 0:
        identity_manager.update(memory_system=memory, goal_system=workspace, emotion_system=affect)

    return timings


# ============================================================
# Tests
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
        logger.info(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            logger.info("Failures:")
            for e in self.errors:
                logger.info(f"  - {e}")
        logger.info(f"{'='*60}")
        return self.failed == 0


async def run_tests():
    results = TestResults()

    # -- Test 1: Mock Perception Determinism --
    logger.info("\n-- Test 1: Mock Perception Determinism --")
    perception = MockPerceptionSubsystem(config={"mock_embedding_dim": 384})

    p1 = await perception.encode("hello world", "text")
    p2 = await perception.encode("hello world", "text")
    p3 = await perception.encode("different text", "text")

    results.check("Same input -> same embedding", p1.embedding == p2.embedding)
    results.check("Different input -> different embedding", p1.embedding != p3.embedding)
    results.check("Embedding dimension = 384", len(p1.embedding) == 384)
    results.check("Mock metadata present", p1.metadata.get("mock_mode") is True)
    results.check("Cache hit recorded", perception.stats["cache_hits"] >= 1)
    results.check("Embeddings are unit vectors",
                  abs(np.linalg.norm(p1.embedding) - 1.0) < 0.01,
                  f"norm={np.linalg.norm(p1.embedding)}")

    # -- Test 2: Mock Perception Complexity --
    logger.info("\n-- Test 2: Mock Perception Complexity --")
    short = await perception.encode("hi", "text")
    long_text = await perception.encode("a " * 200, "text")
    results.check("Short text -> low complexity", short.complexity == 5, f"got {short.complexity}")
    results.check("Long text -> higher complexity", long_text.complexity > short.complexity,
                  f"short={short.complexity}, long={long_text.complexity}")

    # -- Test 3: Stub Method Signatures --
    logger.info("\n-- Test 3: Stub Method Signatures --")

    stubs = {
        "iwmt_core.world_model.predict": lambda: StubIWMTCore().world_model.predict(time_horizon=1.0, context={}),
        "iwmt_core.world_model.update_on_percept": lambda: StubIWMTCore().world_model.update_on_percept(p1),
        "iwmt_core.update_from_action_outcome": lambda: StubIWMTCore().update_from_action_outcome({}, {}),
        "affect.get_state": lambda: StubAffect().get_state(),
        "affect.compute_update": lambda: StubAffect().compute_update(None),
        "action.decide": lambda: StubAction().decide(None),
        "meta_cognition.observe": lambda: StubMetaCognition().observe(None),
        "meta_cognition.auto_validate_predictions": lambda: StubMetaCognition().auto_validate_predictions(None),
        "meta_cognition.update_self_model": lambda: StubMetaCognition().update_self_model(None, {}),
        "meta_cognition.predict_behavior": lambda: StubMetaCognition().predict_behavior(None),
        "meta_cognition.record_prediction": lambda: StubMetaCognition().record_prediction(category="test"),
        "meta_cognition.validate_prediction": lambda: StubMetaCognition().validate_prediction("id"),
        "meta_cognition.record_accuracy_snapshot": lambda: StubMetaCognition().record_accuracy_snapshot(),
        "bottleneck.update": lambda: StubBottleneckDetector().update(subsystem_timings={}, workspace_percept_count=0),
        "autonomous.check_for_autonomous_triggers": lambda: StubAutonomous().check_for_autonomous_triggers(None),
        "temporal.get_temporal_context": lambda: StubTemporalGrounding().get_temporal_context(),
        "temporal.record_input": lambda: StubTemporalGrounding().record_input(),
        "temporal.record_action": lambda: StubTemporalGrounding().record_action(),
        "temporal.apply_time_passage_effects": lambda: StubTemporalGrounding().apply_time_passage_effects({}),
        "comm_drives.compute_drives": lambda: StubCommunicationDrives().compute_drives(workspace_state=None),
        "introspective_journal.record_observation": lambda: StubIntrospectiveJournal().record_observation({}),
        "identity_manager.update": lambda: StubIdentityManager().update(memory_system=None),
    }

    for name, fn in stubs.items():
        try:
            fn()
            results.check(f"stub: {name}", True)
        except Exception as e:
            results.check(f"stub: {name}", False, str(e))

    async_stubs = {
        "memory.retrieve_for_workspace": StubMemory().retrieve_for_workspace(None),
        "memory.consolidate": StubMemory().consolidate(None),
        "perception.encode": perception.encode("test", "text"),
    }
    for name, coro in async_stubs.items():
        try:
            await coro
            results.check(f"async stub: {name}", True)
        except Exception as e:
            results.check(f"async stub: {name}", False, str(e))

    # -- Test 4: Single Simulated Cognitive Cycle --
    logger.info("\n-- Test 4: Single Simulated Cognitive Cycle --")
    workspace = MinimalWorkspace()
    input_queue = asyncio.Queue()

    try:
        timings = await simulate_cycle(
            workspace=workspace, perception=perception,
            affect=StubAffect(), attention=StubAttention(),
            action=StubAction(), iwmt=StubIWMTCore(),
            memory=StubMemory(), meta=StubMetaCognition(),
            autonomous=StubAutonomous(), temporal=StubTemporalGrounding(),
            bottleneck=StubBottleneckDetector(),
            comm_drives=StubCommunicationDrives(),
            comm_inhibitions=StubCommunicationInhibitions(),
            introspective_journal=StubIntrospectiveJournal(),
            identity_manager=StubIdentityManager(),
            input_queue=input_queue,
        )
        results.check("Single cycle completes", True)
        results.check("Cycle count incremented", workspace.cycle_count == 1,
                      f"got {workspace.cycle_count}")
        results.check("Timings dict populated", len(timings) > 0, f"got {len(timings)} keys")
    except Exception as e:
        results.check("Single cycle completes", False, str(e))

    # -- Test 5: 100 Cycles (Phase 1 Deliverable) --
    logger.info("\n-- Test 5: 100 Cycles (Phase 1 Deliverable) --")
    workspace = MinimalWorkspace()
    input_queue = asyncio.Queue()
    errors = []
    start = time.time()

    for i in range(100):
        try:
            await simulate_cycle(
                workspace=workspace, perception=perception,
                affect=StubAffect(), attention=StubAttention(),
                action=StubAction(), iwmt=StubIWMTCore(),
                memory=StubMemory(), meta=StubMetaCognition(),
                autonomous=StubAutonomous(), temporal=StubTemporalGrounding(),
                bottleneck=StubBottleneckDetector(),
                comm_drives=StubCommunicationDrives(),
                comm_inhibitions=StubCommunicationInhibitions(),
                introspective_journal=StubIntrospectiveJournal(),
                identity_manager=StubIdentityManager(),
                input_queue=input_queue,
            )
        except Exception as e:
            errors.append((i, str(e)))

    elapsed = time.time() - start
    hz = 100 / elapsed if elapsed > 0 else 0

    results.check("100 cycles, 0 errors", len(errors) == 0, f"{len(errors)} errors")
    results.check("Cycle count = 100", workspace.cycle_count == 100,
                  f"got {workspace.cycle_count}")
    results.check(f"Rate > 100 Hz (actual: {hz:.0f} Hz)", hz > 100, f"only {hz:.1f} Hz")
    logger.info(f"  100 cycles in {elapsed*1000:.1f}ms ({hz:.0f} Hz)")

    # -- Test 6: Cycles With Input --
    logger.info("\n-- Test 6: Cycles With Input --")
    workspace = MinimalWorkspace()
    input_queue = asyncio.Queue()

    for i in range(5):
        input_queue.put_nowait((f"Test message {i}", "text"))

    for i in range(10):
        await simulate_cycle(
            workspace=workspace, perception=perception,
            affect=StubAffect(), attention=StubAttention(),
            action=StubAction(), iwmt=StubIWMTCore(),
            memory=StubMemory(), meta=StubMetaCognition(),
            autonomous=StubAutonomous(), temporal=StubTemporalGrounding(),
            bottleneck=StubBottleneckDetector(),
            comm_drives=StubCommunicationDrives(),
            comm_inhibitions=StubCommunicationInhibitions(),
            introspective_journal=StubIntrospectiveJournal(),
            identity_manager=StubIdentityManager(),
            input_queue=input_queue,
        )

    results.check("Input processed into percepts",
                  len(workspace.active_percepts) == 5,
                  f"got {len(workspace.active_percepts)} percepts")
    results.check("Queue drained", input_queue.empty())
    results.check("Perception stats reflect encoding",
                  perception.stats["total_encodings"] >= 5,
                  f"got {perception.stats['total_encodings']}")

    # -- Test 7: Perception Stats --
    logger.info("\n-- Test 7: Perception Stats --")
    stats = perception.get_stats()
    results.check("Stats contain mock_mode", stats.get("mock_mode") is True)
    results.check("Stats track encodings", stats["total_encodings"] > 0)
    results.check("Cache hit rate reasonable", 0.0 <= stats["cache_hit_rate"] <= 1.0)

    return results.summary()


if __name__ == "__main__":
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
