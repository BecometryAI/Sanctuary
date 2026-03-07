"""
Minimal Boot Coordinator for Phase 1 integration testing.

Real: affect, attention, action, meta-cognition (+ mock perception)
Stub: everything else, with method signatures matched to cycle_executor.py
and cognitive_loop.py
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Any, Optional

from ..workspace import GlobalWorkspace
from ..attention import AttentionController
from ..affect import AffectSubsystem
from ..action import ActionSubsystem
from ..mock_perception import MockPerceptionSubsystem
from ..meta_cognition import SelfMonitor

logger = logging.getLogger(__name__)


# === Stubs matched to CycleExecutor + CognitiveLoop method calls ===

class StubIdentity:
    def __init__(self):
        self.charter, self.protocols, self.values = {}, {}, []
        self.name = "sanctuary-boot"
    def load_all(self): pass
    def get_value(self, key, default=None): return default
    def get_charter_section(self, section): return ""
    def get_protocol(self, name): return None

class StubBehaviorLog:
    def log(self, *a, **kw): pass
    def get_recent(self, n=10): return []

class StubIdentityManager:
    def __init__(self):
        self.behavior_log = StubBehaviorLog()
        self.identity_vector = {}
    def update(self, **kwargs): pass

class StubWorldModel:
    def __init__(self):
        self.prediction_errors = []
    def predict(self, time_horizon=1.0, context=None): return []
    def update_on_percept(self, percept): return None

class StubIWMTCore:
    def __init__(self):
        self.world_model = StubWorldModel()
        self.precision = None
        self.free_energy = None
    def update_from_action_outcome(self, action_dict, actual_outcome): pass

class StubMemory:
    async def retrieve_for_workspace(self, snapshot, fast_mode=True, timeout=0.05):
        return []
    async def consolidate(self, broadcast_data=None): pass


class StubIntrospectiveJournal:
    def record_observation(self, obs): pass
    def get_recent(self, n=5): return []

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
    def on_interaction(self):
        class _Ctx:
            is_new_session = False
            session_number = 0
        return _Ctx()

class StubTemporalAwareness:
    def __init__(self): self.uptime = 0.0
    def update(self, cycle_time=0.0): self.uptime += cycle_time
    def get_temporal_context(self): return {"uptime_seconds": self.uptime}
    def update_last_interaction_time(self): pass

class StubCommunicationDrives:
    def compute_drives(self, **kw): return []
    def get_drive_summary(self):
        return {"total_drive": 0.0, "active_urges": 0, "strongest_urge": None}
    def get_active_drives(self): return []
    def record_input(self): pass
    def record_output(self): pass

class StubCommunicationInhibitions:
    def __init__(self): self.active_inhibitions = []

class StubLanguageInput:
    async def parse(self, text, context=None):
        from ..workspace import Percept as WP
        from datetime import datetime
        class _Result:
            goals = []
            percept = WP(modality="text", raw=text, embedding=[0.0]*384,
                        complexity=5, timestamp=datetime.now())
        return _Result()

class StubLanguageOutput:
    async def generate(self, ws, context=None): return "[boot mode]"

class StubCheckpointManager:
    def save(self, *a, **kw): return None
    def restore(self, *a, **kw): return False

class StubContinuousConsciousness:
    async def start_idle_loop(self):
        """Run until externally cancelled - matches CognitiveLoop.run() expectations."""
        try:
            while True:
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            pass
    async def update(self, *a, **kw): pass

class StubIntrospectiveLoop:
    async def check_triggers(self, *a, **kw): return False
    async def reflect(self, *a, **kw): return None


# === Boot Coordinator ===

class BootCoordinator:
    """
    Minimal subsystem coordinator for Phase 1 boot testing.
    Real: affect, attention, action, meta-cognition (+ mock perception). Stub: everything else.
    """

    def __init__(self, workspace: GlobalWorkspace, config: Dict[str, Any]):
        self.config = config
        self.workspace = workspace

        logger.info("\U0001f680 BootCoordinator: initializing...")

        self.identity = StubIdentity()
        self.identity_manager = StubIdentityManager()

        self.affect = AffectSubsystem(config=config.get("affect", {}))
        logger.info("  \u2705 AffectSubsystem (real)")

        self.iwmt_core = StubIWMTCore()

        self.attention = AttentionController(
            attention_budget=config.get("attention_budget", 100),
            workspace=workspace,
            affect=self.affect,
            precision_weighting=None,
            emotional_attention=self.affect.emotional_attention_system,
        )
        logger.info("  \u2705 AttentionController (real)")

        self.perception = MockPerceptionSubsystem(config=config.get("perception", {}))
        logger.info("  \u2705 PerceptionSubsystem (mock)")

        self.action = ActionSubsystem(
            config=config.get("action", {}),
            affect=self.affect,
            identity=self.identity,
            behavior_logger=self.identity_manager.behavior_log,
        )
        logger.info("  \u2705 ActionSubsystem (real)")

        workspace.affect = self.affect
        workspace.action_subsystem = self.action
        workspace.perception = self.perception

        self.meta_cognition = SelfMonitor(workspace=workspace)
        logger.info("  ✅ SelfMonitor (real)")
        self.introspective_journal = StubIntrospectiveJournal()
        self.bottleneck_detector = StubBottleneckDetector()
        self.memory = StubMemory()
        self.autonomous = StubAutonomous()
        self.temporal_awareness = StubTemporalAwareness()
        self.temporal_grounding = StubTemporalGrounding()
        self.memory_review = None
        self.existential_reflection = None
        self.pattern_analysis = None
        self.introspective_loop = StubIntrospectiveLoop()
        self.communication_drives = StubCommunicationDrives()
        self.communication_inhibitions = StubCommunicationInhibitions()
        self.communication_decision = None
        self.language_input = StubLanguageInput()
        self.language_output = StubLanguageOutput()
        self.checkpoint_manager = StubCheckpointManager()
        self.continuous_consciousness = StubContinuousConsciousness()
        self.device_registry = None

        logger.info("\U0001f680 BootCoordinator ready: 4 real + mock perception + stubs")

    def initialize_continuous_consciousness(self, cognitive_core):
        return StubContinuousConsciousness()
