"""Phase 1 Boot Configuration.

Minimal configuration for getting CognitiveCore to instantiate and cycle
without requiring heavy ML dependencies (sentence-transformers, torch, etc.)
or external data files.

Usage:
    from sanctuary.mind.cognitive_core.boot_config import create_boot_config
    config = create_boot_config()
    core = CognitiveCore(config=config)
"""

from pathlib import Path
from typing import Dict, Any, Optional
import tempfile


def create_boot_config(
    temp_dir: Optional[Path] = None,
    cycle_rate_hz: float = 10.0,
    mock_perception: bool = True,
) -> Dict[str, Any]:\n    """\n    Create a minimal configuration for Phase 1 boot.\n\n    Args:\n        temp_dir: Base directory for data files. If None, creates a temp dir.\n        cycle_rate_hz: Target cognitive cycle rate.\n        mock_perception: Use mock perception (no sentence-transformers needed).\n\n    Returns:\n        Configuration dict suitable for CognitiveCore.__init__.\n    """\n    if temp_dir is None:\n        temp_dir = Path(tempfile.mkdtemp(prefix="sanctuary_boot_"))\n\n    # Create required directories\n    identity_dir = temp_dir / "identity"\n    journal_dir = temp_dir / "introspection"\n    checkpoint_dir = temp_dir / "checkpoints"\n\n    for d in [identity_dir, journal_dir, checkpoint_dir]:\n        d.mkdir(parents=True, exist_ok=True)\n\n    return {\n        # Core loop settings\n        "cycle_rate_hz": cycle_rate_hz,\n        "attention_budget": 100,\n        "max_queue_size": 100,\n        "log_interval_cycles": 50,\n\n        # Filesystem paths\n        "identity_dir": str(identity_dir),\n        "journal_dir": str(journal_dir),\n\n        # Perception - use mock mode to avoid sentence-transformers\n        "perception": {\n            "mock_mode": mock_perception,\n            "mock_embedding_dim": 384,\n            "cache_size": 100,\n        },\n\n        # Affect - lightweight, no heavy deps\n        "affect": {},\n\n        # Attention\n        "attention": {},\n\n        # Action\n        "action": {},\n\n        # IWMT - enabled with defaults\n        "iwmt": {\n            "enabled": True,\n        },\n\n        # Meta-cognition\n        "meta_cognition": {\n            "action_learner": {},\n        },\n\n        # Memory - minimal config\n        "memory": {},\n\n        # Autonomous initiation\n        "autonomous_initiation": {},\n\n        # Temporal systems\n        "temporal_awareness": {},\n        "temporal_grounding": {},\n\n        # Introspection\n        "introspective_loop": {},\n\n        # Communication\n        "communication": {},\n\n        # Language models - use mock clients\n        "input_llm": {\n            "use_real_model": False,\n        },\n        "output_llm": {\n            "use_real_model": False,\n        },\n\n        # Checkpointing - enabled but with temp dir\n        "checkpointing": {\n            "enabled": True,\n            "auto_save": False,\n            "checkpoint_dir": str(checkpoint_dir),\n            "max_checkpoints": 5,\n            "compression": True,\n            "checkpoint_on_shutdown": False,\n        },\n\n        # Devices - disabled for boot\n        "devices": {\n            "enabled": False,\n        },\n\n        # Identity\n        "identity": {},\n\n        # Timing\n        "timing": {\n            "warn_threshold_ms": 200,\n            "critical_threshold_ms": 500,\n            "track_slow_cycles": True,\n        },\n\n        # Continuous consciousness\n        "continuous_consciousness": {},\n\n        # Memory review\n        "memory_review": {},\n\n        # Existential reflection\n        "existential_reflection": {},\n\n        # Pattern analysis\n        "pattern_analysis": {},\n\n        # Bottleneck detection\n        "bottleneck_detection": {},\n    }\n