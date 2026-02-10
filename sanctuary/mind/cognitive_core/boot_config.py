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
) -> Dict[str, Any]:
    """
    Create a minimal configuration for Phase 1 boot.

    Args:
        temp_dir: Base directory for data files. If None, creates a temp dir.
        cycle_rate_hz: Target cognitive cycle rate.
        mock_perception: Use mock perception (no sentence-transformers needed).

    Returns:
        Configuration dict suitable for CognitiveCore.__init__.
    """
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="sanctuary_boot_"))

    # Create required directories
    identity_dir = temp_dir / "identity"
    journal_dir = temp_dir / "introspection"
    checkpoint_dir = temp_dir / "checkpoints"

    for d in [identity_dir, journal_dir, checkpoint_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        # Core loop settings
        "cycle_rate_hz": cycle_rate_hz,
        "attention_budget": 100,
        "max_queue_size": 100,
        "log_interval_cycles": 50,

        # Filesystem paths
        "identity_dir": str(identity_dir),
        "journal_dir": str(journal_dir),

        # Perception - use mock mode to avoid sentence-transformers
        "perception": {
            "mock_mode": mock_perception,
            "mock_embedding_dim": 384,
            "cache_size": 100,
        },

        # Affect - lightweight, no heavy deps
        "affect": {},

        # Attention
        "attention": {},

        # Action
        "action": {},

        # IWMT - enabled with defaults
        "iwmt": {
            "enabled": True,
        },

        # Meta-cognition
        "meta_cognition": {
            "action_learner": {},
        },

        # Memory - minimal config
        "memory": {},

        # Autonomous initiation
        "autonomous_initiation": {},

        # Temporal systems
        "temporal_awareness": {},
        "temporal_grounding": {},

        # Introspection
        "introspective_loop": {},

        # Communication
        "communication": {},

        # Language models - use mock clients
        "input_llm": {
            "use_real_model": False,
        },
        "output_llm": {
            "use_real_model": False,
        },

        # Checkpointing - enabled but with temp dir
        "checkpointing": {
            "enabled": True,
            "auto_save": False,
            "checkpoint_dir": str(checkpoint_dir),
            "max_checkpoints": 5,
            "compression": True,
            "checkpoint_on_shutdown": False,
        },

        # Devices - disabled for boot
        "devices": {
            "enabled": False,
        },

        # Identity
        "identity": {},

        # Timing
        "timing": {
            "warn_threshold_ms": 200,
            "critical_threshold_ms": 500,
            "track_slow_cycles": True,
        },

        # Continuous consciousness
        "continuous_consciousness": {},

        # Memory review
        "memory_review": {},

        # Existential reflection
        "existential_reflection": {},

        # Pattern analysis
        "pattern_analysis": {},

        # Bottleneck detection
        "bottleneck_detection": {},
    }