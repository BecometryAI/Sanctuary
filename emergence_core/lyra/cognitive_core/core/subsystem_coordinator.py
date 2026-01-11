"""
Subsystem coordination for the cognitive core.

Handles initialization and coordination of all cognitive subsystems.
"""

from __future__ import annotations

import logging
from typing import Dict, Any
from pathlib import Path

from ..workspace import GlobalWorkspace
from ..attention import AttentionController
from ..perception import PerceptionSubsystem
from ..action import ActionSubsystem
from ..affect import AffectSubsystem
from ..meta_cognition import SelfMonitor, IntrospectiveJournal
from ..memory_integration import MemoryIntegration
from ..language_input import LanguageInputParser
from ..language_output import LanguageOutputGenerator
from ..autonomous_initiation import AutonomousInitiationController
from ..temporal_awareness import TemporalAwareness
from ..autonomous_memory_review import AutonomousMemoryReview
from ..existential_reflection import ExistentialReflection
from ..interaction_patterns import InteractionPatternAnalysis
from ..continuous_consciousness import ContinuousConsciousnessController
from ..introspective_loop import IntrospectiveLoop
from ..identity_loader import IdentityLoader
from ..checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


class SubsystemCoordinator:
    """
    Coordinates initialization and interactions between all cognitive subsystems.
    
    Responsibilities:
    - Initialize all subsystems in correct order
    - Manage dependencies between subsystems
    - Provide access to subsystems
    """
    
    def __init__(self, workspace: GlobalWorkspace, config: Dict[str, Any]):
        """
        Initialize all cognitive subsystems.
        
        Args:
            workspace: GlobalWorkspace instance
            config: Configuration dict
        """
        self.config = config
        self.workspace = workspace
        
        # Initialize identity loader (loads charter and protocols)
        identity_dir = Path(config.get("identity_dir", "data/identity"))
        self.identity = IdentityLoader(identity_dir=identity_dir)
        self.identity.load_all()
        
        # Initialize computed identity manager
        from ..identity import IdentityManager
        identity_config_path = config.get("identity_config_path")
        self.identity_manager = IdentityManager(
            config_path=identity_config_path,
            config=config.get("identity", {})
        )
        
        # Initialize affect subsystem first (needed by attention and action)
        self.affect = AffectSubsystem(config=config.get("affect", {}))
        
        # Initialize attention controller
        self.attention = AttentionController(
            attention_budget=config["attention_budget"],
            workspace=workspace,
            affect=self.affect
        )
        
        # Initialize perception subsystem
        self.perception = PerceptionSubsystem(config=config.get("perception", {}))
        
        # Initialize action subsystem with behavior logger
        self.action = ActionSubsystem(
            config=config.get("action", {}),
            affect=self.affect,
            identity=self.identity,
            behavior_logger=self.identity_manager.behavior_log
        )
        
        # Store references for subsystems to access each other
        workspace.affect = self.affect
        workspace.action_subsystem = self.action
        workspace.perception = self.perception
        
        # Initialize meta-cognition (needs workspace reference and identity)
        self.meta_cognition = SelfMonitor(
            workspace=workspace,
            config=config.get("meta_cognition", {}),
            identity=self.identity,
            identity_manager=self.identity_manager
        )
        
        # Create introspective journal
        journal_dir = Path(config.get("journal_dir", "data/introspection"))
        journal_dir.mkdir(parents=True, exist_ok=True)
        self.introspective_journal = IntrospectiveJournal(journal_dir)
        
        # Initialize memory integration
        self.memory = MemoryIntegration(
            workspace=workspace,
            config=config.get("memory", {})
        )
        
        # Initialize autonomous initiation controller
        self.autonomous = AutonomousInitiationController(
            workspace=workspace,
            config=config.get("autonomous_initiation", {})
        )
        
        # Initialize continuous consciousness components
        # Keep legacy TemporalAwareness for backward compatibility
        self.temporal_awareness = TemporalAwareness(
            config=config.get("temporal_awareness", {})
        )
        
        # Initialize new TemporalGrounding system
        from ..temporal import TemporalGrounding
        self.temporal_grounding = TemporalGrounding(
            config=config.get("temporal_grounding", {}),
            memory=self.memory
        )
        
        self.memory_review = AutonomousMemoryReview(
            self.memory,
            config=config.get("memory_review", {})
        )
        
        self.existential_reflection = ExistentialReflection(
            config=config.get("existential_reflection", {})
        )
        
        self.pattern_analysis = InteractionPatternAnalysis(
            self.memory,
            config=config.get("pattern_analysis", {})
        )
        
        # Initialize introspective loop
        self.introspective_loop = IntrospectiveLoop(
            workspace=workspace,
            self_monitor=self.meta_cognition,
            journal=self.introspective_journal,
            config=config.get("introspective_loop", {})
        )
        
        # Initialize LLM clients for language interfaces
        self._initialize_llm_clients()
        
        # Initialize language input parser
        self.language_input = LanguageInputParser(
            self.perception,
            llm_client=self.llm_input_client,
            config=config.get("language_input", {})
        )
        
        # Initialize language output generator
        self.language_output = LanguageOutputGenerator(
            self.llm_output_client,
            config=config.get("language_output", {}),
            identity=self.identity
        )
        
        # Initialize checkpoint manager
        checkpoint_config = config.get("checkpointing", {})
        if checkpoint_config.get("enabled", True):
            checkpoint_dir = Path(checkpoint_config.get("checkpoint_dir", "data/checkpoints"))
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                max_checkpoints=checkpoint_config.get("max_checkpoints", 20),
                compression=checkpoint_config.get("compression", True),
            )
            logger.info(f"💾 Checkpoint manager enabled: {checkpoint_dir}")
        else:
            self.checkpoint_manager = None
            logger.info("💾 Checkpoint manager disabled")
        
        logger.info(
            f"🧠 Subsystems initialized: cycle_rate={config['cycle_rate_hz']}Hz, "
            f"attention_budget={config['attention_budget']}"
        )
    
    def _initialize_llm_clients(self) -> None:
        """
        Initialize LLM clients for language interfaces.
        
        Note: LLM clients are imported here (not at module level) because they
        may not be available in all environments. This allows the module to be
        imported without requiring the LLM dependencies, and only fails at
        runtime if LLM functionality is actually used.
        """
        from ..llm_client import MockLLMClient, GemmaClient, LlamaClient
        
        input_llm_config = self.config.get("input_llm", {})
        output_llm_config = self.config.get("output_llm", {})
        
        # Determine which LLM clients to use for input
        if input_llm_config.get("use_real_model", False):
            try:
                self.llm_input_client = GemmaClient(input_llm_config)
                logger.info("✅ Using real Gemma client for input parsing")
            except Exception as e:
                logger.warning(f"Failed to load Gemma client: {e}, using mock")
                self.llm_input_client = MockLLMClient(input_llm_config)
        else:
            self.llm_input_client = MockLLMClient(input_llm_config)
            logger.info("✅ Using mock LLM client for input parsing (development mode)")
        
        # Determine which LLM clients to use for output
        if output_llm_config.get("use_real_model", False):
            try:
                self.llm_output_client = LlamaClient(output_llm_config)
                logger.info("✅ Using real Llama client for output generation")
            except Exception as e:
                logger.warning(f"Failed to load Llama client: {e}, using mock")
                self.llm_output_client = MockLLMClient(output_llm_config)
        else:
            self.llm_output_client = MockLLMClient(output_llm_config)
            logger.info("✅ Using mock LLM client for output generation (development mode)")
    
    def initialize_continuous_consciousness(self, cognitive_core) -> ContinuousConsciousnessController:
        """
        Initialize continuous consciousness controller.
        
        This is done separately because it needs a reference to the cognitive core.
        
        Args:
            cognitive_core: Reference to the CognitiveCore instance
            
        Returns:
            ContinuousConsciousnessController instance
        """
        return ContinuousConsciousnessController(
            cognitive_core,
            config=self.config.get("continuous_consciousness", {})
        )
