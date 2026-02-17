"""
Subsystem coordination for the cognitive core.

Handles initialization and coordination of all cognitive subsystems.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
from pathlib import Path

from ..workspace import GlobalWorkspace
from ..attention import AttentionController
from ..perception import PerceptionSubsystem
from ..action import ActionSubsystem
from ..affect import AffectSubsystem
from ..meta_cognition import SelfMonitor, IntrospectiveJournal, BottleneckDetector
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


def _try_import_devices():
    """Try to import device modules (optional dependency)."""
    try:
        from ...devices import (
            DeviceRegistry,
            DeviceType,
            MicrophoneDevice,
            CameraDevice,
            SerialSensorDevice,
            HAS_SOUNDDEVICE,
            HAS_OPENCV,
            HAS_SERIAL,
        )
        return {
            "DeviceRegistry": DeviceRegistry,
            "DeviceType": DeviceType,
            "MicrophoneDevice": MicrophoneDevice,
            "CameraDevice": CameraDevice,
            "SerialSensorDevice": SerialSensorDevice,
            "HAS_SOUNDDEVICE": HAS_SOUNDDEVICE,
            "HAS_OPENCV": HAS_OPENCV,
            "HAS_SERIAL": HAS_SERIAL,
        }
    except ImportError as e:
        logger.debug(f"Device modules not available: {e}")
        return None


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

        # Initialize action outcome learner for IWMT (lightweight, only when IWMT enabled)
        iwmt_config = config.get("iwmt", {"enabled": True})
        if iwmt_config.get("enabled", True):
            from ..iwmt_core import IWMTCore
            from ..meta_cognition import ActionOutcomeLearner
            
            # Create action learner for tracking action reliability
            action_learner_config = config.get("meta_cognition", {}).get("action_learner", {})
            action_learner = ActionOutcomeLearner(config=action_learner_config)
            
            self.iwmt_core = IWMTCore(
                iwmt_config,
                action_learner=action_learner
            )
            logger.info("🧠 IWMT Core initialized with action learning integration")
        else:
            self.iwmt_core = None
            logger.info("🧠 IWMT Core disabled (legacy GWT mode)")
        
        # Initialize attention controller with IWMT precision weighting and emotional attention
        self.attention = AttentionController(
            attention_budget=config["attention_budget"],
            workspace=workspace,
            affect=self.affect,
            precision_weighting=self.iwmt_core.precision if self.iwmt_core else None,
            emotional_attention=self.affect.emotional_attention_system
        )
        
        # Initialize perception subsystem (mock mode avoids heavy torch/sentence-transformers)
        perception_config = config.get("perception", {})
        if perception_config.get("mock_mode", False):
            from ..mock_perception import MockPerceptionSubsystem
            self.perception = MockPerceptionSubsystem(config=perception_config)
        else:
            self.perception = PerceptionSubsystem(config=perception_config)
        
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

        # Initialize bottleneck detector for cognitive load monitoring
        self.bottleneck_detector = BottleneckDetector(
            config=config.get("bottleneck_detection", {})
        )
        logger.debug("🔍 Bottleneck detector initialized")

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
        
        # Initialize cross-memory association detector
        from ..memory_associations import MemoryAssociationDetector
        self.memory_associations = MemoryAssociationDetector(
            config=config.get("memory_associations", {})
        )
        logger.debug("🔗 Memory association detector initialized")

        # Initialize goal dynamics for priority adjustment
        from ..goals import GoalDynamics
        self.goal_dynamics = GoalDynamics(
            config=config.get("goal_dynamics", {})
        )
        logger.debug("🎯 Goal dynamics initialized")

        # Initialize communication drive system with validated config
        from ..communication import CommunicationDriveSystem
        self.communication_drives = CommunicationDriveSystem(
            config=config.get("communication", {})
        )
        logger.debug("💬 Communication drive system initialized")

        # Initialize communication inhibition system
        from ..communication import CommunicationInhibitionSystem
        self.communication_inhibitions = CommunicationInhibitionSystem(
            config=config.get("communication", {})
        )
        logger.debug("💬 Communication inhibition system initialized")

        # Initialize communication decision loop with active inference
        from ..communication import CommunicationDecisionLoop
        self.communication_decision = CommunicationDecisionLoop(
            drive_system=self.communication_drives,
            inhibition_system=self.communication_inhibitions,
            config=config.get("communication", {}),
            free_energy_minimizer=self.iwmt_core.free_energy if self.iwmt_core else None,
            world_model=self.iwmt_core.world_model if self.iwmt_core else None
        )
        if self.iwmt_core:
            logger.info("💬 Communication decision loop initialized with active inference")
        else:
            logger.debug("💬 Communication decision loop initialized (legacy mode)")

        # Initialize interruption system for urgent mid-turn communication
        from ..communication import InterruptionSystem
        self.interruption = InterruptionSystem(
            config=config.get("communication", {}).get("interruption", {})
        )
        logger.debug("💬 Interruption system initialized")

        # Initialize communication reflection system (post-hoc evaluation)
        from ..communication import CommunicationReflectionSystem
        self.communication_reflection = CommunicationReflectionSystem(
            config=config.get("communication", {}).get("reflection", {})
        )
        logger.debug("💬 Communication reflection system initialized")

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

        # Initialize device registry for multimodal perception
        self.device_registry = self._initialize_device_registry(config.get("devices", {}))

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

        Supported backends (set via "backend" key in config):
            - "ollama": Local Ollama server (recommended for local use)
            - "gemma" / "llama": HuggingFace transformers (requires GPU + torch)
            - None / missing: Mock client (development mode)
        """
        from ..llm_client import MockLLMClient, GemmaClient, LlamaClient, OllamaClient

        input_llm_config = self.config.get("input_llm", {})
        output_llm_config = self.config.get("output_llm", {})

        self.llm_input_client = self._create_llm_client(
            input_llm_config, "input", MockLLMClient, GemmaClient, LlamaClient, OllamaClient
        )
        self.llm_output_client = self._create_llm_client(
            output_llm_config, "output", MockLLMClient, GemmaClient, LlamaClient, OllamaClient
        )

    def _create_llm_client(self, config, role, MockLLMClient, GemmaClient, LlamaClient, OllamaClient):
        """Create an LLM client based on config. Falls back to mock on failure."""
        if not config.get("use_real_model", False):
            logger.info(f"✅ Using mock LLM client for {role} (development mode)")
            return MockLLMClient(config)

        backend = config.get("backend", "").lower()

        if backend == "ollama":
            try:
                client = OllamaClient(config)
                logger.info(f"✅ Using Ollama client for {role}: model={config.get('model_name', 'default')}")
                return client
            except Exception as e:
                logger.warning(f"Failed to load Ollama client for {role}: {e}, using mock")
                return MockLLMClient(config)

        # Legacy HuggingFace backends
        if role == "input" or backend == "gemma":
            try:
                client = GemmaClient(config)
                logger.info(f"✅ Using real Gemma client for {role}")
                return client
            except Exception as e:
                logger.warning(f"Failed to load Gemma client for {role}: {e}, using mock")
                return MockLLMClient(config)
        else:
            try:
                client = LlamaClient(config)
                logger.info(f"✅ Using real Llama client for {role}")
                return client
            except Exception as e:
                logger.warning(f"Failed to load Llama client for {role}: {e}, using mock")
                return MockLLMClient(config)
    
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

    def _initialize_device_registry(self, device_config: Dict[str, Any]) -> Optional[Any]:
        """
        Initialize device registry for multimodal perception.

        This is done separately because devices are optional dependencies.
        The registry enables plug-and-play support for cameras, microphones,
        and other peripherals.

        Args:
            device_config: Device configuration dict with keys:
                - enabled: bool (default: True)
                - hot_plug_interval: float (default: 5.0)
                - microphone: dict with enabled, sample_rate, etc.
                - camera: dict with enabled, resolution, fps, etc.
                - sensors: dict with enabled, serial_port, etc.

        Returns:
            DeviceRegistry instance or None if devices disabled/unavailable
        """
        if not device_config.get("enabled", True):
            logger.info("📷 Device registry disabled by configuration")
            return None

        # Try to import device modules
        device_modules = _try_import_devices()
        if device_modules is None:
            logger.info("📷 Device registry not available (optional dependencies not installed)")
            return None

        DeviceRegistry = device_modules["DeviceRegistry"]
        DeviceType = device_modules["DeviceType"]
        MicrophoneDevice = device_modules["MicrophoneDevice"]
        CameraDevice = device_modules["CameraDevice"]
        SerialSensorDevice = device_modules["SerialSensorDevice"]

        try:
            # Create registry
            registry = DeviceRegistry(config=device_config)

            # Register available device types
            if device_modules["HAS_SOUNDDEVICE"]:
                registry.register_device_class(DeviceType.MICROPHONE, MicrophoneDevice)
                logger.debug("📷 Registered MicrophoneDevice")

            if device_modules["HAS_OPENCV"]:
                registry.register_device_class(DeviceType.CAMERA, CameraDevice)
                logger.debug("📷 Registered CameraDevice")

            if device_modules["HAS_SERIAL"]:
                registry.register_device_class(DeviceType.SENSOR, SerialSensorDevice)
                logger.debug("📷 Registered SerialSensorDevice")

            logger.info(
                f"📷 Device registry initialized with {len(registry.get_registered_types())} device types"
            )
            return registry

        except Exception as e:
            logger.warning(f"📷 Failed to initialize device registry: {e}")
            return None

    # ========== Subsystem reinitialization for recovery ==========

    def reinitialize_perception(self) -> None:
        """Reinitialize the perception subsystem (stateless — safe to recreate)."""
        self.perception = PerceptionSubsystem(config=self.config.get("perception", {}))
        self.workspace.perception = self.perception
        # Re-wire language input parser's reference
        if hasattr(self, 'language_input') and self.language_input is not None:
            self.language_input.perception = self.perception
        logger.info("Reinitialized perception subsystem")

    def reinitialize_attention(self) -> None:
        """Reinitialize the attention controller (stateless — safe to recreate)."""
        self.attention = AttentionController(
            attention_budget=self.config["attention_budget"],
            workspace=self.workspace,
            affect=self.affect,
            precision_weighting=self.iwmt_core.precision if self.iwmt_core else None,
            emotional_attention=self.affect.emotional_attention_system,
        )
        logger.info("Reinitialized attention subsystem")

    def reinitialize_affect(self) -> None:
        """Reinitialize affect subsystem (resets emotional state to baseline)."""
        self.affect = AffectSubsystem(config=self.config.get("affect", {}))
        self.workspace.affect = self.affect
        # Downstream subsystems hold references that need updating
        self.action.affect = self.affect
        self.reinitialize_attention()  # attention depends on affect
        logger.info("Reinitialized affect subsystem (emotional state reset to baseline)")

    def reinitialize_action(self) -> None:
        """Reinitialize the action subsystem."""
        self.action = ActionSubsystem(
            config=self.config.get("action", {}),
            affect=self.affect,
            identity=self.identity,
            behavior_logger=self.identity_manager.behavior_log,
        )
        self.workspace.action_subsystem = self.action
        logger.info("Reinitialized action subsystem")

    def reinitialize_meta_cognition(self) -> None:
        """Reinitialize meta-cognition (preserves journal, resets monitor)."""
        self.meta_cognition = SelfMonitor(
            workspace=self.workspace,
            config=self.config.get("meta_cognition", {}),
            identity=self.identity,
            identity_manager=self.identity_manager,
        )
        # Re-wire introspective loop reference
        if hasattr(self, 'introspective_loop') and self.introspective_loop is not None:
            self.introspective_loop.self_monitor = self.meta_cognition
        logger.info("Reinitialized meta-cognition subsystem")

    def reinitialize_communication_drives(self) -> None:
        """Reinitialize communication drive system."""
        from ..communication import CommunicationDriveSystem
        self.communication_drives = CommunicationDriveSystem(
            config=self.config.get("communication", {}),
        )
        # Re-wire decision loop reference
        if hasattr(self, 'communication_decision'):
            self.communication_decision.drive_system = self.communication_drives
        logger.info("Reinitialized communication drives")

    def reinitialize_autonomous_initiation(self) -> None:
        """Reinitialize autonomous initiation controller."""
        self.autonomous = AutonomousInitiationController(
            workspace=self.workspace,
            config=self.config.get("autonomous_initiation", {}),
        )
        logger.info("Reinitialized autonomous initiation")

    def reinitialize_bottleneck_detector(self) -> None:
        """Reinitialize bottleneck detector (stateless — safe to recreate)."""
        self.bottleneck_detector = BottleneckDetector(
            config=self.config.get("bottleneck_detection", {}),
        )
        logger.info("Reinitialized bottleneck detector")

    def reinitialize_temporal_grounding(self) -> None:
        """Reinitialize temporal grounding (resets temporal context)."""
        from ..temporal import TemporalGrounding
        self.temporal_grounding = TemporalGrounding(
            config=self.config.get("temporal_grounding", {}),
            memory=self.memory,
        )
        logger.info("Reinitialized temporal grounding")

    def reinitialize_memory(self) -> None:
        """Reinitialize memory integration (preserves stored memories)."""
        self.memory = MemoryIntegration(
            workspace=self.workspace,
            config=self.config.get("memory", {}),
        )
        # Re-wire downstream references
        if hasattr(self, 'temporal_grounding'):
            self.temporal_grounding.memory = self.memory
        if hasattr(self, 'memory_review'):
            self.memory_review.memory = self.memory
        if hasattr(self, 'pattern_analysis'):
            self.pattern_analysis.memory = self.memory
        logger.info("Reinitialized memory integration")

    def reinitialize_iwmt(self) -> None:
        """Reinitialize IWMT core (resets world model and predictions)."""
        iwmt_config = self.config.get("iwmt", {"enabled": True})
        if iwmt_config.get("enabled", True):
            from ..iwmt_core import IWMTCore
            from ..meta_cognition import ActionOutcomeLearner

            action_learner_config = self.config.get("meta_cognition", {}).get("action_learner", {})
            action_learner = ActionOutcomeLearner(config=action_learner_config)
            self.iwmt_core = IWMTCore(iwmt_config, action_learner=action_learner)
            # Re-wire attention precision weighting
            self.attention.precision_weighting = self.iwmt_core.precision
            # Re-wire communication decision loop
            if hasattr(self, 'communication_decision'):
                self.communication_decision.free_energy_minimizer = self.iwmt_core.free_energy
                self.communication_decision.world_model = self.iwmt_core.world_model
            logger.info("Reinitialized IWMT core")

    def connect_device_registry_to_input(self, input_queue) -> bool:
        """
        Connect device registry to the input queue for data routing.

        This must be called after StateManager initializes its queues.
        Device data will be routed to the cognitive pipeline via the input queue.

        Args:
            input_queue: asyncio.Queue or InputQueue instance

        Returns:
            True if connection successful
        """
        if self.device_registry is None:
            return False

        try:
            # Create an adapter that works with both raw asyncio.Queue and InputQueue
            import asyncio

            async def route_device_data(data, modality, source, metadata):
                """Route device data to input queue."""
                try:
                    input_queue.put_nowait((data, modality))
                except asyncio.QueueFull:
                    logger.warning(f"Input queue full, dropping device data from {source}")

            self.device_registry.set_input_callback(route_device_data)
            logger.info("📷 Device registry connected to input queue")
            return True

        except Exception as e:
            logger.error(f"📷 Failed to connect device registry: {e}")
            return False
