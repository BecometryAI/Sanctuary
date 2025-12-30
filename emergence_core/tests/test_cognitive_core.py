"""
Unit tests for cognitive_core placeholder classes.

Tests cover:
- Proper initialization of all classes
- Import structure and module organization
- Type hints and docstring presence
- Data model validation
- PEP 8 compliance
"""

import pytest
from pathlib import Path

from lyra.cognitive_core.core import CognitiveCore
from lyra.cognitive_core.workspace import GlobalWorkspace, WorkspaceContent
from lyra.cognitive_core.attention import AttentionController, AttentionMode, AttentionScore
from lyra.cognitive_core.perception import PerceptionSubsystem, ModalityType, Percept
from lyra.cognitive_core.action import ActionSubsystem, ActionType, Action
from lyra.cognitive_core.affect import AffectSubsystem, EmotionalState
from lyra.cognitive_core.meta_cognition import SelfMonitor, MonitoringLevel, IntrospectiveReport


class TestCognitiveCore:
    """Test CognitiveCore class initialization and structure"""
    
    def test_cognitive_core_initialization_default(self):
        """Test creating CognitiveCore with default parameters"""
        core = CognitiveCore()
        assert core is not None
        assert isinstance(core, CognitiveCore)
    
    def test_cognitive_core_initialization_custom(self):
        """Test creating CognitiveCore with custom parameters"""
        core = CognitiveCore(cycle_frequency=20.0, persistence_dir="/tmp/test")
        assert core is not None
        assert isinstance(core, CognitiveCore)
    
    def test_cognitive_core_has_docstring(self):
        """Test that CognitiveCore has comprehensive docstring"""
        assert CognitiveCore.__doc__ is not None
        assert len(CognitiveCore.__doc__) > 100
        assert "recurrent" in CognitiveCore.__doc__.lower()
    
    def test_cognitive_core_init_signature(self):
        """Test that __init__ has proper type hints"""
        import inspect
        sig = inspect.signature(CognitiveCore.__init__)
        assert 'cycle_frequency' in sig.parameters
        assert 'persistence_dir' in sig.parameters


class TestGlobalWorkspace:
    """Test GlobalWorkspace class initialization and data models"""
    
    def test_workspace_initialization_default(self):
        """Test creating GlobalWorkspace with default parameters"""
        workspace = GlobalWorkspace()
        assert workspace is not None
        assert isinstance(workspace, GlobalWorkspace)
    
    def test_workspace_initialization_custom(self):
        """Test creating GlobalWorkspace with custom parameters"""
        workspace = GlobalWorkspace(capacity=10, persistence_dir="/tmp/test")
        assert workspace is not None
    
    def test_workspace_has_docstring(self):
        """Test that GlobalWorkspace has comprehensive docstring"""
        assert GlobalWorkspace.__doc__ is not None
        assert len(GlobalWorkspace.__doc__) > 100
        assert "conscious" in GlobalWorkspace.__doc__.lower()
        assert "broadcast" in GlobalWorkspace.__doc__.lower()
    
    def test_workspace_content_creation(self):
        """Test creating WorkspaceContent data class"""
        content = WorkspaceContent(
            goals=["test goal"],
            percepts=[{"type": "text", "content": "test"}],
            emotions={"valence": 0.5}
        )
        assert content is not None
        assert len(content.goals) == 1
        assert len(content.percepts) == 1
        assert "valence" in content.emotions
    
    def test_workspace_content_defaults(self):
        """Test WorkspaceContent default values"""
        content = WorkspaceContent()
        assert isinstance(content.goals, list)
        assert isinstance(content.percepts, list)
        assert isinstance(content.emotions, dict)
        assert isinstance(content.memories, list)
        assert isinstance(content.metadata, dict)


class TestAttentionController:
    """Test AttentionController class initialization and enums"""
    
    def test_attention_initialization_default(self):
        """Test creating AttentionController with default parameters"""
        attention = AttentionController()
        assert attention is not None
        assert isinstance(attention, AttentionController)
    
    def test_attention_initialization_custom(self):
        """Test creating AttentionController with custom parameters"""
        attention = AttentionController(
            initial_mode=AttentionMode.DIFFUSE,
            goal_weight=0.5,
            novelty_weight=0.3
        )
        assert attention is not None
    
    def test_attention_mode_enum(self):
        """Test AttentionMode enum values"""
        assert AttentionMode.FOCUSED.value == "focused"
        assert AttentionMode.DIFFUSE.value == "diffuse"
        assert AttentionMode.VIGILANT.value == "vigilant"
        assert AttentionMode.RELAXED.value == "relaxed"
    
    def test_attention_score_creation(self):
        """Test creating AttentionScore data class"""
        score = AttentionScore(
            goal_relevance=0.8,
            novelty=0.6,
            emotional_salience=0.7,
            urgency=0.5,
            total=0.65
        )
        assert score is not None
        assert score.goal_relevance == 0.8
        assert score.total == 0.65
    
    def test_attention_has_docstring(self):
        """Test that AttentionController has comprehensive docstring"""
        assert AttentionController.__doc__ is not None
        assert len(AttentionController.__doc__) > 100
        assert "attention" in AttentionController.__doc__.lower()


class TestPerceptionSubsystem:
    """Test PerceptionSubsystem class initialization and data models"""
    
    def test_perception_initialization_default(self):
        """Test creating PerceptionSubsystem with default parameters"""
        perception = PerceptionSubsystem()
        assert perception is not None
        assert isinstance(perception, PerceptionSubsystem)
    
    def test_perception_initialization_custom(self):
        """Test creating PerceptionSubsystem with custom parameters"""
        perception = PerceptionSubsystem(
            text_encoder_name="test-encoder",
            embedding_dim=512,
            buffer_size=50
        )
        assert perception is not None
    
    def test_modality_type_enum(self):
        """Test ModalityType enum values"""
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.PROPRIOCEPTIVE.value == "proprioceptive"
    
    def test_percept_creation(self):
        """Test creating Percept data class"""
        import numpy as np
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        percept = Percept(
            embedding=embedding,
            modality=ModalityType.TEXT,
            confidence=0.95
        )
        assert percept is not None
        assert percept.modality == ModalityType.TEXT
        assert percept.confidence == 0.95
        assert percept.metadata is not None
    
    def test_perception_has_docstring(self):
        """Test that PerceptionSubsystem has comprehensive docstring"""
        assert PerceptionSubsystem.__doc__ is not None
        assert len(PerceptionSubsystem.__doc__) > 100
        assert "perception" in PerceptionSubsystem.__doc__.lower()


class TestActionSubsystem:
    """Test ActionSubsystem class initialization and data models"""
    
    def test_action_initialization_default(self):
        """Test creating ActionSubsystem with default parameters"""
        action_sys = ActionSubsystem()
        assert action_sys is not None
        assert isinstance(action_sys, ActionSubsystem)
    
    def test_action_initialization_custom(self):
        """Test creating ActionSubsystem with custom parameters"""
        action_sys = ActionSubsystem(
            history_size=50,
            default_action_type=ActionType.WAIT
        )
        assert action_sys is not None
    
    def test_action_type_enum(self):
        """Test ActionType enum values"""
        assert ActionType.COMMUNICATE.value == "communicate"
        assert ActionType.RETRIEVE.value == "retrieve"
        assert ActionType.TOOL_USE.value == "tool_use"
        assert ActionType.INTERNAL.value == "internal"
        assert ActionType.WAIT.value == "wait"
    
    def test_action_creation(self):
        """Test creating Action data class"""
        action = Action(
            action_type=ActionType.COMMUNICATE,
            parameters={"text": "Hello"},
            priority=0.8
        )
        assert action is not None
        assert action.action_type == ActionType.COMMUNICATE
        assert action.priority == 0.8
        assert action.metadata is not None
    
    def test_action_has_docstring(self):
        """Test that ActionSubsystem has comprehensive docstring"""
        assert ActionSubsystem.__doc__ is not None
        assert len(ActionSubsystem.__doc__) > 100
        assert "action" in ActionSubsystem.__doc__.lower()


class TestAffectSubsystem:
    """Test AffectSubsystem class initialization and data models"""
    
    def test_affect_initialization_default(self):
        """Test creating AffectSubsystem with default parameters"""
        affect = AffectSubsystem()
        assert affect is not None
        assert isinstance(affect, AffectSubsystem)
    
    def test_affect_initialization_custom(self):
        """Test creating AffectSubsystem with custom parameters"""
        affect = AffectSubsystem(
            baseline_valence=0.1,
            baseline_arousal=-0.2,
            decay_rate=0.15
        )
        assert affect is not None
    
    def test_emotional_state_creation(self):
        """Test creating EmotionalState data class"""
        state = EmotionalState(
            valence=0.5,
            arousal=0.3,
            dominance=0.2
        )
        assert state is not None
        assert state.valence == 0.5
        assert state.arousal == 0.3
        assert state.dominance == 0.2
        assert state.intensity > 0.0  # Calculated in __post_init__
        assert state.labels is not None
    
    def test_emotional_state_to_vector(self):
        """Test converting EmotionalState to numpy vector"""
        import numpy as np
        state = EmotionalState(valence=0.5, arousal=0.3, dominance=0.2)
        vector = state.to_vector()
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3
        assert vector[0] == 0.5
    
    def test_affect_has_docstring(self):
        """Test that AffectSubsystem has comprehensive docstring"""
        assert AffectSubsystem.__doc__ is not None
        assert len(AffectSubsystem.__doc__) > 100
        assert "emotion" in AffectSubsystem.__doc__.lower()


class TestSelfMonitor:
    """Test SelfMonitor class initialization and data models"""
    
    def test_monitor_initialization_default(self):
        """Test creating SelfMonitor with default parameters"""
        monitor = SelfMonitor()
        assert monitor is not None
        assert isinstance(monitor, SelfMonitor)
    
    def test_monitor_initialization_custom(self):
        """Test creating SelfMonitor with custom parameters"""
        monitor = SelfMonitor(
            monitoring_level=MonitoringLevel.DETAILED,
            history_size=500,
            anomaly_detection=False
        )
        assert monitor is not None
    
    def test_monitoring_level_enum(self):
        """Test MonitoringLevel enum values"""
        assert MonitoringLevel.MINIMAL.value == "minimal"
        assert MonitoringLevel.NORMAL.value == "normal"
        assert MonitoringLevel.DETAILED.value == "detailed"
        assert MonitoringLevel.INTROSPECTIVE.value == "introspective"
    
    def test_introspective_report_creation(self):
        """Test creating IntrospectiveReport data class"""
        from datetime import datetime
        report = IntrospectiveReport(
            timestamp=datetime.now(),
            monitoring_level=MonitoringLevel.NORMAL,
            subsystem_states={"workspace": "active"},
            processing_metrics={"speed": 0.5},
            anomalies=[],
            cognitive_load=0.6,
            coherence_score=0.8,
            insights=["System operating normally"]
        )
        assert report is not None
        assert report.cognitive_load == 0.6
        assert report.coherence_score == 0.8
        assert report.metadata is not None
    
    def test_monitor_has_docstring(self):
        """Test that SelfMonitor has comprehensive docstring"""
        assert SelfMonitor.__doc__ is not None
        assert len(SelfMonitor.__doc__) > 100
        assert "meta" in SelfMonitor.__doc__.lower() or "introspect" in SelfMonitor.__doc__.lower()


class TestModuleStructure:
    """Test module-level structure and imports"""
    
    def test_cognitive_core_module_docstring(self):
        """Test cognitive_core module has proper docstring"""
        import lyra.cognitive_core as cc
        assert cc.__doc__ is not None
        assert "Cognitive Core" in cc.__doc__
        assert "non-linguistic" in cc.__doc__
    
    def test_cognitive_core_exports(self):
        """Test cognitive_core __all__ exports"""
        import lyra.cognitive_core as cc
        assert hasattr(cc, '__all__')
        assert 'CognitiveCore' in cc.__all__
        assert 'GlobalWorkspace' in cc.__all__
        assert 'AttentionController' in cc.__all__
        assert 'PerceptionSubsystem' in cc.__all__
        assert 'ActionSubsystem' in cc.__all__
        assert 'AffectSubsystem' in cc.__all__
        assert 'SelfMonitor' in cc.__all__
    
    def test_interfaces_module_docstring(self):
        """Test interfaces module has proper docstring"""
        import lyra.interfaces as li
        assert li.__doc__ is not None
        assert "Language Interfaces" in li.__doc__
        assert "peripheral" in li.__doc__.lower()
    
    def test_interfaces_exports(self):
        """Test interfaces __all__ exports"""
        import lyra.interfaces as li
        assert hasattr(li, '__all__')
        assert 'LanguageInputParser' in li.__all__
        assert 'LanguageOutputGenerator' in li.__all__
