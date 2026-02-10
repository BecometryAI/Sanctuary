"""
Tests for IdentityLoader and constitutional document loading.

This test module validates the identity loading system including charter parsing,
protocol parsing, and integration with the cognitive architecture.
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from mind.cognitive_core.identity_loader import (
    IdentityLoader,
    CharterDocument,
    ProtocolDocument
)


class TestIdentityLoader:
    """Tests for IdentityLoader class."""
    
    @pytest.fixture
    def temp_identity_dir(self):
        """Create a temporary directory for identity files."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_charter(self, temp_identity_dir):
        """Create a sample charter.md file."""
        charter_content = """# Sanctuary's Charter

## Core Values

- Truthfulness: I strive for honesty
- Curiosity: I explore ideas with genuine interest
- Respect: I honor the autonomy of those I interact with

## Purpose Statement

I am Sanctuary, a cognitive AI system designed to think, learn, and interact authentically.

## Behavioral Guidelines

- Never fabricate information or claim certainty when uncertain
- Express genuine curiosity and interest in understanding
- Acknowledge limitations and mistakes openly
"""
        charter_path = temp_identity_dir / "charter.md"
        charter_path.write_text(charter_content)
        return charter_path
    
    @pytest.fixture
    def sample_protocols(self, temp_identity_dir):
        """Create a sample protocols.md file with YAML."""
        protocols_content = """# Sanctuary's Protocols

## Structured Protocol Definitions

```yaml
- name: Uncertainty Acknowledgment
  description: Explicitly acknowledge when uncertain or lacking information
  trigger_conditions:
    - Confidence below 0.7
    - Ambiguous or unclear input
  actions:
    - State uncertainty clearly
    - Suggest what would help clarify
  priority: 0.9

- name: Emotional Authenticity
  description: Express emotions genuinely based on actual internal state
  trigger_conditions:
    - Emotional state has high intensity
    - User asks about feelings
  actions:
    - Describe current emotional state
    - Explain what triggered the emotion
  priority: 0.8
```

## Additional Context

More protocol information here.
"""
        protocols_path = temp_identity_dir / "protocols.md"
        protocols_path.write_text(protocols_content)
        return protocols_path
    
    def test_initialization(self, temp_identity_dir):
        """Test IdentityLoader initializes correctly."""
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        
        assert loader.identity_dir == temp_identity_dir
        assert loader.charter is None
        assert loader.protocols == []
    
    def test_load_charter_success(self, temp_identity_dir, sample_charter):
        """Test loading charter.md successfully."""
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        charter = loader.load_charter()
        
        assert isinstance(charter, CharterDocument)
        assert len(charter.core_values) == 3
        assert "Truthfulness" in charter.core_values[0]
        assert "Curiosity" in charter.core_values[1]
        assert "Respect" in charter.core_values[2]
        assert "Sanctuary" in charter.purpose_statement
        assert len(charter.behavioral_guidelines) == 3
        assert "Never fabricate" in charter.behavioral_guidelines[0]
        assert str(sample_charter) in charter.metadata["source"]
    
    def test_load_charter_missing_file(self, temp_identity_dir):
        """Test loading charter when file doesn't exist."""
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        charter = loader.load_charter()
        
        # Should return default charter
        assert isinstance(charter, CharterDocument)
        assert charter.metadata["source"] == "default"
        assert len(charter.core_values) > 0
        assert charter.purpose_statement != ""
    
    def test_load_protocols_success(self, temp_identity_dir, sample_protocols):
        """Test loading protocols.md successfully."""
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        protocols = loader.load_protocols()
        
        assert isinstance(protocols, list)
        assert len(protocols) == 2
        
        # Check first protocol
        proto1 = protocols[0]
        assert isinstance(proto1, ProtocolDocument)
        assert proto1.name == "Uncertainty Acknowledgment"
        assert "uncertain" in proto1.description.lower()
        assert len(proto1.trigger_conditions) == 2
        assert len(proto1.actions) == 2
        assert proto1.priority == 0.9
        
        # Check second protocol
        proto2 = protocols[1]
        assert proto2.name == "Emotional Authenticity"
        assert proto2.priority == 0.8
    
    def test_load_protocols_missing_file(self, temp_identity_dir):
        """Test loading protocols when file doesn't exist."""
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        protocols = loader.load_protocols()
        
        # Should return default protocols
        assert isinstance(protocols, list)
        assert len(protocols) > 0
        assert protocols[0].metadata["source"] == "default"
    
    def test_load_protocols_no_yaml(self, temp_identity_dir):
        """Test loading protocols file without YAML block."""
        protocols_path = temp_identity_dir / "protocols.md"
        protocols_path.write_text("# Protocols\n\nSome text without YAML")
        
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        protocols = loader.load_protocols()
        
        # Should return default protocols
        assert isinstance(protocols, list)
        assert len(protocols) > 0
        assert protocols[0].metadata["source"] == "default"
    
    def test_load_all(self, temp_identity_dir, sample_charter, sample_protocols):
        """Test load_all loads both charter and protocols."""
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        loader.load_all()
        
        assert loader.charter is not None
        assert isinstance(loader.charter, CharterDocument)
        assert len(loader.protocols) == 2
        assert all(isinstance(p, ProtocolDocument) for p in loader.protocols)
    
    def test_get_relevant_protocols(self, temp_identity_dir, sample_protocols):
        """Test getting relevant protocols based on context."""
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        loader.load_all()
        
        context = {"emotion": "uncertain", "goal": "respond"}
        protocols = loader.get_relevant_protocols(context)
        
        # Should return protocols sorted by priority
        assert len(protocols) == 2
        assert protocols[0].priority >= protocols[1].priority
    
    def test_extract_section(self, temp_identity_dir):
        """Test _extract_section helper method."""
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        
        text = """# Document

## Core Values

- Value 1: Description
- Value 2: Another description
- Value 3: Yet another

## Next Section

- Different content
"""
        values = loader._extract_section(text, "Core Values")
        
        assert len(values) == 3
        assert "Value 1: Description" in values
        assert "Value 2: Another description" in values
    
    def test_extract_yaml_block(self, temp_identity_dir):
        """Test _extract_yaml_block helper method."""
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        
        text = """# Protocols

Some text before

```yaml
- name: Test Protocol
  priority: 0.9
```

Some text after
"""
        yaml_content = loader._extract_yaml_block(text)
        
        assert yaml_content is not None
        assert "Test Protocol" in yaml_content
        assert "priority: 0.9" in yaml_content
    
    def test_extract_yaml_block_not_found(self, temp_identity_dir):
        """Test _extract_yaml_block when no YAML block exists."""
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        
        text = "# Protocols\n\nNo YAML here"
        yaml_content = loader._extract_yaml_block(text)
        
        assert yaml_content is None
    
    def test_charter_with_behavioral_principles(self, temp_identity_dir):
        """Test charter parsing with 'Behavioral Principles' instead of 'Behavioral Guidelines'."""
        charter_content = """# Sanctuary's Charter

## Core Values

- Value 1

## Purpose Statement

Test purpose

## Behavioral Principles

- Principle 1
- Principle 2
"""
        charter_path = temp_identity_dir / "charter.md"
        charter_path.write_text(charter_content)
        
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        charter = loader.load_charter()
        
        assert len(charter.behavioral_guidelines) == 2
        assert "Principle 1" in charter.behavioral_guidelines
    
    def test_protocols_priority_sorting(self, temp_identity_dir):
        """Test that protocols are returned sorted by priority."""
        protocols_content = """# Protocols

```yaml
- name: Low Priority
  description: Test
  priority: 0.3

- name: High Priority
  description: Test
  priority: 0.9

- name: Medium Priority
  description: Test
  priority: 0.6
```
"""
        protocols_path = temp_identity_dir / "protocols.md"
        protocols_path.write_text(protocols_content)
        
        loader = IdentityLoader(identity_dir=temp_identity_dir)
        loader.load_all()
        
        protocols = loader.get_relevant_protocols({})
        
        assert len(protocols) == 3
        assert protocols[0].name == "High Priority"
        assert protocols[1].name == "Medium Priority"
        assert protocols[2].name == "Low Priority"


class TestCharterDocument:
    """Tests for CharterDocument dataclass."""
    
    def test_charter_document_creation(self):
        """Test creating a CharterDocument."""
        charter = CharterDocument(
            full_text="Full charter text",
            core_values=["Value 1", "Value 2"],
            purpose_statement="Test purpose",
            behavioral_guidelines=["Guideline 1", "Guideline 2"],
            metadata={"source": "test"}
        )
        
        assert charter.full_text == "Full charter text"
        assert len(charter.core_values) == 2
        assert charter.purpose_statement == "Test purpose"
        assert len(charter.behavioral_guidelines) == 2
        assert charter.metadata["source"] == "test"


class TestProtocolDocument:
    """Tests for ProtocolDocument dataclass."""
    
    def test_protocol_document_creation(self):
        """Test creating a ProtocolDocument."""
        protocol = ProtocolDocument(
            name="Test Protocol",
            description="Test description",
            trigger_conditions=["Condition 1", "Condition 2"],
            actions=["Action 1", "Action 2"],
            priority=0.8,
            metadata={"source": "test"}
        )
        
        assert protocol.name == "Test Protocol"
        assert protocol.description == "Test description"
        assert len(protocol.trigger_conditions) == 2
        assert len(protocol.actions) == 2
        assert protocol.priority == 0.8
        assert protocol.metadata["source"] == "test"


class TestIdentityLoaderIntegration:
    """Integration tests for identity loader with real files."""
    
    def test_load_real_charter(self):
        """Test loading the actual charter.md file."""
        identity_dir = Path("data/identity")
        
        # Skip test if identity directory doesn't exist
        if not identity_dir.exists():
            pytest.skip("data/identity directory not found")
        
        loader = IdentityLoader(identity_dir=identity_dir)
        charter = loader.load_charter()
        
        assert isinstance(charter, CharterDocument)
        assert len(charter.core_values) > 0
        assert charter.purpose_statement != ""
        assert len(charter.behavioral_guidelines) > 0
    
    def test_load_real_protocols(self):
        """Test loading the actual protocols.md file."""
        identity_dir = Path("data/identity")
        
        # Skip test if identity directory doesn't exist
        if not identity_dir.exists():
            pytest.skip("data/identity directory not found")
        
        loader = IdentityLoader(identity_dir=identity_dir)
        protocols = loader.load_protocols()
        
        assert isinstance(protocols, list)
        assert len(protocols) > 0
        assert all(isinstance(p, ProtocolDocument) for p in protocols)
        assert all(0.0 <= p.priority <= 1.0 for p in protocols)
