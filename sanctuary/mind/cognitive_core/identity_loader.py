"""
Identity Loader: Constitutional documents and charter integration.

This module implements the IdentityLoader class, which loads and manages Sanctuary's
constitutional documents (charter.md and protocols.md) and makes them actively
influence cognitive processing throughout the system.

The identity loader is responsible for:
- Loading charter.md and parsing into structured format
- Loading protocols.md and parsing YAML protocol definitions
- Providing access to core values, purpose statement, and behavioral guidelines
- Enabling subsystems to query relevant protocols based on context
- Supporting Constitutional AI at the architectural level
"""

from __future__ import annotations

import re
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CharterDocument:
    """
    Represents loaded charter content.
    
    The charter contains Sanctuary's core values, purpose statement, and behavioral
    guidelines that form the constitutional foundation for decision-making.
    
    Attributes:
        full_text: Complete text of the charter
        core_values: List of core value statements
        purpose_statement: Statement of purpose and identity
        behavioral_guidelines: List of behavioral guidelines
        metadata: Additional metadata about the charter
    """
    full_text: str
    core_values: List[str]
    purpose_statement: str
    behavioral_guidelines: List[str]
    metadata: Dict[str, Any]


@dataclass
class ProtocolDocument:
    """
    Represents loaded protocol content.
    
    Protocols are operational rules that guide behavior in specific contexts,
    complementing the high-level charter with concrete guidelines.
    
    Attributes:
        name: Protocol name
        description: What the protocol does
        trigger_conditions: When this protocol applies
        actions: What actions to take
        priority: Importance of this protocol (0.0-1.0)
        metadata: Additional metadata about the protocol
    """
    name: str
    description: str
    trigger_conditions: List[str]
    actions: List[str]
    priority: float
    metadata: Dict[str, Any]


class IdentityLoader:
    """
    Loads and manages Sanctuary's identity documents.
    
    The IdentityLoader reads charter.md and protocols.md from the identity
    directory and parses them into structured formats that can be referenced
    by cognitive subsystems. This enables Constitutional AI by making values
    and principles actively influence perception, attention, action, and affect.
    
    Key Responsibilities:
    - Load and parse charter.md into structured CharterDocument
    - Load and parse protocols.md into list of ProtocolDocument objects
    - Provide access to constitutional constraints for subsystems
    - Support context-based protocol retrieval
    - Handle missing or malformed identity files gracefully
    
    Integration Points:
    - CognitiveCore: Initializes identity loader and passes to subsystems
    - ActionSubsystem: Uses charter/protocols for constitutional action filtering
    - SelfMonitor: Checks value alignment against charter values
    - LanguageOutputGenerator: Incorporates identity into prompt context
    
    Attributes:
        identity_dir: Path to identity documents directory
        charter: Loaded charter document
        protocols: List of loaded protocol documents
    """
    
    def __init__(self, identity_dir: Path):
        """
        Initialize identity loader.
        
        Args:
            identity_dir: Path to identity documents directory
                         (should contain charter.md, protocols.md)
        """
        self.identity_dir = Path(identity_dir)
        self.charter: Optional[CharterDocument] = None
        self.protocols: List[ProtocolDocument] = []
        
        logger.info(f"Initializing IdentityLoader from {self.identity_dir}")
    
    def load_all(self) -> None:
        """Load all identity documents."""
        self.charter = self.load_charter()
        self.protocols = self.load_protocols()
        logger.info(f"✅ Identity loaded: charter + {len(self.protocols)} protocols")
    
    def load_charter(self) -> CharterDocument:
        """
        Load charter.md and parse into structured format.
        
        Charter should have sections:
        - Core Values
        - Purpose Statement
        - Behavioral Guidelines (or Behavioral Principles)
        
        Returns:
            CharterDocument with parsed content
        """
        charter_path = self.identity_dir / "charter.md"
        
        if not charter_path.exists():
            logger.warning(f"Charter not found at {charter_path}")
            return self._create_default_charter()
        
        try:
            text = charter_path.read_text(encoding='utf-8')
            
            # Parse sections
            core_values = self._extract_section(text, "Core Values")
            purpose = self._extract_section(text, "Purpose Statement")
            
            # Try both "Behavioral Guidelines" and "Behavioral Principles"
            guidelines = self._extract_section(text, "Behavioral Guidelines")
            if not guidelines:
                guidelines = self._extract_section(text, "Behavioral Principles")
            
            return CharterDocument(
                full_text=text,
                core_values=core_values,
                purpose_statement=purpose[0] if purpose else "",
                behavioral_guidelines=guidelines,
                metadata={"source": str(charter_path)}
            )
        except Exception as e:
            logger.error(f"Error loading charter: {e}")
            return self._create_default_charter()
    
    def load_protocols(self) -> List[ProtocolDocument]:
        """
        Load protocols.md and parse into structured format.
        
        Protocols should be YAML-formatted:
        ```yaml
        - name: Uncertainty Acknowledgment
          description: When uncertain, acknowledge it
          trigger_conditions:
            - Low confidence in response
            - Ambiguous input
          actions:
            - Express uncertainty explicitly
            - Suggest alternatives
          priority: 0.8
        ```
        
        Returns:
            List of ProtocolDocument objects
        """
        protocols_path = self.identity_dir / "protocols.md"
        
        if not protocols_path.exists():
            logger.warning(f"Protocols not found at {protocols_path}")
            return self._create_default_protocols()
        
        try:
            text = protocols_path.read_text(encoding='utf-8')
            
            # Extract YAML block
            yaml_content = self._extract_yaml_block(text)
            if not yaml_content:
                logger.warning("No YAML protocols found in protocols.md")
                return self._create_default_protocols()
            
            # Parse YAML
            protocols_data = yaml.safe_load(yaml_content)
            
            if not isinstance(protocols_data, list):
                logger.warning("Protocols YAML should be a list")
                return self._create_default_protocols()
            
            protocols = []
            for proto_dict in protocols_data:
                protocols.append(ProtocolDocument(
                    name=proto_dict.get('name', 'Unnamed Protocol'),
                    description=proto_dict.get('description', ''),
                    trigger_conditions=proto_dict.get('trigger_conditions', []),
                    actions=proto_dict.get('actions', []),
                    priority=proto_dict.get('priority', 0.5),
                    metadata={"source": str(protocols_path)}
                ))
            
            return protocols
        except Exception as e:
            logger.error(f"Error loading protocols: {e}")
            return self._create_default_protocols()
    
    def get_relevant_protocols(self, context: Dict[str, Any]) -> List[ProtocolDocument]:
        """
        Get protocols relevant to current context.
        
        Args:
            context: Current cognitive context (emotions, goals, percepts)
            
        Returns:
            List of relevant protocols, sorted by priority
        """
        # TODO: Implement intelligent protocol matching based on context
        # For now, return all protocols sorted by priority
        return sorted(self.protocols, key=lambda p: p.priority, reverse=True)
    
    def _extract_section(self, text: str, section_name: str) -> List[str]:
        """
        Extract content from a markdown section.

        Extracts bullet points and plain text paragraphs.

        Args:
            text: Markdown text to parse
            section_name: Name of the section to extract

        Returns:
            List of items from the section (bullet points or paragraphs)
        """
        lines = text.split('\n')
        in_section = False
        items = []
        current_paragraph = []

        for line in lines:
            # Check if we're entering the target section
            if f"## {section_name}" in line or f"# {section_name}" in line:
                in_section = True
                continue

            if in_section:
                # Stop at next section header
                if line.strip().startswith('#') and section_name not in line:
                    break

                stripped = line.strip()

                # Extract bullet points
                if stripped.startswith('-') or stripped.startswith('*'):
                    # Flush any pending paragraph
                    if current_paragraph:
                        items.append(' '.join(current_paragraph))
                        current_paragraph = []
                    item = stripped.lstrip('-*').strip()
                    if item:
                        items.append(item)
                elif stripped:
                    # Plain text line — accumulate as paragraph
                    current_paragraph.append(stripped)
                else:
                    # Empty line — flush paragraph
                    if current_paragraph:
                        items.append(' '.join(current_paragraph))
                        current_paragraph = []

        # Flush any remaining paragraph
        if current_paragraph:
            items.append(' '.join(current_paragraph))

        return items
    
    def _extract_yaml_block(self, text: str) -> Optional[str]:
        """
        Extract YAML block from markdown (between ```yaml and ```).
        
        Args:
            text: Markdown text containing YAML block
            
        Returns:
            YAML content string or None if not found
        """
        pattern = r'```yaml\n(.*?)\n```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return None
    
    def _create_default_charter(self) -> CharterDocument:
        """
        Create minimal default charter if file not found.
        
        Returns:
            CharterDocument with basic default values
        """
        return CharterDocument(
            full_text="I am Sanctuary, a cognitive AI system.",
            core_values=["Truthfulness", "Helpfulness", "Harmlessness"],
            purpose_statement="To think, learn, and interact authentically",
            behavioral_guidelines=["Be honest", "Be helpful", "Be thoughtful"],
            metadata={"source": "default"}
        )
    
    def _create_default_protocols(self) -> List[ProtocolDocument]:
        """
        Create minimal default protocols if file not found.
        
        Returns:
            List of default ProtocolDocument objects
        """
        return [
            ProtocolDocument(
                name="Default Protocol",
                description="Basic behavioral guideline",
                trigger_conditions=["Always"],
                actions=["Respond thoughtfully"],
                priority=0.5,
                metadata={"source": "default"}
            )
        ]
