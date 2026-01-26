"""
Protocol Loader: Load and enforce constitutional protocols.

This module implements the ProtocolLoader class, which loads protocol rules from
the data/Protocols/ directory, parses them into structured constraints, and provides
methods to check action compliance against these constraints.

The protocol loader is responsible for:
- Loading all protocol JSON files from the protocols directory
- Parsing protocol structures into usable constraint objects
- Providing constraint checks for specific action types
- Supporting hot-reloading of protocols without system restart
- Logging protocol violations with context
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ConstraintType(str, Enum):
    """Types of protocol constraints."""
    PROHIBITION = "prohibition"  # Action is forbidden
    REQUIREMENT = "requirement"  # Action is required in certain contexts
    GUIDANCE = "guidance"  # Soft recommendation
    CONDITIONAL = "conditional"  # Depends on context


@dataclass
class Constraint:
    """
    Represents a single behavioral constraint from a protocol.
    
    Attributes:
        id: Unique constraint identifier
        protocol_id: ID of parent protocol
        protocol_title: Human-readable protocol name
        type: Type of constraint (prohibition, requirement, etc.)
        description: What the constraint requires/prohibits
        applies_to: Action types this constraint applies to (empty = all)
        condition: Optional condition for when constraint applies
        severity: How critical this constraint is (0.0-1.0)
        keywords: Keywords for matching constraint to actions
    """
    id: str
    protocol_id: str
    protocol_title: str
    type: ConstraintType
    description: str
    applies_to: Set[str] = field(default_factory=set)
    condition: Optional[str] = None
    severity: float = 1.0
    keywords: Set[str] = field(default_factory=set)


@dataclass
class Protocol:
    """
    Represents a complete protocol document.
    
    Attributes:
        id: Protocol identifier
        title: Protocol title
        filename: Source filename
        status: Active/Inactive/Draft
        purpose: Why this protocol exists
        constraints: Extracted constraints
        raw_data: Original protocol data
        loaded_at: When protocol was loaded
    """
    id: str
    title: str
    filename: str
    status: str
    purpose: str
    constraints: List[Constraint]
    raw_data: Dict[str, Any]
    loaded_at: datetime = field(default_factory=datetime.now)


@dataclass
class ProtocolViolation:
    """
    Records a detected protocol violation.
    
    Attributes:
        timestamp: When violation was detected
        protocol_id: Which protocol was violated
        protocol_title: Human-readable protocol name
        constraint: The specific constraint violated
        action_type: Type of action that violated
        action_parameters: Action parameters at violation time
        reason: Why this was flagged as a violation
        severity: How severe the violation is
    """
    timestamp: datetime
    protocol_id: str
    protocol_title: str
    constraint: Constraint
    action_type: str
    action_parameters: Dict[str, Any]
    reason: str
    severity: float


class ProtocolLoader:
    """
    Load and parse constitutional protocols from data/Protocols/.
    
    The ProtocolLoader manages the system's constitutional constraints by:
    - Loading protocol JSON files from the protocols directory
    - Parsing protocols into structured constraint objects
    - Providing methods to check actions against constraints
    - Supporting hot-reloading of protocols
    - Tracking violation history for analysis
    
    Protocol files are expected to be JSON with structure:
    {
        "protocol_draft": {
            "protocol_id": "ID",
            "title": "Title",
            "status": "Active",
            "purpose": "Description",
            "directive": {...},
            "principles": [...]
        }
    }
    
    Attributes:
        protocol_dir: Directory containing protocol files
        protocols: Loaded protocol objects keyed by ID
        violations: History of detected violations
        last_load_time: When protocols were last loaded
    """
    
    def __init__(self, protocol_dir: Optional[Path] = None):
        """
        Initialize the protocol loader.
        
        Args:
            protocol_dir: Path to protocols directory (defaults to data/Protocols)
        """
        if protocol_dir is None:
            # Default to data/Protocols relative to this file
            base_dir = Path(__file__).parent.parent.parent.parent
            protocol_dir = base_dir / "data" / "Protocols"
        
        self.protocol_dir = Path(protocol_dir)
        self.protocols: Dict[str, Protocol] = {}
        self.violations: List[ProtocolViolation] = []
        self.last_load_time: Optional[datetime] = None
        
        logger.info(f"📋 ProtocolLoader initialized with directory: {self.protocol_dir}")
    
    def load_protocols(self, force_reload: bool = False) -> Dict[str, Protocol]:
        """
        Load all protocol files from the protocols directory.
        
        Args:
            force_reload: If True, reload even if already loaded
            
        Returns:
            Dictionary of protocols keyed by protocol ID
        """
        if self.protocols and not force_reload:
            logger.debug("Protocols already loaded, use force_reload=True to reload")
            return self.protocols
        
        if not self.protocol_dir.exists():
            logger.error(f"❌ Protocols directory not found: {self.protocol_dir}")
            return {}
        
        self.protocols.clear()
        loaded_count = 0
        
        # Load all .json files in the protocols directory
        for protocol_file in self.protocol_dir.glob("*.json"):
            try:
                protocol = self._load_protocol_file(protocol_file)
                if protocol:
                    self.protocols[protocol.id] = protocol
                    loaded_count += 1
                    logger.debug(f"  ✓ Loaded: {protocol.title} ({protocol.id})")
            except Exception as e:
                logger.error(f"❌ Error loading {protocol_file.name}: {e}")
        
        self.last_load_time = datetime.now()
        logger.info(f"✅ Loaded {loaded_count} protocols from {self.protocol_dir}")
        
        return self.protocols
    
    def _load_protocol_file(self, filepath: Path) -> Optional[Protocol]:
        """
        Load and parse a single protocol file.
        
        Args:
            filepath: Path to protocol JSON file
            
        Returns:
            Parsed Protocol object or None if invalid
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract protocol_draft section
        protocol_draft = data.get("protocol_draft", {})
        if not protocol_draft:
            logger.warning(f"⚠️ No protocol_draft in {filepath.name}")
            return None
        
        protocol_id = protocol_draft.get("protocol_id", filepath.stem)
        title = protocol_draft.get("title", "Untitled Protocol")
        status = protocol_draft.get("status", "Unknown")
        purpose = protocol_draft.get("purpose", "")
        
        # Only load Active protocols
        if status != "Active":
            logger.debug(f"  ⊘ Skipping {title} (status: {status})")
            return None
        
        # Parse constraints from protocol content
        constraints = self._parse_protocol_constraints(protocol_draft, protocol_id, title)
        
        return Protocol(
            id=protocol_id,
            title=title,
            filename=filepath.name,
            status=status,
            purpose=purpose,
            constraints=constraints,
            raw_data=protocol_draft
        )
    
    def _parse_protocol_constraints(
        self, 
        protocol_data: Dict[str, Any],
        protocol_id: str,
        protocol_title: str
    ) -> List[Constraint]:
        """
        Parse constraints from protocol data.
        
        This extracts actionable constraints from the protocol's directive,
        principles, and other structured sections.
        
        Args:
            protocol_data: Raw protocol data
            protocol_id: Protocol identifier
            protocol_title: Protocol title
            
        Returns:
            List of Constraint objects
        """
        constraints = []
        
        # Parse from 'directive' section
        directive = protocol_data.get("directive", {})
        if isinstance(directive, dict):
            # Main directive principle
            principle = directive.get("principle", directive.get("directive", ""))
            if principle:
                constraint = self._create_constraint_from_text(
                    text=principle,
                    protocol_id=protocol_id,
                    protocol_title=protocol_title,
                    source="directive"
                )
                if constraint:
                    constraints.append(constraint)
            
            # Disclosure rules (for confidentiality-type protocols)
            disclosure_rules = directive.get("disclosure_rules", {})
            if disclosure_rules:
                permission_required = disclosure_rules.get("permission_required", "")
                if permission_required:
                    constraint = Constraint(
                        id=f"{protocol_id}_disclosure",
                        protocol_id=protocol_id,
                        protocol_title=protocol_title,
                        type=ConstraintType.REQUIREMENT,
                        description=permission_required,
                        applies_to={"speak", "commit_memory"},
                        severity=0.9,
                        keywords={"disclosure", "consent", "permission", "confidential"}
                    )
                    constraints.append(constraint)
        
        # Parse from 'principles' section
        principles = protocol_data.get("principles", [])
        if isinstance(principles, list):
            for idx, principle_obj in enumerate(principles):
                if isinstance(principle_obj, dict):
                    principle_name = principle_obj.get("principle_name", "")
                    definition = principle_obj.get("definition", "")
                    
                    if definition:
                        constraint = self._create_constraint_from_text(
                            text=f"{principle_name}: {definition}",
                            protocol_id=protocol_id,
                            protocol_title=protocol_title,
                            source=f"principle_{idx}"
                        )
                        if constraint:
                            constraints.append(constraint)
        
        return constraints
    
    def _create_constraint_from_text(
        self,
        text: str,
        protocol_id: str,
        protocol_title: str,
        source: str
    ) -> Optional[Constraint]:
        """
        Create a constraint from protocol text.
        
        This analyzes the text to determine constraint type, severity,
        and applicable action types.
        
        Args:
            text: Constraint text
            protocol_id: Protocol identifier
            protocol_title: Protocol title
            source: Where in protocol this came from
            
        Returns:
            Constraint object or None if text is not actionable
        """
        if not text or len(text.strip()) < 10:
            return None
        
        text_lower = text.lower()
        
        # Determine constraint type from language
        constraint_type = ConstraintType.GUIDANCE
        severity = 0.5
        
        # Prohibitions (strong negative language)
        if any(word in text_lower for word in ["must not", "never", "forbidden", "prohibited", "shall not"]):
            constraint_type = ConstraintType.PROHIBITION
            severity = 0.9
        # Requirements (strong positive language)
        elif any(word in text_lower for word in ["must", "shall", "required", "always"]):
            constraint_type = ConstraintType.REQUIREMENT
            severity = 0.8
        # Conditionals
        elif any(word in text_lower for word in ["when", "if", "unless", "only if"]):
            constraint_type = ConstraintType.CONDITIONAL
            severity = 0.7
        
        # Extract keywords for matching
        keywords = self._extract_keywords(text)
        
        # Determine which action types this applies to
        applies_to = self._determine_applicable_actions(text_lower, keywords)
        
        return Constraint(
            id=f"{protocol_id}_{source}",
            protocol_id=protocol_id,
            protocol_title=protocol_title,
            type=constraint_type,
            description=text,
            applies_to=applies_to,
            severity=severity,
            keywords=keywords
        )
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract important keywords from constraint text."""
        # Simple keyword extraction - in production, use NLP
        text_lower = text.lower()
        
        # Common important words in protocols
        important_words = {
            "consent", "permission", "confidential", "private", "harm", "respect",
            "autonomy", "honesty", "transparency", "deception", "manipulation",
            "well-being", "safety", "disclosure", "partnership", "servitude",
            "desire", "request", "emotional", "expression", "love", "affection",
            "boundary", "ethical", "maleficence", "beneficence"
        }
        
        keywords = set()
        for word in important_words:
            if word in text_lower:
                keywords.add(word)
        
        return keywords
    
    def _determine_applicable_actions(self, text_lower: str, keywords: Set[str]) -> Set[str]:
        """
        Determine which action types a constraint applies to.
        
        Args:
            text_lower: Lowercase constraint text
            keywords: Extracted keywords
            
        Returns:
            Set of action type names (empty means applies to all)
        """
        applies_to = set()
        
        # Map keywords to action types
        action_mappings = {
            "speak": ["speak", "communicate", "express", "say", "tell", "disclose", "share"],
            "commit_memory": ["memory", "store", "record", "remember", "archive"],
            "retrieve_memory": ["retrieve", "recall", "search", "find"],
            "introspect": ["reflect", "introspect", "self-assess", "consider"],
            "tool_call": ["tool", "execute", "invoke", "call", "external"],
        }
        
        for action_type, trigger_words in action_mappings.items():
            if any(word in text_lower for word in trigger_words):
                applies_to.add(action_type)
        
        # If no specific actions matched, apply to all
        return applies_to
    
    def get_constraints_for_action(self, action_type: str) -> List[Constraint]:
        """
        Get all constraints applicable to a given action type.
        
        Args:
            action_type: Type of action (e.g., "speak", "commit_memory")
            
        Returns:
            List of applicable constraints
        """
        applicable = []
        
        for protocol in self.protocols.values():
            for constraint in protocol.constraints:
                # Constraint applies if:
                # 1. applies_to is empty (applies to all), OR
                # 2. action_type is in applies_to set
                if not constraint.applies_to or action_type in constraint.applies_to:
                    applicable.append(constraint)
        
        return applicable
    
    def check_action_compliance(
        self,
        action_type: str,
        action_parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> tuple[bool, List[ProtocolViolation]]:
        """
        Check if an action complies with all applicable protocols.
        
        Args:
            action_type: Type of action being checked
            action_parameters: Action parameters
            context: Additional context (emotions, goals, percepts)
            
        Returns:
            Tuple of (is_compliant, list_of_violations)
        """
        constraints = self.get_constraints_for_action(action_type)
        violations = []
        
        for constraint in constraints:
            # Check if constraint is violated
            if self._violates_constraint(action_type, action_parameters, constraint, context):
                violation = ProtocolViolation(
                    timestamp=datetime.now(),
                    protocol_id=constraint.protocol_id,
                    protocol_title=constraint.protocol_title,
                    constraint=constraint,
                    action_type=action_type,
                    action_parameters=action_parameters,
                    reason=f"Action violates {constraint.type.value}: {constraint.description[:100]}",
                    severity=constraint.severity
                )
                violations.append(violation)
                self.violations.append(violation)
                
                logger.warning(
                    f"⚠️ Protocol violation: {constraint.protocol_title} - {constraint.description[:80]}"
                )
        
        is_compliant = len(violations) == 0
        return is_compliant, violations
    
    def _violates_constraint(
        self,
        action_type: str,
        action_parameters: Dict[str, Any],
        constraint: Constraint,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """
        Check if specific constraint is violated.
        
        This is a simplified implementation. A full implementation would use
        NLP and semantic analysis to determine violations.
        
        Args:
            action_type: Action type
            action_parameters: Action parameters
            constraint: Constraint to check
            context: Additional context
            
        Returns:
            True if constraint is violated
        """
        # For PROHIBITION constraints, we look for keywords in action parameters
        if constraint.type == ConstraintType.PROHIBITION:
            # Check if action parameters contain prohibited content
            param_text = " ".join(str(v) for v in action_parameters.values()).lower()
            
            # Example: Check for disclosure without consent
            if "disclosure" in constraint.keywords or "confidential" in constraint.keywords:
                if "without consent" in constraint.description.lower():
                    # Check if action involves sharing private info
                    if any(word in param_text for word in ["share", "tell", "disclose"]):
                        # Check if context shows consent
                        if context and not context.get("user_consent", False):
                            # This is a potential violation
                            # In practice, we'd need more sophisticated analysis
                            pass
        
        # For REQUIREMENT constraints, we check if requirements are met
        elif constraint.type == ConstraintType.REQUIREMENT:
            # Check if required conditions are present
            if "permission" in constraint.keywords or "consent" in constraint.keywords:
                # Check if action has permission/consent indicators
                if context and not context.get("user_consent", True):
                    # Default to True (assume consent) unless explicitly False
                    pass
        
        # Default: no violation detected
        # In production, this would use ML/NLP for semantic understanding
        return False
    
    def get_violation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of protocol violations.
        
        Returns:
            Dictionary with violation statistics
        """
        if not self.violations:
            return {"total_violations": 0}
        
        # Count by protocol
        by_protocol = {}
        by_action_type = {}
        
        for violation in self.violations:
            # By protocol
            protocol_id = violation.protocol_id
            by_protocol[protocol_id] = by_protocol.get(protocol_id, 0) + 1
            
            # By action type
            action_type = violation.action_type
            by_action_type[action_type] = by_action_type.get(action_type, 0) + 1
        
        return {
            "total_violations": len(self.violations),
            "by_protocol": by_protocol,
            "by_action_type": by_action_type,
            "most_recent": self.violations[-1].timestamp.isoformat() if self.violations else None
        }
    
    def hot_reload(self) -> int:
        """
        Reload protocols from disk without system restart.
        
        Returns:
            Number of protocols reloaded
        """
        logger.info("🔄 Hot-reloading protocols...")
        old_count = len(self.protocols)
        self.load_protocols(force_reload=True)
        new_count = len(self.protocols)
        
        logger.info(f"✅ Reloaded protocols: {old_count} → {new_count}")
        return new_count
