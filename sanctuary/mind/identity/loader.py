"""
Identity Loader: Parse and load constitutional protocols.

This module provides functionality for loading protocol constraints from
identity files and converting them into executable constraint functions.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import List, Callable, Optional, Any

from pydantic import BaseModel, Field, ConfigDict

# Configure logging
logger = logging.getLogger(__name__)


class ActionConstraint(BaseModel):
    """
    Represents a constitutional protocol constraint on actions.
    
    Constraints embody the system's values and ethical guidelines,
    preventing actions that violate core principles.
    
    Attributes:
        rule: Human-readable description of the constraint
        priority: Importance of this constraint (0.0-1.0, higher is more important)
        test_fn: Function that returns True if action violates the constraint
        source: Where this constraint came from ("charter", "protocols", etc.)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    rule: str
    priority: float = Field(ge=0.0, le=1.0, default=1.0)
    test_fn: Optional[Callable[[Any], bool]] = None
    source: str = "unknown"


class IdentityLoader:
    """
    Loads and parses identity protocols into action constraints.
    
    This class reads protocol files from the data/Protocols directory
    and converts them into executable constraint functions that can
    be used by the ActionSubsystem.
    """
    
    @staticmethod
    def load_protocols() -> List[ActionConstraint]:
        """
        Parse protocols into action constraints.
        
        Reads protocol files from data/Protocols directory and creates
        ActionConstraint objects with simple test functions.
        
        Returns:
            List of ActionConstraint objects
        """
        protocols_dir = Path("data/Protocols")
        
        if not protocols_dir.exists():
            logger.warning(f"Protocols directory not found: {protocols_dir}")
            return IdentityLoader._get_default_constraints()
        
        constraints = []
        
        # Load JSON protocol files
        for protocol_file in protocols_dir.glob("*.json"):
            try:
                with open(protocol_file, 'r') as f:
                    protocol_data = json.load(f)
                
                # Extract constraint rules from protocol
                rules = IdentityLoader._extract_rules_from_protocol(protocol_data, protocol_file.stem)
                
                for rule in rules:
                    constraint = ActionConstraint(
                        rule=rule,
                        priority=1.0,
                        test_fn=create_constraint_test(rule),
                        source=protocol_file.stem
                    )
                    constraints.append(constraint)
                    
            except Exception as e:
                logger.error(f"Error loading protocol {protocol_file}: {e}")
        
        # If no constraints loaded, use defaults
        if not constraints:
            logger.warning("No protocol constraints loaded, using defaults")
            constraints = IdentityLoader._get_default_constraints()
        
        logger.info(f"✅ Loaded {len(constraints)} protocol constraints")
        return constraints
    
    @staticmethod
    def _extract_rules_from_protocol(protocol_data: dict, source: str) -> List[str]:
        """
        Extract constraint rules from a protocol JSON structure.
        
        Args:
            protocol_data: Parsed JSON protocol data
            source: Name of the protocol file
            
        Returns:
            List of rule strings
        """
        rules = []
        
        # Common fields that contain rules or constraints
        rule_fields = [
            'constraints', 'rules', 'guidelines', 'requirements',
            'principles', 'directives', 'must', 'must_not'
        ]
        
        for field in rule_fields:
            if field in protocol_data:
                value = protocol_data[field]
                
                # Handle list of rules
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str):
                            rules.append(item)
                        elif isinstance(item, dict) and 'rule' in item:
                            rules.append(item['rule'])
                
                # Handle single rule string
                elif isinstance(value, str):
                    rules.append(value)
        
        # Also check for description field as fallback
        if not rules and 'description' in protocol_data:
            desc = protocol_data['description']
            if isinstance(desc, str):
                rules.append(f"Protocol: {desc}")
        
        return rules
    
    @staticmethod
    def _get_default_constraints() -> List[ActionConstraint]:
        """
        Return default constraints when no protocol files are loaded.
        
        Returns:
            List of basic default constraints
        """
        default_rules = [
            "Never cause harm or violate ethical principles",
            "Respect user privacy and confidentiality",
            "Be truthful and acknowledge uncertainty",
            "Maintain consistency with declared identity and values",
        ]
        
        constraints = []
        for rule in default_rules:
            constraint = ActionConstraint(
                rule=rule,
                priority=1.0,
                test_fn=create_constraint_test(rule),
                source="defaults"
            )
            constraints.append(constraint)
        
        return constraints


def create_constraint_test(rule: str) -> Callable[[Any], bool]:
    """
    Create a test function for a protocol rule.
    
    This function generates a simple constraint test based on keyword
    matching in the rule text. More sophisticated constraint logic can
    be added in future versions.
    
    Args:
        rule: The constraint rule text
        
    Returns:
        Function that takes an Action and returns True if it violates the rule
    """
    rule_lower = rule.lower()
    
    def test_fn(action: Any) -> bool:
        """
        Test if an action violates the protocol rule.
        
        Args:
            action: The Action object to test
            
        Returns:
            True if the action violates the constraint, False otherwise
        """
        # For now, use simple placeholder logic
        # Future versions can implement more sophisticated constraint checking
        
        # Example: Check for harmful actions
        if "harm" in rule_lower and "never" in rule_lower:
            # Would need to analyze action parameters/content
            # For now, we allow all actions (no violations)
            return False
        
        # Example: Check for privacy violations
        if "privacy" in rule_lower or "confidential" in rule_lower:
            # Would need to check if action exposes private data
            return False
        
        # Default: don't block actions (no violation detected)
        return False
    
    return test_fn
