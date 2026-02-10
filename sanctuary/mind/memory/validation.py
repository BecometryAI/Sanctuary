"""
Memory Entry Validation System

Validates memory entries before writing to ensure data consistency and integrity.

Author: Sanctuary Team
Date: January 2, 2026
"""

import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from uuid import UUID
import re

from ..exceptions import ValidationError
from ..logging_config import get_logger

logger = get_logger(__name__)


class MemoryValidator:
    """
    Validator for memory entries before ChromaDB writes.
    
    Validates:
    - Schema compliance
    - Embedding dimensions
    - Metadata structure
    - Content constraints
    
    Example:
        validator = MemoryValidator(expected_embedding_dim=768)
        
        validator.validate_journal_entry(entry)
        validator.validate_embedding(embedding)
        validator.validate_metadata(metadata)
    """
    
    def __init__(
        self,
        expected_embedding_dim: int = 768,
        min_content_length: int = 1,
        max_content_length: int = 50000,
        max_tags: int = 20,
        max_tag_length: int = 50
    ):
        """
        Initialize memory validator.
        
        Args:
            expected_embedding_dim: Expected embedding dimension
            min_content_length: Minimum content length
            max_content_length: Maximum content length
            max_tags: Maximum number of tags
            max_tag_length: Maximum tag length
        """
        self.expected_embedding_dim = expected_embedding_dim
        self.min_content_length = min_content_length
        self.max_content_length = max_content_length
        self.max_tags = max_tags
        self.max_tag_length = max_tag_length
        
        logger.info(
            f"Memory validator initialized: embedding_dim={expected_embedding_dim}, "
            f"content_length={min_content_length}-{max_content_length}"
        )
    
    def validate_embedding(
        self,
        embedding: Union[List[float], Any],
        field_name: str = "embedding"
    ) -> bool:
        """
        Validate embedding dimension and values.
        
        Args:
            embedding: Embedding vector
            field_name: Name of field for error messages
        
        Returns:
            True if valid
        
        Raises:
            ValidationError: If validation fails
        """
        if embedding is None:
            raise ValidationError(
                "Embedding cannot be None",
                field=field_name
            )
        
        # Check if it's a list/array
        if not hasattr(embedding, '__len__'):
            raise ValidationError(
                "Embedding must be a list or array",
                field=field_name,
                value=type(embedding).__name__
            )
        
        # Check dimension
        if len(embedding) != self.expected_embedding_dim:
            raise ValidationError(
                f"Embedding dimension mismatch: expected {self.expected_embedding_dim}, "
                f"got {len(embedding)}",
                field=field_name,
                context={
                    "expected": self.expected_embedding_dim,
                    "actual": len(embedding)
                }
            )
        
        # Check for NaN or inf values
        try:
            for i, val in enumerate(embedding):
                if not isinstance(val, (int, float)):
                    raise ValidationError(
                        f"Embedding contains non-numeric value at index {i}",
                        field=field_name,
                        value=type(val).__name__
                    )
                
                # Check for invalid float values
                if isinstance(val, float):
                    if val != val:  # NaN check
                        raise ValidationError(
                            f"Embedding contains NaN at index {i}",
                            field=field_name
                        )
                    if abs(val) == float('inf'):
                        raise ValidationError(
                            f"Embedding contains infinity at index {i}",
                            field=field_name
                        )
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"Invalid embedding values: {e}",
                field=field_name
            )
        
        return True
    
    def validate_content(
        self,
        content: str,
        field_name: str = "content"
    ) -> bool:
        """
        Validate content text.
        
        Args:
            content: Content text
            field_name: Name of field for error messages
        
        Returns:
            True if valid
        
        Raises:
            ValidationError: If validation fails
        """
        if content is None:
            raise ValidationError(
                "Content cannot be None",
                field=field_name
            )
        
        if not isinstance(content, str):
            raise ValidationError(
                "Content must be a string",
                field=field_name,
                value=type(content).__name__
            )
        
        content_length = len(content)
        
        if content_length < self.min_content_length:
            raise ValidationError(
                f"Content too short: minimum {self.min_content_length} characters",
                field=field_name,
                context={
                    "min_length": self.min_content_length,
                    "actual_length": content_length
                }
            )
        
        if content_length > self.max_content_length:
            raise ValidationError(
                f"Content too long: maximum {self.max_content_length} characters",
                field=field_name,
                context={
                    "max_length": self.max_content_length,
                    "actual_length": content_length
                }
            )
        
        return True
    
    def validate_tags(
        self,
        tags: List[str],
        field_name: str = "tags"
    ) -> bool:
        """
        Validate tags list.
        
        Args:
            tags: List of tags
            field_name: Name of field for error messages
        
        Returns:
            True if valid
        
        Raises:
            ValidationError: If validation fails
        """
        if tags is None:
            return True  # Tags are optional
        
        if not isinstance(tags, list):
            raise ValidationError(
                "Tags must be a list",
                field=field_name,
                value=type(tags).__name__
            )
        
        if len(tags) > self.max_tags:
            raise ValidationError(
                f"Too many tags: maximum {self.max_tags}",
                field=field_name,
                context={
                    "max_tags": self.max_tags,
                    "actual_tags": len(tags)
                }
            )
        
        for i, tag in enumerate(tags):
            if not isinstance(tag, str):
                raise ValidationError(
                    f"Tag at index {i} must be a string",
                    field=f"{field_name}[{i}]",
                    value=type(tag).__name__
                )
            
            if len(tag) > self.max_tag_length:
                raise ValidationError(
                    f"Tag too long: maximum {self.max_tag_length} characters",
                    field=f"{field_name}[{i}]",
                    context={
                        "max_length": self.max_tag_length,
                        "actual_length": len(tag)
                    }
                )
            
            # Tags should be alphanumeric with hyphens/underscores
            if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                raise ValidationError(
                    "Tag contains invalid characters (allowed: a-z, A-Z, 0-9, -, _)",
                    field=f"{field_name}[{i}]",
                    value=tag
                )
        
        return True
    
    def validate_metadata(
        self,
        metadata: Dict[str, Any],
        field_name: str = "metadata"
    ) -> bool:
        """
        Validate metadata dictionary structure.
        
        Args:
            metadata: Metadata dictionary
            field_name: Name of field for error messages
        
        Returns:
            True if valid
        
        Raises:
            ValidationError: If validation fails
        """
        if metadata is None:
            return True  # Metadata is optional
        
        if not isinstance(metadata, dict):
            raise ValidationError(
                "Metadata must be a dictionary",
                field=field_name,
                value=type(metadata).__name__
            )
        
        # Validate metadata values are JSON-serializable
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValidationError(
                    "Metadata keys must be strings",
                    field=f"{field_name}.{key}",
                    value=type(key).__name__
                )
            
            # Check value types (ChromaDB supports limited types)
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValidationError(
                    f"Metadata value for '{key}' has unsupported type",
                    field=f"{field_name}.{key}",
                    value=type(value).__name__,
                    context={"supported_types": ["str", "int", "float", "bool", "None"]}
                )
        
        return True
    
    def validate_journal_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Validate a complete journal entry.
        
        Args:
            entry: Journal entry dictionary
        
        Returns:
            True if valid
        
        Raises:
            ValidationError: If validation fails
        """
        # Required fields
        required_fields = ["content", "summary"]
        for field in required_fields:
            if field not in entry:
                raise ValidationError(
                    f"Missing required field: {field}",
                    field=field
                )
        
        # Validate content
        self.validate_content(entry["content"], "content")
        self.validate_content(entry["summary"], "summary")
        
        # Validate optional fields
        if "tags" in entry:
            self.validate_tags(entry["tags"])
        
        if "metadata" in entry:
            self.validate_metadata(entry["metadata"])
        
        if "embedding" in entry:
            self.validate_embedding(entry["embedding"])
        
        # Validate significance score if present
        if "significance_score" in entry:
            score = entry["significance_score"]
            if not isinstance(score, (int, float)):
                raise ValidationError(
                    "Significance score must be numeric",
                    field="significance_score",
                    value=type(score).__name__
                )
            if not (1 <= score <= 10):
                raise ValidationError(
                    "Significance score must be between 1 and 10",
                    field="significance_score",
                    value=score
                )
        
        return True
    
    def validate_fact_entry(self, entry: Dict[str, Any]) -> bool:
        """
        Validate a fact entry.
        
        Args:
            entry: Fact entry dictionary
        
        Returns:
            True if valid
        
        Raises:
            ValidationError: If validation fails
        """
        # Required fields for facts
        required_fields = ["content"]
        for field in required_fields:
            if field not in entry:
                raise ValidationError(
                    f"Missing required field: {field}",
                    field=field
                )
        
        # Validate content
        self.validate_content(entry["content"], "content")
        
        # Validate optional fields
        if "tags" in entry:
            self.validate_tags(entry["tags"])
        
        if "metadata" in entry:
            self.validate_metadata(entry["metadata"])
        
        if "embedding" in entry:
            self.validate_embedding(entry["embedding"])
        
        return True


# Global validator instance
_global_validator: Optional[MemoryValidator] = None


def get_global_validator() -> MemoryValidator:
    """
    Get or create global memory validator.
    
    Returns:
        Global MemoryValidator instance
    """
    global _global_validator
    if _global_validator is None:
        _global_validator = MemoryValidator()
    return _global_validator


def validate_before_write(entry: Dict[str, Any], entry_type: str = "journal") -> bool:
    """
    Convenience function to validate entry before write.
    
    Args:
        entry: Entry to validate
        entry_type: Type of entry ("journal" or "fact")
    
    Returns:
        True if valid
    
    Raises:
        ValidationError: If validation fails
    """
    validator = get_global_validator()
    
    if entry_type == "journal":
        return validator.validate_journal_entry(entry)
    elif entry_type == "fact":
        return validator.validate_fact_entry(entry)
    else:
        raise ValidationError(
            f"Unknown entry type: {entry_type}",
            field="entry_type",
            value=entry_type
        )
