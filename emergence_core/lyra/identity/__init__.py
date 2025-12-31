"""
Identity module for loading and managing constitutional protocols.

This module provides functionality for loading identity constraints and
protocols that govern the cognitive system's behavior.
"""

from .loader import IdentityLoader, ActionConstraint

__all__ = ['IdentityLoader', 'ActionConstraint']
