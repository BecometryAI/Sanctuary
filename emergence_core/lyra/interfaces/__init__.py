"""
Language Interfaces: Peripheral language I/O adapters.

LLMs are used here for parsing input and generating output, but they
are NOT the cognitive core. The actual "mind" is the non-linguistic
recurrent loop in cognitive_core/.
"""

from __future__ import annotations

from .language_input import LanguageInputParser
from .language_output import LanguageOutputGenerator

__all__ = [
    "LanguageInputParser",
    "LanguageOutputGenerator",
]
