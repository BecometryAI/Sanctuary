"""
DEPRECATED: This module has been replaced by voice_toolkit.py and voice_processor.py

This module contains legacy voice processing functions that have been migrated
to the new voice system architecture. Please use the VoiceToolkit class from
voice_toolkit.py instead.

The functionality has been reorganized as follows:

- VoiceToolkit (voice_toolkit.py)
  - Main interface for voice interactions
  - Manages Discord integration
  - Handles voice channel joining/leaving
  - Coordinates speech and listening

- VoiceProcessor (voice_processor.py)  
  - Core speech processing
  - Text-to-speech generation
  - Speech-to-text transcription
  - Voice model management

For new code, import and use these modules instead.
"""
import warnings

warnings.warn(
    "voice_tools.py is deprecated. Use voice_toolkit.py instead.",
    DeprecationWarning,
    stacklevel=2
)