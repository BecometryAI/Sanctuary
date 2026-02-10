"""
Tools for voice interaction in Sanctuary's specialist system
"""
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from .voice_processor import VoiceProcessor
from .discord_client import SanctuaryDiscordClient

logger = logging.getLogger(__name__)

class VoiceToolkit:
    """Toolkit for voice-enabled specialist interactions"""
    
    def __init__(self, voice_path: Optional[str] = None):
        """
        Initialize voice toolkit
        
        Args:
            voice_path: Path to Sanctuary's voice file
        """
        self.voice_processor = VoiceProcessor(voice_path)
        self.discord_client = SanctuaryDiscordClient(self.voice_processor)
        self.voice_enabled = False
        
    async def initialize(self) -> None:
        """Initialize voice components"""
        try:
            await self.discord_client.start()
            logger.info("Voice toolkit initialized")
        except Exception as e:
            logger.error(f"Failed to initialize voice toolkit: {e}")
            raise
            
    async def shutdown(self) -> None:
        """Clean up voice components"""
        try:
            if self.discord_client:
                await self.discord_client.close()
        except Exception as e:
            logger.error(f"Error shutting down voice toolkit: {e}")
            
    async def join_voice(self, channel_id: str) -> bool:
        """
        Join a voice channel
        
        Args:
            channel_id: Discord voice channel ID
            
        Returns:
            bool indicating success
        """
        success = await self.discord_client.join_voice_channel(channel_id)
        if success:
            self.voice_enabled = True
        return success
        
    async def leave_voice(self) -> None:
        """Leave voice channel"""
        await self.discord_client.leave_voice_channel()
        self.voice_enabled = False
        
    async def speak(self, text: str) -> None:
        """
        Speak text if voice is enabled
        
        Args:
            text: Text to speak
        """
        if self.voice_enabled:
            await self.discord_client.speak(text)
        else:
            logger.debug("Voice not enabled - skipping TTS")