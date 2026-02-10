"""
Audio Gateway Orchestration

Coordinates ASR server and microphone client for seamless integration.
Provides high-level interface for Sanctuary's auditory perception.

Features:
- Unified audio processing interface
- Automatic server/client lifecycle management
- Error handling and reconnection
- Integration with Sanctuary's cognitive pipeline
"""

import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from pathlib import Path

from .asr_server import ASRServer
from .mic_client import MicrophoneClient

logger = logging.getLogger(__name__)


class AudioGateway:
    """High-level audio gateway for Sanctuary's hearing system"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        sample_rate: int = 16000,
        language: str = "en",
        auto_start_server: bool = True
    ):
        """Initialize audio gateway.
        
        Args:
            host: Server host
            port: Server port
            sample_rate: Audio sample rate
            language: Language code for ASR
            auto_start_server: Automatically start ASR server
        """
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.language = language
        self.auto_start_server = auto_start_server
        
        # Components
        self.server: Optional[ASRServer] = None
        self.client: Optional[MicrophoneClient] = None
        
        # State
        self.running = False
        self.server_task: Optional[asyncio.Task] = None
        self.client_task: Optional[asyncio.Task] = None
        
        logger.info("Audio gateway initialized")
    
    async def start(self, transcription_callback: Optional[Callable] = None):
        """Start audio gateway (server + client).
        
        Args:
            transcription_callback: Function to call with transcription results
        """
        logger.info("Starting audio gateway...")
        
        # Start ASR server if needed
        if self.auto_start_server:
            await self._start_server()
        
        # Start microphone client
        await self._start_client(transcription_callback)
        
        self.running = True
        logger.info("Audio gateway running")
    
    async def stop(self):
        """Stop audio gateway."""
        logger.info("Stopping audio gateway...")
        self.running = False
        
        # Stop client
        if self.client:
            await self.client.stop_streaming()
            await self.client.disconnect()
        
        # Stop server
        if self.server:
            await self.server.stop()
        
        # Cancel tasks
        if self.client_task and not self.client_task.done():
            self.client_task.cancel()
            try:
                await self.client_task
            except asyncio.CancelledError:
                pass
        
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Audio gateway stopped")
    
    async def _start_server(self):
        """Start ASR server."""
        logger.info("Starting ASR server...")
        
        self.server = ASRServer(
            host=self.host,
            port=self.port,
            sample_rate=self.sample_rate,
            language=self.language
        )
        
        await self.server.start()
        logger.info("ASR server started")
    
    async def _start_client(self, transcription_callback: Optional[Callable] = None):
        """Start microphone client.
        
        Args:
            transcription_callback: Function to call with transcriptions
        """
        logger.info("Starting microphone client...")
        
        server_url = f"ws://{self.host}:{self.port}"
        self.client = MicrophoneClient(
            server_url=server_url,
            sample_rate=self.sample_rate
        )
        
        # Set callback if provided
        if transcription_callback:
            self.client.set_transcription_callback(transcription_callback)
        
        # Connect to server
        connected = await self.client.connect()
        if not connected:
            raise ConnectionError("Failed to connect microphone client to ASR server")
        
        # Start streaming in background
        self.client_task = asyncio.create_task(self.client.start_streaming())
        logger.info("Microphone client started")
    
    async def change_language(self, language: str):
        """Change ASR language.
        
        Args:
            language: New language code
        """
        self.language = language
        
        if self.client and self.client.connected:
            await self.client.send_control("config", language=language)
            logger.info(f"Language changed to: {language}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get gateway status.
        
        Returns:
            Status dictionary
        """
        server_stats = self.server.get_stats() if self.server else {}
        
        return {
            "running": self.running,
            "server": server_stats,
            "client_connected": self.client.connected if self.client else False,
            "language": self.language,
            "sample_rate": self.sample_rate
        }


# Convenience function for quick setup
async def create_audio_gateway(
    transcription_callback: Callable[[str, dict], None],
    **kwargs
) -> AudioGateway:
    """Create and start audio gateway with callback.
    
    Args:
        transcription_callback: Function to handle transcriptions
        **kwargs: Additional AudioGateway arguments
        
    Returns:
        Running AudioGateway instance
    """
    gateway = AudioGateway(**kwargs)
    await gateway.start(transcription_callback)
    return gateway


async def main():
    """Test audio gateway standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define transcription handler
    def handle_transcription(text: str, metadata: dict):
        """Handle incoming transcriptions."""
        print(f"\n[TRANSCRIPTION] {text}")
        print(f"[CONFIDENCE] {metadata['confidence']:.2%}")
        
        emotional = metadata.get('emotional_context', {})
        if emotional:
            print(f"[EMOTION] Tone: {emotional.get('tone', 'neutral')}")
    
    # Create and run gateway
    gateway = AudioGateway()
    
    try:
        print("\nStarting Audio Gateway...")
        print("Speak into your microphone.")
        print("Press Ctrl+C to stop.\n")
        
        await gateway.start(transcription_callback=handle_transcription)
        
        # Keep running
        while gateway.running:
            await asyncio.sleep(1)
            
            # Print status occasionally
            if hasattr(main, 'counter'):
                main.counter += 1
            else:
                main.counter = 0
            
            if main.counter % 30 == 0:  # Every 30 seconds
                status = gateway.get_status()
                print(f"\n[STATUS] {status['server']['active_connections']} active connections")
        
    except KeyboardInterrupt:
        print("\nShutdown signal received")
    finally:
        await gateway.stop()
        print("Audio gateway stopped")


if __name__ == "__main__":
    asyncio.run(main())
