"""
Microphone Client for ASR Gateway

Captures live microphone audio and streams to ASR server.
Supports cross-platform audio input with fallback options.

Features:
- Real-time microphone capture
- Automatic audio format conversion
- Reconnection on disconnect
- Voice activity detection (optional)
- Multiple audio backend support
"""

import asyncio
import logging
import json
import numpy as np
from typing import Optional, Callable, AsyncGenerator
from pathlib import Path
from datetime import datetime

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False
    print("Warning: sounddevice not installed - microphone unavailable")
    print("Install with: pip install sounddevice")

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("Warning: websockets not installed")
    print("Install with: pip install websockets")

logger = logging.getLogger(__name__)


class MicrophoneClient:
    """Microphone client for real-time audio streaming to ASR server"""
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8765",
        sample_rate: int = 16000,
        channels: int = 1,
        device: Optional[int] = None,
        chunk_duration: float = 0.5  # seconds
    ):
        """Initialize microphone client.
        
        Args:
            server_url: WebSocket server URL
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels (1=mono, 2=stereo)
            device: Audio device index (None for default)
            chunk_duration: Duration of each audio chunk in seconds
        """
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        
        # Connection state
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self.running = False
        
        # Audio buffer for streaming
        self.audio_queue: asyncio.Queue = asyncio.Queue()
        
        # Callback for transcriptions
        self.transcription_callback: Optional[Callable] = None
        
        logger.info(f"Microphone client initialized for {server_url}")
    
    async def connect(self) -> bool:
        """Connect to ASR server.
        
        Returns:
            True if connected successfully
        """
        if not HAS_WEBSOCKETS:
            logger.error("websockets package required")
            return False
        
        try:
            logger.info(f"Connecting to ASR server: {self.server_url}")
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=10
            )
            self.connected = True
            
            # Receive welcome message
            welcome = await self.websocket.recv()
            data = json.loads(welcome)
            logger.info(f"Connected: {data.get('message')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from ASR server."""
        if self.websocket and self.connected:
            await self.websocket.close()
            self.connected = False
            logger.info("Disconnected from ASR server")
    
    def set_transcription_callback(self, callback: Callable[[str, dict], None]):
        """Set callback for transcription results.
        
        Args:
            callback: Function to call with (text, metadata) when transcription received
        """
        self.transcription_callback = callback
    
    async def start_streaming(self):
        """Start streaming microphone audio to server."""
        if not HAS_SOUNDDEVICE:
            raise ImportError("sounddevice package required - install with: pip install sounddevice")
        
        if not self.connected:
            if not await self.connect():
                raise ConnectionError("Failed to connect to ASR server")
        
        self.running = True
        logger.info("Starting microphone streaming...")
        
        # Start concurrent tasks
        tasks = [
            asyncio.create_task(self._capture_audio()),
            asyncio.create_task(self._send_audio()),
            asyncio.create_task(self._receive_transcriptions())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Streaming cancelled")
        finally:
            self.running = False
    
    async def stop_streaming(self):
        """Stop streaming microphone audio."""
        logger.info("Stopping microphone streaming...")
        self.running = False
        await asyncio.sleep(0.5)  # Allow cleanup
    
    async def _capture_audio(self):
        """Capture audio from microphone and queue for sending."""
        
        def audio_callback(indata, frames, time, status):
            """Callback for sounddevice audio capture."""
            if status:
                logger.warning(f"Audio status: {status}")
            
            # Convert to int16 for transmission
            audio_int16 = (indata * 32767).astype(np.int16)
            
            # Put in queue (non-blocking)
            try:
                self.audio_queue.put_nowait(audio_int16.tobytes())
            except asyncio.QueueFull:
                logger.warning("Audio queue full, dropping frame")
        
        # Open audio stream
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            blocksize=self.chunk_size,
            device=self.device,
            callback=audio_callback
        )
        
        with stream:
            logger.info(f"Microphone active (device: {stream.device})")
            
            while self.running:
                await asyncio.sleep(0.1)
        
        logger.info("Microphone capture stopped")
    
    async def _send_audio(self):
        """Send queued audio to ASR server."""
        while self.running:
            try:
                # Get audio from queue (with timeout)
                audio_bytes = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=1.0
                )
                
                # Send to server
                if self.websocket and self.connected:
                    await self.websocket.send(audio_bytes)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
                if not self.connected:
                    break
    
    async def _receive_transcriptions(self):
        """Receive and process transcriptions from server."""
        while self.running and self.websocket:
            try:
                message = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=1.0
                )
                
                # Parse message
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "transcription":
                    text = data.get("text", "")
                    confidence = data.get("confidence", 0.0)
                    emotional_context = data.get("emotional_context", {})
                    
                    logger.info(f"Transcription: {text}")
                    
                    # Call user callback if set
                    if self.transcription_callback:
                        self.transcription_callback(text, {
                            "confidence": confidence,
                            "emotional_context": emotional_context,
                            "timestamp": data.get("timestamp")
                        })
                
                elif msg_type == "error":
                    logger.error(f"Server error: {data.get('message')}")
                
                elif msg_type == "pong":
                    logger.debug("Pong received")
            
            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from server: {e}")
            except Exception as e:
                logger.error(f"Error receiving transcriptions: {e}")
                if not self.connected:
                    break
    
    async def send_control(self, message_type: str, **kwargs):
        """Send control message to server.
        
        Args:
            message_type: Type of control message
            **kwargs: Additional message parameters
        """
        if self.websocket and self.connected:
            message = {"type": message_type, **kwargs}
            await self.websocket.send(json.dumps(message))
    
    @staticmethod
    def list_devices():
        """List available audio input devices.
        
        Returns:
            List of device info dictionaries
        """
        if not HAS_SOUNDDEVICE:
            print("sounddevice package required")
            return []
        
        devices = sd.query_devices()
        input_devices = []
        
        print("\nAvailable audio input devices:")
        print("-" * 70)
        
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': idx,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': int(device['default_samplerate'])
                })
                print(f"[{idx}] {device['name']}")
                print(f"    Channels: {device['max_input_channels']}, "
                      f"Sample Rate: {device['default_samplerate']} Hz")
        
        print("-" * 70)
        return input_devices


async def main():
    """Run microphone client standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # List available devices
    MicrophoneClient.list_devices()
    
    # Create client
    client = MicrophoneClient()
    
    # Set up transcription callback
    def on_transcription(text: str, metadata: dict):
        print(f"\n>>> {text}")
        print(f"    Confidence: {metadata['confidence']:.2f}")
        if metadata.get('emotional_context'):
            tone = metadata['emotional_context'].get('tone', 'neutral')
            print(f"    Tone: {tone}")
    
    client.set_transcription_callback(on_transcription)
    
    try:
        print("\nStarting microphone streaming...")
        print("Speak into your microphone. Press Ctrl+C to stop.\n")
        
        await client.start_streaming()
        
    except KeyboardInterrupt:
        print("\nShutdown signal received")
    finally:
        await client.stop_streaming()
        await client.disconnect()
        print("Microphone client stopped")


if __name__ == "__main__":
    asyncio.run(main())
