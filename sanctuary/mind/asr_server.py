"""
Real-time Audio Streaming Server (ASR Gateway)

Provides WebSocket-based real-time audio streaming for Sanctuary's hearing.
Integrates with WhisperProcessor for live transcription.

Architecture:
- WebSocket server accepts audio streams
- Buffers and processes audio in chunks
- Returns transcriptions with emotional context
- Supports multiple concurrent connections
- Graceful reconnection handling
"""

import asyncio
import logging
import json
from typing import Optional, Dict, Any, Set
from pathlib import Path
import numpy as np
from datetime import datetime

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("Warning: websockets not installed - ASR server unavailable")
    print("Install with: pip install websockets")

from .speech_processor import WhisperProcessor

logger = logging.getLogger(__name__)


class ASRServer:
    """Real-time Audio Streaming Recognition Server"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        sample_rate: int = 16000,
        language: str = "en"
    ):
        """Initialize ASR server.
        
        Args:
            host: Server host address
            port: Server port
            sample_rate: Expected audio sample rate (Hz)
            language: Expected language code
        """
        self.host = host
        self.port = port
        self.sample_rate = sample_rate
        self.language = language
        
        # Initialize Whisper processor
        self.whisper = WhisperProcessor()
        
        # Track active connections
        self.connections: Set[WebSocketServerProtocol] = set()
        
        # Server state
        self.running = False
        self.server = None
        
        logger.info(f"ASR Server initialized on {host}:{port}")
    
    async def start(self):
        """Start the WebSocket server."""
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets package required - install with: pip install websockets")
        
        logger.info(f"Starting ASR server on ws://{self.host}:{self.port}")
        
        self.running = True
        self.server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info("ASR server started successfully")
    
    async def stop(self):
        """Stop the WebSocket server."""
        logger.info("Stopping ASR server...")
        self.running = False
        
        # Close all connections
        if self.connections:
            await asyncio.gather(
                *[conn.close() for conn in self.connections],
                return_exceptions=True
            )
        
        # Stop server
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("ASR server stopped")
    
    async def _handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        # Register connection
        self.connections.add(websocket)
        client_addr = websocket.remote_address
        logger.info(f"New connection from {client_addr}")
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connected",
                "message": "ASR server ready",
                "sample_rate": self.sample_rate,
                "language": self.language
            }))
            
            # Process audio stream
            await self._process_audio_stream(websocket)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed: {client_addr}")
        except Exception as e:
            logger.error(f"Error handling connection from {client_addr}: {e}")
            await self._send_error(websocket, str(e))
        finally:
            # Unregister connection
            self.connections.discard(websocket)
            logger.info(f"Connection cleanup: {client_addr}")
    
    async def _process_audio_stream(self, websocket: WebSocketServerProtocol):
        """Process incoming audio stream from client.
        
        Args:
            websocket: WebSocket connection
        """
        audio_buffer = np.array([], dtype=np.float32)
        chunk_size = self.sample_rate * 2  # 2 seconds of audio
        
        async for message in websocket:
            try:
                # Handle different message types
                if isinstance(message, bytes):
                    # Binary audio data
                    audio_chunk = np.frombuffer(message, dtype=np.int16)
                    audio_chunk = audio_chunk.astype(np.float32) / 32768.0  # Normalize
                    audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                    
                    # Process when buffer is large enough
                    if len(audio_buffer) >= chunk_size:
                        await self._transcribe_and_send(websocket, audio_buffer[:chunk_size])
                        # Keep overlap for context
                        audio_buffer = audio_buffer[chunk_size // 2:]
                
                elif isinstance(message, str):
                    # JSON control message
                    data = json.loads(message)
                    await self._handle_control_message(websocket, data)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await self._send_error(websocket, f"Processing error: {str(e)}")
    
    async def _transcribe_and_send(self, websocket: WebSocketServerProtocol, audio_data: np.ndarray):
        """Transcribe audio and send result to client.
        
        Args:
            websocket: WebSocket connection
            audio_data: Audio data to transcribe
        """
        try:
            # Transcribe with emotional context
            result = await self.whisper._transcribe_with_context(
                audio_data,
                self.language
            )
            
            if result and result["text"].strip():
                # Send transcription result
                response = {
                    "type": "transcription",
                    "text": result["text"],
                    "confidence": result["confidence"],
                    "emotional_context": result.get("emotional_context", {}),
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send(json.dumps(response))
                logger.debug(f"Sent transcription: {result['text'][:50]}...")
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            await self._send_error(websocket, f"Transcription failed: {str(e)}")
    
    async def _handle_control_message(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle control messages from client.
        
        Args:
            websocket: WebSocket connection
            data: Control message data
        """
        msg_type = data.get("type")
        
        if msg_type == "config":
            # Update configuration
            if "language" in data:
                self.language = data["language"]
                logger.info(f"Language changed to: {self.language}")
            
            await websocket.send(json.dumps({
                "type": "config_updated",
                "language": self.language,
                "sample_rate": self.sample_rate
            }))
        
        elif msg_type == "ping":
            # Respond to ping
            await websocket.send(json.dumps({
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }))
        
        elif msg_type == "flush":
            # Client requests to flush buffer
            await websocket.send(json.dumps({
                "type": "flushed",
                "timestamp": datetime.now().isoformat()
            }))
        
        else:
            logger.warning(f"Unknown control message type: {msg_type}")
    
    async def _send_error(self, websocket: WebSocketServerProtocol, error_msg: str):
        """Send error message to client.
        
        Args:
            websocket: WebSocket connection
            error_msg: Error message
        """
        try:
            await websocket.send(json.dumps({
                "type": "error",
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            }))
        except Exception as e:
            logger.error(f"Failed to send error message: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics.
        
        Returns:
            Dictionary with server stats
        """
        return {
            "running": self.running,
            "active_connections": len(self.connections),
            "host": self.host,
            "port": self.port,
            "language": self.language,
            "sample_rate": self.sample_rate
        }


async def main():
    """Run ASR server standalone."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    server = ASRServer(host="0.0.0.0", port=8765)
    
    try:
        await server.start()
        logger.info("ASR server running. Press Ctrl+C to stop.")
        
        # Keep server running
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    finally:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
