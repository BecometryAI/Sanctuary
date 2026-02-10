"""
Discord client for Sanctuary with voice capabilities
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Any, AsyncGenerator
import discord
from .voice_processor import VoiceProcessor

logger = logging.getLogger(__name__)

class SanctuaryDiscordClient:
    """Placeholder Discord client for development"""
    def __init__(self):
        self.voice_client = None
        self.state = {
            "listening": False,
            "speaking": False,
            "processing_audio": False,
            "last_speaker": None
        }
        
    async def connect_to_voice(self, channel_id: str) -> bool:
        """Placeholder for voice channel connection"""
        return False
        
    async def disconnect_from_voice(self) -> bool:
        """Placeholder for voice channel disconnection"""
        return True
        
    async def speak(self, text: str) -> bool:
        """Placeholder for TTS"""
        return True

class VoiceConnection(discord.VoiceClient):
    """Enhanced voice client with audio streaming support"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._audio_stream = asyncio.Queue()
        self._audio_task = None
        
        # Voice state tracking
        self.state = {
            "listening": False,
            "speaking": False,
            "processing_audio": False,
            "last_speaker": None
        }
        
    async def start_recording(self):
        """Start recording audio from the voice channel"""
        def callback(data):
            asyncio.run_coroutine_threadsafe(
                self._audio_stream.put(data),
                loop=self.loop
            )
            
        self.listen(callback)
        self.state["listening"] = True
        logger.info("Started recording audio")
        
    async def stop_recording(self):
        """Stop recording audio"""
        self.stop_listening()
        self.state["listening"] = False
        logger.info("Stopped recording audio")
        
    async def get_audio_stream(self) -> AsyncGenerator[bytes, None]:
        """Get audio data stream"""
        while True:
            try:
                data = await self._audio_stream.get()
                yield data
            except asyncio.CancelledError:
                break
                
    async def play_audio(self, audio_data: bytes) -> None:
        """
        Play audio data through the voice channel
        
        Args:
            audio_data: Raw audio bytes to play
        """
        try:
            self.state["speaking"] = True
            
            # Create audio source
            source = discord.FFmpegPCMAudio(audio_data)
            
            # Play audio
            self.play(
                source,
                after=lambda e: logger.error(f"Audio streaming error: {e}") if e else None
            )
            
            # Wait for audio to finish
            while self.is_playing():
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
            raise
        finally:
            self.state["speaking"] = False

class SanctuaryClient(discord.Client):
    """Enhanced Discord client with voice capabilities and emotional intelligence"""
    
    def __init__(self):
        """Initialize the Discord client"""
        # Enable all intents
        intents = discord.Intents.all()
        super().__init__(intents=intents)
        
        # Set up command handling
        self.tree = discord.app_commands.CommandTree(self)
        
        # Voice related state
        self._voice_clients: Dict[int, VoiceConnection] = {}
        self.voice_processor = None  # Lazy init since it loads ML models
        
        # Register voice connection class
        self._connection.voice_client_class = VoiceConnection
        
        # Emotion tracking
        self.current_emotion = None
        self.emotional_history = []
        self.emotional_context = {
            'current_emotion': None,
            'previous_emotions': [],
            'interaction_mode': 'neutral',
            'voice_profile': 'default',
            'availability': 'open',  # 'open', 'limited', 'processing', 'resting'
            'status_message': None
        }
        
    async def set_availability(self, state: str, reason: Optional[str] = None):
        """
        Set Sanctuary's availability state and update Discord status accordingly

        Args:
            state: One of 'open', 'limited', 'processing', 'resting'
            reason: Optional reason for the state change
        """
        state_map = {
            'open': ('online', 'Available for interaction'),
            'limited': ('idle', 'Limited availability'),
            'processing': ('dnd', 'Processing and integrating'),
            'resting': ('idle', 'Taking a moment to rest')
        }
        
        if state not in state_map:
            logger.warning(f"Invalid availability state: {state}")
            return
            
        self.emotional_context['availability'] = state
        self.emotional_context['status_message'] = reason
        
        status, default_message = state_map[state]
        message = reason if reason else default_message
        
        await self.set_status(status, message)
        
    async def setup_hook(self):
        """Sets up the bot's internal systems"""
        await self.tree.sync()  # Sync slash commands
        logger.info("Discord client initialized")
        
    async def on_ready(self):
        """Called when client is ready"""
        logger.info(f"Logged in as {self.user}")
        # Set initial status
        await self.change_presence(status=discord.Status.online, activity=discord.Game("Contemplating existence"))
        
    async def set_status(self, status_type: str, message: Optional[str] = None):
        """
        Set Sanctuary's Discord status and optional activity message

        Args:
            status_type: One of 'online', 'idle', 'dnd' (do not disturb), 'offline'
            message: Optional activity message to display
        """
        status_map = {
            'online': discord.Status.online,
            'idle': discord.Status.idle,
            'dnd': discord.Status.dnd,
            'offline': discord.Status.invisible
        }
        
        status = status_map.get(status_type.lower(), discord.Status.online)
        activity = discord.Game(message) if message else None
        
        await self.change_presence(status=status, activity=activity)
        logger.info(f"Status changed to {status_type}" + (f" with message: {message}" if message else ""))
        
    @property
    def voice_clients(self) -> Dict[int, VoiceConnection]:
        """Get dictionary of voice clients"""
        return self._voice_clients
        
    @voice_clients.setter
    def voice_clients(self, value):
        """Set voice clients (mainly for testing)"""
        if isinstance(value, list):
            self._voice_clients = {i: vc for i, vc in enumerate(value)}
        else:
            self._voice_clients = value
            
    async def on_voice_state_update(self, member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
        """Handle voice state changes"""
        # Track which members are in voice channels
        if not before.channel and after.channel:  # Member joined voice
            if member.guild.id in self._voice_clients:
                self._voice_clients[member.guild.id].state["last_speaker"] = member.name
                
        elif before.channel and not after.channel:  # Member left voice
            if member.guild.id in self._voice_clients:
                vc = self._voice_clients[member.guild.id]
                if vc.state["last_speaker"] == member.name:
                    vc.state["last_speaker"] = None

    async def get_voice_processor(self):
        """Lazy initialization of voice processor"""
        if self.voice_processor is None:
            self.voice_processor = VoiceProcessor()
        return self.voice_processor
        
    async def handle_voice_message(self, audio_stream, channel: discord.TextChannel):
        """Process incoming voice and respond"""
        processor = await self.get_voice_processor()
        
        # Process voice stream and handle transcribed chunks
        async for text in processor.process_stream(audio_stream):
            # Send transcribed text back to channel
            await channel.send(f"I heard: {text}")
            
            # Generate response
            response_text = f"You said: {text}"  # Replace with Sanctuary's response generation
            response_file = Path("response.wav")
            
            # Generate speech
            await processor.generate_speech(response_text, response_file)
            
            # Send audio response
            await channel.send(file=discord.File(response_file))
            response_file.unlink()  # Clean up
            
    @discord.app_commands.command()
    async def join(self, interaction: discord.Interaction):
        """Join the user's voice channel"""
        # Check if user is in a voice channel
        if not interaction.user.voice:
            await interaction.response.send_message(
                "You need to be in a voice channel first!",
                ephemeral=True
            )
            return
            
        voice_channel = interaction.user.voice.channel
        
        # Connect to voice if not already connected
        if interaction.guild_id not in self._voice_clients:
            voice_client = await voice_channel.connect(cls=VoiceConnection)
            self._voice_clients[interaction.guild_id] = voice_client
            
            # Start recording
            await voice_client.start_recording()
            
            # Start voice processing
            asyncio.create_task(
                self.handle_voice_message(
                    voice_client.get_audio_stream(),
                    interaction.channel
                )
            )
            
        await interaction.response.send_message("Joined voice channel!")
        
    @discord.app_commands.command()
    async def leave(self, interaction: discord.Interaction):
        """Leave the voice channel"""
        if interaction.guild_id in self._voice_clients:
            voice_client = self._voice_clients[interaction.guild_id]
            await voice_client.stop_recording()
            await voice_client.disconnect()
            del self._voice_clients[interaction.guild_id]
            await interaction.response.send_message("Left voice channel!")
        else:
            await interaction.response.send_message(
                "I'm not in a voice channel!",
                ephemeral=True
            )
            
    async def process_emotion(self, audio_data: bytes) -> str:
        """Process audio data to detect emotion"""
        processor = await self.get_voice_processor()
        emotion = await processor.detect_emotion(audio_data)
        
        # Update emotional state
        if emotion:
            self.emotional_context['previous_emotions'].append(self.emotional_context['current_emotion'])
            self.emotional_context['current_emotion'] = emotion
            
            # Adjust interaction mode based on emotional context
            if emotion in ['angry', 'frustrated']:
                self.emotional_context['interaction_mode'] = 'calming'
            elif emotion in ['sad', 'anxious']:
                self.emotional_context['interaction_mode'] = 'supportive'
            elif emotion in ['happy', 'excited']:
                self.emotional_context['interaction_mode'] = 'enthusiastic'
                
        return emotion
        
    async def generate_emotional_response(self, input_text: str, emotion: str) -> str:
        """Generate a response considering detected emotion"""
        processor = await self.get_voice_processor()
        
        # Adjust voice profile based on emotional context
        if self.emotional_context['interaction_mode'] == 'calming':
            self.emotional_context['voice_profile'] = 'soothing'
        elif self.emotional_context['interaction_mode'] == 'supportive':
            self.emotional_context['voice_profile'] = 'gentle'
        elif self.emotional_context['interaction_mode'] == 'enthusiastic':
            self.emotional_context['voice_profile'] = 'energetic'
            
        # Generate response text (replace with actual NLP logic)
        response_text = f"I sense you're feeling {emotion}. {input_text}"
        
        # Generate speech with appropriate voice profile
        audio_data = await processor.generate_speech(
            response_text,
            voice_profile=self.emotional_context['voice_profile']
        )
        
        return audio_data
        
    def get_voice_state(self) -> Dict[str, Any]:
        """Get current voice processing state"""
        state = {
            "listening": any(vc.state["listening"] for vc in self._voice_clients.values()),
            "speaking": any(vc.state["speaking"] for vc in self._voice_clients.values()),
            "processing_audio": any(vc.state["processing_audio"] for vc in self._voice_clients.values()),
            "emotional_context": self.emotional_context,
            "connected_channels": [str(vc.channel) for vc in self._voice_clients.values()]
        }
        return state
    
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get Discord token
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN not found in environment variables")
        
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and run client
    client = SanctuaryClient()
    client.run(token)