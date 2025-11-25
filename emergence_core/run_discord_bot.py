"""
Discord Bot Startup Script for Lyra

This script connects Lyra's cognitive architecture to Discord,
enabling voice and text interactions with emotion-aware responses.

Usage:
    python run_discord_bot.py

Environment Variables Required:
    DISCORD_BOT_TOKEN - Your Discord bot token
    DISCORD_GUILD_ID - (Optional) Primary guild ID
    DISCORD_VOICE_CHANNEL_ID - (Optional) Auto-join voice channel
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import discord
from discord.ext import commands

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from lyra.discord_client import LyraClient
from lyra.router import LyraRouter
from lyra.voice_processor import VoiceProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('discord_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed - using system environment variables only")

class LyraBot:
    """Main bot coordinator connecting Discord client with Lyra's cognitive systems"""
    
    def __init__(self):
        """Initialize the bot and its components"""
        # Validate required environment variables
        self.bot_token = os.getenv("DISCORD_BOT_TOKEN")
        if not self.bot_token:
            raise ValueError("DISCORD_BOT_TOKEN environment variable is required!")
        
        # Optional configuration
        self.guild_id = os.getenv("DISCORD_GUILD_ID")
        self.voice_channel_id = os.getenv("DISCORD_VOICE_CHANNEL_ID")
        self.auto_join_voice = os.getenv("AUTO_JOIN_VOICE", "false").lower() == "true"
        
        # Initialize components
        logger.info("Initializing Lyra's cognitive architecture...")
        self.client = LyraClient()
        self.router = None  # Lazy init to avoid loading models during import
        self.temp_audio_dir = Path("temp_audio")
        self.temp_audio_dir.mkdir(exist_ok=True)
        
        # Configure event handlers
        self.setup_event_handlers()
        
        logger.info("Lyra Discord Bot initialized successfully")
    
    def setup_event_handlers(self):
        """Set up Discord event handlers"""
        
        @self.client.event
        async def on_ready():
            """Called when bot successfully connects to Discord"""
            logger.info(f"Logged in as {self.client.user} (ID: {self.client.user.id})")
            logger.info(f"Connected to {len(self.client.guilds)} guilds")
            
            # Initialize router (now that bot is ready)
            if self.router is None:
                logger.info("Loading Lyra's cognitive models...")
                self.router = LyraRouter(base_dir=Path(__file__).parent)
                await self.router.initialize()
                logger.info("Lyra's cognitive models loaded successfully")
            
            # Initialize voice processor
            if self.client.voice_processor is None:
                logger.info("Initializing voice processor...")
                self.client.voice_processor = VoiceProcessor()
                logger.info("Voice processor ready")
            
            # Set initial status
            await self.client.set_availability('open', 'Ready for contemplation and conversation')
            
            # Auto-join voice channel if configured
            if self.auto_join_voice and self.voice_channel_id:
                channel = self.client.get_channel(int(self.voice_channel_id))
                if channel and isinstance(channel, discord.VoiceChannel):
                    await self.join_voice_channel(channel)
        
        @self.client.event
        async def on_message(message: discord.Message):
            """Process incoming messages"""
            # Ignore messages from bots (including self)
            if message.author.bot:
                return
            
            # Ignore empty messages
            if not message.content and not message.attachments:
                return
            
            try:
                # Show typing indicator
                async with message.channel.typing():
                    # Process message with router
                    response = await self.process_message(message)
                    
                    # Send text response
                    await self.send_response(message.channel, response)
                    
                    # If in voice channel, also speak the response
                    if message.guild and message.guild.voice_client:
                        await self.speak_in_voice(message.guild.voice_client, response)
                        
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                await message.channel.send("I encountered an error while processing your message. Please try again.")
        
        @self.client.tree.command(name="join", description="Join your current voice channel")
        async def join_voice(interaction: discord.Interaction):
            """Join the user's voice channel"""
            if not interaction.user.voice:
                await interaction.response.send_message("You need to be in a voice channel first!")
                return
            
            channel = interaction.user.voice.channel
            await interaction.response.defer()
            
            success = await self.join_voice_channel(channel)
            if success:
                await interaction.followup.send(f"Joined {channel.name}")
            else:
                await interaction.followup.send("Failed to join voice channel")
        
        @self.client.tree.command(name="leave", description="Leave the current voice channel")
        async def leave_voice(interaction: discord.Interaction):
            """Leave voice channel"""
            if not interaction.guild.voice_client:
                await interaction.response.send_message("I'm not in a voice channel!")
                return
            
            await interaction.response.defer()
            await interaction.guild.voice_client.disconnect()
            await interaction.followup.send("Left voice channel")
        
        @self.client.tree.command(name="volume", description="Set Lyra's voice volume")
        async def set_volume(interaction: discord.Interaction, volume: int):
            """Set voice output volume (0-100)"""
            if volume < 0 or volume > 100:
                await interaction.response.send_message("Volume must be between 0 and 100")
                return
            
            if self.client.voice_processor:
                self.client.voice_processor.set_volume(volume / 100.0)
                await interaction.response.send_message(f"Volume set to {volume}%")
            else:
                await interaction.response.send_message("Voice processor not initialized")
        
        @self.client.tree.command(name="status", description="Set Lyra's availability status")
        async def set_status(interaction: discord.Interaction, state: str, reason: Optional[str] = None):
            """Set availability status"""
            valid_states = ['open', 'limited', 'processing', 'resting']
            if state not in valid_states:
                await interaction.response.send_message(f"Invalid state. Use: {', '.join(valid_states)}")
                return
            
            await self.client.set_availability(state, reason)
            await interaction.response.send_message(f"Status updated to: {state}")
    
    async def process_message(self, message: discord.Message) -> str:
        """Process message through Lyra's cognitive architecture"""
        # Extract text content
        text_content = message.content
        
        # Process image attachments
        image = None
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('image/'):
                # Download image
                image_data = await attachment.read()
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_data))
                logger.info(f"Processing image attachment: {attachment.filename}")
                break
        
        # Process audio attachments
        audio_path = None
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('audio/'):
                # Download audio
                audio_path = self.temp_audio_dir / attachment.filename
                await attachment.save(audio_path)
                logger.info(f"Processing audio attachment: {attachment.filename}")
                
                # Transcribe audio
                if self.client.voice_processor:
                    result = await self.client.voice_processor.transcribe_audio(audio_path)
                    text_content = f"{text_content}\n\n[Audio transcription: {result['text']}]"
                    text_content += f"\n[Detected emotion: {result['emotion']}]"
                break
        
        # Process through router
        context = {
            "discord_user": str(message.author),
            "discord_channel": str(message.channel),
            "has_image": image is not None,
            "has_audio": audio_path is not None
        }
        
        response = await self.router.process_message(
            message=text_content,
            context=context,
            image=image
        )
        
        # Clean up temporary audio file
        if audio_path and audio_path.exists():
            audio_path.unlink()
        
        return response.content
    
    async def send_response(self, channel: discord.TextChannel, text: str):
        """Send response to Discord channel, splitting if needed"""
        # Discord has 2000 character limit
        max_length = 2000
        
        if len(text) <= max_length:
            await channel.send(text)
        else:
            # Split into chunks
            chunks = []
            current_chunk = ""
            
            for line in text.split('\n'):
                if len(current_chunk) + len(line) + 1 <= max_length:
                    current_chunk += line + '\n'
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = line + '\n'
            
            if current_chunk:
                chunks.append(current_chunk)
            
            # Send chunks
            for chunk in chunks:
                await channel.send(chunk)
    
    async def join_voice_channel(self, channel: discord.VoiceChannel) -> bool:
        """Join a voice channel"""
        try:
            if channel.guild.voice_client:
                await channel.guild.voice_client.move_to(channel)
            else:
                await channel.connect()
            
            logger.info(f"Joined voice channel: {channel.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to join voice channel: {e}")
            return False
    
    async def speak_in_voice(self, voice_client: discord.VoiceClient, text: str):
        """Speak text in voice channel using TTS"""
        try:
            # Generate speech
            audio_path = self.temp_audio_dir / "response.wav"
            self.client.voice_processor.generate_speech(
                text=text,
                output_path=audio_path,
                emotion=self.client.current_emotion or "neutral"
            )
            
            # Play audio
            if voice_client.is_connected():
                source = discord.FFmpegPCMAudio(str(audio_path))
                voice_client.play(source)
                
                # Wait for playback to finish
                while voice_client.is_playing():
                    await asyncio.sleep(0.1)
            
            # Clean up
            if audio_path.exists():
                audio_path.unlink()
                
        except Exception as e:
            logger.error(f"Error speaking in voice channel: {e}", exc_info=True)
    
    async def start(self):
        """Start the bot"""
        try:
            logger.info("Starting Lyra Discord Bot...")
            await self.client.start(self.bot_token)
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
        except Exception as e:
            logger.error(f"Bot error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Lyra Discord Bot...")
        
        # Disconnect from voice channels
        for voice_client in self.client.voice_clients:
            await voice_client.disconnect()
        
        # Close client
        await self.client.close()
        
        logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    # Create and start bot
    bot = LyraBot()
    await bot.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
