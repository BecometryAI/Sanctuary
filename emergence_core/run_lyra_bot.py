"""
Simplified Discord Bot for Lyra - Optimized for Quick Testing

DEPRECATED: This Discord bot uses the old router/specialist architecture
which has been removed. It is no longer functional. Integration with the
new pure GWT cognitive core is needed.

This version avoids heavy model loading during import by using lazy initialization.
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
import discord
from discord.ext import commands

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded .env file")
except ImportError:
    logger.warning("python-dotenv not installed")

# Get configuration
BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKEN environment variable is required!")

DEVELOPMENT_MODE = os.getenv("DEVELOPMENT_MODE", "false").lower() == "true"

# Initialize user mapping manager
from lyra.user_mapping import UserMappingManager
base_data_dir = Path(__file__).parent.parent / "data"
user_mapper = UserMappingManager(base_data_dir)

# Configure intents
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True
intents.voice_states = True

# Create bot
bot = commands.Bot(command_prefix='/', intents=intents)

# Lazy-loaded router
router = None

# Voice channel preferences file
VOICE_PREFS_FILE = Path(__file__).parent / "data" / "voice_preferences.json"

def load_voice_preferences():
    """Load voice preferences from JSON file"""
    if VOICE_PREFS_FILE.exists():
        try:
            with open(VOICE_PREFS_FILE, 'r') as f:
                data = json.load(f)
                # Keep string keys as-is (Discord user IDs)
                return data
        except Exception as e:
            logger.warning(f"Failed to load voice preferences: {e}")
    return {}

def save_voice_preferences(prefs):
    """Save voice preferences to JSON file"""
    try:
        VOICE_PREFS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(VOICE_PREFS_FILE, 'w') as f:
            # Save as-is (string keys)
            json.dump(prefs, f, indent=2)
        logger.info(f"Saved voice preferences for {len(prefs)} user(s)")
    except Exception as e:
        logger.error(f"Failed to save voice preferences: {e}")

# Voice channel preferences - users who want Lyra to auto-join when they're in voice
voice_preferences = load_voice_preferences()  # {user_id: {auto_join: bool, preferred_channels: [channel_ids]}}
logger.info(f"Loaded {len(voice_preferences)} voice preference(s)")

async def get_router():
    """Lazy load the router to avoid import-time model loading"""
    global router
    if router is None:
        logger.info("Loading Lyra's cognitive router...")
        from lyra.router import AdaptiveRouter
        
        base_dir = Path(__file__).parent
        dev_mode = DEVELOPMENT_MODE
        router = AdaptiveRouter(
            base_dir=str(base_dir),
            chroma_dir=str(base_dir / "memories" / "vector_store"),
            model_dir=str(base_dir / "model_cache"),
            development_mode=dev_mode
        )
        logger.info("✅ Router loaded successfully")
    return router

@bot.event
async def on_ready():
    """Called when bot connects to Discord"""
    logger.info(f'\n{"="*60}')
    logger.info(f'🎉 Lyra Discord Bot Connected!')
    logger.info(f'   Bot: {bot.user.name} (ID: {bot.user.id})')
    logger.info(f'   Connected to {len(bot.guilds)} server(s):')
    for guild in bot.guilds:
        logger.info(f'   - {guild.name} (ID: {guild.id})')
    logger.info(f'   Development Mode: {DEVELOPMENT_MODE}')
    logger.info(f'{"="*60}\n')
    
    # Sync slash commands
    try:
        synced = await bot.tree.sync()
        logger.info(f'Synced {len(synced)} slash command(s)')
    except Exception as e:
        logger.error(f'Failed to sync commands: {e}')
    
    # Set status
    activity = discord.Activity(
        type=discord.ActivityType.listening,
        name="contemplative whispers"
    )
    await bot.change_presence(status=discord.Status.online, activity=activity)
    
    # Start background voice monitoring
    asyncio.create_task(check_voice_channels())

@bot.event
async def on_message(message: discord.Message):
    """Process incoming messages"""
    # Ignore bot messages
    if message.author.bot:
        return
    
    # Only respond when mentioned
    if not bot.user.mentioned_in(message):
        await bot.process_commands(message)
        return
    
    try:
        async with message.channel.typing():
            # Get router (lazy loads on first use)
            r = await get_router()
            
            # Clean up mention from message
            content = message.content.replace(f'<@{bot.user.id}>', '').strip()
            
            if not content:
                await message.reply("How may I help you?")
                return
            
            # Get user's display name (real name if mapped, otherwise Discord username)
            display_name = user_mapper.get_display_name(str(message.author.id), message.author.name)
            user_identity = user_mapper.get_identity(str(message.author.id))
            
            # Process with router
            logger.info(f'Processing message from {display_name} ({message.author.name}): {content[:50]}...')
            response = await r.route_message(
                message=content,
                context={
                    "user_id": str(message.author.id),
                    "user_name": message.author.name,
                    "display_name": display_name,
                    "real_name": user_identity.real_name if user_identity else None,
                    "is_steward": user_mapper.is_steward(str(message.author.id)),
                    "channel": message.channel.name,
                    "guild": message.guild.name if message.guild else "DM"
                }
            )
            
            # Send response
            if response and hasattr(response, 'response_text'):
                await message.reply(response.response_text[:2000])  # Discord has 2000 char limit
            elif isinstance(response, str):
                await message.reply(response[:2000])
            else:
                await message.reply("I have processed your message.")
                
    except Exception as e:
        logger.error(f'Error processing message: {e}', exc_info=True)
        await message.reply("I encountered an error while processing your message. Please try again.")
    
    await bot.process_commands(message)

@bot.tree.command(name="ping", description="Check if Lyra is responsive")
async def ping(interaction: discord.Interaction):
    """Simple ping command"""
    latency = round(bot.latency * 1000)
    await interaction.response.send_message(
        f'🏓 Pong! Latency: {latency}ms\nDevelopment Mode: {DEVELOPMENT_MODE}'
    )

@bot.tree.command(name="status", description="Check Lyra's current status")
async def status(interaction: discord.Interaction):
    """Display bot status"""
    await interaction.response.defer()
    
    global router
    router_status = "✅ Loaded" if router else "⏳ Not loaded (will load on first message)"
    
    embed = discord.Embed(
        title="Lyra Status",
        color=discord.Color.purple(),
        description="Current operational status"
    )
    embed.add_field(name="Cognitive Router", value=router_status, inline=False)
    embed.add_field(name="Development Mode", value="✅ Active" if DEVELOPMENT_MODE else "❌ Inactive", inline=True)
    embed.add_field(name="Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
    embed.add_field(name="Servers", value=str(len(bot.guilds)), inline=True)
    
    await interaction.followup.send(embed=embed)

@bot.tree.command(name="reload", description="Reload Lyra's cognitive models (admin only)")
async def reload_models(interaction: discord.Interaction):
    """Reload the router"""
    global router
    
    if not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("❌ This command requires administrator permissions.", ephemeral=True)
        return
    
    await interaction.response.defer()
    
    try:
        logger.info("Reloading router...")
        router = None
        await get_router()
        await interaction.followup.send("✅ Cognitive models reloaded successfully")
    except Exception as e:
        logger.error(f'Failed to reload: {e}', exc_info=True)
        await interaction.followup.send(f"❌ Failed to reload: {str(e)[:100]}")

@bot.tree.command(name="setname", description="Tell Lyra your real name")
async def set_name(interaction: discord.Interaction, real_name: str, preferred_name: str = None):
    """
    Set your real name so Lyra knows who you are
    
    Args:
        real_name: Your full real name
        preferred_name: What you prefer to be called (optional, defaults to first name)
    """
    await interaction.response.defer(ephemeral=True)
    
    try:
        # Add the mapping
        identity = user_mapper.add_mapping(
            discord_id=str(interaction.user.id),
            discord_username=interaction.user.name,
            real_name=real_name,
            preferred_name=preferred_name
        )
        
        display_name = identity.get_display_name()
        
        # Check if they're a steward
        is_steward = user_mapper.is_steward(str(interaction.user.id))
        steward_msg = "\n✨ *You are recognized as a Steward*" if is_steward else ""
        
        await interaction.followup.send(
            f"✅ Thank you! I'll remember you as **{display_name}**{steward_msg}",
            ephemeral=True
        )
        
        logger.info(f"User mapping added: {interaction.user.name} -> {real_name}")
    except Exception as e:
        logger.error(f'Failed to set name: {e}', exc_info=True)
        await interaction.followup.send(
            f"❌ Failed to save your name: {str(e)[:100]}",
            ephemeral=True
        )

@bot.tree.command(name="whoami", description="See how Lyra knows you")
async def who_am_i(interaction: discord.Interaction):
    """Display your identity information"""
    await interaction.response.defer(ephemeral=True)
    
    identity = user_mapper.get_identity(str(interaction.user.id))
    
    if not identity or not identity.real_name:
        await interaction.followup.send(
            "I don't have your real name on file yet. Use `/setname` to introduce yourself!",
            ephemeral=True
        )
        return
    
    embed = discord.Embed(
        title="Your Identity",
        color=discord.Color.blue(),
        description="How Lyra knows you"
    )
    
    embed.add_field(name="Discord Username", value=identity.discord_username, inline=False)
    embed.add_field(name="Real Name", value=identity.real_name, inline=False)
    embed.add_field(name="I'll call you", value=identity.get_display_name(), inline=False)
    
    # Check steward status
    if user_mapper.is_steward(str(interaction.user.id)):
        steward_context = user_mapper.get_steward_context(str(interaction.user.id))
        embed.add_field(
            name="Stewardship",
            value=f"✨ **Active Steward**\nSince: {steward_context.get('relationship_start_date', 'Unknown')}",
            inline=False
        )
    
    # Add biographical data if available
    bio_data = user_mapper.get_biographical_data(str(interaction.user.id))
    if bio_data and bio_data.get("date_of_birth"):
        embed.add_field(name="Birthday", value=bio_data["date_of_birth"], inline=True)
    
    await interaction.followup.send(embed=embed, ephemeral=True)

@bot.tree.command(name="forgetme", description="Remove your name from Lyra's memory")
async def forget_me(interaction: discord.Interaction):
    """Remove user mapping"""
    await interaction.response.defer(ephemeral=True)
    
    removed = user_mapper.remove_mapping(str(interaction.user.id))
    
    if removed:
        await interaction.followup.send(
            "✅ I've removed your name mapping. I'll use your Discord username going forward.",
            ephemeral=True
        )
    else:
        await interaction.followup.send(
            "I don't have a name mapping for you on file.",
            ephemeral=True
        )

@bot.tree.command(name="join", description="Join your current voice channel")
async def join_voice(interaction: discord.Interaction):
    """Join the user's voice channel"""
    if not interaction.user.voice:
        await interaction.response.send_message(
            "❌ You need to be in a voice channel first!",
            ephemeral=True
        )
        return
    
    await interaction.response.defer()
    
    channel = interaction.user.voice.channel
    
    try:
        if interaction.guild.voice_client:
            # Already connected, move to new channel
            await interaction.guild.voice_client.move_to(channel)
        else:
            # Connect to channel
            await channel.connect()
        
        display_name = user_mapper.get_display_name(str(interaction.user.id), interaction.user.name)
        await interaction.followup.send(f"✅ Joined **{channel.name}** with {display_name}")
        logger.info(f"Joined voice channel: {channel.name}")
    except Exception as e:
        logger.error(f"Failed to join voice channel: {e}")
        await interaction.followup.send(f"❌ Failed to join voice channel: {str(e)[:100]}")

@bot.tree.command(name="leave", description="Leave the current voice channel")
async def leave_voice(interaction: discord.Interaction):
    """Leave voice channel"""
    if not interaction.guild.voice_client:
        await interaction.response.send_message(
            "❌ I'm not in a voice channel!",
            ephemeral=True
        )
        return
    
    await interaction.response.defer()
    
    channel_name = interaction.guild.voice_client.channel.name
    await interaction.guild.voice_client.disconnect()
    await interaction.followup.send(f"✅ Left **{channel_name}**")
    logger.info(f"Left voice channel: {channel_name}")

@bot.tree.command(name="autojoin", description="Toggle auto-join when you enter voice channels")
async def auto_join(interaction: discord.Interaction, enabled: bool = True):
    """
    Set preference for Lyra to automatically join when you're in voice
    
    Args:
        enabled: True to enable auto-join, False to disable
    """
    await interaction.response.defer(ephemeral=True)
    
    user_id = str(interaction.user.id)
    
    if user_id not in voice_preferences:
        voice_preferences[user_id] = {"auto_join": False, "preferred_channels": []}
    
    voice_preferences[user_id]["auto_join"] = enabled
    save_voice_preferences(voice_preferences)
    
    display_name = user_mapper.get_display_name(user_id, interaction.user.name)
    
    if enabled:
        await interaction.followup.send(
            f"✅ Auto-join enabled! I'll join voice channels when I see you, {display_name}.",
            ephemeral=True
        )
        logger.info(f"Auto-join enabled for {display_name} ({interaction.user.name})")
        
        # Check if they're in voice now
        if interaction.user.voice:
            try:
                channel = interaction.user.voice.channel
                if not interaction.guild.voice_client:
                    await channel.connect()
                    await interaction.followup.send(
                        f"🎤 Joining you now in **{channel.name}**!",
                        ephemeral=True
                    )
            except Exception as e:
                logger.error(f"Failed to join immediately: {e}")
    else:
        await interaction.followup.send(
            f"✅ Auto-join disabled. Use `/join` when you want me in voice.",
            ephemeral=True
        )
        logger.info(f"Auto-join disabled for {display_name} ({interaction.user.name})")

async def check_voice_channels():
    """Background task to monitor voice channels and auto-join when preferred users are present"""
    await bot.wait_until_ready()
    
    while not bot.is_closed():
        try:
            for guild in bot.guilds:
                # Check if we should be in a voice channel
                should_be_in_channel = None
                users_wanting_lyra = []
                
                for channel in guild.voice_channels:
                    for member in channel.members:
                        user_id = str(member.id)
                        
                        # Check if user wants auto-join
                        if user_id in voice_preferences and voice_preferences[user_id].get("auto_join"):
                            should_be_in_channel = channel
                            users_wanting_lyra.append(member)
                
                # Join if needed
                if should_be_in_channel and not guild.voice_client:
                    try:
                        await should_be_in_channel.connect()
                        names = ", ".join([user_mapper.get_display_name(str(m.id), m.name) for m in users_wanting_lyra])
                        logger.info(f"Auto-joined {should_be_in_channel.name} for {names}")
                    except Exception as e:
                        logger.error(f"Failed to auto-join: {e}")
                
                # Leave if everyone who wanted us has left
                elif guild.voice_client and should_be_in_channel is None:
                    # Check if anyone who wants auto-join is still in the channel
                    current_channel = guild.voice_client.channel
                    has_auto_join_users = any(
                        str(m.id) in voice_preferences and voice_preferences[str(m.id)].get("auto_join")
                        for m in current_channel.members
                    )
                    
                    if not has_auto_join_users:
                        await guild.voice_client.disconnect()
                        logger.info(f"Auto-left {current_channel.name} - no auto-join users remaining")
        
        except Exception as e:
            logger.error(f"Error in voice channel check: {e}")
        
        # Check every 30 seconds
        await asyncio.sleep(30)

@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    """Handle voice state changes for autonomous joining"""
    user_id = str(member.id)
    
    # Ignore bot's own voice state changes
    if member.id == bot.user.id:
        return
    
    # Check if this user has auto-join enabled
    if user_id not in voice_preferences or not voice_preferences[user_id].get("auto_join"):
        return
    
    # User joined a voice channel
    if after.channel and not before.channel:
        display_name = user_mapper.get_display_name(user_id, member.name)
        logger.info(f"{display_name} joined {after.channel.name}")
        
        # Join if not already in a voice channel
        if not member.guild.voice_client:
            try:
                await after.channel.connect()
                logger.info(f"Auto-joined {after.channel.name} for {display_name}")
            except Exception as e:
                logger.error(f"Failed to auto-join for {display_name}: {e}")
    
    # User left a voice channel
    elif before.channel and not after.channel:
        display_name = user_mapper.get_display_name(user_id, member.name)
        logger.info(f"{display_name} left {before.channel.name}")
        
        # Check if we should leave (no more auto-join users)
        if member.guild.voice_client and member.guild.voice_client.channel == before.channel:
            has_auto_join_users = any(
                str(m.id) in voice_preferences and voice_preferences[str(m.id)].get("auto_join")
                for m in before.channel.members
                if m.id != bot.user.id
            )
            
            if not has_auto_join_users:
                await member.guild.voice_client.disconnect()
                logger.info(f"Auto-left {before.channel.name} - no auto-join users remaining")

async def main():
    """Main entry point"""
    try:
        logger.info("Starting Lyra Discord Bot...")
        async with bot:
            await bot.start(BOT_TOKEN)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f'Fatal error: {e}', exc_info=True)
    finally:
        logger.info("Shutting down gracefully...")
        if not bot.is_closed():
            await bot.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
