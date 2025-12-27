"""
Quick test to verify Discord bot connection
"""
import os
import discord
from discord.ext import commands

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get token
token = os.getenv("DISCORD_BOT_TOKEN")
if not token:
    print("❌ DISCORD_BOT_TOKEN not found in environment")
    exit(1)

print(f"✅ Token found: {token[:20]}...")

# Create minimal bot
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    print(f'\n✅ Bot connected successfully!')
    print(f'   Logged in as: {bot.user.name} (ID: {bot.user.id})')
    print(f'   Connected to {len(bot.guilds)} server(s)')
    for guild in bot.guilds:
        print(f'   - {guild.name} (ID: {guild.id})')
    print(f'\n🎉 Discord connection test PASSED!')
    print(f'\nTo stop the bot, press Ctrl+C')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    
    if bot.user.mentioned_in(message):
        await message.channel.send(f"👋 Hello! I'm a test bot. The connection works!")
    
    await bot.process_commands(message)

@bot.command()
async def ping(ctx):
    """Simple ping command"""
    await ctx.send(f'🏓 Pong! Latency: {round(bot.latency * 1000)}ms')

print("🔄 Connecting to Discord...")
try:
    bot.run(token)
except discord.LoginFailure:
    print("❌ Invalid token! Please check your DISCORD_BOT_TOKEN")
except Exception as e:
    print(f"❌ Error: {e}")
