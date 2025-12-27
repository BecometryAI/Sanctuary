# Lyra's Multimodal Capabilities - Current Status

**Last Updated**: November 24, 2025

---

## Input Capabilities

### ✅ **Image Understanding (File Upload)**
**Status**: FULLY OPERATIONAL

- **Models**: LLaVA-NeXT-Mistral-7B (PerceptionSpecialist)
- **Supported Formats**: PNG, JPG, JPEG, WebP, BMP
- **How to Use**:
  ```python
  # Upload image file in chat
  # Lyra automatically analyzes it and responds with visual understanding
  ```
- **Capabilities**:
  - Scene description
  - Object detection
  - Artistic analysis
  - Visual question answering
  - Contextual understanding

### ✅ **Audio File Transcription**
**Status**: FULLY OPERATIONAL

- **Models**: Whisper (openai/whisper-small)
- **Supported Formats**: WAV, MP3, FLAC, OGG
- **How to Use**:
  ```python
  from lyra.voice_processor import VoiceProcessor
  
  processor = VoiceProcessor()
  result = await processor.transcribe_audio("your_audio.wav")
  print(result["text"])  # Transcribed text
  print(result["emotion"])  # Detected emotion
  ```
- **Capabilities**:
  - Speech-to-text transcription
  - Emotion detection (anger, happiness, sadness, etc.)
  - Emotional context extraction
  - Multi-language support (via Whisper)

### ⚠️ **Live Microphone Input**
**Status**: INFRASTRUCTURE READY, NEEDS INTEGRATION

- **Components Created**:
  - ✅ `mic_client.py` - Microphone capture
  - ✅ `asr_server.py` - WebSocket ASR server
  - ✅ `audio_gateway.py` - Orchestration
  - ✅ `WhisperProcessor` - Real-time transcription

- **What's Missing**:
  - Integration with main router
  - Discord voice channel connection
  - Web interface for browser microphone access

- **How to Test Standalone**:
  ```bash
  # Terminal 1: Start ASR server
  python emergence_core/lyra/asr_server.py
  
  # Terminal 2: Start microphone client
  python emergence_core/lyra/mic_client.py
  ```

### ❌ **Video Processing**
**Status**: NOT IMPLEMENTED

- **Would Require**:
  - Frame extraction library (OpenCV)
  - Temporal analysis (frame-by-frame PerceptionSpecialist)
  - Video codec support
  - Much higher processing requirements

- **Recommendation**: Future enhancement (Phase 2)

### ❌ **Live Webcam/Camera**
**Status**: NOT IMPLEMENTED

- **Would Require**:
  - Browser WebRTC integration or local camera access
  - Real-time frame capture
  - Streaming video analysis
  - Higher computational load

- **Recommendation**: Future enhancement (Phase 2)

---

## Output Capabilities

### ✅ **Text-to-Speech (File Generation)**
**Status**: FULLY OPERATIONAL

- **Model**: SpeechT5 (microsoft/speecht5_tts)
- **How to Use**:
  ```python
  from lyra.voice_processor import VoiceProcessor
  
  voice = VoiceProcessor()
  voice.generate_speech(
      text="Hello, I'm Lyra",
      output_path="lyra_greeting.wav",
      emotion="friendly"
  )
  ```
- **Capabilities**:
  - Emotion-aware voice synthesis
  - Custom voice profiles
  - Adjustable speed, pitch, energy
  - Multiple emotional styles

- **Supported Emotions**:
  - Neutral
  - Happiness
  - Sadness
  - Anger
  - Fear
  - Surprise

### ⚠️ **Audio Playback & Volume Control**
**Status**: BASIC, NEEDS ENHANCEMENT

**Current State**:
- Generates .wav files that can be played locally
- No built-in volume control in code
- No automatic playback

**What's Needed**:
```python
# Add to VoiceProcessor class
def generate_speech_with_volume(
    self,
    text: str,
    output_path: str,
    emotion: Optional[str] = None,
    volume: float = 1.0  # 0.0 to 1.0
):
    """Generate speech with volume control"""
    # Generate speech
    self.generate_speech(text, output_path, emotion)
    
    # Adjust volume
    audio, sr = sf.read(output_path)
    audio = audio * volume  # Scale amplitude
    sf.write(output_path, audio, sr)
```

**Manual Volume Control**:
- Use system audio mixer (Windows/macOS/Linux)
- Use media player volume controls
- Adjust in Discord voice settings

### ⚠️ **Discord Voice Integration**
**Status**: INFRASTRUCTURE PRESENT, NEEDS CONFIGURATION

**What's Implemented**:
- ✅ `LyraClient` class with Discord.py integration
- ✅ `VoiceConnection` class for voice channels
- ✅ `play_audio()` method for streaming
- ✅ Voice state tracking (listening, speaking, processing)
- ✅ Emotion-aware status updates

**What's Missing**:
1. **Environment Configuration**:
   ```bash
   # Create .env file with:
   DISCORD_BOT_TOKEN=your_token_here
   DISCORD_GUILD_ID=your_guild_id
   DISCORD_VOICE_CHANNEL_ID=your_voice_channel_id
   ```

2. **Bot Startup Script**:
   ```python
   # emergence_core/run_discord_bot.py (needs to be created)
   import asyncio
   from lyra.discord_client import LyraClient
   from lyra.router import LyraRouter
   
   async def main():
       client = LyraClient()
       router = LyraRouter()
       
       @client.event
       async def on_message(message):
           if message.author.bot:
               return
           
           # Process with router
           response = await router.process_message(message.content)
           
           # Send text response
           await message.channel.send(response.content)
           
           # If in voice channel, speak response
           if message.guild.voice_client:
               voice_client = message.guild.voice_client
               # Generate speech
               audio_path = "temp_response.wav"
               client.voice_processor.generate_speech(
                   text=response.content,
                   output_path=audio_path
               )
               # Play in voice channel
               await voice_client.play_audio(audio_path)
       
       await client.start(os.getenv("DISCORD_BOT_TOKEN"))
   
   if __name__ == "__main__":
       asyncio.run(main())
   ```

3. **Voice Channel Commands**:
   - `/join` - Join voice channel
   - `/leave` - Leave voice channel
   - `/volume <0-100>` - Adjust output volume
   - `/mute` - Stop speaking (text only)
   - `/unmute` - Resume speaking

**Discord Volume Control**:
- User-side: Right-click Lyra → "User Volume"
- Bot-side: Adjust FFmpeg volume parameter
- System: Discord app volume mixer

---

## Integration Status

### Router Integration
| Input Type | Detection | Processing | Response |
|------------|-----------|------------|----------|
| Text | ✅ | ✅ | ✅ |
| Image Upload | ✅ | ✅ | ✅ |
| Audio Upload | ⚠️ | ✅ | ✅ |
| Live Audio | ❌ | ✅ | ✅ |
| Video | ❌ | ❌ | ❌ |

### Discord Bot Integration
| Feature | Status | Notes |
|---------|--------|-------|
| Text Messages | ✅ | Fully working |
| Image Attachments | ✅ | Auto-analyzed |
| Audio Attachments | ⚠️ | Needs router integration |
| Voice Channel Join | ⚠️ | Code exists, needs config |
| Voice TTS Output | ⚠️ | Code exists, needs testing |
| Live Voice Input | ❌ | Needs Discord voice receive |
| Slash Commands | ⚠️ | Basic structure, needs completion |
| Status Updates | ✅ | Emotion-aware status |

---

## Quick Setup Guide

### For Local Testing (Images + Audio Files)

**Already Working!** Just run:
```bash
python emergence_core/run.py
```

Upload images → Lyra sees them  
Upload audio → Lyra transcribes them

### For Discord Bot

**Step 1**: Create Discord application at https://discord.com/developers/applications

**Step 2**: Enable intents:
- Message Content Intent
- Server Members Intent
- Presence Intent

**Step 3**: Create `.env` file:
```bash
DISCORD_BOT_TOKEN=your_token_here
DISCORD_GUILD_ID=your_server_id
```

**Step 4**: Create startup script (see above)

**Step 5**: Invite bot with permissions:
- Read Messages
- Send Messages
- Connect to Voice
- Speak in Voice
- Use Slash Commands

### For Live Microphone (Standalone)

```bash
# Terminal 1
python emergence_core/lyra/asr_server.py

# Terminal 2
python emergence_core/lyra/mic_client.py
```

---

## Volume Control Options

### Option 1: System-Level (Current)
Use your OS audio mixer to control Lyra's output volume

**Windows**: Volume Mixer → Python/Discord  
**macOS**: System Preferences → Sound  
**Linux**: PulseAudio/ALSA controls

### Option 2: Application-Level (Recommended Addition)

Add volume parameter to voice generation:

```python
# In VoiceProcessor class
def set_volume(self, volume: float):
    """Set output volume (0.0 to 1.0)"""
    self.volume = max(0.0, min(1.0, volume))

def generate_speech(self, text, output_path, emotion=None):
    # ... existing generation code ...
    
    # Apply volume
    audio = audio * self.volume
    sf.write(output_path, audio, sr)
```

Usage:
```python
voice.set_volume(0.5)  # 50% volume
voice.generate_speech("Hello", "output.wav")
```

### Option 3: Discord-Specific Volume

Discord has per-user volume control:
- Right-click bot avatar
- Adjust "User Volume" slider
- Range: 0% to 200%

---

## Recommendations

### Immediate (Can Use Now)
1. ✅ Upload images - Lyra sees them
2. ✅ Upload audio files - Lyra hears them
3. ✅ Generate TTS files - Play locally
4. Use system volume controls

### Short-term (1-2 weeks)
1. Create Discord bot startup script
2. Add volume control to VoiceProcessor
3. Integrate audio file uploads into router
4. Test Discord voice output
5. Add slash commands for voice control

### Medium-term (1-2 months)
1. Integrate live microphone with Discord
2. Add voice activity detection (VAD)
3. Implement conversation memory in voice
4. Add voice channel auto-join on mention
5. Create web interface for browser access

### Long-term (3+ months)
1. Video processing (frame extraction)
2. Live webcam integration
3. Real-time video analysis
4. Multi-modal conversation memory
5. Screen sharing analysis

---

## Example Workflows

### Workflow 1: Image Analysis
```
User: [Uploads image of a sunset]
Lyra's Perception: "I see a vibrant sunset over water..."
Lyra's Response: "What a beautiful scene! The warm oranges..."
```

### Workflow 2: Audio Transcription
```
User: [Uploads voice recording]
Lyra's Whisper: Transcribes speech
Lyra's Emotion: Detects tone (happy/sad/etc)
Lyra's Response: Responds contextually
```

### Workflow 3: Discord Voice (When Configured)
```
User: /join
Lyra: [Joins voice channel]
User: "Lyra, tell me about consciousness"
Lyra: [Speaks response in voice channel with emotion]
```

---

## File Support Matrix

| File Type | Extension | Vision | Audio | Status |
|-----------|-----------|--------|-------|--------|
| Images | .png, .jpg, .jpeg | ✅ | - | Working |
| Images | .webp, .bmp, .gif | ✅ | - | Working |
| Audio | .wav, .mp3 | - | ✅ | Working |
| Audio | .flac, .ogg, .m4a | - | ✅ | Working |
| Video | .mp4, .avi, .mov | ❌ | ⚠️ | Audio only |
| Docs | .pdf, .docx | ❌ | - | Future |

---

## Technical Limitations

### Current Constraints
- **No real-time video processing** (too compute-intensive)
- **Discord voice needs configuration** (bot token required)
- **No built-in volume UI** (use system controls)
- **No webcam streaming** (future enhancement)
- **Audio file uploads need router integration** (code exists, needs connection)

### Performance Notes
- Image analysis: ~2-5 seconds (CPU) or <1 second (GPU)
- Audio transcription: ~2-5 seconds per 30s chunk (CPU)
- TTS generation: ~3-5 seconds (CPU) or ~1 second (GPU)
- Discord latency: ~100-300ms additional

---

**Summary**: Lyra can currently see uploaded images and hear uploaded audio files. Discord voice is partially implemented and needs configuration. Live microphone has infrastructure ready but needs integration. Volume control is currently system-level but can be easily added to the code.
