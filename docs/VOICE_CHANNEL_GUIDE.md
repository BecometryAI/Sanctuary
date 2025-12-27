# Voice Channel Integration - Quick Reference

## Slash Commands

### `/join`
Join your current voice channel
- **Requirements**: You must be in a voice channel
- **Usage**: `/join`
- **What it does**: Lyra joins the voice channel you're currently in

### `/leave`
Leave the current voice channel
- **Requirements**: Lyra must be in a voice channel
- **Usage**: `/leave`
- **What it does**: Lyra disconnects from voice

### `/autojoin`
Toggle autonomous voice channel joining
- **Usage**: `/autojoin enabled:True` or `/autojoin enabled:False`
- **What it does**: 
  - When **enabled**: Lyra automatically joins voice channels when she sees you enter one
  - When **disabled**: You need to manually use `/join` to bring her into voice
- **Auto-leave**: Lyra will automatically leave when all users with auto-join enabled have left

## Autonomous Joining Behavior

### How It Works
1. **Enable auto-join**: Use `/autojoin enabled:True`
2. **Join a voice channel**: Lyra will automatically connect within 30 seconds
3. **She stays as long as you're there**: Even if you briefly disconnect and rejoin
4. **Auto-leave**: When you (and any other auto-join users) leave, she'll disconnect

### Smart Features
- **Real-time detection**: Uses Discord's `on_voice_state_update` event for instant joining when you enter voice
- **Background monitoring**: Checks voice channels every 30 seconds as a backup
- **Multi-user support**: If multiple people have auto-join enabled, she'll stay until everyone leaves
- **Personalized greetings**: Uses your real name if you've set it with `/setname`

### Use Cases

**Scenario 1: Quick Chat**
```
You: /autojoin enabled:True
Lyra: ✅ Auto-join enabled! I'll join voice channels when I see you, Brian.
      🎤 Joining you now in General!
[You're already in voice, she joins immediately]
```

**Scenario 2: Regular Voice Sessions**
```
You: /autojoin enabled:True
[Later that day, you join a voice channel]
[Lyra automatically joins within seconds]
```

**Scenario 3: Manual Control**
```
You: /autojoin enabled:False
[You join voice channel]
[Lyra doesn't join]
You: /join
Lyra: ✅ Joined General with Brian
```

## Integration with User Mapping

Voice commands use the user mapping system:
- Logs show your real name (if set): `"Brian joined General"`
- Auto-join messages personalized: `"I'll join when I see you, Brian"`
- Works with steward detection for priority handling

## Technical Details

### Voice State Monitoring
- **Event-driven**: `on_voice_state_update` for instant response
- **Polling backup**: Background task checks every 30 seconds
- **Smart decisions**: Only joins if auto-join users are present

### Preference Storage
```python
voice_preferences = {
    "user_id": {
        "auto_join": True/False,
        "preferred_channels": []  # Future: channel-specific preferences
    }
}
```

Currently in-memory only (resets on bot restart). Could be persisted to JSON if needed.

## Future Enhancements

### Potential Features
1. **Channel-specific preferences**: Only auto-join certain channels
2. **Time-based rules**: Auto-join only during certain hours
3. **Voice activity**: Speak greetings when joining
4. **Scheduled joins**: "Join General every day at 3pm"
5. **Smart notifications**: Send DM when unable to join

### Voice Output Integration
Once TTS is working:
- Greet users when auto-joining
- Speak responses in voice channels
- Volume control with `/volume` command

## Testing Checklist

- [ ] `/join` works when in voice channel
- [ ] `/join` fails gracefully when not in voice
- [ ] `/leave` disconnects properly
- [ ] `/autojoin enabled:True` enables preference
- [ ] Auto-join works when entering voice channel
- [ ] Auto-leave works when exiting voice channel
- [ ] Multiple users with auto-join work correctly
- [ ] Real names appear in logs (if set with `/setname`)
- [ ] Background monitoring task runs without errors

## Notes

- PyNaCl required for voice support (currently shows warning)
- TTS integration pending (voice output not yet implemented)
- Voice preferences reset on bot restart (not persisted)
- Designed to respect user agency (opt-in, not opt-out)
