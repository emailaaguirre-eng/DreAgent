# Lea Assistant - Voice Setup Guide

## Required Installations for Voice Features

### 1. Install Required Packages

Open a terminal/command prompt and run:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
# Core voice packages
pip install gtts pygame SpeechRecognition pyaudio

# Optional but recommended
pip install playsound
```

### 2. Windows-Specific Notes

**For PyAudio on Windows**, you may need to install it separately:
```bash
pip install pipwin
pipwin install pyaudio
```

Or download the wheel file from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

### 3. Verify Installation

After installation, run the program. It will check for:
- ✅ `gtts` - Text-to-speech (Lea speaks responses)
- ✅ `pygame` - Seamless audio playback (no media player windows)
- ✅ `SpeechRecognition` - Voice input recognition
- ✅ `pyaudio` - Microphone access

## Voice Features

### 🎤 Continuous Listening Mode (NEW!)

**Perfect for hands-free conversations!**

1. Go to **⚙️ Settings**
2. Enable **"🔄 Continuous Listening Mode"**
3. Click the mic button **once** to start
4. Speak your message
5. Lea will automatically:
   - Send your message
   - Process the response
   - Speak the response (if TTS enabled)
   - **Automatically start listening again** for your next message

**No more clicking the mic button repeatedly!**

### ⌨️ Push-to-Talk (Optional)

For even easier control:

1. Go to **⚙️ Settings**
2. Set a **Push-to-Talk Key** (e.g., "Space", "Ctrl+Space", "F1")
3. Press and hold the key to speak
4. Release to send (if not in continuous mode)

### 🎙️ Voice-Only Mode

Hide text during voice conversations for a cleaner experience:

1. Go to **⚙️ Settings**
2. Enable **"🎙️ Voice-Only Mode"**
3. You'll see visual indicators instead of text
4. Perfect for hands-free conversations

## Recommended Setup for Hands-Free Use

1. **Enable TTS** (Text-to-Speech) - Lea will speak responses
2. **Enable Continuous Listening** - Auto-restart after each response
3. **Enable Voice-Only Mode** (optional) - Cleaner interface
4. **Set up microphone** - Select your preferred mic in Settings

## Troubleshooting

### Microphone Not Working?
- Check Windows Privacy Settings > Microphone
- Close other apps using the microphone (Zoom, Teams, etc.)
- Try selecting a different microphone in Settings
- Test your microphone using the "🎤 Test Selected Microphone" button

### Audio Playback Issues?
- Install `pygame` for seamless playback: `pip install pygame`
- Without pygame, audio will open in your default media player

### Speech Recognition Errors?
- Make sure you have an internet connection (uses Google's service)
- Check microphone permissions in Windows Settings
- Try speaking more clearly or closer to the microphone

## Quick Start

1. Install packages: `pip install -r requirements.txt`
2. Run the program
3. Go to **⚙️ Settings**
4. Enable **Continuous Listening Mode**
5. Click the **🎤 mic button** once
6. Start talking! Lea will automatically listen for your next message after responding.

Enjoy your hands-free conversations with Lea! 🐦

