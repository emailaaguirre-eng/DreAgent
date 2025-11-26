# Lea Assistant - Dependencies

## ✅ Installation Status

All **required** dependencies are installed and ready to use!

### Required Packages (All Installed ✓)
- **PyQt6** - GUI framework
- **openai** - OpenAI API client
- **python-dotenv** - Environment variable management
- **requests** - HTTP requests

### Optional Packages (Most Installed ✓)
- **SpeechRecognition** ✓ - Speech-to-text functionality
- **pyaudio** ✓ - Microphone access for speech recognition
- **edge-tts** ✓ - High-quality offline text-to-speech
- **gtts** ✓ - Google Text-to-Speech (fallback)
- **pyttsx3** ✓ - Cross-platform TTS engine
- **pygame** ✓ - Audio playback
- **playsound** ○ - Alternative audio playback (optional, pygame is sufficient)

## 📦 Installation Files

### Quick Install
Run one of these scripts to install all dependencies:

**Windows Batch:**
```batch
install_dependencies.bat
```

**PowerShell:**
```powershell
.\install_dependencies.ps1
```

**Manual Install:**
```bash
pip install -r requirements.txt
```

### Check Dependencies
To verify all packages are installed:
```bash
python check_dependencies.py
```

## 📝 Notes

- **playsound** is optional and may have build issues on some systems. **pygame** is already installed and provides all necessary audio playback functionality.
- All critical features will work with the currently installed packages.
- If you need to reinstall dependencies, use `pip install -r requirements.txt`

## 🔧 Troubleshooting

If you encounter issues:

1. **Upgrade pip first:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Install packages individually if batch install fails:**
   ```bash
   pip install PyQt6 openai python-dotenv requests
   pip install SpeechRecognition pyaudio edge-tts gtts pyttsx3 pygame
   ```

3. **For pyaudio issues on Windows:**
   - May need to install from a wheel file or use conda
   - Alternative: `pip install pipwin` then `pipwin install pyaudio`

4. **Check Python version:**
   - Requires Python 3.8 or higher
   - Check with: `python --version`

## ✅ Current Status

All required dependencies are installed and the application is ready to run!

