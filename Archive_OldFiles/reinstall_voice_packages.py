"""
Script to uninstall and reinstall all voice/TTS packages
"""
import subprocess
import sys

# All voice-related packages
packages = [
    "edge-tts",
    "gtts",
    "SpeechRecognition",
    "pygame",
    "pyaudio",
]

print("=" * 60)
print("Voice Package Reinstallation Script")
print("=" * 60)
print()

# Step 1: Uninstall packages
print("Step 1: Uninstalling packages...")
print("-" * 60)
for package in packages:
    try:
        print(f"Uninstalling {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", package, "-y"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  [OK] {package} uninstalled")
        else:
            print(f"  [INFO] {package} may not have been installed")
    except Exception as e:
        print(f"  [ERROR] Error uninstalling {package}: {e}")

print()
print("Step 2: Installing packages...")
print("-" * 60)

# Step 2: Reinstall packages
for package in packages:
    try:
        print(f"Installing {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  [OK] {package} installed successfully")
        else:
            print(f"  [ERROR] Error installing {package}")
            print(f"    {result.stderr}")
    except Exception as e:
        print(f"  [ERROR] Error installing {package}: {e}")

print()
print("Step 3: Verifying installations...")
print("-" * 60)

# Step 3: Verify installations
import_checks = {
    "edge-tts": "import edge_tts",
    "gtts": "from gtts import gTTS",
    "SpeechRecognition": "import speech_recognition",
    "pygame": "import pygame",
    "pyaudio": "import pyaudio",
}

all_ok = True
for package, import_cmd in import_checks.items():
    try:
        exec(import_cmd)
        print(f"  [OK] {package} - OK")
    except ImportError as e:
        print(f"  [FAILED] {package} - FAILED: {e}")
        all_ok = False
    except Exception as e:
        print(f"  [ERROR] {package} - ERROR: {e}")
        all_ok = False

print()
print("=" * 60)
if all_ok:
    print("[SUCCESS] All packages installed and verified successfully!")
    print()
    print("Next steps:")
    print("1. Restart the Lea Assistant program")
    print("2. Go to Settings")
    print("3. Enable TTS and select a voice")
    print("4. Test the microphone if needed")
else:
    print("[WARNING] Some packages failed to install. Check the errors above.")
print("=" * 60)

