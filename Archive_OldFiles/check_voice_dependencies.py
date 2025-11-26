"""
Quick script to check if all voice dependencies are installed
"""
import sys

print("=" * 60)
print("Lea Assistant - Voice Dependencies Check")
print("=" * 60)
print()

missing = []
optional_missing = []

# Core dependencies
print("Checking core dependencies...")
try:
    import PyQt6
    print("[OK] PyQt6")
except ImportError:
    print("[MISSING] PyQt6 - REQUIRED")
    missing.append("PyQt6")

try:
    import dotenv
    print("[OK] python-dotenv")
except ImportError:
    print("[MISSING] python-dotenv - REQUIRED")
    missing.append("python-dotenv")

try:
    from openai import OpenAI
    print("[OK] openai")
except ImportError:
    print("[MISSING] openai - REQUIRED")
    missing.append("openai")

try:
    import requests
    print("[OK] requests")
except ImportError:
    print("[MISSING] requests - REQUIRED")
    missing.append("requests")

print()
print("Checking voice dependencies...")

# Text-to-Speech
try:
    from gtts import gTTS
    print("[OK] gtts (Text-to-Speech)")
except ImportError:
    print("[MISSING] gtts - REQUIRED for TTS")
    missing.append("gtts")

try:
    import pygame
    print("[OK] pygame (Seamless audio playback)")
except ImportError:
    print("[OPTIONAL] pygame - MISSING (Recommended)")
    optional_missing.append("pygame")
    print("   Without pygame, audio will open in media player")

# Speech Recognition
try:
    import speech_recognition as sr
    print("[OK] SpeechRecognition")
except ImportError:
    print("[MISSING] SpeechRecognition - REQUIRED for voice input")
    missing.append("SpeechRecognition")

try:
    import pyaudio
    print("[OK] pyaudio (Microphone access)")
except ImportError:
    print("[MISSING] pyaudio - REQUIRED for microphone")
    missing.append("pyaudio")
    print("   On Windows, you may need: pip install pipwin && pipwin install pyaudio")
    print("   Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")

try:
    import playsound
    print("[OK] playsound (Fallback audio player)")
except ImportError:
    print("[OPTIONAL] playsound - MISSING (Optional fallback)")
    optional_missing.append("playsound")

print()
print("=" * 60)
print("Summary")
print("=" * 60)

if not missing and not optional_missing:
    print("[SUCCESS] All dependencies are installed! You're ready to go!")
elif not missing:
    print("[SUCCESS] All required dependencies are installed!")
    if optional_missing:
        print("[INFO] Optional packages missing (but not critical):")
        for pkg in optional_missing:
            print(f"   - {pkg}")
else:
    print("[ERROR] Missing required packages:")
    for pkg in missing:
        print(f"   - {pkg}")
    print()
    print("To install missing packages, run:")
    print("  pip install -r requirements.txt")
    print()
    if "pyaudio" in missing:
        print("For PyAudio on Windows, try:")
        print("  pip install pipwin")
        print("  pipwin install pyaudio")
        print()
        print("Or download the wheel file from:")
        print("  https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio")

print("=" * 60)

