#!/usr/bin/env python3
"""
Dependency checker for Lea Assistant
Checks if all required and optional packages are installed
"""

import sys
import importlib
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

# Required packages (application won't work without these)
REQUIRED_PACKAGES = {
    'PyQt6': 'PyQt6',
    'openai': 'openai',
    'dotenv': 'python-dotenv',
    'requests': 'requests',
}

# Optional packages (features will be disabled if missing)
OPTIONAL_PACKAGES = {
    'speech_recognition': 'SpeechRecognition',
    'pyaudio': 'pyaudio',
    'edge_tts': 'edge-tts',
    'gtts': 'gtts',
    'pyttsx3': 'pyttsx3',
    'pygame': 'pygame',
    'playsound': 'playsound',
}

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("Lea Assistant - Dependency Checker")
    print("=" * 60)
    print()
    
    # Check required packages
    print("REQUIRED PACKAGES (must be installed):")
    print("-" * 60)
    required_missing = []
    for import_name, package_name in REQUIRED_PACKAGES.items():
        installed, error = check_package(import_name, import_name)
        status = "[OK] INSTALLED" if installed else "[X] MISSING"
        print(f"{status} - {package_name}")
        if not installed:
            required_missing.append(package_name)
    
    print()
    
    # Check optional packages
    print("OPTIONAL PACKAGES (features disabled if missing):")
    print("-" * 60)
    optional_missing = []
    for import_name, package_name in OPTIONAL_PACKAGES.items():
        installed, error = check_package(import_name, import_name)
        status = "[OK] INSTALLED" if installed else "[ ] OPTIONAL"
        print(f"{status} - {package_name}")
        if not installed:
            optional_missing.append(package_name)
    
    print()
    print("=" * 60)
    
    # Summary
    if required_missing:
        print("[X] CRITICAL: Missing required packages!")
        print("   Install with: pip install " + " ".join(required_missing))
        print()
        print("   Or run: pip install -r requirements.txt")
        return 1
    else:
        print("[OK] All required packages are installed!")
        if optional_missing:
            print(f"[!] {len(optional_missing)} optional package(s) missing (features may be limited)")
            print("   Install with: pip install " + " ".join(optional_missing))
        else:
            print("[OK] All optional packages are installed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())

