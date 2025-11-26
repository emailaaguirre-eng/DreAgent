"""
Quick script to reset voice settings to defaults
"""
import json
from pathlib import Path

settings_file = Path(__file__).parent / "lea_settings.json"

if settings_file.exists():
    try:
        with open(settings_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reset problematic settings
        data['tts_enabled'] = False
        data['voice_only_mode'] = False
        
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print("Settings reset successfully!")
        print("- TTS disabled")
        print("- Voice-only mode disabled")
        print("\nRestart the program to apply changes.")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Settings file not found. Creating default settings...")
    default_settings = {
        'tts_enabled': False,
        'tts_voice_id': ["en", "com"],
        'edge_tts_voice': "en-US-AriaNeural",
        'microphone_device_index': None,
        'voice_only_mode': False,
        'continuous_listening': False,
        'push_to_talk_key': None
    }
    with open(settings_file, 'w', encoding='utf-8') as f:
        json.dump(default_settings, f, indent=2)
    print("Default settings created!")

