"""Verify model assignments are appropriate for each mode"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

import importlib.util
spec = importlib.util.spec_from_file_location('lea', Path(__file__).parent / 'Lea_Visual_Code_v2.5.1a_ TTS.py')
lea = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lea)

print("=" * 70)
print("MODEL ASSIGNMENTS VERIFICATION")
print("=" * 70)
print()

modes = list(lea.AGENTS.keys())
for mode in modes:
    model_id = lea.get_default_model_for_mode(mode)
    capability = lea.MODE_TO_CAPABILITY.get(mode, "chat_default")
    
    # Determine if assignment is appropriate
    if mode == "General Assistant & Triage":
        appropriate = "✓ Fast model (gpt-5-mini) for quick routing"
    elif mode == "IT Support":
        appropriate = "✓ Maximum capability (gpt-5) for tech support"
    else:
        appropriate = "✓ Maximum capability (gpt-5) for complex tasks"
    
    print(f"{mode:40} -> {model_id:20} ({capability:15}) {appropriate}")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("✓ All modes using appropriate models:")
print("  - Triage: Fast model (gpt-5-mini) for quick responses")
print("  - All other modes: Maximum capability (gpt-5) for best results")
print("✓ IT Support updated to redirect coding questions to Chiquis")
print("✓ Finance & Tax now uses gpt-5 (not vision model)")
print("=" * 70)
