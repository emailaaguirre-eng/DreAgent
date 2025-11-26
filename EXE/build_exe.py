"""
Build Lea as a standalone executable
Creates a single EXE file with Python and all dependencies bundled
"""

import PyInstaller.__main__
import os
import sys
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).resolve().parent

# Main script to package
main_script = script_dir / "Lea_Visual_Code_v2.5_ TTS.py"

# Icon file (if available)
icon_file = script_dir / "assets" / "Hummingbird_LEA_v1_Logo_Neon_Green.ico"
if not icon_file.exists():
    icon_file = script_dir / "assets" / "Hummingbird_LEA_v1_Logo_Neon_Green.png"

# Additional data files to include
additional_files = [
    ("assets", "assets"),
    ("lea_tasks.py", "."),
    ("custom_tasks_example.py", "."),
    ("universal_file_reader.py", "."),
    ("lea_backup_system.py", "."),
    ("installer.py", "."),  # Include installer for first-time setup
]

# Build PyInstaller arguments
pyinstaller_args = [
    str(main_script),
    "--name=LeaAssistant",
    "--onefile",  # Create a single executable file
    "--windowed",  # No console window (GUI only)
    "--clean",  # Clean cache before building
    "--noconfirm",  # Overwrite output without asking
]

# Add icon if it exists
if icon_file.exists():
    pyinstaller_args.append(f"--icon={icon_file}")

# Add additional files
for src, dst in additional_files:
    src_path = script_dir / src
    if src_path.exists():
        pyinstaller_args.append(f"--add-data={src}{os.pathsep}{dst}")

# Add hidden imports (PyQt6 sometimes needs these)
hidden_imports = [
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "openai",
    "dotenv",
    "msal",
    "openpyxl",
    "pandas",
    "gtts",
    "speech_recognition",
]

for imp in hidden_imports:
    pyinstaller_args.append(f"--hidden-import={imp}")

# Output directory - build to dist first, then we'll copy to Lea_Executable
output_dir = script_dir / "dist"
pyinstaller_args.append(f"--distpath={output_dir}")

# Work directory
work_dir = script_dir / "build"
pyinstaller_args.append(f"--workpath={work_dir}")

# Spec file location
spec_file = script_dir / "LeaAssistant.spec"
pyinstaller_args.append(f"--specpath={script_dir}")

print("=" * 60)
print("Building Lea Assistant Executable")
print("=" * 60)
print(f"Main script: {main_script}")
print(f"Icon: {icon_file if icon_file.exists() else 'Not found'}")
print(f"Output: {output_dir / 'LeaAssistant.exe'}")
print("=" * 60)
print()

# Run PyInstaller
try:
    PyInstaller.__main__.run(pyinstaller_args)
    print()
    print("=" * 60)
    print("✓ Build Complete!")
    print(f"Executable location: {output_dir / 'LeaAssistant.exe'}")
    print("=" * 60)
except Exception as e:
    print()
    print("=" * 60)
    print("✗ Build Failed!")
    print(f"Error: {e}")
    print("=" * 60)
    print()
    print("Make sure PyInstaller is installed:")
    print("  pip install pyinstaller")
    sys.exit(1)

