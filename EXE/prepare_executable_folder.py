"""
Prepare Lea_Executable folder with all necessary files for distribution
Run this after building the executable to copy everything needed
"""

import shutil
from pathlib import Path

def prepare_executable_folder():
    """Copy executable and all necessary files to Lea_Executable folder"""
    script_dir = Path(__file__).resolve().parent
    
    # Source and destination paths
    dist_dir = script_dir / "dist"
    exe_source = dist_dir / "LeaAssistant.exe"
    executable_folder = script_dir / "Lea_Executable"
    
    # Create Lea_Executable folder if it doesn't exist
    executable_folder.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Preparing Lea_Executable Folder for Distribution")
    print("=" * 60)
    
    # Check if executable exists
    if not exe_source.exists():
        print(f"\n❌ ERROR: Executable not found at {exe_source}")
        print("   Please run 'python build_exe.py' first to build the executable.")
        return False
    
    # Copy executable
    print(f"\n📦 Copying executable...")
    exe_dest = executable_folder / "LeaAssistant.exe"
    shutil.copy2(exe_source, exe_dest)
    print(f"   ✓ Copied: LeaAssistant.exe")
    
    # Files that should already be in Lea_Executable (created separately)
    expected_files = [
        "README.txt",
        "SETUP_GUIDE.md",
        ".env.example",
        "COPY_HERE.txt"
    ]
    
    print(f"\n📄 Checking for documentation files...")
    for filename in expected_files:
        file_path = executable_folder / filename
        if file_path.exists():
            print(f"   ✓ Found: {filename}")
        else:
            print(f"   ⚠ Missing: {filename} (should be created separately)")
    
    print(f"\n✅ Preparation complete!")
    print(f"\n📁 Executable folder location: {executable_folder}")
    print(f"\n📦 Ready to share:")
    print(f"   1. Zip the entire 'Lea_Executable' folder")
    print(f"   2. Share the zip file")
    print(f"   3. Recipients extract and follow README.txt")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    prepare_executable_folder()

