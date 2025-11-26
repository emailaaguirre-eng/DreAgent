"""
Simple Python script to create Lea Assistant backup zip
Run this with: python create_backup_simple.py
"""
import os
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

def create_backup():
    """Create a complete backup zip of Lea Assistant"""
    
    source_dir = Path(__file__).parent
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    backup_name = f"Lea_Assistant_Backup_{timestamp}.zip"
    backup_path = source_dir / backup_name
    
    print("=" * 60)
    print("Creating Lea Assistant Backup")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Backup: {backup_path}")
    print()
    
    # Files to include
    files_to_include = [
        # Main application
        "Lea_Visual_Code_v2.5.1a_ TTS.py",
        
        # Required modules
        "model_registry.py",
        "universal_file_reader.py",
        "lea_tasks.py",
        
        # Configuration (optional - user should add their own)
        "lea_settings.json",
        
        # Documentation
        "FIXES_SUMMARY.md",
    ]
    
    # Directories to include
    dirs_to_include = [
        "assets"
    ]
    
    # Create zip file
    with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        print("Adding files...")
        
        # Add main files
        for filename in files_to_include:
            filepath = source_dir / filename
            if filepath.exists():
                zipf.write(filepath, filename)
                print(f"  [OK] {filename}")
            else:
                print(f"  [SKIP] {filename} (not found)")
        
        # Add directories
        for dirname in dirs_to_include:
            dirpath = source_dir / dirname
            if dirpath.exists() and dirpath.is_dir():
                for root, dirs, files in os.walk(dirpath):
                    for file in files:
                        filepath = Path(root) / file
                        arcname = filepath.relative_to(source_dir)
                        zipf.write(filepath, arcname)
                print(f"  [OK] {dirname}/ (directory)")
            else:
                print(f"  [SKIP] {dirname}/ (not found)")
        
        # Create README for backup
        readme_content = f"""# Lea Assistant Backup

This backup was created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files Included

### Main Application
- Lea_Visual_Code_v2.5.1a_ TTS.py - Main Lea application

### Required Modules
- model_registry.py - Model registry and capability mapping
- universal_file_reader.py - Universal file reading support
- lea_tasks.py - Task execution system

### Configuration
- lea_settings.json - User settings
- .env - **NOT INCLUDED** (contains sensitive API keys - create your own)

### Assets
- assets/ - Application icons and splash screens

## Setup Instructions

1. Extract this zip file to a directory
2. Install Python 3.8 or higher
3. Install required packages:
   ```
   pip install PyQt6 openai python-dotenv
   ```
4. Optional packages (for full functionality):
   ```
   pip install gtts pygame SpeechRecognition pyaudio Pillow
   ```
5. Create a .env file in the extracted directory with your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
6. Run Lea:
   ```
   python "Lea_Visual_Code_v2.5.1a_ TTS.py"
   ```

## Features

- Multi-agent system with 7 specialized modes
- Model registry with automatic fallback
- Universal file reading
- Task execution system
- TTS (Text-to-Speech) support
- Speech recognition support
- Image analysis support

## Troubleshooting

If you encounter issues:
1. Check that all required Python packages are installed
2. Verify your .env file has a valid OPENAI_API_KEY
3. Check the console for error messages
4. See FIXES_SUMMARY.md for known issues and fixes

## Version

This backup includes fixes for:
- Mode switching crashes
- Model availability issues
- TTS feature blocking
- Unicode encoding errors

Backup created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        zipf.writestr("BACKUP_README.txt", readme_content)
        print("  [OK] BACKUP_README.txt")
    
    # Get file size
    size_mb = backup_path.stat().st_size / (1024 * 1024)
    
    print()
    print("=" * 60)
    print("Backup created successfully!")
    print("=" * 60)
    print(f"Location: {backup_path}")
    print(f"Size: {size_mb:.2f} MB")
    print()
    print("Note: .env file was NOT included for security reasons.")
    print("      Create your own .env file with your OPENAI_API_KEY")
    print("=" * 60)
    
    return backup_path

if __name__ == "__main__":
    try:
        create_backup()
    except Exception as e:
        print(f"\nError creating backup: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

