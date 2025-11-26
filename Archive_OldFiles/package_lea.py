"""
Lea Package Creator
Creates a portable zip file with everything needed to run Lea on another computer
Excludes personal data, secrets, and temporary files
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def create_lea_package():
    """Create a portable Lea package"""
    
    # Get the project directory
    project_dir = Path(__file__).parent.resolve()
    package_name = f"Lea_Portable_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    package_dir = project_dir / package_name
    zip_file = project_dir / f"{package_name}.zip"
    
    print("📦 Creating Lea Portable Package...")
    print(f"   Source: {project_dir}")
    print(f"   Package: {package_name}")
    print()
    
    # Create package directory
    package_dir.mkdir(exist_ok=True)
    
    # Essential files to include
    essential_files = [
        "Lea_Visual_Code_v2.5.1a_ TTS.py",  # Main program
        "requirements.txt",                  # Dependencies
        "lea_update_checker.py",             # Update checker
        "UPDATE_SYSTEM_README.md",           # Update docs
    ]
    
    # Optional files (if they exist)
    optional_files = [
        "universal_file_reader.py",         # File reader (if exists)
        "lea_tasks.py",                      # Task system (if exists)
        ".env.example",                      # Environment template
    ]
    
    # Directories to include (if they exist)
    essential_dirs = [
        "assets",                            # Icons, splash screens
    ]
    
    # Files/directories to exclude
    exclude_patterns = [
        # Personal data
        ".env",                              # Contains secrets
        "lea_history.json",                  # Personal chat history
        "lea_settings.json",                 # Personal settings
        "outlook_token_cache.json",          # Personal tokens
        "outlook_recommendations.json",      # Personal recommendations
        "last_update_check.json",            # Not essential
        "update_check.log",                   # Log file
        "lea_crash.log",                     # Crash log
        
        # Temporary/cache
        "__pycache__",                       # Python cache
        "*.pyc",                             # Compiled Python
        "*.pyo",                              # Optimized Python
        ".pytest_cache",                     # Test cache
        
        # Generated directories
        "backups",                           # Backup files
        "downloads",                         # Downloaded files
        "memory",                            # Memory cache
        
        # Package files
        "*.zip",                             # Don't include other packages
        "package_lea.py",                    # This script
    ]
    
    # Copy essential files
    print("📄 Copying essential files...")
    copied_count = 0
    for file_name in essential_files:
        src = project_dir / file_name
        if src.exists():
            dst = package_dir / file_name
            shutil.copy2(src, dst)
            print(f"   ✅ {file_name}")
            copied_count += 1
        else:
            print(f"   ⚠️  {file_name} (not found)")
    
    # Copy optional files
    print("\n📄 Copying optional files...")
    for file_name in optional_files:
        src = project_dir / file_name
        if src.exists():
            dst = package_dir / file_name
            shutil.copy2(src, dst)
            print(f"   ✅ {file_name}")
            copied_count += 1
    
    # Copy essential directories
    print("\n📁 Copying directories...")
    for dir_name in essential_dirs:
        src = project_dir / dir_name
        if src.exists() and src.is_dir():
            dst = package_dir / dir_name
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"   ✅ {dir_name}/")
            copied_count += 1
    
    # Create .env.example if it doesn't exist
    env_example = package_dir / ".env.example"
    if not env_example.exists():
        env_template = """# Lea Assistant Environment Variables
# Copy this file to .env and fill in your values

# OpenAI API Key (Required)
OPENAI_API_KEY=your_openai_api_key_here

# Outlook/Microsoft Graph (Optional)
OUTLOOK_CLIENT_ID=your_outlook_client_id_here
OUTLOOK_TENANT_ID=your_tenant_id_or_common

# Web Search (Optional)
# SERPAPI_API_KEY=your_serpapi_key_here
"""
        with open(env_example, 'w') as f:
            f.write(env_template)
        print(f"   ✅ Created .env.example")
    
    # Create README for package
    readme_content = f"""# Lea Assistant - Portable Package

This package contains everything you need to run Lea on a new computer.

## Package Contents

- **Lea_Visual_Code_v2.5.1a_ TTS.py** - Main Lea program
- **requirements.txt** - Python package dependencies
- **lea_update_checker.py** - Automatic update checker
- **assets/** - Icons and splash screens
- **.env.example** - Environment variables template

## Setup Instructions

### 1. Install Python
- Download Python 3.8+ from https://www.python.org/
- Make sure to check "Add Python to PATH" during installation

### 2. Install Dependencies
Open a terminal/command prompt in this folder and run:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
1. Copy `.env.example` to `.env`
2. Edit `.env` and add your API keys:
   - `OPENAI_API_KEY` - Required (get from https://platform.openai.com/)
   - `OUTLOOK_CLIENT_ID` - Optional (for Outlook integration)
   - `OUTLOOK_TENANT_ID` - Optional (for Outlook integration)

### 4. Run Lea
```bash
python "Lea_Visual_Code_v2.5.1a_ TTS.py"
```

Or double-click the file if Python is associated with .py files.

## Features

- ✅ Multi-agent system (7 specialized modes)
- ✅ Outlook/Email integration
- ✅ Text-to-Speech
- ✅ Speech Recognition
- ✅ File reading and processing
- ✅ Task automation
- ✅ Automatic update checking
- ✅ Memory system
- ✅ And much more!

## Troubleshooting

- **Import errors**: Make sure all dependencies are installed (`pip install -r requirements.txt`)
- **API errors**: Check your `.env` file has correct API keys
- **Outlook not working**: Make sure you've set up Azure app registration and added Client ID to `.env`

## Support

For issues or questions, check the update system README or review the code comments.

---
Package created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_file = package_dir / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"   ✅ Created README.md")
    
    # Create setup script for Windows
    setup_script = package_dir / "setup.bat"
    setup_content = """@echo off
echo ========================================
echo Lea Assistant - Setup Script
echo ========================================
echo.
echo This will install all required packages.
echo.
pause

echo Installing packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Copy .env.example to .env
echo 2. Edit .env and add your API keys
echo 3. Run: python "Lea_Visual_Code_v2.5.1a_ TTS.py"
echo.
pause
"""
    with open(setup_script, 'w') as f:
        f.write(setup_content)
    print(f"   ✅ Created setup.bat")
    
    # Create zip file
    print(f"\n📦 Creating zip file...")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            # Skip excluded patterns
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                file_path = Path(root) / file
                # Skip excluded files
                if any(pattern in file for pattern in exclude_patterns):
                    continue
                
                # Get relative path for zip
                arcname = file_path.relative_to(package_dir)
                zipf.write(file_path, arcname)
                print(f"   📄 {arcname}")
    
    # Clean up package directory
    print(f"\n🧹 Cleaning up...")
    shutil.rmtree(package_dir)
    
    # Final summary
    zip_size_mb = zip_file.stat().st_size / (1024 * 1024)
    print(f"\n✅ Package created successfully!")
    print(f"   📦 File: {zip_file.name}")
    print(f"   📊 Size: {zip_size_mb:.2f} MB")
    print(f"   📍 Location: {zip_file.parent}")
    print(f"\n🎉 Ready to share or backup!")
    print(f"\n💡 To use on another computer:")
    print(f"   1. Extract the zip file")
    print(f"   2. Run setup.bat (Windows) or install requirements manually")
    print(f"   3. Copy .env.example to .env and add your API keys")
    print(f"   4. Run Lea!")

if __name__ == "__main__":
    try:
        create_lea_package()
    except Exception as e:
        print(f"\n❌ Error creating package: {e}")
        import traceback
        traceback.print_exc()

