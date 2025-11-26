# Quick Start - Building the Executable

## 🚀 Fast Track

1. **Open a terminal/command prompt in this EXE folder**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

3. **Build the executable**:
   ```bash
   python build_exe.py
   ```

4. **Find your executable**:
   - Location: `EXE\dist\LeaAssistant.exe`
   - The executable is ready to use!

## 📋 Detailed Steps

### Step 1: Prerequisites
- Python 3.8 or higher installed
- pip package manager

### Step 2: Install Dependencies
Run one of these commands:
```bash
# Windows Batch
install_dependencies.bat

# PowerShell
.\install_dependencies.ps1

# Or manually
pip install -r requirements.txt
pip install pyinstaller
```

### Step 3: Build
```bash
python build_exe.py
```

This will:
- Create a single `LeaAssistant.exe` file
- Bundle all Python dependencies
- Include all assets and resources
- Output to `EXE\dist\LeaAssistant.exe`

### Step 4: Test
Run the executable:
```bash
.\dist\LeaAssistant.exe
```

### Step 5: Distribute (Optional)
To prepare a distribution package:
```bash
python prepare_executable_folder.py
```

This creates a `Lea_Executable` folder with everything needed for distribution.

## ⚠️ Important Notes

- The executable is **self-contained** - no Python installation needed on target machines
- First-time users will need to create a `.env` file (use `.env.example` as template)
- The executable includes the installer for first-time setup
- Build time: Usually 2-5 minutes depending on your system

## 🐛 Troubleshooting

**Build fails?**
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Make sure PyInstaller is installed: `pip install pyinstaller`
- Check that all source files are in this EXE folder

**Executable doesn't run?**
- Make sure you have a `.env` file with your API keys
- Check Windows Defender isn't blocking it
- Try running from command line to see error messages

**Need help?**
- See `EXECUTABLE_INSTRUCTIONS.md` for detailed instructions
- See `README_BUILD.md` for build documentation
- See `README_EXE_FOLDER.md` for folder contents

