# EXE Build Folder

This folder contains all the files needed to build Lea Assistant as a standalone executable.

## 📁 Contents

### Build Scripts
- **`build_exe.py`** - Main script to build the executable using PyInstaller
- **`prepare_executable_folder.py`** - Prepares the distribution folder after building
- **`installer.py`** - First-time setup installer for executable version

### Source Files
- **`Lea_Visual_Code_v2.5_ TTS.py`** - Main application file
- **`lea_tasks.py`** - Task execution system
- **`outlook_integration.py`** - Outlook/Microsoft Graph integration
- **`model_registry.py`** - AI model management
- **`universal_file_reader.py`** - File reading utilities
- **`lea_backup_system.py`** - Backup system
- **`workflow_system.py`** - Workflow automation
- **`custom_tasks_example.py`** - Example custom tasks

### Configuration
- **`.env.example`** - Template for environment variables (copy to `.env` and fill in)
- **`requirements.txt`** - Python dependencies

### Installation Scripts
- **`install_dependencies.bat`** - Windows batch script to install dependencies
- **`install_dependencies.ps1`** - PowerShell script to install dependencies

### Documentation
- **`EXECUTABLE_INSTRUCTIONS.md`** - Instructions for building the executable
- **`README_BUILD.md`** - Build documentation

### Assets
- **`assets/`** - Icons, images, and other resources

## 🚀 How to Build the Executable

### Prerequisites
1. Python 3.8+ installed
2. All dependencies installed: `pip install -r requirements.txt`
3. PyInstaller installed: `pip install pyinstaller`

### Build Steps

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. **Build the executable**:
   ```bash
   python build_exe.py
   ```
   
   This will create `LeaAssistant.exe` in the `dist` folder.

3. **Prepare distribution folder** (optional):
   ```bash
   python prepare_executable_folder.py
   ```
   
   This copies the executable and necessary files to `Lea_Executable` folder.

## 📝 Notes

- The executable will be a single `.exe` file with all dependencies bundled
- First-time users will need to run the installer to set up their configuration
- Make sure to have a `.env` file with your API keys before building (or users will need to create one)
- The executable includes all necessary Python modules and assets

## 🔧 Customization

Before building, you can customize:
- Agent name and user name in `.env.example` (or create `.env` file)
- Icon in `assets/` folder (should be `.ico` format for Windows)
- Build options in `build_exe.py`

## 📦 Distribution

After building:
1. The executable will be in the `dist` folder
2. Use `prepare_executable_folder.py` to create a distribution package
3. Include `.env.example` and setup instructions for end users
4. Zip the entire distribution folder for sharing

