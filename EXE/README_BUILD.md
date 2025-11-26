# Building Lea Assistant as an Executable

This guide explains how to create a standalone EXE file that can run on any Windows machine without requiring Python or package installation.

## Prerequisites

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```

2. Install all Lea dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Building the Executable

### Option 1: Using the build script (Recommended)
```bash
python build_exe.py
```

### Option 2: Manual PyInstaller command
```bash
pyinstaller --name=LeaAssistant --onefile --windowed --icon=assets\Hummingbird_LEA_v1_Logo_Neon_Green.ico --add-data="assets;assets" --add-data="lea_tasks.py;." --add-data="custom_tasks_example.py;." --add-data="universal_file_reader.py;." "Lea_Visual_Code_v2.5_ TTS.py"
```

## Output

The executable will be created at:
- `F:\Dre_Programs\LeaAssistant\dist\LeaAssistant.exe`

## Preparing for Distribution

After building, prepare the distribution folder:

```bash
python prepare_executable_folder.py
```

This will:
- Copy the executable to `Lea_Executable/` folder
- Verify all documentation files are present
- The folder is ready to zip and share!

The `Lea_Executable/` folder contains:
- `LeaAssistant.exe` - The main executable
- `README.txt` - Quick start guide
- `SETUP_GUIDE.md` - Detailed setup instructions
- `.env.example` - Template for API keys
- `COPY_HERE.txt` - Instructions for distribution

## Using the Executable

1. Copy `LeaAssistant.exe` to any Windows machine
2. **First Run**: The installer will prompt you to customize:
   - Agent name (what to call the assistant)
   - Your name
   - Personality description (optional)
3. Create a `.env` file in the same folder with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   OUTLOOK_CLIENT_ID=your_client_id (optional)
   OUTLOOK_CLIENT_SECRET=your_secret (optional)
   OUTLOOK_TENANT_ID=your_tenant_id (optional)
   ```
4. Run `LeaAssistant.exe` - no Python installation needed!

**Note**: Each user can customize the assistant with their own names and personality when they first run it. See `EXECUTABLE_INSTRUCTIONS.md` for detailed setup instructions to share with recipients.

## Notes

- The EXE file will be large (100-200MB) because it bundles Python and all dependencies
- First run may be slower as files are extracted
- The EXE is portable - you can put it on a USB drive and run it anywhere
- Settings and history will be saved in the same folder as the EXE

## Troubleshooting

If the EXE doesn't work:
1. Check that all dependencies are installed before building
2. Try building with `--console` instead of `--windowed` to see error messages
3. Make sure the `.env` file is in the same folder as the EXE

