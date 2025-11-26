# Installation Guide for Lea Multi-Agent System

## Quick Install

1. **Install Python 3.8 or higher** (if not already installed)
   - Download from: https://www.python.org/downloads/
   - Make sure to check "Add Python to PATH" during installation

2. **Navigate to the program directory**
   ```bash
   cd C:\Dre_Programs\LeaAssistant
   ```

3. **Install all required packages**
   ```bash
   pip install -r requirements.txt
   ```

## Required Packages (Core)

These are **essential** for the program to run:

- **python-dotenv** - Loads API keys from .env file
- **requests** - Handles web search API calls
- **openai** - Connects to OpenAI API
- **PyQt6** - Provides the graphical user interface

## Optional Packages (Automation)

These are **optional** - only needed if you want computer automation features:

- **pyautogui** - Mouse and keyboard automation
- **keyboard** - Advanced keyboard event handling

To install automation packages separately:
```bash
pip install pyautogui keyboard
```

## Setup Steps

1. **Copy your .env file** to the new computer with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   SERPAPI_API_KEY=your_key_here
   ```

2. **Copy universal_file_reader.py** (if you have it) to the same directory

3. **Run the program:**
   ```bash
   python Leacurser1.1.py
   ```

## Verification

After installation, run:
```bash
pip list
```

You should see:
- python-dotenv
- requests
- openai
- PyQt6
- (Optional) pyautogui
- (Optional) keyboard

## Troubleshooting

**If PyQt6 installation fails:**
- On Windows: Usually works fine
- On Linux: May need: `sudo apt-get install python3-pyqt6` or `sudo yum install python3-qt6`
- On Mac: May need: `brew install pyqt6`

**If automation doesn't work:**
- Make sure pyautogui and keyboard are installed
- On Linux, may need: `sudo apt-get install python3-tk python3-dev`
- On Mac, may need: `brew install python-tk`

## Minimum System Requirements

- Python 3.8 or higher
- Windows 10/11, macOS 10.14+, or Linux
- 4GB RAM minimum
- Internet connection (for API calls)

