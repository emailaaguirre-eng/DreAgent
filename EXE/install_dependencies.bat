@echo off
REM Install all dependencies for Lea Assistant
echo ========================================
echo Lea Assistant - Dependency Installer
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo Python found!
echo.

REM Upgrade pip first
echo [1/3] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip, continuing anyway...
)
echo.

REM Install core dependencies
echo [2/3] Installing core dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install some dependencies
    echo Please check the error messages above
    pause
    exit /b 1
)
echo.

REM Verify critical packages
echo [3/3] Verifying installation...
python -c "import PyQt6; print('✓ PyQt6 installed')" 2>nul || echo ✗ PyQt6 NOT installed
python -c "import openai; print('✓ OpenAI installed')" 2>nul || echo ✗ OpenAI NOT installed
python -c "import dotenv; print('✓ python-dotenv installed')" 2>nul || echo ✗ python-dotenv NOT installed
python -c "import requests; print('✓ requests installed')" 2>nul || echo ✗ requests NOT installed
python -c "import speech_recognition; print('✓ SpeechRecognition installed')" 2>nul || echo ✗ SpeechRecognition NOT installed
python -c "import edge_tts; print('✓ edge-tts installed')" 2>nul || echo ✗ edge-tts NOT installed
python -c "import gtts; print('✓ gtts installed')" 2>nul || echo ✗ gtts NOT installed
python -c "import pygame; print('✓ pygame installed')" 2>nul || echo ✗ pygame NOT installed
echo.

echo ========================================
echo Installation complete!
echo ========================================
echo.
echo Note: Some optional packages may show as not installed if they failed.
echo The application will work with core packages, but some features may be disabled.
echo.
pause

