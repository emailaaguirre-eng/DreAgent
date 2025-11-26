# PowerShell script to install all dependencies for Lea Assistant
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Lea Assistant - Dependency Installer" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from https://www.python.org/" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Upgrade pip first
Write-Host "[1/3] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: Failed to upgrade pip, continuing anyway..." -ForegroundColor Yellow
}
Write-Host ""

# Install core dependencies
Write-Host "[2/3] Installing core dependencies..." -ForegroundColor Yellow
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install some dependencies" -ForegroundColor Red
    Write-Host "Please check the error messages above" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Verify critical packages
Write-Host "[3/3] Verifying installation..." -ForegroundColor Yellow
$packages = @(
    @{Name="PyQt6"; Import="PyQt6"},
    @{Name="OpenAI"; Import="openai"},
    @{Name="python-dotenv"; Import="dotenv"},
    @{Name="requests"; Import="requests"},
    @{Name="SpeechRecognition"; Import="speech_recognition"},
    @{Name="edge-tts"; Import="edge_tts"},
    @{Name="gtts"; Import="gtts"},
    @{Name="pygame"; Import="pygame"}
)

foreach ($pkg in $packages) {
    try {
        python -c "import $($pkg.Import)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ $($pkg.Name) installed" -ForegroundColor Green
        } else {
            Write-Host "✗ $($pkg.Name) NOT installed" -ForegroundColor Red
        }
    } catch {
        Write-Host "✗ $($pkg.Name) NOT installed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: Some optional packages may show as not installed if they failed." -ForegroundColor Yellow
Write-Host "The application will work with core packages, but some features may be disabled." -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"

