# Create Lea Assistant Backup Zip
# This script creates a complete backup of all files needed to run Lea

$ErrorActionPreference = "Stop"

$sourceDir = "c:\Dre_Programs\LeaAssistant"
$backupName = "Lea_Assistant_Backup_$(Get-Date -Format 'yyyy-MM-dd_HHmmss')"
$backupZip = "$sourceDir\$backupName.zip"

Write-Host "Creating Lea Assistant Backup..." -ForegroundColor Green
Write-Host "Source: $sourceDir" -ForegroundColor Cyan
Write-Host "Backup: $backupZip" -ForegroundColor Cyan
Write-Host ""

# Files to include
$filesToInclude = @(
    # Main application
    "Lea_Visual_Code_v2.5.1a_ TTS.py",
    
    # Required modules
    "model_registry.py",
    "universal_file_reader.py",
    "lea_tasks.py",
    
    # Configuration (if exists)
    ".env",
    "lea_settings.json",
    
    # Documentation
    "FIXES_SUMMARY.md",
    "README.md"
)

# Directories to include
$dirsToInclude = @(
    "assets"
)

# Create temp directory for backup
$tempBackupDir = "$env:TEMP\LeaBackup_$(Get-Date -Format 'yyyyMMddHHmmss')"
New-Item -ItemType Directory -Path $tempBackupDir -Force | Out-Null

Write-Host "Copying files..." -ForegroundColor Yellow

# Copy main files
foreach ($file in $filesToInclude) {
    $sourcePath = Join-Path $sourceDir $file
    if (Test-Path $sourcePath) {
        $destPath = Join-Path $tempBackupDir $file
        Copy-Item $sourcePath $destPath -Force
        Write-Host "  [OK] $file" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $file (not found)" -ForegroundColor Yellow
    }
}

# Copy directories
foreach ($dir in $dirsToInclude) {
    $sourcePath = Join-Path $sourceDir $dir
    if (Test-Path $sourcePath) {
        $destPath = Join-Path $tempBackupDir $dir
        Copy-Item $sourcePath $destPath -Recurse -Force
        Write-Host "  [OK] $dir/ (directory)" -ForegroundColor Green
    } else {
        Write-Host "  [SKIP] $dir/ (not found)" -ForegroundColor Yellow
    }
}

# Create README for backup
$readmeContent = @"
# Lea Assistant Backup

This backup was created on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

## Files Included

### Main Application
- Lea_Visual_Code_v2.5.1a_ TTS.py - Main Lea application

### Required Modules
- model_registry.py - Model registry and capability mapping
- universal_file_reader.py - Universal file reading support
- lea_tasks.py - Task execution system

### Configuration
- .env - Environment variables (API keys, etc.) - **IMPORTANT: Contains sensitive data**
- lea_settings.json - User settings

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
5. Copy .env file and add your OPENAI_API_KEY:
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

Backup created: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
"@

$readmePath = Join-Path $tempBackupDir "BACKUP_README.txt"
$readmeContent | Out-File $readmePath -Encoding UTF8

Write-Host "  [OK] BACKUP_README.txt" -ForegroundColor Green

# Create zip file
Write-Host ""
Write-Host "Creating zip file..." -ForegroundColor Yellow

if (Test-Path $backupZip) {
    Remove-Item $backupZip -Force
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory($tempBackupDir, $backupZip)

# Clean up temp directory
Remove-Item $tempBackupDir -Recurse -Force

Write-Host ""
Write-Host "Backup created successfully!" -ForegroundColor Green
Write-Host "Location: $backupZip" -ForegroundColor Cyan
Write-Host "Size: $([math]::Round((Get-Item $backupZip).Length / 1MB, 2)) MB" -ForegroundColor Cyan
Write-Host ""

