"""
Lea Backup System
Automated daily backups with 10-day retention
Backs up to: F:\Dre_Programs\LeaAssistant\backups and OneDrive
"""

import os
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('F:/Dre_Programs/LeaAssistant/backup.log'),
        logging.StreamHandler()
    ]
)

# Backup locations
LEA_HOME = Path("F:/Dre_Programs/LeaAssistant")
BACKUP_LOCAL = LEA_HOME / "backups"
BACKUP_ONEDRIVE = Path("C:/Users/email/OneDrive/LeaAssistant_Backups")
RETENTION_DAYS = 10

# Files to backup
BACKUP_FILES = [
    "Lea_Visual_Code_v2.5_ TTS.py",
    "lea_tasks.py",
    "custom_tasks_example.py",
    "universal_file_reader.py",
    "lea_settings.json",
    "lea_history.json",
    "lea_tasks_config.json",
    ".env",  # Backup .env file to preserve all credentials including tenant_id
]

# Directories to backup
BACKUP_DIRS = [
    "memory",
    "assets",
]


def create_backup_destinations():
    """Create backup destination directories if they don't exist"""
    for backup_dir in [BACKUP_LOCAL, BACKUP_ONEDRIVE]:
        backup_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Backup directory ready: {backup_dir}")


def get_backup_name():
    """Generate backup folder name with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"LeaBackup_{timestamp}"


def backup_files(source_dir: Path, backup_dir: Path, backup_name: str):
    """Backup files and directories to backup location"""
    backup_path = backup_dir / backup_name
    backup_path.mkdir(exist_ok=True)
    
    files_backed_up = 0
    dirs_backed_up = 0
    
    # Backup individual files
    for file_name in BACKUP_FILES:
        source_file = source_dir / file_name
        if source_file.exists():
            dest_file = backup_path / file_name
            try:
                shutil.copy2(source_file, dest_file)
                files_backed_up += 1
                logging.info(f"  Backed up: {file_name}")
            except Exception as e:
                logging.error(f"  Failed to backup {file_name}: {e}")
    
    # Backup directories
    for dir_name in BACKUP_DIRS:
        source_dir_path = source_dir / dir_name
        if source_dir_path.exists() and source_dir_path.is_dir():
            dest_dir_path = backup_path / dir_name
            try:
                shutil.copytree(source_dir_path, dest_dir_path, dirs_exist_ok=True)
                dirs_backed_up += 1
                logging.info(f"  Backed up directory: {dir_name}")
            except Exception as e:
                logging.error(f"  Failed to backup directory {dir_name}: {e}")
    
    # Create backup manifest
    manifest = {
        "backup_date": datetime.now().isoformat(),
        "backup_name": backup_name,
        "files_count": files_backed_up,
        "directories_count": dirs_backed_up,
        "source": str(source_dir),
        "destination": str(backup_path)
    }
    
    manifest_file = backup_path / "backup_manifest.json"
    with open(manifest_file, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    
    return files_backed_up, dirs_backed_up


def cleanup_old_backups(backup_dir: Path):
    """Delete backups older than retention period"""
    if not backup_dir.exists():
        return 0
    
    cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
    deleted_count = 0
    
    for backup_folder in backup_dir.iterdir():
        if not backup_folder.is_dir() or not backup_folder.name.startswith("LeaBackup_"):
            continue
        
        try:
            # Extract date from folder name (format: LeaBackup_YYYYMMDD_HHMMSS)
            date_str = backup_folder.name.replace("LeaBackup_", "")
            backup_date = datetime.strptime(date_str.split("_")[0], "%Y%m%d")
            
            if backup_date < cutoff_date:
                shutil.rmtree(backup_folder)
                deleted_count += 1
                logging.info(f"  Deleted old backup: {backup_folder.name}")
        except Exception as e:
            logging.warning(f"  Could not process backup folder {backup_folder.name}: {e}")
    
    return deleted_count


def run_backup():
    """Run the backup process"""
    logging.info("=" * 60)
    logging.info("Starting Lea Backup Process")
    logging.info("=" * 60)
    
    # Create backup destinations
    create_backup_destinations()
    
    backup_name = get_backup_name()
    total_files = 0
    total_dirs = 0
    
    # Backup to local F: drive
    logging.info(f"\nBacking up to: {BACKUP_LOCAL}")
    try:
        files, dirs = backup_files(LEA_HOME, BACKUP_LOCAL, backup_name)
        total_files += files
        total_dirs += dirs
        logging.info(f"✓ Local backup complete: {files} files, {dirs} directories")
    except Exception as e:
        logging.error(f"✗ Local backup failed: {e}")
    
    # Backup to OneDrive
    logging.info(f"\nBacking up to: {BACKUP_ONEDRIVE}")
    try:
        files, dirs = backup_files(LEA_HOME, BACKUP_ONEDRIVE, backup_name)
        total_files += files
        total_dirs += dirs
        logging.info(f"✓ OneDrive backup complete: {files} files, {dirs} directories")
    except Exception as e:
        logging.error(f"✗ OneDrive backup failed: {e}")
    
    # Cleanup old backups
    logging.info(f"\nCleaning up backups older than {RETENTION_DAYS} days...")
    deleted_local = cleanup_old_backups(BACKUP_LOCAL)
    deleted_onedrive = cleanup_old_backups(BACKUP_ONEDRIVE)
    
    logging.info("=" * 60)
    logging.info("Backup Process Complete")
    logging.info(f"  Files backed up: {total_files}")
    logging.info(f"  Directories backed up: {total_dirs}")
    logging.info(f"  Old backups deleted (local): {deleted_local}")
    logging.info(f"  Old backups deleted (OneDrive): {deleted_onedrive}")
    logging.info("=" * 60)


if __name__ == "__main__":
    run_backup()

