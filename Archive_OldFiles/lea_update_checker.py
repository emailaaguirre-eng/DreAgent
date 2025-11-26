"""
Lea Update Checker
Automatically checks for outdated packages and script updates
"""

import subprocess
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import requests

# Version of this update checker
UPDATE_CHECKER_VERSION = "1.0.0"

# Lea script version (update this when you release new versions)
LEA_SCRIPT_VERSION = "2.5.1a"

# GitHub repository info (if you host Lea on GitHub)
# Leave empty if not using GitHub for updates
GITHUB_REPO_OWNER = ""  # e.g., "yourusername"
GITHUB_REPO_NAME = ""   # e.g., "LeaAssistant"

class UpdateChecker:
    """Check for package and script updates"""
    
    def __init__(self, requirements_file: Path = None, check_interval_days: int = 7):
        self.requirements_file = requirements_file or Path(__file__).parent / "requirements.txt"
        self.check_interval_days = check_interval_days
        self.last_check_file = Path(__file__).parent / "last_update_check.json"
        self.update_log_file = Path(__file__).parent / "update_check.log"
        
        # Setup logging
        logging.basicConfig(
            filename=str(self.update_log_file),
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    
    def should_check_updates(self) -> bool:
        """Check if enough time has passed since last check"""
        if not self.last_check_file.exists():
            return True
        
        try:
            with open(self.last_check_file, 'r') as f:
                data = json.load(f)
                last_check = datetime.fromisoformat(data.get("last_check", "2000-01-01"))
                days_since = (datetime.now() - last_check).days
                return days_since >= self.check_interval_days
        except Exception as e:
            logging.warning(f"Error reading last check file: {e}")
            return True
    
    def record_check(self):
        """Record that we checked for updates"""
        try:
            data = {
                "last_check": datetime.now().isoformat(),
                "checker_version": UPDATE_CHECKER_VERSION
            }
            with open(self.last_check_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logging.error(f"Error recording check: {e}")
    
    def check_package_updates(self) -> Dict[str, any]:
        """Check for outdated Python packages"""
        result = {
            "success": False,
            "outdated": [],
            "error": None,
            "summary": ""
        }
        
        try:
            # Run pip list --outdated
            process = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if process.returncode == 0:
                outdated = json.loads(process.stdout)
                result["outdated"] = outdated
                result["success"] = True
                
                if outdated:
                    result["summary"] = f"Found {len(outdated)} outdated package(s):\n"
                    for pkg in outdated:
                        result["summary"] += f"  • {pkg['name']}: {pkg['version']} → {pkg['latest_version']}\n"
                else:
                    result["summary"] = "All packages are up to date! ✅"
            else:
                result["error"] = process.stderr or "Unknown error"
                logging.error(f"pip list --outdated failed: {result['error']}")
        
        except subprocess.TimeoutExpired:
            result["error"] = "Update check timed out"
            logging.error("Update check timed out")
        except json.JSONDecodeError as e:
            result["error"] = f"Failed to parse pip output: {e}"
            logging.error(f"JSON decode error: {e}")
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"Error checking package updates: {e}")
        
        return result
    
    def check_script_version(self) -> Dict[str, any]:
        """Check for Lea script updates (if using GitHub)"""
        result = {
            "success": False,
            "update_available": False,
            "current_version": LEA_SCRIPT_VERSION,
            "latest_version": None,
            "release_url": None,
            "error": None
        }
        
        if not GITHUB_REPO_OWNER or not GITHUB_REPO_NAME:
            result["error"] = "GitHub repository not configured"
            return result
        
        try:
            # Check GitHub releases API
            url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/releases/latest"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                release_data = response.json()
                latest_version = release_data.get("tag_name", "").lstrip("v")
                result["latest_version"] = latest_version
                result["release_url"] = release_data.get("html_url")
                
                # Simple version comparison (you might want more sophisticated logic)
                if latest_version != LEA_SCRIPT_VERSION:
                    result["update_available"] = True
                    result["success"] = True
                else:
                    result["success"] = True
            else:
                result["error"] = f"GitHub API returned status {response.status_code}"
        
        except requests.exceptions.RequestException as e:
            result["error"] = f"Failed to check GitHub: {e}"
            logging.error(f"GitHub check error: {e}")
        except Exception as e:
            result["error"] = str(e)
            logging.error(f"Error checking script version: {e}")
        
        return result
    
    def get_update_summary(self) -> Dict[str, any]:
        """Get comprehensive update summary"""
        summary = {
            "packages": None,
            "script": None,
            "timestamp": datetime.now().isoformat(),
            "should_check": self.should_check_updates()
        }
        
        if summary["should_check"]:
            summary["packages"] = self.check_package_updates()
            summary["script"] = self.check_script_version()
            self.record_check()
        
        return summary
    
    def generate_update_report(self) -> str:
        """Generate human-readable update report"""
        summary = self.get_update_summary()
        report = "📦 Lea Update Check Report\n"
        report += "=" * 50 + "\n\n"
        
        if not summary["should_check"]:
            report += "⏭️  Skipping check (checked recently)\n"
            report += "Next check will run automatically.\n"
            return report
        
        # Package updates
        if summary["packages"]:
            pkg_result = summary["packages"]
            report += "📚 Package Updates:\n"
            if pkg_result["success"]:
                if pkg_result["outdated"]:
                    report += f"⚠️  Found {len(pkg_result['outdated'])} outdated package(s):\n\n"
                    for pkg in pkg_result["outdated"]:
                        report += f"  • {pkg['name']}\n"
                        report += f"    Current: {pkg['version']}\n"
                        report += f"    Latest:  {pkg['latest_version']}\n\n"
                    report += "💡 To update all packages, run:\n"
                    report += "   pip install --upgrade -r requirements.txt\n"
                else:
                    report += "✅ All packages are up to date!\n\n"
            else:
                report += f"❌ Error checking packages: {pkg_result.get('error', 'Unknown error')}\n\n"
        
        # Script updates
        if summary["script"]:
            script_result = summary["script"]
            report += "📝 Script Updates:\n"
            if script_result["success"]:
                if script_result["update_available"]:
                    report += f"🆕 Update available!\n"
                    report += f"   Current version: {script_result['current_version']}\n"
                    report += f"   Latest version: {script_result['latest_version']}\n"
                    if script_result["release_url"]:
                        report += f"   Download: {script_result['release_url']}\n"
                else:
                    report += f"✅ Script is up to date (v{script_result['current_version']})\n"
            else:
                if "GitHub repository not configured" not in str(script_result.get("error", "")):
                    report += f"⚠️  Could not check for script updates: {script_result.get('error', 'Unknown error')}\n"
                else:
                    report += "ℹ️  Script version checking not configured (GitHub repo not set)\n"
        
        report += "\n" + "=" * 50 + "\n"
        report += f"Checked: {summary['timestamp']}\n"
        
        return report

def check_and_notify(show_dialog: bool = True):
    """Check for updates and optionally show notification"""
    checker = UpdateChecker()
    
    if not checker.should_check_updates():
        return None
    
    summary = checker.get_update_summary()
    
    # Generate report
    report = checker.generate_update_report()
    
    # Log to file
    logging.info(f"Update check completed:\n{report}")
    
    # Show dialog if requested
    if show_dialog:
        try:
            from PyQt6.QtWidgets import QMessageBox, QApplication
            from PyQt6.QtCore import Qt
            
            # Check if we have updates
            has_updates = False
            update_text = ""
            
            if summary["packages"] and summary["packages"].get("outdated"):
                has_updates = True
                outdated_count = len(summary["packages"]["outdated"])
                update_text += f"📦 {outdated_count} package(s) need updating\n"
            
            if summary["script"] and summary["script"].get("update_available"):
                has_updates = True
                update_text += f"📝 Script update available\n"
            
            if has_updates:
                msg = QMessageBox()
                msg.setWindowTitle("🔄 Updates Available")
                msg.setText("Updates are available for Lea!")
                msg.setInformativeText(update_text + "\nSee update report for details.")
                msg.setDetailedText(report)
                msg.setStandardButtons(QMessageBox.StandardButton.Ok)
                msg.exec()
            else:
                # Optional: Show "all up to date" message (comment out if too frequent)
                # msg = QMessageBox()
                # msg.setWindowTitle("✅ Up to Date")
                # msg.setText("All packages are up to date!")
                # msg.exec()
                pass
        
        except Exception as e:
            logging.error(f"Error showing update dialog: {e}")
    
    return summary

if __name__ == "__main__":
    # Run update check from command line
    print(check_and_notify(show_dialog=False))

