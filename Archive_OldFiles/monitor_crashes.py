"""
Crash Monitor - Monitors the program for crashes and issues
Run this alongside the main program to catch problems
"""

import sys
import os
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

class CrashMonitor:
    def __init__(self):
        self.program_path = Path(__file__).parent / "Lea_Visual_Code_v2.5.1a_ TTS.py"
        self.log_file = Path(__file__).parent / "crash_monitor.log"
        self.process = None
        self.start_time = None
        self.crash_count = 0
        self.restart_count = 0
        self.max_restarts = 5
        
    def log(self, message, level="INFO"):
        """Log a message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except:
            pass
        
        print(log_entry.strip())
    
    def start_program(self):
        """Start the monitored program"""
        if not self.program_path.exists():
            self.log(f"Program not found: {self.program_path}", "ERROR")
            return False
        
        try:
            self.log(f"Starting program: {self.program_path.name}")
            self.process = subprocess.Popen(
                [sys.executable, str(self.program_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.start_time = time.time()
            self.log(f"Program started (PID: {self.process.pid})")
            return True
        except Exception as e:
            self.log(f"Failed to start program: {e}", "ERROR")
            return False
    
    def monitor(self):
        """Monitor the program for crashes"""
        if not self.start_program():
            return
        
        self.log("Crash monitor started. Monitoring program...")
        self.log("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process has exited
                    return_code = self.process.returncode
                    elapsed = time.time() - self.start_time if self.start_time else 0
                    
                    # Get output
                    stdout, stderr = self.process.communicate()
                    
                    if return_code == 0:
                        self.log(f"Program exited normally after {elapsed:.1f}s")
                        break
                    else:
                        self.crash_count += 1
                        self.log(f"Program crashed! (Exit code: {return_code}, Runtime: {elapsed:.1f}s)", "ERROR")
                        
                        if stderr:
                            self.log(f"Error output:\n{stderr}", "ERROR")
                        
                        # Attempt restart if within limit
                        if self.restart_count < self.max_restarts:
                            self.restart_count += 1
                            self.log(f"Attempting restart {self.restart_count}/{self.max_restarts}")
                            time.sleep(2)  # Wait before restart
                            if not self.start_program():
                                self.log("Failed to restart program", "ERROR")
                                break
                        else:
                            self.log(f"Max restarts ({self.max_restarts}) reached. Stopping.", "ERROR")
                            break
                
                # Check every second
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.log("Monitor stopped by user")
            if self.process and self.process.poll() is None:
                self.log("Terminating program...")
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
        
        # Final summary
        self.log("\n" + "="*60)
        self.log("MONITORING SUMMARY")
        self.log("="*60)
        self.log(f"Total crashes detected: {self.crash_count}")
        self.log(f"Total restarts attempted: {self.restart_count}")
        if self.start_time:
            total_runtime = time.time() - self.start_time
            self.log(f"Total runtime: {total_runtime:.1f}s")

def main():
    """Run crash monitor"""
    print("="*60)
    print("LEA ASSISTANT CRASH MONITOR")
    print("="*60)
    print("This will monitor the program for crashes")
    print("="*60)
    
    monitor = CrashMonitor()
    monitor.monitor()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Monitor error: {traceback.format_exc()}")

