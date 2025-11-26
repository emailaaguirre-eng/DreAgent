"""
Simple runner for stress test - double-click to run
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    stress_test = script_dir / "stress_test_enhanced.py"
    
    print("="*60)
    print("LEA ASSISTANT STRESS TEST RUNNER")
    print("="*60)
    print(f"\nRunning stress test from: {script_dir}")
    print("="*60 + "\n")
    
    try:
        # Run the stress test
        result = subprocess.run(
            [sys.executable, str(stress_test)],
            cwd=str(script_dir),
            capture_output=False
        )
        
        print("\n" + "="*60)
        print("STRESS TEST COMPLETED")
        print("="*60)
        print(f"\nExit code: {result.returncode}")
        print(f"Check 'stress_test_report.txt' for detailed results")
        print("\nPress Enter to exit...")
        input()
        
    except Exception as e:
        print(f"\n[ERROR] Failed to run stress test: {e}")
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)

