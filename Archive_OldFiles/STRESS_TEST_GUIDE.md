# Stress Testing Guide

## Overview

The stress tests help identify issues that might not appear during normal use. When you run them, they'll reveal problems that can then be fixed.

## Running Stress Tests

### Quick Start
```bash
run_stress_test.bat
```

Or run directly:
```bash
python stress_test_enhanced.py
```

## What the Stress Tests Check

### 1. Rapid Mode Switching
- Tests switching between modes 100 times rapidly
- **Issues to look for:**
  - Model assignment errors
  - Memory leaks from mode switching
  - UI freezing during switches

### 2. Concurrent File Operations
- Tests 20 simultaneous file operations
- **Issues to look for:**
  - File locking problems
  - Thread conflicts
  - Resource exhaustion

### 3. Message History Management
- Processes 1000 messages and tests history limiting
- **Issues to look for:**
  - Memory growth
  - History not being limited properly
  - Performance degradation

### 4. Thread Cleanup Under Stress
- Creates and cleans up 50 threads rapidly
- **Issues to look for:**
  - Thread leaks
  - Crashes during cleanup
  - Signal disconnection errors

### 5. Memory Leak Detection
- Creates many objects and checks cleanup
- **Issues to look for:**
  - Objects not being garbage collected
  - Memory growth over time

### 6. Error Recovery
- Tests how the program handles various errors
- **Issues to look for:**
  - Crashes on errors
  - Poor error messages
  - Incomplete error handling

### 7. API Call Simulation
- Simulates rapid API calls with failures
- **Issues to look for:**
  - Thread blocking
  - Poor retry logic
  - Resource leaks

## Interpreting Results

### ✅ All Tests Pass
Great! The program handles stress well. However, continue monitoring during actual use.

### ❌ Tests Fail
**This is valuable!** The failures tell you exactly what needs fixing:

1. **Check the error messages** - They'll tell you what operation failed
2. **Review stress_test_report.txt** - Contains detailed error information
3. **Look for patterns** - Multiple failures of the same type indicate a systemic issue

### Common Issues and Fixes

#### Issue: Thread Cleanup Failures
**Symptoms:** Thread cleanup test fails
**Fix:** 
- Ensure all signals are disconnected before `deleteLater()`
- Use timeouts on `thread.wait()`
- Check for proper cleanup in error handlers

#### Issue: Memory Leaks
**Symptoms:** Memory leak detection fails
**Fix:**
- Ensure large objects are cleared when not needed
- Check history limiting is working
- Verify file handles are closed

#### Issue: Mode Switching Errors
**Symptoms:** Rapid mode switching fails
**Fix:**
- Check `MODE_MODEL_DEFAULTS` has all modes
- Verify model assignment logic
- Ensure UI updates are thread-safe

#### Issue: Concurrent Operation Failures
**Symptoms:** File operations fail under load
**Fix:**
- Check for proper thread synchronization
- Ensure file locks are released
- Verify resource cleanup

## Using the Crash Monitor

To monitor the program while it runs:

```bash
python monitor_crashes.py
```

This will:
- Start the program
- Monitor for crashes
- Automatically restart on crash (up to 5 times)
- Log all crashes to `crash_monitor.log`

## Fixing Issues Found

### Step 1: Identify the Problem
- Read the error message carefully
- Check `stress_test_report.txt` for details
- Note which test failed

### Step 2: Locate the Code
- Search for the failing operation in the main code
- Check related functions and classes

### Step 3: Apply Fix
- Fix the specific issue
- Add error handling if missing
- Improve cleanup if needed

### Step 4: Re-test
- Run the stress test again
- Verify the fix works
- Check for new issues

## Example: Fixing a Thread Cleanup Issue

**Error from stress test:**
```
[ERROR] Thread cleanup 23: RuntimeError: wrapped C/C++ object has been deleted
```

**Fix:**
1. Find where threads are cleaned up
2. Ensure signals are disconnected before deletion
3. Add try/except around cleanup
4. Use `deleteLater()` instead of direct deletion

**Code fix:**
```python
# Before (problematic)
thread.quit()
thread.wait()  # Can cause issues
del thread

# After (fixed)
thread.quit()
if not thread.wait(1000):  # Timeout
    thread.terminate()
try:
    thread.finished.disconnect()  # Disconnect signals
except:
    pass
thread.deleteLater()  # Safe deletion
```

## Best Practices

1. **Run stress tests regularly** - Catch issues early
2. **Fix issues immediately** - Don't let them accumulate
3. **Test after fixes** - Verify the fix works
4. **Monitor in production** - Use crash monitor during actual use
5. **Keep logs** - Review `stress_test_report.txt` and `crash_monitor.log`

## Next Steps After Stress Testing

1. **Fix all failures** - Address each issue found
2. **Re-run tests** - Verify fixes work
3. **Manual testing** - Test the specific scenarios that failed
4. **Production monitoring** - Use crash monitor during real use

Remember: **Stress test failures are opportunities to improve!** Each failure tells you exactly what needs to be fixed.

