# Lea Assistant Testing Guide

## Quick Test Run

To run all tests, simply execute:
```bash
run_tests.bat
```

Or run tests individually:
```bash
python test_lea_stability.py
python test_lea_runtime.py
```

## Test Suites

### 1. Stability Tests (`test_lea_stability.py`)

Tests critical components for stability and crash prevention:

- **Import Verification**: Ensures all required modules can be imported
- **Model Assignments**: Verifies model mappings are correct
- **Thread Safety**: Tests PyQt threading patterns
- **File Operations**: Checks file reading capabilities
- **Error Handling**: Verifies exception handling works
- **Concurrent Operations**: Simulates multiple simultaneous operations
- **Memory Management**: Tests history limiting and data structures
- **Signal/Slot Connections**: Tests PyQt signal system
- **Pressure Test**: Runs 50 rapid operations to test stability

### 2. Runtime Tests (`test_lea_runtime.py`)

Tests the actual program structure:

- **Program Import**: Checks if main file can be loaded
- **Configuration**: Verifies .env and directory structure
- **Dependencies**: Checks if required packages are installed
- **Critical Functions**: Verifies key functions and classes exist

## What the Tests Check

### Crash Prevention
- ✅ Thread cleanup with timeouts
- ✅ Signal disconnection before deletion
- ✅ Proper error handling
- ✅ No blocking operations in main thread
- ✅ Memory management (history limiting)

### Stability
- ✅ Concurrent operation handling
- ✅ Rapid operation stress testing
- ✅ Resource cleanup
- ✅ Exception recovery

### Functionality
- ✅ Model assignments per mode
- ✅ File upload capabilities
- ✅ Thread safety patterns

## Interpreting Results

### ✅ All Tests Pass
The program appears stable and ready to use. All critical components are working correctly.

### ⚠️ Warnings
Some optional features may not be available, but core functionality should work.

### ❌ Failures
Review the specific failures. Common issues:
- Missing dependencies (install with `pip install`)
- Configuration issues (check .env file)
- Import errors (check Python path)

## Manual Testing Checklist

After automated tests pass, manually test:

1. **Startup**: Launch the program - does it start without errors?
2. **Mode Switching**: Switch between different agent modes
3. **Model Selection**: Verify correct models are assigned per mode
4. **File Upload**: Try uploading a text file
5. **Image Upload**: Try uploading an image
6. **Send Message**: Send a message and wait for response
7. **Rapid Operations**: Send multiple messages quickly
8. **Thread Cleanup**: Close program - does it exit cleanly?

## Common Issues and Fixes

### Import Errors
```bash
pip install PyQt6 openai pillow pygame
```

### Missing .env File
Create `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

### Thread Errors
- Ensure all signals are disconnected before thread deletion
- Use timeouts on thread.wait() calls
- Check for proper cleanup in error handlers

## Performance Benchmarks

Expected performance:
- **Startup**: < 3 seconds
- **Message Send**: < 5 seconds (depends on API)
- **File Upload**: < 2 seconds for small files
- **Mode Switch**: < 1 second

If performance is significantly worse, check:
- Network connection
- API response times
- System resources

