@echo off
REM Batch script to run all tests for Lea Assistant

echo ========================================
echo LEA ASSISTANT TEST SUITE
echo ========================================
echo.

echo Running Stability Tests...
echo.
python test_lea_stability.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Stability tests had failures!
    pause
    exit /b 1
)

echo.
echo ========================================
echo.

echo Running Runtime Tests...
echo.
python test_lea_runtime.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Runtime tests had issues!
    pause
    exit /b 1
)

echo.
echo ========================================
echo All tests completed!
echo ========================================
pause

