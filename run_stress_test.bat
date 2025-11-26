@echo off
REM Run enhanced stress test

echo ========================================
echo LEA ASSISTANT ENHANCED STRESS TEST
echo ========================================
echo.
echo This will run comprehensive stress tests
echo to identify potential issues.
echo.
pause

cd /d "F:\Dre_Programs\LeaAssistant"
python stress_test_enhanced.py

echo.
echo ========================================
echo Stress test completed!
echo Check stress_test_report.txt for details
echo ========================================
pause

