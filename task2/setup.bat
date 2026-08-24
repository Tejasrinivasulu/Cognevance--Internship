@echo off
cd /d "%~dp0"
echo Installing packages...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Failed. Install Python 3.10+ and check "Add Python to PATH".
    pause
    exit /b 1
)
echo.
echo Setup done!
pause
