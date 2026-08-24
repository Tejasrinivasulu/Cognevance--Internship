@echo off
cd /d "%~dp0"
echo Running inference...
python inference.py --image outputs\sample_sneaker.png --model outputs\best_model.pt
if errorlevel 1 (
    echo.
    echo Failed. Run train.bat first.
    pause
    exit /b 1
)
pause
