@echo off
cd /d "%~dp0"
echo Training model...
python train.py --epochs 3 --batch-size 32 --max-train-samples 3000
if errorlevel 1 (
    echo.
    echo Training failed. Run setup.bat first.
    pause
    exit /b 1
)
echo.
echo Training done! Check the outputs folder.
pause
