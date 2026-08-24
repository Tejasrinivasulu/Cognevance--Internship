@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo.
echo ========================================
echo  Customer Churn Prediction - Running
echo ========================================
echo.

set "PY="

REM 1) Prefer known local Python installs
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" (
    set "PY=%LocalAppData%\Programs\Python\Python312\python.exe"
    goto :found
)
if exist "%LocalAppData%\Programs\Python\Python313\python.exe" (
    set "PY=%LocalAppData%\Programs\Python\Python313\python.exe"
    goto :found
)
if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PY=%LocalAppData%\Programs\Python\Python311\python.exe"
    goto :found
)

REM 2) Try py launcher
where py >nul 2>&1
if %errorlevel%==0 (
    set "PY=py"
    goto :found
)

REM 3) Try python on PATH (skip Microsoft Store stub if broken)
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "delims=" %%i in ('where python') do (
        echo %%i | find /i "WindowsApps" >nul
        if errorlevel 1 (
            set "PY=%%i"
            goto :found
        )
    )
)

echo.
echo ERROR: Python was not found.
echo.
echo Fix:
echo  1. Install Python from https://www.python.org/downloads/
echo  2. CHECK "Add python.exe to PATH"
echo  3. Open a NEW Command Prompt and try again
echo.
pause
exit /b 1

:found
echo Using Python: %PY%
echo.

echo [1/2] Installing packages...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo [2/2] Running pipeline...
"%PY%" src\run_pipeline.py
if errorlevel 1 (
    echo.
    echo ERROR: pipeline failed.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  DONE - check model\ and images\
echo ========================================
echo.
pause
