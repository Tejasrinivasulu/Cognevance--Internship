@echo off
cd /d "%~dp0"

where py >nul 2>&1
if %errorlevel%==0 (
    set PY=py
    goto :run
)

where python >nul 2>&1
if %errorlevel%==0 (
    set PY=python
    goto :run
)

echo Python not found. Install Python 3 and add it to PATH.
pause
exit /b 1

:run
%PY% -m notebook notebooks\Customer_Churn_Prediction.ipynb
