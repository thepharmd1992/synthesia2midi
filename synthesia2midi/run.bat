@echo off
title Synthesia2MIDI
echo Starting Synthesia2MIDI...
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check if we're in Git Bash and open a new cmd window if so
if defined MSYSTEM (
    start cmd /c "%~f0"
    exit
)

REM Prefer the repo virtual environment if present.
set "VENV_PY=%SCRIPT_DIR%..\.venv\Scripts\python.exe"
if exist "%VENV_PY%" (
    "%VENV_PY%" run.py
    goto :done
)

REM Try py command first (only if it actually runs), then python
py -V >nul 2>&1
if %errorlevel% == 0 (
    py run.py
    goto :done
)

python -V >nul 2>&1
if %errorlevel% == 0 (
    python run.py
    goto :done
)

python3 -V >nul 2>&1
if %errorlevel% == 0 (
    python3 run.py
    goto :done
)

echo Error: Python not found.
echo Run setup_windows.bat from the repo root, or install Python 3 and add it to PATH.
pause
exit /b 1

:done
pause
