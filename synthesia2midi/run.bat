@echo off
title Synthesia2MIDI
echo Starting Synthesia2MIDI...
cd /d "%~dp0"

REM Check if we're in Git Bash and open a new cmd window if so
if defined MSYSTEM (
    start cmd /c "%~f0"
    exit
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

echo Error: Python not found. Please install Python and add it to PATH.
pause
exit /b 1

:done
pause
