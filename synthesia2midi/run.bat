@echo off
title Synthesia2MIDI
echo Starting Synthesia2MIDI...
cd /d "%~dp0"

REM Check if we're in Git Bash and open a new cmd window if so
if defined MSYSTEM (
    start cmd /c "%~f0"
    exit
)

REM Try py command first, then python
where py >nul 2>&1
if %errorlevel% == 0 (
    py run.py
) else (
    where python >nul 2>&1
    if %errorlevel% == 0 (
        python run.py
    ) else (
        echo Error: Python not found. Please install Python and add it to PATH.
        pause
        exit /b 1
    )
)

pause