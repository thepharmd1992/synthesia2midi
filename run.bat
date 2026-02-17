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
set "VENV_PY=%SCRIPT_DIR%\.venv\Scripts\python.exe"
set "RUST_EDITOR_DIR=%SCRIPT_DIR%\tools\midi_touchup_editor_rust"
set "RUST_EDITOR_BIN=%RUST_EDITOR_DIR%\target\release\midi-touchup-editor.exe"
set "NEED_SETUP=0"

call :check_deps

if "%NEED_SETUP%"=="1" (
    echo Missing dependencies: %MISSING_REASON%
    echo Running setup_windows.bat...
    call "%SCRIPT_DIR%\setup_windows.bat" launched
    if errorlevel 1 (
        echo Setup failed. See messages above.
        pause
        exit /b %errorlevel%
    )
    call :check_deps
    if "%NEED_SETUP%"=="1" (
        echo Setup completed but dependencies are still missing: %MISSING_REASON%
        echo Fix the issue above, then run setup_windows.bat again.
        pause
        exit /b 1
    )
)

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
exit /b 0

:check_deps
set "NEED_SETUP=0"
set "MISSING_REASON="
if not exist "%VENV_PY%" call :add_missing "Python venv"
if exist "%RUST_EDITOR_DIR%" if not exist "%RUST_EDITOR_BIN%" call :add_missing "Rust touch-up editor"
set "FFMPEG_OK=0"
where ffmpeg >nul 2>&1 && set "FFMPEG_OK=1"
if exist "%SCRIPT_DIR%synthesia2midi\ffmpeg\ffmpeg.exe" set "FFMPEG_OK=1"
if "%FFMPEG_OK%"=="0" call :add_missing "FFmpeg"
exit /b 0

:add_missing
if defined MISSING_REASON (
    set "MISSING_REASON=%MISSING_REASON%, %~1"
) else (
    set "MISSING_REASON=%~1"
)
set "NEED_SETUP=1"
exit /b 0
