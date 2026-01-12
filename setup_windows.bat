@echo off
setlocal

REM If double-clicked, the console can close too quickly to read errors.
REM Relaunch ourselves in a persistent window.
if "%~1"=="" (
  cmd /k "%~f0" launched
  exit /b 0
)

REM Use pushd to support UNC paths (including \\wsl.localhost\...).
pushd "%~dp0" >nul 2>&1
if %errorlevel% neq 0 (
  echo ERROR: Could not switch to the repository directory:
  echo   %~dp0
  echo.
  echo If this is a \\wsl.localhost\\... path, Windows CMD cannot use it as a working directory.
  echo Options:
  echo  1 - Copy or clone the repo into a normal Windows folder, for example C:\\Users\\%USERNAME%\\midi, then run this again, OR
  echo  2 - Run the Linux setup inside WSL: bash setup.sh
  echo.
  pause
  exit /b 1
)

echo == Synthesia2MIDI setup ==

REM Prefer the Python launcher on Windows
where py >nul 2>&1
if %errorlevel%==0 (
  set PY=py -3
) else (
  where python >nul 2>&1
  if %errorlevel%==0 (
    set PY=python
  ) else (
    echo ERROR: Python was not found.
    echo This app needs Python 3.
    echo.
    set /p INSTALL_PY=Install Python now? (Y/N): 
    if /I "%INSTALL_PY%"=="Y" (
      where winget >nul 2>&1
      if %errorlevel%==0 (
        echo Installing Python with winget...
        winget install --id Python.Python.3 --scope user --accept-source-agreements --accept-package-agreements
        echo.
        echo Python installation attempted. Please re-run this script after install completes.
        pause
      ) else (
        echo Winget not found. Opening the Python download page...
        start "" "https://www.python.org/downloads/"
        echo After installing, re-run this script.
        pause
      )
    ) else (
      echo Install Python from https://www.python.org/downloads/ and re-run this script.
      echo During install, enable: Add python.exe to PATH.
      pause
    )
    popd
    exit /b 1
  )
)

if not exist ".venv" (
  echo Creating virtual environment (.venv)...
  %PY% -m venv .venv
  if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    pause
    popd
    exit /b 1
  )
)

echo Installing Python dependencies...
".venv\\Scripts\\python.exe" -m pip install --upgrade pip
if %errorlevel% neq 0 (
  echo ERROR: pip upgrade failed.
  pause
  popd
  exit /b 1
)

".venv\\Scripts\\python.exe" -m pip install -r "synthesia2midi\\requirements.txt"
if %errorlevel% neq 0 (
  echo ERROR: dependency install failed.
  pause
  popd
  exit /b 1
)

where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
  echo.
  echo NOTE: FFmpeg was not found.
  echo Some video workflows like YouTube downloads and video-to-frames conversion need FFmpeg.
  echo Download: https://ffmpeg.org/download.html
  echo.
)

echo Launching app...
".venv\\Scripts\\python.exe" "synthesia2midi\\run.py"

echo.
echo Done.
pause
popd
