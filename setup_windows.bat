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

REM Prefer the Python launcher on Windows, but verify it actually runs.
set "PY_CMD="
set "PY_LAUNCHER="
for /f "delims=" %%I in ('where py 2^>nul') do (
  if not defined PY_LAUNCHER set "PY_LAUNCHER=%%I"
)
if not defined PY_LAUNCHER if exist "%LOCALAPPDATA%\Programs\Python\Launcher\py.exe" (
  set "PY_LAUNCHER=%LOCALAPPDATA%\Programs\Python\Launcher\py.exe"
)
if defined PY_LAUNCHER (
  "%PY_LAUNCHER%" -3 -c "import sys" >nul 2>&1
  if not errorlevel 1 set "PY_CMD="%PY_LAUNCHER%" -3"
)

if not defined PY_CMD (
  set "PY_PATH="
  for /f "delims=" %%I in ('where python 2^>nul') do (
    if not defined PY_PATH set "PY_PATH=%%I"
  )
  if not defined PY_PATH if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PY_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
  )
  if defined PY_PATH (
    "%PY_PATH%" -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY_CMD="%PY_PATH%""
  )
)

if not defined PY_CMD (
  echo ERROR: Python was not found.
  echo This app needs Python 3.
  echo.
  set /p INSTALL_PY=Install Python now? Y or N:
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

set "PY=%PY_CMD%"

if not exist ".venv" (
  echo Creating virtual environment at .venv...
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
