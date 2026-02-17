@echo off
setlocal
set "PAUSE_ON_ERROR=1"

REM If double-clicked, the console can close too quickly to read errors.
REM Relaunch ourselves in a persistent window.
if "%~1"=="" (
  cmd /k "%~f0" launched
  exit /b 0
)
if /I "%~1"=="launched" set "PAUSE_ON_ERROR=0"

REM Handle running from inside a zip file opened in Explorer.
set "SCRIPT_DIR=%~dp0"
echo %SCRIPT_DIR% | findstr /I ".zip" >nul
if %errorlevel%==0 (
  echo ERROR: This setup is running from inside a zip archive.
  echo Please extract the zip to a normal folder, then run setup_windows.bat again.
  call :maybe_pause
  exit /b 1
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
  call :maybe_pause
  exit /b 1
)

echo == Synthesia2MIDI setup ==

if not exist "synthesia2midi\\requirements.txt" (
  echo ERROR: synthesia2midi\\requirements.txt not found.
  echo Make sure the repo is fully extracted and run this from the repo root.
  call :maybe_pause
  popd
  exit /b 1
)

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
      call :maybe_pause
    ) else (
      echo Winget not found. Opening the Python download page...
      start "" "https://www.python.org/downloads/"
      echo After installing, re-run this script.
      call :maybe_pause
    )
  ) else (
    echo Install Python from https://www.python.org/downloads/ and re-run this script.
    echo During install, enable: Add python.exe to PATH.
    call :maybe_pause
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
    call :maybe_pause
    popd
    exit /b 1
  )
)

echo Installing Python dependencies...
set "PIP_CACHE_DIR=%CD%\\.pip-cache"
".venv\\Scripts\\python.exe" -m pip install --upgrade pip
if errorlevel 1 (
  echo ERROR: pip upgrade failed.
  call :maybe_pause
  popd
  exit /b 1
)

".venv\\Scripts\\python.exe" -m pip install -r "synthesia2midi\\requirements.txt"
if errorlevel 1 (
  echo ERROR: dependency install failed.
  call :maybe_pause
  popd
  exit /b 1
)

call :ensure_ffmpeg
if errorlevel 1 (
  popd
  exit /b 1
)

call :ensure_rust_touchup
if errorlevel 1 (
  echo.
  echo ERROR: Rust touch-up editor could not be installed or built.
  echo Please resolve the error above, then re-run setup_windows.bat.
  echo.
  call :maybe_pause
  popd
  exit /b 1
)

echo Launching app...
".venv\\Scripts\\python.exe" "run.py"

echo.
echo Done.
pause
popd

goto :eof

:ensure_ffmpeg
set "FFMPEG_LOCAL=%CD%\\synthesia2midi\\ffmpeg\\ffmpeg.exe"
where ffmpeg >nul 2>&1
if %errorlevel%==0 exit /b 0
if exist "%FFMPEG_LOCAL%" exit /b 0

echo.
echo FFmpeg not found. Attempting automatic install...
set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
set "FFMPEG_ZIP=%TEMP%\\ffmpeg-release-essentials.zip"
set "FFMPEG_TMP=%TEMP%\\ffmpeg_extract"
set "FFMPEG_DIR=%CD%\\synthesia2midi\\ffmpeg"

where powershell >nul 2>&1
if errorlevel 1 (
  echo ERROR: PowerShell was not found; cannot auto-install FFmpeg.
  goto :ffmpeg_fail
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $zip='%FFMPEG_ZIP%'; $dest='%FFMPEG_TMP%'; $url='%FFMPEG_URL%'; Invoke-WebRequest -Uri $url -OutFile $zip; if(Test-Path $dest){Remove-Item $dest -Recurse -Force}; Expand-Archive -Path $zip -DestinationPath $dest -Force; $ff=Get-ChildItem -Path $dest -Filter ffmpeg.exe -Recurse ^| Select-Object -First 1; if(-not $ff){throw 'ffmpeg.exe not found in archive'}; New-Item -ItemType Directory -Force -Path '%FFMPEG_DIR%' ^| Out-Null; Copy-Item -Path $ff.FullName -Destination '%FFMPEG_DIR%\\ffmpeg.exe' -Force"
if errorlevel 1 goto :ffmpeg_fail

if exist "%FFMPEG_LOCAL%" (
  echo FFmpeg installed: %FFMPEG_LOCAL%
  exit /b 0
)

:ffmpeg_fail
echo.
echo ERROR: FFmpeg auto-install failed.
echo Please install FFmpeg manually:
echo   1) Open your web browser and go to:
echo      https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
echo   2) Download the zip file.
echo   3) Open the zip, open the "bin" folder, and copy "ffmpeg.exe".
echo   4) Create this folder if it does not exist:
echo      %CD%\\synthesia2midi\\ffmpeg
echo   5) Paste "ffmpeg.exe" into that folder:
echo      %FFMPEG_LOCAL%
echo   6) Run setup_windows.bat again.
call :maybe_pause
exit /b 1

:ensure_rust_touchup
set "RUST_EDITOR_DIR=%CD%\\tools\\midi_touchup_editor_rust"
set "RUST_EDITOR_BIN=%RUST_EDITOR_DIR%\\target\\release\\midi-touchup-editor.exe"

if not exist "%RUST_EDITOR_DIR%" (
  exit /b 0
)

if exist "%RUST_EDITOR_BIN%" (
  echo Rust touch-up editor already present.
  exit /b 0
)

call :detect_cargo

if not defined CARGO_EXE (
  echo.
  echo Rust toolchain ^(cargo^) was not found.
  echo MIDI Touch-Up Editor now requires Rust.
  echo.
  set "NEED_RUSTUP=0"
  where winget >nul 2>&1
  if errorlevel 1 (
    echo Winget was not found. Will try Rust direct download.
    set "NEED_RUSTUP=1"
  ) else (
    echo Installing Rust with winget...
    winget install --id Rustlang.Rustup -e --scope user --accept-source-agreements --accept-package-agreements
    if errorlevel 1 set "NEED_RUSTUP=1"
  )

  call :detect_cargo
  if not defined CARGO_EXE set "NEED_RUSTUP=1"

  call :maybe_install_rustup
  if errorlevel 1 goto :rust_install_failed
)

if not defined CARGO_EXE (
  echo.
  echo Rust toolchain ^(cargo^) is still unavailable.
  echo Please install Rust manually:
  echo   1^) Click Start, type "Command Prompt", and press Enter.
  echo   2^) Copy and paste this command, then press Enter:
  echo      winget install --id Rustlang.Rustup -e
  echo   3^) If that fails, open this page and run the installer:
  echo      https://www.rust-lang.org/tools/install
  echo   4^) After it finishes, run setup_windows.bat again.
  echo.
  call :maybe_pause
  exit /b 1
)

echo Building Rust MIDI Touch-Up Editor...
pushd "%RUST_EDITOR_DIR%" >nul 2>&1
if %errorlevel% neq 0 (
  echo ERROR: Could not open Rust editor directory:
  echo   %RUST_EDITOR_DIR%
  call :maybe_pause
  exit /b 1
)

"%CARGO_EXE%" build --release
if %errorlevel% neq 0 (
  echo ERROR: Rust build failed.
  echo Retry manually:
  echo   cd tools\midi_touchup_editor_rust
  echo   cargo build --release
  popd
  call :maybe_pause
  exit /b 1
)
popd

if exist "%RUST_EDITOR_BIN%" (
  echo Rust touch-up editor ready: %RUST_EDITOR_BIN%
  exit /b 0
)

echo ERROR: Rust build completed but binary was not found:
echo   %RUST_EDITOR_BIN%
call :maybe_pause
exit /b 1

:rust_install_failed
echo.
echo ERROR: Rust auto-install failed.
echo Please install Rust manually:
echo   1^) Click Start, type "Command Prompt", and press Enter.
echo   2^) Copy and paste this command, then press Enter:
echo      winget install --id Rustlang.Rustup -e
echo   3^) If that fails, open this page and run the installer:
echo      https://www.rust-lang.org/tools/install
echo   4^) After it finishes, run setup_windows.bat again.
call :maybe_pause
exit /b 1

:maybe_install_rustup
if "%NEED_RUSTUP%"=="1" (
  echo Installing Rust via direct download...
  call :install_rust_with_rustup
  if errorlevel 1 exit /b 1
  call :detect_cargo
)
exit /b 0

:detect_cargo
set "CARGO_EXE="
for /f "delims=" %%I in ('where cargo 2^>nul') do (
  if not defined CARGO_EXE set "CARGO_EXE=%%I"
)
if not defined CARGO_EXE if exist "%USERPROFILE%\.cargo\bin\cargo.exe" (
  set "CARGO_EXE=%USERPROFILE%\.cargo\bin\cargo.exe"
)
exit /b 0

:install_rust_with_rustup
set "RUSTUP_URL=https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe"
if /I "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "RUSTUP_URL=https://static.rust-lang.org/rustup/dist/aarch64-pc-windows-msvc/rustup-init.exe"
if /I "%PROCESSOR_ARCHITECTURE%"=="x86" set "RUSTUP_URL=https://static.rust-lang.org/rustup/dist/i686-pc-windows-msvc/rustup-init.exe"
set "RUSTUP_EXE=%TEMP%\\rustup-init.exe"

where powershell >nul 2>&1
if errorlevel 1 (
  echo ERROR: PowerShell was not found; cannot auto-install Rust.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; Invoke-WebRequest -Uri '%RUSTUP_URL%' -OutFile '%RUSTUP_EXE%'"
if errorlevel 1 exit /b 1

"%RUSTUP_EXE%" -y --profile minimal
if errorlevel 1 exit /b 1

exit /b 0

:maybe_pause
if "%PAUSE_ON_ERROR%"=="1" (
  echo.
  echo Press any key to close this window...
  pause >nul
)
exit /b 0
