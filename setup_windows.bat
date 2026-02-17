@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM If double-clicked, keep the window open
if "%~1"=="" (
  cmd /k "%~f0" launched
  exit /b 0
)

if /I "%~1"=="launched" (
  echo == Synthesia2MIDI setup ==
  echo Starting the guided installer...
  echo.
) else (
  echo == Synthesia2MIDI setup ==
)

call :find_python
if errorlevel 1 exit /b 1

if not exist "logs" mkdir "logs"
set "BOOT_LOG=%SCRIPT_DIR%logs\\installer_bootstrap.log"
echo == bootstrap started at %DATE% %TIME% ==>> "%BOOT_LOG%"

if not exist ".venv" (
  echo [1/3] Creating Python environment...
  "%PY_EXE%" %PY_ARGS% -m venv .venv
  if errorlevel 1 (
    echo.
    echo ERROR: Could not create the Python environment.
    echo Please reinstall Python and try again.
    pause
    exit /b 1
  )
 ) else (
  echo [1/3] Python environment already exists.
)

echo [2/3] Installing installer UI ^(textual^).
echo       First run can take several minutes while pip resolves/downloads packages.
echo       Live output is shown below.
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade textual
if errorlevel 1 (
  echo.
  echo ERROR: Could not install the installer UI.
  echo See the log for details: %BOOT_LOG%
  pause
  exit /b 1
)

echo [3/3] Launching guided installer UI...
".venv\Scripts\python.exe" -u "installer\tui_installer.py"

echo.
echo Installer finished.
pause
exit /b 0

:find_python
call :find_python_internal
if defined PY_EXE exit /b 0

echo Python not found. Installing Python automatically...
call :install_python_auto
if errorlevel 1 goto :python_manual

call :find_python_internal
if defined PY_EXE exit /b 0

:python_manual
echo.
echo Python could not be installed automatically.
echo Please install Python 3 manually:
echo 1^) Open your browser and go to python.org/downloads
echo 2^) Click "Download Python 3.x"
echo 3^) Run the installer and check "Add python.exe to PATH"
echo 4^) Click "Install Now" and wait for it to finish
echo 5^) Run setup_windows.bat again
pause
exit /b 1

:find_python_internal
set "PY_EXE="
set "PY_ARGS="

for /f "delims=" %%I in ('where py 2^>nul') do (
  if not defined PY_EXE (
    set "PY_EXE=%%I"
    set "PY_ARGS=-3"
  )
)

if not defined PY_EXE (
  for /f "delims=" %%I in ('where python 2^>nul') do (
    if not defined PY_EXE set "PY_EXE=%%I"
  )
)

if not defined PY_EXE if exist "%LOCALAPPDATA%\Programs\Python\Launcher\py.exe" (
  set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Launcher\py.exe"
  set "PY_ARGS=-3"
)

if not defined PY_EXE (
  for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if not defined PY_EXE if exist "%%D\python.exe" set "PY_EXE=%%D\python.exe"
  )
)

call :validate_python
exit /b 0

:validate_python
if not defined PY_EXE exit /b 0
"%PY_EXE%" %PY_ARGS% -c "import sys" >nul 2>&1
if errorlevel 1 (
  set "PY_EXE="
  set "PY_ARGS="
)
exit /b 0

:install_python_auto
where winget >nul 2>&1
if errorlevel 1 goto :install_python_direct

echo Trying to install Python with winget...
winget install --id Python.Python.3 --scope user --accept-source-agreements --accept-package-agreements
if errorlevel 1 (
  echo Winget install failed. Trying a direct download instead...
  goto :install_python_direct
)
call :find_python_internal
if defined PY_EXE exit /b 0
echo Winget completed but Python was not found. Trying a direct download instead...
goto :install_python_direct

:install_python_direct
where powershell >nul 2>&1
if errorlevel 1 exit /b 1

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; $base='https://www.python.org/ftp/python/'; $content=(Invoke-WebRequest -UseBasicParsing -Uri $base).Content; $versions=[regex]::Matches($content, 'href=.*?(\d+\.\d+\.\d+)/') | ForEach-Object { $_.Groups[1].Value } | Where-Object { $_ -match '^\d+\.\d+\.\d+$' } | Sort-Object {[version]$_} -Descending; if(-not $versions){ throw 'No Python versions found'; }; $suffix='-amd64'; if($env:PROCESSOR_ARCHITECTURE -eq 'ARM64'){ $suffix='-arm64' } elseif($env:PROCESSOR_ARCHITECTURE -eq 'x86'){ $suffix='' }; $selected=$null; foreach($v in $versions){ $file='python-' + $v + $suffix + '.exe'; $url=$base + $v + '/' + $file; try { Invoke-WebRequest -UseBasicParsing -Method Head -Uri $url | Out-Null; $selected=$url; break } catch {} }; if(-not $selected){ throw 'No suitable Python installer found'; }; $out=$env:TEMP + '\python-installer.exe'; Invoke-WebRequest -UseBasicParsing -Uri $selected -OutFile $out; Start-Process -FilePath $out -ArgumentList '/quiet InstallAllUsers=0 PrependPath=1 Include_test=0' -Wait"
if errorlevel 1 exit /b 1
exit /b 0
