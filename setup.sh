#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "== Synthesia2MIDI setup =="
echo "Starting the guided installer..."

fail() {
  echo
  echo "Setup failed."
  echo "Read the message above, fix the problem, then run this again."
  read -r -p "Press Enter to close..."
  exit 1
}

BOOT_LOG="$ROOT_DIR/logs/installer_bootstrap.log"
mkdir -p "$ROOT_DIR/logs"

PYTHON_BIN="python3"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found. Installing Python automatically..."
  if command -v brew >/dev/null 2>&1; then
    brew install python || fail
  else
    echo "Homebrew is not installed."
    echo "Open https://brew.sh and follow the install steps."
    echo "Then run this setup again."
    fail
  fi
fi

if [[ ! -d ".venv" ]]; then
  echo "Creating Python environment..."
  "$PYTHON_BIN" -m venv .venv || fail
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing installer UI... (first time can take a few minutes)"
python -m pip install --disable-pip-version-check --upgrade textual >> "$BOOT_LOG" 2>&1 || fail

echo "Launching installer..."
python "installer/tui_installer.py" || fail
