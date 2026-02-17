#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "== Synthesia2MIDI setup =="
echo "Starting the guided installer..."
echo

fail() {
  echo
  echo "Setup failed."
  echo "Read the message above, fix the problem, then run this again."
  read -r -p "Press Enter to close..."
  exit 1
}

BOOT_LOG="$ROOT_DIR/logs/installer_bootstrap.log"
mkdir -p "$ROOT_DIR/logs"
{
  echo "== bootstrap started at $(date) =="
} >> "$BOOT_LOG"

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
  echo "[1/3] Creating Python environment..."
  "$PYTHON_BIN" -m venv .venv || fail
else
  echo "[1/3] Python environment already exists."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[2/3] Installing installer UI (textual)."
echo "      First run can take several minutes while pip resolves/downloads packages."
echo "      Live output is shown below and also logged to: $BOOT_LOG"
python -m pip install --disable-pip-version-check --upgrade textual 2>&1 | tee -a "$BOOT_LOG" || fail

echo "[3/3] Launching guided installer UI..."
python -u "installer/tui_installer.py" || fail
