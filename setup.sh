#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "== Synthesia2MIDI setup =="

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python 3 was not found."
  echo "Install Python and re-run this script."
  echo "- macOS: brew install python"
  echo "- Windows: https://www.python.org/downloads/"
  echo "- Linux: sudo apt install python3"
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment at .venv..."
  if ! "$PYTHON_BIN" -m venv .venv; then
    echo "ERROR: Failed to create virtual environment."
    echo "On Linux you may need: sudo apt install python3-venv"
    exit 1
  fi
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -r synthesia2midi/requirements.txt

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo
  echo "NOTE: FFmpeg was not found."
  echo "Some video workflows (YouTube downloads and video-to-frames conversion) need FFmpeg."
  echo "- macOS: brew install ffmpeg"
  echo "- Linux: sudo apt install ffmpeg"
  echo
fi

echo "Launching app..."
python synthesia2midi/run.py
