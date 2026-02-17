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
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "Install Python and re-run this script."
    if command -v brew >/dev/null 2>&1; then
      read -r -p "Install Python now with Homebrew? (Y/N): " INSTALL_PY
      if [[ "${INSTALL_PY^^}" == "Y" ]]; then
        brew install python
        echo "Python installation attempted. Please re-run this script after install completes."
      fi
    else
      echo "Homebrew not found. Install it from https://brew.sh and re-run this script."
      echo "Or download Python from https://www.python.org/downloads/"
    fi
  else
    echo "Install Python and re-run this script."
    echo "- macOS: brew install python"
    echo "- Windows: https://www.python.org/downloads/"
    echo "- Linux: sudo apt install python3"
  fi
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
# Always grab the latest yt-dlp (YouTube changes frequently)
python -m pip install --upgrade yt-dlp

RUST_EDITOR_DIR="tools/midi_touchup_editor_rust"
RUST_EDITOR_BIN="${RUST_EDITOR_DIR}/target/release/midi-touchup-editor"

if [[ -d "$RUST_EDITOR_DIR" ]]; then
  echo "Checking Rust MIDI Touch-Up Editor..."

  # rustup commonly installs cargo under ~/.cargo/bin without shell profile reload.
  if ! command -v cargo >/dev/null 2>&1 && [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  fi

  if command -v cargo >/dev/null 2>&1; then
    echo "Building Rust MIDI Touch-Up Editor..."
    if (cd "$RUST_EDITOR_DIR" && cargo build --release); then
      echo "Rust touch-up editor ready: $RUST_EDITOR_BIN"
    else
      echo
      echo "WARNING: Rust touch-up editor build failed."
      echo "The core app will run, but Edit MIDI touch-up may be unavailable until this succeeds."
      echo "Retry manually:"
      echo "  cd $RUST_EDITOR_DIR && cargo build --release"
      echo
    fi
  else
    echo
    echo "NOTE: Rust toolchain (cargo) was not found."
    echo "The core app will run, but MIDI Touch-Up Editor requires a Rust build."
    echo "Install Rust and re-run setup, or build manually:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    echo "  source \"$HOME/.cargo/env\""
    echo "  cd $RUST_EDITOR_DIR && cargo build --release"
    echo
  fi
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo
  echo "NOTE: FFmpeg was not found."
  echo "Some video workflows (YouTube downloads and video-to-frames conversion) need FFmpeg."
  echo "- macOS: brew install ffmpeg"
  echo "- Linux: sudo apt install ffmpeg"
  echo
fi

echo "Launching app..."
python run.py
