#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

pause_on_error() {
  echo
  echo "Setup failed."
  echo "Read the message above, fix the problem, then run this again."
  read -r -p "Press Enter to close..."
}

trap 'pause_on_error' ERR

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
  pause_on_error
  exit 1
fi

if [[ ! -d ".venv" ]]; then
  echo "Creating virtual environment at .venv..."
  if ! "$PYTHON_BIN" -m venv .venv; then
    echo "ERROR: Failed to create virtual environment."
    echo "On Linux you may need: sudo apt install python3-venv"
    pause_on_error
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

ensure_cargo_on_path() {
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi
  if [[ -x "$HOME/.cargo/bin/cargo" ]]; then
    export PATH="$HOME/.cargo/bin:$PATH"
  fi
  if command -v cargo >/dev/null 2>&1; then
    return 0
  fi
  if [[ -f "$HOME/.cargo/env" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.cargo/env" >/dev/null 2>&1 || true
  fi
  command -v cargo >/dev/null 2>&1
}

ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    return 0
  fi

  echo
  echo "FFmpeg not found. Attempting automatic install..."

  if [[ "$(uname -s)" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      brew install ffmpeg
    else
      echo "ERROR: Homebrew not found; cannot auto-install FFmpeg."
      echo "Please do this:"
      echo "  1) Open Safari and go to https://brew.sh"
      echo "  2) Follow the install steps shown there."
      echo "  3) After Homebrew finishes, open Terminal and run:"
      echo "     brew install ffmpeg"
      return 1
    fi
  else
    if command -v apt-get >/dev/null 2>&1; then
      if command -v sudo >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
      else
        echo "ERROR: sudo not found; cannot auto-install FFmpeg."
        echo "Please ask someone with administrator access to install FFmpeg."
        echo "On Ubuntu/Debian the command is:"
        echo "  sudo apt-get install -y ffmpeg"
        return 1
      fi
    else
      echo "ERROR: FFmpeg not found and no supported package manager was detected."
      echo "Please install FFmpeg using your system's app installer."
      echo "If you are on Ubuntu/Debian:"
      echo "  sudo apt-get install -y ffmpeg"
      return 1
    fi
  fi

  command -v ffmpeg >/dev/null 2>&1
}

maybe_install_rust_toolchain() {
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required to install Rust automatically."
    echo "Please install curl and run this setup again."
    return 1
  fi

  echo "Installing Rust toolchain (rustup)..."
  if curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; then
    return 0
  fi
  echo "ERROR: Rust installation failed."
  return 1
}

if [[ -d "$RUST_EDITOR_DIR" ]]; then
  echo "Checking Rust MIDI Touch-Up Editor..."

  if ensure_cargo_on_path || maybe_install_rust_toolchain; then
    # rustup install may add cargo after setup shell startup.
    ensure_cargo_on_path || true
    echo "Building Rust MIDI Touch-Up Editor..."
    if (cd "$RUST_EDITOR_DIR" && cargo build --release); then
      echo "Rust touch-up editor ready: $RUST_EDITOR_BIN"
    else
      echo
      echo "ERROR: Rust touch-up editor build failed."
      echo "Please do this:"
      echo "  1) Open Terminal."
      echo "  2) Run: cd $RUST_EDITOR_DIR"
      echo "  3) Run: cargo build --release"
      echo "Retry manually:"
      echo "  cd $RUST_EDITOR_DIR && cargo build --release"
      echo
      pause_on_error
      exit 1
    fi
  else
    echo
    echo "ERROR: Rust toolchain (cargo) was not found or could not be installed."
    echo "Please install Rust manually:"
    echo "  1) Open Terminal."
    echo "  2) Copy and paste this command, then press Enter:"
    echo "     curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    echo "  3) Close Terminal and open it again."
    echo "  4) Run this setup again."
    echo
    pause_on_error
    exit 1
  fi
fi

if ! ensure_ffmpeg; then
  echo
  echo "ERROR: FFmpeg is required. Please install it and re-run setup."
  pause_on_error
  exit 1
fi

echo "Launching app..."
python run.py
