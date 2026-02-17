#!/bin/bash

# Allow double-click execution on macOS; run from repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# First-run checks: if environment or Rust editor is missing, run setup.
NEED_SETUP=0
if [ ! -x ".venv/bin/python" ]; then
  NEED_SETUP=1
fi

RUST_EDITOR_DIR="tools/midi_touchup_editor_rust"
RUST_EDITOR_BIN="${RUST_EDITOR_DIR}/target/release/midi-touchup-editor"
if [ -d "$RUST_EDITOR_DIR" ] && [ ! -f "$RUST_EDITOR_BIN" ]; then
  NEED_SETUP=1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  NEED_SETUP=1
fi

if [ "$NEED_SETUP" -eq 1 ]; then
  echo "Missing dependencies; running setup.sh..."
  /bin/bash "./setup.sh"
  if [ $? -ne 0 ]; then
    echo
    echo "Setup failed. Fix the error above, then re-run setup.sh."
    read -r -p "Press Enter to close..."
    exit 1
  fi
fi

# Re-check required components after setup.
NEED_SETUP=0
if [ ! -x ".venv/bin/python" ]; then
  NEED_SETUP=1
fi
if [ -d "$RUST_EDITOR_DIR" ] && [ ! -f "$RUST_EDITOR_BIN" ]; then
  NEED_SETUP=1
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  NEED_SETUP=1
fi

if [ "$NEED_SETUP" -eq 1 ]; then
  echo
  echo "Dependencies are still missing. Re-run setup.sh after fixing the issue."
  read -r -p "Press Enter to close..."
  exit 1
fi

# Activate venv for normal runs.
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# Launch the app
python run.py
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo
  echo "The app closed with an error."
  read -r -p "Press Enter to close..."
  exit $EXIT_CODE
fi
