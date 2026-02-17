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

if [ "$NEED_SETUP" -eq 1 ]; then
  echo "Missing first-run components; running setup.sh..."
  exec /bin/bash "./setup.sh"
fi

# Activate venv for normal runs.
if [ -f ".venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

# Launch the app
exec python run.py
