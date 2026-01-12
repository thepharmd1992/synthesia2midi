#!/bin/bash

# Allow double-click execution on macOS; run from repo root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
  source ".venv/bin/activate"
fi

# Launch the app
exec python3 synthesia2midi/run.py
