#!/bin/bash
# Launch macOS setup from a double-clickable command file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "./setup.sh" ]]; then
  exec /bin/bash "./setup.sh"
else
  echo "setup.sh not found in ${SCRIPT_DIR}."
  exit 1
fi
