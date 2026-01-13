# Synthesia2MIDI

Synthesia2MIDI is a PySide6 desktop app that analyzes Synthesia videos, detects key presses frame-by-frame, and exports a MIDI file.

This project is **not affiliated with Synthesia**.

## What It Does

- Load a video of a piano keyboard.
- Define/adjust key overlays 
- Calibrate unlit keys and lit exemplars.
- Run detection 
- Convert detected key states into note on/off events and write a `.mid`.

## Quick Start

### Requirements

- Python 3.12+ recommended
- FFmpeg is recommended (used for YouTube downloads and optional video-to-frames conversion)
  - Windows: https://ffmpeg.org/download.html
  - macOS: `brew install ffmpeg`
  - Linux: install via your package manager (e.g., `sudo apt install ffmpeg`)

### Setup (recommended)

- Windows: run `setup_windows.bat`
- macOS/Linux: run `./setup.sh` (or `bash setup.sh`)

### Manual install (optional)

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
#   .\\.venv\\Scripts\\Activate.ps1
# Windows (cmd.exe):
#   .\\.venv\\Scripts\\activate.bat
pip install -r synthesia2midi/requirements.txt
```

### Run (GUI)

```bash
python synthesia2midi/run.py
```

### Windows convenience launcher

On Windows you can also run `synthesia2midi\\run.bat`.

## Architecture (High-Level)

The code is organized with one-way dependencies:

`GUI → workflows → detection → core`

- `synthesia2midi/synthesia2midi/main.py`: main window and top-level wiring
- `synthesia2midi/synthesia2midi/gui/`: Qt UI (canvas, controls, signals)
- `synthesia2midi/synthesia2midi/workflows/`: orchestration (load/calibrate/convert)
- `synthesia2midi/synthesia2midi/detection/`: detection methods and ROI utilities
- `synthesia2midi/synthesia2midi/core/`: application state and persistence

## Key Concepts

- **Overlay**: a rectangular ROI corresponding to one piano key.
- **Unlit calibration**: captures each key’s baseline color and HSV histogram.
- **Lit exemplars**: reference samples for what a “lit” key looks like (colors and histograms).
- **Progression ratio**: normalized distance from unlit → current relative to unlit → lit exemplar.
- **Histogram detection**: optional rule using ROI HSV histograms (Bhattacharyya distance).
- **Delta detection**: optional frame-to-frame change rule to improve press/release timing.

## Configuration and Artifacts

- Runtime artifacts (logs, screenshots, extracted frames, videos) are intentionally not tracked by git.
- Settings and overlay calibration are persisted via the app’s config workflow.

## Third-Party Licenses

See `THIRD_PARTY_NOTICES.md` for a list of direct Python dependencies and their licenses.

## Known Limitations / Notes

- The Visual Threshold Monitor currently does not compute histogram hits for display; it reflects color progression and delta timing behavior.

## Contributing / Support

Bug fix PRs are welcome. Please keep changes small and focused, and include clear reproduction steps (and logs/screenshots if relevant). For larger features or refactors, please open an issue first; I may ask you to maintain changes in your own fork.
