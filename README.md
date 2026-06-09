# Synthesia2MIDI

Synthesia2MIDI is a PySide6 desktop app that analyzes Synthesia videos, detects key presses frame-by-frame, and exports a MIDI file.

This project is **not affiliated with Synthesia**.

Acknowledgments: see [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).

![Synthesia2MIDI GUI](docs/GUI.png)

## What It Does

- Load a video of a piano keyboard.
- Define/adjust key overlays 
- Calibrate unlit keys and lit exemplars.
- Run detection 
- Convert detected key states into note on/off events and write a `.mid`.

## Download

End users should use GitHub Releases, not the repo setup flow.

- macOS Apple Silicon: download the `Synthesia2MIDI-macos-arm64-<tag>.zip` asset, unzip it, then open the app. If Gatekeeper blocks launch, open System Settings > Privacy & Security and use `Open Anyway`.
- Windows x64: download the `Synthesia2MIDI-windows-x64-<tag>.zip` asset, unzip it, then launch `Synthesia2MIDI.exe`. If SmartScreen warns, click `More info` and `Run anyway`.

## Quick Start

### Requirements

- Python 3.10+ is required; Python 3.12+ is recommended. On macOS, Apple's system `python3` may be too old, so install a current Python first if `python3 --version` reports less than 3.10.
- FFmpeg is required and must be available on `PATH`.
  - Windows: `winget install Gyan.FFmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: install via your package manager, e.g. `sudo apt install ffmpeg`
- Rust toolchain (`cargo`) is optional for the main app, but needed to build the MIDI Touch-Up Editor binary.
  - Windows: `winget install --id Rustlang.Rustup -e`
  - macOS/Linux: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

### Setup

Run from the repo root.

macOS/Linux with Python 3.10+ on `PATH`:

```bash
python3 setup_env.py
# If python3 is too old, install a current Python and run e.g. python3.12 setup_env.py
```

Windows:

```powershell
py setup_env.py
```

The setup script creates/updates the repo-local `.venv`, installs Python dependencies, verifies FFmpeg, and builds the Rust Touch-Up Editor if `cargo` is available. If FFmpeg is missing, setup fails with install instructions.

You do not need to activate `.venv` manually; `run.py` uses it automatically.

### Run (GUI)

```bash
python3 run.py
```

On Windows, use `py run.py`.

### Packaged Release Build

Portable end-user bundles are built from version tags:

```bash
.venv/bin/python packaging/build_release.py --version v0.1.0
```

The build script bundles FFmpeg/ffprobe from `PATH`, bundles the Rust touch-up editor and soundfont assets, installs Deno if needed, creates a zip under `dist/release/`, and smoke-launches the packaged app offscreen.

### Touch-Up Editor (Rust binary)

The setup script builds the Rust MIDI Touch-Up Editor automatically when `cargo` is available. Without Cargo, the main app still works; touch-up editing is unavailable until the binary is built.

## Architecture (High-Level)

The code is organized with one-way dependencies:

`GUI → workflows → detection → core`

- `synthesia2midi/synthesia2midi/main.py`: main window and top-level wiring
- `synthesia2midi/synthesia2midi/gui/`: Qt UI (canvas, controls, signals)
- `synthesia2midi/synthesia2midi/workflows/`: orchestration (load/calibrate/convert)
- `synthesia2midi/synthesia2midi/detection/`: detection methods and ROI utilities
- `synthesia2midi/synthesia2midi/core/`: application state and persistence

## Developer / Agent Docs

- Documentation index: [docs/README.md](docs/README.md)
- Canonical agent operating model: [AGENTS.md](AGENTS.md)
- Project state and durable decisions: [PROJECT_LOG.md](PROJECT_LOG.md)
- Architecture map: [ARCHITECTURE.MD](ARCHITECTURE.MD)
- Task ownership and handoff boundaries: [docs/task-boundaries.md](docs/task-boundaries.md)
- Detailed testing and verification runbook: [docs/testing.md](docs/testing.md)

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

See `THIRD_PARTY_NOTICES.md` for a list of third-party dependencies, included tools, and assets.

## License

This repository is licensed under GPL-3.0-only. See `LICENSE`.

## Known Limitations / Notes

- The Visual Threshold Monitor currently does not compute histogram hits for display; it reflects color progression and delta timing behavior.

## Contributing / Support

Bug fix PRs are welcome. Please keep changes small and focused, and include clear reproduction steps (and logs/screenshots if relevant). For larger features or refactors, please open an issue first; I may ask you to maintain changes in your own fork.
