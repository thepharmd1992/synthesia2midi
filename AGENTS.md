# Synthesia2MIDI Agent Contract

This repo is optimized for coding-agent development. Keep changes small, explicit, and verifiable.

## Current Goal

Make the app easier for agents to understand and modify by shrinking `main.py`, splitting the manual auto-detector into focused stages, adding tests, and documenting stable boundaries while preserving current user-visible behavior.

## Non-Negotiables

- Preserve existing app behavior unless the task explicitly says otherwise.
- Preserve per-video config/calibration compatibility unless the task explicitly includes a migration.
- Do not commit generated media, logs, extracted frames, MIDI files, `.venv`, or Rust `target/` output.
- Start every task with `git status --short --branch` and protect unrelated user/agent changes.
- For multi-step refactors or setup changes, make frequent small git commits after passing relevant verification so changes can be reverted cleanly.
- One bounded task per change. Do not opportunistically refactor neighboring systems.
- Add or update tests before behavior changes and before risky refactors.
- Keep docs focused on stable contracts, commands, decisions, and boundaries. Do not paste stale code inventories.

## Architecture Direction

Target dependency direction:

```text
GUI/composition -> workflows/controllers -> detection/conversion/core
```

`main.py` should become a thin compatibility facade and eventual entrypoint shim. Move behavior into focused controllers/workflows with explicit ownership. Track remaining work in `docs/main-py-refactor-checklist.md`; update that checklist after each extraction checkpoint.

## Verification Defaults

Run from repo root:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

For GUI wiring/refactor work, also run:

```bash
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY'
from PySide6.QtWidgets import QApplication
from synthesia2midi.main import Video2MidiApp
app = QApplication([])
w = Video2MidiApp()
assert hasattr(w, 'control_panel')
assert hasattr(w, 'keyboard_canvas')
w.close()
app.quit()
print('offscreen Video2MidiApp smoke ok')
PY
```

For setup/launcher work, also run:

```bash
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_setup_and_launch.py
.venv/bin/python setup_env.py --check
```

For Rust touch-up editor work:

```bash
cd tools/midi_touchup_editor_rust
cargo check
```

## Setup / Launcher Contract

- Supported user commands are `python3 setup_env.py` / `python3 run.py` on macOS/Linux, and `py setup_env.py` / `py run.py` on Windows, from the repo root.
- Keep `.venv` as an implementation detail; users should not need to activate it manually.
- FFmpeg is required. Setup and launch should fail clearly if `ffmpeg` is unavailable on `PATH`.
- Do not reintroduce Textual/TUI installers or OS-specific setup/run wrapper scripts unless explicitly requested.

## Task Boundaries

- GUI/window/menu/dialog wiring: `frontend-eng`
- workflows/controllers/state/config/detection/conversion: `backend-eng`
- setup/CI/release scripts: `ops`
- architecture docs, review gates, operating model: `pm` / `reviewer`
- detection tuning/research and fixture design: `researcher` / `analyst`

See `docs/task-boundaries.md` for the full table.

## Refactor Rules for `main.py`

- Characterize before extracting high-coupling behavior.
- Keep thin wrapper methods until `ControlSignalManager` and other signal wiring are updated.
- Controllers that own `QThread`, `QProcess`, or modeless dialogs must preserve Qt object lifetime; use QObject parenting or explicit owner references.
- Preserve video-load ordering exactly unless tests and task scope say otherwise.
- Name color spaces explicitly when moving ROI/calibration code: RGB, BGR, HSV.

## Refactor Rules for Manual Auto-Detection

- Keep `synthesia2midi.detection.monolithic_detector.MonolithicPianoDetector` import/API compatibility unless a task explicitly includes migration.
- Preserve auto-detect tuning parameter names in `detector_defaults.py`; existing per-video config can depend on them.
- Add or update synthetic detector characterization tests before changing black-key scanning, white-key solvers, or note assignment.
- Keep behavior stages separate: black-key detection, white-key geometry/solvers, note assignment, visualization.

## Kanban

Tenant: `synthesia2midi`

Root roadmap epic: `t_77771f49`

Useful commands:

```bash
hermes kanban list --tenant synthesia2midi --json
hermes kanban show t_77771f49 --json
```
