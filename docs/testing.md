# Testing and Verification

## Local Environment

Recommended local setup with Python 3.10+ (3.12+ recommended):

```bash
python3 setup_env.py --dev
# If python3 is too old on macOS, install a current Python and run e.g. python3.12 setup_env.py --dev
```

`setup_env.py --dev` creates `.venv`, installs Python dependencies plus `pytest`/`ruff`, verifies FFmpeg, and builds the Rust touch-up editor when Cargo is available. FFmpeg is required; missing FFmpeg is a setup failure.

## Default Gate

Run from repo root:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

## Setup / Launcher Gate

```bash
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_setup_and_launch.py
.venv/bin/python setup_env.py --check
```

`run.py` should remain the only launcher. It re-execs through `.venv` automatically and fails clearly if setup or FFmpeg is missing.

## Import Smoke

`tests/test_import_smoke.py` imports the core app modules plus the manual auto-detector stage modules. Run the default pytest gate after adding detector modules so new files are covered by import smoke.

```bash
PYTHONPATH=synthesia2midi .venv/bin/python - <<'PY'
from synthesia2midi.detection.monolithic_detector import MonolithicPianoDetector
from synthesia2midi.detection.auto_detect_adapter import AutoDetectAdapter
from synthesia2midi.detection.factory import DetectionFactory
print('auto detector imports ok')
print(DetectionFactory.get_available_methods())
PY
```

## Detector Characterization

The manual ROI auto-detector has synthetic characterization coverage in `tests/test_monolithic_detector_characterization.py`:

```bash
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_monolithic_detector_characterization.py
```

These tests use temporary generated images, not checked-in media. Add or extend synthetic fixtures before changing black-key scanning, white-key solvers, or note assignment.

## GUI Smoke

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

## Rust Editor Gate

```bash
cd tools/midi_touchup_editor_rust
cargo check
```

## Test Strategy

- Prefer synthetic NumPy frames and temporary files over checked-in media.
- Unit-test pure utilities first: note mapping, param coercion, ROI extraction, MIDI save/read, config round-trip.
- Mark any real-video, ffmpeg, or network tests as slow/manual; do not run them in default CI.
- For behavior-preserving refactors, add characterization tests or smoke checks before moving code.
