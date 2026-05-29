# Testing and Verification

Run all commands from the repository root unless a section says otherwise.

## Local Environment

Recommended local setup with Python 3.10+ (3.12+ recommended):

```bash
python3 setup_env.py --dev
# If python3 is too old on macOS/Linux, install a current Python and run e.g. python3.12 setup_env.py --dev
```

`setup_env.py --dev` creates `.venv`, installs the app dependencies plus `pytest`/`ruff`, verifies FFmpeg, and builds the Rust touch-up editor when Cargo is available. FFmpeg is required for app setup; missing FFmpeg is a setup failure.

You do not have to activate the venv for the commands below because they call `.venv/bin/python` directly. If you prefer an activated shell:

```bash
source .venv/bin/activate
python -m pytest
```

On Windows, use the venv Python at `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

## Baseline Pytest Suite

The baseline tests are intended to avoid real videos, network access, and visible GUI windows. They use synthetic inputs, temp files, mocks, subprocess import smoke checks, and Qt's offscreen platform where needed.

`pyproject.toml` supplies the pytest config (`testpaths = ["tests"]` and quiet output), and `tests/conftest.py` adds the package root to `sys.path` and defaults `QT_QPA_PLATFORM=offscreen`. From a normal repo-root checkout, no manual `PYTHONPATH` or Qt environment prefix is needed for pytest.

Copy/paste from repo root:

```bash
python3 setup_env.py --dev
.venv/bin/python -m pytest --collect-only
.venv/bin/python -m pytest
```

Expected scaffold baseline: pytest collects the tests under `tests/` and runs without opening a GUI window, downloading anything, or requiring real video fixtures. The current suite may print existing PySide6 `QFontDatabase` deprecation warnings; those warnings are not test failures.

## Default Gate

Use this before handing off code changes:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
.venv/bin/python -m pytest
```

## Setup / Launcher Gate

```bash
.venv/bin/python -m pytest tests/test_setup_and_launch.py
.venv/bin/python setup_env.py --check
```

`run.py` should remain the only launcher. It re-execs through `.venv` automatically and fails clearly if setup or FFmpeg is missing.

## Import Smoke

`tests/test_import_smoke.py` imports the core app modules plus the manual auto-detector stage modules. Run the default pytest gate after adding detector modules so new files are covered by import smoke.

For an ad hoc import check outside pytest, set `PYTHONPATH` explicitly because `tests/conftest.py` is not loaded:

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
.venv/bin/python -m pytest tests/test_monolithic_detector_characterization.py
```

These tests use temporary generated images, not checked-in media. Add or extend synthetic fixtures before changing black-key scanning, white-key solvers, or note assignment.

## GUI Smoke

The pytest GUI smoke uses offscreen Qt through `tests/conftest.py`:

```bash
.venv/bin/python -m pytest tests/test_video2midi_app_smoke.py
```

For an ad hoc GUI smoke outside pytest, set both `PYTHONPATH` and `QT_QPA_PLATFORM` explicitly:

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

## Troubleshooting

- `ModuleNotFoundError: No module named 'synthesia2midi'`: run pytest from the repository root with `.venv/bin/python -m pytest`, not from inside `tests/`. Confirm `tests/conftest.py` is present. For ad hoc Python snippets, prefix the command with `PYTHONPATH=synthesia2midi`.
- `No module named pytest`: rerun `python3 setup_env.py --dev`, or install pytest into the repo venv with `.venv/bin/python -m pip install pytest`.
- Qt platform/plugin errors or a visible GUI during tests: run through pytest so `tests/conftest.py` sets `QT_QPA_PLATFORM=offscreen`; for ad hoc snippets, prefix with `QT_QPA_PLATFORM=offscreen`.
- Windows path errors: replace `.venv/bin/python` with `.venv\Scripts\python.exe`.

## Test Strategy

- Prefer synthetic NumPy frames and temporary files over checked-in media.
- Unit-test pure utilities first: note mapping, param coercion, ROI extraction, MIDI save/read, config round-trip.
- Mark any real-video, FFmpeg-heavy, or network tests as slow/manual; do not run them in the default baseline suite.
- For behavior-preserving refactors, add characterization tests or smoke checks before moving code.
