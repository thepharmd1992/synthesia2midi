# Synthesia2MIDI Agent-First Refactor Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task after Jeff explicitly dispatches the Kanban roadmap.

**Goal:** Make Synthesia2MIDI safe and efficient for coding-agent development while preserving current user-visible behavior.

**Architecture:** Refactor by characterization-first extraction. Keep `main.py` as a temporary compatibility facade while moving bounded responsibilities into controllers/workflows, then shrink the facade only after tests and smoke checks protect the behavior.

**Tech Stack:** Python 3.12, PySide6, OpenCV, NumPy, midiutil/mido, yt-dlp, Rust 2021, egui/eframe, midly, cpal/rustysynth.

---

## Current Facts

- Repo: project checkout root
- Kanban tenant: `synthesia2midi`
- Root Kanban epic: `t_77771f49`
- `main.py` is currently ~2.9k lines and owns Qt shell, menus, dialogs, video loading, calibration interaction dispatch, conversion actions, FFmpeg frame extraction, and Rust touch-up editor process lifecycle.
- Blocking syntax error in `main.py` was fixed in the parent planning session.
- Verified in the parent planning session:
  - `.venv/bin/python -m compileall -q synthesia2midi`
  - `PYTHONPATH=synthesia2midi .venv/bin/python` imports auto-detector modules and `DetectionFactory`
  - `PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python` can instantiate `Video2MidiApp`
- No tracked test suite exists yet.
- Current CI only does install, compileall, and a `Video2MidiApp` import smoke on Windows/macOS.

## Non-Goals for the First Refactor Wave

- Do not redesign detection algorithms while extracting `main.py` responsibilities.
- Do not change config/calibration file formats unless a card explicitly adds a migration.
- Do not introduce a broad plugin framework, event bus, or generalized agent harness.
- Do not optimize UI design or add new user features.
- Do not commit large media fixtures, generated MIDI files, logs, extracted frames, `.venv`, or Rust `target/` outputs.

## Global Verification Commands

Run from repo root unless noted.

```bash
git status --short --branch
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi .venv/bin/python - <<'PY'
from synthesia2midi.detection.monolithic_detector import MonolithicPianoDetector
from synthesia2midi.detection.auto_detect_adapter import AutoDetectAdapter
from synthesia2midi.detection.factory import DetectionFactory
print('auto detector imports ok')
print(DetectionFactory.get_available_methods())
PY
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

Rust editor gate:

```bash
cd tools/midi_touchup_editor_rust
cargo check
```

Once pytest exists:

```bash
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

---

## Kanban Roadmap

All cards were created in `triage` with workspace pointing at the project checkout so nothing dispatches until Jeff explicitly promotes/dispatches.

| ID | Title | Assignee |
|---|---|---|
| `t_77771f49` | project: synthesia2midi agent-first refactor roadmap | pm |
| `t_162eeb6f` | M0: establish tracked agent operating model and project log | pm |
| `t_407a23d8` | M0: add Python test scaffold and baseline characterization tests | backend-eng |
| `t_bba30a31` | M0: harden CI gates for Python and Rust touch-up editor | ops |
| `t_2f43096c` | M1: characterize main.py responsibilities before extraction | analyst |
| `t_2926a400` | M1: extract video-to-frames worker/controller from main.py | backend-eng |
| `t_8ca2f8f2` | M1: extract MIDI touch-up launcher/process controller | backend-eng |
| `t_c4437a18` | M1: extract shared video session coordinator | backend-eng |
| `t_d65a82ed` | M2: extract calibration wizard and auto-detect tuning controller | backend-eng |
| `t_7829ca08` | M2: extract calibration interaction, spark, and shadow controllers | backend-eng |
| `t_09729a5d` | M2: add detection/conversion reliability tests and fix confirmed utility bugs | backend-eng |
| `t_795498c5` | M3: review architecture boundaries after first extraction wave | reviewer |

Dry-run dispatch after creation returned no spawned tasks.

---

## Task 0: Fix Compile Blocker

**Objective:** Restore Python compilation and auto-detector imports.

**Status:** Done in parent planning session.

**Files:**
- Modified: `synthesia2midi/synthesia2midi/main.py`

**Change:** Fixed nested quote syntax error in the `_open_video_file` logging f-string.

**Verification:**

```bash
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi .venv/bin/python - <<'PY'
from synthesia2midi.detection.monolithic_detector import MonolithicPianoDetector
from synthesia2midi.detection.auto_detect_adapter import AutoDetectAdapter
from synthesia2midi.detection.factory import DetectionFactory
print('auto detector imports ok')
PY
```

---

## Task 1: Establish Agent Operating Model

**Objective:** Give future coding agents a small, reliable onboarding surface.

**Kanban:** `t_162eeb6f`

**Files:**
- Modify: `.gitignore` if choosing tracked `AGENTS.md`
- Create: `AGENTS.md` or `docs/agent-operating-model.md`
- Create: `PROJECT_LOG.md`
- Create: `docs/task-boundaries.md`

**Steps:**

1. Decide tracked root `AGENTS.md` vs tracked `docs/agent-operating-model.md`.
   - Recommendation: track `AGENTS.md`; remove the ignore rule for it.
2. Add `PROJECT_LOG.md` with original goal, definition of done, current focus, blockers, parking lot, and decisions.
3. Add task-boundary table mapping file areas to assignees:
   - GUI: `frontend-eng`
   - workflows/state/detection/conversion: `backend-eng`
   - CI/setup: `ops`
   - docs/spec/review: `pm`/`reviewer`
4. Keep command sequences in one canonical place; link instead of duplicating.

**Verification:**

```bash
git diff --check
```

---

## Task 2: Add Python Test Scaffold and Baseline Characterization Tests

**Objective:** Create tests before large refactors so agents can move code safely.

**Kanban:** `t_407a23d8`

**Files:**
- Create: `tests/`
- Create: `tests/conftest.py`
- Create: `tests/test_overlay_config.py` or `tests/test_auto_detect_param_specs.py`
- Create/modify: `docs/testing.md`
- Optionally create: `pyproject.toml` for pytest config if the repo wants config in one file.

**First tests:**

1. `OverlayConfig` MIDI mapping:
   - C4 = 60
   - A4 = 69
   - A0 = 21
   - octave transpose clamps to 0..127
2. Auto-detect param coercion:
   - unknown keys dropped
   - bool strings normalize
   - numeric params clamp
   - odd-only params become odd
3. Import smoke:
   - imports package modules with `QT_QPA_PLATFORM=offscreen`
   - does not require real video/network.

**Verification:**

```bash
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
```

---

## Task 3: Harden CI

**Objective:** Make CI catch the exact failures agents are likely to introduce.

**Kanban:** `t_bba30a31`

**Files:**
- Modify: `.github/workflows/ci.yml`
- Optionally modify: `synthesia2midi/requirements.txt` or add dev/test requirements if needed.

**CI additions:**

1. Set:

```yaml
env:
  PYTHONPATH: synthesia2midi
  QT_QPA_PLATFORM: offscreen
```

2. Install pytest/dev tools.
3. Run:
   - `python -m compileall -q synthesia2midi`
   - import smoke
   - `python -m pytest -q`
4. Add Rust editor check:

```bash
cd tools/midi_touchup_editor_rust
cargo check
```

5. Optional low-churn ruff gate:

```bash
ruff check synthesia2midi --select=E9,F63,F7,F82
```

**Verification:**

```bash
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
cd tools/midi_touchup_editor_rust && cargo check
```

---

## Task 4: Characterize `main.py` Before Extraction

**Objective:** Convert the audit into stable extraction targets and behavior checks.

**Kanban:** `t_2f43096c`

**Files:**
- Create: `docs/refactor-main.md` or add to `docs/architecture.md`
- Add tests/smokes from Task 2 if not already present.

**Document these current responsibilities:**

- app bootstrap/logging
- Qt main window shell
- menu/action/hotkey creation
- startup/local/YouTube dialogs
- video session lifecycle
- FFmpeg video-to-frame conversion
- MIDI conversion action
- Rust touch-up process integration
- calibration wizard and auto-detect tuning dialog
- overlay click calibration dispatch
- spark/shadow ROI calibration
- frame navigation and UI update facade

**Behavior traps to lock down:**

- Qt object lifetime for `QThread`, `QProcess`, modeless dialogs.
- `ControlSignalManager` currently expects old main-window methods.
- Video-load ordering is fragile and must be preserved.
- Color spaces: Qt/canvas RGB vs OpenCV BGR/HSV.
- `unsaved_changes` and auto-save semantics during calibration.

**Verification:**

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
```

---

## Task 5: Extract Video-to-Frames Controller

**Objective:** First low-risk extraction from `main.py`.

**Kanban:** `t_2926a400`

**Files:**
- Create: `synthesia2midi/synthesia2midi/workflows/video_to_frames.py`
- Modify: `synthesia2midi/synthesia2midi/main.py`
- Possibly modify: `synthesia2midi/synthesia2midi/gui/signal_manager.py` only if wrappers are not enough.

**Move:**

- `VideoToFramesWorker`
- `_handle_video_to_frames_request`
- `_on_conversion_progress`
- `_on_conversion_finished`

**Design:**

- Controller owns the worker reference.
- Main window keeps a thin wrapper while existing signal wiring remains coupled to main method names.

**Verification:**

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY'
from PySide6.QtWidgets import QApplication
from synthesia2midi.main import Video2MidiApp
app = QApplication([])
w = Video2MidiApp()
assert hasattr(w, 'control_panel')
w.close()
app.quit()
print('smoke ok')
PY
```

---

## Task 6: Extract MIDI Touch-Up Controller

**Objective:** Move Rust touch-up editor process/path/dialog lifecycle out of `main.py`.

**Kanban:** `t_8ca2f8f2`

**Files:**
- Create: `synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py`
- Modify: `synthesia2midi/synthesia2midi/main.py`

**Move:**

- `_show_conversion_complete_dialog_with_touchup`
- `_open_midi_touchup_editor_from_picker`
- `_open_midi_touchup_editor`
- `_resolve_midi_touchup_binary_path`
- `_show_midi_touchup_setup_dialog`
- `_handle_midi_touchup_process_finished`
- `_cleanup_midi_touchup_process`
- `_remove_midi_touchup_process_ref`
- `_shutdown_midi_touchup_processes`

**Design:**

- Use a QObject-backed controller parented to the main window.
- Preserve `_is_closing` semantics so close shutdown does not spawn extra dialogs.

**Verification:**

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
cd tools/midi_touchup_editor_rust && cargo check
```

---

## Task 7: Extract Video Session Coordinator

**Objective:** Deduplicate local and YouTube video load paths.

**Kanban:** `t_c4437a18`

**Files:**
- Create: `synthesia2midi/synthesia2midi/workflows/video_session_coordinator.py`
- Optionally create: `synthesia2midi/synthesia2midi/gui/video_file_picker.py`
- Modify: `synthesia2midi/synthesia2midi/main.py`

**Preserve ordering:**

1. Close old session.
2. Reset state to defaults.
3. Load video path.
4. Assign `self.video_session`.
5. Update video controls/canvas/frame slider.
6. Initialize `CalibrationWorkflow`, `AutoCalibrationWorkflow`, `DetectionManager`, `ConversionWorkflow`.
7. Update canvas detection wrapper.
8. Apply loaded-config or missing-config UI state.
9. Display initial frame.
10. Enable/disable buttons.
11. Resize window.

**Verification:**

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY'
from PySide6.QtWidgets import QApplication
from synthesia2midi.main import Video2MidiApp
app = QApplication([])
w = Video2MidiApp()
assert hasattr(w, 'video_loading_workflow')
assert hasattr(w, 'video_controls')
w.close()
app.quit()
print('smoke ok')
PY
```

---

## Task 8: Extract Calibration Wizard and Auto-Detect Tuning Controller

**Objective:** Move wizard/dialog lifecycle out of `main.py` after lower-risk extractions are stable.

**Kanban:** `t_d65a82ed`

**Files:**
- Create: `synthesia2midi/synthesia2midi/workflows/calibration_wizard_controller.py`
- Modify: `synthesia2midi/synthesia2midi/main.py`

**Move:**

- `_invoke_calibration_wizard`
- keyboard region selection request/selected handlers
- edit-current-calibration handlers
- auto-detect tuning context cache/build/apply methods
- tuning dialog finished handler

**Risks:**

- Modeless dialog lifetime.
- Wizard cleanup timing.
- Callback state flags.

**Verification:**

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY'
from PySide6.QtWidgets import QApplication
from synthesia2midi.main import Video2MidiApp
app = QApplication([])
w = Video2MidiApp()
assert hasattr(w, 'keyboard_canvas')
assert hasattr(w, 'control_panel')
w.close()
app.quit()
print('smoke ok')
PY
```

---

## Task 9: Extract Calibration Interaction, Spark, and Shadow Controllers

**Objective:** Remove the largest domain-heavy branch logic from `main.py`.

**Kanban:** `t_7829ca08`

**Files:**
- Create: `synthesia2midi/synthesia2midi/workflows/calibration_interactions.py`
- Create: `synthesia2midi/synthesia2midi/workflows/spark_calibration_controller.py`
- Create: `synthesia2midi/synthesia2midi/workflows/shadow_calibration_controller.py`
- Modify: `synthesia2midi/synthesia2midi/main.py`

**Move:**

- `_handle_overlay_selection`
- `_handle_color_pick`
- lit exemplar calibration start/change handlers if not already elsewhere
- spark ROI/calibration methods
- shadow ROI/calibration methods
- `_extract_roi`

**Risks:**

- BGR/RGB confusion.
- Auto-save and `unsaved_changes` behavior.
- Click dispatch mode handling.

**Verification:**

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python - <<'PY'
from PySide6.QtWidgets import QApplication
from synthesia2midi.main import Video2MidiApp
app = QApplication([])
w = Video2MidiApp()
assert callable(w._handle_overlay_selection)
w.close()
app.quit()
print('smoke ok')
PY
```

---

## Task 10: Add Detection/Conversion Reliability Tests and Fix Confirmed Bugs

**Objective:** Cover pure logic and fix confirmed utility bugs found during audit.

**Kanban:** `t_09729a5d`

**Files:**
- Modify: `synthesia2midi/synthesia2midi/detection/roi_utils.py`
- Modify: `synthesia2midi/synthesia2midi/midi_generator.py`
- Create tests under `tests/`

**Confirmed bugs to test first:**

1. `roi_utils.adjust_overlay_for_crop()` returns `None` because it constructs `OverlayConfig` without required note fields.
2. `MidiWriter.save_to_disk("out.mid")` fails because `os.path.dirname("out.mid") == ""` and `os.makedirs("")` raises.

**Other high-value tests:**

- `DetectionFactory` chooses standard vs spark-integrated correctly.
- `StandardDetection` detects a synthetic lit ROI and ignores unlit/missing calibration.
- `ConfigManager` saves/loads overlay JSON and INI in `tmp_path`.

**Verification:**

```bash
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q synthesia2midi
```

---

## Task 11: Architecture Boundary Review

**Objective:** Stop after the first extraction wave and check whether the repo is actually more agent-friendly.

**Kanban:** `t_795498c5`

**Files:**
- Read-only review across changed files.
- Update docs only if boundary docs are stale.

**Review checklist:**

- Is `main.py` smaller and mostly a facade/composition root?
- Did any extracted controller become a new god object?
- Are dependencies still roughly `GUI → workflows → detection → core`?
- Are Qt object lifetimes still safe?
- Do tests/smokes cover the moved seams?
- Can a fresh agent pick up the next card from docs and Kanban without tribal context?

**Verification:**

```bash
git status --short --branch
git diff --stat
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
cd tools/midi_touchup_editor_rust && cargo check
```

---

## Recommended Extraction Order

1. Video-to-frames controller.
2. MIDI touch-up controller.
3. Shared video session coordinator.
4. Calibration wizard / auto-detect tuning controller.
5. Calibration interaction + spark/shadow controllers.
6. Menu/layout extraction if still worth it after behavior-heavy logic is gone.
7. Move `Video2MidiApp` from `main.py` to `gui/main_window.py`; leave `main.py` as import/entrypoint shim.

Do not start with layout/menu extraction unless a frontend card needs it. It reduces line count but does less for behavioral risk than moving process/session/calibration controllers.

## Risk Register

| Risk | Mitigation |
|---|---|
| Qt object lifetime breaks after extraction | Controllers that own `QProcess`, `QThread`, or dialogs should inherit/hold QObject parents and be smoke-tested offscreen. |
| Signal manager still expects main-window method names | Keep thin wrappers in main window until a later signal-manager card updates bindings. |
| Video load ordering changes subtly | Document the order and add characterization checks before moving coordinator code. |
| BGR/RGB/HSV mix-ups during calibration extraction | Name frame formats in function names/docstrings and add small synthetic ROI tests. |
| Existing config files break | Preserve load/save behavior and add `tmp_path` round-trip tests. |
| Docs become stale pseudo-truth | Docs should capture boundaries, commands, decisions, and contracts, not full code inventories. |
| Over-agenting creates churn | Keep cards small; do not dispatch the whole roadmap blindly. Promote one milestone at a time. |

## Suggested First Dispatch Batch

Do not dispatch the whole board. Start with M0 only:

1. `t_162eeb6f` — agent operating model/project log.
2. `t_407a23d8` — Python test scaffold.
3. `t_bba30a31` — CI hardening.

After M0 is reviewed, dispatch M1 extraction cards one at a time or in this limited parallel shape:

- `t_2926a400` video-to-frames extraction and `t_8ca2f8f2` MIDI touch-up extraction can run in parallel if they touch disjoint `main.py` regions and use separate branches/worktrees.
- `t_c4437a18` video session coordinator should wait until those land because it touches higher-coupling workflow initialization.
