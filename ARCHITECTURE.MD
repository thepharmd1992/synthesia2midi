# Synthesia2MIDI Architecture

This document is the agent-friendly map of the repo. Read it before changing cross-cutting GUI, workflow, calibration, detection, conversion, setup, or Rust touch-up code.

## Product Flow

Synthesia2MIDI is a desktop Qt app that converts Synthesia-style piano videos into MIDI files:

1. Load a video, image sequence, or YouTube download.
2. Build or load key overlays.
3. Calibrate unlit keys and lit exemplars.
4. Detect pressed keys frame-by-frame.
5. Convert key transitions into MIDI events.
6. Optionally open the result in the Rust MIDI touch-up editor.

## Dependency Direction

Keep dependencies flowing downward:

```text
Qt composition / signal wiring
  -> GUI controllers / workflows
    -> detection, conversion, video, config, core state
      -> pure-ish utilities and models
```

Rules:

- `main.py` composes the app. Do not put workflow bodies back into it.
- `ControlSignalManager` wires Qt signals to focused controllers/workflows.
- Controllers may coordinate widgets, app state, and workflows.
- Detection/conversion/core modules should not depend on `Video2MidiApp` unless a legacy API makes it unavoidable.
- Preserve per-video config compatibility unless the task explicitly includes migration.

## Entry Points and Setup

| Path | Responsibility |
|---|---|
| `run.py` | User-facing launcher; uses `.venv` internally. |
| `setup_env.py` | Cross-platform setup/check script. Do not replace it with OS-specific wrappers without a clear reason. |
| `synthesia2midi/synthesia2midi/main.py` | Qt root window composition, menus, hotkeys, lifecycle, public UI update adapter methods. |
| `tools/midi_touchup_editor_rust/` | Separate Rust touch-up editor process launched by Python. |

Supported user commands from repo root:

```bash
python3 setup_env.py
python3 run.py
```

Windows equivalents:

```powershell
py setup_env.py
py run.py
```

## Main Window Boundary

`Video2MidiApp` is intentionally a root-window facade. It should own:

- application state object construction;
- controller/workflow construction;
- root widget layout and menu/hotkey wiring;
- Qt lifecycle hooks: `resizeEvent`, `showEvent`, `closeEvent`;
- public `UIUpdateInterface` adapter methods used by managers/controllers.

It should not own:

- video loading workflow bodies;
- conversion/export behavior;
- calibration or auto-detection workflows;
- overlay mutation logic;
- detection parameter state machines;
- MIDI touch-up process lifecycle;
- trim/range business logic.

Current `main.py` guardrail after the refactor: about 500-700 lines, composition only.

## GUI Controllers

| Module | Owner / Boundary |
|---|---|
| `gui/signal_manager.py` | Central Qt signal wiring. Prefer wiring signals directly to controller methods instead of adding `main.py` wrappers. |
| `gui/video_session_ui_controller.py` | File/YouTube video entrypoints, frame navigation interval, trim/range UI handlers, video-to-frames request forwarding. |
| `gui/midi_conversion_controller.py` | User-triggered MIDI conversion flow: output path selection, conversion result UI, touch-up prompt. |
| `gui/midi_touchup_controller.py` | Rust touch-up editor binary discovery, launch, process tracking, result parsing, shutdown. |
| `gui/main_action_controller.py` | Small menu/control actions that mutate app state or delegate to managers: detection toggles, overlay color/size actions, octave/FPS, visual monitor, exemplar toggles. |
| `gui/calibration_wizard_controller.py` | Manual calibration wizard, keyboard-region selection, auto-detect tuning dialog flow. |
| `gui/calibration_effects_controller.py` | Facade over spark, shadow, and overlay calibration effect controllers. |
| `gui/calibration_interaction_controller.py` | Canvas color-pick and overlay-selection dispatch into calibration workflows. |
| `gui/spark_calibration_controller.py` | Spark ROI and spark calibration behavior. |
| `gui/shadow_calibration_controller.py` | Shadow ROI and shadow calibration behavior. |
| `gui/overlay_interaction_controller.py` | Overlay drawing-mode changes. |
| `gui/video_controls.py` | Frame display, frame slider, navigation controls. |
| `gui/display_manager.py` | Canvas/UI display toggles and current-frame display coordination. |
| `gui/window_manager.py` | Resize/position behavior. |

When adding a new user action, first ask: can `ControlSignalManager` wire directly to an existing controller? If yes, do that. If no, add a focused controller method, not a `Video2MidiApp` wrapper.

## Workflows and Managers

| Module | Responsibility |
|---|---|
| `workflows/video_loading.py` | Loading videos/frame sequences and config for a path. |
| `workflows/video_session_coordinator.py` | Fragile post-load order: close old session, reset state, load session into controls/canvas, initialize video-bound workflows, apply config/no-config UI, display initial frame, enable controls, resize. Preserve this ordering. |
| `workflows/video_to_frames.py` | FFmpeg frame-series conversion and worker lifecycle. |
| `workflows/calibration.py` | Manual calibration workflow and overlay generation. |
| `workflows/auto_calibration.py` | Auto-calibration orchestration. |
| `workflows/conversion.py` | Frame processing and MIDI file generation. |
| `workflows/detection_manager.py` | Detection mode/threshold/toggle orchestration and wrappers used by the canvas. |
| `workflows/overlay_manager.py` | Overlay selection/alignment/size mutation. |
| `workflows/parameter_manager.py` | Parameter synchronization between app state and UI. |

## Detection Architecture

Keep `synthesia2midi.detection.monolithic_detector.MonolithicPianoDetector` import/API compatibility unless a task explicitly removes it with migration coverage.

The detector is split into behavior stages:

| Module | Responsibility |
|---|---|
| `detection/monolithic_detector.py` | API-compatible facade. |
| `detection/black_key_detector.py` | Black-key thresholding, scanning, and recovery. |
| `detection/white_key_geometry.py` | Reusable white-key span/valley geometry helpers. |
| `detection/white_key_lattice_model.py`, `white_key_lattice_solver.py`, `black_note_center_map.py`, `black_residual_warp.py` | D-lattice white-key reconstruction path. |
| `detection/white_key_boundary_solver.py` | Boundary/separator fallback white-key detection. |
| `detection/black_note_assignment.py`, `white_note_assignment.py`, `note_assignment.py`, `note_parsing.py` | Note anchors, note scanning, fallback assignment, and overlay adjustment. |
| `detection/detector_visualization.py` | Visualization output. |
| `detection/detector_defaults.py` | Stable auto-detect tuning parameter defaults. |
| `detection/roi_utils.py` | ROI extraction/color helpers. Name RGB/BGR/HSV formats explicitly at call sites. |

Behavior traps:

- Do not rename tuning parameters casually; per-video configs can depend on them.
- Add or update synthetic detector characterization tests before changing black/white-key geometry or note assignment.
- OpenCV image paths are usually BGR/HSV; Qt/canvas paths often use RGB.

## State and Persistence

| Module | Responsibility |
|---|---|
| `core/app_state.py` | Central app state model. |
| `core/state_manager.py` | Reset/default state operations. |
| `config_manager.py` | Per-video `.ini` and overlay sidecar persistence. |
| `video_loader.py` | Video/frame-sequence session abstraction. |
| `midi_generator.py` | MIDI writer wrapper/utility. |

Per-video settings are persisted through `.ini` files and `_overlays.json` sidecars derived from the video path. Backward compatibility is required unless a task explicitly includes migration and tests.

## Python / Rust Boundary

The Python app launches the Rust MIDI touch-up editor as a separate process.

Expected release binary path:

```text
tools/midi_touchup_editor_rust/target/release/midi-touchup-editor      # macOS/Linux
tools/midi_touchup_editor_rust/target/release/midi-touchup-editor.exe  # Windows
```

Process contract:

- Python passes MIDI file paths to the Rust process.
- Python reads a JSON result line from stdout.
- `MidiTouchupController` owns process references and closes them during app shutdown.

If this contract changes, update both Python and Rust and run Python tests plus `cargo check`.

## Testing and Verification

Default gate from repo root:

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

For setup/launcher work:

```bash
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_setup_and_launch.py
.venv/bin/python setup_env.py --check
```

For Rust touch-up editor work:

```bash
cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml
```

## Refactor Rules

- Characterize behavior before extracting high-coupling logic.
- Prefer direct signal-to-controller wiring over compatibility wrappers.
- Keep Qt object lifetime explicit for `QThread`, `QProcess`, and modeless dialogs.
- Preserve video-load ordering in `VideoSessionCoordinator` unless tests and task scope say otherwise.
- Keep changes bounded and commit after passing relevant verification.
- Update this file and `docs/main-py-refactor-checklist.md` when architecture or refactor status changes.
