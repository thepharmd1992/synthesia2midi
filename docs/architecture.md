# Architecture

## Current System

Synthesia2MIDI is a desktop app that turns Synthesia-style piano videos into MIDI files.

Core flow:

1. Load a video, image sequence, or YouTube download.
2. Build or load key overlays.
3. Calibrate unlit and lit key exemplars.
4. Detect pressed keys frame by frame.
5. Convert key transitions into MIDI notes.
6. Optionally open the generated MIDI in the Rust touch-up editor.

## Target Dependency Direction

```text
GUI/composition -> workflows/controllers -> detection/conversion/core
```

- GUI/composition owns Qt widgets, menus, dialogs, and signal wiring.
- Workflows/controllers orchestrate user actions and long-running operations.
- Detection/conversion/core own algorithmic behavior and state models.

## Refactor Objective

`main.py` is being reduced from a god object into a thin main-window facade and eventual entrypoint shim. The manual auto-detector has also been split from one monolithic implementation file into focused detection stages. Behavior moves into focused controllers/workflows/detection modules while compatibility facades preserve existing signal wiring and detector imports during the transition.

## Current Extraction State

The first GUI/workflow refactor wave is complete:

- `workflows/video_to_frames.py` owns FFmpeg frame-series conversion and worker lifecycle.
- `gui/midi_touchup_controller.py` owns Rust MIDI touch-up editor launch/process handling.
- `workflows/video_session_coordinator.py` owns shared post-load video session wiring for local and YouTube paths.
- `gui/calibration_wizard_controller.py` owns manual calibration wizard and auto-detect tuning dialog flow.
- `gui/calibration_effects_controller.py` is a small facade over focused calibration interaction/spark/shadow/overlay controllers.
- `gui/calibration_interaction_controller.py` owns overlay-click calibration dispatch and color-pick delegation.
- `gui/spark_calibration_controller.py` owns spark ROI and spark calibration behavior.
- `gui/shadow_calibration_controller.py` owns shadow ROI and shadow calibration behavior.
- `gui/overlay_interaction_controller.py` owns overlay drawing mode changes.

`main.py` remains a compatibility facade for existing signal wiring and UI-update interfaces. Its responsibility is now composition, menus/dialog entrypoints, small compatibility wrappers, and app-level coordination that has not yet earned another extraction.

The manual ROI auto-detector now keeps `MonolithicPianoDetector` as an API-compatible facade in `detection/monolithic_detector.py`, with behavior split by stage:

- `detection/black_key_detector.py` owns black-key thresholding/scanning/recovery.
- `detection/white_key_geometry.py` owns reusable white-key span/valley geometry helpers.
- `detection/white_key_lattice_model.py`, `white_key_lattice_solver.py`, `black_note_center_map.py`, and `black_residual_warp.py` own the D-lattice white-key reconstruction path.
- `detection/white_key_boundary_solver.py` owns boundary/separator fallback white-key detection.
- `detection/black_note_assignment.py`, `white_note_assignment.py`, `note_assignment.py`, and `note_parsing.py` own note anchors, note scanning, fallback assignment, and post-assignment overlay adjustment.
- `detection/detector_visualization.py` owns final visualization output.
- `detection/detector_defaults.py` owns default auto-detect tuning parameters.

## Extraction Order

1. Video-to-frames worker/controller.
2. MIDI touch-up launcher/process controller.
3. Shared video session coordinator.
4. Calibration wizard and auto-detect tuning controller.
5. Calibration interaction, spark, and shadow controllers.
6. Manual ROI auto-detector stage extraction.
7. Optional menu/layout extraction after behavior-heavy logic is moved.

## Behavior Traps

- Qt object lifetime matters for `QThread`, `QProcess`, and modeless dialogs.
- Existing `ControlSignalManager` expects several main-window method names; keep wrappers until wiring changes deliberately.
- Video-load ordering is fragile: close previous session, reset state, load video, update controls/canvas, create video-bound workflows, apply config/no-config UI, display initial frame, enable controls, resize.
- OpenCV paths use BGR/HSV; Qt/canvas display paths often use RGB. Name formats explicitly in extracted code.
- Manual ROI auto-detection is behavior-sensitive: keep `MonolithicPianoDetector` import/method compatibility, preserve tuning parameter names, and add synthetic-frame characterization before changing black/white-key geometry or note assignment.
- Calibration handlers often auto-save and manipulate `unsaved_changes`; preserve those semantics.

## Python/Rust Boundary

The Python app launches the Rust MIDI touch-up editor as a separate process. The expected release binary lives under `tools/midi_touchup_editor_rust/target/release/` as `midi-touchup-editor` on macOS/Linux or `midi-touchup-editor.exe` on Windows. The Python host passes MIDI paths and reads a JSON result line from stdout; that process contract must remain stable unless a task explicitly updates both sides.

## Persistence Boundary

Per-video settings are persisted through `.ini` files and `_overlays.json` sidecars derived from the video path. Backward compatibility is required unless a task explicitly includes migration and tests.
