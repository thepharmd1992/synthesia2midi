# `main.py` Extraction Group Plan

This document records the first three safe extraction groups for the historical `Video2MidiApp` god-object refactor. It is based on the pre-refactor `main.py` inventory from `e987c9c^` (2,867 lines) plus the current architecture map and controller boundaries. Current `HEAD` has already implemented these boundaries; keep this file as the exact characterization map for replaying or reviewing the extraction sequence.

Principle: extract one responsibility at a time, preserve old behavior through characterization checks, and keep `main.py` as a temporary compatibility facade until signal callers can be rewired directly.

## Prerequisites and blockers

- Fix compile blockers before characterization. The historical baseline had an `_open_video_file` f-string syntax error; no extraction is safe until `compileall` can run.
- Preserve unrelated working-tree changes. Inspect `git status --short --branch` and relevant diffs before editing, and do not touch unrelated test/code changes.
- Start every extraction checkpoint with the repo preflight from [`../AGENTS.md`](../AGENTS.md) and the relevant verification gate from [`testing.md`](testing.md).
- For any GUI wiring extraction, also run the GUI smoke from [`testing.md`](testing.md).
- Do not combine these groups with detection algorithm, config format, calibration math, or UI redesign changes.

## Group 1 — Video-to-frame-series conversion worker/controller

Rationale: this is the lowest-risk first extraction. It is already a cohesive asynchronous workflow with a clear `QThread` lifetime, a single user entrypoint, and minimal dependency on the rest of `Video2MidiApp`.

Destination:

- `synthesia2midi/synthesia2midi/workflows/video_to_frames.py`
- Classes: `VideoToFramesWorker`, `VideoToFramesController`
- Temporary facade in `main.py` is acceptable for one checkpoint; next checkpoint can rewire `ControlSignalManager` to the controller directly.

Exact methods/functions to move from historical `main.py`:

- `VideoToFramesWorker.__init__(video_path, output_dir, quality=90)`
- `VideoToFramesWorker.run()`
- `Video2MidiApp._handle_video_to_frames_request()` -> `VideoToFramesController.handle_request()`
- `Video2MidiApp._on_conversion_progress(message)` -> `VideoToFramesController.on_progress(message)`
- `Video2MidiApp._on_conversion_finished(success, message)` -> `VideoToFramesController.on_finished(success, message)`

Inputs and state read:

- `app.app_state.video.filepath`
- `app.control_panel.video_to_frames_button`
- filesystem state for the selected video path and sibling `<basename>_frames` directory
- FFmpeg availability from `synthesia2midi.utils.ffmpeg_helper`
- frame-series naming convention: `<original_video_basename>_frames/frame_%06d.jpg`

Outputs and state written:

- Creates or overwrites the frame-series output directory.
- Runs FFmpeg with `-q:v` derived from quality `90` and `-vf format=bgr24`.
- Disables the control-panel button while conversion runs and restores its label to `Reset Video -> Frame Series` on completion.
- Stores the active worker reference on the controller; mirror it to `app.video_to_frames_worker` only as a temporary compatibility bridge.
- Emits/logs progress and shows success/failure `QMessageBox` dialogs.

Qt object and signal ownership concerns:

- `VideoToFramesWorker` remains a `QThread` with `progress_updated = Signal(str)` and `conversion_finished = Signal(bool, str)`.
- The controller must keep a strong reference to the worker until `conversion_finished` fires; otherwise the thread can be garbage-collected.
- `on_finished` must call `deleteLater()`, clear both controller and compatibility references, and re-enable the button for success and failure paths.
- Signal source is `ControlPanelQt.video_to_frames_requested`; after the extraction, prefer `ControlSignalManager` wiring to `mw.video_session_ui_controller.handle_video_to_frames_request` or directly to `mw.video_to_frames_controller.handle_request`, not back to `main.py`.

Behavior to preserve with characterization checks:

- No loaded video shows the existing `Video to Frames` warning and does not create a worker.
- Missing FFmpeg shows the existing install guidance and does not create a worker.
- If a loaded path is a `_frames` directory, the workflow searches for the sibling original video file with extensions `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`.
- If a directory is loaded but is not a frame-series directory, warn and abort.
- If the selected video path is invalid, warn and abort.
- Confirmation dialog text includes source video and output path and defaults to `No`.
- Button disabled/enabled lifecycle is identical around the running worker.
- Success counts written `frame_*.jpg` files and reports the output directory.
- Failure preserves the special `Is a directory` and `No such file or directory` messages, plus stderr tail handling.

Suggested first tests/smokes:

- Unit-test the controller with a fake app/control panel and monkeypatched FFmpeg helper.
- Unit-test worker command construction with `subprocess.run` monkeypatched.
- Offscreen app smoke to verify signal wiring still instantiates.

## Group 2 — Rust MIDI touch-up editor process controller

Rationale: this group is cohesive process lifecycle code. It has high Qt lifetime risk, but it is isolated from detection/video-load behavior and has a clear owner: a controller responsible for `QProcess` references and result parsing.

Destination:

- `synthesia2midi/synthesia2midi/gui/midi_touchup_controller.py`
- Class: `MidiTouchupController`
- `main.py` should retain only construction (`self.midi_touchup_controller = MidiTouchupController(self)`) and app-close delegation.

Exact methods/functions to move from historical `main.py`:

- `Video2MidiApp._show_conversion_complete_dialog_with_touchup(midi_output_path)` -> `MidiTouchupController.show_conversion_complete_dialog(midi_output_path)`
- `Video2MidiApp._open_midi_touchup_editor_from_picker()` -> `MidiTouchupController.open_from_picker()`
- `Video2MidiApp._open_midi_touchup_editor(midi_path)` -> `MidiTouchupController.open_editor(midi_path)`
- `Video2MidiApp._resolve_midi_touchup_binary_path()` -> `MidiTouchupController.resolve_binary_path()`
- `Video2MidiApp._show_midi_touchup_setup_dialog(midi_path)` -> `MidiTouchupController.show_setup_dialog(midi_path)`
- `Video2MidiApp._handle_midi_touchup_process_finished(process, source_midi_path, exit_code)` -> `MidiTouchupController.handle_process_finished(process, source_midi_path, exit_code)`
- `Video2MidiApp._cleanup_midi_touchup_process(process)` -> `MidiTouchupController.cleanup_process(process)`
- `Video2MidiApp._remove_midi_touchup_process_ref(process)` -> `MidiTouchupController.remove_process_ref(process)`
- `Video2MidiApp._shutdown_midi_touchup_processes()` -> `MidiTouchupController.shutdown_processes()`

Inputs and state read:

- MIDI path from conversion result or file picker.
- Repo root for `videos/` picker default and Rust binary path lookup.
- `app._is_closing` to suppress user-facing dialogs during shutdown.
- `QProcess` stdout/stderr, including the final JSON result line.
- OS name for executable filename and setup command wording.

Outputs and state written:

- Launches the Rust editor with `--midi <path> --result-json --theme neothesia`.
- Tracks live `QProcess` objects in `MidiTouchupController.processes`; `Video2MidiApp` does not mirror process refs.
- Shows missing-file, missing-binary, launch-failure, saved, and failure dialogs.
- Removes process references and deletes processes on finish or shutdown.

Qt object and signal ownership concerns:

- Parent each `QProcess` to the main app/window so Qt owns the process object lifetime.
- Keep a Python reference in the controller list until finished/destroyed.
- Connect `process.destroyed` to reference cleanup.
- Connect `process.finished` with a closure that captures both the process and the source MIDI path.
- `cleanup_process` must tolerate already-deleted process handles (`RuntimeError`) and must terminate/kill before `deleteLater()`.
- `closeEvent` must set `app._is_closing = True` before `shutdown_processes()` so process-finish dialogs do not appear while the app exits.

Behavior to preserve with characterization checks:

- Conversion success dialog offers `Open Touch-Up Editor` and `Done`; only the open button launches the editor.
- File picker starts in repo `videos/` if it exists, otherwise user home.
- Missing MIDI path shows `MIDI file not found` and does not launch a process.
- Missing Rust binary shows setup guidance with `python3 setup_env.py` on macOS/Linux and `py setup_env.py` on Windows.
- Binary resolution checks both `midi-touchup-editor` and the alternate crate-name binary.
- Launch failure cleans up the process ref and suppresses the dialog if `_is_closing` is true.
- Finished process with JSON `status == "saved"` and exit code `0` shows the saved-path dialog.
- Finished process with JSON `status == "cancelled"` and exit code `0` logs and shows no dialog.
- Failure path includes source MIDI, stdout, and stderr in the error dialog.
- App close shuts down all live touch-up processes without leaking refs.

Suggested first tests/smokes:

- Unit-test binary resolution with temporary executable paths.
- Unit-test stdout JSON parsing for `saved`, `cancelled`, malformed output, and nonzero exit.
- Offscreen app smoke plus direct `closeEvent`/shutdown test with a fake process if possible.
- `cargo check --manifest-path tools/midi_touchup_editor_rust/Cargo.toml` when Rust code or binary contract is touched.

## Group 3 — Video session loading coordinator and file/YouTube entrypoints

Rationale: the local-file and YouTube paths historically duplicated the same fragile post-load ordering. Extract this before calibration or detection controllers so later work has one canonical session-setup path. Keep trim/range handlers out of this first pass unless tests already cover them.

Destinations:

- `synthesia2midi/synthesia2midi/gui/video_session_ui_controller.py` for file/YouTube dialogs and thin UI entrypoints.
- `synthesia2midi/synthesia2midi/workflows/video_session_coordinator.py` for shared post-load session wiring.
- `main.py` should construct `self.video_session_ui_controller` and `self.video_session_coordinator`, then wire menu/startup actions to the UI controller.

Exact methods/functions to move or split from historical `main.py`:

- `Video2MidiApp._show_youtube_download_dialog()` -> `VideoSessionUiController.show_youtube_download_dialog()`
- `Video2MidiApp._open_video_file()` -> `VideoSessionUiController.open_video_file()` plus shared coordinator call
- `Video2MidiApp._handle_youtube_video_downloaded(filepath)` -> `VideoSessionUiController.handle_youtube_video_downloaded(filepath)` plus shared coordinator call
- duplicated post-load body inside `_open_video_file` and `_handle_youtube_video_downloaded` -> `VideoSessionCoordinator.load_path(filepath, log_prefix, update_fps_display)` and `VideoSessionCoordinator.apply_loaded_session(video_session, log_prefix, update_fps_display)`
- helper logic derived from those bodies -> `VideoSessionCoordinator._close_existing_session()`, `_set_canvas_video_session()`, `_initialize_video_bound_workflows()`, `_apply_loaded_configuration_ui()`

Do not include in this first extraction unless characterization tests already exist:

- `_handle_start_frame_change`, `_handle_end_frame_change`
- `_handle_processing_start_frame_change`, `_handle_processing_end_frame_change`
- `_handle_trim_video_request`
- `_initialize_processing_range_defaults`
- frame-navigation menu wrappers (`_update_nav_interval`, `_handle_frame_nav_interval`)

Inputs and state read:

- Selected local path from `QFileDialog` or downloaded path from `YouTubeDownloadDialog.video_downloaded`.
- `app.video_session`, `app.app_state`, `app.state_manager`, `app.video_loading_workflow`, `app.config_manager`.
- Existing UI objects: `keyboard_canvas`, `video_controls`, `control_panel`, `window_manager`.
- Video info from `VideoLoadingWorkflow.get_video_info()`, especially whether a video-specific `.ini` loaded.

Outputs and state written:

- Closes any existing `VideoSession` before loading a new path.
- Resets app state to defaults before load.
- Assigns `app.video_session` and updates `VideoControls` and `KeyboardCanvas` with the new session.
- Updates frame slider limits, control-panel frame limits, optional FPS display, trim controls, conversion/wizard button enablement, and current frame display.
- Initializes video-bound workflows: `CalibrationWorkflow`, `AutoCalibrationWorkflow`, `DetectionManager`, and `ConversionWorkflow`.
- Recreates `keyboard_canvas.detect_pressed_func` using the current detection wrapper.
- Clears overlays and resets `unsaved_changes` in the no-config branch.
- Resizes/repositions the window after the loaded session is applied.

Qt object and signal ownership concerns:

- `QFileDialog` and `YouTubeDownloadDialog` stay parented to the app/main window.
- `YouTubeDownloadDialog.video_downloaded` should connect to `VideoSessionUiController.handle_youtube_video_downloaded`.
- File menu `Open Video` and startup-dialog `open_local_file` should connect to `VideoSessionUiController.open_video_file`.
- File menu `Download Youtube Video` and startup-dialog `download_from_youtube` should connect to `VideoSessionUiController.show_youtube_download_dialog`.
- `ControlSignalManager` should connect `control_panel.youtube_video_downloaded` to the UI controller, not to `main.py`.
- `DetectionManager` receives `app._update_current_frame_display` as a temporary adapter; do not move this adapter until display/update facade ownership is separately characterized.

Behavior to preserve with characterization checks:

- Cancelling the startup, local-file, or YouTube dialogs leaves the app open with no loaded video and logs the same cancellation intent.
- Local open path uses the non-native dialog, permits file or image-sequence directory selection, starts at the project root, and keeps single-selection behavior in list/tree views.
- YouTube dialog defaults output to repo `videos/` and loads the downloaded file after `video_downloaded` fires.
- Existing video session is closed before state reset and new load.
- Failed load returns without partially initializing a new session.
- Local-file load updates FPS display; YouTube load preserves the historical no-FPS-update difference unless a separate behavior-change task says otherwise.
- Config-loaded branch updates controls from state, trim controls, processing range defaults, initial frame, convert button, wizard button, and window size.
- No-config branch clears overlays, marks no unsaved changes, updates controls/trim controls, displays the correct initial frame, disables convert, enables wizard, and resizes.
- The order of `video_session` assignment, `VideoControls.set_video_session`, `KeyboardCanvas.set_video_session`, workflow initialization, detection wrapper refresh, and initial display remains stable.

Suggested first tests/smokes:

- Unit-test `VideoSessionCoordinator.apply_loaded_session()` against a fake app to assert call order and branch behavior for config-loaded vs no-config.
- Unit-test `VideoSessionUiController.open_video_file()` with a fake dialog or isolate path handling behind a small helper.
- Offscreen `Video2MidiApp` smoke to verify menu/startup/signal wiring still instantiates.
- A manual GUI smoke remains useful for loading a local video and confirming frame slider/range controls after the extraction.

## Follow-on groups intentionally deferred

After these first three groups, the next extraction candidates should be planned separately:

1. MIDI conversion UI flow: `_start_conversion_process` and conversion-result UI into `MidiConversionController`.
2. Calibration wizard and auto-detect tuning dialog cluster.
3. Calibration interaction, spark/shadow ROI, and overlay mutation controllers.
4. Detection parameter/live-feedback/menu action wrappers into `MainActionController` or focused controllers.
5. Thin compatibility API audit: keep only `UIUpdateInterface` methods and true app-shell lifecycle methods in `main.py`.

Keeping these deferred avoids mixing process lifecycle, session ordering, calibration state, and detection behavior in one checkpoint.
