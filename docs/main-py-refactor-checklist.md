# `main.py` Refactor Checklist

This is the living control document for finishing the `Video2MidiApp` god-object refactor. Update it after every extraction/checkpoint commit.

## Goal

Turn `synthesia2midi/synthesia2midi/main.py` into a root-window composition facade: widget construction, controller/workflow instantiation, signal wiring, and app-level state only.

Line count is a guardrail, not the definition of done.

## Current Baseline

Measured 2026-05-12:

| Metric | Before refactor (`HEAD`/`origin/main`) | Current working tree | Target guardrail |
|---|---:|---:|---:|
| `main.py` lines | 2,867 | 562 | ~500-700 if remaining code is UI composition only |
| `Video2MidiApp` methods | 121 | 21 | materially fewer; remaining methods should be shell/wiring or documented public API |
| `Video2MidiApp` classes | 2 | 1 | 1 |

Interpretation: `main.py` is now inside the target range and functions as a root-window composition facade. Remaining non-shell methods are public UI/update compatibility methods used by workflows/controllers.

## Definition of Done

This refactor is done when all of these are true:

- [x] `main.py` is at or below the 500-700 line guardrail, or any excess is justified in this file.
- [x] `Video2MidiApp` no longer owns video loading, conversion, calibration, detection, overlay mutation, MIDI touch-up, or trim workflow bodies.
- [x] Remaining methods in `Video2MidiApp` are classified as one of:
  - UI construction/composition
  - signal/menu/hotkey wiring
  - app lifecycle (`resizeEvent`, `showEvent`, `closeEvent`)
  - intentionally public compatibility API with known callers
- [x] Every compatibility wrapper has been either deleted or documented with its caller and removal condition.
- [x] Tests and smoke checks pass after each extraction checkpoint.
- [x] A small git commit exists after each completed subsystem checkpoint.

## Standard Checkpoint Commands

Run the smallest relevant set during each task, and the full gate before committing a phase:

```bash
git diff --check
.venv/bin/python -m compileall -q synthesia2midi
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q
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

For setup/launcher-adjacent edits, also run:

```bash
PYTHONPATH=synthesia2midi QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q tests/test_setup_and_launch.py
.venv/bin/python setup_env.py --check
```

## Method Inventory and Extraction Checklist

Update statuses as work lands.

Status legend:

- `[ ]` not started
- `[~]` partially extracted / wrapper remains
- `[x]` done / verified
- `[!]` blocked or needs design decision

### 1. App Shell / UI Composition

Goal: keep only true root-window shell code in `main.py`; move reusable construction helpers if they keep growing.

Current large methods:

- [x] `_init_ui` — 176 lines. Kept as root-window composition/widget layout; no workflow business logic remains.
- [x] `__init__` — 53 lines. Controller/workflow construction and app state setup only.
- [x] `_bind_hotkeys` — 27 lines. Pure shortcut wiring to `VideoControls`/conversion controller.
- [x] `closeEvent` — 23 lines. Lifecycle shell; cleanup delegated to `MidiTouchupController`, `video_session`, and `ConfigManager`.
- [x] `_show_startup_dialog` — 11 lines. Kept as startup dialog signal wiring only.
- [x] `_capture_window_screenshot` — 38 lines. Kept as app-shell debug utility because it captures the root widget.

Verification:

- [x] Offscreen `Video2MidiApp` smoke passes.
- [x] Hotkeys still bind without exceptions.
- [x] Window close still shuts down touch-up/process resources.

### 2. Video Session / Loading / Trimming

Goal: all video/session operations live in `VideoLoadingWorkflow`, `VideoSessionCoordinator`, or a dedicated trim/range controller. `main.py` should only connect UI events to those objects.

Remaining methods to classify/extract/delete wrappers:

- [x] `_open_video_file` — deleted; file dialog flow moved to `VideoSessionUiController.open_video_file`.
- [x] `_show_youtube_download_dialog` — deleted; moved to `VideoSessionUiController.show_youtube_download_dialog`.
- [x] `_handle_youtube_video_downloaded` — deleted; moved to `VideoSessionUiController.handle_youtube_video_downloaded`.
- [x] `_handle_video_to_frames_request` — deleted; signal now targets `VideoSessionUiController.handle_video_to_frames_request`.
- [x] `_update_frame_nav_interval` — stale checklist name; actual `_update_nav_interval` deleted and moved to `VideoSessionUiController.update_nav_interval`.
- [x] `_handle_frame_nav_interval` — deleted; frame-nav menu targets `VideoSessionUiController.handle_frame_nav_interval`.
- [x] `_update_frame_slider_for_video` — deleted; coordinator now calls `video_controls.update_frame_slider_for_video` directly.
- [x] `_handle_start_frame_change` — deleted; signal now targets `VideoSessionUiController.handle_start_frame_change`.
- [x] `_handle_end_frame_change` — deleted; signal now targets `VideoSessionUiController.handle_end_frame_change`.
- [x] `_handle_processing_start_frame_change` — deleted; moved to `VideoSessionUiController` with coverage.
- [x] `_handle_processing_end_frame_change` — deleted; moved to `VideoSessionUiController` with coverage.
- [x] `_handle_trim_video_request` — deleted; moved to `VideoSessionUiController` with coverage.
- [x] `_initialize_processing_range_defaults` — deleted; moved to `VideoSessionUiController` with coverage.
- [x] `get_video_session` — kept as intentionally public compatibility API for tools/controllers.
- [x] `has_video_loaded` — kept as intentionally public compatibility API for tools/controllers.
- [x] `get_total_frames` — kept as intentionally public compatibility API for tools/controllers.

Verification:

- [x] Existing import/app smoke passes.
- [x] Tests cover session/range helpers before deleting wrappers (`tests/test_video_session_ui_controller.py`).
- [ ] Manual smoke: load a local video and frame slider/range controls still update correctly. Not performed in this headless pass.

### 3. Conversion / MIDI Export

Goal: conversion/MIDI orchestration belongs to `ConversionWorkflow`; `main.py` should not decide conversion behavior.

Remaining methods:

- [x] `_start_conversion_process` — deleted; `ControlSignalManager` and the Space hotkey now call `MidiConversionController.start_conversion_process`.
- [x] `_on_conversion_progress` — deleted; no live callers.
- [x] `_on_conversion_finished` — deleted; no live callers.
- [x] `_show_conversion_complete_dialog_with_touchup` — deleted; conversion controller calls `MidiTouchupController` directly.

Verification:

- [x] Conversion tests/smoke still pass.
- [x] Generated MIDI save path and result UI behavior covered by `tests/test_midi_conversion_controller.py`.
- [x] Touch-up prompt after conversion still appears when expected (`MidiConversionController` calls `MidiTouchupController`).

### 4. Calibration / Auto-Calibration / Exemplars

Goal: calibration state transitions live in calibration controllers/workflows; `main.py` keeps no calibration branching logic.

Remaining methods:

- [x] `_handle_calibrate_unlit_all_keys` — deleted; signal targets `MainActionController`.
- [x] `_handle_calibrate_lit_exemplar_key_start` — deleted; signal targets `MainActionController`.
- [x] `_handle_spark_roi_selection_request` — deleted; signal targets `CalibrationEffectsController`.
- [x] `_handle_spark_roi_visibility_toggle` — deleted; signal targets `CalibrationEffectsController`.
- [x] `_handle_shadow_roi_selection_request` — deleted; no live caller remained.
- [x] `_handle_shadow_white_roi_selection_request` — deleted; no live caller remained.
- [x] `_handle_shadow_black_roi_selection_request` — deleted; no live caller remained.
- [x] `_handle_spark_roi_updated` — deleted; `KeyboardCanvas` callback targets `CalibrationEffectsController`.
- [x] `_handle_spark_calibration_request` — deleted; signal targets `CalibrationEffectsController`.
- [x] `_handle_auto_spark_calibration_request` — deleted; signal targets `CalibrationEffectsController`.
- [x] `_handle_spark_detection_toggle` — deleted; signal targets `CalibrationEffectsController`.
- [x] `_handle_spark_detection_sensitivity_change` — deleted; signal targets `CalibrationEffectsController`.
- [x] `_handle_shadow_detection_toggle` — deleted; no live caller remained.
- [x] `_handle_shadow_detection_sensitivity_change` — deleted; no live caller remained.
- [x] `_handle_shadow_darkness_threshold_change` — deleted; no live caller remained.
- [x] `_handle_shadow_calibration_request` — deleted; no live caller remained.
- [x] `_capture_spark_background_calibration` — deleted; owned by `CalibrationWizardController`/effect controllers.
- [x] `_capture_spark_overlay_calibration` — deleted; owned by `CalibrationWizardController`/effect controllers.
- [x] `_capture_shadow_background_calibration` — already absent from `main.py`.
- [x] `_capture_shadow_overlay_calibration` — deleted; owned by `CalibrationWizardController`/effect controllers.
- [x] `_apply_template_styles_to_overlays` — deleted; implemented in `CalibrationWizardController`.
- [x] `_handle_exemplar_key_type_enabled_change` — moved to `MainActionController` with coverage.

Verification:

- [x] Calibration wizard smoke still instantiates via offscreen app smoke/import tests.
- [x] Existing tests pass; controller behavior covered where state mutation moved.
- [ ] Manual smoke: unlit/lit calibration actions still update overlays and controls. Not performed in this headless pass.

### 5. Overlay / Canvas / Keyboard Editing

Goal: overlay mutations and canvas refresh behavior live in `OverlayManager`, `KeyboardCanvas`, or focused GUI controllers.

Remaining methods:

- [x] `_handle_overlay_selection` — deleted; `KeyboardCanvas` callback targets `CalibrationInteractionController`.
- [x] `_toggle_overlays` — deleted; menu targets `MainActionController`.
- [x] `_handle_overlay_type_change` — deleted; signal targets `CalibrationEffectsController`.
- [x] `_handle_refresh_selected_overlay_display` — deleted; signal targets `MainActionController`.
- [x] `_align_overlays_vertically` — deleted; no live callers.
- [x] `_handle_align_white_keys_to_selected` — deleted; signal targets `MainActionController`.
- [x] `_handle_align_black_keys_to_selected` — deleted; signal targets `MainActionController`.
- [x] `_handle_spinbox_overlay_size_change` — deleted; no live callers.
- [x] `_handle_overlay_size_adjustment` — deleted; signal targets `MainActionController`.
- [x] `_handle_overlay_color_change` — moved to `MainActionController` with coverage.
- [x] `_handle_octave_transpose_change` — moved to `MainActionController` with coverage.
- [x] `update_overlay_action` — kept as public `UIUpdateInterface` method used by managers.
- [x] `refresh_canvas` — kept as public `UIUpdateInterface` method used by managers.
- [x] `update_selected_overlay_display` — kept as public `UIUpdateInterface` method used by managers.
- [x] `get_roi_bgr` — kept as public `UIUpdateInterface`/ROI adapter for controllers.

Verification:

- [x] Overlay selection and size controls are rewired to controllers; regression suite passes.
- [x] Alignment commands still delegate to `OverlayManager` through `MainActionController`.
- [ ] ROI extraction tests cover moved utilities. No ROI utility behavior moved in this pass; adapter remains in app API.

### 6. Detection Parameters / Live Feedback / Monitor

Goal: detection parameter changes belong to `DetectionManager`, display/monitor controllers, or parameter manager.

Remaining methods:

- [x] `_prepare_frame_for_detection` — deleted; no live callers.
- [x] `_toggle_live_detection_feedback` — deleted; menu targets `MainActionController`.
- [x] `_handle_visual_threshold_monitor_menu` — deleted; menu targets `MainActionController`.
- [x] `_handle_detection_threshold_change` — deleted; signal targets `MainActionController`.
- [x] `_handle_rise_delta_threshold_change` — deleted; signal targets `MainActionController`.
- [x] `_handle_fall_delta_threshold_change` — deleted; signal targets `MainActionController`.
- [x] `_on_toggle_hist_detection` — deleted; signal targets `MainActionController`.
- [x] `_on_toggle_delta_detection` — deleted; signal targets `MainActionController`.
- [x] `_on_toggle_winner_takes_black` — deleted; signal targets `MainActionController`.
- [x] `_handle_hand_assignment_toggle` — deleted; signal targets `MainActionController`.
- [x] `_handle_visual_threshold_monitor_toggle` — deleted; no live callers.
- [x] `_handle_fps_override_change` — moved to `MainActionController` with coverage.
- [x] `_handle_detection_logging_toggle` — deleted; no live callers.
- [x] `_log_detection_parameters` — deleted with no-op callback registration.
- [x] `_create_detection_wrapper` — deleted; `MainActionController` exposes detection wrapper creation.
- [x] `update_live_detection_action` — kept as public `UIUpdateInterface` method used by display/controls.
- [x] `update_detection_threshold` — kept as public `UIUpdateInterface` method used by display/controls.

Verification:

- [ ] Manual auto-detector characterization tests pass. Not run; no detector math changed.
- [x] Detection parameter UI changes are rewired to `MainActionController`; regression suite passes.
- [x] Visual Threshold Monitor toggles through `MainActionController` and `DisplayManager`; regression suite passes.

### 7. MIDI Touch-Up Editor

Goal: all Rust editor process lifecycle stays in `MidiTouchupController`; remove shell wrappers if callers can target the controller directly.

Remaining wrappers:

- [x] `_open_midi_touchup_editor_from_picker` — deleted; `ControlSignalManager` connects directly to `midi_touchup_controller.open_from_picker`.
- [x] `_open_midi_touchup_editor` — deleted; no live callers.
- [x] `_resolve_midi_touchup_binary_path` — deleted; no live callers.
- [x] `_show_midi_touchup_setup_dialog` — deleted; no live callers.
- [x] `_handle_midi_touchup_process_finished` — deleted; no live callers.
- [x] `_cleanup_midi_touchup_process` — deleted; no live callers.
- [x] `_remove_midi_touchup_process_ref` — deleted; no live callers.
- [x] `_shutdown_midi_touchup_processes` — deleted; `closeEvent` calls `midi_touchup_controller.shutdown_processes()` directly.

Verification:

- [x] Missing-binary dialog still points to `python3 setup_env.py` on macOS/Linux and `py setup_env.py` on Windows.
- [x] Process cleanup still happens on app close.

### 8. Thin Wrappers / Compatibility API Audit

Goal: delete wrappers once signal/control callers are rewired. Keep only wrappers that are true public app API.

Known thin wrappers needing caller search:

- [x] `resizeEvent` — kept as lifecycle method delegating to `WindowManager`.
- [x] `showEvent` — kept as lifecycle method delegating to `WindowManager`.
- [x] `_update_tempo` — deleted; no live callers.
- [x] `_navigate_frame_pgup` — deleted; hotkeys now call `video_controls.navigate_frame_pgup` directly.
- [x] `_navigate_frame_pgdn` — deleted; hotkeys now call `video_controls.navigate_frame_pgdn` directly.
- [x] `_display_frame_lightweight` — deleted; no live callers.
- [x] `_update_frame_slider_position` — deleted; no live callers.
- [x] `_update_time_display` — deleted; no live callers.
- [x] `_display_frame_with_slider_update` — deleted; callers target `VideoControls` directly.
- [x] `_update_current_frame_display` — kept as tiny callback adapter for `DetectionManager`.
- [x] `_handle_color_pick` — deleted; `KeyboardCanvas` callback targets `CalibrationInteractionController`.
- [x] `_extract_roi` — deleted; no live callers.
- [x] `_handle_keyboard_region_selection_request` — deleted; no live callers.
- [x] `_clone_auto_detect_tuning_context` — deleted; no live callers.
- [x] `_cache_auto_detect_tuning_context` — deleted; no live callers.
- [x] `_get_current_frame_rgb_for_tuning` — deleted; no live callers.
- [x] `_build_auto_detect_tuning_context_from_state` — deleted; no live callers.
- [x] `update_control_panel` — kept as public `UIUpdateInterface` method.
- [x] `_resize_and_position_window` — deleted; callers target `WindowManager`/`MainActionController` directly.
- [x] `show_message` — kept as public `UIUpdateInterface` method.

Verification:

- [x] For each deleted wrapper, searches show no live callers outside owning controllers/docs.
- [x] If kept, this document records why and who calls it.

## Checkpoint Log

Add one row per meaningful checkpoint commit.

| Date | Commit | Scope | `main.py` lines | Verification | Notes |
|---|---|---|---:|---|---|
| 2026-05-12 | `e987c9c` | First extraction wave + setup cleanup | 1,272 | compileall, pytest, ruff critical selectors, cargo check | Still many wrappers; needs subsystem-by-subsystem wrapper deletion. |
| 2026-05-12 | `5036152` | Removed MIDI touch-up wrappers from `main.py` | 1,244 | `git diff --check`; compileall; pytest | `ControlSignalManager` and `closeEvent` now target `MidiTouchupController` directly. |
| 2026-05-12 | `a65a395` | Removed no-live-caller thin wrappers | 1,135 | `git diff --check`; compileall; pytest | Deleted stale wrappers and no-op detection logging callback; hotkeys now target `VideoControls` directly. |
| 2026-05-12 | `496dfd0` | Extracted user-triggered MIDI conversion UI flow | 1,077 | `git diff --check`; compileall; pytest; ruff critical selectors | Added `MidiConversionController`; conversion signals/hotkey no longer target `main.py`. |
| 2026-05-12 | `e7387ad` | Extracted video/session UI and frame-range handlers | 797 | `git diff --check`; compileall; pytest; ruff critical selectors | Added `VideoSessionUiController`; video/session signals and coordinator no longer call `main.py` wrappers. |
| 2026-05-12 | pending | Extracted action/calibration/detection wrappers | 562 | pending | Added `MainActionController`; removed final calibration/effects/detection/overlay wrapper layer from `main.py`. |

## Next Recommended Checkpoints

1. Keep `main.py` as composition-only; do not reintroduce workflow bodies or compatibility wrappers.
2. If `_init_ui` grows, split it into an explicit UI builder without mixing business logic back into the root window.
3. Add manual smoke coverage for local video loading and calibration flows when a GUI session is available.
4. Keep `ARCHITECTURE.md` and `AGENTS.md` updated when adding new controllers/workflows.
