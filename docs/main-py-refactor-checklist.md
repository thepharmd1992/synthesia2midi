# `main.py` Refactor Checklist

This is the living control document for finishing the `Video2MidiApp` god-object refactor. Update it after every extraction/checkpoint commit.

## Goal

Turn `synthesia2midi/synthesia2midi/main.py` into a root-window composition facade: widget construction, controller/workflow instantiation, signal wiring, and app-level state only.

Line count is a guardrail, not the definition of done.

## Current Baseline

Measured 2026-05-12:

| Metric | Before refactor (`HEAD`/`origin/main`) | Current working tree | Target guardrail |
|---|---:|---:|---:|
| `main.py` lines | 2,867 | 1,077 | ~500-700 if remaining code is UI composition only |
| `Video2MidiApp` methods | 121 | 117 | materially fewer; remaining methods should be shell/wiring or documented public API |
| `Video2MidiApp` classes | 2 | 1 | 1 |

Interpretation: the first pass moved large implementation bodies out, but `main.py` still has too many compatibility wrappers and mixed responsibilities.

## Definition of Done

This refactor is done when all of these are true:

- [ ] `main.py` is at or below the 500-700 line guardrail, or any excess is justified in this file.
- [ ] `Video2MidiApp` no longer owns video loading, conversion, calibration, detection, overlay mutation, MIDI touch-up, or trim workflow bodies.
- [ ] Remaining methods in `Video2MidiApp` are classified as one of:
  - UI construction/composition
  - signal/menu/hotkey wiring
  - app lifecycle (`resizeEvent`, `showEvent`, `closeEvent`)
  - intentionally public compatibility API with known callers
- [ ] Every compatibility wrapper has been either deleted or documented with its caller and removal condition.
- [ ] Tests and smoke checks pass after each extraction checkpoint.
- [ ] A small git commit exists after each completed subsystem checkpoint.

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

- [ ] `_init_ui` — 179 lines. Split into smaller UI builder methods/classes if it remains bulky after workflow wrappers are removed.
- [ ] `__init__` — 54 lines. Acceptable only if it remains controller construction and state setup.
- [ ] `_bind_hotkeys` — 27 lines. Keep if it is pure shortcut wiring; otherwise move to a hotkey binder helper.
- [ ] `closeEvent` — 23 lines. Keep lifecycle shell; delegate cleanup to owned controllers.
- [ ] `_show_startup_dialog` — 11 lines. Decide whether this belongs in shell or a dialog helper.
- [ ] `_capture_window_screenshot` — 38 lines. Move to a screenshot/debug utility or remove if not product behavior.

Verification:

- [ ] Offscreen `Video2MidiApp` smoke passes.
- [ ] Hotkeys still bind without exceptions.
- [ ] Window close still shuts down touch-up/process resources.

### 2. Video Session / Loading / Trimming

Goal: all video/session operations live in `VideoLoadingWorkflow`, `VideoSessionCoordinator`, or a dedicated trim/range controller. `main.py` should only connect UI events to those objects.

Remaining methods to classify/extract/delete wrappers:

- [ ] `_open_video_file` — 74 lines; likely should be moved almost entirely to video session workflow/controller.
- [ ] `_show_youtube_download_dialog` — 14 lines.
- [ ] `_handle_youtube_video_downloaded` — 9 lines.
- [ ] `_handle_video_to_frames_request` — 3 lines.
- [ ] `_update_frame_nav_interval` — 8 lines.
- [ ] `_handle_frame_nav_interval` — 15 lines.
- [ ] `_update_frame_slider_for_video` — 3 lines.
- [ ] `_handle_start_frame_change` — 9 lines.
- [ ] `_handle_end_frame_change` — 9 lines.
- [ ] `_handle_processing_start_frame_change` — 29 lines.
- [ ] `_handle_processing_end_frame_change` — 29 lines.
- [ ] `_handle_trim_video_request` — 38 lines.
- [ ] `_initialize_processing_range_defaults` — 24 lines.
- [ ] `get_video_session` — 3 lines; keep only if external callers need it.
- [ ] `has_video_loaded` — 3 lines; keep only if external callers need it.
- [ ] `get_total_frames` — 3 lines; keep only if external callers need it.

Verification:

- [ ] Existing import/app smoke passes.
- [ ] Tests cover session/range helpers before deleting wrappers.
- [ ] Manual smoke: load a local video and frame slider/range controls still update correctly.

### 3. Conversion / MIDI Export

Goal: conversion/MIDI orchestration belongs to `ConversionWorkflow`; `main.py` should not decide conversion behavior.

Remaining methods:

- [x] `_start_conversion_process` — deleted; `ControlSignalManager` and the Space hotkey now call `MidiConversionController.start_conversion_process`.
- [x] `_on_conversion_progress` — deleted; no live callers.
- [x] `_on_conversion_finished` — deleted; no live callers.
- [x] `_show_conversion_complete_dialog_with_touchup` — deleted; conversion controller calls `MidiTouchupController` directly.

Verification:

- [ ] Conversion tests/smoke still pass.
- [ ] Generated MIDI save path and progress UI behavior unchanged.
- [ ] Touch-up prompt after conversion still appears when expected.

### 4. Calibration / Auto-Calibration / Exemplars

Goal: calibration state transitions live in calibration controllers/workflows; `main.py` keeps no calibration branching logic.

Remaining methods:

- [ ] `_handle_calibrate_unlit_all_keys`
- [ ] `_handle_calibrate_lit_exemplar_key_start`
- [ ] `_handle_spark_roi_selection_request`
- [ ] `_handle_spark_roi_visibility_toggle`
- [ ] `_handle_shadow_roi_selection_request`
- [ ] `_handle_shadow_white_roi_selection_request`
- [ ] `_handle_shadow_black_roi_selection_request`
- [ ] `_handle_spark_roi_updated`
- [ ] `_handle_spark_calibration_request`
- [ ] `_handle_auto_spark_calibration_request`
- [ ] `_handle_spark_detection_toggle`
- [ ] `_handle_spark_detection_sensitivity_change`
- [ ] `_handle_shadow_detection_toggle`
- [ ] `_handle_shadow_detection_sensitivity_change`
- [ ] `_handle_shadow_darkness_threshold_change`
- [ ] `_handle_shadow_calibration_request`
- [ ] `_capture_spark_background_calibration`
- [ ] `_capture_spark_overlay_calibration`
- [ ] `_capture_shadow_background_calibration`
- [ ] `_capture_shadow_overlay_calibration`
- [ ] `_apply_template_styles_to_overlays`
- [ ] `_handle_exemplar_key_type_enabled_change` — 22 lines; move into calibration/exemplar controller.

Verification:

- [ ] Calibration wizard smoke still instantiates.
- [ ] Existing calibration tests pass or are added before wrapper deletion.
- [ ] Manual smoke: unlit/lit calibration actions still update overlays and controls.

### 5. Overlay / Canvas / Keyboard Editing

Goal: overlay mutations and canvas refresh behavior live in `OverlayManager`, `KeyboardCanvas`, or focused GUI controllers.

Remaining methods:

- [ ] `_handle_overlay_selection`
- [ ] `_toggle_overlays`
- [ ] `_handle_overlay_type_change`
- [ ] `_handle_refresh_selected_overlay_display`
- [x] `_align_overlays_vertically` — deleted; no live callers.
- [ ] `_handle_align_white_keys_to_selected`
- [ ] `_handle_align_black_keys_to_selected`
- [x] `_handle_spinbox_overlay_size_change` — deleted; no live callers.
- [ ] `_handle_overlay_size_adjustment` — 10 lines.
- [ ] `_handle_overlay_color_change` — 10 lines.
- [ ] `_handle_octave_transpose_change` — 19 lines.
- [ ] `update_overlay_action`
- [ ] `refresh_canvas`
- [ ] `update_selected_overlay_display`
- [ ] `get_roi_bgr`

Verification:

- [ ] Overlay selection and size controls still update canvas/control panel.
- [ ] Alignment commands still preserve black/white key behavior.
- [ ] ROI extraction tests cover moved utilities.

### 6. Detection Parameters / Live Feedback / Monitor

Goal: detection parameter changes belong to `DetectionManager`, display/monitor controllers, or parameter manager.

Remaining methods:

- [x] `_prepare_frame_for_detection` — deleted; no live callers.
- [ ] `_toggle_live_detection_feedback`
- [ ] `_handle_visual_threshold_monitor_menu`
- [ ] `_handle_detection_threshold_change`
- [ ] `_handle_rise_delta_threshold_change`
- [ ] `_handle_fall_delta_threshold_change`
- [ ] `_on_toggle_hist_detection`
- [ ] `_on_toggle_delta_detection`
- [ ] `_on_toggle_winner_takes_black`
- [ ] `_handle_hand_assignment_toggle`
- [x] `_handle_visual_threshold_monitor_toggle` — deleted; no live callers.
- [ ] `_handle_fps_override_change` — 23 lines.
- [x] `_handle_detection_logging_toggle` — deleted; no live callers.
- [x] `_log_detection_parameters` — deleted with no-op callback registration.
- [ ] `_create_detection_wrapper`
- [ ] `update_live_detection_action`
- [ ] `update_detection_threshold`

Verification:

- [ ] Manual auto-detector characterization tests pass.
- [ ] Detection parameter UI changes still update detector/control state.
- [ ] Visual Threshold Monitor still opens/toggles as before.

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

- [ ] Missing-binary dialog still points to `python3 setup_env.py` on macOS/Linux and `py setup_env.py` on Windows.
- [ ] Process cleanup still happens on app close.

### 8. Thin Wrappers / Compatibility API Audit

Goal: delete wrappers once signal/control callers are rewired. Keep only wrappers that are true public app API.

Known thin wrappers needing caller search:

- [ ] `resizeEvent`
- [ ] `showEvent`
- [x] `_update_tempo` — deleted; no live callers.
- [x] `_navigate_frame_pgup` — deleted; hotkeys now call `video_controls.navigate_frame_pgup` directly.
- [x] `_navigate_frame_pgdn` — deleted; hotkeys now call `video_controls.navigate_frame_pgdn` directly.
- [x] `_display_frame_lightweight` — deleted; no live callers.
- [x] `_update_frame_slider_position` — deleted; no live callers.
- [x] `_update_time_display` — deleted; no live callers.
- [ ] `_display_frame_with_slider_update`
- [ ] `_update_current_frame_display`
- [ ] `_handle_color_pick`
- [x] `_extract_roi` — deleted; no live callers.
- [x] `_handle_keyboard_region_selection_request` — deleted; no live callers.
- [x] `_clone_auto_detect_tuning_context` — deleted; no live callers.
- [x] `_cache_auto_detect_tuning_context` — deleted; no live callers.
- [x] `_get_current_frame_rgb_for_tuning` — deleted; no live callers.
- [x] `_build_auto_detect_tuning_context_from_state` — deleted; no live callers.
- [ ] `update_control_panel`
- [ ] `_resize_and_position_window`
- [ ] `show_message`

Verification:

- [ ] For each deleted wrapper, `rg "wrapper_name" synthesia2midi tests` returns no live callers.
- [ ] If kept, this document records why and who calls it.

## Checkpoint Log

Add one row per meaningful checkpoint commit.

| Date | Commit | Scope | `main.py` lines | Verification | Notes |
|---|---|---|---:|---|---|
| 2026-05-12 | `e987c9c` | First extraction wave + setup cleanup | 1,272 | compileall, pytest, ruff critical selectors, cargo check | Still many wrappers; needs subsystem-by-subsystem wrapper deletion. |
| 2026-05-12 | `5036152` | Removed MIDI touch-up wrappers from `main.py` | 1,244 | `git diff --check`; compileall; pytest | `ControlSignalManager` and `closeEvent` now target `MidiTouchupController` directly. |
| 2026-05-12 | `a65a395` | Removed no-live-caller thin wrappers | 1,135 | `git diff --check`; compileall; pytest | Deleted stale wrappers and no-op detection logging callback; hotkeys now target `VideoControls` directly. |
| 2026-05-12 | pending | Extracted user-triggered MIDI conversion UI flow | 1,077 | pending | Added `MidiConversionController`; conversion signals/hotkey no longer target `main.py`. |

## Next Recommended Checkpoints

1. Commit current broad working tree before more extraction.
2. Tackle video/session loading wrappers first because it contains the largest remaining workflow bodies.
3. Tackle conversion/MIDI export next.
4. Tackle overlay/detection/calibration wrappers after adding or confirming focused tests.
5. Re-run this inventory after each subsystem and update the baseline table/checkpoint log.
