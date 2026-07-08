# Task 3 Report: Overlay Quick-Adjustment Values And Reset

## Outcome
- Status: complete
- Commit: `f1f9abf` - `Show overlay adjustment values and reset controls`

## RED Evidence
Ran the new focused test before implementation:

```bash
.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls -v
```

Failure:
- `AttributeError: 'ControlPanelQt' object has no attribute 'left_slant_value_label'`

This confirmed the test was exercising the missing visible value and reset controls, not an unrelated path.

## GREEN Evidence
Ran the focused checks after implementation:

```bash
.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls -v
.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls tests/test_main_window_layout.py::test_overlay_size_controls_stack_for_narrow_settings_window -v
.venv/bin/python -m pytest tests/test_main_window_layout.py -v
```

Results:
- Focused overlay quick-adjust test: pass
- Brief-required layout pair: pass
- Full layout file: `13 passed`

## Files Changed
- `synthesia2midi/synthesia2midi/gui/controls_qt.py`
- `tests/test_main_window_layout.py`

## Self-Review
- Added transient overlay adjustment state inside `ControlPanelQt` so each control can display its current offset locally.
- Wired all six overlay size controls through `_apply_overlay_adjustment(...)` so the emitted `overlay_size_adjustment_requested(str, str, int)` deltas stay consistent with the brief.
- Added reset buttons and value labels without changing overlay manager semantics beyond the same delta emissions.
- Kept the change scoped to the control panel and the layout test; no unrelated calibration or readiness behavior was touched.

## Concerns
- The full layout file passed, but the UI change only has direct test coverage for the slant controls. The white/black overlay rows are wired through the same helper and should behave the same way, but they are not individually asserted in this task.
- The test run still prints an existing Qt font deprecation warning from `font_helper.py`; it is unrelated to this change.

## Fix
- Added shared quick-adjust coverage in `tests/test_main_window_layout.py` for one white control (`white_width`) and one black control (`black_height`), asserting the value label, reset button text, and inverse-delta reset emission.
- Regenerated `synthesia2midi_{es,ja,ko,pt_BR,ru,zh_CN}.ts` with `pyside6-lupdate -no-obsolete` and rebuilt the matching `.qm` files with `pyside6-lrelease` so `Current:` and `Reset` are present in production translation assets.
- Refreshed `docs/localization/translation-agent-packet.json` and `docs/localization/ui-string-manifest.json` from the updated source strings.
- Verification:
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls tests/test_main_window_layout.py::test_overlay_size_controls_stack_for_narrow_settings_window tests/test_main_window_layout.py::test_overlays_tab_exposes_white_and_black_quick_adjust_controls -v` -> `3 passed, 1 warning`
  - `.venv/bin/python -m pytest tests/test_localization.py tests/test_ui_string_audit.py -v` -> `22 passed`
- Commit: `6c08488` (`Cover overlay quick-adjust localization`)

## Fix
- Removed the hard `72 px` maximum width from the shared overlay reset button helper in `synthesia2midi/synthesia2midi/gui/controls_qt.py` so translated `Reset` labels can use their natural size.
- Extended `tests/test_main_window_layout.py` to assert the reset button is not width-capped too tightly by checking `maximumWidth()` stays unbounded for the overlay quick-adjust controls.
- Verification:
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls tests/test_main_window_layout.py::test_overlays_tab_exposes_white_and_black_quick_adjust_controls -v` -> `2 passed in 0.47s`

## Fix
- Reviewer issue addressed: overlay quick-adjust labels and reset now only advance from controller-confirmed steps, instead of assuming every requested delta actually changed overlay geometry.
- `synthesia2midi/synthesia2midi/workflows/overlay_manager.py` now rejects quick-adjust requests that would partially apply because a width/height target would drop below `1 px` or a slant step would clamp past `[-45, 45]`.
- `synthesia2midi/synthesia2midi/gui/main_action_controller.py` now confirms successful overlay adjustments back to `ControlPanelQt`, while still preserving the existing no-overlays UI behavior used by the layout tests.
- `synthesia2midi/synthesia2midi/gui/controls_qt.py` no longer increments or clears the displayed adjustment optimistically; reset emits the compensating delta only from confirmed state.
- Added focused regressions in `tests/test_main_window_layout.py` for:
  - a white-width shrink request where one targeted overlay is already `1 px` wide
  - a right-slant increase request where one targeted overlay is already at `45` degrees
- Verification:
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls tests/test_main_window_layout.py::test_overlays_tab_exposes_white_and_black_quick_adjust_controls -v` -> `2 passed in 0.36s`
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py::test_white_width_quick_adjust_does_not_drift_when_any_target_would_underflow tests/test_main_window_layout.py::test_right_slant_quick_adjust_does_not_drift_when_rotation_would_clamp -v` -> `2 passed in 0.33s`
  - `git diff --check` -> clean
- Commit: `b2bd427` (`Fix overlay quick-adjust drift accounting`)

## Fix
- Files changed:
  - `synthesia2midi/synthesia2midi/gui/main_action_controller.py`
  - `synthesia2midi/synthesia2midi/gui/controls_qt.py`
  - `tests/test_main_window_layout.py`
- Behavior changed:
  - Removed the phantom success path in `MainActionController.handle_overlay_size_adjustment(...)`; quick-adjust clicks with no overlays still emit the request, but they no longer advance the visible `Current` value or reset basis.
  - Added `ControlPanelQt.clear_overlay_adjustments()` plus an overlay-basis signature check in `update_controls_from_state()` so cached quick-adjust values are cleared when overlays disappear or when a different baseline geometry is now on screen.
  - Kept the bool-return overlay-manager contract intact; the panel still only advances after confirmed applied steps.
  - Reworked the quick-adjust layout tests to use the lighter control-panel plus `MainActionController`/`OverlayManager` path with real overlays, and added regressions for empty-state clicks plus stale-cache reset after baseline changes and overlay removal.
- Tests run:
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls tests/test_main_window_layout.py::test_overlays_tab_exposes_white_and_black_quick_adjust_controls -v`
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlay_quick_adjust_empty_state_does_not_change_display tests/test_main_window_layout.py::test_overlay_quick_adjust_values_reset_when_overlay_baseline_changes tests/test_main_window_layout.py::test_overlay_quick_adjust_values_reset_when_overlays_are_cleared tests/test_main_window_layout.py::test_white_width_quick_adjust_does_not_drift_when_any_target_would_underflow tests/test_main_window_layout.py::test_right_slant_quick_adjust_does_not_drift_when_rotation_would_clamp -v`
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py -v`
  - `git diff --check`
- Results:
  - Focused reviewer command: `2 passed in 0.33s`
  - Added/existing drift coverage: `5 passed in 0.33s`
  - Full layout file: `19 passed, 10 warnings in 0.55s`
  - `git diff --check`: clean
  - Remaining unrelated warning: only `.venv/bin/python -m pytest tests/test_main_window_layout.py -v` emits the existing `DeprecationWarning` from `synthesia2midi/synthesia2midi/utils/font_helper.py:32` (`QFontDatabase.QFontDatabase()`).

## Fix
- Files changed:
  - `synthesia2midi/synthesia2midi/gui/controls_qt.py`
  - `synthesia2midi/synthesia2midi/gui/main_action_controller.py`
  - `synthesia2midi/synthesia2midi/workflows/overlay_manager.py`
  - `tests/test_main_window_layout.py`
- Behavior changed:
  - Restored `MainActionController.handle_overlay_size_adjustment(...)` to the pre-Task-3 pass-through call into `OverlayManager`.
  - Restored `OverlayManager.adjust_overlay_sizes(...)` and `_adjust_overlay_slant(...)` to their pre-Task-3 no-return, per-overlay behavior from `0132644`.
  - Moved the visible-step guard into `ControlPanelQt`, so the local `Current` value and reset basis only advance when the present overlays can accept the full requested width/height or slant step without underflow or clamp.
  - Empty overlay state no longer creates fake displayed quick-adjust values, and `update_controls_from_state()` still clears cached values when overlays disappear or the geometry baseline changes.
- Tests run:
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls tests/test_main_window_layout.py::test_overlays_tab_exposes_white_and_black_quick_adjust_controls -v`
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlay_quick_adjust_empty_state_does_not_change_display tests/test_main_window_layout.py::test_overlay_quick_adjust_values_reset_when_overlay_baseline_changes tests/test_main_window_layout.py::test_overlay_quick_adjust_values_reset_when_overlays_are_cleared tests/test_main_window_layout.py::test_white_width_quick_adjust_does_not_drift_when_any_target_would_underflow tests/test_main_window_layout.py::test_right_slant_quick_adjust_does_not_drift_when_rotation_would_clamp -v`
  - `git diff --check`
- Results:
  - Reviewer quick-adjust pair: `2 passed in 0.56s`
  - Empty/baseline/underflow/clamp coverage: `5 passed in 0.59s`
  - `git diff --check`: clean

## Fix
- Files changed:
  - `synthesia2midi/synthesia2midi/gui/controls_qt.py`
  - `tests/test_main_window_layout.py`
- Behavior changed:
  - `ControlPanelQt._apply_overlay_adjustment(...)` now always emits `overlay_size_adjustment_requested(key_color, dimension, delta)` for the user click, even when the current overlays are empty, would underflow, or would slant-clamp.
  - The visible quick-adjust value and reset basis only advance when `_can_apply_overlay_adjustment(...)` says the requested step is safe for the overlays currently on screen.
  - Reset remains local-counted only: when the visible value is `0`, reset emits nothing.
  - `synthesia2midi/synthesia2midi/workflows/overlay_manager.py` is no longer touched by the Task 3 diff, and `synthesia2midi/synthesia2midi/gui/main_action_controller.py` still has no diff versus `0132644`.
- Tests run:
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlays_tab_exposes_left_and_right_slant_controls tests/test_main_window_layout.py::test_overlays_tab_exposes_white_and_black_quick_adjust_controls -v`
  - `.venv/bin/python -m pytest tests/test_main_window_layout.py::test_overlay_quick_adjust_empty_state_does_not_change_display tests/test_main_window_layout.py::test_overlay_quick_adjust_values_reset_when_overlay_baseline_changes tests/test_main_window_layout.py::test_overlay_quick_adjust_values_reset_when_overlays_are_cleared tests/test_main_window_layout.py::test_white_width_quick_adjust_does_not_drift_when_any_target_would_underflow tests/test_main_window_layout.py::test_right_slant_quick_adjust_does_not_drift_when_rotation_would_clamp -v`
  - `git diff --check`
- Results:
  - Reviewer quick-adjust pair: `2 passed in 0.39s`
  - Empty/baseline/underflow/clamp coverage: `5 passed in 0.41s`
  - `git diff --check`: clean
