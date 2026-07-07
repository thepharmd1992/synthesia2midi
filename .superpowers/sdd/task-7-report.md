# Task 7 Report: Assisted Wizard Flow

## What changed

- Added assisted-calibration controller wiring in `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`.
- Imported the pure assisted-calibration facade APIs and added:
  - `_frame_provider_rgb(...)`
  - `_proposal_summary_text(...)`
  - `_run_assisted_auto_calibration(...)`
- Updated `_handle_keyboard_region_selected(...)` so that, after successful keyboard ROI autodetect and overlay creation, the controller:
  - reads the wizard baseline frame context,
  - captures unlit references from the baseline frame,
  - scans later frames through overlay ROIs,
  - shows a confirmation dialog with the proposed `LW/LB/RW/RB` legacy family-slot assignments,
  - applies the accepted proposal,
  - saves the current config,
  - then continues into the existing auto-detect tuning dialog flow.
- Added a regression test in `tests/test_bugfix_regressions.py` covering the accepted assisted-calibration proposal path and config save.

## RED test evidence

- Command:
  - `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_keyboard_region_selection_runs_assisted_calibration_and_saves -q`
- Result:
  - `FAILED`
- Expected failure observed:
  - `AttributeError: module 'synthesia2midi.gui.calibration_wizard_controller' has no attribute 'apply_assisted_calibration_proposal'`

## GREEN test evidence

- Command:
  - `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_keyboard_region_selection_runs_assisted_calibration_and_saves tests/test_bugfix_regressions.py::test_auto_detect_keyboard_region_marks_overlay_generation_source_auto -q`
- Result:
  - `2 passed`

- Command:
  - `git diff --check`
- Result:
  - clean output

## Files changed

- `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`
- `tests/test_bugfix_regressions.py`
- `.superpowers/sdd/task-7-report.md`

## Self-review

- Kept the implementation inside the controller boundary; no workflow or wizard changes were necessary.
- Preserved one-way dependency flow: GUI controller calls the pure detection facade and existing workflow save hook.
- Kept the assisted step synthetic-testable: no real video files, no network, no visible GUI requirement.
- Added a defensive guard so the assisted flow is skipped when the wizard context does not provide a usable baseline RGB frame.
- Left the existing modeless auto-detect tuning flow intact after the assisted proposal step.

## Concerns

- Task 8 still owns the unlit warning/cancel behavior inside the assisted flow. This task does not add extra warning UX beyond the proposal confirmation requested here.

---

## Review follow-up: assisted wizard flow findings

### What you fixed

- Added `QApplication.processEvents()` inside the synchronous assisted-scan progress callback so the `QProgressDialog` cancel button is processed during scan.
- Added a controller-side guard that rejects empty assisted proposals before confirmation, apply, or save when:
  - `proposal.candidate_count == 0`, or
  - no assignment is both enabled and backed by an RGB sample.
- Added an informational message using `translate("CalibrationWizardController", ...)` for the no-result case so existing exemplar state is left unchanged.
- Added regression coverage for:
  - scan cancel behavior through the progress callback,
  - no-result proposals not applying or saving,
  - explicit decline/no-save behavior,
  - the accepted-path assisted calibration save flow using a controller-level proposal stub.

### Test evidence

- `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_keyboard_region_selection_runs_assisted_calibration_and_saves -q`
  - `1 passed`
- `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_auto_detect_keyboard_region_marks_overlay_generation_source_auto -q`
  - `1 passed`
- `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_assisted_calibration_scan_cancel_processes_events_and_stops tests/test_bugfix_regressions.py::test_assisted_calibration_no_result_does_not_apply_or_save tests/test_bugfix_regressions.py::test_assisted_calibration_decline_does_not_apply_or_save -q`
  - `3 passed`
- `git diff --check`
  - clean

### Files changed

- `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`
- `tests/test_bugfix_regressions.py`
- `.superpowers/sdd/task-7-report.md`

### Self-review

- Kept the implementation inside the GUI controller boundary; analyzer code was unchanged.
- Used controller-focused proposal stubs for the save/decline/no-result regressions so the tests verify controller decisions rather than analyzer heuristics.
- The no-result guard is intentionally strict: assisted calibration now only persists when at least one enabled assignment carries a real RGB exemplar.
- The cancel fix is limited to event pumping in the existing synchronous path, matching the review request without introducing thread or workflow churn.

---

## Review follow-up: restore unlit calibration on non-acceptance

### What you fixed

- Snapshotted each overlay's `unlit_reference_color` and `unlit_hist` before assisted baseline capture in `_run_assisted_auto_calibration(...)`.
- Restored that per-overlay unlit state on every post-capture early return:
  - canceled scan,
  - no-result proposal,
  - declined proposal.
- Left the accepted path unchanged so accepted assisted calibration still keeps the baseline-captured unlit references and then applies/saves the proposal.
- Extended the assisted calibration regressions to prove overlay unlit samples are unchanged across cancel, no-result, and decline exits.

### Test evidence

- RED:
  - `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_assisted_calibration_no_result_does_not_apply_or_save tests/test_bugfix_regressions.py::test_assisted_calibration_decline_does_not_apply_or_save tests/test_bugfix_regressions.py::test_assisted_calibration_scan_cancel_processes_events_and_stops -q`
  - `3 failed`
  - Failure cause: overlay `unlit_reference_color` had been overwritten by baseline capture on each non-acceptance path.
- GREEN:
  - `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_assisted_calibration_no_result_does_not_apply_or_save tests/test_bugfix_regressions.py::test_assisted_calibration_decline_does_not_apply_or_save tests/test_bugfix_regressions.py::test_keyboard_region_selection_runs_assisted_calibration_and_saves -q`
  - `3 passed`
- Additional regression:
  - `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_assisted_calibration_scan_cancel_processes_events_and_stops -q`
  - `1 passed`
- Hygiene:
  - `git diff --check`
  - clean

### Files changed

- `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`
- `tests/test_bugfix_regressions.py`
- `.superpowers/sdd/task-7-report.md`

### Self-review

- Kept the fix narrow and controller-local; assisted calibration helpers and analyzer internals were unchanged.
- Used explicit restore points instead of altering acceptance behavior, so the accepted assisted flow still preserves baseline capture as requested.
- Added cancel-path assertions in addition to the requested no-result/decline coverage because the same mutation bug affected all non-acceptance exits.
