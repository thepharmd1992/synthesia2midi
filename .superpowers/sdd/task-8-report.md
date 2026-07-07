# Task 8 Report: Unlit Warning In Assisted Flow

## What changed

- Updated `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py` so `_run_assisted_auto_calibration(...)` now calls `assess_unlit_frame(...)` on the baseline frame before `capture_unlit_references_from_frame(...)`.
- Added a soft, bypassable `QMessageBox.warning(...)` when the baseline frame appears to contain lit keys.
- Kept the assisted flow cancellable: if the user chooses Cancel, the method returns `False` before any unlit references are written, before the scan starts, and before proposal application or save.
- Used `QCoreApplication.translate` via the existing `translate` alias for the new warning title and body strings.
- Added `test_assisted_calibration_unlit_warning_cancel_skips_apply` to `tests/test_bugfix_regressions.py` to prove the cancel path short-circuits the assisted flow.

## RED test evidence

- Command:
  - `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_assisted_calibration_unlit_warning_cancel_skips_apply -q`
- Result:
  - `FAILED`
- Expected failure observed:
  - `AssertionError: scan should not run after warning cancel`

## GREEN test evidence

- Command:
  - `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_assisted_calibration_unlit_warning_cancel_skips_apply tests/test_bugfix_regressions.py::test_keyboard_region_selection_runs_assisted_calibration_and_saves -q`
- Result:
  - `2 passed`

- Command:
  - `git diff --check`
- Result:
  - clean output

## Files changed

- `synthesia2midi/synthesia2midi/gui/calibration_wizard_controller.py`
- `tests/test_bugfix_regressions.py`
- `.superpowers/sdd/task-8-report.md`

## Self-review

- Kept the change inside the GUI controller boundary and used the existing pure assessment helper rather than adding new workflow or detection plumbing.
- Preserved the one-way dependency direction: GUI controller code calls the detection facade; no reverse coupling was introduced.
- The warning is soft and bypassable, matching the task brief, and the Cancel path exits before any scan or write.
- The regression test is synthetic and does not require a real video file or visible GUI interaction.

## Concerns

- The warning strings are translated through `translate(...)`, but translation asset updates are still deferred to Task 9.
