Task 3 Report: Capture Unlit References From The Baseline Frame

What changed
- Added `capture_unlit_references_from_frame(frame_rgb, overlays)` to `synthesia2midi/detection/assisted_calibration.py`.
- The new helper samples each overlay through the existing ROI sampling path, then stores `unlit_reference_color` and `unlit_hist` from the sampled ROI.
- Updated `synthesia2midi/synthesia2midi/workflows/calibration.py` to assess the current frame with `assess_unlit_frame` before unlit calibration overwrites existing overlay data.
- Added a soft, bypassable warning dialog using `QCoreApplication.translate`.
- Added regression coverage in `tests/test_assisted_calibration.py` and `tests/test_bugfix_regressions.py`.

RED evidence
- Initial red run:
  - `.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_capture_unlit_references_sets_rgb_and_histogram tests/test_bugfix_regressions.py::test_unlit_calibration_warns_when_frame_has_likely_lit_key -q`
  - Result: import error for missing `capture_unlit_references_from_frame`.
- Workflow warning red run in isolation:
  - `.venv/bin/python -m pytest tests/test_bugfix_regressions.py::test_unlit_calibration_warns_when_frame_has_likely_lit_key -q --ignore=tests/test_assisted_calibration.py`
  - Result: failed because no warning was emitted.

GREEN evidence
- Targeted green run:
  - `.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_capture_unlit_references_sets_rgb_and_histogram tests/test_bugfix_regressions.py::test_unlit_calibration_warns_when_frame_has_likely_lit_key -q`
  - Result: 2 passed.
- Broader green run:
  - `.venv/bin/python -m pytest tests/test_assisted_calibration.py tests/test_bugfix_regressions.py -q`
  - Result: 34 passed.

Files changed
- `synthesia2midi/detection/assisted_calibration.py`
- `synthesia2midi/synthesia2midi/workflows/calibration.py`
- `tests/test_assisted_calibration.py`
- `tests/test_bugfix_regressions.py`
- `.superpowers/sdd/task-3-report.md`

Self-review
- The new capture helper stays on the existing ROI sampling path, so overlay truncation semantics are preserved.
- The workflow warning is soft and returns early only when the user cancels.
- The current unlit overwrite loop was left intact per task brief.

Concerns
- Translation assets are not updated in this task; only the runtime `translate` calls were added.
- The workflow still depends on `current_frame_rgb` being present on the canvas to show the warning path.
