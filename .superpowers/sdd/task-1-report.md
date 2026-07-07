# Task 1 Report: Pure Data Models And Overlay Sampling

## What changed
- Added `synthesia2midi/synthesia2midi/detection/assisted_calibration.py` with:
  - pure data model dataclasses (`LikelyLitOverlay`, `UnlitFrameAssessment`, `ExemplarCandidate`, `AssignedExemplar`, `ExemplarAssignmentResult`, `AssistedCalibrationProposal`, `ExemplarScanSettings`)
  - pure helper functions:
    - `overlay_note_label(overlay: OverlayConfig) -> str`
    - `overlay_key_color(overlay: OverlayConfig) -> str`
    - `sample_overlay_rgb(frame_rgb: np.ndarray, overlay: OverlayConfig) -> tuple[int, int, int] | None`
    - `sample_overlay_bgr(frame_rgb: np.ndarray, overlay: OverlayConfig) -> np.ndarray | None`
  - internal ROI clipping helper `_overlay_bounds`
  - `LW/LB/RW/RB` mapped via suffix to legacy color family (`W` unless suffix is `B`, which maps to `B`)
- Added `tests/test_assisted_calibration.py` with the required 3 tests:
  - `test_overlay_sampling_uses_clipped_integer_roi`
  - `test_overlay_sampling_returns_none_for_empty_roi`
  - `test_overlay_note_label_and_key_color_use_existing_overlay_data`

## RED/GREEN test evidence
- RED (`.venv/bin/python -m pytest tests/test_assisted_calibration.py -q`):
  - Failed at collection with `ModuleNotFoundError: No module named 'synthesia2midi.detection.assisted_calibration'` (as expected while file did not yet exist).
- GREEN (`.venv/bin/python -m pytest tests/test_assisted_calibration.py -q`):
  - `... [100%]`

## Files changed
- `synthesia2midi/synthesia2midi/detection/assisted_calibration.py` (new)
- `tests/test_assisted_calibration.py` (new)
- `.superpowers/sdd/task-1-report.md` (new)

## Self-review
- Verified ROI behavior is deterministic and pure:
  - Uses integer-rounded overlay bounds
  - Clips to frame bounds
  - Returns `None` if ROI is empty or outside frame
  - BGR sampling is derived from ROI after RGB-to-BGR conversion, preserving color channel order expected by existing cv2 workflows
- Kept implementation GUI-free and non-deterministic-free.
- Kept changes scoped to requested files.

## Concerns
- No functional concerns identified for Task 1 scope.

---

## Task 1 Review Follow-up: Assisted Calibration Sampling Semantics

## What changed
- Updated `synthesia2midi/detection/assisted_calibration.py` so `_overlay_bounds()` now truncates overlay coordinates and dimensions with `int(...)` before clipping, matching the app's existing ROI slicing path.
- Added a regression test in `tests/test_assisted_calibration.py`:
  - `test_overlay_sampling_truncates_fractional_overlay_bounds`

## RED evidence
- Command:
  - `.venv/bin/python -m pytest tests/test_assisted_calibration.py -q -k truncates_fractional_overlay_bounds`
- Failure:
  - Expected `sample_overlay_rgb(frame, overlay)` to return `tuple(frame[1, 1])`
  - Actual value was `(30, 31, 32)` instead of `(15, 16, 17)`, proving the helper was sampling the rounded ROI pixel set

## GREEN evidence
- Command:
  - `.venv/bin/python -m pytest tests/test_assisted_calibration.py -q`
- Result:
  - `.... [100%]`

## Notes
- Scope stayed limited to the requested calibration helper, regression test, and report file.
