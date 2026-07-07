# Task 4: Lit Exemplar Scanner

Implementation commit: `fc3eb81` (`feat: scan lit exemplar candidates`)

## What changed

- Added `scan_lit_exemplar_candidates(...)` to `synthesia2midi/detection/assisted_calibration.py`.
- Added the supporting internal helpers for candidate construction and per-frame overlay scanning.
- Kept the scanner on overlay ROIs only, reusing the existing RGB/BGR sampling helpers and histogram extraction path.
- Preserved the existing unlit-reference semantics and the ROI truncation behavior already covered by the helper tests.
- Added focused tests for candidate discovery and cancellation in `tests/test_assisted_calibration.py`.

## RED / GREEN evidence

RED:

- Command: `.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_scanner_finds_lit_candidates_from_overlay_deltas tests/test_assisted_calibration.py::test_scanner_honors_cancel_callback -q`
- Result before implementation: `ImportError: cannot import name 'scan_lit_exemplar_candidates'`

GREEN:

- Same focused command passed after implementation.
- Full module check: `.venv/bin/python -m pytest tests/test_assisted_calibration.py -q`
- Result: `11 passed`

## Files changed

- `synthesia2midi/detection/assisted_calibration.py`
- `tests/test_assisted_calibration.py`
- `.superpowers/sdd/task-4-report.md`

## Self-review

- The scanner stays inside the detection layer and does not introduce GUI coupling or new dependencies.
- ROI sampling continues to use the existing clipped overlay helpers, so fractional and out-of-bounds behavior is unchanged.
- The scan logic now uses a lit coarse frame as a refinement anchor for all overlays, which is necessary to recover exemplars that appear between coarse strides.

## Concerns

- The scanner keeps the existing per-key top-N pruning but does not add a stronger deduplication layer across repeated coarse windows. That matches the current bounded candidate model, but it can retain multiple nearby hits for the same key when a long run contains repeated lit frames.
