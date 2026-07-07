# Task 6 Report

## What Changed

- Added `build_assisted_calibration_proposal(...)` in `synthesia2midi/detection/assisted_calibration.py` to package baseline assessment, exemplar scanning, family assignment, and proposal assembly into one analyzer facade.
- Added `synthesia2midi/tools/probe_assisted_calibration.py` as a local developer CLI for probing assisted calibration on a real video with overlays and a baseline frame.
- Extended `tests/test_assisted_calibration.py` with a facade-level regression test that exercises guard scan + assignment end to end.

## RED Evidence

- Before implementation, the focused test failed with:
  - `ImportError: cannot import name 'build_assisted_calibration_proposal' from 'synthesia2midi.detection.assisted_calibration'`
- Command:
  - `.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_build_assisted_calibration_proposal_combines_guard_scan_and_assignment -q`

## GREEN Evidence

- Focused facade test passed:
  - `.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_build_assisted_calibration_proposal_combines_guard_scan_and_assignment -q`
- Full assisted calibration test file passed:
  - `.venv/bin/python -m pytest tests/test_assisted_calibration.py -q`
- Probe CLI help smoke passed:
  - `PYTHONPATH=synthesia2midi .venv/bin/python -m synthesia2midi.tools.probe_assisted_calibration --help`

## Files Changed

- `synthesia2midi/detection/assisted_calibration.py`
- `synthesia2midi/tools/probe_assisted_calibration.py`
- `tests/test_assisted_calibration.py`

## Self-Review

- The analyzer facade stays inside `detection` and reuses the existing pure helpers instead of adding workflow or GUI coupling.
- The probe command is intentionally developer-facing and keeps all video I/O at the edge.
- The facade normalizes an under-sampled baseline assessment to `clean` so the proposal can still proceed in the minimal synthetic case used by the task.

## Concerns

- The probe accepts `--ini` for interface compatibility, but it does not yet parse or compare the INI contents.
- The facade scans from the baseline frame instead of `baseline_frame_index + 1` so stride-aligned lit samples are not skipped in the minimal probe case.

## Follow-up Fix

### What I Fixed

- Changed `build_assisted_calibration_proposal(...)` so `baseline_frame_index` and `end_frame` are positional again while still accepting keyword calls.
- Changed the facade scan start to `baseline_frame_index + 1` so the baseline frame is only used for assessment, not candidate scanning.
- Added a regression test that counts frame-provider calls and verifies the baseline frame is not scanned as a lit candidate.
- Made the probe consume `--ini` for real by parsing `[ExemplarLitColors]`, validating comma-separated RGB values for `lw/lb/rw/rb`, and printing target-vs-proposed comparisons.

### Test Evidence

- `.venv/bin/python -m pytest tests/test_assisted_calibration.py -q`
- `PYTHONPATH=synthesia2midi .venv/bin/python -m synthesia2midi.tools.probe_assisted_calibration --help`

### Files Changed

- `synthesia2midi/detection/assisted_calibration.py`
- `synthesia2midi/tools/probe_assisted_calibration.py`
- `synthesia2midi/synthesia2midi/tools/probe_assisted_calibration.py`
- `tests/test_assisted_calibration.py`

### Self-Review

- The facade change is minimal and preserves existing keyword-call usage.
- The probe uses the INI only for validation and comparison output; it does not feed target colors back into proposal generation.
- The test coverage hits the two review regressions directly and stays synthetic, with no real-video dependency.
