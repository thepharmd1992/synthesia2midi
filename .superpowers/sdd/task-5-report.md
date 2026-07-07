# Task 5 Report: Color-Family Assignment And Proposal Application

## What changed

- Added `assign_exemplar_slots()` in `synthesia2midi/detection/assisted_calibration.py`.
  - Groups exemplar candidates into at most two families using hue distance only.
  - Chooses the best white and black exemplar within each family by confidence, then by earlier frame, then by `key_id` as a local tiebreaker.
  - Treats `LW/LB/RW/RB` strictly as storage slots. Assignment is based on color family plus key color, not physical keyboard position.
  - Disables the second slot pair when only one family is present.
- Added `apply_assisted_calibration_proposal()` in `synthesia2midi/detection/assisted_calibration.py`.
  - Applies enabled flags, lit RGB colors, and histograms from the proposal assignment result into `AppState.detection`.
  - Clears disabled slot colors and histograms.
  - Marks `app_state.unsaved_changes = True`.
- Added Task 5 tests in `tests/test_assisted_calibration.py`.
  - Verifies two-family assignment by hue instead of position.
  - Verifies absent second-family slots are disabled.
  - Verifies proposal application updates colors, histograms, enabled flags, and unsaved state.

## RED test evidence

Command:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py::test_assign_exemplar_slots_maps_two_color_families_by_hue_not_position tests/test_assisted_calibration.py::test_assign_exemplar_slots_disables_absent_second_family tests/test_assisted_calibration.py::test_apply_assisted_calibration_proposal_updates_colors_histograms_and_enabled_slots -q
```

Result:

- Exit code `4`
- Import-time failure as expected because `apply_assisted_calibration_proposal` did not exist yet
- Pytest output included:
  - `ImportError: cannot import name 'apply_assisted_calibration_proposal' from 'synthesia2midi.detection.assisted_calibration'`

## GREEN test evidence

Command:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py -q
```

Result:

- Exit code `0`
- `15 passed`

Additional verification:

```bash
.venv/bin/python -m compileall -q synthesia2midi
git diff --check
```

Result:

- Both commands exited `0`

## Files changed

- `synthesia2midi/detection/assisted_calibration.py`
- `tests/test_assisted_calibration.py`
- `.superpowers/sdd/task-5-report.md`

## Self-review

- Kept the change inside the owned module and test file.
- Did not alter ROI truncation logic, unlit-frame assessment behavior, or event stabilization in lit scanning.
- Did not introduce physical-position assignment logic.
- Did not add user-visible strings or GUI dependencies.
- The implementation depends on overlay ROI samples and existing HSV/RGB helpers only.

## Concerns

- Family ordering is currently deterministic by hue-family sort (`cool` family first, then warmer family). That matches the task brief examples, but it is still a policy choice for mapping the legacy `LW/LB` and `RW/RB` storage slots.
- This task does not yet cover edge cases where one family contains only white or only black exemplars; those slots stay enabled and are reported as missing, which is consistent with the current assignment result model.

## Follow-up Review Fixes (Findings 1,2)

### What added/fixed

- Added `test_assign_exemplar_slots_enables_partial_family_with_missing_partner_as_missing` in `tests/test_assisted_calibration.py` to cover a present family with only white candidates.
- Strengthened `test_apply_assisted_calibration_proposal_updates_colors_histograms_and_enabled_slots` to directly assert histogram write/clear behavior for enabled/disabled slots.
- Left production code unchanged because existing `assign_exemplar_slots` and `apply_assisted_calibration_proposal` already satisfy the targeted behavior.

### Test evidence

Command:

```bash
.venv/bin/python -m pytest tests/test_assisted_calibration.py -q
```

Result:

- Exit code `0`
- `16 passed`

### Files changed

- `tests/test_assisted_calibration.py`
- `.superpowers/sdd/task-5-report.md`

### Self-review

- Scope stayed narrow to test updates plus report append.
- Added assertions directly target reviewer findings without broadening behavior expectations.
- No unrelated test files, docs, or production behavior touched.
